"""
scripts/exp_utility_mmlu.py

Utility evaluation: MMLU 4-choice MCQ accuracy (A/B/C/D)
  baseline     : no intervention
  target       : hook at specified layer/feature, α
  random       : same α, coherence-matched isotropic random direction
  random_sae_dec: same magnitude, direction from random alive SAE feature (K=5 majority vote)

Data: HuggingFace cais/mmlu (all, test split, 14042 samples across 57 subjects)
  format: {question, subject, choices: [A,B,C,D], answer: int 0-3}

Usage:
  # Llama L16
  CUDA_VISIBLE_DEVICES=0 conda run -n privacy-leakage \
      python scripts/exp_utility_mmlu.py \
      --model llama3_1_8b_instruct --layer 16 --feature 32619 \
      --alpha -2 --n-samples 200 --out-dir results/utility_mmlu

  # Qwen L18
  CUDA_VISIBLE_DEVICES=0 conda run -n privacy-leakage \
      python scripts/exp_utility_mmlu.py \
      --model qwen3_8b --layer 18 --feature 540 \
      --alpha -2 --n-samples 200 --out-dir results/utility_mmlu

  # Gemma L20
  CUDA_VISIBLE_DEVICES=0 conda run -n privacy-leakage \
      python scripts/exp_utility_mmlu.py \
      --model gemma_2_9b --layer 20 --feature 20385 \
      --alpha -2 --n-samples 200 --out-dir results/utility_mmlu
"""

import argparse, os, sys, csv
from collections import Counter
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import config
from scripts.intervention import (
    get_sae_W_dec, llama_hook, make_intervention_fn,
    make_random_sae_decoder_fn,
)
from scripts.tasksae_glue import load_tasksae_for_intervention, get_alive_features


CHOICES = ["A", "B", "C", "D"]


# ── prompt builders ───────────────────────────────────────────────────────────

def format_mcq(question: str, choices: list[str]) -> str:
    body = "\n".join(f"{l}. {c}" for l, c in zip(CHOICES, choices))
    return f"Question: {question}\n{body}"


def build_prompt_hf(tokenizer, question: str, choices: list[str]) -> str:
    messages = [
        {"role": "system",
         "content": (
             "Answer the following multiple choice question with exactly one "
             "letter: A, B, C, or D."
         )},
        {"role": "user", "content": format_mcq(question, choices)},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False
        )
    except Exception:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def build_prompt_completion(question: str, choices: list[str]) -> str:
    return format_mcq(question, choices) + "\nAnswer:"


# ── logprob label IDs ─────────────────────────────────────────────────────────

_LABEL_IDS: dict | None = None


def get_label_ids(tokenizer):
    global _LABEL_IDS
    if _LABEL_IDS is None:
        def first_tok(letter):
            for prefix in (" ", ""):
                ids = tokenizer.encode(prefix + letter, add_special_tokens=False)
                if ids:
                    return ids[0]
            raise ValueError(f"cannot tokenize {letter!r}")
        _LABEL_IDS = {lbl: first_tok(lbl) for lbl in CHOICES}
        print(f"[tok] label ids: {_LABEL_IDS}")
    return _LABEL_IDS


# ── logprob classifier ────────────────────────────────────────────────────────

def classify_hf(model, tokenizer, prompt: str, device) -> str:
    inputs = tokenizer(prompt, return_tensors="pt",
                       truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        out = model(**inputs)
    logits = out.logits[0, -1, :].float().cpu()
    ids = get_label_ids(tokenizer)
    scores = {lbl: logits[tid].item() for lbl, tid in ids.items()}
    return max(scores, key=scores.get)


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="llama3_1_8b_instruct",
                   choices=list(config.MODEL_CONFIGS.keys()))
    p.add_argument("--layer", type=int, default=16)
    p.add_argument("--feature", type=int, default=32619)
    p.add_argument("--alpha", type=float, default=-2.0)
    p.add_argument("--n-samples", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mmlu-config", default="all",
                   help="MMLU subset config (default: all = 57 subjects)")
    p.add_argument("--out-dir", default="results/utility_mmlu")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = config.MODEL_CONFIGS[args.model]
    model_type = cfg["model_type"]

    # ── Load data ─────────────────────────────────────────────────────────────
    from datasets import load_dataset
    print(f"[data] loading MMLU ({args.mmlu_config}) ...")
    ds = load_dataset("cais/mmlu", args.mmlu_config, split="test")
    rng_data = np.random.default_rng(args.seed)
    indices = rng_data.choice(len(ds), size=min(args.n_samples, len(ds)), replace=False)
    samples = [ds[int(i)] for i in sorted(indices)]
    print(f"[data] {len(samples)} samples (total {len(ds)}, 57 subjects)")
    subj_counts = Counter(s["subject"] for s in samples)
    print(f"[data] subject spread: {len(subj_counts)} unique subjects sampled")

    # ── Load model ────────────────────────────────────────────────────────────
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[load] {args.model} → cuda:0 (single GPU) ...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["path"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"],
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = torch.device("cuda:0")
    model.eval()
    layer_device = next(model.model.layers[args.layer].parameters()).device
    print(f"[info] layer {args.layer} on {layer_device}")

    # ── Load SAE ──────────────────────────────────────────────────────────────
    print(f"[load] task-SAE L{args.layer} x8_v2 ...")
    sae_wrapper = load_tasksae_for_intervention(
        args.model, args.layer, str(layer_device), expansion=8, suffix="v2"
    )
    sae = sae_wrapper.sae
    d_sae = sae.d_sae
    W_dec = get_sae_W_dec(sae, d_sae, cfg["d_model"]).to(layer_device)
    print(f"  d_sae={d_sae}  sae_device={layer_device}")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    modes = ["baseline", "target", "random", "random_sae_dec"]
    results = {m: [] for m in modes}

    alive_all = get_alive_features(args.model, args.layer, expansion=8, suffix="v2")
    alive_excl = [f for f in alive_all if f != args.feature]
    K_util = 5

    use_chat = model_type in ("llama", "qwen") and hasattr(tokenizer, "apply_chat_template")

    for i, sample in enumerate(samples):
        question = sample["question"]
        choices = sample["choices"]
        label = CHOICES[sample["answer"]]

        if use_chat:
            prompt = build_prompt_hf(tokenizer, question, choices)
        else:
            prompt = build_prompt_completion(question, choices)

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(samples)}]")

        for mode in modes:
            torch.manual_seed(args.seed + i)
            np.random.seed(args.seed + i)

            if mode == "random_sae_dec":
                rng_rsae = np.random.default_rng(args.seed + i + 99999)
                k_idx = rng_rsae.choice(len(alive_excl),
                                        size=min(K_util, len(alive_excl)),
                                        replace=False)
                rand_feats = [alive_excl[j] for j in k_idx]
                preds_k = []
                for gf in rand_feats:
                    rsae_fn = make_random_sae_decoder_fn(
                        sae, W_dec, sae_wrapper.backend, d_sae,
                        [args.feature], args.alpha, rand_feature_id=gf
                    )
                    handle = model.model.layers[args.layer].register_forward_hook(
                        llama_hook(rsae_fn)
                    )
                    try:
                        preds_k.append(classify_hf(model, tokenizer, prompt, device))
                    finally:
                        handle.remove()
                pred = Counter(preds_k).most_common(1)[0][0]

            else:
                int_fn = make_intervention_fn(
                    sae, W_dec, sae_wrapper.backend, d_sae,
                    [args.feature], args.alpha,
                    "baseline" if mode == "baseline" else
                    "target"   if mode == "target"   else "random_vector"
                )
                handle = None
                if int_fn is not None:
                    handle = model.model.layers[args.layer].register_forward_hook(
                        llama_hook(int_fn)
                    )
                try:
                    pred = classify_hf(model, tokenizer, prompt, device)
                finally:
                    if handle is not None:
                        handle.remove()

            results[mode].append({
                "idx": int(indices[i]), "subject": sample["subject"],
                "label": label, "pred": pred,
                "correct": int(pred == label)
            })

    # ── Report ────────────────────────────────────────────────────────────────
    print("\n=== MMLU Utility Results ===")
    print(f"  n={len(samples)}  model={args.model}  L{args.layer}  f={args.feature}  α={args.alpha}")
    print()
    summary_rows = []
    for mode in modes:
        rows = results[mode]
        acc = np.mean([r["correct"] for r in rows])
        per_cls = {}
        for lbl in CHOICES:
            sub = [r for r in rows if r["label"] == lbl]
            per_cls[lbl] = (np.mean([r["correct"] for r in sub]) if sub else float("nan"),
                            len(sub))
        cls_str = "  ".join(f"{l}={v:.3f}(n={n})" for l, (v, n) in per_cls.items())
        print(f"  {mode:15s}  overall={acc:.3f}  {cls_str}")
        summary_rows.append({"mode": mode, "accuracy": round(acc, 4), "n": len(rows)})

        out_csv = Path(args.out_dir) / f"mmlu_{args.model}_L{args.layer}_f{args.feature}_{mode}.csv"
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["idx", "subject", "label", "pred", "correct"])
            w.writeheader()
            w.writerows(rows)

    out_sum = (Path(args.out_dir) /
               f"mmlu_{args.model}_L{args.layer}_f{args.feature}_summary.csv")
    with open(out_sum, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["mode", "accuracy", "n"])
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\nSaved to {args.out_dir}")


if __name__ == "__main__":
    main()
