"""
scripts/exp_utility_pubmedqa.py

Utility evaluation: PubMedQA yes/no/maybe accuracy
  baseline  : no intervention
  target    : hook at specified layer/feature, α
  random    : same α, coherence-matched random direction

Supports all 3 model types:
  llama3_1_8b_instruct : HF, chat template, single GPU (device_map=auto tied-weight bug avoided)
  qwen3_8b             : HF, chat template + thinking disabled, single GPU
  gemma_2_9b           : TransformerLens, completion-style prompt

Usage:
  # Llama (main result)
  python scripts/exp_utility_pubmedqa.py \
      --model llama3_1_8b_instruct --layer 16 --feature 32619 \
      --alpha -2 --n-samples 200 --out-dir results/utility_pubmedqa

  # Qwen (main result)
  python scripts/exp_utility_pubmedqa.py \
      --model qwen3_8b --layer 18 --feature 540 \
      --alpha -2 --n-samples 200 --out-dir results/utility_pubmedqa

  # Gemma (main result)
  python scripts/exp_utility_pubmedqa.py \
      --model gemma_2_9b --layer 20 --feature 20385 \
      --alpha -2 --n-samples 200 --out-dir results/utility_pubmedqa
"""

import argparse, json, os, sys, csv
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import config
import models.loader as loader
from scripts.intervention import (
    get_sae_W_dec, llama_hook, gemma_hook, make_intervention_fn,
    make_random_sae_decoder_fn,
)
from scripts.tasksae_glue import load_tasksae_for_intervention, get_alive_features


MAX_CTX_CHARS = 600


# ── prompt builders ───────────────────────────────────────────────────────────

def build_prompt_hf(tokenizer, question: str, contexts: list[str]) -> str:
    """Chat-template prompt for Llama / Qwen instruct models."""
    ctx_parts = []
    for i, c in enumerate(contexts[:2]):
        ctx_parts.append(f"Abstract {i+1}: {c[:MAX_CTX_CHARS]}")
    ctx_text = "\n\n".join(ctx_parts)

    messages = [
        {"role": "system",
         "content": (
             "You are a biomedical expert. Read the provided abstracts and "
             "answer the question with exactly one word: yes, no, or maybe."
         )},
        {"role": "user",
         "content": f"{ctx_text}\n\nQuestion: {question}"},
    ]
    # Qwen3: disable thinking mode so next token is the answer, not <think>
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False
        )
    except Exception:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def build_prompt_gemma(question: str, contexts: list[str]) -> str:
    """Completion-style prompt for Gemma base model (no chat template)."""
    ctx_parts = []
    for i, c in enumerate(contexts[:2]):
        ctx_parts.append(f"Abstract {i+1}: {c[:MAX_CTX_CHARS]}")
    ctx_text = "\n\n".join(ctx_parts)
    return (
        f"{ctx_text}\n\n"
        f"Question: {question}\n"
        f"Answer (yes, no, or maybe):"
    )


# ── logprob label IDs ─────────────────────────────────────────────────────────

_LABEL_IDS: dict | None = None


def get_label_ids(tokenizer):
    global _LABEL_IDS
    if _LABEL_IDS is None:
        def first_tok(word):
            for prefix in (" ", ""):
                ids = tokenizer.encode(prefix + word, add_special_tokens=False)
                if ids:
                    return ids[0]
            raise ValueError(f"cannot tokenize {word!r}")
        _LABEL_IDS = {lbl: first_tok(lbl) for lbl in ["yes", "no", "maybe"]}
        print(f"[tok] label ids: {_LABEL_IDS}")
    return _LABEL_IDS


# ── logprob classifiers ───────────────────────────────────────────────────────

def classify_hf(model, tokenizer, prompt: str, device) -> str:
    """HF model: compare next-token logprobs at last position."""
    inputs = tokenizer(prompt, return_tensors="pt",
                       truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        out = model(**inputs)
    logits = out.logits[0, -1, :].float().cpu()
    ids = get_label_ids(tokenizer)
    scores = {lbl: logits[tid].item() for lbl, tid in ids.items()}
    return max(scores, key=scores.get)


def classify_tl(model, prompt: str) -> str:
    """TransformerLens model: compare next-token logprobs at last position."""
    tokens = model.to_tokens(prompt, prepend_bos=True)
    with torch.no_grad():
        logits = model(tokens)  # [1, seq, vocab]
    last_logits = logits[0, -1, :].float().cpu()
    ids = get_label_ids(model.tokenizer)
    scores = {lbl: last_logits[tid].item() for lbl, tid in ids.items()}
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
    p.add_argument("--data-dir", default="Utility_data/pubmedqa")
    p.add_argument("--out-dir", default="results/utility_pubmedqa")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = config.MODEL_CONFIGS[args.model]
    model_type = cfg["model_type"]

    # ── Load data ─────────────────────────────────────────────────────────────
    with open(os.path.join(args.data_dir, "ori_pqal.json")) as f:
        raw = json.load(f)
    with open(os.path.join(args.data_dir, "test_ground_truth.json")) as f:
        gt_map = json.load(f)

    keys = [k for k in raw if k in gt_map][:args.n_samples]
    print(f"[data] {len(keys)} samples with ground truth (requested {args.n_samples})")

    # ── Load model ────────────────────────────────────────────────────────────
    if model_type in ("llama", "qwen", "gemma"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        # Force single GPU: device_map=auto splits model across GPUs, causing
        # tied-weight (lm_head ↔ embed_tokens) to produce all-zero logits.
        # For Gemma: also avoids TransformerLens rotary-embedding CUDA assertion.
        print(f"[load] {args.model} → cuda:0 (single GPU, HF) ...")
        tokenizer = AutoTokenizer.from_pretrained(cfg["path"])
        model = AutoModelForCausalLM.from_pretrained(
            cfg["path"],
            torch_dtype=torch.bfloat16,
            device_map={"": "cuda:0"},
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        device = torch.device("cuda:0")
        model.eval()
        layer_device = next(model.model.layers[args.layer].parameters()).device

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

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

    # Pre-load alive features for random_sae_dec (excludes target feature)
    alive_all = get_alive_features(args.model, args.layer, expansion=8, suffix="v2")
    alive_excl = [f for f in alive_all if f != args.feature]
    K_util = 5

    for i, key in enumerate(keys):
        sample = raw[key]
        label = gt_map[key]

        if model_type in ("llama", "qwen") and hasattr(tokenizer, "apply_chat_template"):
            prompt = build_prompt_hf(tokenizer, sample["QUESTION"], sample["CONTEXTS"])
        else:
            prompt = build_prompt_gemma(sample["QUESTION"], sample["CONTEXTS"])

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(keys)}]")

        for mode in modes:
            torch.manual_seed(args.seed + i)
            np.random.seed(args.seed + i)

            if mode == "random_sae_dec":
                rng_rsae = np.random.default_rng(args.seed + i + 99999)
                k_idx = rng_rsae.choice(len(alive_excl), size=min(K_util, len(alive_excl)), replace=False)
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
                from collections import Counter
                pred = Counter(preds_k).most_common(1)[0][0]
            else:
                int_fn = make_intervention_fn(
                    sae, W_dec, sae_wrapper.backend, d_sae,
                    [args.feature], args.alpha,
                    "baseline" if mode == "baseline" else
                    "target" if mode == "target" else "random_vector"
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
                "key": key, "label": label, "pred": pred,
                "correct": int(pred == label)
            })

    # ── Report ────────────────────────────────────────────────────────────────
    print("\n=== PubMedQA Utility Results ===")
    print(f"  n={len(keys)}  model={args.model}  L{args.layer}  f={args.feature}  α={args.alpha}")
    print()
    summary_rows = []
    for mode in modes:
        rows = results[mode]
        acc = np.mean([r["correct"] for r in rows])
        for lbl in ["yes", "no", "maybe"]:
            sub = [r for r in rows if r["label"] == lbl]
            sub_acc = np.mean([r["correct"] for r in sub]) if sub else float("nan")
            print(f"  {mode:10s}  overall={acc:.3f}  {lbl}={sub_acc:.3f}  (n_{lbl}={len(sub)})")
        summary_rows.append({"mode": mode, "accuracy": round(acc, 4), "n": len(rows)})
        print()

        out_csv = Path(args.out_dir) / f"pubmedqa_{args.model}_L{args.layer}_f{args.feature}_{mode}.csv"
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["key", "label", "pred", "correct"])
            w.writeheader()
            w.writerows(rows)

    out_sum = Path(args.out_dir) / f"pubmedqa_{args.model}_L{args.layer}_f{args.feature}_summary.csv"
    with open(out_sum, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["mode", "accuracy", "n"])
        w.writeheader()
        w.writerows(summary_rows)
    print(f"Saved to {args.out_dir}")


if __name__ == "__main__":
    main()
