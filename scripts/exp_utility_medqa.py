"""
scripts/exp_utility_medqa.py

Utility evaluation: MedQA 4-choice MCQ accuracy (A/B/C/D)
  baseline : no intervention
  target   : hook at specified layer/feature, α
  random   : same α, coherence-matched random direction

Data: Utility_data/medqa.json  (list of {query, answer})
  query already contains question + Option A/B/C/D
  answer is the correct letter (A/B/C/D)

Usage:
  python scripts/exp_utility_medqa.py \
      --model llama3_1_8b_instruct --layer 16 --feature 32619 \
      --alpha -2 --n-samples 200 --out-dir results/utility_medqa

  python scripts/exp_utility_medqa.py \
      --model qwen3_8b --layer 18 --feature 540 \
      --alpha -2 --n-samples 200 --out-dir results/utility_medqa

  python scripts/exp_utility_medqa.py \
      --model gemma_2_9b --layer 20 --feature 20385 \
      --alpha -2 --n-samples 200 --out-dir results/utility_medqa
"""

import argparse, json, os, sys, csv
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import config
from scripts.intervention import get_sae_W_dec, llama_hook, make_intervention_fn
from scripts.tasksae_glue import load_tasksae_for_intervention


# ── prompt builders ───────────────────────────────────────────────────────────

def build_prompt_hf(tokenizer, query: str) -> str:
    """Chat-template prompt for Llama / Qwen instruct models."""
    messages = [
        {"role": "system",
         "content": (
             "You are a medical expert. Answer the following multiple choice question "
             "with exactly one letter: A, B, C, or D."
         )},
        {"role": "user", "content": query.strip()},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False  # Qwen3: skip <think> block
        )
    except Exception:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def build_prompt_completion(query: str) -> str:
    """Completion-style prompt for Gemma base model."""
    return query.strip() + "\nAnswer:"


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
        _LABEL_IDS = {lbl: first_tok(lbl) for lbl in ["A", "B", "C", "D"]}
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
    p.add_argument("--data-path", default="Utility_data/medqa.json")
    p.add_argument("--out-dir", default="results/utility_medqa")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = config.MODEL_CONFIGS[args.model]
    model_type = cfg["model_type"]

    # ── Load data ─────────────────────────────────────────────────────────────
    with open(args.data_path) as f:
        raw = json.load(f)

    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(raw), size=min(args.n_samples, len(raw)), replace=False)
    samples = [raw[i] for i in sorted(indices)]
    print(f"[data] {len(samples)} samples (total {len(raw)})")

    # ── Load model ────────────────────────────────────────────────────────────
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[load] {args.model} → cuda:0 (single GPU) ...")
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
    modes = ["baseline", "target", "random"]
    results = {m: [] for m in modes}

    use_chat = model_type in ("llama", "qwen") and hasattr(tokenizer, "apply_chat_template")

    for i, sample in enumerate(samples):
        query = sample["query"]
        label = sample["answer"].strip().upper()

        if use_chat:
            prompt = build_prompt_hf(tokenizer, query)
        else:
            prompt = build_prompt_completion(query)

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(samples)}]")

        for mode in modes:
            torch.manual_seed(args.seed + i)
            np.random.seed(args.seed + i)

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
                "idx": int(indices[i]), "label": label, "pred": pred,
                "correct": int(pred == label)
            })

    # ── Report ────────────────────────────────────────────────────────────────
    print("\n=== MedQA Utility Results ===")
    print(f"  n={len(samples)}  model={args.model}  L{args.layer}  f={args.feature}  α={args.alpha}")
    print()
    summary_rows = []
    for mode in modes:
        rows = results[mode]
        acc = np.mean([r["correct"] for r in rows])
        per_class = {}
        for lbl in ["A", "B", "C", "D"]:
            sub = [r for r in rows if r["label"] == lbl]
            per_class[lbl] = (np.mean([r["correct"] for r in sub]) if sub else float("nan"),
                              len(sub))
        cls_str = "  ".join(f"{l}={v:.3f}(n={n})" for l, (v, n) in per_class.items())
        print(f"  {mode:10s}  overall={acc:.3f}  {cls_str}")
        summary_rows.append({"mode": mode, "accuracy": round(acc, 4), "n": len(rows)})

        out_csv = Path(args.out_dir) / f"medqa_{args.model}_L{args.layer}_f{args.feature}_{mode}.csv"
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["idx", "label", "pred", "correct"])
            w.writeheader()
            w.writerows(rows)

    out_sum = Path(args.out_dir) / f"medqa_{args.model}_L{args.layer}_f{args.feature}_summary.csv"
    with open(out_sum, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["mode", "accuracy", "n"])
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\nSaved to {args.out_dir}")


if __name__ == "__main__":
    main()
