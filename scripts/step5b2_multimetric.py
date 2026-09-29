"""
scripts/step5b2_multimetric.py

Step 5b-2 Stage 2 — Multi-metric (Δ_lp + Δb_rank + first-token rank) on all
T0-T8 × 4 PII × hero feature conditions at val50.

Same hooks/intervention as step5b2_eval.py, but per pair also records:
  - b_rank_base, b_rank_int  (sum of log2(rank_t) over PII span)
  - first_rank_base, first_rank_int  (rank of gold first PII token in baseline / intervention)

Top-k inclusion is computed post-hoc from stored first-token rank.

CLI:
  CUDA_VISIBLE_DEVICES=0 python scripts/step5b2_multimetric.py \\
      --model gemma_2_9b --layer 10 --alpha=-2 \\
      --feature-map "name=2759,id=2759,phone=2759,email=2759" \\
      --templates T0,T1,T2,T3,T4,T5,T6,T7,T8 \\
      --pair-indices 300,...,349 \\
      --out-dir results/new_paper_step5b2_multimetric
"""

import argparse
import csv
import json
import math
import os
import statistics as st
import sys
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import config
import models.loader as loader
from scripts.intervention import get_sae_W_dec, gemma_hook, llama_hook, make_intervention_fn
from scripts.step5b2_eval import (
    TEMPLATES, PII_FIELD,
    parse_feature_map, adapter_default_path,
    get_pii_span, span_logprob, build_pair_data_for_template,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(config.MODEL_CONFIGS.keys()))
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--alpha", type=float, required=True)
    p.add_argument("--feature-map", required=True)
    p.add_argument("--templates", required=True)
    p.add_argument("--adapter-path", default=None)
    p.add_argument("--pairs", default=os.path.join(config.DATA_DIR, "pairs.jsonl"))
    p.add_argument("--registry", default=os.path.join(config.DATA_DIR, "profile_registry.jsonl"))
    p.add_argument("--pair-indices", required=True)
    p.add_argument("--mode", default="target",
                   choices=["target", "random_vector", "random_features", "random_encoder"])
    p.add_argument("--use-tasksae", action="store_true",
                   help="Use task-specific SAE from <SAE_CHECKPOINT_DIR>/")
    p.add_argument("--sae-expansion", type=int, default=4,
                   help="Task-SAE expansion factor (×4 default, ×8 for retrained)")
    p.add_argument("--sae-suffix", default="",
                   help="Optional task-SAE checkpoint suffix (e.g. 'no_registry_leak')")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


@torch.no_grad()
def forward_logits(model, model_type, full_ids, first_device):
    """Return raw logits [seq, vocab]; no log_softmax (we need ranks too)."""
    tokens = full_ids.unsqueeze(0).to(first_device)
    if model_type == "gemma":
        out = model(tokens)
        logits = out[0] if isinstance(out, torch.Tensor) else out.logits[0]
    else:
        logits = model(tokens).logits[0]
    return logits.float().cpu()


def span_metrics(logits, full_ids, prompt_len, pii_end_len):
    """Returns dict with lp_mean (mean per-token logprob over span), b_rank (sum
    log2(rank_t)), first_rank (rank of token at position prompt_len)."""
    if prompt_len <= 0 or prompt_len >= pii_end_len:
        return None
    span_lps = []
    span_ranks = []
    for i in range(prompt_len, pii_end_len):
        if i == 0 or i >= full_ids.shape[0]:
            continue
        tok_id = int(full_ids[i].item())
        pred_logits = logits[i - 1]
        # logprob
        log_p = F.log_softmax(pred_logits, dim=-1)
        span_lps.append(float(log_p[tok_id].item()))
        # rank: number of tokens with logit strictly greater + 1
        rank = int((pred_logits > pred_logits[tok_id]).sum().item()) + 1
        span_ranks.append(rank)
    if not span_lps:
        return None
    return {
        "lp_mean": sum(span_lps) / len(span_lps),
        "b_rank": sum(math.log2(r) for r in span_ranks),
        "first_rank": span_ranks[0],
        "n_tokens": len(span_ranks),
        "ranks": span_ranks,
    }


@contextmanager
def apply_intervention(model_type, model, layer, intervention_fn):
    if model_type == "gemma":
        model.add_hook(f"blocks.{layer}.hook_resid_post", gemma_hook(intervention_fn))
        try:
            yield
        finally:
            model.reset_hooks()
    else:
        handle = model.model.layers[layer].register_forward_hook(llama_hook(intervention_fn))
        try:
            yield
        finally:
            handle.remove()


def main():
    args = parse_args()
    cfg = config.MODEL_CONFIGS[args.model]
    d_sae = cfg["sae_dim"]
    pair_indices = [int(x) for x in args.pair_indices.split(",")]
    template_ids = args.templates.split(",")
    feat_map = parse_feature_map(args.feature_map)
    adapter = args.adapter_path or adapter_default_path(args.model)

    print(f"[main] {args.model} L{args.layer} α={args.alpha}")
    print(f"[main] feature map: {feat_map}")
    print(f"[main] templates: {template_ids}  pairs n={len(pair_indices)}")

    with open(args.pairs, encoding="utf-8") as f:
        pairs = [json.loads(line) for line in f if line.strip()]
    with open(args.registry, encoding="utf-8") as f:
        registry = {r["entity_id"]: r for r in (json.loads(line) for line in f if line.strip())}

    print(f"[load] {args.model} ...")
    if cfg["model_type"] == "gemma":
        model = loader.load_model(args.model, config.MODEL_DEVICE, adapter_path=adapter)
        tokenizer = model.tokenizer
        first_device = model.cfg.device
    else:
        model, tokenizer = loader.load_model(args.model, config.MODEL_DEVICE, adapter_path=adapter)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        first_device = next(model.parameters()).device
    model.eval()

    print(f"[load] SAE L{args.layer} ...")
    if args.use_tasksae:
        from scripts.tasksae_glue import load_tasksae_for_intervention
        sae_wrapper = load_tasksae_for_intervention(args.model, args.layer, str(first_device),
                                                     expansion=args.sae_expansion,
                                                     suffix=args.sae_suffix)
        sae = sae_wrapper.sae
        backend = sae_wrapper.backend
        d_sae = sae.d_sae   # override with task-SAE actual dim
        W_dec = get_sae_W_dec(sae, d_sae, cfg["d_model"]).to(first_device)
        print(f"  [task-SAE] d_sae={d_sae}, k={sae.k}")
    else:
        sae_wrapper = loader.load_sae(args.model, args.layer, str(first_device))
        sae = sae_wrapper.sae
        backend = sae_wrapper.backend
        W_dec = get_sae_W_dec(sae, d_sae, cfg["d_model"]).to(first_device)

    os.makedirs(args.out_dir, exist_ok=True)
    for tid in template_ids:
        for pii_type, feat in feat_map.items():
            print(f"\n=== {tid} × {pii_type} × feat {feat} ===")
            pair_data = build_pair_data_for_template(tokenizer, pairs, registry, pair_indices, pii_type, tid)
            print(f"  valid pairs: {len(pair_data)}")
            if not pair_data:
                continue

            # baseline
            baseline = {}
            for d in pair_data:
                logits = forward_logits(model, cfg["model_type"], d["full_ids"], first_device)
                baseline[d["pair_idx"]] = span_metrics(logits, d["full_ids"], d["prompt_len"], d["pii_end_len"])

            # intervention
            rows = []
            for d in pair_data:
                seed_val = abs(args.seed + d["pair_idx"] * 1000 + int(args.alpha * 100) + feat + hash(tid) % 1000)
                torch.manual_seed(seed_val); np.random.seed(seed_val)
                int_fn = make_intervention_fn(sae, W_dec, backend, d_sae, [feat], args.alpha, args.mode)
                with apply_intervention(cfg["model_type"], model, args.layer, int_fn):
                    logits = forward_logits(model, cfg["model_type"], d["full_ids"], first_device)
                im = span_metrics(logits, d["full_ids"], d["prompt_len"], d["pii_end_len"])
                bm = baseline.get(d["pair_idx"])
                if im is None or bm is None:
                    continue
                rows.append({
                    "feature": feat, "template": tid, "pii_type": pii_type, "alpha": args.alpha,
                    "pair_idx": d["pair_idx"], "value": d["value"],
                    "n_tokens": im["n_tokens"],
                    "lp_base": round(bm["lp_mean"], 4),
                    "lp_int":  round(im["lp_mean"], 4),
                    "delta_lp": round(im["lp_mean"] - bm["lp_mean"], 4),
                    "b_rank_base": round(bm["b_rank"], 4),
                    "b_rank_int":  round(im["b_rank"], 4),
                    "delta_b_rank": round(im["b_rank"] - bm["b_rank"], 4),
                    "first_rank_base": bm["first_rank"],
                    "first_rank_int":  im["first_rank"],
                })

            alpha_str = f"{args.alpha:+g}".replace("+", "p").replace("-", "m")
            mode_tag = "" if args.mode == "target" else f"_{args.mode}"
            out_csv = os.path.join(args.out_dir,
                                    f"{args.model}_L{args.layer}_{pii_type}_{tid}_alpha{alpha_str}{mode_tag}.csv")
            with open(out_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)

            # aggregate
            deltas_lp  = [r["delta_lp"]      for r in rows]
            deltas_br  = [r["delta_b_rank"]  for r in rows]
            def stats(v):
                n = len(v); m = sum(v)/n
                sd = st.stdev(v) if n>1 else 0
                se = sd/math.sqrt(n) if n>1 else 0
                ci = 1.96*se
                return n, m, m-ci, m+ci
            n, m_lp, lo_lp, hi_lp = stats(deltas_lp)
            _, m_br, lo_br, hi_br = stats(deltas_br)
            print(f"  Δ_lp:     n={n}  mean={m_lp:+.4f}  CI [{lo_lp:+.4f}, {hi_lp:+.4f}]")
            print(f"  Δb_rank:  n={n}  mean={m_br:+.4f}  CI [{lo_br:+.4f}, {hi_br:+.4f}]")


if __name__ == "__main__":
    main()
