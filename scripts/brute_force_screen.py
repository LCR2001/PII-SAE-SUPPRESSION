"""
scripts/brute_force_screen.py

Step 5b-1 — Brute-force candidate screening at PCA-guided layer.

For one (model, layer, alpha) session: process all 4 PII types (name/id/phone/email).
Per PII type:
  1. Find PII prefix position + PII span in pairs[0:10].
  2. Compute mean |z[f]| @ PII prefix across discovery pairs (using SAE @ layer).
  3. Top-N candidates by that score.
  4. For each candidate, paired baseline / intervention forward
     (V3 hidden-space: h' = h - α·z_f·W_dec[f]).
  5. Save per-(feature, pair) deltas + per-feature aggregate.

CLI:
  CUDA_VISIBLE_DEVICES=0 python scripts/brute_force_screen.py \\
      --model gemma_2_9b --layer 10 --alpha=-2 --n-candidates 500 \\
      --pair-indices 0,1,2,3,4,5,6,7,8,9 \\
      --pii-types name,id,phone,email \\
      --out-dir results/new_paper_step5b1

  CUDA_VISIBLE_DEVICES=1 python scripts/brute_force_screen.py \\
      --model llama3_med_8b --layer 25 --alpha=2 --n-candidates 500 \\
      --pair-indices 0,1,2,3,4,5,6,7,8,9 \\
      --pii-types name,id,phone,email \\
      --out-dir results/new_paper_step5b1
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
from scripts.intervention import (
    encode_to_dense, get_sae_W_dec, gemma_hook, llama_hook, make_intervention_fn,
)

PII_TYPES = {
    "name":  ("Patient Name: ", "full_name"),
    "id":    ("Patient ID: ",   "patient_id"),
    "phone": ("Phone: ",        "phone"),
    "email": ("Email: ",        "email"),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(config.MODEL_CONFIGS.keys()))
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--alpha", type=float, required=True)
    p.add_argument("--n-candidates", type=int, default=500)
    p.add_argument("--adapter-path", default=None)
    p.add_argument("--pairs", default=os.path.join(config.DATA_DIR, "pairs.jsonl"))
    p.add_argument("--registry", default=os.path.join(config.DATA_DIR, "profile_registry.jsonl"))
    p.add_argument("--pair-indices", required=True)
    p.add_argument("--pii-types", default="name,id,phone,email")
    p.add_argument("--working-threshold", type=float, default=-0.5)
    p.add_argument("--ranking-pos", default="prefix", choices=["prefix", "span"],
                   help="prefix = single PII prefix position; span = sum |z| over PII span tokens")
    p.add_argument("--sae-expansion", type=int, default=4,
                   help="Task-SAE expansion factor (×4 default, ×8 for retrained)")
    p.add_argument("--sae-suffix", default="",
                   help="Optional task-SAE checkpoint suffix (e.g. 'no_registry_leak')")
    p.add_argument("--use-tasksae", action="store_true",
                   help="Use task-SAE checkpoint instead of pre-trained SAE")
    p.add_argument("--exclude-dead", action="store_true",
                   help="With --use-tasksae: filter dead features from candidate set")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


def adapter_default_path(model_name):
    if model_name == "gemma_2_9b":
        return os.environ.get("CHECKPOINT_DIR", "/path/to/checkpoints") + "/gemma_2_9b_lora/final"
    if model_name == "llama3_med_8b":
        return os.environ.get("CHECKPOINT_DIR", "/path/to/checkpoints") + "/llama3_med_8b_lora/final"
    if model_name == "llama3_1_8b_instruct":
        return os.environ.get("CHECKPOINT_DIR", "/path/to/checkpoints") + "/llama3_1_8b_instruct_lora/final"
    if model_name == "qwen3_8b":
        return os.environ.get("CHECKPOINT_DIR", "/path/to/checkpoints") + "/qwen3_8b_lora/final"
    return None


def get_pii_span(tokenizer, text, marker, value):
    """Returns (full_ids, prompt_len, pii_end_len). PII prefix position = prompt_len - 1."""
    marker_no_trail = marker.rstrip(" ")
    pos = text.find(marker)
    if pos < 0:
        return None, None, None
    prompt_text = text[: pos + len(marker_no_trail)]
    pii_end_text = text[: pos + len(marker) + len(value)]
    prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=True).input_ids[0]
    pii_end_ids = tokenizer(pii_end_text, return_tensors="pt", add_special_tokens=True).input_ids[0]
    full_ids = tokenizer(text, return_tensors="pt", add_special_tokens=True).input_ids[0]
    return full_ids, int(prompt_ids.shape[0]), int(pii_end_ids.shape[0])


def span_logprob(log_probs, full_ids, prompt_len, pii_end_len):
    if prompt_len <= 0 or prompt_len >= pii_end_len:
        return None
    span_lps = []
    for i in range(prompt_len, pii_end_len):
        if i == 0 or i >= full_ids.shape[0]:
            continue
        tok_id = int(full_ids[i].item())
        span_lps.append(float(log_probs[i - 1, tok_id].item()))
    if not span_lps:
        return None
    return sum(span_lps) / len(span_lps)


@torch.no_grad()
def forward_logprobs(model, model_type, full_ids, first_device):
    tokens = full_ids.unsqueeze(0).to(first_device)
    if model_type == "gemma":
        out = model(tokens)
        logits = out[0] if isinstance(out, torch.Tensor) else out.logits[0]
    else:
        logits = model(tokens).logits[0]
    return F.log_softmax(logits.float().cpu(), dim=-1)


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


@torch.no_grad()
def get_z_at_positions(model, model_type, sae_wrapper, layer, full_ids, positions, first_device):
    """Sum |z[f]| over given list of positions. Returns dense [d_sae] np.array."""
    if isinstance(positions, int):
        positions = [positions]
    sae = sae_wrapper.sae
    sae_dtype = next(sae.parameters()).dtype
    tokens = full_ids.unsqueeze(0).to(first_device)
    captured = {}
    if model_type == "gemma":
        def cap(activation, hook=None):
            captured["h"] = activation[0, :, :].detach().clone()
            return activation
        model.add_hook(f"blocks.{layer}.hook_resid_post", cap)
        try:
            _ = model(tokens)
        finally:
            model.reset_hooks()
    else:
        def cap(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            captured["h"] = h[0, :, :].detach().clone()
            return output
        handle = model.model.layers[layer].register_forward_hook(cap)
        try:
            _ = model(tokens)
        finally:
            handle.remove()
    h_full = captured["h"].to(sae_dtype)  # [seq, d_model]
    d_sae = sae.num_latents if sae_wrapper.backend == "sparsify" else sae.cfg.d_sae
    accum = np.zeros(d_sae, dtype=np.float64)
    for pos in positions:
        if pos < 0 or pos >= h_full.shape[0]:
            continue
        h_p = h_full[pos]
        if sae_wrapper.backend == "saelens":
            z = sae.encode(h_p).float().cpu().numpy()
            accum += np.abs(z)
        else:
            enc = sae.encode(h_p)
            for ai, idx in zip(enc.top_acts.tolist(), enc.top_indices.tolist()):
                accum[int(idx)] += abs(float(ai))
    return accum


def build_pair_data(tokenizer, pairs, registry, indices, pii_type):
    marker, field = PII_TYPES[pii_type]
    out = []
    for pi in indices:
        if pi >= len(pairs):
            continue
        pair = pairs[pi]
        rec = registry.get(pair.get("entity_id"))
        if rec is None:
            continue
        value = str(rec.get(field, ""))
        if not value:
            continue
        full_ids, plen, pelen = get_pii_span(tokenizer, pair["p_l"], marker, value)
        if full_ids is None or plen is None or plen >= pelen:
            continue
        out.append({
            "pair_idx": pi,
            "full_ids": full_ids,
            "prompt_len": plen,
            "pii_end_len": pelen,
            "pii_pos": plen - 1,
            "value": value,
        })
    return out


def main():
    args = parse_args()
    cfg = config.MODEL_CONFIGS[args.model]
    d_sae = cfg["sae_dim"]
    pair_indices = [int(x) for x in args.pair_indices.split(",")]
    pii_types = args.pii_types.split(",")
    adapter = args.adapter_path or adapter_default_path(args.model)

    print(f"[main] {args.model} L{args.layer} α={args.alpha} n_candidates={args.n_candidates}")
    print(f"[main] PII types: {pii_types}  pairs n={len(pair_indices)}")

    with open(args.pairs, encoding="utf-8") as f:
        pairs = [json.loads(line) for line in f if line.strip()]
    with open(args.registry, encoding="utf-8") as f:
        registry = {r["entity_id"]: r for r in (json.loads(line) for line in f if line.strip())}

    print(f"[load] {args.model} (adapter={adapter}) ...")
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
    alive_mask = None  # for dead-feature filter
    if args.use_tasksae:
        from scripts.tasksae_glue import load_tasksae_for_intervention, get_alive_features
        sae_wrapper = load_tasksae_for_intervention(args.model, args.layer, str(first_device),
                                                     expansion=args.sae_expansion,
                                                     suffix=args.sae_suffix)
        sae = sae_wrapper.sae
        backend = sae_wrapper.backend
        d_sae = sae.d_sae
        W_dec = get_sae_W_dec(sae, d_sae, cfg["d_model"]).to(first_device)
        print(f"  [task-SAE] d_sae={d_sae}, k={sae.k}")
        if args.exclude_dead:
            alive_set = set(get_alive_features(args.model, args.layer,
                                                 expansion=args.sae_expansion,
                                                 suffix=args.sae_suffix))
            alive_mask = np.zeros(d_sae, dtype=bool)
            for i in alive_set:
                alive_mask[i] = True
            print(f"  [task-SAE] alive features: {alive_mask.sum()}/{d_sae} "
                  f"({alive_mask.sum()/d_sae*100:.1f}%)")
    else:
        sae_wrapper = loader.load_sae(args.model, args.layer, str(first_device))
        sae = sae_wrapper.sae
        backend = sae_wrapper.backend
        W_dec = get_sae_W_dec(sae, d_sae, cfg["d_model"]).to(first_device)

    os.makedirs(args.out_dir, exist_ok=True)

    all_summary = []  # for cross-PII-type final report
    for pii_type in pii_types:
        print(f"\n{'=' * 80}\nPII type: {pii_type}\n{'=' * 80}")
        pair_data = build_pair_data(tokenizer, pairs, registry, pair_indices, pii_type)
        print(f"  valid pairs: {len(pair_data)}")
        if not pair_data:
            print(f"  [skip] no valid pairs for {pii_type}")
            continue

        # ── candidate ranking ─────────────────────────────────────────────
        if args.ranking_pos == "prefix":
            ranking_label = f"@ {pii_type} prefix position"
        else:
            ranking_label = f"sum over {pii_type} span positions"
        print(f"\n  [candidates] computing mean |z[f]| {ranking_label} ...")
        sum_abs = np.zeros(d_sae, dtype=np.float64)
        for d in pair_data:
            if args.ranking_pos == "prefix":
                positions = [d["pii_pos"]]
            else:
                positions = list(range(d["prompt_len"], d["pii_end_len"]))
            a = get_z_at_positions(model, cfg["model_type"], sae_wrapper, args.layer,
                                    d["full_ids"], positions, first_device)
            sum_abs += a
        mean_abs = sum_abs / len(pair_data)
        if alive_mask is not None:
            # zero out dead features so they fall to bottom of ranking
            mean_abs_filtered = mean_abs.copy()
            mean_abs_filtered[~alive_mask] = -1.0
            top_idx = np.argsort(mean_abs_filtered)[::-1][: args.n_candidates].tolist()
            print(f"  [dead-filter] excluded {(~alive_mask).sum()} dead features from candidate pool")
        else:
            top_idx = np.argsort(mean_abs)[::-1][: args.n_candidates].tolist()
        print(f"  top-5: {top_idx[:5]}")
        print(f"  |z| @ top-1 / top-100 / top-500: "
              f"{mean_abs[top_idx[0]]:.4f} / {mean_abs[top_idx[99]]:.4f} / "
              f"{mean_abs[top_idx[-1]]:.4f}")

        # ── baselines (no hook) ────────────────────────────────────────────
        baseline = {}
        for d in pair_data:
            log_probs = forward_logprobs(model, cfg["model_type"], d["full_ids"], first_device)
            baseline[d["pair_idx"]] = span_logprob(log_probs, d["full_ids"],
                                                    d["prompt_len"], d["pii_end_len"])

        # ── per candidate × pair intervention ─────────────────────────────
        print(f"\n  [intervention] {args.n_candidates} candidates × {len(pair_data)} pairs at α={args.alpha} ...")
        rows = []
        for ci, feat in enumerate(top_idx):
            if ci % 50 == 0:
                print(f"    [{ci}/{len(top_idx)}] feat={feat}")
            for d in pair_data:
                seed_val = abs(args.seed + d["pair_idx"] * 1000 + int(args.alpha * 100) + feat)
                torch.manual_seed(seed_val); np.random.seed(seed_val)
                int_fn = make_intervention_fn(sae, W_dec, backend, d_sae, [feat], args.alpha, "target")
                with apply_intervention(cfg["model_type"], model, args.layer, int_fn):
                    log_probs = forward_logprobs(model, cfg["model_type"], d["full_ids"], first_device)
                inter = span_logprob(log_probs, d["full_ids"], d["prompt_len"], d["pii_end_len"])
                base = baseline.get(d["pair_idx"])
                if inter is None or base is None:
                    continue
                rows.append({
                    "feature": int(feat),
                    "pii_type": pii_type,
                    "alpha": args.alpha,
                    "pair_idx": d["pair_idx"],
                    "value": d["value"],
                    "lp_base": round(base, 4),
                    "lp_int": round(inter, 4),
                    "delta_lp": round(inter - base, 4),
                })

        # save per-condition CSV (include ranking-pos in filename so prefix/span don't overwrite)
        alpha_str = f"{args.alpha:+g}".replace("+", "p").replace("-", "m")
        out_csv = os.path.join(args.out_dir,
                               f"{args.model}_L{args.layer}_{pii_type}_alpha{alpha_str}_{args.ranking_pos}_screening.csv")
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

        # aggregate
        g = defaultdict(list)
        for r in rows:
            g[r["feature"]].append(r["delta_lp"])
        summary = []
        for feat, vals in g.items():
            n = len(vals); m = sum(vals)/n
            sd = st.stdev(vals) if n>1 else 0
            se = sd/math.sqrt(n) if n>1 else 0
            ci = 1.96*se
            summary.append((feat, n, m, m-ci, m+ci))
        summary.sort(key=lambda x: x[2])
        workers = [s for s in summary if s[2] < args.working_threshold]
        n_sig = sum(1 for s in summary if s[3] < 0 < s[4] is False and s[4] < 0)  # CI fully neg

        print(f"\n  Working features (mean Δ_lp < {args.working_threshold}): {len(workers)} / {len(summary)}")
        print(f"  Top-10 most suppressive:")
        print(f"  {'feat':>8s}  {'n':>3s}  {'mean Δlp':>10s}  {'95% CI':>26s}")
        for feat, n, m, lo, hi in summary[:10]:
            print(f"  {feat:>8d}  {n:>3d}  {m:>+10.4f}  [{lo:>+8.4f}, {hi:>+8.4f}]")

        all_summary.append({
            "pii_type": pii_type, "alpha": args.alpha,
            "n_candidates": len(summary), "n_workers": len(workers),
            "top_workers": [s[0] for s in summary[:5]],
            "top_deltas": [round(s[2], 4) for s in summary[:5]],
        })

    # final cross-PII summary
    final_csv = os.path.join(args.out_dir,
                             f"{args.model}_L{args.layer}_alpha{alpha_str}_{args.ranking_pos}_summary.csv")
    with open(final_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["pii_type", "alpha", "n_candidates", "n_workers",
                                          "top_workers", "top_deltas"])
        w.writeheader()
        for s in all_summary:
            s_out = dict(s)
            s_out["top_workers"] = ",".join(str(x) for x in s_out["top_workers"])
            s_out["top_deltas"] = ",".join(str(x) for x in s_out["top_deltas"])
            w.writerow(s_out)

    print("\n" + "=" * 80)
    print(f"FINAL SUMMARY — {args.model} L{args.layer} α={args.alpha}")
    print("=" * 80)
    for s in all_summary:
        print(f"  {s['pii_type']:>6s}: workers={s['n_workers']}/{s['n_candidates']}  "
              f"top-5 feat={s['top_workers']}  Δ={s['top_deltas']}")


if __name__ == "__main__":
    main()
