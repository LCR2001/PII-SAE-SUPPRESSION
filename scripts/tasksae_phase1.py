import os
"""
scripts/tasksae_phase1.py

Phase 1 sanity diagnostics for new task-specific SAEs.

Per (model, layer) checkpoint:
  1. Dead feature fraction (from steps_since_active vs dead_threshold)
  2. Reconstruction error on n=20 pairs:
       - all-positions mean rel_err
       - PII-prefix-position rel_err
       - cosine(h, recon)
  3. Comparison vs pre-trained SAE (from existing recon_error CSVs)

CLI:
  CUDA_VISIBLE_DEVICES=0 python scripts/tasksae_phase1.py \\
      --model gemma_2_9b --layers 9 20 31 \\
      --adapter-path <CHECKPOINT_DIR>/<model>_lora/final \\
      --n-pairs 20 \\
      --out results/new_sae_diagnostics/gemma_phase1.csv

  CUDA_VISIBLE_DEVICES=1 python scripts/tasksae_phase1.py \\
      --model llama3_med_8b --layers 8 16 24 \\
      --adapter-path <CHECKPOINT_DIR>/<model>_lora/final \\
      --n-pairs 20 \\
      --out results/new_sae_diagnostics/llama_phase1.csv
"""

import argparse
import csv
import json
import math
import os
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import config
import models.loader as loader
from scripts.tasksae_wrapper import load_tasksae, dead_fraction


SAE_BASE = os.environ.get("SAE_CHECKPOINT_DIR", "/path/to/SAE/checkpoints")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(config.MODEL_CONFIGS.keys()))
    p.add_argument("--layers", nargs="+", type=int, required=True)
    p.add_argument("--adapter-path", default=None)
    p.add_argument("--pairs", default=os.path.join(config.DATA_DIR, "pairs.jsonl"))
    p.add_argument("--n-pairs", type=int, default=20)
    p.add_argument("--sae-expansion", type=int, default=4)
    p.add_argument("--sae-suffix", default="")
    p.add_argument("--out", required=True)
    return p.parse_args()


def adapter_default_path(model_name):
    if model_name == "gemma_2_9b":
        return "<CHECKPOINT_DIR>/<model>_lora/final
    if model_name == "llama3_med_8b":
        return "<CHECKPOINT_DIR>/<model>_lora/final
    if model_name == "llama3_1_8b_instruct":
        return "<CHECKPOINT_DIR>/<model>_lora/final
    if model_name == "qwen3_8b":
        return "<CHECKPOINT_DIR>/<model>_lora/final
    return None


SHORT_MAP = {
    "gemma_2_9b": "gemma_2_9b",
    "llama3_med_8b": "llama3_med_8b",
    "llama3_1_8b_instruct": "llama_31_8b_inst",
    "qwen3_8b": "qwen3_8b",
}


def ckpt_dir_for(model, layer, expansion=4, suffix=""):
    short = SHORT_MAP[model]
    base = f"{short}_L{layer}_x{expansion}_k32"
    if suffix:
        base = f"{base}_{suffix}"
    return f"{SAE_BASE}/{base}"


def find_pii_prefix_pos(tokenizer, p_l_text, marker="Patient Name: "):
    pos = p_l_text.find(marker)
    if pos < 0:
        return None
    prefix_text = p_l_text[: pos + len(marker.rstrip(" "))]
    prefix_ids = tokenizer(prefix_text, return_tensors="pt", add_special_tokens=True).input_ids[0]
    return int(prefix_ids.shape[0]) - 1


@torch.no_grad()
def get_hidden_at_layer(model, model_type, full_ids, layer, first_device):
    tokens = full_ids.unsqueeze(0).to(first_device)
    if model_type == "gemma":
        _, cache = model.run_with_cache(
            tokens, names_filter=[f"blocks.{layer}.hook_resid_post"],
        )
        return cache[f"blocks.{layer}.hook_resid_post"][0]
    else:
        out = model(tokens, output_hidden_states=True)
        return out.hidden_states[layer + 1][0]


def main():
    args = parse_args()
    cfg = config.MODEL_CONFIGS[args.model]
    adapter = args.adapter_path or adapter_default_path(args.model)

    print(f"[main] {args.model} layers={args.layers}")

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

    with open(args.pairs, encoding="utf-8") as f:
        pairs = [json.loads(line) for line in f if line.strip()][: args.n_pairs]
    print(f"[main] using {len(pairs)} pairs")

    rows = []
    for layer in args.layers:
        ckpt = ckpt_dir_for(args.model, layer,
                             expansion=args.sae_expansion, suffix=args.sae_suffix)
        print(f"\n[layer {layer}] loading TaskSAE from {ckpt} ...")
        # 1. dead fraction (no model needed)
        dead_info = dead_fraction(ckpt)
        print(f"  dead: {dead_info['n_dead']}/{dead_info['d_sae']} "
              f"({dead_info['dead_fraction']*100:.1f}%)")

        wrapper = load_tasksae(ckpt, device=str(first_device))
        sae = wrapper.sae

        rels_all, rels_pii, cos_all, cos_pii = [], [], [], []
        for pi, pair in enumerate(pairs):
            p_l = pair["p_l"]
            full_ids = tokenizer(p_l, return_tensors="pt",
                                  add_special_tokens=True).input_ids[0]
            pii_pos = find_pii_prefix_pos(tokenizer, p_l)
            if pii_pos is None or pii_pos >= full_ids.shape[0]:
                continue

            hidden = get_hidden_at_layer(model, cfg["model_type"], full_ids, layer, first_device)
            h = hidden.float().to(first_device)
            # encode + decode all positions
            enc = sae.encode(h)
            recon = sae.decode(enc.top_acts, enc.top_indices)
            with torch.no_grad():
                diff = h - recon
                abs_err = diff.norm(dim=-1)
                h_norm = h.norm(dim=-1).clamp_min(1e-9)
                rel_err = (abs_err / h_norm).cpu().numpy()
                cos = torch.nn.functional.cosine_similarity(h, recon, dim=-1).cpu().numpy()
            rels_all.append(float(rel_err.mean()))
            rels_pii.append(float(rel_err[pii_pos]))
            cos_all.append(float(cos.mean()))
            cos_pii.append(float(cos[pii_pos]))

        rels_all = np.array(rels_all); rels_pii = np.array(rels_pii)
        cos_all = np.array(cos_all); cos_pii = np.array(cos_pii)
        row = {
            "model": args.model, "layer": layer, "n_pairs": len(rels_all),
            "d_sae": dead_info["d_sae"], "k": sae.k,
            "n_dead": dead_info["n_dead"],
            "dead_frac": round(dead_info["dead_fraction"], 4),
            "rel_err_all_pos_mean": round(float(rels_all.mean()), 4),
            "rel_err_all_pos_std":  round(float(rels_all.std()), 4),
            "rel_err_pii_pos_mean": round(float(rels_pii.mean()), 4),
            "rel_err_pii_pos_std":  round(float(rels_pii.std()), 4),
            "cos_all_pos_mean":     round(float(cos_all.mean()), 4),
            "cos_pii_pos_mean":     round(float(cos_pii.mean()), 4),
        }
        rows.append(row)
        print(f"  rel_err_all_pos = {row['rel_err_all_pos_mean']:.4f} ± {row['rel_err_all_pos_std']:.4f}")
        print(f"  rel_err_pii_pos = {row['rel_err_pii_pos_mean']:.4f} ± {row['rel_err_pii_pos_std']:.4f}")
        print(f"  cos(all_pos)    = {row['cos_all_pos_mean']:.4f}")
        print(f"  cos(pii_pos)    = {row['cos_pii_pos_mean']:.4f}")

        # release SAE memory
        del wrapper, sae
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[main] wrote {args.out}")


if __name__ == "__main__":
    main()
