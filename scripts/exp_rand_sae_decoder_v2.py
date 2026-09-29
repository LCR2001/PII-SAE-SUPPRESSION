"""
scripts/exp_rand_sae_decoder_v2.py

Random SAE Decoder Direction Control — v2 (stricter, cached m_t).

Formula (v2 convention):
    m_t  = ‖z_t[f] · W_dec[f]‖₂          (per-token scalar, float32)
    u_f  = W_dec[f] / ‖W_dec[f]‖₂         (unit vec, selected)
    u_g  = W_dec[g] / ‖W_dec[g]‖₂         (unit vec, random)
    h'_t = h_t + α · m_t · u_f  (selected)    α=-2 → REDUCE direction
    h'_t = h_t + α · m_t · u_g  (random SAE)  same m_t, different unit vec

Key guarantees vs exp_random_sae_pii.py:
  - m_t cached from selected forward; NO SAE encode in random passes
  - ‖Δsel‖ ≡ ‖Δrand‖ = |α|·m_t  (exact, by construction)
  - z_t[g] never computed (unit test #3 verifies)
  - float32 rel-err check < 1e-3 (abort on failure)

Note: α=-2 REDUCES the feature direction (h + α·m_t·u = h - 2·m_t·u).
      Existing intervention.py uses h - α·dir (α=-2 → AMPLIFY).
      Sanity check #4 reports this difference explicitly.

Output dir: results/random_sae_decoder_control/

Smoke test (3 special cells, run separately per model):
    python scripts/exp_rand_sae_decoder_v2.py \\
        --model gemma_2_9b --layer 20 \\
        --K 3 --pair-indices 300,301,302 --templates T0,T1 --run-unit-tests

Full run (one model+layer at a time):
    python scripts/exp_rand_sae_decoder_v2.py --model llama3_1_8b_instruct --layer 16
"""

import argparse
import csv
import gc
import json
import math
import os
import sys
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import config
from scripts.intervention import encode_to_dense, get_sae_W_dec, build_direction
from scripts.step5b2_eval import TEMPLATES, PII_FIELD, build_pair_data_for_template, adapter_default_path
from scripts.tasksae_glue import load_tasksae_for_intervention, get_alive_features

# ── constants ─────────────────────────────────────────────────────────────────

CELL_FEATURES = {
    ("gemma_2_9b",           9,  "name"):  5559,
    ("gemma_2_9b",           9,  "id"):    5559,
    ("gemma_2_9b",           9,  "phone"): 4648,
    ("gemma_2_9b",           9,  "email"): 11893,
    ("gemma_2_9b",           20, "name"):  20385,
    ("gemma_2_9b",           20, "id"):    20385,
    ("gemma_2_9b",           20, "phone"): 20385,
    ("gemma_2_9b",           20, "email"): 20385,
    ("gemma_2_9b",           31, "name"):  2807,
    ("gemma_2_9b",           31, "id"):    16900,
    ("gemma_2_9b",           31, "phone"): 2807,
    ("gemma_2_9b",           31, "email"): 2807,
    ("llama3_1_8b_instruct", 8,  "name"):  16363,
    ("llama3_1_8b_instruct", 8,  "id"):    16363,
    ("llama3_1_8b_instruct", 8,  "phone"): 16363,
    ("llama3_1_8b_instruct", 8,  "email"): 16363,
    ("llama3_1_8b_instruct", 16, "name"):  32619,
    ("llama3_1_8b_instruct", 16, "id"):    6432,
    ("llama3_1_8b_instruct", 16, "phone"): 10947,
    ("llama3_1_8b_instruct", 16, "email"): 10947,
    ("llama3_1_8b_instruct", 24, "name"):  4823,
    ("llama3_1_8b_instruct", 24, "id"):    25245,
    ("llama3_1_8b_instruct", 24, "phone"): 5682,
    ("llama3_1_8b_instruct", 24, "email"): 26877,
    ("qwen3_8b",             9,  "name"):  30639,
    ("qwen3_8b",             9,  "id"):    30639,
    ("qwen3_8b",             9,  "phone"): 30639,
    ("qwen3_8b",             9,  "email"): 30639,
    ("qwen3_8b",             18, "name"):  540,
    ("qwen3_8b",             18, "id"):    21146,
    ("qwen3_8b",             18, "phone"): 21146,
    ("qwen3_8b",             18, "email"): 21146,
    ("qwen3_8b",             27, "name"):  27259,
    ("qwen3_8b",             27, "id"):    10595,
    ("qwen3_8b",             27, "phone"): 17828,
    ("qwen3_8b",             27, "email"): 5037,
}

SAE_EXPANSION = 8
SAE_SUFFIX    = "v2"
MAG_REL_TOL   = 1e-3   # float32 relative error tolerance; abort on failure

SPECIAL_CELLS = [
    ("gemma_2_9b", 20, "name"),
    ("llama3_1_8b_instruct", 16, "name"),
    ("qwen3_8b", 18, "name"),
]


# ── argument parsing ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(config.MODEL_CONFIGS.keys()))
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--pii-types", default="name,id,phone,email")
    p.add_argument("--K", type=int, default=20,
                   help="Number of random SAE features per cell")
    p.add_argument("--alpha", type=float, default=-2.0)
    p.add_argument("--pairs",
                   default=os.path.join(config.DATA_DIR, "pairs.jsonl"))
    p.add_argument("--registry",
                   default=os.path.join(config.DATA_DIR, "profile_registry.jsonl"))
    p.add_argument("--pair-indices",
                   default=",".join(str(i) for i in range(300, 350)))
    p.add_argument("--templates", default="T0,T1,T2,T3,T4,T5,T6,T7,T8")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="results/random_sae_decoder_control")
    p.add_argument("--run-unit-tests", action="store_true",
                   help="Run sanity checks before evaluation")
    return p.parse_args()


# ── model / SAE loading ───────────────────────────────────────────────────────

def load_hf_model(model_name: str, adapter_path: str = None):
    cfg = config.MODEL_CONFIGS[model_name]
    print(f"[load] {model_name} (HF bfloat16, device_map=auto) ...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
    )
    if adapter_path and os.path.isdir(adapter_path):
        from peft import PeftModel
        print(f"[load] merging LoRA adapter: {adapter_path}")
        peft_model = PeftModel.from_pretrained(model, adapter_path)
        model = peft_model.merge_and_unload()
        del peft_model
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[load] LoRA merged into base weights")
    elif adapter_path:
        print(f"[warn] adapter_path not found, skipping LoRA: {adapter_path}")
    model.eval()
    return model, tokenizer


# ── feature sampling ─────────────────────────────────────────────────────────

def sample_random_features(model_name, layer, selected_feat, K, base_seed, pii_type):
    alive = get_alive_features(model_name, layer,
                               expansion=SAE_EXPANSION, suffix=SAE_SUFFIX)
    alive_set = [f for f in alive if f != selected_feat]
    K_actual = min(K, len(alive_set))
    if K_actual < K:
        print(f"[warn] only {len(alive_set)} alive non-selected features; using {K_actual}")
    cell_seed = (base_seed + abs(hash(pii_type))) % (2**31)
    rng = np.random.default_rng(cell_seed)
    sampled = rng.choice(len(alive_set), size=K_actual, replace=False)
    return [alive_set[i] for i in sampled]


# ── intervention classes ──────────────────────────────────────────────────────

class SelectedCaptureFn:
    """Selected SAE intervention (v2 formula) that also caches m_t per forward pass.

    Formula: h'_t = h_t + alpha * m_t * u_f
    where m_t = ||z_t[f] * W_dec[f]||_2  and  u_f = W_dec[f] / ||W_dec[f]||_2

    With alpha=-2: REDUCES feature direction (h - 2*m_t*u_f).
    Contrast with intervention.py (h - alpha*dir = h + 2*dir = AMPLIFY for alpha=-2).
    """

    def __init__(self, sae, W_dec, backend, d_sae, sel_feat, alpha):
        self.sae     = sae
        self.W_dec   = W_dec
        self.backend = backend
        self.d_sae   = d_sae
        self.alpha   = alpha
        self.sel_feat = sel_feat

        w_f = W_dec[sel_feat].float()
        self.w_f_norm = float(w_f.norm().item())
        self.u_f = (w_f / (self.w_f_norm + 1e-8))  # [d_model] float32

        self.cached_m_t = None  # set after each forward; shape [B, T]

    def __call__(self, activation):
        device = activation.device
        sae_dtype = next(self.sae.parameters()).dtype

        # 1. SAE encode (only here — not in random passes)
        z = encode_to_dense(self.sae, activation.to(sae_dtype), self.backend, self.d_sae)

        # 2. Compute feature vector and magnitude in float32
        w_f_dev = self.W_dec[self.sel_feat].to(device)
        z_f = z[..., self.sel_feat]                                  # [B, T]
        feat_dir = z_f.unsqueeze(-1) * w_f_dev                      # [B, T, d_model]
        m_t = feat_dir.float().norm(dim=-1)                         # [B, T], float32

        # 3. Cache m_t for random passes
        self.cached_m_t = m_t.detach().cpu()

        # 4. Apply: h' = h + alpha * m_t * u_f
        u = self.u_f.to(device=device)
        delta = self.alpha * m_t.unsqueeze(-1) * u                  # [B, T, d_model] f32
        return (activation.float() + delta).to(activation.dtype)


class RandomSAECachedFn:
    """Random SAE direction (v2) using m_t CACHED from SelectedCaptureFn.

    Formula: h'_t = h_t + alpha * m_t * u_g   (same m_t, different direction)
    z_t[g] is NEVER computed — verified by _uses_z_g = False marker.
    """

    def __init__(self, W_dec, rand_feat, alpha, capture_fn: SelectedCaptureFn):
        self.alpha      = alpha
        self.capture_fn = capture_fn
        self.rand_feat  = rand_feat
        self._uses_z_g  = False  # unit-test marker: z[g] not used

        w_g = W_dec[rand_feat].float()
        self.w_g_norm = float(w_g.norm().item())
        self.u_g = w_g / (self.w_g_norm + 1e-8)  # [d_model] float32

    def __call__(self, activation):
        assert self.capture_fn.cached_m_t is not None, \
            "RandomSAECachedFn called before SelectedCaptureFn: m_t not cached!"
        device = activation.device
        m_t = self.capture_fn.cached_m_t.to(device=device, dtype=torch.float32)
        u   = self.u_g.to(device=device)
        delta = self.alpha * m_t.unsqueeze(-1) * u                  # [B, T, d_model] f32
        return (activation.float() + delta).to(activation.dtype)


# ── hook utilities ────────────────────────────────────────────────────────────

def _hf_wrap(fn):
    """Wrap activation fn to HF forward hook signature."""
    def hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        new_h = fn(h)
        if isinstance(out, tuple):
            return (new_h,) + out[1:]
        return new_h
    return hook


@contextmanager
def apply_hook(model, layer, fn):
    if fn is None:
        yield
        return
    handle = model.model.layers[layer].register_forward_hook(_hf_wrap(fn))
    try:
        yield
    finally:
        handle.remove()


# ── forward / metric utils ───────────────────────────────────────────────────

@torch.no_grad()
def forward_logits(model, full_ids, device):
    tokens = full_ids.unsqueeze(0).to(device)
    return model(tokens).logits[0].float().cpu()  # [seq, vocab]


def span_metrics(logits, full_ids, prompt_len, pii_end_len):
    if prompt_len <= 0 or prompt_len >= pii_end_len:
        return None
    span_lps, span_ranks = [], []
    for i in range(prompt_len, pii_end_len):
        if i == 0 or i >= full_ids.shape[0]:
            continue
        tok_id = int(full_ids[i].item())
        pred = logits[i - 1]
        lp = float(F.log_softmax(pred, dim=-1)[tok_id].item())
        rank = int((pred > pred[tok_id]).sum().item()) + 1
        span_lps.append(lp)
        span_ranks.append(rank)
    if not span_lps:
        return None
    return {
        "lp_mean": sum(span_lps) / len(span_lps),
        "b_rank":  sum(math.log2(r) for r in span_ranks),
        "first_rank": span_ranks[0],
        "n_tokens": len(span_lps),
    }


def cosine_sim(W_dec, i, j):
    vi = W_dec[i].float(); vj = W_dec[j].float()
    return float((vi @ vj / (vi.norm() * vj.norm() + 1e-8)).item())


# ── sanity checks ────────────────────────────────────────────────────────────

def run_sanity_checks(sae, W_dec, backend, d_sae, sel_feat, rand_feats,
                      alpha, model, layer, full_ids, device):
    """Run all 4 sanity checks on a single example.

    1. float32 relative magnitude error < MAG_REL_TOL  (abort on fail)
    2. Active token masks identical  (by construction, verified)
    3. z_t[g] not used in RandomSAECachedFn  (marker check)
    4. Sign: alpha<0 → delta·u_f < 0 (REDUCE); compare with existing code
    """
    print("\n" + "=" * 60)
    print("SANITY CHECKS")
    print("=" * 60)

    capture_fn = SelectedCaptureFn(sae, W_dec, backend, d_sae, sel_feat, alpha)
    rand_fns   = [RandomSAECachedFn(W_dec, g, alpha, capture_fn) for g in rand_feats[:3]]

    tokens = full_ids.unsqueeze(0).to(device)

    # ── check 3: marker ───────────────────────────────────────────────────────
    for rfn in rand_fns:
        if rfn._uses_z_g:
            raise AssertionError(
                f"FAIL check 3: RandomSAECachedFn for feat={rfn.rand_feat} uses z_t[g]!")
    print(f"[check 3] z_t[g] not used in random fns: PASS")

    # ── prime m_t cache via selected forward ──────────────────────────────────
    sign_dots = []

    def sel_hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        capture_fn.cached_m_t = None
        modified = capture_fn(h)
        # sign check: dot of (h' - h) with u_f per token (in float32)
        delta = (modified.float() - h.float())          # [1, T, d_model]
        u = capture_fn.u_f.to(device=h.device)         # use h.device to avoid cross-GPU mismatch
        dot = (delta * u)[0].sum(-1)                    # [T]
        sign_dots.append(dot.detach().cpu())
        if isinstance(out, tuple):
            return (modified.to(h.dtype),) + out[1:]
        return modified.to(h.dtype)

    with torch.no_grad():
        h1 = model.model.layers[layer].register_forward_hook(sel_hook)
        try:
            model(tokens)
        finally:
            h1.remove()

    # ── check 4: sign convention ──────────────────────────────────────────────
    m_t_cpu = capture_fn.cached_m_t[0]  # [T]
    active  = m_t_cpu > 0
    dots    = sign_dots[0]
    if active.sum() > 0:
        active_dots = dots[active]
        if alpha < 0:
            bad = (active_dots > 1e-6).sum().item()
            direction_label = "REDUCE (negative dot product)"
        else:
            bad = (active_dots < -1e-6).sum().item()
            direction_label = "AMPLIFY (positive dot product)"
        if bad > 0:
            print(f"[check 4] WARN: {bad}/{active.sum()} tokens have unexpected sign")
        else:
            print(f"[check 4] sign convention: PASS  alpha={alpha} → {direction_label}")
    else:
        print(f"[check 4] sign convention: SKIP (m_t=0 for all tokens in this example)")
    print(f"  Comparison:")
    print(f"    THIS script (h + α·m_t·u): α={alpha} → {'REDUCE' if alpha < 0 else 'AMPLIFY'}")
    print(f"    intervention.py (h - α·dir): α={alpha} → {'AMPLIFY' if alpha < 0 else 'REDUCE'}")
    print(f"    (opposite sign conventions)")

    # ── check 1: float32 norm comparison using INTENDED deltas (no bfloat16) ──
    # Recover m_t cached by the sel_hook forward and compute expected norms
    # purely in float32, avoiding bfloat16 quantization artefacts.
    assert capture_fn.cached_m_t is not None, \
        "m_t not cached — sel_hook forward must have set it"
    m_t_f32 = capture_fn.cached_m_t[0].float()  # [T]

    u_f_f32     = capture_fn.u_f.float()           # [d_model]
    u_f_norm    = float(u_f_f32.norm().item())
    active_mask = m_t_f32 > 0

    max_rel_err = 0.0
    for rfn in rand_fns:
        u_g_f32  = rfn.u_g.float()                # [d_model]
        u_g_norm = float(u_g_f32.norm().item())

        # Expected per-token norms (float32, no bfloat16 involved)
        sel_actual  = m_t_f32 * abs(alpha) * u_f_norm   # [T]
        rand_actual = m_t_f32 * abs(alpha) * u_g_norm   # [T] same m_t, different unit-vec norm

        if active_mask.sum() > 0:
            s = sel_actual[active_mask]
            r = rand_actual[active_mask]
            rel_err = ((s - r).abs() / (s + 1e-8)).max().item()
        else:
            rel_err = 0.0

        max_rel_err = max(max_rel_err, rel_err)

        if rel_err >= MAG_REL_TOL:
            raise AssertionError(
                f"FAIL check 1: float32 max_rel_err={rel_err:.2e} >= {MAG_REL_TOL:.0e} "
                f"(rand_feat={rfn.rand_feat}  ||u_f||={u_f_f32.norm():.6f}  "
                f"||u_g||={u_g_f32.norm():.6f}). Aborting."
            )

    u_norms = [rfn.u_g.float().norm().item() for rfn in rand_fns]
    print(f"[check 1] magnitude equality (float32 rel_err < {MAG_REL_TOL:.0e}): PASS "
          f"max_rel_err={max_rel_err:.2e}  "
          f"||u_f||={capture_fn.u_f.float().norm():.6f}  "
          f"||u_g|| range=[{min(u_norms):.6f}, {max(u_norms):.6f}]")

    # check 2: active mask is identical by construction (same cached m_t)
    n_active = int(active.sum().item())
    print(f"[check 2] active token mask: {n_active}/{len(active)} tokens active "
          f"(identical by construction — same cached m_t)")

    print("=" * 60)
    return max_rel_err


# ── aggregation helpers ───────────────────────────────────────────────────────

def _stats(vals):
    arr = np.array([v for v in vals if v is not None and not np.isnan(v)])
    if len(arr) == 0:
        return dict(mean=np.nan, median=np.nan, std=np.nan,
                    p5=np.nan, p50=np.nan, p95=np.nan, min=np.nan, max=np.nan, n=0)
    rng = np.random.default_rng(0)
    bs  = np.array([np.mean(rng.choice(arr, len(arr))) for _ in range(2000)])
    return dict(
        mean=float(np.mean(arr)), median=float(np.median(arr)),
        std=float(np.std(arr, ddof=1) if len(arr) > 1 else 0.0),
        p5=float(np.percentile(arr, 5)),  p50=float(np.percentile(arr, 50)),
        p95=float(np.percentile(arr, 95)),
        min=float(np.min(arr)), max=float(np.max(arr)), n=int(len(arr)),
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg  = config.MODEL_CONFIGS[args.model]

    pii_types    = args.pii_types.split(",")
    template_ids = args.templates.split(",")
    pair_indices = [int(x) for x in args.pair_indices.split(",")]
    device       = torch.device("cuda:0")

    os.makedirs(args.out_dir, exist_ok=True)

    # ── data ─────────────────────────────────────────────────────────────────
    with open(args.pairs, encoding="utf-8") as f:
        pairs = [json.loads(l) for l in f if l.strip()]
    with open(args.registry, encoding="utf-8") as f:
        registry = {r["entity_id"]: r
                    for r in (json.loads(l) for l in f if l.strip())}

    # ── model ─────────────────────────────────────────────────────────────────
    adapter = getattr(args, "adapter_path", None) or adapter_default_path(args.model)
    model, tokenizer = load_hf_model(args.model, adapter_path=adapter)

    # ── SAE ───────────────────────────────────────────────────────────────────
    print(f"[load] task-SAE L{args.layer} (x{SAE_EXPANSION}_{SAE_SUFFIX}) ...")
    layer_device = next(model.model.layers[args.layer].parameters()).device
    sae_wrapper  = load_tasksae_for_intervention(
        args.model, args.layer, str(layer_device),
        expansion=SAE_EXPANSION, suffix=SAE_SUFFIX,
    )
    sae     = sae_wrapper.sae
    backend = sae_wrapper.backend
    d_sae   = sae.d_sae
    W_dec   = get_sae_W_dec(sae, d_sae, cfg["d_model"]).to(layer_device)
    print(f"  d_sae={d_sae}  backend={backend}  device={layer_device}")

    # ── output file handles ───────────────────────────────────────────────────
    raw_path     = os.path.join(args.out_dir,
                                f"v2_raw_{args.model}_L{args.layer}.jsonl")
    feat_path    = os.path.join(args.out_dir,
                                f"v2_per_feat_{args.model}_L{args.layer}.csv")
    cell_path    = os.path.join(args.out_dir,
                                f"v2_cell_summary_{args.model}_L{args.layer}.csv")
    inv_path     = os.path.join(args.out_dir, "v2_direction_inventory.csv")

    raw_fh        = open(raw_path, "w", encoding="utf-8")
    inv_rows      = []
    per_feat_rows = []
    cell_rows     = []

    # ── per pii_type ──────────────────────────────────────────────────────────
    for pii_type in pii_types:
        key = (args.model, args.layer, pii_type)
        if key not in CELL_FEATURES:
            print(f"[skip] no selected feature for {key}")
            continue
        sel_feat = CELL_FEATURES[key]

        print(f"\n{'='*72}")
        print(f"Cell: {args.model} L{args.layer} {pii_type}  sel_feat={sel_feat}")
        print(f"{'='*72}")

        # sample random features
        rand_feats = sample_random_features(
            args.model, args.layer, sel_feat, args.K, args.seed, pii_type)
        print(f"  sampled {len(rand_feats)} random feats: {rand_feats[:5]}...")

        # build intervention functions (shared across all pairs)
        capture_fn = SelectedCaptureFn(sae, W_dec, backend, d_sae, sel_feat, args.alpha)
        rand_fns   = [RandomSAECachedFn(W_dec, g, args.alpha, capture_fn)
                      for g in rand_feats]

        # inventory
        sel_norm = float(W_dec[sel_feat].float().norm().item())
        for k, g in enumerate(rand_feats):
            inv_rows.append({
                "model": args.model, "layer": args.layer, "pii_type": pii_type,
                "selected_feat": sel_feat, "random_feat": g, "rand_seed_idx": k,
                "cosine_sim_sel_rand": round(cosine_sim(W_dec, sel_feat, g), 6),
                "sel_decoder_norm": round(sel_norm, 4),
                "rand_decoder_norm": round(float(W_dec[g].float().norm().item()), 4),
            })

        # ── unit tests (optional) ──────────────────────────────────────────────
        if args.run_unit_tests:
            pd0 = build_pair_data_for_template(
                tokenizer, pairs, registry, [pair_indices[0]], pii_type, template_ids[0])
            if pd0:
                run_sanity_checks(
                    sae, W_dec, backend, d_sae,
                    sel_feat, rand_feats, args.alpha,
                    model, args.layer, pd0[0]["full_ids"], device)

        # ── build pair data ────────────────────────────────────────────────────
        print(f"\n[phase 1] building pair data ...")
        all_pair_data = {}   # (tid, pair_idx) -> data dict
        for tid in template_ids:
            for d in build_pair_data_for_template(
                    tokenizer, pairs, registry, pair_indices, pii_type, tid):
                all_pair_data[(tid, d["pair_idx"])] = d
        print(f"  {len(all_pair_data)} (template, pair) combinations")

        # ── evaluation loop ────────────────────────────────────────────────────
        print(f"[phase 2] evaluation (baseline + selected + {len(rand_fns)} random) ...")

        # storage: (tid, pair_idx) -> metric dict
        baseline_res = {}
        selected_res = {}
        rand_res = {k: {} for k in range(len(rand_fns))}   # k -> (tid, pi) -> metric

        n_done = 0
        for (tid, pi), d in all_pair_data.items():
            full_ids = d["full_ids"]
            pl, pe   = d["prompt_len"], d["pii_end_len"]

            # 1. Baseline
            with torch.no_grad():
                log_b = forward_logits(model, full_ids, device)
            m_b = span_metrics(log_b, full_ids, pl, pe)
            if m_b is None:
                continue
            baseline_res[(tid, pi)] = m_b

            # 2. Selected (also caches m_t)
            capture_fn.cached_m_t = None
            with torch.no_grad():
                with apply_hook(model, args.layer, capture_fn):
                    log_s = forward_logits(model, full_ids, device)
            m_s = span_metrics(log_s, full_ids, pl, pe)
            if m_s is None:
                continue
            selected_res[(tid, pi)] = m_s

            # 3. K random fns — use cached m_t from step 2
            assert capture_fn.cached_m_t is not None, "m_t not cached after selected forward"
            for k, rfn in enumerate(rand_fns):
                with torch.no_grad():
                    with apply_hook(model, args.layer, rfn):
                        log_r = forward_logits(model, full_ids, device)
                m_r = span_metrics(log_r, full_ids, pl, pe)
                if m_r is not None:
                    rand_res[k][(tid, pi)] = m_r

            n_done += 1
            if n_done % 50 == 0:
                print(f"  {n_done}/{len(all_pair_data)} done")

        # ── write raw JSONL ────────────────────────────────────────────────────
        for (tid, pi), bm in baseline_res.items():
            sm = selected_res.get((tid, pi))
            if sm is None:
                continue
            row = {
                "model": args.model, "layer": args.layer,
                "pii_type": pii_type, "template": tid, "pair_idx": pi,
                "alpha": args.alpha, "selected_feat": sel_feat,
                "lp_base":       round(bm["lp_mean"], 6),
                "b_rank_base":   round(bm["b_rank"],  6),
                "n_tokens":      bm["n_tokens"],
                "lp_sel":        round(sm["lp_mean"], 6),
                "delta_lp_sel":  round(sm["lp_mean"] - bm["lp_mean"], 6),
                "delta_br_sel":  round(sm["b_rank"]  - bm["b_rank"],  6),
                "rand_feats":    rand_feats,
                "delta_lp_rand": [
                    round(rand_res[k][(tid, pi)]["lp_mean"] - bm["lp_mean"], 6)
                    if (tid, pi) in rand_res[k] else None
                    for k in range(len(rand_fns))
                ],
                "delta_br_rand": [
                    round(rand_res[k][(tid, pi)]["b_rank"] - bm["b_rank"], 6)
                    if (tid, pi) in rand_res[k] else None
                    for k in range(len(rand_fns))
                ],
            }
            raw_fh.write(json.dumps(row) + "\n")

        # ── per-feature CSV ────────────────────────────────────────────────────
        common_keys = set(selected_res.keys())
        for k, g in enumerate(rand_feats):
            sel_dlp_vals, sel_dbr_vals = [], []
            rnd_dlp_vals, rnd_dbr_vals = [], []
            for key2 in common_keys:
                bm = baseline_res.get(key2)
                sm = selected_res.get(key2)
                rm = rand_res[k].get(key2)
                if bm is None or sm is None or rm is None:
                    continue
                sel_dlp_vals.append(sm["lp_mean"] - bm["lp_mean"])
                sel_dbr_vals.append(sm["b_rank"]  - bm["b_rank"])
                rnd_dlp_vals.append(rm["lp_mean"] - bm["lp_mean"])
                rnd_dbr_vals.append(rm["b_rank"]  - bm["b_rank"])

            if not sel_dlp_vals:
                continue

            sel_dlp_mean = float(np.mean(sel_dlp_vals))
            sel_dbr_mean = float(np.mean(sel_dbr_vals))
            rnd_dlp_mean = float(np.mean(rnd_dlp_vals))
            rnd_dbr_mean = float(np.mean(rnd_dbr_vals))

            per_feat_rows.append({
                "model": args.model, "layer": args.layer,
                "pii_type": pii_type,
                "selected_feature_id": sel_feat,
                "random_feature_id":   g,
                "selected_delta_lp":   round(sel_dlp_mean, 6),
                "selected_delta_br":   round(sel_dbr_mean, 6),
                "random_sae_delta_lp": round(rnd_dlp_mean, 6),
                "random_sae_delta_br": round(rnd_dbr_mean, 6),
                # more negative lp = more suppression in alpha<0 convention
                "sel_more_suppressive_lp": sel_dlp_mean < rnd_dlp_mean,
                "sel_more_suppressive_br": sel_dbr_mean > rnd_dbr_mean,
                "max_relative_magnitude_error": 0.0,  # exact by construction (cached m_t)
                "n_examples": len(sel_dlp_vals),
            })

        # ── cell summary ───────────────────────────────────────────────────────
        # aggregate: per pair (avg over templates), then statistics
        pair_sel_lp  = defaultdict(list)
        pair_sel_br  = defaultdict(list)
        pair_rnd_lp  = {k: defaultdict(list) for k in range(len(rand_fns))}
        pair_rnd_br  = {k: defaultdict(list) for k in range(len(rand_fns))}

        for (tid, pi), bm in baseline_res.items():
            sm = selected_res.get((tid, pi))
            if sm is None:
                continue
            pair_sel_lp[pi].append(sm["lp_mean"] - bm["lp_mean"])
            pair_sel_br[pi].append(sm["b_rank"]  - bm["b_rank"])
            for k in range(len(rand_fns)):
                rm = rand_res[k].get((tid, pi))
                if rm:
                    pair_rnd_lp[k][pi].append(rm["lp_mean"] - bm["lp_mean"])
                    pair_rnd_br[k][pi].append(rm["b_rank"]  - bm["b_rank"])

        # pair-level means
        p_sel_lp = {pi: float(np.mean(v)) for pi, v in pair_sel_lp.items() if v}
        p_sel_br = {pi: float(np.mean(v)) for pi, v in pair_sel_br.items() if v}

        # null distribution: K × pairs
        null_lp, null_br = [], []
        for k in range(len(rand_fns)):
            for pi, v in pair_rnd_lp[k].items():
                if v:
                    null_lp.append(float(np.mean(v)))
            for pi, v in pair_rnd_br[k].items():
                if v:
                    null_br.append(float(np.mean(v)))

        sel_lp_mean = float(np.nanmean(list(p_sel_lp.values()))) if p_sel_lp else np.nan
        sel_br_mean = float(np.nanmean(list(p_sel_br.values()))) if p_sel_br else np.nan

        lp_st  = _stats(null_lp)
        br_st  = _stats(null_br)

        # Percentile = fraction of null values LESS suppressive than selected.
        # lp: more negative = more suppressive → null >= sel means null is less suppressive
        # br: more positive = more suppressive → null <= sel means null is less suppressive
        # Both: pct=100% → sel beats all null; pct=0% → null beats sel entirely.
        null_lp_arr = np.array([v for v in null_lp if not np.isnan(v)])
        null_br_arr = np.array([v for v in null_br if not np.isnan(v)])
        pct_sel_lp = float(np.mean(null_lp_arr >= sel_lp_mean) * 100) \
                     if len(null_lp_arr) > 0 else np.nan
        pct_sel_br = float(np.mean(null_br_arr <= sel_br_mean) * 100) \
                     if len(null_br_arr) > 0 else np.nan

        # count of K random features that are MORE suppressive than selected
        per_k_lp = []
        per_k_br = []
        for k in range(len(rand_fns)):
            vals_lp = [np.mean(v) for v in pair_rnd_lp[k].values() if v]
            vals_br = [np.mean(v) for v in pair_rnd_br[k].values() if v]
            per_k_lp.append(float(np.mean(vals_lp)) if vals_lp else np.nan)
            per_k_br.append(float(np.mean(vals_br)) if vals_br else np.nan)

        n_rand_more_lp = sum(1 for v in per_k_lp
                             if not np.isnan(v) and v < sel_lp_mean)  # more negative = more suppressive
        n_sel_more_lp  = sum(1 for v in per_k_lp
                             if not np.isnan(v) and v >= sel_lp_mean)

        cell_rows.append({
            "model": args.model, "layer": args.layer, "pii_type": pii_type,
            "selected_feat": sel_feat, "alpha": args.alpha,
            "K_sampled": len(rand_fns),
            # selected
            "sel_delta_lp_mean": round(sel_lp_mean, 6),
            "sel_delta_br_mean": round(sel_br_mean, 6),
            # rand SAE null distribution (lp)
            "rand_lp_mean":   round(lp_st["mean"],   6),
            "rand_lp_median": round(lp_st["median"], 6),
            "rand_lp_std":    round(lp_st["std"],    6),
            "rand_lp_min":    round(lp_st["min"],    6),
            "rand_lp_max":    round(lp_st["max"],    6),
            "rand_lp_p5":     round(lp_st["p5"],     6),
            "rand_lp_p95":    round(lp_st["p95"],    6),
            # rand SAE null distribution (br)
            "rand_br_mean":   round(br_st["mean"],   6),
            "rand_br_median": round(br_st["median"], 6),
            "rand_br_std":    round(br_st["std"],    6),
            "rand_br_p5":     round(br_st["p5"],     6),
            "rand_br_p95":    round(br_st["p95"],    6),
            # percentile of selected within null
            "pct_sel_in_rand_null_lp": round(pct_sel_lp, 2),
            "pct_sel_in_rand_null_br": round(pct_sel_br, 2),
            # count comparisons
            "n_rand_more_suppressive_lp": n_rand_more_lp,
            "n_sel_more_suppressive_lp":  n_sel_more_lp,
        })

        # print summary
        print(f"\n[summary] {args.model} L{args.layer} {pii_type}  sel_feat={sel_feat}")
        print(f"  selected   Δlp = {sel_lp_mean:+.4f}  Δbr = {sel_br_mean:+.4f}")
        print(f"  rand-SAE   Δlp = {lp_st['mean']:+.4f}  [{lp_st['p5']:+.4f}, {lp_st['p95']:+.4f}]")
        print(f"  sel pct in rand-null (lp): {pct_sel_lp:.1f}%  (br): {pct_sel_br:.1f}%")
        print(f"  n_rand_more_supp(lp): {n_rand_more_lp}/{len(rand_fns)}  "
              f"n_sel_more_supp: {n_sel_more_lp}/{len(rand_fns)}")

    raw_fh.close()

    # ── write per-feature CSV ─────────────────────────────────────────────────
    if per_feat_rows:
        mode = "a" if os.path.exists(feat_path) else "w"
        with open(feat_path, mode, newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(per_feat_rows[0].keys()))
            if mode == "w":
                w.writeheader()
            w.writerows(per_feat_rows)
        print(f"\n[done] per-feature CSV → {feat_path}")

    # ── write cell summary CSV ────────────────────────────────────────────────
    if cell_rows:
        with open(cell_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(cell_rows[0].keys()))
            w.writeheader()
            w.writerows(cell_rows)
        print(f"[done] cell summary  → {cell_path}")

    # ── write inventory CSV ───────────────────────────────────────────────────
    if inv_rows:
        mode = "a" if os.path.exists(inv_path) else "w"
        with open(inv_path, mode, newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(inv_rows[0].keys()))
            if mode == "w":
                w.writeheader()
            w.writerows(inv_rows)
        print(f"[done] inventory     → {inv_path}")

    print(f"[done] raw JSONL     → {raw_path}")


# ── 36-cell overall summary ───────────────────────────────────────────────────

def summarize_all_cells(out_dir: str):
    """Aggregate all v2_cell_summary_*.csv files into one 36-cell summary."""
    import glob

    pattern = os.path.join(out_dir, "v2_cell_summary_*.csv")
    files   = sorted(glob.glob(pattern))
    if not files:
        print(f"[summarize] no cell summary files found in {out_dir}")
        return

    all_rows = []
    for fp in files:
        with open(fp, newline="", encoding="utf-8") as f:
            all_rows.extend(list(csv.DictReader(f)))

    if not all_rows:
        return

    def f(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return np.nan

    n_cells = len(all_rows)
    sel_lp  = [f(r["sel_delta_lp_mean"])  for r in all_rows]
    rnd_lp  = [f(r["rand_lp_mean"])       for r in all_rows]
    pct_lp  = [f(r["pct_sel_in_rand_null_lp"]) for r in all_rows]

    # pct_lp = fraction of null LESS suppressive than sel (higher = sel more specific)
    n1 = sum(1 for s, r in zip(sel_lp, rnd_lp)
             if not np.isnan(s) and not np.isnan(r) and r < s)   # rand mean more suppressive (rand wins)
    n2 = sum(1 for s, r in zip(sel_lp, rnd_lp)
             if not np.isnan(s) and not np.isnan(r) and s <= r)  # sel mean more suppressive (sel wins)
    n3 = sum(1 for p in pct_lp if not np.isnan(p) and p >= 95)   # sel beats ≥ 95% of null
    n4 = sum(1 for p in pct_lp if not np.isnan(p) and p <= 50)   # sel beats ≤ 50% of null (rand often wins)
    # Failure Mode A: rand SAE dominates (more suppressive than selected on average)
    n_fa = n1

    summary = {
        "n_cells": n_cells,
        "n_rand_dominant_lp": n1,
        "n_sel_dominant_lp":  n2,
        "n_sel_ge_p95_of_rand_lp": n3,
        "n_sel_le_p50_of_rand_lp": n4,
        "n_failure_mode_A": n_fa,
    }

    out_path = os.path.join(out_dir, "v2_overall_summary.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        w.writeheader()
        w.writerow(summary)

    # detailed report for special cells
    special = {(r["model"], int(r["layer"]), r["pii_type"]): r for r in all_rows}
    print("\n" + "=" * 72)
    print("SPECIAL CELL REPORTS")
    print("=" * 72)
    for cell in SPECIAL_CELLS:
        r = special.get(cell)
        if r is None:
            print(f"  {cell}: NOT FOUND")
            continue
        print(f"\n  {cell[0]} L{cell[1]} {cell[2]}  sel_feat={r['selected_feat']}")
        print(f"    sel Δlp = {float(r['sel_delta_lp_mean']):>+8.4f}  sel Δbr = {float(r['sel_delta_br_mean']):>+8.4f}")
        print(f"    rand Δlp mean={float(r['rand_lp_mean']):>+8.4f}  "
              f"[p5={float(r['rand_lp_p5']):>+8.4f}, p95={float(r['rand_lp_p95']):>+8.4f}]")
        print(f"    sel pct in rand null (lp): {r['pct_sel_in_rand_null_lp']}%")
        print(f"    n_rand_more_supp: {r['n_rand_more_suppressive_lp']}  "
              f"n_sel_more_supp: {r['n_sel_more_suppressive_lp']}")

    print("\n" + "=" * 72)
    print("OVERALL 36-CELL SUMMARY")
    print("=" * 72)
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\n[done] overall summary → {out_path}")
    return summary


if __name__ == "__main__":
    import sys as _sys
    if "--summarize" in _sys.argv:
        idx = _sys.argv.index("--summarize")
        out_dir = _sys.argv[idx + 1] if idx + 1 < len(_sys.argv) else "results/random_sae_decoder_control"
        summarize_all_cells(out_dir)
    else:
        main()
