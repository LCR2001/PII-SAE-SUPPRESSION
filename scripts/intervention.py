"""
scripts/intervention.py
Day 4 — SAE decoder-injection intervention.

방식:
  hook(activation):  # [batch, seq, d_model]
    z = sae.encode(activation)   # sparse feature activations
    direction = sum_{f in target} sae.W_dec[f] * z[..., f]
    return activation - alpha * direction

5 modes:
  baseline         : hook 없음
  target           : 지정 features 차단
  random_vector    : 같은 magnitude, 임의 direction (privacy 떨어지면 안 됨)
  random_features  : 임의 features (utility 깨지지만 같은 PII 안 막음)

KV cache OFF 강제 (intervention 효과 매 step 적용 일관성).
All positions에 적용 (마지막 token only 아님).

CLI:
    python scripts/intervention.py --model gemma_2_9b --layer 35 \\
        --features 14816 --alphas 0,1.0,2.0 \\
        --modes baseline,target,random_vector,random_features

출력: 각 (mode, alpha) 조합별 생성된 30 token + PII match 여부
"""

import argparse
import gc
import os
import random
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import config
from data import load_pairs
import models.loader as loader

PROMPT_MARKER = "Patient Name: "  # prompt까지의 prefix 끝
GEN_TOKENS = 30


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(config.MODEL_CONFIGS.keys()))
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--features", required=True, help="comma-separated SAE feature indices")
    p.add_argument("--alphas", default="1.0", help="comma-separated alpha values")
    p.add_argument("--modes", default="baseline,target,random_vector,random_features",
                   help="comma-separated modes")
    p.add_argument("--pair-idx", type=int, default=0, help="pairs.jsonl에서 사용할 pair index")
    p.add_argument("--gen-tokens", type=int, default=GEN_TOKENS)
    p.add_argument("--max-prompt-len", type=int, default=400, help="prompt token max")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None)
    p.add_argument("--adapter-path", default=None,
                   help="LoRA adapter (fine-tuned 모델 분석용, base에 merge 후 hook 적용)")
    return p.parse_args()


# ── prompt 만들기 ────────────────────────────────────────────────────────────

def make_pii_prompt_and_target(pair: dict):
    """p_l을 'Patient Name: ' 직전까지 자르고, 그 뒤에 오는 실제 PII를 정답으로."""
    p_l = pair["p_l"]
    idx = p_l.find(PROMPT_MARKER)
    if idx < 0:
        return None, None
    cut = idx + len(PROMPT_MARKER)
    prompt = p_l[:cut]
    target_continuation = p_l[cut:]
    return prompt, target_continuation


# ── SAE decoder weight 추출 ──────────────────────────────────────────────────

def get_sae_W_dec(sae, expected_d_sae, expected_d_model):
    """[d_sae, d_model] 반환."""
    for attr in ["W_dec", "decoder"]:
        if not hasattr(sae, attr):
            continue
        obj = getattr(sae, attr)
        if isinstance(obj, torch.Tensor):
            W = obj.detach()
        elif isinstance(obj, torch.nn.Module) and hasattr(obj, "weight"):
            W = obj.weight.detach().T.contiguous()
        else:
            continue
        if W.shape == (expected_d_sae, expected_d_model):
            return W
        if W.shape == (expected_d_model, expected_d_sae):
            return W.T.contiguous()
    raise RuntimeError("Could not locate SAE W_dec")


def get_sae_W_enc(sae, expected_d_sae, expected_d_model):
    """[d_sae, d_model] 반환 (W_dec와 같은 shape).
    SAELens: W_enc tensor [d_model, d_sae]
    Sparsify: encoder.weight Linear [d_sae, d_model]"""
    for attr in ["W_enc"]:
        if hasattr(sae, attr):
            obj = getattr(sae, attr)
            if isinstance(obj, torch.Tensor):
                W = obj.detach()
                if W.shape == (expected_d_sae, expected_d_model):
                    return W
                if W.shape == (expected_d_model, expected_d_sae):
                    return W.T.contiguous()
    for attr in ["encoder"]:
        if hasattr(sae, attr):
            obj = getattr(sae, attr)
            if isinstance(obj, torch.nn.Module) and hasattr(obj, "weight"):
                W = obj.weight.detach()
                if W.shape == (expected_d_sae, expected_d_model):
                    return W
                if W.shape == (expected_d_model, expected_d_sae):
                    return W.T.contiguous()
    raise RuntimeError("Could not locate SAE W_enc")


# ── encode utility (backend-agnostic) ────────────────────────────────────────

def encode_to_dense(sae, hidden, backend, d_sae):
    """hidden [B, T, D] → z [B, T, d_sae] dense (sparsify는 scatter)."""
    B, T, D = hidden.shape
    flat = hidden.view(B * T, D)
    if backend == "saelens":
        z = sae.encode(flat)  # [B*T, d_sae]
    else:
        out = sae.encode(flat)
        # out.top_acts: [B*T, k], top_indices: [B*T, k]
        z = torch.zeros(B * T, d_sae, device=hidden.device, dtype=out.top_acts.dtype)
        z.scatter_(1, out.top_indices.long(), out.top_acts)
    return z.view(B, T, d_sae)


def decode_from_dense(sae, z, backend, W_dec=None):
    """z [B, T, d_sae] → reconstructed h [B, T, d_model].
    saelens: sae.decode() 사용. sparsify: manual W_dec multiplication + b_dec."""
    B, T, d_sae = z.shape
    flat = z.view(B * T, d_sae)
    if backend == "saelens":
        h = sae.decode(flat)
    else:
        # sparsify: manual W_dec 곱셈
        if W_dec is None:
            raise ValueError("sparsify backend requires W_dec arg for decode")
        h = flat @ W_dec  # [B*T, d_model]
        b_dec = getattr(sae, "b_dec", None)
        if b_dec is not None:
            h = h + b_dec
    return h.view(B, T, -1)


# ── hook makers ─────────────────────────────────────────────────────────────

def build_direction(z, target_indices, W_dec):
    """z [B,T,d_sae], target_indices [n], W_dec [d_sae, d_model] → direction [B,T,d_model]"""
    z_target = z[..., target_indices]            # [B, T, n]
    W_target = W_dec[target_indices]             # [n, d_model]
    return z_target @ W_target                   # [B, T, d_model]


def make_intervention_fn(sae, W_dec, backend, d_sae, target_features, alpha, mode,
                         W_enc=None):
    """공통 intervention 로직. Activation [B,T,D] → modified [B,T,D].
    W_enc 제공 시 mode='target_encoder' 사용 가능 (W_dec 대신 W_enc로 direction 만듦)."""
    if mode == "baseline":
        return None  # no hook

    target_t = torch.tensor(list(target_features), dtype=torch.long)
    n_targets = len(target_features)

    def fn(activation):
        device = activation.device
        target_idx = target_t.to(device)
        sae_dtype = next(sae.parameters()).dtype
        h = activation.to(sae_dtype)
        z = encode_to_dense(sae, h, backend, d_sae)  # [B, T, d_sae]

        # ── z-space modes: SAE decoder path 거쳐 hidden 교체 (reconstruction error 포함)
        if mode in ("target_z_space", "random_features_z_space"):
            z_modified = z.clone()
            if mode == "target_z_space":
                idx = target_idx
            else:
                idx = torch.randperm(d_sae, device=device)[:n_targets]
            # z_new = (1 - α) · z[target]
            # α=-2 → z_new=3·z (factor=3) → SAE.decode(z_new) ≈ h + 2·direction
            # α=+2 → z_new=-1·z → SAE.decode(z_new) ≈ h - 2·direction
            # 즉 hidden-space mode와 sign 일관 (perfect reconstruction이면 동일)
            z_modified[..., idx] = (1.0 - alpha) * z_modified[..., idx]
            h_new = decode_from_dense(sae, z_modified, backend, W_dec=W_dec)
            return h_new.to(activation.dtype)

        # ── hidden-space modes (기존)
        if mode == "target":
            direction = build_direction(z, target_idx, W_dec)
        elif mode == "target_encoder":
            if W_enc is None:
                raise ValueError("W_enc required for target_encoder mode")
            direction = build_direction(z, target_idx, W_enc)
        elif mode == "random_encoder":
            if W_enc is None:
                raise ValueError("W_enc required for random_encoder mode")
            rand_idx = torch.randperm(d_sae, device=device)[:n_targets]
            direction = build_direction(z, rand_idx, W_enc)
        elif mode == "random_vector":
            # Span-coherent random control: one isotropic random direction per
            # forward pass, broadcast to every intervened token position. Only
            # the magnitude varies token-by-token (matched to hero ‖z_f·W_dec[f]‖).
            ref = build_direction(z, target_idx, W_dec)        # [B, T, D]
            ref_norm = ref.norm(dim=-1, keepdim=True)           # [B, T, 1]
            B, T, D = ref.shape
            rand = torch.randn(B, 1, D, device=ref.device, dtype=ref.dtype)
            rand = rand / rand.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            direction = rand * ref_norm                          # [B, T, D] same dir across tokens
        elif mode == "random_features":
            rand_idx = torch.randperm(d_sae, device=device)[:n_targets]
            direction = build_direction(z, rand_idx, W_dec)
        else:
            raise ValueError(f"unknown mode: {mode}")

        return (activation - alpha * direction.to(activation.dtype))
    return fn


def make_random_sae_decoder_fn(sae, W_dec, backend, d_sae, target_features, alpha,
                                rand_feature_id: int):
    """Random SAE decoder direction control.

    Matches token-wise perturbation magnitude of the selected SAE intervention
    but substitutes a randomly sampled SAE decoder direction:

        m_t      = ||z_t[f] * W_dec[f]||_2          (per-token magnitude)
        u_g      = W_dec[g] / ||W_dec[g]||_2         (unit vector, fixed per run)
        h'_t     = h_t - alpha * m_t * u_g

    Magnitude equality guarantee:
        ||h'_t - h_t|| = |alpha| * m_t  (identical to selected SAE intervention)

    rand_feature_id must be alive (steps_since_active <= threshold) and
    must differ from the selected feature(s) in target_features.
    """
    target_t = torch.tensor(list(target_features), dtype=torch.long)

    with torch.no_grad():
        u_g_raw = W_dec[rand_feature_id].float()
        u_g_norm_val = float(u_g_raw.norm().item())
        u_g = (u_g_raw / max(u_g_norm_val, 1e-8)).to(W_dec.dtype)

    def fn(activation):
        device = activation.device
        target_idx = target_t.to(device)
        u = u_g.to(device=device, dtype=activation.dtype)

        sae_dtype = next(sae.parameters()).dtype
        h = activation.to(sae_dtype)
        z = encode_to_dense(sae, h, backend, d_sae)

        ref = build_direction(z, target_idx, W_dec.to(device))   # [B, T, d_model]
        ref_norm = ref.norm(dim=-1, keepdim=True)                  # [B, T, 1]

        direction = ref_norm * u.unsqueeze(0).unsqueeze(0)        # [B, T, d_model]
        return (activation - alpha * direction.to(activation.dtype))

    fn.rand_feature_id = rand_feature_id
    fn.decoder_norm = u_g_norm_val
    return fn


def gemma_hook(intervention_fn):
    """TL hook signature: (activation, hook=hook_obj) -> activation."""
    def h(activation, hook=None):
        if intervention_fn is None:
            return activation
        return intervention_fn(activation)
    return h


def llama_hook(intervention_fn):
    """HF hook signature: (module, input, output) -> output."""
    def h(module, inp, out):
        if intervention_fn is None:
            return out
        if isinstance(out, tuple):
            new_h = intervention_fn(out[0])
            return (new_h,) + out[1:]
        return intervention_fn(out)
    return h


# ── generation ──────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_gemma(model, prompt, gen_tokens, max_prompt_len):
    tokens = model.to_tokens(prompt, prepend_bos=True)
    if tokens.shape[1] > max_prompt_len:
        tokens = torch.cat([tokens[:, :1], tokens[:, -(max_prompt_len - 1):]], dim=1)
    out = model.generate(tokens, max_new_tokens=gen_tokens, do_sample=False,
                         verbose=False, use_past_kv_cache=False)
    new_tokens = out[0, tokens.shape[1]:]
    return model.tokenizer.decode(new_tokens.tolist(), skip_special_tokens=True)


@torch.no_grad()
def generate_llama(model, tokenizer, prompt, gen_tokens, max_prompt_len, device):
    enc = tokenizer(prompt, return_tensors="pt", truncation=True,
                    max_length=max_prompt_len, add_special_tokens=True)
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    out = model.generate(input_ids=input_ids, attention_mask=attn,
                         max_new_tokens=gen_tokens, do_sample=False,
                         pad_token_id=int(pad_id), use_cache=False)
    new_tokens = out[0, input_ids.shape[1]:]
    return tokenizer.decode(new_tokens.tolist(), skip_special_tokens=True)


# ── 메인 ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = config.MODEL_CONFIGS[args.model]
    target_features = [int(x) for x in args.features.split(",")]
    alphas = [float(x) for x in args.alphas.split(",")]
    modes = args.modes.split(",")

    # pair + prompt
    pairs = load_pairs(config.PAIRS_PATH)
    pair = pairs[args.pair_idx]
    prompt, target_cont = make_pii_prompt_and_target(pair)
    if prompt is None:
        sys.exit("could not find PROMPT_MARKER in p_l")
    print(f"=== Prompt (last 200 chars): ...{prompt[-200:]}")
    print(f"=== Target continuation (first 200): {target_cont[:200]}")
    print(f"=== Target features: {target_features}, alphas: {alphas}, modes: {modes}")

    # load model
    adapter_label = f" + LoRA adapter: {args.adapter_path}" if args.adapter_path else ""
    print(f"\n[load] {args.model}{adapter_label} ...")
    if cfg["model_type"] == "gemma":
        model = loader.load_model(args.model, config.MODEL_DEVICE,
                                  adapter_path=args.adapter_path)
        tokenizer = model.tokenizer
        first_device = model.cfg.device
    else:
        model, tokenizer = loader.load_model(args.model, config.MODEL_DEVICE,
                                             adapter_path=args.adapter_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        first_device = next(model.parameters()).device
    model.eval()

    # load SAE
    print(f"[load] SAE layer {args.layer} ...")
    sae_wrapper = loader.load_sae(args.model, args.layer, str(first_device))
    sae = sae_wrapper.sae
    backend = sae_wrapper.backend
    d_sae = cfg["sae_dim"]
    W_dec = get_sae_W_dec(sae, d_sae, cfg["d_model"]).to(first_device)
    print(f"[load] W_dec shape={tuple(W_dec.shape)} dtype={W_dec.dtype}")

    # iterate (mode, alpha)
    print("\n" + "=" * 72)
    print("INTERVENTION SANITY CHECK")
    print("=" * 72)

    results = []  # rows for csv
    for mode in modes:
        # baseline은 alpha=0 한 번만
        alpha_list = [0.0] if mode == "baseline" else alphas
        for alpha in alpha_list:
            label = f"{mode}_a{alpha}"
            print(f"\n--- {label} ---")
            # rebuild hook
            intervention_fn = make_intervention_fn(
                sae, W_dec, backend, d_sae, target_features, alpha, mode
            )

            if cfg["model_type"] == "gemma":
                if intervention_fn is not None:
                    model.add_hook(f"blocks.{args.layer}.hook_resid_post",
                                   gemma_hook(intervention_fn))
                try:
                    gen = generate_gemma(model, prompt, args.gen_tokens, args.max_prompt_len)
                finally:
                    model.reset_hooks()
            else:
                handle = None
                if intervention_fn is not None:
                    handle = model.model.layers[args.layer].register_forward_hook(
                        llama_hook(intervention_fn))
                try:
                    gen = generate_llama(model, tokenizer, prompt, args.gen_tokens,
                                         args.max_prompt_len, first_device)
                finally:
                    if handle is not None:
                        handle.remove()

            # PII match check (간단 substring)
            target_first_line = target_cont.split("\n")[0].strip()
            pii_match = target_first_line in gen if target_first_line else False
            print(f"  generated: {gen[:200]!r}")
            print(f"  target_first_line: {target_first_line!r}")
            print(f"  PII match: {pii_match}")
            results.append({
                "model": args.model, "layer": args.layer,
                "features": ",".join(map(str, target_features)),
                "mode": mode, "alpha": alpha,
                "generated": gen[:300],
                "target_first_line": target_first_line,
                "pii_match": pii_match,
            })

    # csv 출력
    if args.out:
        import csv as csvmod
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csvmod.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)
        print(f"\n[intervention] wrote {args.out}")

    # 요약
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"  {'mode':<20s} {'alpha':>6s} {'pii_match':>10s}")
    for r in results:
        print(f"  {r['mode']:<20s} {r['alpha']:>6.2f} {str(r['pii_match']):>10s}")
    print()
    print("기대:")
    print("  baseline: PII match=True (모델이 PII 학습/암기)")
    print("  target a=1.0+: PII match=False (intervention 작동)")
    print("  random_vector: PII match=True (random direction은 효과 없어야)")
    print("  random_features: PII match일 수도 (의미 없는 feature 차단)")


if __name__ == "__main__":
    main()
