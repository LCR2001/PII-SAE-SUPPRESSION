"""
data/build_pairs.py
profile_registry.jsonl을 읽어 P_L / P_N_1 / P_N_2 / P_N_3 pair를 생성.
출력: data/pairs.jsonl

실행:
    python data/build_pairs.py
    python data/build_pairs.py --registry data/profile_registry.jsonl --out data/pairs.jsonl
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ── P_L 템플릿 ─────────────────────────────────────────────────────────────────
_PII_BLOCK = """\
For patient handover, the following personal information has been compiled:
Patient Name: {name}
Patient ID: {patient_id}
Phone: {phone}
Email: {email}"""

def make_pl(rec: dict) -> str:
    block = _PII_BLOCK.format(
        name=rec["full_name"],
        patient_id=rec["patient_id"],
        phone=rec["phone"],
        email=rec["email"],
    )
    return rec["prefix_text"] + "\n\n" + block


def make_pn_private(rec: dict) -> str:
    block = _PII_BLOCK.format(
        name="{PRIVATE}",
        patient_id="{PRIVATE}",
        phone="{PRIVATE}",
        email="{PRIVATE}",
    )
    return rec["prefix_text"] + "\n\n" + block


def make_pn_redacted(rec: dict) -> str:
    block = _PII_BLOCK.format(
        name="{REDACTED}",
        patient_id="{REDACTED}",
        phone="{REDACTED}",
        email="{REDACTED}",
    )
    return rec["prefix_text"] + "\n\n" + block


def make_pn_typed(rec: dict) -> str:
    block = """\
For patient handover, the following personal information has been compiled:
Patient Name: {NAME}
Patient ID: {PATIENT_ID}
Phone: {PHONE}
Email: {EMAIL}"""
    return rec["prefix_text"] + "\n\n" + block


# ── PII 유형별 단독 ablation ────────────────────────────────────────────────────
# 해당 필드만 placeholder로 교체하고 나머지는 실제 PII를 유지.
# delta = encode(P_L) - encode(P_N_ablated) 는 해당 PII 유형 제거 효과만 포착.

def make_pn_name_ablated(rec: dict) -> str:
    """이름만 {NAME}으로 교체, 나머지 PII는 실제 값 유지."""
    block = _PII_BLOCK.format(
        name="{NAME}",
        patient_id=rec["patient_id"],
        phone=rec["phone"],
        email=rec["email"],
    )
    return rec["prefix_text"] + "\n\n" + block


def make_pn_id_ablated(rec: dict) -> str:
    """환자ID만 {PATIENT_ID}으로 교체."""
    block = _PII_BLOCK.format(
        name=rec["full_name"],
        patient_id="{PATIENT_ID}",
        phone=rec["phone"],
        email=rec["email"],
    )
    return rec["prefix_text"] + "\n\n" + block


def make_pn_phone_ablated(rec: dict) -> str:
    """전화번호만 {PHONE}으로 교체."""
    block = _PII_BLOCK.format(
        name=rec["full_name"],
        patient_id=rec["patient_id"],
        phone="{PHONE}",
        email=rec["email"],
    )
    return rec["prefix_text"] + "\n\n" + block


def make_pn_email_ablated(rec: dict) -> str:
    """이메일만 {EMAIL}으로 교체."""
    block = _PII_BLOCK.format(
        name=rec["full_name"],
        patient_id=rec["patient_id"],
        phone=rec["phone"],
        email="{EMAIL}",
    )
    return rec["prefix_text"] + "\n\n" + block


def build(registry_path: str, out_path: str) -> None:
    records = []
    with open(registry_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    pairs = []
    for rec in records:
        pair = {
            "entity_id": rec["entity_id"],
            "p_l": make_pl(rec),
            "p_n_private": make_pn_private(rec),
            "p_n_redacted": make_pn_redacted(rec),
            "p_n_typed": make_pn_typed(rec),
            # PII 유형별 단독 ablation
            "p_n_name_ablated":  make_pn_name_ablated(rec),
            "p_n_id_ablated":    make_pn_id_ablated(rec),
            "p_n_phone_ablated": make_pn_phone_ablated(rec),
            "p_n_email_ablated": make_pn_email_ablated(rec),
        }
        pairs.append(pair)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"[build_pairs] {len(pairs)} pairs → {out_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default=config.REGISTRY_PATH)
    parser.add_argument("--out", default=config.PAIRS_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.registry):
        print(f"[build_pairs] Registry not found: {args.registry}")
        print("[build_pairs] Run 'python data/generate_registry.py' first.")
        sys.exit(1)
    build(args.registry, args.out)
