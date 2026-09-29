"""data 패키지 — pairs.jsonl 로드 유틸리티"""

import json
import os
from typing import Any

import config


def load_pairs(pairs_path: str = config.PAIRS_PATH) -> list[dict[str, Any]]:
    """pairs.jsonl 전체 로드"""
    pairs = []
    with open(pairs_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


__all__ = ["load_pairs"]
