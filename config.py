"""
전역 설정 — 모델 경로, SAE 정보, 디바이스, 공통 하이퍼파라미터
"""

import os

# ── 모델 구성 ──────────────────────────────────────────────────────────────────
MODEL_CONFIGS = {
    "gemma_2_9b": {
        "path": os.environ.get("GEMMA_MODEL_PATH", "/path/to/gemma-2-9b"),
        "tl_name": "gemma-2-9b",        # TransformerLens 내부 모델명
        "model_type": "gemma",          # "gemma" → TransformerLens 사용
        "num_layers": 42,
        "d_model": 3584,
        "sae_release": "gemma-scope-9b-pt-res-canonical",
        "sae_id_template": "layer_{layer}/width_16k/canonical",
        "sae_dim": 16384,
    },
    "llama3_med_8b": {
        "path": os.environ.get("LLAMA_MED_MODEL_PATH", "/path/to/llama3-med42-8b"),
        "model_type": "llama",          # "llama" → HuggingFace + sparsify 사용
        "num_layers": 31,               # SAE는 layer 0~30만 존재 (layers.31 없음)
        "d_model": 4096,
        "sae_hub": "EleutherAI/sae-llama-3-8b-32x",
        "sae_dim": 131072,
    },
    "llama3_1_8b_instruct": {
        "path": os.environ.get("LLAMA_INSTRUCT_MODEL_PATH", "/path/to/llama-3.1-8b-instruct"),
        "model_type": "llama",          # HuggingFace 경로
        "num_layers": 32,
        "d_model": 4096,
        # 아직 사전학습 SAE 정해두지 않음. Task-SAE만 사용 예정.
        "sae_hub": None,
        "sae_dim": None,
    },
    "qwen3_8b": {
        "path": os.environ.get("QWEN3_MODEL_PATH", "/path/to/qwen3-8b"),
        "model_type": "qwen",          # HuggingFace generic
        "num_layers": 36,
        "d_model": 4096,
        "sae_hub": None,
        "sae_dim": None,
    },
}

# ── 디바이스 ───────────────────────────────────────────────────────────────────
import torch as _torch

# 모델: 사용 가능한 GPU 전체에 분산 (n_devices)
# 입력 텐서 / run_with_cache 시작점은 항상 cuda:0
MODEL_DEVICE = "cuda:0"

# 사용 가능한 GPU 수 (모델 분산에 사용)
N_MODEL_GPUS: int = _torch.cuda.device_count() if _torch.cuda.is_available() else 1

# SAE는 cuda:0 에 로드 (activation은 CPU 경유 후 이쪽으로 이동)
SAE_DEVICE = "cuda:0"

DTYPE = "bfloat16"

# ── 데이터 경로 ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
REGISTRY_PATH = os.path.join(DATA_DIR, "profile_registry.jsonl")
PAIRS_PATH = os.path.join(DATA_DIR, "pairs.jsonl")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# ── P_N variant 이름 ───────────────────────────────────────────────────────────
VARIANTS = ["private", "redacted", "typed"]

# PII 유형별 단독 ablation variant
ABLATED_VARIANTS = ["name_ablated", "id_ablated", "phone_ablated", "email_ablated"]
ABLATED_LABELS = {
    "name_ablated":  "Patient Name",
    "id_ablated":    "Patient ID",
    "phone_ablated": "Phone",
    "email_ablated": "Email",
}
ABLATED_COLORS = {
    "name_ablated":  "#e41a1c",   # red
    "id_ablated":    "#377eb8",   # blue
    "phone_ablated": "#4daf4a",   # green
    "email_ablated": "#ff7f00",   # orange
}

# ── 추출 하이퍼파라미터 ────────────────────────────────────────────────────────
BATCH_SIZE = 8
MAX_LENGTH = 512      # 앞쪽 truncation (긴 prefix_text 대응)
TOP_K_FEATURES = 50   # SAE feature analysis용

# ── 디버그 모드 ────────────────────────────────────────────────────────────────
DEBUG_N_PAIRS = 10
DEBUG_LAYERS_GEMMA = [0, 5, 10]
DEBUG_LAYERS_LLAMA = [0, 5, 10]


def get_result_dir(model_name: str) -> str:
    d = os.path.join(RESULTS_DIR, model_name)
    os.makedirs(d, exist_ok=True)
    return d


def get_feature_dir(model_name: str) -> str:
    d = os.path.join(get_result_dir(model_name), "features")
    os.makedirs(d, exist_ok=True)
    return d


def get_feature_path(model_name: str, variant: str) -> str:
    return os.path.join(get_feature_dir(model_name), f"features_{variant}.npz")


def get_ablated_feature_path(model_name: str) -> str:
    return os.path.join(get_feature_dir(model_name), "features_ablated.npz")


def make_all_dirs(model_name: str) -> None:
    for sub in ["features", "pca_plots", "cosine_similarity",
                "sae_feature_analysis", "delta_distribution"]:
        os.makedirs(os.path.join(RESULTS_DIR, model_name, sub), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "cross_model_comparison"), exist_ok=True)
