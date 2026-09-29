"""
scripts/tasksae_glue.py

Helper to load a task-specific SAE in a form compatible with the existing
intervention pipeline (sparsify-style backend).

Returns a wrapper-shaped object so existing scripts can do:
    sae_wrapper = load_tasksae_for_intervention(model, layer, device)
    sae = sae_wrapper.sae        # has .encode (top-k), W_dec, b_dec, k, num_latents
    backend = sae_wrapper.backend  # "sparsify" so existing branches apply

Provides the same interface as models.loader.load_sae(...) returns.
"""

import os
from scripts.tasksae_wrapper import TaskSAE


SAE_BASE = os.environ.get("SAE_CHECKPOINT_DIR", "/path/to/SAE/checkpoints")


class TaskSAEInterventionWrapper:
    """sparsify-compatible wrapper around TaskSAE."""
    def __init__(self, sae: TaskSAE):
        self.sae = sae
        # Tag as 'sparsify' so existing intervention.encode_to_dense scatters top-k
        self.backend = "sparsify"

    def to(self, device):
        self.sae = self.sae.to(device)
        return self


SHORT_MAP = {
    "gemma_2_9b": "gemma_2_9b",
    "llama3_med_8b": "llama3_med_8b",
    "llama3_1_8b_instruct": "llama_31_8b_inst",
    "qwen3_8b": "qwen3_8b",
}


def ckpt_dir_for(model: str, layer: int, expansion: int = 4, suffix: str = "") -> str:
    short = SHORT_MAP[model]
    base = f"{short}_L{layer}_x{expansion}_k32"
    if suffix:
        base = f"{base}_{suffix}"
    return f"{SAE_BASE}/{base}"


def load_tasksae_for_intervention(model: str, layer: int, device: str = "cuda:0",
                                   expansion: int = 4,
                                   suffix: str = "") -> TaskSAEInterventionWrapper:
    ckpt = ckpt_dir_for(model, layer, expansion, suffix)
    if not os.path.exists(os.path.join(ckpt, "sae.pt")):
        raise FileNotFoundError(f"task-SAE not found: {ckpt}")
    sae = TaskSAE(ckpt, device=device)
    return TaskSAEInterventionWrapper(sae)


def get_alive_features(model: str, layer: int,
                        expansion: int = 4, suffix: str = "") -> list:
    """Return indices of non-dead features (steps_since_active <= dead_threshold)."""
    import torch, json
    ckpt = ckpt_dir_for(model, layer, expansion, suffix)
    sd = torch.load(os.path.join(ckpt, "sae.pt"), map_location="cpu", weights_only=False)
    cfg = json.load(open(os.path.join(ckpt, "config.json")))
    threshold = cfg.get("dead_threshold", 1000)
    steps = sd["steps_since_active"]
    alive_mask = (steps <= threshold)
    return alive_mask.nonzero(as_tuple=True)[0].tolist()
