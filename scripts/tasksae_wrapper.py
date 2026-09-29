"""
scripts/tasksae_wrapper.py

Wrapper for the new task-specific TopK SAE checkpoints in
<SAE_CHECKPOINT_DIR>/{name}/sae.pt.

Checkpoint format (PyTorch state dict):
  pre_bias        : [d_model]
  encoder.weight  : [d_sae, d_model]
  encoder.bias    : [d_sae]
  decoder.weight  : [d_model, d_sae]
  steps_since_active : [d_sae]   (training tracker; for dead-feature diagnostics)

TopK encode:
  pre = encoder(x − pre_bias) + bias
  values, indices = topk(pre, k)
  z is dense [d_sae] with z[indices] = values, else 0
TopK decode:
  x' = decoder z + pre_bias

This wrapper exposes a sparsify-Sae-compatible interface with `encode` returning
EncoderOutput-like (top_acts, top_indices) so we can reuse the pipeline.
"""

import json
import os
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class EncoderOutput:
    top_acts: Tensor
    top_indices: Tensor


class TaskSAE(torch.nn.Module):
    def __init__(self, ckpt_dir: str, device: str = "cpu"):
        super().__init__()
        cfg_path = os.path.join(ckpt_dir, "config.json")
        sd_path = os.path.join(ckpt_dir, "sae.pt")
        if not os.path.exists(sd_path):
            raise FileNotFoundError(f"sae.pt not found in {ckpt_dir}")
        with open(cfg_path) as f:
            self.cfg = json.load(f)
        self.d_model = self.cfg["d_model"]
        self.d_sae = self.cfg["d_model"] * self.cfg["expansion"]
        self.k = self.cfg["k"]
        self.layer = self.cfg["layer"]
        self.dead_threshold = self.cfg.get("dead_threshold", 1000)
        self.num_latents = self.d_sae

        sd = torch.load(sd_path, map_location=device, weights_only=False)
        self.pre_bias = torch.nn.Parameter(sd["pre_bias"].to(device))
        self.encoder_weight = torch.nn.Parameter(sd["encoder.weight"].to(device))
        self.encoder_bias = torch.nn.Parameter(sd["encoder.bias"].to(device))
        self.decoder_weight = torch.nn.Parameter(sd["decoder.weight"].to(device))
        self.register_buffer("steps_since_active", sd["steps_since_active"].to(device))
        self.W_dec = self.decoder_weight  # alias for compatibility
        self.b_dec = self.pre_bias  # alias for decode_from_dense (sparsify branch)

    @property
    def device(self):
        return self.pre_bias.device

    @property
    def dtype(self):
        return self.pre_bias.dtype

    def parameters(self, recurse=True):
        return iter([self.pre_bias, self.encoder_weight, self.encoder_bias, self.decoder_weight])

    @torch.no_grad()
    def encode(self, x: Tensor) -> EncoderOutput:
        """x: [..., d_model] -> EncoderOutput(top_acts, top_indices) with shape [..., k]."""
        x = x.to(self.pre_bias.dtype)
        x_centered = x - self.pre_bias
        pre = x_centered @ self.encoder_weight.T + self.encoder_bias  # [..., d_sae]
        # ReLU then top-k (standard TopK SAE)
        pre = torch.relu(pre)
        top_vals, top_idx = pre.topk(self.k, dim=-1)
        return EncoderOutput(top_acts=top_vals, top_indices=top_idx)

    @torch.no_grad()
    def decode(self, top_acts: Tensor, top_indices: Tensor) -> Tensor:
        """top_acts/top_indices: [..., k]. Returns reconstructed x: [..., d_model]."""
        # Sum: x' = sum_i top_acts[i] * decoder_weight[:, top_indices[i]] + pre_bias
        # Use scatter into dense then matmul (works for any shape).
        shape = top_acts.shape[:-1]  # batch dims
        flat_acts = top_acts.reshape(-1, self.k)
        flat_idx = top_indices.reshape(-1, self.k)
        # gather the relevant decoder columns: [N, k, d_model]
        cols = self.decoder_weight[:, flat_idx]  # [d_model, N, k]
        cols = cols.permute(1, 2, 0)            # [N, k, d_model]
        recon = (flat_acts.unsqueeze(-1) * cols).sum(dim=1)  # [N, d_model]
        recon = recon + self.pre_bias.unsqueeze(0)
        return recon.reshape(*shape, self.d_model)

    @torch.no_grad()
    def encode_dense(self, x: Tensor) -> Tensor:
        """Convenience: encode + scatter to dense [..., d_sae]."""
        out = self.encode(x)
        shape = x.shape[:-1]
        flat_acts = out.top_acts.reshape(-1, self.k)
        flat_idx = out.top_indices.reshape(-1, self.k)
        N = flat_acts.shape[0]
        dense = torch.zeros(N, self.d_sae, device=x.device, dtype=x.dtype)
        dense.scatter_(1, flat_idx, flat_acts.to(x.dtype))
        return dense.reshape(*shape, self.d_sae)


class TaskSAEWrapper:
    """Wrapper compatible with models.sae_wrapper.SAEWrapper (sparsify backend)."""
    def __init__(self, sae: TaskSAE):
        self.sae = sae
        self.backend = "tasksae"  # custom tag so existing branches still work via sparsify-style API

    @torch.no_grad()
    def encode(self, hidden: Tensor) -> Tensor:
        """For single-position vector, returns dense [d_sae]."""
        return self.sae.encode_dense(hidden)

    @torch.no_grad()
    def decode(self, features: Tensor) -> Tensor:
        # If features is dense [..., d_sae], take its top-k for compatibility
        if features.shape[-1] == self.sae.d_sae:
            top_vals, top_idx = features.topk(self.sae.k, dim=-1)
            return self.sae.decode(top_vals, top_idx)
        raise ValueError("decode expects dense [d_sae] input")

    def to(self, device):
        self.sae = self.sae.to(device)
        return self


def load_tasksae(ckpt_dir: str, device: str = "cuda:0") -> TaskSAEWrapper:
    sae = TaskSAE(ckpt_dir, device=device)
    return TaskSAEWrapper(sae)


def dead_fraction(ckpt_dir: str) -> dict:
    """Return dead feature stats from the checkpoint's tracker + metrics.json."""
    sd_path = os.path.join(ckpt_dir, "sae.pt")
    sd = torch.load(sd_path, map_location="cpu", weights_only=False)
    cfg = json.load(open(os.path.join(ckpt_dir, "config.json")))
    threshold = cfg.get("dead_threshold", 1000)
    steps = sd["steps_since_active"]
    d_sae = steps.shape[0]
    dead = (steps > threshold).sum().item()
    return {"d_sae": d_sae, "n_dead": dead, "dead_fraction": dead / d_sae}
