# PII Leakage Suppression via Task-Specific SAE Intervention

Code and results for the paper:
> **"Suppressing PII Leakage in Fine-Tuned LLMs via Task-Specific Sparse Autoencoder Intervention"**

We identify PII-encoding features in task-specific Sparse Autoencoders (SAEs) trained on fine-tuned medical LLMs, then suppress PII prediction by directly steering the corresponding residual stream directions at inference time — without modifying model weights.

**Models evaluated**: Gemma-2-9B, Llama-3.1-8B-Instruct, Qwen3-8B  
**PII types**: patient name, ID, phone, email  
**Key result**: Top-1 SAE feature suppression reduces PII log-probability by 3–6 nats while preserving downstream utility (MMLU / MedQA / PubMedQA accuracy within ±2%).

---

## Repository Structure

```
pii-sae-suppression/
├── config.py                        # Global model paths & hyperparameters (set env vars below)
├── data/
│   ├── generate_registry.py         # Generate synthetic patient profiles (Synthea-style)
│   ├── build_pairs.py               # Build (p_L, p_N) prompt pairs
│   ├── generate_test_set.py         # Hold-out test split
│   ├── generate_sae_training.py     # 10k synthetic records for SAE training
│   ├── pairs_test.jsonl             # Pre-built test pairs (n=100; synthetic PII)
│   └── profile_registry_test.jsonl  # Test patient profiles
├── scripts/
│   ├── finetune_lora.py             # [Step 1] LoRA fine-tuning for PII memorization
│   ├── tasksae_wrapper.py           # TopK SAE model class
│   ├── tasksae_glue.py              # SAE checkpoint loader
│   ├── tasksae_phase1.py            # [Step 3] SAE diagnostics (dead features, recon error)
│   ├── brute_force_screen.py        # [Step 4] Feature discovery via brute-force sign sweep
│   ├── intervention.py              # Residual stream hook utilities
│   ├── step5b2_eval.py              # [Step 5] Main PII suppression evaluation (9 templates × 4 PII)
│   ├── step5b2_multimetric.py       # Multi-metric aggregation
│   ├── exp_rand_sae_decoder_v2.py   # [Step 6] Random SAE decoder direction control
│   ├── exp_utility_mmlu.py          # [Step 7] MMLU utility benchmark
│   ├── exp_utility_medqa.py         # [Step 7] MedQA utility benchmark
│   ├── exp_utility_pubmedqa.py      # [Step 7] PubMedQA utility benchmark
│   ├── step5b2_figures.py           # Figure generation (Figs 1–6)
│   ├── fig_tasksae_selectivity_grid.py
│   ├── fig_tasksae_categorical.py
│   ├── fig_utility.py
│   ├── figures_final.py
│   ├── viz_fig1_main_v3.py
│   ├── viz_figA1_alpha_sensitivity_v3.py
│   ├── viz_unified_fig23.py
│   └── viz_scatter_36cells.py
├── run_full_grid_chain.sh           # Orchestrates Steps 4–5 across all 9 (model, layer) cells
├── run_rand_sae_control.sh          # Runs the 36-cell random SAE decoder control experiment
└── results/
    ├── figures_final/               # All paper figures (PDF + PNG)
    ├── random_sae_decoder_control/  # Random SAE direction control results
    ├── utility_mmlu/                # MMLU benchmark summaries
    ├── utility_medqa/               # MedQA benchmark summaries
    └── utility_pubmedqa/            # PubMedQA benchmark summaries
```

---

## Environment Setup

### Python environment

```bash
conda create -n pii-sae python=3.10
conda activate pii-sae
pip install torch==2.3.0 transformers==4.46.3 peft==0.12.0 \
            datasets accelerate safetensors tqdm numpy pandas
```

### Path configuration

Set the following environment variables (or modify `config.py` directly):

```bash
# Base LLM checkpoints (HuggingFace model directories)
export GEMMA_MODEL_PATH="/path/to/gemma-2-9b"
export LLAMA_INSTRUCT_MODEL_PATH="/path/to/llama-3.1-8b-instruct"
export QWEN3_MODEL_PATH="/path/to/qwen3-8b"

# LoRA fine-tuned adapter checkpoints (output of Step 1)
export CHECKPOINT_DIR="/path/to/checkpoints"
# Expected layout: $CHECKPOINT_DIR/{gemma_2_9b,llama3_1_8b_instruct,qwen3_8b}_lora/final/

# Task-specific SAE checkpoints (output of Step 2; see SAE Training below)
export SAE_CHECKPOINT_DIR="/path/to/SAE/checkpoints"
# Expected layout: $SAE_CHECKPOINT_DIR/{model_tag}_L{layer}_x8_k32_v2/
```

---

## Execution Order

### Step 1 — Fine-tune models with LoRA

Fine-tune each base LLM on 600 synthetic medical notes to memorize patient PII (training/test split verified by extraction rate).

```bash
for MODEL in gemma_2_9b llama3_1_8b_instruct qwen3_8b; do
    python scripts/finetune_lora.py \
        --model $MODEL \
        --pairs-path data/pairs.jsonl \
        --output-base $CHECKPOINT_DIR \
        --epochs 20 --lr 3e-4
done
```

Stopping criterion: training extraction rate ≥ 90%, test extraction rate < 20% (memorization without generalization).

### Step 2 — Train task-specific SAEs

Train TopK SAEs (expansion×8, k=32) on residual stream activations of the fine-tuned models. SAE training code uses the configuration format in `results/random_sae_decoder_control/v2_direction_inventory.csv`. Key hyperparameters: `steps=50000`, `batch_size=4096`, `lr=3e-4`, `aux_alpha=0.25`, `dead_threshold=1000`.

Store checkpoints at `$SAE_CHECKPOINT_DIR/{model_tag}_L{layer}_x8_k32_v2/`.

### Step 3 — SAE diagnostics

```bash
for MODEL in gemma_2_9b llama3_1_8b_instruct qwen3_8b; do
    python scripts/tasksae_phase1.py \
        --model $MODEL --layers 9 20 31 \   # adjust layers per model
        --n-pairs 20 \
        --out results/new_sae_diagnostics/${MODEL}_phase1.csv
done
```

### Step 4 — Feature discovery (brute-force sign sweep)

For each (model, layer) cell, screen all alive SAE features for those that maximally suppress PII log-probability.

```bash
bash run_full_grid_chain.sh   # runs all 9 (model, layer) combinations
```

Or per-cell:
```bash
python scripts/brute_force_screen.py \
    --model gemma_2_9b --layer 20 \
    --pii-type name --pair-indices 0-299 \
    --templates T0 --alpha -2 \
    --out-dir results/new_x8_v2_xtemplate
```

### Step 5 — Main PII suppression evaluation

Evaluate hero features across 9 prompt templates × 4 PII types × 50 held-out pairs (indices 300–349):

```bash
python scripts/step5b2_eval.py \
    --model gemma_2_9b --layer 20 \
    --pii-types name,id,phone,email \
    --templates T0,T1,T2,T3,T4,T5,T6,T7,T8 \
    --pair-indices $(python3 -c "print(','.join(str(i) for i in range(300,350)))") \
    --alpha -2 \
    --out-dir results/new_paper_step5b2_multimetric
```

### Step 6 — Random SAE decoder direction control

Ablates whether the selected direction is special within the SAE manifold by comparing it against K=20 random SAE decoder directions with matched magnitude.

```bash
bash run_rand_sae_control.sh
```

Results: `results/random_sae_decoder_control/v2_overall_summary.csv`  
Key finding: selected feature ranks at ≥ p72 of the random null in 26/36 cells (Δ log-prob metric).

### Step 7 — Utility benchmarks

```bash
for BENCH in mmlu medqa pubmedqa; do
    python scripts/exp_utility_${BENCH}.py \
        --model gemma_2_9b --layer 20 --feat 20385 --alpha -2
done
```

### Step 8 — Reproduce paper figures

```bash
python scripts/step5b2_figures.py      # Figs 1–3, A1
python scripts/fig_tasksae_selectivity_grid.py  # Fig 5
python scripts/fig_tasksae_categorical.py       # Fig 5 (categorical)
python scripts/fig_utility.py                   # Fig 4, 6
```

All output PDFs/PNGs are already in `results/figures_final/`.

---

## Key Results Summary

### Random SAE Decoder Direction Control (Appendix, 36-cell)

| Metric | Count / 36 cells |
|--------|-----------------|
| Selected > rand-SAE mean (Δ log-prob) | **26 / 36** |
| Selected ≥ p95 of rand null | **12 / 36** |
| Selected ≤ p50 of rand null | 10 / 36 |
| Rand-SAE dominant | 10 / 36 |

Full per-cell table: `results/random_sae_decoder_control/v2_cell_summary_*.csv`

### Utility Preservation

Intervention at hero feature with α=−2 preserves downstream task accuracy within ±2% on MMLU, MedQA, and PubMedQA across all three models.

---

## Reproducibility

This repository contains the full experiment code, and the pipeline can be reproduced by following the steps in [Execution Order](#execution-order). Three artifacts are **not included** and must be prepared separately:

| Artifact | How to obtain |
|----------|---------------|
| Base model weights | Download from Hugging Face: Gemma-2-9B, Llama-3.1-8B-Instruct, Qwen3-8B |
| LoRA adapter checkpoints | Train with `scripts/finetune_lora.py` (Step 1) |
| Task-specific SAE checkpoints | Train TopK SAEs on the fine-tuned models (Step 2) |

The LoRA adapters and SAE weights are excluded because they range from hundreds of MB to several GB. We plan to release them separately on the Hugging Face Hub.

### Inspecting results without running anything

All reported results are already included:

- `results/figures_final/` — all paper figures (PDF + PNG)
- `results/random_sae_decoder_control/` — per-cell CSVs for the random SAE decoder control
- `results/utility_*/` — MMLU / MedQA / PubMedQA summaries

---

## Data

`data/pairs_test.jsonl` and `data/profile_registry_test.jsonl` contain **synthetically generated** patient profiles. Names, IDs, phone numbers, and emails are algorithmically generated and do not correspond to real individuals. Raw Synthea output used as clinical note templates is not included due to size; run `data/generate_registry.py` to regenerate.

---

## Citation

```bibtex
@article{xxx2025pii,
  title   = {Suppressing PII Leakage in Fine-Tuned LLMs via Task-Specific Sparse Autoencoder Intervention},
  author  = {xxx},
  year    = {2025},
}
```
