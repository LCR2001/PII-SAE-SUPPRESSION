#!/usr/bin/env bash
# Full 36-cell Random SAE Decoder Direction Control experiment
# Runs 9 (model, layer) jobs sequentially; each handles 4 PII types × 9 templates × 50 pairs

set -e
PY=${PY:-python3}
OUT=results/random_sae_decoder_control
PAIRS=$(python3 -c "print(','.join(str(i) for i in range(300,350)))")
TEMPLATES="T0,T1,T2,T3,T4,T5,T6,T7,T8"
K=20
ALPHA=-2
SEED=42

export CUDA_VISIBLE_DEVICES=0

echo "======================================================================"
echo "Random SAE Decoder Control — full 36-cell run"
echo "K=$K  alpha=$ALPHA  seed=$SEED  templates=$TEMPLATES  pairs=300-349"
echo "======================================================================"

run_cell() {
    local MODEL=$1; local LAYER=$2
    echo ""
    echo ">>> [$(date '+%H:%M:%S')] $MODEL  L$LAYER"
    $PY scripts/exp_rand_sae_decoder_v2.py \
        --model "$MODEL" --layer "$LAYER" \
        --pii-types name,id,phone,email \
        --K "$K" --alpha "$ALPHA" --seed "$SEED" \
        --pair-indices "$PAIRS" --templates "$TEMPLATES" \
        --out-dir "$OUT"
    echo "<<< [$(date '+%H:%M:%S')] $MODEL L$LAYER done"
}

# ── Gemma-2-9B ─────────────────────────────────────────────────────────────────
run_cell gemma_2_9b 9
run_cell gemma_2_9b 20
run_cell gemma_2_9b 31

# ── Llama-3.1-8B-Instruct ──────────────────────────────────────────────────────
run_cell llama3_1_8b_instruct 8
run_cell llama3_1_8b_instruct 16
run_cell llama3_1_8b_instruct 24

# ── Qwen3-8B ───────────────────────────────────────────────────────────────────
run_cell qwen3_8b 9
run_cell qwen3_8b 18
run_cell qwen3_8b 27

echo ""
echo "======================================================================"
echo "All 9 model+layer jobs done. Generating 36-cell summary ..."
echo "======================================================================"
$PY scripts/exp_rand_sae_decoder_v2.py --summarize "$OUT"

echo ""
echo "[done] Results in $OUT/"
