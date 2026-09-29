#!/bin/bash
# Full-grid chain: extends task-SAE + PT-SAE cross-layer evaluation to 4 PIIs × 9 templates.
# Pre-requisite: Phase 1 (6 combos × name) already complete.
# This script runs Stage B (id/phone/email discovery) + Stage C (multi-metric × 4 PII × 9T × target+random).

set -e
PY=${PY:-python3}

LOG=results/cross_layer_discovery/full_grid_chain.log
PAIRS=$(python3 -c "print(','.join(str(i) for i in range(300, 350)))")

echo "=== START $(date '+%F %T') ===" > "$LOG"

# ============================================================================
# STAGE B — discovery for id/phone/email at 7 combos
# (Gemma Task L31 skipped — already have feat 12512 multi-PII detector)
# ============================================================================

for cfg in \
  "llama3_med_8b:8:tasksae:llama_task_L8" \
  "llama3_med_8b:16:tasksae:llama_task_L16" \
  "llama3_med_8b:8:pt:llama_pt_L8" \
  "llama3_med_8b:16:pt:llama_pt_L16" \
  "gemma_2_9b:20:tasksae:gemma_task_L20" \
  "gemma_2_9b:20:pt:gemma_pt_L20" \
  "gemma_2_9b:31:pt:gemma_pt_L31"; do
  IFS=":" read -r MODEL LAYER SAETYPE OUTSUB <<< "$cfg"
  TASKFLAG=""
  EXCLUDE=""
  if [[ "$SAETYPE" == "tasksae" ]]; then
    TASKFLAG="--use-tasksae"
    EXCLUDE="--exclude-dead"
  fi
  echo "--- [B] $MODEL L$LAYER $SAETYPE × {id,phone,email} ---" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES=0 $PY scripts/brute_force_screen.py \
    --model "$MODEL" --layer "$LAYER" --alpha=-2 --n-candidates 500 \
    $TASKFLAG $EXCLUDE \
    --pair-indices 0,1,2,3,4,5,6,7,8,9 \
    --pii-types id,phone,email \
    --out-dir results/cross_layer_discovery/${OUTSUB}_extra >> "$LOG" 2>&1
done

echo "=== STAGE B DONE $(date '+%F %T') ===" | tee -a "$LOG"

# ============================================================================
# Parse heroes from all discovery outputs → write heroes.json
# ============================================================================
$PY <<'EOF' >> "$LOG" 2>&1
import csv, json, os
from pathlib import Path

R = Path("results")
DD = R / "cross_layer_discovery"

# For each (model, layer, SAE), pick top-1 hero per PII based on the summary CSV
def pick_hero(model_tag, layer, sub_extra_or_name, pii):
    """Parse summary csv ({model}_L{layer}_alpham2_prefix_summary.csv) → top1 feature for pii."""
    # name comes from 'name'-only subdir; id/phone/email come from '_extra' subdir
    sub_name = sub_extra_or_name.rsplit("_extra", 1)[0]  # base sub
    if pii == "name":
        path = DD / sub_name / f"{model_tag}_L{layer}_alpham2_prefix_summary.csv"
    else:
        path = DD / f"{sub_name}_extra" / f"{model_tag}_L{layer}_alpham2_prefix_summary.csv"
    if not path.exists():
        return None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        if r.get("pii_type") == pii:
            feats = r.get("top_workers", "")
            if feats:
                return int(feats.split(",")[0].strip().lstrip("[").rstrip("]"))
    return None


# Hardcode hero for Gemma Task L31 (known from new_sae_phase3)
KNOWN = {
    ("gemma_2_9b", 31, "tasksae"): {"id": 12512, "phone": 12512, "email": 12512},
}

CONFIGS = [
    ("llama3_med_8b", 8,  "tasksae", "llama_task_L8"),
    ("llama3_med_8b", 16, "tasksae", "llama_task_L16"),
    ("llama3_med_8b", 8,  "pt",      "llama_pt_L8"),
    ("llama3_med_8b", 16, "pt",      "llama_pt_L16"),
    ("gemma_2_9b",    20, "tasksae", "gemma_task_L20"),
    ("gemma_2_9b",    31, "tasksae", None),  # use KNOWN
    ("gemma_2_9b",    20, "pt",      "gemma_pt_L20"),
    ("gemma_2_9b",    31, "pt",      "gemma_pt_L31"),
]

heroes = {}
for model, layer, sae, sub in CONFIGS:
    key = f"{model}_L{layer}_{sae}"
    heroes[key] = {}
    for pii in ["name", "id", "phone", "email"]:
        # check KNOWN first
        k = (model, layer, sae)
        if k in KNOWN and pii in KNOWN[k]:
            heroes[key][pii] = KNOWN[k][pii]
            continue
        if sub is None:
            heroes[key][pii] = None
            continue
        h = pick_hero(model, layer, sub, pii)
        heroes[key][pii] = h
    print(f"{key}: {heroes[key]}")

out = R / "cross_layer_discovery" / "heroes.json"
with open(out, "w") as f:
    json.dump(heroes, f, indent=2)
print(f"saved {out}")
EOF

# ============================================================================
# STAGE C — multi-metric for all 8 combos × 4 PII × 9T × {target, random}
# ============================================================================

$PY <<'EOF' >> "$LOG" 2>&1
import json, subprocess, os, sys
from pathlib import Path

R = Path("results")
heroes = json.load(open(R / "cross_layer_discovery" / "heroes.json"))

# Out dir per combo (so they don't collide with peak-layer data)
def out_dir(model, layer, sae):
    base = R / "cross_layer_xtemplate" / f"{model}_L{layer}_{sae}"
    base.mkdir(parents=True, exist_ok=True)
    return str(base)

CONFIGS = [
    ("llama3_med_8b", 8,  "tasksae", "-2"),
    ("llama3_med_8b", 16, "tasksae", "-2"),
    ("llama3_med_8b", 8,  "pt",      "-2"),
    ("llama3_med_8b", 16, "pt",      "-2"),
    ("gemma_2_9b",    20, "tasksae", "-2"),
    ("gemma_2_9b",    31, "tasksae", "-2"),
    ("gemma_2_9b",    20, "pt",      "-2"),
    ("gemma_2_9b",    31, "pt",      "-2"),
]

PAIRS = ",".join(str(i) for i in range(300, 350))
PY_BIN = os.environ.get("PY", "python3")
SCRIPT = "scripts/step5b2_multimetric.py"

for model, layer, sae, alpha in CONFIGS:
    key = f"{model}_L{layer}_{sae}"
    h = heroes.get(key, {})
    fmap_parts = []
    for pii in ["name", "id", "phone", "email"]:
        if h.get(pii) is not None:
            fmap_parts.append(f"{pii}={h[pii]}")
    if not fmap_parts:
        print(f"[skip] {key}: no heroes")
        continue
    fmap = ",".join(fmap_parts)
    out = out_dir(model, layer, sae)
    use_task = ["--use-tasksae"] if sae == "tasksae" else []
    for mode in ["target", "random_vector"]:
        cmd = [
            "env", "CUDA_VISIBLE_DEVICES=0", PY_BIN, SCRIPT,
            "--model", model, "--layer", str(layer), "--alpha=" + alpha,
            "--feature-map", fmap,
            "--templates", "T0,T1,T2,T3,T4,T5,T6,T7,T8",
            "--pair-indices", PAIRS,
            "--mode", mode,
            *use_task,
            "--out-dir", out,
        ]
        print(f"[C] {key} {mode}: feature-map={fmap}")
        sys.stdout.flush()
        ret = subprocess.run(cmd, capture_output=False)
        if ret.returncode != 0:
            print(f"[C ERROR] {key} {mode} returncode={ret.returncode}")
EOF

echo "=== STAGE C DONE $(date '+%F %T') ===" | tee -a "$LOG"
echo "=== ALL DONE $(date '+%F %T') ===" | tee -a "$LOG"
