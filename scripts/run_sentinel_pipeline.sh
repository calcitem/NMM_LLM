#!/usr/bin/env bash
# run_sentinel_pipeline.sh — run Stage 4 → Stage 5 → deploy
# Usage: bash scripts/run_sentinel_pipeline.sh [--device cuda|cpu]
set -euo pipefail

DEVICE="${1:-cuda}"
DB_PATH="/mnt/windows/NMM_DB/Malom_Standard_Ultra-strong_1.1.0/Std_DD_89adjusted"
PYTHON=".venv/bin/python"

echo "=== Sentinel pipeline: Stage 4 → Stage 5 → deploy ==="
echo "Device: $DEVICE"
echo ""

# Stage 5 only (Stage 4 already running / complete externally in this session)
# For a full fresh run from scratch, uncomment the Stage 4 block below.

# ── Stage 4 (fresh, --drop-db-features) ────────────────────────────────────
# echo "--- Stage 4: structural training (no DB features) ---"
# $PYTHON scripts/train_sentinel.py \
#   --config configs/sentinel_stage4.yaml \
#   --game-dir data/games \
#   --db-path "$DB_PATH" \
#   --drop-db-features \
#   --aux-wdl --lambda-wdl 0.3 \
#   --device "$DEVICE"
# echo "Stage 4 complete."

# ── Stage 5 (fine-tune, DB features re-enabled) ─────────────────────────────
echo "--- Stage 5: DB-feature fine-tuning ---"
$PYTHON scripts/train_sentinel.py \
  --config configs/sentinel_stage5.yaml \
  --game-dir data/games \
  --db-path "$DB_PATH" \
  --resume learned_ai/sentinel/checkpoints/stage4/best.pt \
  --epochs 35 \
  --aux-wdl --lambda-wdl 0.3 \
  --device "$DEVICE"
# Note: --epochs 35 = Stage 4's 27 best epoch + 8 Stage 5 fine-tune epochs.
# The resume sets start_epoch=27; the loop runs range(27,35) = 8 epochs.
echo "Stage 5 complete."

# ── Deploy ──────────────────────────────────────────────────────────────────
echo "--- Deploying Stage 5 checkpoint ---"
BACKUP="learned_ai/sentinel/checkpoints/best-$(date +%Y%m%d-%H%M%S)-pre-deploy.pt"
if [ -f learned_ai/sentinel/checkpoints/best.pt ]; then
  cp learned_ai/sentinel/checkpoints/best.pt "$BACKUP"
  echo "Previous best.pt backed up to: $BACKUP"
fi
cp learned_ai/sentinel/checkpoints/stage5/best.pt \
   learned_ai/sentinel/checkpoints/best.pt
echo "Deployed: stage5/best.pt → checkpoints/best.pt"
echo ""
echo "=== Pipeline complete. Restart the Flask server to load the new sentinel. ==="
