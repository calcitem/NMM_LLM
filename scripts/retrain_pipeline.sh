#!/usr/bin/env bash
# retrain_pipeline.sh — Full sentinel retraining: Stage 1 → 2 → 4 → 5 → deploy
# Includes human_games for broader coverage; FEN dedup active in dataset.py.
# Usage: bash scripts/retrain_pipeline.sh [cuda|cpu]
set -euo pipefail

DEVICE="${1:-cuda}"
DB_PATH="/mnt/windows/NMM_DB/Malom_Standard_Ultra-strong_1.1.0/Std_DD_89adjusted"
PYTHON=".venv/bin/python"
HUMAN_DIR="data/human_games"

echo "=== Sentinel retrain pipeline ==="
echo "Device: $DEVICE  |  Human games: $HUMAN_DIR"
echo ""

# ── Stage 1: structural foundation (no DB, decisive-only) ───────────────────
echo "--- Stage 1: structural training (heuristic labels, no DB) ---"
$PYTHON scripts/train_sentinel.py \
  --config configs/sentinel_stage1.yaml \
  --game-dir data/games \
  --human-game-dir "$HUMAN_DIR" \
  --drop-db-features \
  --decisive-only \
  --device "$DEVICE"
echo "Stage 1 complete."
echo ""

# ── Stage 2: DB calibration (resume Stage 1, drop DB feature slots) ─────────
echo "--- Stage 2: DB calibration (resume Stage 1) ---"
$PYTHON scripts/train_sentinel.py \
  --config configs/sentinel_stage2.yaml \
  --game-dir data/games \
  --human-game-dir "$HUMAN_DIR" \
  --db-path "$DB_PATH" \
  --resume learned_ai/sentinel/checkpoints/stage1/best.pt \
  --drop-db-features \
  --aux-wdl --lambda-wdl 0.3 \
  --device "$DEVICE"
echo "Stage 2 complete."
echo ""

# ── Stage 4: corrected full training (fresh, drop DB feature slots) ──────────
echo "--- Stage 4: full structural training (fresh start, no DB features) ---"
$PYTHON scripts/train_sentinel.py \
  --config configs/sentinel_stage4.yaml \
  --game-dir data/games \
  --human-game-dir "$HUMAN_DIR" \
  --db-path "$DB_PATH" \
  --drop-db-features \
  --aux-wdl --lambda-wdl 0.3 \
  --device "$DEVICE"
echo "Stage 4 complete."
echo ""

# ── Stage 5: DB feature fine-tuning (resume Stage 4) ────────────────────────
# Epoch arithmetic: read Stage 4's saved epoch counter and add 8.
STAGE4_EPOCH=$($PYTHON -c "
import torch
ck = torch.load('learned_ai/sentinel/checkpoints/stage4/best.pt', map_location='cpu')
print(ck.get('epoch', 29))
")
STAGE5_EPOCHS=$((STAGE4_EPOCH + 8))
echo "--- Stage 5: DB feature fine-tuning (resume Stage 4 @ epoch $STAGE4_EPOCH → --epochs $STAGE5_EPOCHS) ---"
$PYTHON scripts/train_sentinel.py \
  --config configs/sentinel_stage5.yaml \
  --game-dir data/games \
  --human-game-dir "$HUMAN_DIR" \
  --db-path "$DB_PATH" \
  --resume learned_ai/sentinel/checkpoints/stage4/best.pt \
  --epochs "$STAGE5_EPOCHS" \
  --aux-wdl --lambda-wdl 0.3 \
  --device "$DEVICE"
echo "Stage 5 complete."
echo ""

# ── Deploy Stage 5 as production checkpoint ──────────────────────────────────
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
echo "=== Sentinel retrain pipeline complete. Restart Flask to load new checkpoint. ==="
