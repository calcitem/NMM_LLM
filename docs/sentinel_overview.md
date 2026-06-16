# Sentinel AI — Overview & Training Guide

The Sentinel is a learned overlay on top of the heuristic GameAI engine. It watches each position, scores candidate moves by quality, and can redirect the engine toward better choices without replacing it.

---

## Architecture

**SentinelNet** is a move-level quality scorer.  Each inference example is one candidate move in one position; the network outputs a single float in `[0, 1]` representing move quality from the mover's perspective (1.0 = winning move, 0.5 = draw, 0.0 = losing move).

```
Input: 58-float feature vector (see Feature Vector section below)

Shared trunk:
  Linear(58 → 128) → ReLU → Dropout
  Linear(128 → 64) → ReLU → Dropout
  Linear(64  → 32) → ReLU → Dropout

Quality head (always active):
  Linear(32 → 1) → Sigmoid  →  move_quality ∈ [0, 1]

Auxiliary WDL head (optional, --aux-wdl):
  Linear(32 → 3)  →  logits [loss, draw, win]  (cross-entropy during training only)
```

At inference `SentinelAdvisor.advise()` returns `move_quality` for each candidate and flags the top recommendation if it differs significantly from the engine's first choice.

---

## Intervention modes

| Mode | Behaviour |
|---|---|
| `advisory` | Logs advice; never changes the move. Badge shown in UI. |
| `score_adjust` | Nudges heuristic scores proportional to sentinel quality delta before final sort. |
| `reconsider` | On high-confidence bad moves: tries LLM override → deeper search → second-best fallback. |

All sentinel calls are wrapped in `try/except`; failures always fall through to the heuristic move.

---

## Feature vector (58 floats)

| Range | Size | Content |
|---|---|---|
| `[0:20)` | 20 | Board context (piece counts, phase, mills, mobility — mover-normalised) |
| `[20:40)` | 20 | Move features (from/to square, closes mill, captures, fly-phase flag, …) |
| `[40:58)` | 18 | Counterfactual context (heuristic rank/score vs candidates; DB-derived stats at training time) |

**DB-derived slots `[41:46)` and `[48:58)`** are populated from `ExternalSolvedDB.query_all_moves()` during training (win/loss fractions, WDL indicators, DTM quality scores).  At inference these slots are always **zero** — the DB is never queried at runtime.

> **Important:** training with DB features enabled causes the model to learn  
> `output ≈ feat[57]` (this-move DTM quality = training label).  
> This gives near-zero Spearman r at inference (all DB slots are 0 at test time).  
> **Always use `--drop-db-features` in training** to zero those slots and force the  
> model to learn from board structure and move geometry instead.

---

## Training stages

Training is a four-stage curriculum.  Each stage saves its best checkpoint in `learned_ai/sentinel/checkpoints/stageN/`.  Stage N+1 resumes from stage N's `best.pt`.

### Stage 1 — Structural foundation

Learn purely from board structure.  No DB, no DB feature slots.  Heuristic quality scores are the training labels.

```
Config:  configs/sentinel_stage1.yaml
Command: .venv/bin/python scripts/train_sentinel.py \
           --config configs/sentinel_stage1.yaml \
           --game-dir data/games \
           --drop-db-features \
           --decisive-only \
           --device cuda
```

Key settings: `external_db_enabled: false`, `dropout: 0.3`, `lr: 0.001`, `epochs: 20`

What the model learns: which board patterns are structurally strong — piece counts, mill pressure, mobility, piece placement — using only normalised heuristic scores as labels.

---

### Stage 2 — DB calibration

Resume from Stage 1.  Malom DB provides strong WDL + DTM labels for every legal move.  `--drop-db-features` still zeroes the DB indicator slots, so the model updates its *structural* weights toward DB ground truth rather than memorising the oracle signal directly.

```
Config:  configs/sentinel_stage2.yaml
Command: .venv/bin/python scripts/train_sentinel.py \
           --config configs/sentinel_stage2.yaml \
           --game-dir data/games \
           --db-path /mnt/windows/NMM_DB/Malom_Standard_Ultra-strong_1.1.0/Std_DD_89adjusted \
           --resume learned_ai/sentinel/checkpoints/stage1/best.pt \
           --drop-db-features \
           --aux-wdl --lambda-wdl 0.3 \
           --device cuda
```

Key settings: `external_db_enabled: true`, `dropout: 0.2`, `lr: 0.0003`, `epochs: 30`

What the model learns: to recognise winning and losing positions from board features alone, calibrated against the solved database.  DTM-graded labels (`dtm_quality()`) give fine-grained quality in `[0,1]` rather than binary WDL.

---

### Stage 3 — Trajectory fine-tuning (historical — had feature leakage bug)

> **Archived.** Stage 3 (`configs/sentinel_stage3.yaml`) was designed to fine-tune on  
> game-outcome trajectories but was run without `--drop-db-features`.  This caused  
> `feat[57]` (this-move DTM quality) to equal the training label for 86% of examples.  
> The model learned to copy `feat[57] → output`; at inference (`feat[57] = 0`) Spearman  
> r was ~0.10 (near-random).  The checkpoint at `checkpoints/stage3/best.pt` (epoch 2,  
> val-loss 0.307) should not be used for production.

Stage 4 below is the corrected replacement.

---

### Stage 4 — Corrected full training

Train from scratch with `--drop-db-features` active throughout.  DTM-graded labels provide accurate supervision; the model cannot shortcut on oracle features that are absent at inference.

```
Config:  configs/sentinel_stage4.yaml
Command: .venv/bin/python scripts/train_sentinel.py \
           --config configs/sentinel_stage4.yaml \
           --game-dir data/games \
           --db-path /mnt/windows/NMM_DB/Malom_Standard_Ultra-strong_1.1.0/Std_DD_89adjusted \
           --drop-db-features \
           --aux-wdl --lambda-wdl 0.3 \
           --device cuda
```

Key settings: `dropout: 0.2`, `lr: 0.001`, `epochs: 30`, fresh start (no `--resume`)

Optionally add human games for broader coverage:
```
           --human-game-dir data/human_games \
```

Optionally boost the played move by game outcome:
```
           --trajectory-weight \
```

---

### Stage 5 — DB feature fine-tuning (light)

Resume from Stage 4.  DB feature slots are now **visible** (no `--drop-db-features`).  At a very low learning rate the model learns to exploit WDL, DTM, and win-fraction signals from the solved DB when they are available, while retaining the structural weights built in Stage 4.  Few epochs prevent overwriting Stage 4 learning.

```
Config:  configs/sentinel_stage5.yaml
Command: .venv/bin/python scripts/train_sentinel.py \
           --config configs/sentinel_stage5.yaml \
           --game-dir data/games \
           --db-path /mnt/windows/NMM_DB/Malom_Standard_Ultra-strong_1.1.0/Std_DD_89adjusted \
           --resume learned_ai/sentinel/checkpoints/stage4/best.pt \
           --epochs 35 \
           --aux-wdl --lambda-wdl 0.3 \
           --device cuda
```

Key settings: `dropout: 0.1`, `lr: 0.00005`, `epochs: 8` fine-tune, resumes from Stage 4

> **Epoch arithmetic:** the `--resume` flag loads Stage 4's epoch counter (e.g. 27).
> Pass `--epochs N` where N = Stage 4 epoch + 8 (e.g. `--epochs 35`).
> The training loop runs `range(27, 35)` = exactly 8 fine-tuning epochs.
> Omitting `--epochs` causes the config's `epochs: 8` to produce an empty range and no training.

What the model learns: to use DB quality signals (`feat[41-57]`) when they are populated — improving predictions for positions where the game's endgame DB or external DB provides information — without losing the board-structure knowledge from Stage 4.

> **Curriculum rationale:** Stage 4 forces the model to build genuine structural  
> knowledge (DB features zeroed).  Stage 5 then teaches it to additionally exploit  
> DB signals when present.  The result is a model that degrades gracefully:  
> strong at inference without DB info, stronger still when DB info is available.

---

## Deploying the checkpoint

After all stages complete, back up the previous production checkpoint and promote the new one:

```bash
# Back up what's live (do this first, before running any new training)
cp learned_ai/sentinel/checkpoints/best.pt \
   learned_ai/sentinel/checkpoints/best-YYYYMMDD-backup.pt

# Promote Stage 5 output to production
cp learned_ai/sentinel/checkpoints/stage5/best.pt \
   learned_ai/sentinel/checkpoints/best.pt
```

The checkpoint path used at runtime is configured in `web/app.py` and defaults to `learned_ai/sentinel/checkpoints/best.pt`.

Restart the Flask server to pick up the new checkpoint (it is loaded once at startup).

---

## Evaluation

```bash
.venv/bin/python scripts/eval_sentinel.py \
  --checkpoint learned_ai/sentinel/checkpoints/stage4/best.pt \
  --game-dir data/games \
  --db-path /mnt/windows/NMM_DB/Malom_Standard_Ultra-strong_1.1.0/Std_DD_89adjusted
```

Key metric: **Spearman r** between predicted quality and Malom DTM quality across the held-out val set.  A well-trained model (Stage 4) should reach r > 0.4.  Stage 3 (leaky) was r ≈ 0.10.

---

## Graceful degradation

If `best.pt` is missing or PyTorch is not installed, the sentinel silently disables itself at startup.  The game runs identically.  No crash, no error shown to the user.
