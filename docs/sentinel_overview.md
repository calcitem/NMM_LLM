# Sentinel AI — Overview & Training Guide

The Sentinel is a learned overlay on top of the heuristic GameAI engine. It watches each position, scores candidate moves by quality, and can redirect the engine toward better choices without replacing it.

---

## Architecture

**SentinelNet** is a move-level quality scorer. Each inference example is one candidate move in one position; the network outputs a single float in `[0, 1]` representing move quality from the mover's perspective (1.0 = winning move, 0.5 = draw, 0.0 = losing move).

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
| `score_adjust` | Re-ranks candidates using a blend of heuristic rank and sentinel quality. The engine's search result is anchored at rank 0; other candidates are blended 60% heuristic / 40% sentinel. |
| `reconsider` | On high-confidence bad moves: tries LLM override → deeper search → second-best fallback. |

All sentinel calls are wrapped in `try/except`; failures always fall through to the heuristic move.

---

## Feature vector (58 floats)

| Range | Size | Content |
|---|---|---|
| `[0:20)` | 20 | Board context (piece counts, phase, mills, mobility — mover-normalised) |
| `[20:40)` | 20 | Move features (from/to square, closes mill, captures, fly-phase flag, …) |
| `[40:58)` | 18 | Counterfactual context (heuristic rank/score vs candidates; DB-derived stats at training time) |

**DB-derived slots `[41:46)` and `[48:58)`** are populated from `ExternalSolvedDB.query_all_moves()` during training (win/loss fractions, WDL indicators, DTM quality scores). At inference these slots are always **zero** — the DB is never queried at runtime.

> **Important:** training with DB features enabled causes the model to learn  
> `output ≈ feat[57]` (this-move DTM quality = training label).  
> This gives near-zero Spearman r at inference (all DB slots are 0 at test time).  
> **Always use `--drop-db-features` in training** to zero those slots and force the  
> model to learn from board structure and move geometry instead.

---

## Training dataset

For every played position in every game file, **all legal moves** are enumerated (not just the played move). Each legal move gets one `MoveExample` with its own feature vector and quality label. This trains the model to rank moves within a position rather than merely predict the played move.

**FEN deduplication:** Each unique board position (identified by `board_fen_before`) is included at most once per split. Positions seen in earlier game files are skipped when encountered again. This prevents common opening positions — which appear in hundreds of games with identical early moves — from flooding the gradient signal. Train and val each deduplicate independently.

The dataset is split at the **game-file level** (no ply-level leakage between train and val).

**Human games:** Pass `--human-game-dir data/human_games` to include human-vs-human JSONL records alongside AI self-play games.

---

## Training stages

Training is a four-stage curriculum. Each stage saves its best checkpoint in `learned_ai/sentinel/checkpoints/stageN/`. Stage N+1 resumes from stage N's `best.pt` where applicable.

### Stage 1 — Structural foundation

Learn purely from board structure. No DB, no DB feature slots. Heuristic quality scores are the training labels.

```
Config:  configs/sentinel_stage1.yaml
Command: .venv/bin/python scripts/train_sentinel.py \
           --config configs/sentinel_stage1.yaml \
           --game-dir data/games \
           --human-game-dir data/human_games \
           --drop-db-features \
           --decisive-only \
           --device cuda
```

Key settings: `external_db_enabled: false`, `dropout: 0.3`, `lr: 0.001`, `epochs: 20`

What the model learns: which board patterns are structurally strong — piece counts, mill pressure, mobility, piece placement — using only normalised heuristic scores as labels.

---

### Stage 2 — DB calibration

Resume from Stage 1. Malom DB provides strong WDL + DTM labels for every legal move. `--drop-db-features` still zeroes the DB indicator slots, so the model updates its *structural* weights toward DB ground truth rather than memorising the oracle signal directly.

```
Config:  configs/sentinel_stage2.yaml
Command: .venv/bin/python scripts/train_sentinel.py \
           --config configs/sentinel_stage2.yaml \
           --game-dir data/games \
           --human-game-dir data/human_games \
           --db-path /mnt/windows/NMM_DB/Malom_Standard_Ultra-strong_1.1.0/Std_DD_89adjusted \
           --resume learned_ai/sentinel/checkpoints/stage1/best.pt \
           --drop-db-features \
           --aux-wdl --lambda-wdl 0.3 \
           --device cuda
```

Key settings: `external_db_enabled: true`, `dropout: 0.2`, `lr: 0.0003`, `epochs: 30`

---

### Stage 3 — Archived (feature leakage)

> **Archived.** Stage 3 was designed to fine-tune on game-outcome trajectories but was run without `--drop-db-features`. This caused `feat[57]` (this-move DTM quality) to equal the training label for 86% of examples. The model learned to copy `feat[57] → output`; at inference (`feat[57] = 0`) Spearman r was ~0.10 (near-random). The checkpoint at `checkpoints/stage3/best.pt` should not be used for production.

Stage 4 below is the corrected replacement.

---

### Stage 4 — Corrected full training

Train from scratch with `--drop-db-features` active throughout. DTM-graded labels provide accurate supervision; the model cannot shortcut on oracle features that are absent at inference.

```
Config:  configs/sentinel_stage4.yaml
Command: .venv/bin/python scripts/train_sentinel.py \
           --config configs/sentinel_stage4.yaml \
           --game-dir data/games \
           --human-game-dir data/human_games \
           --db-path /mnt/windows/NMM_DB/Malom_Standard_Ultra-strong_1.1.0/Std_DD_89adjusted \
           --drop-db-features \
           --aux-wdl --lambda-wdl 0.3 \
           --device cuda
```

Key settings: `dropout: 0.2`, `lr: 0.001`, `epochs: 30`, fresh start (no `--resume`)

---

### Stage 5 — DB feature fine-tuning (light)

Resume from Stage 4. DB feature slots are now **visible** (no `--drop-db-features`). At a very low learning rate the model learns to exploit WDL, DTM, and win-fraction signals from the solved DB when they are available, while retaining the structural weights built in Stage 4. Few epochs prevent overwriting Stage 4 learning.

```
Config:  configs/sentinel_stage5.yaml
Command: .venv/bin/python scripts/train_sentinel.py \
           --config configs/sentinel_stage5.yaml \
           --game-dir data/games \
           --human-game-dir data/human_games \
           --db-path /mnt/windows/NMM_DB/Malom_Standard_Ultra-strong_1.1.0/Std_DD_89adjusted \
           --resume learned_ai/sentinel/checkpoints/stage4/best.pt \
           --epochs 38 \
           --aux-wdl --lambda-wdl 0.3 \
           --device cuda
```

Key settings: `dropout: 0.1`, `lr: 0.00005`, `epochs: 8` fine-tune, resumes from Stage 4

> **Epoch arithmetic:** The `--resume` flag loads Stage 4's epoch counter.  
> Pass `--epochs N` where N = Stage 4 best epoch + 8.  
> With `epochs: 30` in the Stage 4 config, use `--epochs 38`.  
> Omitting `--epochs` causes the config's `epochs: 8` to produce an empty range and no training.

---

## One-command pipeline (recommended)

Runs all four stages in sequence, dynamically computes Stage 5's epoch count, and promotes the final checkpoint to `best.pt`:

```bash
bash scripts/retrain_pipeline.sh cuda
# or: bash scripts/retrain_pipeline.sh cpu
```

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

### Offline quality metrics (`eval_sentinel.py`)

Evaluates the deployed checkpoint against Malom DB ground truth. DB feature slots are **zeroed** to simulate live inference conditions.

```bash
.venv/bin/python scripts/eval_sentinel.py \
  --checkpoint learned_ai/sentinel/checkpoints/best.pt \
  --game-dir data/games \
  --db-path /mnt/windows/NMM_DB/Malom_Standard_Ultra-strong_1.1.0/Std_DD_89adjusted \
  --limit 200
```

Key metrics:

| Metric | Meaning | Target |
|---|---|---|
| `win_acc` | % of DB-win moves scored > 0.5 | > 60% |
| `loss_acc` | % of DB-loss moves scored < 0.5 | > 65% |
| `top1_win_rate` | % of positions (win available) where sentinel ranks a win #1 | > 75% |
| `critical_miss` | % of positions (win available) where sentinel ranks a loss #1 | < 15% |
| `spearman_r` | Move ranking correlation with DTM quality | limited by features |
| `bad_move_recall` | Loss-in-≤10 moves scored < 0.4 | > 50% |

> **Note on Spearman r:** With DB features zeroed at inference, r ≈ 0.10 is expected.
> The top-1 win rate (76.5% on Stage 4+5) is the more actionable metric for game play.

### Known results

| Checkpoint | win_acc | loss_acc | top1_win_rate | critical_miss | spearman_r |
|---|---|---|---|---|---|
| Stage 3 (leaky — do not use) | ~85% | ~65% | ~76% | ~20% | ~0.10 |
| Stage 4+5 | 41.6% | 64.9% | 76.5% | 20.0% | 0.10 |

Stage 4+5 win_acc is lower because the model doesn't read feat[57] at inference; top-1 win rate is equivalent, meaning game-play quality is unchanged or better.

---

### Game-play benchmark (`bench_sentinel.py`)

Tests whether sentinel/value-net actually improve the engine's win rate in self-play.

```bash
# Sanity check — identical configs should be near 50/50
.venv/bin/python scripts/bench_sentinel.py --games 200 --difficulty 4

# Sentinel (score_adjust) vs baseline
.venv/bin/python scripts/bench_sentinel.py --games 200 --difficulty 4 \
  --white-sentinel score_adjust

# Sentinel + value_net vs baseline
.venv/bin/python scripts/bench_sentinel.py --games 200 --difficulty 4 \
  --white-sentinel score_adjust --white-value-net

# Value net alone vs baseline
.venv/bin/python scripts/bench_sentinel.py --games 200 --difficulty 4 \
  --white-value-net

# Sentinel in reconsider mode vs baseline
.venv/bin/python scripts/bench_sentinel.py --games 200 --difficulty 4 \
  --white-sentinel reconsider
```

Each config plays White in half the games and Black in the other half to cancel first-mover bias. Results are reported as **Config A edge** in percentage points.

`--time-budget 0.25` (default) gives ~13 s/game; 200 games ≈ 45 min.
Increase to `--time-budget 1.0` for stronger play at the cost of longer runtime.

---

## Graceful degradation

If `best.pt` is missing or PyTorch is not installed, the sentinel silently disables itself at startup. The game runs identically. No crash, no error shown to the user.
