# NMM Script Command Reference

## Board Layout

```
a7───────────d7───────────g7
│            │            │
│   b6───────d6──────f6  │
│   │         │         │  │
│   │   c5───d5───e5   │  │
│   │   │           │  │  │
a4──b4──c4       e4──f4───g4
│   │   │           │  │  │
│   │   c3───d3───e3   │  │
│   │         │         │  │
│   b2───────d2──────f2  │
│            │            │
a1───────────d1───────────g1
```

## Rebuild Rust Engine

```
bash scripts/build_rust.sh
```

Run whenever `native/nmm_core/src/` files change.

## Self-Play Game Generation

```
python tools/self_play.py --games 500 --no-llm --white 7 --black 3 --parallel 4  \
  --game-dir data/games/self_play --random-difficulty
```

| Flag | Default | Description |
| - | - | - |
| `--games N` | 10 | Number of games to play |
| `--white D` | 5 | White difficulty (1–10) |
| `--black D` | 5 | Black difficulty (1–10) |
| `--parallel N` | 1 | Parallel workers |
| `--game-dir PATH` | — | Output directory for game JSONL files |
| `--random-difficulty` | off | Randomise difficulty each game within range |
| `--min-difficulty D` | 1 | Lower bound for random difficulty |
| `--max-difficulty D` | 9 | Upper bound for random difficulty |
| `--blunder P` | 0.0 | Probability of blunder move |
| `--no-llm` | off | Skip LLM commentary |
| `--swap` | off | Alternate which side plays first |
| `--personalities LIST` | — | Comma-separated personality names |
| `--white-personality NAME` | — | Specific personality for White |
| `--black-personality NAME` | — | Specific personality for Black |
| `--verbose` | off | Per-game status lines |


## Value Net — Basic Training (train_value_net.py)

```
.venv/bin/python tools/train_value_net.py  \
  --games-dir data/games --games-dir data/human_games  \
  --decisive-only --epochs 30 --output data/value_net.npz
```

| Flag | Default | Description |
| - | - | - |
| `--games-dir PATH` | — | Game directory (repeatable for multiple dirs) |
| `--output PATH` | — | Output .npz path |
| `--epochs N` | 30 | Training epochs |
| `--lr F` | 0.001 | Learning rate |
| `--batch-size N` | 256 | Mini-batch size |
| `--decisive-only` | off | Skip drawn games |


## Value Net — Human-Filtered V3 (train_value_net_filtered.py)

```
.venv/bin/python tools/train_value_net_filtered.py
```

| Flag | Default | Description |
| - | - | - |
| `--games-dir PATH` | `data/human_games` | Source game directory |
| `--output PATH` | auto | Output .npz path |
| `--epochs N` | 100 | Training epochs |
| `--lr F` | 3e-4 | Learning rate |
| `--batch-size N` | 256 | Mini-batch size |
| `--val-frac F` | 0.1 | Validation fraction |
| `--patience N` | 10 | Early-stop patience (epochs) |
| `--weight-decay F` | 1e-4 | L2 regularisation |
| `--placement-blend F` | 0.35 | Weight of placement-phase positions |
| `--heuristic-scale F` | auto | Scale for heuristic target labels |
| `--min-elo N` | 0 | Minimum player ELO filter |
| `--decisive-only` | off | Skip drawn games |


**Benchmark all nets once V3 is trained:**

```
.venv/bin/python tools/bench_vn_filtered.py --diff 4 --budget 3.0 --games-per-pair 10

# Longer run (200 games)
.venv/bin/python tools/bench_vn_filtered.py --diff 5 --budget 3.0 --games-per-pair 20
```

| Flag | Default | Description |
| - | - | - |
| `--diff D` | 4 | AI difficulty for benchmark games |
| `--budget F` | 3.0 | Per-move time budget (seconds) |
| `--games-per-pair N` | 10 | Games per config pair (must be even) |
| `--out PATH` | — | JSON results output path |


## Value Net — Phase Trajectory Training (train_vn_trajectory.py)

Trains THREE phase-specific value nets (placement / movement / fly) saved as `data/value_net_phase_place.npz`, `data/value_net_phase_move.npz`, `data/value_net_phase_fly.npz`. At inference the web app loads all three as a `PhaseValueNet` and dispatches `predict()` to the correct sub-net based on the game phase.

Reward per position: `malom_sign × best_composite_quality` where composite = 0.6 × sentinel + 0.4 × heuristic (same signals as GAP net, computed live per trajectory position). Winner moves are randomly sampled from ALL winning successors (not just best DTW).

**Full training run — all three phases (default settings):**

```
.venv/bin/python scripts/train_vn_trajectory.py
```

Defaults: 5000 starts · 40 epochs · traj depth 40 · sentinel loaded automatically · bench 2000 accuracy.

**Train only one phase:**

```
.venv/bin/python scripts/train_vn_trajectory.py --phase move
```

**Smaller run to check quality first:**

```
.venv/bin/python scripts/train_vn_trajectory.py --n-starts 500 --epochs 20
```

**Fine-tune from existing phase nets:**

```
.venv/bin/python scripts/train_vn_trajectory.py  \
  --continue-from data/value_net_phase  \
  --n-starts 2000 --epochs 20
```

**Benchmark only (no training):**

```
.venv/bin/python scripts/train_vn_trajectory.py  \
  --epochs 0 --bench-accuracy 2000 --bench-traj 300 --bench-games 50
```

| Flag | Default | Description |
| - | - | - |
| `--db PATH` | `data/human_db.sqlite` | Human DB for starting positions |
| `--malom-db PATH` | `.../Std_DD_89adjusted` | Malom DB directory |
| `--sentinel PATH` | `learned_ai/sentinel/checkpoints/best.pt` | Sentinel checkpoint for composite quality (falls back to heuristic-only if missing) |
| `--out PATH` | `data/value_net_phase` | Base output path (no extension); creates _place/_move/_fly.npz |
| `--phase PHASE` | all | Which phase(s) to train: `place`, `move`, `fly`, or `all` |
| `--n-starts N` | 5000 | Starting positions to sample from human_db |
| `--traj-depth N` | 40 | Max plies per trajectory |
| `--min-placed N` | 7 | Min total pieces placed in start position |
| `--bucket-cap N` | none | Max starts per placement-stage bucket |
| `--seed N` | 42 | RNG seed for winner-move random sampling |
| `--use-heuristic` | off | Use heuristic AI for loser moves (slower) |
| `--heuristic-difficulty D` | 4 | Heuristic AI difficulty (only with `--use-heuristic`) |
| `--heuristic-time F` | 0.05 | Time budget per AI move in seconds (only with `--use-heuristic`) |
| `--epochs N` | 40 | Training epochs |
| `--lr F` | 8e-4 | Learning rate |
| `--batch N` | 512 | Mini-batch size |
| `--continue-from PATH` | — | Base path of existing phase nets to fine-tune |
| `--bench-accuracy N` | 2000 | Positions for accuracy test (move-phase net) |
| `--bench-traj N` | 300 | Positions for trajectory-follow test |
| `--bench-games N` | 0 | Full games vs raw heuristic (0 = skip) |
| `--bench-gap` | off | Also benchmark GAP net for comparison |
| `--difficulty D` | 6 | Difficulty for full-game benchmark |
| `--vn-blend N` | 80 | VN blend % for full-game benchmark |
| `--time-budget F` | 0.5 | Per-move budget for full-game benchmark |


## Trajectory Value Net — Round-Robin Benchmark (bench_trajectory_value_net.py)

Tests five configs against each other: Baseline, TrajVN at 10/30/60% blend, and GapNet.

**Full round-robin (10 pairs):**

```
.venv/bin/python scripts/bench_trajectory_value_net.py --games 20 --difficulty 4
```

**Single matchup:**

```
.venv/bin/python scripts/bench_trajectory_value_net.py --matchup TrajVN-30% Baseline --games 40
```

**Custom blend percentages:**

```
.venv/bin/python scripts/bench_trajectory_value_net.py --blends 20 40 80 --games 20
```

| Flag | Default | Description |
| - | - | - |
| `--games N` | 20 | Games per matchup pair (must be even) |
| `--difficulty D` | 4 | AI difficulty for all configs |
| `--time-limit F` | — | Per-move time limit in seconds |
| `--vn-path PATH` | `data/value_net_trajectory.npz` | Trajectory value net path |
| `--gap-path PATH` | `data/gap_net.npz` | Gap net path |
| `--blends PCT…` | `10 30 60` | Value-net blend percentages to test |
| `--matchup A B` | — | Single matchup between two named configs instead of round-robin |


Configs: `Baseline`, `TrajVN-10%`, `TrajVN-30%`, `TrajVN-60%`, `GapNet`

## GAP Net — Build Dataset + Train

**Step 1 — Build training dataset:**

```
.venv/bin/python scripts/build_gap_dataset.py
```

| Flag | Default | Description |
| - | - | - |
| `--db PATH` | `data/human_db.sqlite` | Human DB source |
| `--sentinel PATH` | `learned_ai/sentinel/checkpoints/best.pt` | Sentinel checkpoint |
| `--value-net PATH` | `data/value_net.npz` | Value net checkpoint |
| `--out PATH` | `data/gap_net_training.npz` | Output training data |
| `--samples-per-category N` | 15000 | Samples per WDL category |
| `--dtw-threshold N` | 15 | DTW threshold for gap labelling |


**Step 2 — Train the GAP net:**

```
.venv/bin/python tools/train_gap_net.py --epochs 80
```

| Flag | Default | Description |
| - | - | - |
| `--epochs N` | 80 | Training epochs |
| `--lr F` | 0.001 | Learning rate |
| `--data PATH` | `data/gap_net_training.npz` | Training data |
| `--out PATH` | `data/gap_net.npz` | Output net |


## Benchmarking — Sentinel / GAP Net / Tournament

**Base vs base (sanity check):**

```
.venv/bin/python scripts/bench_sentinel.py --games 200 --difficulty 4
```

**Sentinel (20% gap) vs base:**

```
.venv/bin/python scripts/bench_sentinel.py --games 200 --difficulty 4  \
  --white-sentinel score_adjust
```

**GAP net vs base:**

```
.venv/bin/python scripts/bench_sentinel.py --games 200 --difficulty 4 --white-gap-net
```

**Sentinel + GAP net vs base:**

```
.venv/bin/python scripts/bench_sentinel.py --games 200 --difficulty 4  \
  --white-sentinel score_adjust --white-gap-net
```

| Flag | Default | Description |
| - | - | - |
| `--games N` | 4 | Number of games |
| `--difficulty D` | 4 | AI difficulty (1–10) |
| `--white-sentinel MODE` | — | Sentinel mode for White: `score_adjust` etc. |
| `--black-sentinel MODE` | — | Sentinel mode for Black |
| `--sentinel-path PATH` | best.pt | Sentinel checkpoint |
| `--sentinel-scale F` | 0.20 | Min gap fraction for sentinel intervention |
| `--white-value-net` | off | Enable value net for White |
| `--black-value-net` | off | Enable value net for Black |
| `--vn-blend N` | 0 | Value net blend % |
| `--white-gap-net` | off | Enable GAP net for White |
| `--black-gap-net` | off | Enable GAP net for Black |
| `--time-budget F` | — | Per-move time budget override |
| `--suite` | off | Run preset benchmark suite |
| `--round-robin` | off | Round-robin all configurations |


**Full round-robin tournament (S0/S10/S20/S30 + VN blends):**

```
.venv/bin/python tools/bench_tournament.py --diff 4 --budget 3.0 --games-per-pair 10
```

| Flag | Default | Description |
| - | - | - |
| `--diff D` | 4 | AI difficulty |
| `--budget F` | 3.0 | Per-move time budget (seconds) |
| `--games-per-pair N` | 10 | Games per config pair (must be even) |
| `--out PATH` | `eval_results.json` | JSON results output |


**Sentinel v2 benchmark (after training v2/best.pt):**

```
.venv/bin/python tools/bench_sentinel_v2.py --diff 4 --budget 3.0 --games-per-pair 10 --gap 20
```

| Flag | Default | Description |
| - | - | - |
| `--diff D` | 4 | AI difficulty |
| `--budget F` | 3.0 | Per-move time budget (seconds) |
| `--games-per-pair N` | 10 | Games per config pair |
| `--old-ckpt PATH` | — | Old sentinel checkpoint for comparison |
| `--new-ckpt PATH` | — | New sentinel checkpoint to test |
| `--out PATH` | — | JSON results output |


## Opening Audit

```
.venv/bin/python scripts/audit_openings.py --games 5 --diff 4
```

| Flag | Default | Description |
| - | - | - |
| `--games N` | 0 | Games to simulate per opening (0 = eval only) |
| `--diff D` | 3 | Heuristic difficulty for simulation |
| `--threshold F` | 0.06 | Eval margin for W/B vs equal classification |
| `--sim-margin F` | 0.08 | Win-rate margin for simulation classification |
| `--only-id ID` | — | Audit a single opening by ID |
| `--dry-run` | off | Print report without writing files |
| `--seed N` | 42 | Random seed |


## Human DB — Import PlayOK Games

```
python tools/import_playok.py  \
  --archive ~/playok_archive/games  \
  --output data/human_games
```

| Flag | Default | Description |
| - | - | - |
| `--archive PATH` | `~/playok_archive/games` | Input archive directory |
| `--output PATH` | `data/human_games` | Output directory for JSONL files |
| `--dry-run` | off | Count games without writing |
| `--validate-only` | off | Check legality without writing |
| `--limit N` | — | Stop after N new games |
| `--verbose` | off | Per-game status lines |


## Human DB — Rebuild from Scratch

```
.venv/bin/python tools/build_human_db.py  \
  --games-dir data/human_games  \
  --extra-dirs data/games  \
  --malom-db /mnt/windows/NMM_DB/Malom_Standard_Ultra-strong_1.1.0/Std_DD_89adjusted  \
  --output data/human_db.sqlite  \
  --rebuild
```

| Flag | Default | Description |
| - | - | - |
| `--games-dir PATH` | `data/human_games` | Primary game directory |
| `--extra-dirs PATH…` | — | Additional directories |
| `--output PATH` | `data/human_db.sqlite` | Output SQLite path |
| `--malom-db PATH` | — | Malom DB directory for WDL annotation |
| `--no-malom` | off | Skip Malom annotation |
| `--rebuild` | off | Clear DB and reprocess everything from scratch |
| `--update` | off | Only process new or changed files |


> **Note:** Do not use both `--rebuild` and `--update`. Use neither to append without checking.

## Human DB — Incremental Update (SHA-tracked)

```
.venv/bin/python tools/build_human_db_sha.py  \
  --update  \
  --games-dir data/human_games  \
  --output data/human_db.sqlite  \
  --malom-db /mnt/windows/NMM_DB/Malom_Standard_Ultra-strong_1.1.0/Std_DD_89adjusted
```

| Flag | Default | Description |
| - | - | - |
| `--games-dir PATH` | `data/human_games` | Primary game directory |
| `--extra-dirs PATH…` | — | Additional directories |
| `--output PATH` | `data/human_db.sqlite` | Output SQLite path |
| `--malom-db PATH` | — | Malom DB directory |
| `--no-malom` | off | Skip Malom annotation |
| `--update` | off | Only process files whose SHA-256 changed |
| `--rebuild` | off | Clear DB and reprocess from scratch |


## Full Game DB — Build

```
python tools/build_fullgame_db.py  \
  --expand-from-games data/games  \
  --min-seed-frequency 3  \
  --early-expand-depth 4  \
  --expand-depth 6  \
  --output /mnt/windows/NMM_DB/fullgame.bin  \
  --temp-db /mnt/windows/NMM_DB/  \
  --max-db-gb 40
```

| Flag | Default | Description |
| - | - | - |
| `--expand-from-games DIR` | `data/games` | Human game records to seed from |
| `--output PATH` | `data/fullgame.bin` | Output binary file |
| `--db-dir DIR` | — | Shorthand for `--output <dir>/fullgame.bin` |
| `--temp-db PATH` | alongside output | Temporary SQLite build DB (use large drive) |
| `--max-db-gb GB` | 10.0 | Stop BFS when temp DB exceeds this size |
| `--max-gb GB` | 6.0 | Abort when process RSS exceeds this (GB) |
| `--min-seed-frequency N` | 2 | Min human-game visits to seed a BFS position |
| `--expand-depth D` | 4 | BFS depth for late-game positions |
| `--early-expand-depth D` | 2× expand-depth | BFS depth for early-game positions |
| `--max-expand-positions N` | unlimited | Hard cap on BFS positions |
| `--passes N` | 6 | Backpropagation passes for W/L labelling |
| `--dry-run` | off | Synthetic build (no disk write) |
| `--quiet` | off | Suppress progress logging |


## Endgame DB — Build

```
python tools/build_endgame_db.py --build-all --skip-existing  \
  --out-dir /mnt/windows/NMM_DB
```

| Flag | Default | Description |
| - | - | - |
| `--out-dir PATH` | `data/endgame` | Directory for `endgame_*.wdl` files |
| `--build-all` | off | Build all tables in dependency order |
| `--max-sum N` | 11 | Max nW+nB when using `--build-all` |
| `--nW N` | — | White piece count (single table build) |
| `--nB N` | — | Black piece count (single table build) |
| `--skip-existing` | off | Skip tables whose .wdl already exists |
| `--quiet` | off | Suppress per-pass logging |


## Sentinel v2 — Train

```
.venv/bin/python scripts/train_sentinel.py  \
  --game-dir data/games  \
  --human-game-dir data/human_games  \
  --ai-game-dir data/ai_games  \
  --db-path /mnt/windows/NMM_DB/Malom_Standard_Ultra-strong_1.1.0/Std_DD_89adjusted  \
  --drop-db-features  \
  --aux-wdl --lambda-wdl 0.4  \
  --contrastive --lambda-contrastive 0.4  \
  --curriculum  \
  --epochs 50 --epochs-phase1 10  \
  --lr-phase1 5e-3 --lr-phase2 5e-3  \
  --out-dir learned_ai/sentinel/checkpoints/v2  \
  --device cuda
```

| Flag | Default | Description |
| - | - | - |
| `--game-dir PATH` | `data/games` | AI self-play game directory |
| `--human-game-dir PATH` | — | Human game directory |
| `--ai-game-dir PATH` | — | Additional AI game directory |
| `--db-path PATH` | — | Malom DB directory |
| `--dataset PATH` | — | Preprocessed .npz (skips replay) |
| `--drop-db-features` | off | Zero out DB-derived input features |
| `--aux-wdl` | off | Add auxiliary WDL prediction head |
| `--lambda-wdl F` | 0.3 | Weight for WDL auxiliary loss |
| `--contrastive` | off | Enable contrastive loss |
| `--lambda-contrastive F` | 0.3 | Weight for contrastive loss |
| `--curriculum` | off | Phase 1 → Phase 2 curriculum schedule |
| `--epochs N` | — | Total training epochs |
| `--epochs-phase1 N` | — | Epochs in phase 1 |
| `--lr-phase1 F` | — | Learning rate for phase 1 |
| `--lr-phase2 F` | 1e-4 | Learning rate for phase 2 |
| `--out-dir PATH` | — | Checkpoint output directory |
| `--device` | `cpu` | `cpu` or `cuda` |
| `--resume PATH` | — | Resume from checkpoint |
| `--decisive-only` | off | Skip drawn games |
| `--trajectory-weight` | off | Weight samples by trajectory quality |
| `--limit N` | — | Max game files to load |
| `--config PATH` | — | JSON config file (overrides flags) |


## Learned AI — Imitation Data Generation

Run these before training specialist networks.

**Step 1 — AI self-play imitation data (~10h):**

```
.venv/bin/python scripts/gen_imitation_data.py  \
  --games 1000 --diff 7  \
  --sentinel learned_ai/sentinel/checkpoints/best.pt
```

| Flag | Default | Description |
| - | - | - |
| `--games N` | 2000 | Games to generate |
| `--diff D` | 3 | AI difficulty |
| `--sentinel PATH` | — | Sentinel checkpoint |
| `--malom PATH` | — | Malom DB path |
| `--value-net PATH` | `data/value_net.npz` | Value net checkpoint |
| `--out PATH` | auto | Output .npz |
| `--max-ply N` | 300 | Max plies per game |
| `--seed N` | 42 | Random seed |


**Step 2 — Human game imitation data (62-float, legacy):**

```
.venv/bin/python scripts/gen_human_imitation_data.py
```

| Flag | Default | Description |
| - | - | - |
| `--games-dir PATH` | `data/games` | Source game directory |
| `--out PATH` | `learned_ai/data/human_imitation.npz` | Output .npz |
| `--sentinel PATH` | `learned_ai/sentinel/checkpoints/best.pt` | Sentinel checkpoint |
| `--malom PATH` | — | Malom DB path |
| `--value-net PATH` | `data/value_net.npz` | Value net |
| `--won-weight F` | 1.0 | Sample weight for winner positions |
| `--draw-weight F` | 0.3 | Sample weight for draw positions |
| `--loser-weight F` | 0.5 | Sample weight for loser positions from human-won games |


**Step 2b — Human game imitation data v2 (122-float, for v2 specialists; ~6–8h with sentinel):**

Uses `encode_position_with_lookahead` with 15-ply LookaheadAdvisor + GapNet. Cap `--max-moves 120` prevents stalling on very long game files.

```
.venv/bin/python scripts/gen_human_imitation_data_v2.py  \
  --gap-net data/gap_net.npz --max-moves 120
```

Output: `learned_ai/data/human_imitation2.npz` — 13,040 positions, 122-float features.

| Flag | Default | Description |
| - | - | - |
| `--games-dir PATH` | `data/human_games` | Source game directory |
| `--out PATH` | `learned_ai/data/human_imitation2.npz` | Output .npz |
| `--sentinel PATH` | `best.pt` | Sentinel checkpoint |
| `--value-net PATH` | `data/value_net.npz` | Value net |
| `--gap-net PATH` | `data/gap_net.npz` | GapNet (blunder density) |
| `--max-moves N` | 120 | Cap moves per game to avoid stalls |


## Learned AI — Specialist Training v3 (Opening / Midgame / Endgame)

Three independent phase specialists (opening / midgame / endgame). Each one sits on top of the classical engine's alpha-beta search and re-ranks its top-K candidates.

**Per-move features (126 floats each row):**
- **62 base** — sentinel score, heuristic + VN blended eval and delta, counterfactual block, `is_engine_top1` flag, and the 58-float sentinel/board context.
- **60 lookahead** — 15 half-plies × 4 signals (heuristic + VN + sentinel + gap). Training simulates only `--sim-ply-depth` half-plies (default 5) and pads to full width; inference always runs full 15.
- **4 top-K extras** — `ab_score_norm`, `ab_rank_norm`, `human_freq`, `human_rank`.

**Value input (80 floats):** 23 encoder base + 9 history (last 3 moves' from/to/capture as normalised indices) + 48 raw-board one-hot (24 positions × 2 colours).

**Model:** `ScaffoldedPolicyNet` — policy MLP `126 → 512 → 256 → 128 → 1`, value MLP `80 → 256 → 128 → 64 → 1`. ~289 k params.

**Difficulty:** 20 levels. Log-scale per-move opponent budget: L1 ≈ 1 ms → L15+ caps at 2 s (mid/end) or 1 s (opening). **Advancement:** Sanmill superiority-probability gate — `P(true score > target) ≥ 0.95` on the last 50 games; target ramps 55% (L1) → 60% (L20) with time-of-flight relaxation to a 51% floor after 1000+ stalled games. Checked every 10 games once `games_at_level ≥ 20`.

### Prerequisites

Generate the 122-float human imitation warm-start dataset once (`human_imitation2.npz`, ~6-8 h wall):

```
.venv/bin/python scripts/gen_human_imitation_data_v2.py \
  --gap-net data/gap_net.npz --max-moves 120
```

### Fresh training runs (recommended flags)

Speed flags: `--sim-ply-depth 5` (~3× lookahead speed-up during training; inference stays at 15) and `--minimal-rollouts` (one primary rollout per game, no confirm / retry). Launch each specialist in its own terminal / tmux pane — they train independently and in parallel.

**Opening specialist — fresh:**

```
.venv/bin/python scripts/train_s_open_v2.py \
  --max-games 30000 --batch-games 10 \
  --sim-ply-depth 5 --minimal-rollouts \
  --self-play-ratio 0.05
```

**Midgame specialist — fresh:**

```
.venv/bin/python scripts/train_s_mid_v2.py \
  --max-games 30000 --batch-games 10 \
  --sim-ply-depth 5 --minimal-rollouts \
  --self-play-ratio 0.05
```

**Endgame specialist — fresh:**

```
.venv/bin/python scripts/train_s_end_v2.py \
  --max-games 30000 --batch-games 10 \
  --sim-ply-depth 5 --minimal-rollouts \
  --self-play-ratio 0.05 \
  --malom /mnt/windows/NMM_DB/Malom_Standard_Ultra-strong_1.1.0/Std_DD_89adjusted
```

### Resume from best checkpoint

```
.venv/bin/python scripts/train_s_open_v2.py --auto-resume-best --max-games 30000 --batch-games 10 --sim-ply-depth 5 --minimal-rollouts
.venv/bin/python scripts/train_s_mid_v2.py  --auto-resume-best --max-games 30000 --batch-games 10 --sim-ply-depth 5 --minimal-rollouts
.venv/bin/python scripts/train_s_end_v2.py  --auto-resume-best --max-games 30000 --batch-games 10 --sim-ply-depth 5 --minimal-rollouts --malom /mnt/windows/NMM_DB/Malom_Standard_Ultra-strong_1.1.0/Std_DD_89adjusted
```

### Smoke test (2-5 games each, no warm-start)

```
.venv/bin/python scripts/train_s_open_v2.py --max-games 5 --no-s1a-warmstart --sim-ply-depth 5 --minimal-rollouts
.venv/bin/python scripts/train_s_mid_v2.py  --max-games 5 --no-s1a-warmstart --sim-ply-depth 5 --minimal-rollouts
.venv/bin/python scripts/train_s_end_v2.py  --max-games 5 --no-s1a-warmstart --sim-ply-depth 5 --minimal-rollouts --malom /mnt/windows/NMM_DB/Malom_Standard_Ultra-strong_1.1.0/Std_DD_89adjusted
```

### Common flags (all three v2/v3 specialists)

| Flag | Default | Description |
| - | - | - |
| `--sentinel PATH` | `best.pt` | Sentinel checkpoint |
| `--value-net PATH` | `data/value_net.npz` | Trajectory value net |
| `--gap-net PATH` | `data/gap_net.npz` | Gap net (blunder density) |
| `--out-dir PATH` | `learned_ai/checkpoints/scaffolded/s_*_v2` | Checkpoint output |
| `--s1a-data PATH` | `data/human_imitation2.npz` | Pre-RL imitation warm-start data |
| `--no-s1a-warmstart` | off | Skip s1a warm-start (start RL from scratch) |
| `--batch-games N` | 1 | Parallel primary rollouts via ThreadPoolExecutor. 10 recommended on 16+ cores; diminishing returns beyond 24. |
| `--sim-ply-depth N` | 5 | LookaheadAdvisor simulated depth during training. Feature width still 60 floats via padding; inference runs full 15. |
| `--minimal-rollouts` | off | Skip retry + confirm rollouts (branches already off by default). ~3× training throughput at the cost of sample efficiency. |
| `--max-games N` | 5000 | Games (soft cap; specialist stops early on hitting max difficulty). |
| `--diff-max N` | 20 | Maximum difficulty level. |
| `--diff-start N` | 1 | Starting difficulty level. |
| `--time-budget F` | -1 (auto per-level) | Override per-move budget for the opponent's α-β search. |
| `--self-play-ratio F` | 0.5 | Fraction of games vs frozen model. 0.05 recommended once RL is stable. |
| `--lr F` | 1e-4 | Learning rate. |
| `--entropy-coef F` | 0.01 | Entropy regularisation coefficient. |
| `--update-every N` | 16 | Policy update interval (steps). |
| `--rolling-win N` | 50 | Rolling window for the Sanmill advance test. |
| `--resume PATH` | — | Explicit checkpoint to resume from. |
| `--auto-resume-best` | off | Auto-resume from `s_*_v2/best.pt`. |
| `--ppo` | off | Use PPO instead of A2C. |
| `--seed N` | 42 | Random seed. |
| `--malom PATH` | — | (endgame only) Malom perfect DB directory for endgame reward + lookahead early-exit. |

### Notes for overnight runs

- Each specialist runs independently. Launch all three in parallel, one per terminal / tmux pane.
- At `--batch-games 10` on a 16-core CUDA box, expect **~300-1100 games/hour at diff 1** with the v3 speed flags applied.
- Watch `htop` — if all CPU cores are pegged, raise `--batch-games` cautiously (10 → 16 → 24). Beyond 24 you'll see diminishing returns from Python GIL + memory pressure.
- Advance-check log line format: `[s_open_v2] advance-check @ diff 3: P=0.982 ≥ 0.95 (target=0.545, score=0.760)`. When the P-value stays < 0.5 for 5000+ games at a level, more games won't help — the model has plateaued.


## Learned AI — Specialist Benchmark (bench_scaffolded.py)

Runs the **v2 SpecialistRouter** (opening + midgame + endgame specialists routed by phase) as one player, versus a matrix of heuristic-opponent configurations at multiple difficulties. Colours alternate every game. Streams results to `data/bench/scaffolded_v2_<timestamp>.jsonl` (one row per matchup) — safe for overnight runs.

**Opponent configurations** (all share `value_net_blend=20`, sentinel `_sentinel_activation_prob=0.20`):

| Config | Description |
| --- | --- |
| `raw` | GameAI only (no sentinel / vn / gap) |
| `sentinel` | GameAI + sentinel score_adjust (20% intervention) |
| `vn` | GameAI + value_net (blend 20%) |
| `gap` | GameAI + gap_net (blunder-zone exploitation) |
| `sv` | GameAI + sentinel + value_net |
| `full` | GameAI + sentinel + value_net + gap_net |
| `deep` | Full stack + max_search_depth=25 (extended tactical search) |

**Heuristic time budget**: by default (`--time-budget -1`), each opponent move uses the SAME per-difficulty cap the game applies in real play:

| Diff | Cap | Notes |
| --- | --- | --- |
| 1–5 | 15 s | Reduced to 3 s (first 2 moves) / 10 s (≤4 pieces on board) automatically |
| 6 | 30 s | Same early-placement reductions |
| 7 | 45 s | Same early-placement reductions |
| 8–10 | 60 s | Same early-placement reductions |

Pass `--time-budget SECONDS` (positive value) to override with a flat cap — useful for fast smoke tests since the game-native caps are slow (they mirror real interactive play).

**Quick smoke test (10 games, diff 5, two configs, capped at 2 s/move):**

```
.venv/bin/python scripts/bench_scaffolded.py --games 10 --difficulties 5 \
    --opponents raw,full --time-budget 2.0
```

**Overnight sweep at game-native per-difficulty budgets:**

```
.venv/bin/python scripts/bench_scaffolded.py --games 40 --difficulties 3,5,7,9
```

**Deeper specialist lookahead (25 plies) at game-native budgets:**

```
.venv/bin/python scripts/bench_scaffolded.py --games 40 --difficulties 5,7,9 \
    --specialist-ply-depth 25
```

| Flag | Default | Description |
| --- | --- | --- |
| `--games N` | 40 | Games per matchup (alternating colours) |
| `--difficulties LIST` | `3,5,7,9` | Comma-separated GameAI difficulties (1–10) |
| `--opponents LIST` | `raw,sentinel,vn,gap,sv,full,deep` | Which configs to test (comma-separated) |
| `--time-budget F` | `-1` (game-native) | Per-move heuristic budget. ≤ 0 → use game's per-difficulty caps (15/30/45/60 s + early reductions). Positive → flat override. |
| `--specialist-ply-depth N` | 15 | LookaheadAdvisor ply depth used by the specialists |
| `--max-plies N` | 400 | Max plies per game before draw |
| `--sentinel-path PATH` | `learned_ai/sentinel/checkpoints/best.pt` | Sentinel checkpoint |
| `--value-net-path PATH` | `data/value_net.npz` | Value net checkpoint |
| `--gap-net-path PATH` | `data/gap_net.npz` | Gap net checkpoint |
| `--malom-path PATH` | `/mnt/windows/NMM_DB/Malom_Standard_Ultra-strong_1.1.0/Std_DD_89adjusted` | Malom perfect DB directory |
| `--out-dir PATH` | `data/bench` | Output directory for the JSONL stream |
| `--quiet` | off | Suppress per-game outcome dots |

**Output**: `data/bench/scaffolded_v2_<YYYYMMDD_HHMMSS>.jsonl`, one row per matchup with fields `config, difficulty, games, wins, draws, losses, win_rate, draw_rate, score, elapsed_s, avg_s_per_game, time_budget_s, time_budget_mode, specialist_ply_depth, timestamp`. `time_budget_mode` records whether that row used `game_native_per_diff` or `flat_override`. Score = `(wins + 0.5 × draws) / games`. A final markdown table is printed to stdout.

**Prerequisite**: v2 specialist checkpoints must exist at `learned_ai/checkpoints/scaffolded/{s_open_v2,s_mid_v2,s_end_v2}/best.pt`.


## Learned AI — Extended Tactical Search Benchmark

Head-to-head: extended tactical search (fast_eval=False) vs no-extended (fast_eval=True).

**100-game result at 1s/move (completed 2026-07-10):** Extended 51 — No-Extended 10 — Draws 39 (51% vs 10%, 39% draws). Extended wins 5:1. Result is decisive.

**Run this bench yourself:**

```
.venv/bin/python /tmp/bench_ext_vs_noext.py
```

Variables at top of script: `BUDGET` (seconds/move), `N_GAMES` (total games).

## Learned AI — Overseer Training (RETIRED)

The overseer meta-layer has been removed. Specialists now act directly for their own phase (place → opening, move ≥6 pieces → midgame, move/fly ≤5 pieces → endgame). The scripts below still exist but are not used in the v2 pipeline.

```
.venv/bin/python scripts/train_scaffolded_overseer_parallel.py  \
  --midgame-ckpt  learned_ai/checkpoints/scaffolded/s_mid/best.pt  \
  --endgame-ckpt  learned_ai/checkpoints/scaffolded/s_end/best.pt  \
  --opening-ckpt  learned_ai/checkpoints/scaffolded/s_open/best.pt  \
  --max-games 10000 --max-ply 140  \
  --malom /mnt/windows/NMM_DB/Malom_Standard_Ultra-strong_1.1.0/Std_DD_89adjusted  \
  --workers 8
```

## Puzzle Generators

### Opening / Placement Puzzles

Uses Malom DB path from `data/settings.json`.

```
.venv/bin/python tools/placement_puzzle_generator.py  \
  --depth random --max-winning-moves 2 --side random
```

| Flag | Default | Description |
| - | - | - |
| `--side W\|B\|random` | random | Which side has the winning move |
| `--depth 0\|4\|5\|6\|7` | 0 | Target win depth in winner moves (0 = random) |
| `--max-winning-moves N` | 2 | Reject positions with more than N winning first moves |
| `--count N` | 0 | Puzzles to generate (0 = run forever) |
| `--attempts N` | 3000 | Positions sampled per puzzle attempt |
| `--out PATH` | `data/puzzles/` | Output directory |
| `--print` | off | Print each puzzle JSON to stdout |


### Midgame Puzzles (Malom DB)

```
.venv/bin/python tools/malom_puzzle_generator.py  \
  --depth 6 --max-winning-moves 2 --side random  \
  --min-pieces 4 --max-pieces 16
```

| Flag | Default | Description |
| - | - | - |
| `--side W\|B\|random` | random | Which side has the winning move |
| `--depth 0\|4\|5\|6\|7` | 0 | Target win depth (0 = random) |
| `--max-winning-moves N` | 2 | Reject positions with more than N winning first moves |
| `--min-pieces N` | 4 | Minimum pieces per side |
| `--max-pieces N` | 7 | Maximum pieces per side (raise for richer midgame) |
| `--count N` | 0 | Puzzles to generate (0 = run forever) |
| `--attempts N` | 3000 | Positions sampled per puzzle attempt |
| `--out PATH` | `data/puzzles/` | Output directory |
| `--print` | off | Print each puzzle JSON to stdout |


### Endgame Puzzles (Retrograde DB)

```
.venv/bin/python tools/puzzle_generator.py  \
  --depth random --max-winning-moves 2 --side random --random-db
```

| Flag | Default | Description |
| - | - | - |
| `--side W\|B\|random` | random | Which side has the winning move |
| `--depth 3\|4\|5\|6\|7\|random` | random | Target win depth in winner moves |
| `--max-winning-moves N` | 2 | Reject positions with more than N winning first moves |
| `--db FILE\|random` | random | Specific endgame .wdl file from `data/endgame/` |
| `--random-db` | off | Pick a new random DB file for every attempt (cross-table) |
| `--count N` | 0 | Puzzles to generate (0 = run forever) |
| `--attempts N` | 5000 | Positions sampled per puzzle attempt |
| `--out PATH` | `data/puzzles/` | Output directory |
| `--print` | off | Print each puzzle JSON to stdout |


