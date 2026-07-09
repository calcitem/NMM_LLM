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
bash scripts/build\_rust.sh
```

Run whenever `native/nmm\_core/src/` files change.


## Self-Play Game Generation

```
python tools/self\_play.py --games 500 --no-llm --white 7 --black 3 --parallel 4 \\  
  --game-dir data/games/self\_play --random-difficulty
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



## Value Net — Basic Training (train\_value\_net.py)

```
.venv/bin/python tools/train\_value\_net.py \\  
  --games-dir data/games --games-dir data/human\_games \\  
  --decisive-only --epochs 30 --output data/value\_net.npz
```

| Flag | Default | Description |
| - | - | - |
| `--games-dir PATH` | — | Game directory (repeatable for multiple dirs) |
| `--output PATH` | — | Output .npz path |
| `--epochs N` | 30 | Training epochs |
| `--lr F` | 0.001 | Learning rate |
| `--batch-size N` | 256 | Mini-batch size |
| `--decisive-only` | off | Skip drawn games |



## Value Net — Human-Filtered V3 (train\_value\_net\_filtered.py)

```
.venv/bin/python tools/train\_value\_net\_filtered.py
```

| Flag | Default | Description |
| - | - | - |
| `--games-dir PATH` | `data/human\_games` | Source game directory |
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
.venv/bin/python tools/bench\_vn\_filtered.py --diff 4 --budget 3.0 --games-per-pair 10  
  
\# Longer run (200 games)  
.venv/bin/python tools/bench\_vn\_filtered.py --diff 5 --budget 3.0 --games-per-pair 20
```

| Flag | Default | Description |
| - | - | - |
| `--diff D` | 4 | AI difficulty for benchmark games |
| `--budget F` | 3.0 | Per-move time budget (seconds) |
| `--games-per-pair N` | 10 | Games per config pair (must be even) |
| `--out PATH` | — | JSON results output path |



## Value Net — Phase Trajectory Training (train\_vn\_trajectory.py)

Trains THREE phase-specific value nets (placement / movement / fly) saved as
`data/value\_net\_phase\_place.npz`, `data/value\_net\_phase\_move.npz`, `data/value\_net\_phase\_fly.npz`.
At inference the web app loads all three as a `PhaseValueNet` and dispatches
`predict()` to the correct sub-net based on the game phase.

Reward per position: `malom_sign × best_composite_quality` where composite = 0.6 × sentinel + 0.4 × heuristic (same signals as GAP net, computed live per trajectory position). Winner moves are randomly sampled from ALL winning successors (not just best DTW).

**Full training run — all three phases (default settings):**

```
.venv/bin/python scripts/train\_vn\_trajectory.py
```

Defaults: 5000 starts · 40 epochs · traj depth 40 · sentinel loaded automatically · bench 2000 accuracy.

**Train only one phase:**

```
.venv/bin/python scripts/train\_vn\_trajectory.py --phase move
```

**Smaller run to check quality first:**

```
.venv/bin/python scripts/train\_vn\_trajectory.py --n-starts 500 --epochs 20
```

**Fine-tune from existing phase nets:**

```
.venv/bin/python scripts/train\_vn\_trajectory.py \
  --continue-from data/value\_net\_phase \
  --n-starts 2000 --epochs 20
```

**Benchmark only (no training):**

```
.venv/bin/python scripts/train\_vn\_trajectory.py \
  --epochs 0 --bench-accuracy 2000 --bench-traj 300 --bench-games 50
```

| Flag | Default | Description |
| - | - | - |
| `--db PATH` | `data/human\_db.sqlite` | Human DB for starting positions |
| `--malom-db PATH` | `.../Std\_DD\_89adjusted` | Malom DB directory |
| `--sentinel PATH` | `learned\_ai/sentinel/checkpoints/best.pt` | Sentinel checkpoint for composite quality (falls back to heuristic-only if missing) |
| `--out PATH` | `data/value\_net\_phase` | Base output path (no extension); creates \_place/\_move/\_fly.npz |
| `--phase PHASE` | all | Which phase(s) to train: `place`, `move`, `fly`, or `all` |
| `--n-starts N` | 5000 | Starting positions to sample from human\_db |
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


## Trajectory Value Net — Round-Robin Benchmark (bench\_trajectory\_value\_net.py)

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
.venv/bin/python scripts/build\_gap\_dataset.py
```

| Flag | Default | Description |
| - | - | - |
| `--db PATH` | `data/human\_db.sqlite` | Human DB source |
| `--sentinel PATH` | `learned\_ai/sentinel/checkpoints/best.pt` | Sentinel checkpoint |
| `--value-net PATH` | `data/value\_net.npz` | Value net checkpoint |
| `--out PATH` | `data/gap\_net\_training.npz` | Output training data |
| `--samples-per-category N` | 15000 | Samples per WDL category |
| `--dtw-threshold N` | 15 | DTW threshold for gap labelling |


**Step 2 — Train the GAP net:**

```
.venv/bin/python tools/train\_gap\_net.py --epochs 80
```

| Flag | Default | Description |
| - | - | - |
| `--epochs N` | 80 | Training epochs |
| `--lr F` | 0.001 | Learning rate |
| `--data PATH` | `data/gap\_net\_training.npz` | Training data |
| `--out PATH` | `data/gap\_net.npz` | Output net |



## Benchmarking — Sentinel / GAP Net / Tournament

**Base vs base (sanity check):**

```
.venv/bin/python scripts/bench\_sentinel.py --games 200 --difficulty 4
```

**Sentinel (20% gap) vs base:**

```
.venv/bin/python scripts/bench\_sentinel.py --games 200 --difficulty 4 \\  
  --white-sentinel score\_adjust
```

**GAP net vs base:**

```
.venv/bin/python scripts/bench\_sentinel.py --games 200 --difficulty 4 --white-gap-net
```

**Sentinel + GAP net vs base:**

```
.venv/bin/python scripts/bench\_sentinel.py --games 200 --difficulty 4 \\  
  --white-sentinel score\_adjust --white-gap-net
```

| Flag | Default | Description |
| - | - | - |
| `--games N` | 4 | Number of games |
| `--difficulty D` | 4 | AI difficulty (1–10) |
| `--white-sentinel MODE` | — | Sentinel mode for White: `score\_adjust` etc. |
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
.venv/bin/python tools/bench\_tournament.py --diff 4 --budget 3.0 --games-per-pair 10
```

| Flag | Default | Description |
| - | - | - |
| `--diff D` | 4 | AI difficulty |
| `--budget F` | 3.0 | Per-move time budget (seconds) |
| `--games-per-pair N` | 10 | Games per config pair (must be even) |
| `--out PATH` | `eval\_results.json` | JSON results output |


**Sentinel v2 benchmark (after training v2/best.pt):**

```
.venv/bin/python tools/bench\_sentinel\_v2.py --diff 4 --budget 3.0 --games-per-pair 10 --gap 20
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
.venv/bin/python scripts/audit\_openings.py --games 5 --diff 4
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
python tools/import\_playok.py \\  
  --archive ~/playok\_archive/games \\  
  --output data/human\_games
```

| Flag | Default | Description |
| - | - | - |
| `--archive PATH` | `~/playok\_archive/games` | Input archive directory |
| `--output PATH` | `data/human\_games` | Output directory for JSONL files |
| `--dry-run` | off | Count games without writing |
| `--validate-only` | off | Check legality without writing |
| `--limit N` | — | Stop after N new games |
| `--verbose` | off | Per-game status lines |



## Human DB — Rebuild from Scratch

```
.venv/bin/python tools/build\_human\_db.py \\  
  --games-dir data/human\_games \\  
  --extra-dirs data/games \\  
  --malom-db /mnt/windows/NMM\_DB/Malom\_Standard\_Ultra-strong\_1.1.0/Std\_DD\_89adjusted \\  
  --output data/human\_db.sqlite \\  
  --rebuild
```

| Flag | Default | Description |
| - | - | - |
| `--games-dir PATH` | `data/human\_games` | Primary game directory |
| `--extra-dirs PATH…` | — | Additional directories |
| `--output PATH` | `data/human\_db.sqlite` | Output SQLite path |
| `--malom-db PATH` | — | Malom DB directory for WDL annotation |
| `--no-malom` | off | Skip Malom annotation |
| `--rebuild` | off | Clear DB and reprocess everything from scratch |
| `--update` | off | Only process new or changed files |


> **Note:** Do not use both `--rebuild` and `--update`. Use neither to append without checking.


## Human DB — Incremental Update (SHA-tracked)

```
.venv/bin/python tools/build\_human\_db\_sha.py \\  
  --update \\  
  --games-dir data/human\_games \\  
  --output data/human\_db.sqlite \\  
  --malom-db /mnt/windows/NMM\_DB/Malom\_Standard\_Ultra-strong\_1.1.0/Std\_DD\_89adjusted
```

| Flag | Default | Description |
| - | - | - |
| `--games-dir PATH` | `data/human\_games` | Primary game directory |
| `--extra-dirs PATH…` | — | Additional directories |
| `--output PATH` | `data/human\_db.sqlite` | Output SQLite path |
| `--malom-db PATH` | — | Malom DB directory |
| `--no-malom` | off | Skip Malom annotation |
| `--update` | off | Only process files whose SHA-256 changed |
| `--rebuild` | off | Clear DB and reprocess from scratch |



## Full Game DB — Build

```
python tools/build\_fullgame\_db.py \\  
  --expand-from-games data/games \\  
  --min-seed-frequency 3 \\  
  --early-expand-depth 4 \\  
  --expand-depth 6 \\  
  --output /mnt/windows/NMM\_DB/fullgame.bin \\  
  --temp-db /mnt/windows/NMM\_DB/ \\  
  --max-db-gb 40
```

| Flag | Default | Description |
| - | - | - |
| `--expand-from-games DIR` | `data/games` | Human game records to seed from |
| `--output PATH` | `data/fullgame.bin` | Output binary file |
| `--db-dir DIR` | — | Shorthand for `--output \<dir\>/fullgame.bin` |
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
python tools/build\_endgame\_db.py --build-all --skip-existing \\  
  --out-dir /mnt/windows/NMM\_DB
```

| Flag | Default | Description |
| - | - | - |
| `--out-dir PATH` | `data/endgame` | Directory for `endgame\_\*.wdl` files |
| `--build-all` | off | Build all tables in dependency order |
| `--max-sum N` | 11 | Max nW+nB when using `--build-all` |
| `--nW N` | — | White piece count (single table build) |
| `--nB N` | — | Black piece count (single table build) |
| `--skip-existing` | off | Skip tables whose .wdl already exists |
| `--quiet` | off | Suppress per-pass logging |



## Sentinel v2 — Train

```
.venv/bin/python scripts/train\_sentinel.py \\  
  --game-dir data/games \\  
  --human-game-dir data/human\_games \\  
  --ai-game-dir data/ai\_games \\  
  --db-path /mnt/windows/NMM\_DB/Malom\_Standard\_Ultra-strong\_1.1.0/Std\_DD\_89adjusted \\  
  --drop-db-features \\  
  --aux-wdl --lambda-wdl 0.4 \\  
  --contrastive --lambda-contrastive 0.4 \\  
  --curriculum \\  
  --epochs 50 --epochs-phase1 10 \\  
  --lr-phase1 5e-3 --lr-phase2 5e-3 \\  
  --out-dir learned\_ai/sentinel/checkpoints/v2 \\  
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
.venv/bin/python scripts/gen\_imitation\_data.py \\  
  --games 1000 --diff 7 \\  
  --sentinel learned\_ai/sentinel/checkpoints/best.pt
```

| Flag | Default | Description |
| - | - | - |
| `--games N` | 2000 | Games to generate |
| `--diff D` | 3 | AI difficulty |
| `--sentinel PATH` | — | Sentinel checkpoint |
| `--malom PATH` | — | Malom DB path |
| `--value-net PATH` | `data/value\_net.npz` | Value net checkpoint |
| `--out PATH` | auto | Output .npz |
| `--max-ply N` | 300 | Max plies per game |
| `--seed N` | 42 | Random seed |


**Step 2 — Human game imitation data (~45s, run after or in parallel):**

```
.venv/bin/python scripts/gen\_human\_imitation\_data.py
```

| Flag | Default | Description |
| - | - | - |
| `--games-dir PATH` | `data/games` | Source game directory |
| `--out PATH` | `learned\_ai/data/human\_imitation.npz` | Output .npz |
| `--sentinel PATH` | `learned\_ai/sentinel/checkpoints/best.pt` | Sentinel checkpoint |
| `--malom PATH` | — | Malom DB path |
| `--value-net PATH` | `data/value\_net.npz` | Value net |
| `--won-weight F` | 1.0 | Sample weight for winner positions |
| `--draw-weight F` | 0.3 | Sample weight for draw positions |
| `--loser-weight F` | 0.5 | Sample weight for loser positions from human-won games |



## Learned AI — Specialist Training (Opening / Midgame / Endgame)

Run all three in parallel (independent networks).

**Opening specialist:**

```
.venv/bin/python scripts/train\_scaffolded\_opening.py \\  
  --max-games 10000 --max-ply 140 \\  
  --malom /mnt/windows/NMM\_DB/Malom\_Standard\_Ultra-strong\_1.1.0/Std\_DD\_89adjusted
```

**Midgame specialist:**

```
.venv/bin/python scripts/train\_scaffolded\_midgame.py \\  
  --max-games 10000 --max-ply 140 \\  
  --malom /mnt/windows/NMM\_DB/Malom\_Standard\_Ultra-strong\_1.1.0/Std\_DD\_89adjusted
```

**Endgame specialist:**

```
.venv/bin/python scripts/train\_scaffolded\_endgame.py \\  
  --max-games 10000 --max-ply 140 \\  
  --malom /mnt/windows/NMM\_DB/Malom\_Standard\_Ultra-strong\_1.1.0/Std\_DD\_89adjusted
```

Common flags (all three specialists):

| Flag | Default | Description |
| - | - | - |
| `--malom PATH` | — | Malom DB directory |
| `--sentinel PATH` | `best.pt` | Sentinel checkpoint |
| `--value-net PATH` | `data/value\_net.npz` | Value net |
| `--out-dir PATH` | `learned\_ai/checkpoints/scaffolded/s\_\*` | Checkpoint output |
| `--max-games N` | 5000 | Training games |
| `--max-ply N` | auto | Max plies per game |
| `--lr F` | auto | Learning rate |
| `--entropy-coef F` | auto | Entropy regularisation coefficient |
| `--update-every N` | auto | Policy update interval (games) |
| `--rolling-win N` | auto | Rolling window for win-rate tracking |
| `--resume PATH` | — | Explicit checkpoint to resume from |
| `--auto-resume-best` | off | Auto-resume from `s\_\*/best.pt` |
| `--ppo` | off | Use PPO instead of A2C |
| `--seed N` | 42 | Random seed |



## Learned AI — Overseer Training

Run after all three specialists converge.

```
.venv/bin/python scripts/train\_scaffolded\_overseer\_parallel.py \\  
  --midgame-ckpt  learned\_ai/checkpoints/scaffolded/s\_mid/best.pt \\  
  --endgame-ckpt  learned\_ai/checkpoints/scaffolded/s\_end/best.pt \\  
  --opening-ckpt  learned\_ai/checkpoints/scaffolded/s\_open/best.pt \\  
  --max-games 10000 --max-ply 140 \\  
  --malom /mnt/windows/NMM\_DB/Malom\_Standard\_Ultra-strong\_1.1.0/Std\_DD\_89adjusted \\  
  --workers 8
```

**With self-play and auto-resume (post game 150):**

```
.venv/bin/python scripts/train\_scaffolded\_overseer\_parallel.py \\  
  --midgame-ckpt  learned\_ai/checkpoints/scaffolded/s\_mid/best.pt \\  
  --endgame-ckpt  learned\_ai/checkpoints/scaffolded/s\_end/best.pt \\  
  --opening-ckpt  learned\_ai/checkpoints/scaffolded/s\_open/best.pt \\  
  --max-games 10000 --max-ply 140 \\  
  --malom /mnt/windows/NMM\_DB/Malom\_Standard\_Ultra-strong\_1.1.0/Std\_DD\_89adjusted \\  
  --workers 8 --self-play-ratio 0.1 --max-branches-per-game 0 --auto-resume-best
```

| Flag | Default | Description |
| - | - | - |
| `--opening-ckpt PATH` | — | Opening specialist checkpoint |
| `--midgame-ckpt PATH` | — | Midgame specialist checkpoint |
| `--endgame-ckpt PATH` | — | Endgame specialist checkpoint |
| `--malom PATH` | — | Malom DB directory |
| `--workers N` | 4 | Parallel worker processes |
| `--max-games N` | 5000 | Training games |
| `--max-ply N` | auto | Max plies per game |
| `--max-ply-branch N` | auto | Max plies for branch games |
| `--self-play-ratio F` | auto | Fraction of games using self-play |
| `--max-branches-per-game N` | 0 | Branch games per main game |
| `--auto-resume-best` | off | Auto-resume from `s\_over/best.pt` |
| `--resume PATH` | — | Explicit checkpoint to resume |
| `--scratch` | off | Start from scratch (ignore existing ckpt) |
| `--out-dir PATH` | `s\_over/` | Checkpoint output directory |
| `--s1b-data PATH` | `learned\_ai/data/human\_imitation.npz` | Human imitation data for refresher |
| `--s1b-refresher-epochs N` | auto | Epochs per refresher cycle |
| `--no-s1b-refresher` | off | Disable human imitation refresher |
| `--gameai-depth N` | 7 | Depth for in-loop heuristic AI opponent |
| `--human-db PATH` | — | Human DB path |
| `--no-lookahead` | off | Disable lookahead in policy |
| `--sentinel PATH` | `best.pt` | Sentinel checkpoint |
| `--value-net PATH` | `data/value\_net.npz` | Value net |
| `--lr F` | auto | Learning rate |
| `--seed N` | 42 | Random seed |



## Puzzle Generators

### Opening / Placement Puzzles

Uses Malom DB path from `data/settings.json`.

```
.venv/bin/python tools/placement\_puzzle\_generator.py \\  
  --depth random --max-winning-moves 2 --side random
```

| Flag | Default | Description |
| - | - | - |
| `--side W\\|B\\|random` | random | Which side has the winning move |
| `--depth 0\\|4\\|5\\|6\\|7` | 0 | Target win depth in winner moves (0 = random) |
| `--max-winning-moves N` | 2 | Reject positions with more than N winning first moves |
| `--count N` | 0 | Puzzles to generate (0 = run forever) |
| `--attempts N` | 3000 | Positions sampled per puzzle attempt |
| `--out PATH` | `data/puzzles/` | Output directory |
| `--print` | off | Print each puzzle JSON to stdout |


### Midgame Puzzles (Malom DB)

```
.venv/bin/python tools/malom\_puzzle\_generator.py \\  
  --depth 6 --max-winning-moves 2 --side random \\  
  --min-pieces 4 --max-pieces 16
```

| Flag | Default | Description |
| - | - | - |
| `--side W\\|B\\|random` | random | Which side has the winning move |
| `--depth 0\\|4\\|5\\|6\\|7` | 0 | Target win depth (0 = random) |
| `--max-winning-moves N` | 2 | Reject positions with more than N winning first moves |
| `--min-pieces N` | 4 | Minimum pieces per side |
| `--max-pieces N` | 7 | Maximum pieces per side (raise for richer midgame) |
| `--count N` | 0 | Puzzles to generate (0 = run forever) |
| `--attempts N` | 3000 | Positions sampled per puzzle attempt |
| `--out PATH` | `data/puzzles/` | Output directory |
| `--print` | off | Print each puzzle JSON to stdout |


### Endgame Puzzles (Retrograde DB)

```
.venv/bin/python tools/puzzle\_generator.py \\  
  --depth random --max-winning-moves 2 --side random --random-db
```

| Flag | Default | Description |
| - | - | - |
| `--side W\\|B\\|random` | random | Which side has the winning move |
| `--depth 3\\|4\\|5\\|6\\|7\\|random` | random | Target win depth in winner moves |
| `--max-winning-moves N` | 2 | Reject positions with more than N winning first moves |
| `--db FILE\\|random` | random | Specific endgame .wdl file from `data/endgame/` |
| `--random-db` | off | Pick a new random DB file for every attempt (cross-table) |
| `--count N` | 0 | Puzzles to generate (0 = run forever) |
| `--attempts N` | 5000 | Positions sampled per puzzle attempt |
| `--out PATH` | `data/puzzles/` | Output directory |
| `--print` | off | Print each puzzle JSON to stdout |


