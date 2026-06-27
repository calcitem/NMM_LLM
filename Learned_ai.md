# Learned AI — Architecture & Training Plan

This document defines the complete architecture, reward structure, training pipeline, and
implementation strategy for the NMM learned AI system.  The history of failures is kept because
each attempt narrowed the diagnosis.

---

## Failure History (Summary)

| Attempt | Algorithm | Root cause of failure |
|-|-|-|
| REINFORCE v1–v3 | Terminal-only reward | 95% loss rate; every log_prob uniformly pushed down; collapse |
| A2C raw board | TD per-step | 84-float one-hot gives no strategic context; model can't discover NMM strategy from 10k games of losing |
| Scaffolded A2C s2b | RL on full game | Model learns difficulty-specific exploits; cannot transfer to the next level |

**The core insight from all failures:** A model learning NMM must start with meaningful signal — not
raw board pixels.  The scaffolded feature vector (sentinel + heuristic quality per move)
solves the cold-start problem.  The remaining challenge is **generalisation across difficulty levels**
and **phase-appropriate strategy**.

---

## Best Results — Specialist Training

### Opening Specialist (2026-06-23)

The Opening Specialist (`train_scaffolded_opening.py`) reached **difficulty 6 of 7** in 1,149 games,
with a perfect **100% win rate at difficulty 5** (recorded at advancement game 521).

| Difficulty | Best hwr | Advancement game |
|-|-|-|
| 1 | 0.480 | 52 |
| 2 | 0.375 | 226 |
| 3 | 0.389 | 338 |
| 4 | 0.440 | 521 |
| 5 | **1.000** | 690 |
| 6 | 0.220 (peak) | — (stalled ~12–14%) |

Difficulty 5 was beaten with a perfect sweep — every heuristic game in the rolling window was a win.
Difficulty 6 is proving substantially harder; the model plateaus around 12–14% win rate.
The opening specialist checkpoint (`s_open/best.pt`) is production-ready for the Overseer.

**Why it stalled at difficulty 6 — and what this reveals about midgame/endgame play:**

The opening specialist has **no midgame or endgame reward signal**.  All reward fires during
placement and the first 6 movement turns only.  After that, the only gradient is the retroactive
outcome score spread thinly across all plies.

Its wins at difficulties 1–5 were carried almost entirely by **opening position quality** — it
places pieces so well that the heuristic opponent at those levels makes enough midgame errors to
lose from a disadvantaged position.  The opening AI does not understand midgame tactics or
endgame conversion; it just arrives at a structurally strong position and benefits from opponent
mistakes.

At difficulty 6 the opponent is strong enough that a good opening is no longer sufficient —
real midgame and endgame play is required to convert the advantage.  The opening specialist has
none of that, so it stalls.  This is the expected and correct behaviour.

**Option B GameAI Handoff (2026-06-24):**

To score opening quality fairly without requiring the specialist to play midgame/endgame well,
after the placement phase + `OPENING_EXTENSION_PLY=6` movement turns, two `GameAI` instances
at the current training difficulty (`handoff_difficulty`) take over both sides and play the game
to completion.  No trajectory steps are recorded from the handoff point.  The game outcome then
reflects whether the opening produced a genuinely strong position, not whether the specialist
could convert it unassisted.

### Endgame Specialist (2026-06-24)

The Endgame Specialist (`train_scaffolded_endgame.py`) reached **difficulty 6** and stalled,
with best win rate 0.54.  Checkpoint `s_end/best.pt` is production-ready for endgame play.

| Difficulty | Best wr | Advancement game |
|-|-|-|
| 3 | 0.778 | start |
| 4 | 0.400 | 772 |
| 5 | 0.500 | 1013 |
| 6 | **0.54** (peak, stalled) | 1223 |

Advanced through difficulties 3–5 quickly (~100–120 games each).  At difficulty 6, the model
accumulated 694 training games and achieved a rolling peak of 0.54 win rate.  Could not reach
the 60% heuristic win threshold for difficulty 7 advancement.

---

## Architecture: Three Specialists + Overseer

Four separate networks are trained: one for each phase of the game, and an Overseer that
consults all three and makes the final decision.

```
                   ┌──────────────────────────────────────────┐
                   │             Overseer Net                   │
                   │   Input: 85 floats per move                │
                   │   [0:62)   62 scaffold base features       │
                   │   [62:77)  15 lookahead features (5-ply)   │
                   │   [77:80)  specialist probs (open/mid/end) │
                   │   [80:82)  GameAI alpha-beta (score, best) │
                   │   [82:85)  HumanDB (win_rate, freq, seen)  │
                   │                                            │
                   │   Reward: win/loss outcome (retroactive)   │
                   │         + specialist-filtered Malom win    │
                   │   Draws: 0 (neutral). No mill bonus.       │
                   └──────────────────────────────────────────┘
                          ↑          ↑          ↑
               ┌──────────┘   ┌──────┘   └──────────┐
               │              │                      │
    ┌──────────────┐ ┌──────────────┐     ┌──────────────┐
    │  Opening Net │ │  Midgame Net │     │  Endgame Net │
    │  77 floats   │ │  77 floats   │     │  77 floats   │
    │  Reward:     │ │  Reward:     │     │  Reward:     │
    │  sentinel +  │ │  sentinel +  │     │  Malom DTM   │
    │  heuristic   │ │  heuristic + │     │  + endgame   │
    │  advantage   │ │  Malom R/P   │     │  DB WDL      │
    └──────────────┘ └──────────────┘     └──────────────┘
```

All four networks use the same `ScaffoldedPolicyNet` architecture.
Specialists: `move_feat_dim=77`.  Overseer: `move_feat_dim=85`.

---

## Malom Database — Scope Clarification

**The Malom DB covers ALL positions in the game** — every legal position at every piece count
has a win/draw/loss label and a distance-to-mate (DTM) value.  It is not restricted to endgame.

This means Malom is the ground-truth training signal for all four specialists and the Overseer.
It is used as a reward/penalty signal (not as input features at inference) wherever it fires.

The user's own **retrograde endgame databases** (7v3, 4v3, 3v3, 4v4, etc.) are a separate
concept and cover endgame positions only.  These are distinct from Malom.

| Stage | Malom role |
|-|-|
| Opening specialist | No Malom reward (GAMMA=DELTA=0). Reward is sentinel + heuristic only. |
| Midgame specialist | Per-move reward (+0.15) when opponent enters losing state; penalty (−0.15) when opponent enters winning state |
| Endgame specialist | DTM-quality reward + trap bonus; plus EndgameSolvedDB WDL reward (+0.20 win / −0.10 loss) |
| Overseer | Specialist-filtered Malom win only: fires when the active phase specialist's top-1 also hits a Malom win. No loss penalty. No mill bonus. Main signal is game outcome. |

---

## Feature Vectors

### Specialist input (77 floats per candidate move)

**Base 62 floats** (indices 0–61):

| Slice | Source | Content |
|-|-|-|
| [0:20) | `board_context_features()` | Phase, piece counts, mobility, mills — same for all moves |
| [20:40) | `move_features()` | From/to/capture indices, mill flags, resulting state |
| [40:58) | `counterfactual_features()` | Heuristic rank, score; DB WDL/DTM slots (zero at inference) |
| [58] | Sentinel score | Quality score for this move [0,1] (0.5 when no sentinel loaded) |
| [59] | Blended absolute eval | 0.5×h_abs + 0.5×vn_abs, mapped [0,1] |
| [60] | is_top1 | 1.0 if heuristic #1 ranked move |
| [61] | Blended signed delta | tanh(0.5×h_delta + 0.5×vn_delta) |

Malom DB slots [40:58) are **zero at inference**.  Malom is reward-only for specialists.

**Lookahead 15 floats** (indices 62–76, `LookaheadAdvisor`, all 3 specialists + Overseer):

5 half-plies; each ply records 3 signals from the learner's perspective:

| Signal | Meaning |
|-|-|
| `h_norm` | `(evaluate(board, learner) + 1) / 2` — heuristic strength [0,1] |
| `vn_norm` | `(value_net.predict(board, learner) + 1) / 2` — value net [0,1] |
| `sent_mean` | Mean sentinel score for the side to move; flipped when opponent to move [0,1] |

Both sides play the static-heuristic-best move at plies 2–5.  The trajectory terminates early
if a terminal position or an exact Endgame-DB WDL is hit.
`use_sentinel=True` in all specialists and the Overseer — real sentinel scores in all plies.

### Overseer input (85 floats per candidate move)

The first 77 floats are identical to the specialist input.  Eight extra floats are appended:

| Index | Source | Content |
|-|-|-|
| [77] | Opening specialist | Softmax probability this move gets from Opening Net |
| [78] | Midgame specialist | Softmax probability this move gets from Midgame Net |
| [79] | Endgame specialist | Softmax probability this move gets from Endgame Net |
| [80] | GameAI score_norm | Alpha-beta search score for this move, normalised to [0,1] |
| [81] | GameAI is_best | 1.0 if this is the move GameAI's search selects, else 0.0 |
| [82] | HumanDB win_rate | Win rate when humans played this move (0.5 if not in DB) |
| [83] | HumanDB freq_norm | Relative frequency humans played this move (0 if not in DB) |
| [84] | HumanDB seen_flag | 1.0 if this position+move appears in the human game database |

**GameAI depth:** 3 during training, 5 at gameplay inference.  The GameAI instance runs
`score_root_moves(board, depth=D)` which performs a full alpha-beta search to depth D and
returns per-move normalised scores.  This gives the Overseer insight into what the classical
engine would do from the current position.

**HumanDB:** `data/human_db.sqlite` — 22,895 games, 642,703 positions.  Per-move win rates
and frequencies are looked up by board state key (with symmetry canonicalisation).

### Value head input (23 floats)

| Slice | Content |
|-|-|
| [0:20) | Board context features |
| [20] | `evaluate(board, player)` — absolute position strength |
| [21] | `max(sentinel_scores)` across legal moves |
| [22] | `mean(sentinel_scores)` across legal moves |

### Network structure (all four nets)

```
Policy: shared MLP applied independently to each move's feature row
  dim → 128 → 64 → 1  (scalar logit per move)
  softmax over k legal moves → policy distribution

Value: board-level MLP
  23 → 64 → 32 → tanh → 1  (scalar in [-1, 1])
```

Specialists: `move_feat_dim=77`, ~20K parameters.
Overseer: `move_feat_dim=85`, ~21K parameters.

---

## Reward Structures

### Opening Specialist rewards

Phase gate: placement phase only, plus OPENING_EXTENSION_PLY (6) moves into movement.

```
r_sentinel  = 0.20 × (sentinel_score_played − mean_sentinel_score)
r_heuristic = 0.15 × tanh(h_after − h_before)
r_mill      = 0.20 × mills_closed_this_move
r_retro     = 0.50 × outcome × 0.98^(plies_remaining)   [retroactive, game end]
```

No Malom reward (GAMMA=DELTA=0.0) — reward is purely sentinel + heuristic quality.
This was the reward structure that achieved difficulty 6 with 100% win rate at difficulty 5.

### Midgame Specialist rewards

Phase gate: movement phase only.
Rollouts start from real-game positions at movement turn 10 (±2) — 70% pool, 30% new_game().

```
r_sentinel   = 0.20 × (sentinel_score_played − mean_sentinel_score)
r_heuristic  = 0.15 × tanh(h_after − h_before)
r_mill       = 0.25 × mills_closed_this_move          [un-gated, fires always]
r_malom_win  = +0.15  if opponent enters Malom losing state
r_malom_loss = −0.15  if opponent enters Malom winning state
r_retro      =  0.50 × outcome × 0.98^(plies_remaining)
```

### Endgame Specialist rewards

Phase gate: total pieces < 12 or fly phase.
Rollouts start from real-game positions with < 12 pieces — 70% pool, 30% new_game().

```
r_malom_win  = 0.40 × dtm_quality(move)    if Malom says this move wins
               dtm_quality = 1 − dtm/100   (win-in-1 ≈ 0.99, win-in-50 ≈ 0.50)
r_malom_trap = 0.25                         if resulting opponent position is Malom "loss"
r_endgame_db = +0.20   if opponent is now in EndgameSolvedDB losing position
r_endgame_db = −0.10   if opponent is now in EndgameSolvedDB winning position
r_mill       = 0.15 × mills_closed
r_retro      = 0.50 × outcome × 0.98^(plies_remaining)
```

### Overseer rewards

Win-first design.  Main signal is the game outcome retroactively spread across all moves.
The only per-move bonus fires when the Malom database and the active phase specialist agree.

**Phase boundaries:**
- Opening: `board.phase == "place"` (placement turns 1–9 per side)
- Midgame: movement phase with ≥ 12 total pieces on board
- Endgame: movement phase with < 12 total pieces on board

```
# Game outcome (retroactive across all learner moves):
r_retro      =  0.60 × outcome × 0.98^(plies_remaining)
  outcome: win = +1.0,  loss = −1.0,  draw = 0.0

# Per-move bonus (sparse — fires only when specialist endorses Malom):
r_malom_win  = +0.40 × dtm_quality(move)
  fires when: chosen move is Malom "win" AND
              active phase specialist's top-1 is also Malom "win"

# Disabled (kept in code, may re-enable):
# r_malom_loss = −0.30 × dtm_quality(move)   if Malom losing trajectory
# r_mill       = +0.20 / +0.05 per mill formed
```

**Starting difficulty:** 1 (curriculum: 1 → 2 → … → 7, advance at 60% win rate each level).
**Draws:** neutral (0.0); no reward, no penalty.

---

## Lookahead Advisor (all four models)

The 5-ply lookahead is used by **all three specialists and the Overseer** — not just the Overseer.
`use_sentinel=True` everywhere: real sentinel scores computed at every ply.

### Algorithm

```
For each legal move m at the current position:
    board_1 = board.apply_move(m)                  # learner move (candidate being scored)
    board_2 = heuristic_best(board_1)              # opponent responds (static heuristic best)
    board_3 = heuristic_best(board_2)              # learner responds (static heuristic best)
    board_4 = heuristic_best(board_3)              # opponent responds
    board_5 = heuristic_best(board_4)              # learner responds

    At each ply depth d (1–5), record 3 signals from learner's perspective:
      h_norm[d]    = (evaluate(board_d, learner) + 1) / 2
      vn_norm[d]   = (value_net.predict(board_d, learner) + 1) / 2
      sent_mean[d] = mean sentinel score for side to move; flipped when opponent

    Early-terminate trajectory if:
      - Terminal position (winner known)
      - EndgameSolvedDB returns exact WDL (→ fill remaining plies with WDL value)
```

This produces a (k, 15) block appended to the (k, 62) base → (k, 77) total specialist input.

---

## Position Pool Sampling

Specialists need diverse starting positions from the phase they are training on.

### Midgame pool

`load_position_pool(root, phase="midgame", movement_turn=10, window=2)`

Walks every JSONL game file in `data/games`, `data/human_games`, `data/ai_games`.
For each game, finds the first movement-phase move, counts 10 more, and records boards
at movement turns 8–12 (turn 10 ± window=2).  Stops scanning that game once past the window.

70% of training games start from a pool position; 30% start from `new_game()`.
Learner colour is set to `board.turn` for pool starts.

### Endgame pool

`load_position_pool(root, phase="endgame", min_pieces=4, max_pieces=11)`

Reads every `board_fen_before` entry from the same JSONL directories.
Keeps positions where total pieces ∈ [4, 11].  Deduplicates by FEN.

70% of training games start from a pool position; 30% start from `new_game()`.

---

## Opening Book Enforcement

100% of opening specialist training games follow a randomly chosen line from the combined pool
(`book_openings.json` + `learned_openings.json`, ~120 lines).
The learner's first 4 placement moves are forced to match the line's positions for the learner's
colour.  If the matching position is not in the legal moves list, the learner plays freely from
that point.

---

## Anti-Overfitting: Difficulty Diversity

**1. Mixed-difficulty batches**

15% of heuristic games at each difficulty level are played against a randomly chosen
lower difficulty.  These games count toward `mixed_win_history` but not toward the
advancement check, so they cannot artificially inflate the advancement win rate.

**2. s1b Refresher on every difficulty advance**

Re-anchors the model to human-game positions before each new difficulty.
Prevents the model from specialising entirely on how the heuristic engine moves.

**3. Opponent time-budget variation**

Heuristic opponent time budget varied ±40% each game.

**4. Opening-line diversity**

120 opening lines from two sources; 50% of games follow a random line.

**5. Advancement threshold**

Wins ≥ 60% at each difficulty level (Overseer: pure win rate, draws neutral).
Opening/midgame specialists use the draw-inclusive path; Overseer does not.

---

## Training Pipeline

### Stage 1 — Imitation warmup

```bash
.venv/bin/python scripts/gen_imitation_data.py \
    --games 2000 --diff 3 \
    --sentinel learned_ai/sentinel/checkpoints/best.pt \
    --malom /mnt/windows/NMM_DB/Malom_Standard_Ultra-strong_1.1.0/Std_DD_89adjusted

.venv/bin/python scripts/train_scaffolded_s1.py \
    --data learned_ai/data/imitation_scaffolded.npz --epochs 10
# → learned_ai/checkpoints/scaffolded/s1/best.pt
```

### Stage 1b — Human-game fine-tuning

```bash
.venv/bin/python scripts/gen_human_imitation_data.py \
    --malom /mnt/windows/NMM_DB/Malom_Standard_Ultra-strong_1.1.0/Std_DD_89adjusted

.venv/bin/python scripts/train_scaffolded_s1b.py \
    --base-ckpt learned_ai/checkpoints/scaffolded/s1/best.pt --epochs 10 --lr 0.009
# → learned_ai/checkpoints/scaffolded/s1b/best.pt
```

### Stage 2 — Opening Specialist ✓ TRAINED

**Result:** Reached difficulty 6/7.  Perfect 100% win rate at difficulty 5.
Checkpoint `s_open/best.pt` is production-ready for opening play.
Training uses `use_sentinel=True` in the 5-ply lookahead, and Option B GameAI handoff
(after placement + 6 movement plies, two GameAI at current difficulty play out the rest).

```bash
# Resume training (auto-resumes from s_open/best.pt)
.venv/bin/python scripts/train_scaffolded_opening.py \
    --max-games 10000 --max-ply 140 \
    --malom /mnt/windows/NMM_DB/Malom_Standard_Ultra-strong_1.1.0/Std_DD_89adjusted
# → learned_ai/checkpoints/scaffolded/s_open/best{N}.pt, best.pt
```

To restart from a specific difficulty:
```bash
.venv/bin/python scripts/train_scaffolded_opening.py \
    --resume learned_ai/checkpoints/scaffolded/s_open/best1.pt \
    --diff-start 2 --max-games 10000 --max-ply 140
```

**Note:** `--malom` is passed for Malom feature encoding (DB probe flags in base features),
not for reward — opening rewards are sentinel + heuristic only (GAMMA=DELTA=0).

### Stage 3 — Midgame Specialist

Rewards: sentinel delta + heuristic delta + mill + Malom opponent-state reward/penalty.
Sentinel used as feature source (ALPHA=0.20 reward weight) AND in 5-ply lookahead.
Positions sampled from real games at movement turn 10 ±2.

```bash
.venv/bin/python scripts/train_scaffolded_midgame.py \
    --max-games 10000 --max-ply 140 \
    --malom /mnt/windows/NMM_DB/Malom_Standard_Ultra-strong_1.1.0/Std_DD_89adjusted
# → learned_ai/checkpoints/scaffolded/s_mid/best{N}.pt, best.pt
```

### Stage 4 — Endgame Specialist ✓ TRAINED

**Result:** Reached difficulty 6/7.  Best win rate 0.54 at difficulty 6.
Checkpoint `s_end/best.pt` is production-ready for endgame play.

Rewards: Malom DTM win quality + trap bonus + EndgameSolvedDB WDL.
Positions sampled from real games with < 12 pieces total.

```bash
# Resume training
.venv/bin/python scripts/train_scaffolded_endgame.py \
    --max-games 10000 --max-ply 140 \
    --malom /mnt/windows/NMM_DB/Malom_Standard_Ultra-strong_1.1.0/Std_DD_89adjusted
# → learned_ai/checkpoints/scaffolded/s_end/best{N}.pt, best.pt
```

### Stage 5 — Overseer

Requires all three specialist checkpoints to be complete first.
The Overseer uses 85-float features: 77 base+lookahead + 3 specialist probs + 2 GameAI + 3 HumanDB.
GameAI runs depth=7 during training, depth=5 at gameplay inference.

**Reward design (revised 2026-06-25):** Win-first.  Main signal is game outcome.
Specialist-filtered Malom win bonus: fires only when the active phase specialist's top-1
also agrees with Malom's winning trajectory.  No mill bonus.  No Malom loss penalty.
Draws score 0 (neutral).  Starts at difficulty 1 and advances at 60% win rate per level.

**Early result (2026-06-25, game 10):** diff 1 — hwr=0.333, awr=0.600, malom=74.4%.
Significantly stronger start than the previous diff-3 run (which showed hwr≈0.08 at game 10).

#### Serial training (single process)

```bash
# Fresh start (clears old logs, ignores previous s_over checkpoints)
.venv/bin/python scripts/train_scaffolded_overseer.py \
    --scratch \
    --opening-ckpt  learned_ai/checkpoints/scaffolded/s_open-retired/best.pt \
    --midgame-ckpt  learned_ai/checkpoints/scaffolded/s_mid/best.pt \
    --endgame-ckpt  learned_ai/checkpoints/scaffolded/s_end/best.pt \
    --malom /mnt/windows/NMM_DB/Malom_Standard_Ultra-strong_1.1.0/Std_DD_89adjusted \
    --max-games 10000 --max-ply 140
# Optional: --human-db data/human_db.sqlite (default)
# → learned_ai/checkpoints/scaffolded/s_over/best.pt

# Resume training (auto-resumes from s_over/best.pt)
.venv/bin/python scripts/train_scaffolded_overseer.py \
    --auto-resume-best \
    --opening-ckpt  learned_ai/checkpoints/scaffolded/s_open-retired/best.pt \
    --midgame-ckpt  learned_ai/checkpoints/scaffolded/s_mid/best.pt \
    --endgame-ckpt  learned_ai/checkpoints/scaffolded/s_end/best.pt \
    --malom /mnt/windows/NMM_DB/Malom_Standard_Ultra-strong_1.1.0/Std_DD_89adjusted \
    --max-games 10000 --max-ply 140
```

#### Parallel training (multi-process, recommended)

`train_scaffolded_overseer_parallel.py` runs rollouts across N worker processes via
`ProcessPoolExecutor`, then trains in the main process on the collected trajectories.
Workers each load all three specialists and run full 85-float overseer games independently;
the main process batches the results and calls the A2C/PPO update.

GameAI depth defaults to 7 (vs 3 in the serial version) — workers run on CPU so the deeper
search doesn't block training.

**Key difference from serial:** state dict is serialised as numpy arrays when queued to workers
(avoids PyTorch's fd-sharing socket race under high worker counts).

```bash
# Fresh start — parallel, 8 workers
.venv/bin/python scripts/train_scaffolded_overseer_parallel.py \
    --scratch \
    --opening-ckpt  learned_ai/checkpoints/scaffolded/s_open-retired/best.pt \
    --midgame-ckpt  learned_ai/checkpoints/scaffolded/s_mid/best.pt \
    --endgame-ckpt  learned_ai/checkpoints/scaffolded/s_end/best.pt \
    --malom /mnt/windows/NMM_DB/Malom_Standard_Ultra-strong_1.1.0/Std_DD_89adjusted \
    --max-games 10000 --max-ply 140 \
    --workers 8
# → learned_ai/checkpoints/scaffolded/s_over/best.pt  (same output dir as serial)

# Resume training
.venv/bin/python scripts/train_scaffolded_overseer_parallel.py \
    --auto-resume-best \
    --opening-ckpt  learned_ai/checkpoints/scaffolded/s_open-retired/best.pt \
    --midgame-ckpt  learned_ai/checkpoints/scaffolded/s_mid/best.pt \
    --endgame-ckpt  learned_ai/checkpoints/scaffolded/s_end/best.pt \
    --malom /mnt/windows/NMM_DB/Malom_Standard_Ultra-strong_1.1.0/Std_DD_89adjusted \
    --max-games 10000 --max-ply 140 \
    --workers 8
```

**Notable parallel-only flags:**

| Flag | Default | Effect |
|-|-|-|
| `--workers N` | 4 | Number of worker processes for parallel rollouts |
| `--gameai-depth D` | 7 | Alpha-beta depth used inside workers (serial default is 3) |
| `--s1b-data PATH` | `learned_ai/data/human_imitation.npz` | Data for s1b refresher run before workers spawn |
| `--s1b-refresher-epochs N` | 3/10 | Epochs for loser/winner refresher passes |
| `--max-branches-per-game N` | 0 (disabled) | Branch rollouts per game for extra data |
| `--branch-every N` | — | How often to sample a branch game |

---

## Advancement Criteria

| Specialist | Advance condition | Window |
|-|-|-|
| Opening | `(wr ≥ 30% AND dr ≥ 30%) OR wr ≥ 50%` | 50-game rolling, full-difficulty heuristic games only |
| Midgame | `(wr ≥ 30% AND dr ≥ 30%) OR wr ≥ 50%` | 50-game rolling, full-difficulty heuristic games only |
| Endgame | `(wr ≥ 30% AND dr ≥ 30%) OR wr ≥ 60%` | 50-game rolling, full-difficulty heuristic games only |
| Overseer | `wr ≥ 60%` (wins only — draws score 0, not 0.5) | 50-game rolling, full-difficulty heuristic games only |

**Checkpoint safety:** `best{N}.pt` is saved both at every periodic log interval AND at the
moment of advancement (if not already saved).  This prevents the case where the deque fills
to exactly 50 entries between log checkpoints and advancement fires without saving the weights.

**Exit criterion:** same threshold as advancement, evaluated vs the maximum difficulty level.

---

## Recovery Mechanism

If rolling win rate drops below 12% for 30+ consecutive games:
- Reload `best{N}.pt` for the current difficulty.
- Reset optimizer (fresh Adam at `lr_base`).
- Reset temperature to `TEMP_START` (0.50).
- Clear win history.

---

## LR Scaling

```
scale  = clamp(win_rate / 0.35, 0.5, 2.0)
new_lr = lr_base × scale
```

---

## Integration with Existing Engine (Wiring into Gameplay)

At deployment the Coordinator selects the agent mode via `ai/coordinator.py`.
The `scaffolded` mode uses the Overseer checkpoint (`s_over/best.pt`) as the primary decision maker.

### Inference path per learner turn

1. `encode_position_with_lookahead(board, player, sentinel_advisor, db, value_net, lookahead_advisor)`
   → 77-float feature matrix (k, 77) for k legal moves.

2. **Specialists only (77-float path):** Forward through the relevant specialist net → logits → choose move.

3. **Overseer (85-float path):** Call `build_overseer_extras(base_77, board, enc, player, spec_open, spec_mid, spec_end, gameai, human_db, gameai_depth=5)` to extend to (k, 85), then forward through OverseerNet.

### ScaffoldedAgent wiring

`ScaffoldedAgent` handles both specialists and overseer at inference:

```python
from learned_ai.agents.scaffolded_agent import ScaffoldedAgent
from ai.game_ai import GameAI
from ai.human_db import HumanDB

# Specialist agent (77-float):
agent = ScaffoldedAgent(
    color="W",
    checkpoint_path="learned_ai/checkpoints/scaffolded/s_open/best.pt",
    sentinel_advisor=sentinel,
    value_net=value_net,
)

# Overseer agent (85-float):
gameai = GameAI(color="W", difficulty=5)   # depth=5 at inference
human_db = HumanDB("data/human_db.sqlite")
spec_open = _load_specialist("...s_open/best.pt")
spec_mid  = _load_specialist("...s_mid/best.pt")
spec_end  = _load_specialist("...s_end/best.pt")
overseer_agent = ScaffoldedAgent(
    color="W",
    checkpoint_path="learned_ai/checkpoints/scaffolded/s_over/best.pt",
    sentinel_advisor=sentinel,
    value_net=value_net,
    is_overseer=True,
    spec_open=spec_open, spec_mid=spec_mid, spec_end=spec_end,
    gameai=gameai, human_db=human_db, gameai_depth=5,
)
# ScaffoldedAgent.choose_move() returns a move dict — drop-in replacement for GameAI.choose_move()
```

The `is_overseer=True` flag tells `ScaffoldedAgent` to call `build_overseer_extras` before the
forward pass, extending the 77-float base to 85 floats.

### Coordinator modes

- `heuristic` — existing GameAI engine
- `scaffolded` — Overseer checkpoint (`s_over/best.pt`) via ScaffoldedAgent(is_overseer=True)
- `overseer_overlay` — heuristic picks the move; Overseer probabilities shown as "O:XX%"

---

## Files

| File | Purpose |
|-|-|
| `learned_ai/models/scaffolded_encoder.py` | `encode_position()` + `encode_position_with_lookahead()` — builds (k,62) or (k,77) feat matrix |
| `learned_ai/models/scaffolded_net.py` | `ScaffoldedPolicyNet` — policy (dim→128→64→1) + value head |
| `learned_ai/models/lookahead_advisor.py` | `LookaheadAdvisor` — 5-ply forward simulation, all models |
| `learned_ai/models/overseer_extras.py` | `build_overseer_extras()` — 77→85 float extension for Overseer |
| `learned_ai/training/scaffolded_a2c.py` | `ScaffoldedStep`, `scaffolded_a2c_update()`, `scaffolded_ppo_update()` |
| `learned_ai/training/position_pool.py` | `load_position_pool()` — midgame turn-10 and endgame <12-piece pools |
| `learned_ai/agents/scaffolded_agent.py` | `ScaffoldedAgent` — inference wrapper (specialists + overseer) |
| `ai/game_ai.py` | `GameAI` — classical engine; `score_root_moves(board, depth)` for Overseer features |
| `ai/human_db.py` | `HumanDB` — 22,895 human game SQLite; `query_moves()` for Overseer features |
| `scripts/gen_imitation_data.py` | Stage 1 supervised dataset (heuristic self-play, Malom soft labels) |
| `scripts/train_scaffolded_s1.py` | Stage 1: imitation training |
| `scripts/gen_human_imitation_data.py` | Stage 1b supervised dataset (human games) |
| `scripts/train_scaffolded_s1b.py` | Stage 1b: human-game fine-tune (policy head only) |
| `scripts/train_scaffolded_opening.py` | Stage 2: Opening Specialist |
| `scripts/train_scaffolded_midgame.py` | Stage 3: Midgame Specialist |
| `scripts/train_scaffolded_endgame.py` | Stage 4: Endgame Specialist |
| `scripts/train_scaffolded_overseer.py` | Stage 5: Overseer (serial, single process) |
| `scripts/train_scaffolded_overseer_parallel.py` | Stage 5: Overseer (parallel, N workers via ProcessPoolExecutor) |
| `scripts/bench_scaffolded.py` | Headless benchmark vs heuristic configs |
| `tests/test_scaffolded_policy.py` | 25 unit tests (encoder, net, A2C, agent) |

---

## Success Metrics

| Stage | Target | Status |
|-|-|-|
| Stage 1 | Policy loss decreasing; plays better than random | ✓ |
| Stage 1b | Human-deviated positions weighted; policy loss < Stage 1 | ✓ |
| Stage 2 (Opening) | Reaches diff 7 with wins ≥ 30% + draws ≥ 30% | Reached diff 6; 100% wr at diff 5 ✓; retraining with sentinel lookahead |
| Stage 3 (Midgame) | Reaches diff 7; Malom trap rate > 20% in midgame | Pending |
| Stage 4 (Endgame) | Reaches diff 7; Malom move-match rate > 60% | Reached diff 6; best wr 0.54 ✓; s_end/best.pt production-ready |
| Stage 5 (Overseer) | Reaches diff 7; wins ≥ 60% vs diff 7 | In progress — diff 1; hwr=0.333 at game 10 (2026-06-25) |
