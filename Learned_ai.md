# Learned AI — Training Plan

This document tracks every attempt to train a learned NMM agent, why each one failed,
and the current plan.  The history matters: each failure narrowed the diagnosis.

---

## Failure History

### Attempt 1–3: REINFORCE (v1/v2/v3)

Three REINFORCE attempts all ended in **policy collapse** within 500 games.

| Version | What changed | Killed at | Final win rate | Root cause |
|---------|--------------|-----------|----------------|------------|
| v1 | baseline REINFORCE | game 4372 | 7.5% | sentinel over-filtered, value collapse, T=1.0 |
| v2 | + Malom move quality reward | game ~200 | 2.3% | too early to judge; restarted |
| v3 | + Malom trap reward, T=0.2→0.6, reward=1.0 | game 500 | 2.0% (declining) | three compounded bugs |

**Root cause of all three:** Terminal-only reward in a 40-ply game means one gradient
signal per game.  With 95% loss rate, every log_prob in every game got pushed down
uniformly.  The model collapsed to near-random before it could discover any winning moves.

---

### Attempt 4: A2C with raw board state

Switched from REINFORCE to **A2C** (actor-critic, per-step TD bootstrapping) to solve the
variance problem.  Three concrete bug fixes: win_reward 2.0→1.0, lr 1e-4→5e-6, temperature
0.5→0.2 annealed to 0.6.

**Model:** `NMMNet` — 84-float raw board one-hot → MLP backbone (256→256→128) → 5 phase
heads → 624 fixed action logits.

**Result after 10,000 games vs heuristic difficulty 2:** **2% win rate peak, 2% final.**
Many games hit the 200-step draw cap.  Loss never meaningfully decreased.

**Why this also failed — the real root cause, finally identified:**

The 84-float raw board encoding gives the model no strategic context.  It receives 24
positions × 3-way one-hot, side-to-move, phase, and piece counts — nothing about
which moves are good, which pieces are in danger, what the heuristic thinks, or
what Malom says.  To win from this starting point, the model would have to independently
rediscover all of NMM strategy through trial and error against a competent heuristic.

After 10,000 games it had still not managed this.  The signal-to-noise ratio is simply
too low: the heuristic consistently outplays a confused random-ish policy, so almost
every game is a loss, and the per-move TD signal only helps if the model can already
produce occasionally good moves to learn from.

This is not an algorithm problem.  A2C is correct.  The problem is the **input
representation**.

---

## The Real Problem: No Scaffold

The pattern across all four attempts is the same:

> A model learning NMM from raw board encodings must discover the entire strategy of the
> game before it can produce a single positive gradient signal in a game against a
> competent opponent.  It can't do this from 10,000 games of repeated losing.

The sentinel does not have this problem.  It was trained with 58 rich per-move features:
resulting piece counts, mobility, mill threats, heuristic rank, DB win rates, DTM quality.
It knew, from day one, that closing a mill is good and walking into a trap is bad.

The learned policy needs the same scaffold.

---

## New Approach: Scaffolded Meta-Policy

**Core idea:** Instead of learning NMM from scratch, teach the model to learn *when to
agree or disagree with the experts* (sentinel, heuristic engine, Malom DB).

The model is not given the raw board.  It is given, for every legal move:
- What the sentinel thinks of this move (quality score)
- What the heuristic engine thinks (rank, absolute evaluation, delta from current position)
- What Malom says (WDL, distance-to-win)
- The structural move properties (mill closing, piece counts, mobility, etc.)

The task becomes: "given everything the experts know about this move, should I play it?"
This is learnable in far fewer games because the model starts with meaningful signal.

It is fine if the model remains dependent on the sentinel and heuristic at deployment — it
has them available anyway.  The goal is to *improve on top of that baseline*, not to
replace it.

---

## Architecture: ScaffoldedPolicyNet

### Per-move input (62 floats)

| Slice | Source | Content |
|-------|--------|---------|
| [0:20) | `feature_builder.board_context_features()` | Phase, piece counts, mobility, mills — same for all moves in this position |
| [20:40) | `feature_builder.move_features()` | From/to/capture indices, mill flags, resulting state |
| [40:58) | `feature_builder.counterfactual_features()` | Heuristic rank, normalised score, **DB win/loss fracs, DTM quality** — the DB slots are **zero at inference and in Stages 1 & 2**; first populated in Stage 3 when `encode_position(db=db)` is called |
| [58] | `SentinelAdvisor.advise().move_scores[i]` | Sentinel's quality score for this move, [0,1] |
| [59] | `0.5 * h_abs_norm + 0.5 * vn_abs_norm` | Blended absolute eval of resulting position (heuristic + value-net, each mapped [0,1]) |
| [60] | `is_top1` | 1.0 if this is the heuristic engine's #1 ranked move |
| [61] | `tanh(0.5 * h_delta + 0.5 * vn_delta)` | Blended signed improvement (heuristic + value-net delta, then tanh) |

> **Option B blend** — features 59 and 61 blend the value net (VN_BLEND=0.5) into
> the existing heuristic slots to preserve the 62-dim checkpoint format.  When
> `value_net=None`, they fall back to pure heuristic values (backward-compatible
> with `s1b/best.pt` at inference).
>
> **Future — Option A** (implement when next checkpoint is trained from scratch):
> Extend to 64 floats — add `vn_score_abs` as feature [62] and `vn_delta_tanh` as
> feature [63], keeping heuristic features [59] and [61] pure.  This gives the model
> a clean, separable view of both evaluators.  Changes required:
> - `scaffolded_encoder.py`: `MOVE_FEAT_DIM = 64`; add vn features as independent
>   entries instead of blending; remove `VN_BLEND`
> - `scaffolded_net.py`: no change (reads `move_feat_dim` from config)
> - All training/gen scripts: no change (they already pass `value_net`)
> - Old 62-dim checkpoints will be incompatible — start a fresh Stage 1 run

### Value head input (23 floats)

| Slice | Content |
|-------|---------|
| [0:20) | Board context features (same as above) |
| [20] | `evaluate(board, player, strength_mode=True)` — absolute position strength |
| [21] | `max(sentinel_scores)` across legal moves |
| [22] | `mean(sentinel_scores)` across legal moves |

### Network structure

```
Policy: shared MLP applied independently to each move's 62-float row
  62 → 128 → 64 → 1  (scalar logit per move)
  softmax over k legal moves → policy distribution

Value: board-level MLP
  23 → 64 → 32 → tanh → 1  (scalar in [-1, 1])
```

Variable move count handled naturally: k varies by position, no padding or masking needed.

**Parameter count:** ~20K total (policy 16K + value 4K).  Deliberately small — the heavy
lifting is done by the feature computation, not the network.

Files:
- `learned_ai/models/scaffolded_encoder.py` — `encode_position()` → `EncodedPosition`
- `learned_ai/models/scaffolded_net.py` — `ScaffoldedPolicyNet`
- `learned_ai/agents/scaffolded_agent.py` — `ScaffoldedAgent` (inference)
- `learned_ai/training/scaffolded_a2c.py` — `ScaffoldedStep` + `scaffolded_a2c_update()` + PPO variant

---

## Malom: Training Signal vs Feature Visibility

Malom (the ultra-strong external solver) plays two separate roles in the pipeline:

| Role | Stage 1 | Stage 2 | Stage 3 | Inference |
|------|---------|---------|---------|-----------|
| **Soft label targets** (KL loss) | ✓ queried separately | — | — | — |
| **Per-move reward shaping** | — | ✓ queried separately | ✓ queried separately | — |
| **Malom slots in feature vector [40:58)** | **zero** | **zero** | **non-zero** | **zero** |

The invariant is: **the model never sees Malom WDL/DTM in its input features at inference**.
Stages 1 and 2 call `encode_position(db=None)`, which zeros the DB slots in [40:58).
Malom data is queried independently by the training scripts and used only to:
- Shape the soft label distribution (Stage 1)
- Compute `r_malom_win` and `r_malom_trap` rewards (Stages 2 & 3)

Stage 3 is the only exception: it calls `encode_position(db=db)`, so the model begins
to see Malom WDL/DTM as live input features — simultaneously with the supervised KL loss.
At inference, `db=None` is always passed (Malom unavailable), so the model generalises
using what it learned from heuristics, value net, sentinel, and board structure.

---

## Reward Structure

### Per-move shaped rewards (dense — every learner turn)

```
r_sentinel   = 0.15 × (sentinel_score_played − mean_sentinel_score)
               Did we play above the average quality sentinel sees for this position?

r_heuristic  = 0.10 × tanh(h_after − h_before)
               Did the heuristic evaluation improve after our move?

r_value_net  = 0.10 × tanh(vn_after − vn_before)
               Did the value-net evaluation improve after our move?
               (fires only when --value-net is provided; zero otherwise)

r_malom_win  = 0.25 × dtm_quality(move)    if Malom says this move wins
               dtm_quality = 1 − dtm/100, so win-in-1 ≈ 0.99, win-in-50 ≈ 0.50
               Winning moves rewarded more for faster wins.

r_malom_trap = 0.15                        if resulting opponent position is Malom "loss"
               Opponent is now provably losing — clearest possible signal.
```

These fire every turn, giving dense gradient signal regardless of whether the game is won
or lost.  The model learns to improve positions incrementally.

### Game-level retroactive rescoring (after game ends)

```
outcome = +1.0 (win) | −1.0 (loss) | +0.15 (draw < 100 plies) | −0.05 (draw ≥ 100 plies)

for each move t in trajectory (from end):
    r_t += 0.50 × outcome × 0.98^(plies_remaining)
```

This retroactively credits early moves in a winning game and penalises early moves in
a losing game, decayed so recent moves get more credit.  It prevents the per-move
signals from drowning out the ultimate game result.

### Why this is better than the previous reward structure

| Old (A2C, raw board) | New (Scaffolded) |
|----------------------|------------------|
| r_malom = 0.1 × Δ(WDL) — small, infrequent (endgame only) | r_sentinel fires every move, calibrated to quality delta |
| Terminal outcome dominates | Per-move + retroactive are balanced |
| Model needs to form strategic understanding to get any reward | Model gets reward for playing sentinel/heuristic-aligned moves from game 1 |

---

## Training Pipeline

### Stage 1 — Imitation warmup

**Goal:** Give the model a solid starting policy before RL begins.  Without this, the
model is near-random and the per-move rewards are noisy.

**Method:** Play heuristic (diff 3) vs heuristic self-play with book-guided White
openings and `top_n=2` move diversity.  At each position, `encode_position(db=None)` is
called — **Malom slots [40:58) are zero in the feature matrix**.  The Malom DB is then
queried *separately* to compute a **soft label distribution** over all legal moves
(DTM-graded per move):

```
weight[move] = _WDL_SCALE[wdl] × dtm_quality(wdl, dtm)
    where _WDL_SCALE = {"win": 1.0, "draw": 0.4, "loss": 0.1}
    and   dtm_quality = 1 − dtm/100   (win-in-1 ≈ 0.99, win-in-50 ≈ 0.50)
```

When the Malom DB has no entry for a move, the sentinel score is used instead.
The resulting (k,) distribution is normalised to sum to 1.

Training loss: **cross-entropy with soft labels** (equivalent to KL divergence) for
policy, **MSE against heuristic h_eval** for value.  This teaches the model to prefer
Malom-winning moves proportional to how quickly they win — but from features that match
inference time (no Malom in the vector).

**Balance fix (2026-06-21):** Without intervention, heuristic self-play with a tight
time budget produced heavily Black-biased outcomes (0/76/24 W/B/D over 100 test games).
Root cause: B-47 intentional asymmetry in `tactical_move_bonus` rates Black's early
placements higher, compounded by NMM second-player advantage at shallow search depth.
Fixes applied in `gen_imitation_data.py`:
- White's 1st and 2nd placements are forced to a randomly selected book opening line
  (`data/openings/book_openings.json`), giving White sound structural starts.
- `top_n=2` for all non-book moves — picks randomly from the top-2 scored moves,
  breaking deterministic Black-wins-every-game spirals.

**Expected outcome:** Model learns to prefer Malom-winning moves from day one.  Confirmed
by checking that policy loss (cross-entropy with soft label) decreases each epoch.

**Commands:**
```bash
# Generate dataset (~10h for 2000 games at diff 3)
.venv/bin/python scripts/gen_imitation_data.py \
    --games 2000 --diff 3 \
    --sentinel learned_ai/sentinel/checkpoints/best.pt \
    --malom /mnt/windows/NMM_DB/Malom_Standard_Ultra-strong_1.1.0/Std_DD_89adjusted

# Train imitation model (~10 min)
.venv/bin/python scripts/train_scaffolded_s1.py \
    --data learned_ai/data/imitation_scaffolded.npz \
    --epochs 20

# Checkpoint → learned_ai/checkpoints/scaffolded/s1/best.pt
```

---

### Stage 1.5 — Human-game fine-tuning

**Goal:** Teach the policy to replicate patterns from human games where the human won,
giving it knowledge the heuristic doesn't have — human tactical intuition vs AI at
difficulty 3.

**What it uses:** All `data/games/*.jsonl` game records where `human_color` is set and
the human won or drew.  For each human move, the board is reconstructed from the
`board_fen_before` field, encoded with `encode_position()`, and the human's chosen move
is recorded as the target.

**Weighting:**
- Won game moves: weight = 1.0
- Draw game moves: weight = 0.3
- Lost game moves: skipped entirely
- Positions where the human deviated from the heuristic's top-1 pick (in won games):
  extra 1.5× bonus weight — these carry the most human-specific signal

**Numbers (2026-06-19, 451 human games):**
- 5,014 positions total (4,219 from won games, 795 from draws)
- Human deviated from heuristic top-1 in 3,658 / 4,219 won-game positions (87%)

**Method:** Fine-tune policy head only from `s1/best.pt`.  Value head is frozen to
preserve Stage 1's position evaluation.  Weighted cross-entropy loss, LR=0.009, 10 epochs.

**Commands:**
```bash
# Extract human game data (run once, ~45s — re-run after new games are played)
# Can run in parallel with gen_imitation_data.py once that has started
.venv/bin/python scripts/gen_human_imitation_data.py \
    --malom /mnt/windows/NMM_DB/Malom_Standard_Ultra-strong_1.1.0/Std_DD_89adjusted
# Output → learned_ai/data/human_imitation.npz

# Fine-tune from Stage 1 checkpoint (~1 min)
.venv/bin/python scripts/train_scaffolded_s1b.py \
    --base-ckpt learned_ai/checkpoints/scaffolded/s1/best.pt \
    --epochs 10 --lr 0.009
# Checkpoint → learned_ai/checkpoints/scaffolded/s1b/best.pt
```

Stage 2 should resume from `s1b/best.pt` instead of `s1/best.pt`.

---

### Stage 2 — A2C self-play with full scaffolded rewards

**Goal:** Learn to actually win, using dense per-move rewards + retroactive rescoring.

**Opponent:** Heuristic engine, difficulty 2 → 3.
**Algorithm:** A2C (or `--ppo` for PPO).
**Temperature:** 0.5 annealed to 0.9 (model starts near imitation prior; explore more as training progresses).
**LR:** 1e-4 (model is new, not fine-tuning a pre-trained checkpoint — we can use a higher rate).

**Malom separation:** `encode_position(db=None)` is called on every position — **Malom
slots [40:58) remain zero**, matching inference time.  Malom is queried separately:
`db.query_all_moves()` drives `r_malom_win`; `db.query_state()` drives `r_malom_trap`.
The model learns what winning moves look like from rewards, not from seeing Malom answers
in its inputs.

**Curriculum:**
- Start vs diff 2.
- Advance to diff 3 when rolling-200 win rate ≥ 60%.
- Exit when rolling-200 win rate ≥ 60% at diff 3.

**Why higher LR than old Stage 2:** Old Stage 2 used 5e-6 to protect a pre-trained
checkpoint.  ScaffoldedNet starts from Stage 1 imitation, not a pre-trained NMMNet
backbone — a fresh model that needs to learn.  1e-4 is appropriate.

**Current script:** `train_scaffolded_s2_diagnostic.py` — same algorithm as `train_scaffolded_s2.py`
but with richer JSONL logging (per-component reward breakdown, chosen-move probability,
policy sharpness, Malom hit rate) and self-adjusting temperature/LR-backoff.

**Commands:**
```bash
# Standard run
.venv/bin/python scripts/train_scaffolded_s2_diagnostic.py --max-games 10000

# Resumes automatically from s1b/best.pt (falls back to s1/best.pt if absent)
# Checkpoint → learned_ai/checkpoints/scaffolded/s2/best.pt
# Logs → learned_ai/checkpoints/scaffolded/s2/train_log.jsonl
#         learned_ai/checkpoints/scaffolded/s2/update_log.jsonl

# PPO variant
.venv/bin/python scripts/train_scaffolded_s2_diagnostic.py --ppo --max-games 10000
```

**Training performance (Stage 2 — three runs, 128 log entries total):**

Three separate runs are recorded in `s2/train_log.jsonl`:

| Run | Games | Difficulty | Peak win-200 | Outcome |
|-----|-------|------------|-------------|---------|
| Run 1 | 50–300 | 2 | 0.22 | Abandoned early |
| Run 2 | 50–5800 | 2 | 0.37 (~game 2500) | Stagnated; never advanced |
| Run 3 | 2850–3100 | 2 → **3** | 0.455 → advanced | Beat diff 3 at 91% |

Key observations:
- **Run 2 stagnation:** Win rate peaked at ~0.37 around game 2500 then declined as
  temperature annealed past 0.90 toward 1.0, making play increasingly stochastic.  By
  game 5800 the win rate had dropped to ~0.14.  The 60% advance threshold was never
  crossed; this run was abandoned.
- **Run 3 breakthrough:** Resuming from the best checkpoint, the model reached 0.455 at
  game 3000 — crossing the internal threshold — and advanced to difficulty 3 at game 3050.
  The first difficulty-3 reading showed 100% (window artefact: only a handful of games
  played), settling at 91% by game 3100.
- **Difficulty 3 already mastered:** 91% win rate immediately on arrival at diff 3
  confirms the model had already internalised difficulty-3 strategy during diff-2 training.
  This is why Stage 2b starts directly at difficulty 4.

---

### Stage 2b — Self-play with branched mid-game rollouts

**Goal:** Broaden trajectory diversity without the training/inference gap of an undo
mechanism.  Two additions on top of Stage 2:

**Self-play (50% of main games):** The live model (temperature-sampled) plays against a
periodically frozen copy of itself (refreshed every 50 games, `argmax` mode).  The other
50% use the heuristic opponent from Stage 2.  Mixing prevents the model from only learning
to beat itself.

**Branched rollouts:** Every 10 learner turns, the current board state is snapshotted.
After the main game ends, up to 2 of those snapshots are used as starting points for
fresh independent rollouts (model vs frozen copy).  Each branch is stored as a completely
separate trajectory — it never shares a gradient-update batch with the game it was spawned
from, so there is **no gradient contamination for shared positions**.

**Game-stage diversity — phase buckets:** Branch points are classified as:
- `opening` — placement phase, < 10 pieces placed total
- `midgame` — late placement or early movement (10+ placed, ≥ 12 on board)
- `endgame` — movement phase with < 12 pieces on board

A rolling counter (300-game window) caps how many branches can come from any single bucket
(`MAX_PER_BUCKET = 80`).  Once a bucket saturates, new branches from that phase are
skipped.  This ensures the training set always spans beginning, middle, and end-game play.

**Why not the "rewind" approach:** Rewinding and replaying within the same trajectory
causes the same board positions to receive contradictory gradient signals (credited from
path A in one update, penalised from path B in another).  It also trains a skill (undoing
moves) that doesn't exist at inference.  Independent branches avoid both problems.

**Curriculum (five-level):**
- Start at difficulty 4 (`DIFF_START=4`; the model already beats diff 3 at 100%).
- Advance: diff 4 → 5 at ≥ 70% rolling-200 win rate; 5 → 6 at ≥ 65%; 6 → 7 at ≥ 60%.
- Exit: ≥ 70% win rate vs difficulty 7 (`EXIT_THRESHOLD=0.70`, `DIFF_MAX=7`).
- When a difficulty threshold is crossed, `win_history` is cleared so progress against the
  new opponent is measured fresh.

**Commands:**
```bash
# From s1b (default — no s2 checkpoint yet)
.venv/bin/python scripts/train_scaffolded_s2b.py --max-games 5000

# From s2/best.pt
.venv/bin/python scripts/train_scaffolded_s2b.py --auto-resume-s2 --max-games 5000

# Resume a previous s2b run
.venv/bin/python scripts/train_scaffolded_s2b.py --auto-resume-best --max-games 5000

# PPO variant
.venv/bin/python scripts/train_scaffolded_s2b.py --auto-resume-best --ppo --max-games 5000

# Disable branching — pure self-play only (to isolate effects)
.venv/bin/python scripts/train_scaffolded_s2b.py --auto-resume-best --max-branches-per-game 0

# More aggressive branching (3 per game, every 8 moves)
.venv/bin/python scripts/train_scaffolded_s2b.py --auto-resume-best --max-branches-per-game 3 --branch-every 8

# Checkpoint → learned_ai/checkpoints/scaffolded/s2b/best.pt
# Logs → learned_ai/checkpoints/scaffolded/s2b/train_log.jsonl
#         learned_ai/checkpoints/scaffolded/s2b/update_log.jsonl
```

**What to watch in the logs (`train_log.jsonl`):**
- `bucket_opening / bucket_midgame / bucket_endgame` — all should be non-zero; if one
  saturates, branches from that phase auto-suppress
- `game_type: "branch"` entries with varying `phase_bucket` — confirms game-stage coverage
- Win rate trend vs Stage 2 — should be equal or better; if notably worse, reduce
  `--max-branches-per-game` to 1

---

### Stage 3 — Malom supervised fine-tuning

**Goal:** Sharpen strategy with explicit DB supervision.  When Malom knows the WDL for
all legal moves, push the policy toward the winning move(s), weighted by DTM quality.

**This is the first and only stage where the model sees Malom WDL/DTM in its input
features.**  `encode_position(db=db)` is called, populating the DB slots in [40:58).
The model can now condition its policy directly on Malom information, and the SL loss
explicitly pushes weights toward Malom-preferred move distributions.

**Loss:** `total = 0.6 × A2C_loss + 0.4 × KL(policy → malom_target)`

The Malom target is a probability distribution over legal moves:
- Win moves: weight = dtm_quality (faster wins get more weight)
- Draw moves: weight = 0.15
- Loss moves: weight = 0.0

The SL loss fires only when Malom has entries — in endgame and tractable midgame positions.
It is zero in opening positions where the DB is unavailable.

**Opponent:** Heuristic diff 3, advancing to diff 4.
**LR:** 3e-5 (fine-tuning, lower than Stage 2).

**Exit criterion:** Rolling-200 win rate ≥ 40% vs diff 4.

**Commands:**
```bash
# Standard run — auto-resumes from s2b/best.pt (falls back to s2/best.pt if absent)
.venv/bin/python scripts/train_scaffolded_s3.py \
    --malom /mnt/windows/NMM_DB/Malom_Standard_Ultra-strong_1.1.0/Std_DD_89adjusted

# Explicit checkpoint
.venv/bin/python scripts/train_scaffolded_s3.py \
    --resume learned_ai/checkpoints/scaffolded/s2b/best.pt \
    --malom /mnt/windows/NMM_DB/Malom_Standard_Ultra-strong_1.1.0/Std_DD_89adjusted

# Checkpoint → learned_ai/checkpoints/scaffolded/s3/best.pt
```

---

## Overseer — Live Overlay & Player Mode

`OverseerAdvisor` in `learned_ai/models/overseer.py` wraps the `ScaffoldedPolicyNet`
checkpoint and exposes per-move pick probabilities for two purposes:

### Advisory overlay (always active when checkpoint found)

Runs alongside the heuristic engine on every AI turn.  Each legal move in the diagnostic
panel gets an "O:XX%" label showing the policy network's probability for that move.
Helps evaluate whether the trained policy agrees with the heuristic's pick.

Checkpoint search order (most-trained first): `s3/best.pt → s2b/best.pt → s2/best.pt → s1b/best.pt → s1/best.pt`

### Overseer player mode (selectable in UI)

When enabled (`use_overseer_player=True`), replaces the heuristic engine's choice with
the policy network's **argmax** move.  Used to evaluate the ScaffoldedPolicyNet in live
play without writing a full game loop.  Activated via `"use_overseer_player": true` in
the WebSocket game-start message; exposed as a toggle in the game UI.

Both modes call `encode_position()` → `ScaffoldedPolicyNet.policy_probs()`.  The sentinel,
Malom DB, and value net are wired in from the server's global advisors so the policy sees
the same features it was trained on.

**Bug fixed (b9cb779):** Overseer was leaking Malom DB features from the encoding context
into positions where the DB had no entry — a silent feature-value shift that degraded
overlay accuracy.  Fixed by resetting the DB query context correctly per call.

---

## Integration with Existing Engine

The `ScaffoldedAgent` has a `choose_move(board)` interface identical to `HeuristicAgent`
and `LearnedAgent`.  At deployment it calls:

1. `SentinelAdvisor.advise()` — already called by the Coordinator anyway
2. `evaluate(board_after, player, strength_mode=True)` — per legal move, instantaneous static eval
3. `ScaffoldedPolicyNet.policy_logits()` — tiny MLP, <1ms

Total added latency: negligible.  The sentinel was already the bottleneck.

Overseer player mode (see above) is the current live-test path while Stage 2 A2C is
being re-run.  Full `ScaffoldedAgent` integration into `ai/coordinator.py` follows once
Stage 2 converges.

---

## Success Metrics

| Stage | Target | Why |
|-------|--------|-----|
| Stage 1 | Val policy loss decreasing; model plays legal moves better than random | Confirms imitation is working |
| Stage 1.5 | Human-deviated positions weighted; policy loss lower than s1 | Human intuition added on top of heuristic baseline |
| Stage 2 | Rolling-200 win rate ≥ 60% vs difficulty 2 in < 3,000 games | Per-move rewards should produce visible improvement within 500 games |
| Stage 2 → diff 3 | Rolling-200 win rate ≥ 60% | |
| Stage 2b | ≥70% win rate vs diff 7; `bucket_*` all non-zero in logs | Five-level curriculum: 70%→65%→60%→70% exit at diff 7 |
| Stage 3 | Rolling-200 win rate ≥ 40% vs difficulty 4 | Fine-tuning on top of a working Stage 2/2b model |

Early warning (check after 500 games of Stage 2): if win rate is still below 5% and
not trending up, the per-move reward signal is not working.  Check:
- Are sentinel scores varying across moves? (should not all be 0.5)
- Is `h_delta` varying? (should see positive and negative values)
- Is A2C advantage non-zero? (log in training loop)

---

## What Changed vs All Previous Attempts

| | REINFORCE | A2C raw board | **Scaffolded A2C** |
|--|-----------|---------------|-------------------|
| State input | 84-float one-hot board | 84-float one-hot board | **62-float expert features per move** |
| Knows what a good move looks like? | No | No | **Yes — sentinel + heuristic say so explicitly** |
| Reward density | 1 signal / game | per-step TD | **per-step TD + dense per-move shaping** |
| Starting quality | random | imitation prior | **imitation prior at heuristic level** |
| Must discover NMM strategy from scratch? | Yes | Yes | **No** |
| Model can be dependent on sentinel/heuristic? | N/A | N/A | **Yes — they're available at deployment** |

---

## Benchmarking

The agent is given **sentinel + value net by default** — use `--no-agent-sentinel` to
disable sentinel for ablation.

```bash
# Quick smoke test after Stage 1 (5 games vs diff 2)
.venv/bin/python scripts/bench_scaffolded.py \
    --checkpoint learned_ai/checkpoints/scaffolded/s1/best.pt \
    --games 5 --difficulties 2 --opponents raw

# Full benchmark after Stage 2b
.venv/bin/python scripts/bench_scaffolded.py \
    --checkpoint learned_ai/checkpoints/scaffolded/s2b/best.pt \
    --games 40 --difficulties 2,3,4

# Compare s2 vs s2b at diff 3
.venv/bin/python scripts/bench_scaffolded.py \
    --checkpoint learned_ai/checkpoints/scaffolded/s2b/best.pt \
    --compare   learned_ai/checkpoints/scaffolded/s2/best.pt \
    --games 40 --difficulties 3

# Agent without sentinel (ablation)
.venv/bin/python scripts/bench_scaffolded.py \
    --checkpoint learned_ai/checkpoints/scaffolded/s2b/best.pt \
    --games 40 --difficulties 3 --no-agent-sentinel
```

---

## Files

| File | Purpose |
|------|---------|
| `learned_ai/models/scaffolded_encoder.py` | `encode_position()` — builds (k,62) feat matrix + (23,) value input |
| `learned_ai/models/scaffolded_net.py` | `ScaffoldedPolicyNet` — per-move MLP policy + value head |
| `learned_ai/models/overseer.py` | `OverseerAdvisor` — advisory overlay + Overseer player mode |
| `learned_ai/training/scaffolded_a2c.py` | `ScaffoldedStep`, `scaffolded_a2c_update()`, `scaffolded_ppo_update()` |
| `learned_ai/agents/scaffolded_agent.py` | `ScaffoldedAgent` — inference wrapper for gameplay |
| `scripts/gen_imitation_data.py` | Generate supervised dataset from heuristic self-play (book-guided White, `top_n=2`) |
| `scripts/train_scaffolded_s1.py` | Stage 1: imitation training with Malom soft labels (cross-entropy) |
| `scripts/gen_human_imitation_data.py` | Extract human-game dataset from `data/games/*.jsonl` |
| `scripts/train_scaffolded_s1b.py` | Stage 1.5: human-game fine-tune (policy head only) |
| `scripts/train_scaffolded_s2_diagnostic.py` | Stage 2: A2C/PPO with rich per-component reward diagnostics and self-adjusting LR/temperature |
| `scripts/train_scaffolded_s2b.py` | Stage 2b: self-play (model vs frozen copy) + branched mid-game rollouts with phase-bucket saturation cap |
| `scripts/train_scaffolded_s3.py` | Stage 3: Malom supervised fine-tuning; resumes from s2b/best.pt (falls back to s2/best.pt) |
| `scripts/bench_scaffolded.py` | Headless benchmark: ScaffoldedAgent (sentinel + value net on by default) vs heuristic configs |
| `tests/test_scaffolded_policy.py` | 25 unit tests (encoder shapes, net forward, A2C update, agent legality) |
