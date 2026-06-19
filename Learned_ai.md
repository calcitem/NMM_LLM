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
| [40:58) | `feature_builder.counterfactual_features()` | Heuristic rank, normalised score, DB win/loss fracs, DTM quality |
| [58] | `SentinelAdvisor.advise().move_scores[i]` | Sentinel's quality score for this move, [0,1] |
| [59] | `evaluate(board_after, player, strength_mode=True)` → mapped [0,1] | Absolute heuristic evaluation of the resulting position |
| [60] | `is_top1` | 1.0 if this is the heuristic engine's #1 ranked move |
| [61] | `tanh(h_after − h_before)` | Heuristic improvement after this move |

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

## Reward Structure

### Per-move shaped rewards (dense — every learner turn)

```
r_sentinel   = 0.15 × (sentinel_score_played − mean_sentinel_score)
               Did we play above the average quality sentinel sees for this position?

r_heuristic  = 0.10 × tanh(h_after − h_before)
               Did the heuristic evaluation improve after our move?

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

**Method:** Play heuristic (diff 3) vs heuristic self-play.  At each position, record the
heuristic's actual move as the supervised target.  Train with cross-entropy (policy) +
MSE (value against heuristic evaluation).

**Why this works:** The heuristic already plays well.  Imitating it gives the model a
baseline that generates non-trivial per-move reward from the first game of Stage 2.

**Expected outcome:** Model learns to play at roughly heuristic quality.  Confirmed
by checking that policy cross-entropy loss decreases and val accuracy > 20%.

**Commands:**
```bash
# Generate dataset (~2h for 2000 games)
.venv/bin/python scripts/gen_imitation_data.py \
    --games 2000 --diff 3 \
    --sentinel learned_ai/sentinel/checkpoints/best.pt

# Train imitation model (~10 min)
.venv/bin/python scripts/train_scaffolded_s1.py \
    --data learned_ai/data/imitation_scaffolded.npz \
    --epochs 20

# Checkpoint → learned_ai/checkpoints/scaffolded/s1/best.pt
```

---

### Stage 2 — A2C self-play with full scaffolded rewards

**Goal:** Learn to actually win, using dense per-move rewards + retroactive rescoring.

**Opponent:** Heuristic engine, difficulty 2 → 3.
**Algorithm:** A2C (or `--ppo` for PPO).
**Temperature:** 0.5 annealed to 1.2 (model starts near imitation prior; explore more as training progresses).
**LR:** 1e-4 (model is new, not fine-tuning a pre-trained checkpoint — we can use a higher rate).

**Curriculum:**
- Start vs diff 2.
- Advance to diff 3 when rolling-200 win rate ≥ 60%.
- Exit when rolling-200 win rate ≥ 60% at diff 3.

**Why higher LR than old Stage 2:** Old Stage 2 used 5e-6 to protect a pre-trained
checkpoint.  ScaffoldedNet starts from Stage 1 imitation, not a pre-trained NMMNet
backbone — a fresh model that needs to learn.  1e-4 is appropriate.

**Commands:**
```bash
.venv/bin/python scripts/train_scaffolded_s2.py \
    --sentinel learned_ai/sentinel/checkpoints/best.pt \
    --max-games 10000

# Resumes automatically from s1/best.pt
# Checkpoint → learned_ai/checkpoints/scaffolded/s2/best.pt
# Log → learned_ai/checkpoints/scaffolded/s2/train_log.jsonl
```

To use PPO instead:
```bash
.venv/bin/python scripts/train_scaffolded_s2.py --ppo --max-games 10000
```

---

### Stage 3 — Malom supervised fine-tuning

**Goal:** Sharpen strategy with explicit DB supervision.  When Malom knows the WDL for
all legal moves, push the policy toward the winning move(s), weighted by DTM quality.

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
.venv/bin/python scripts/train_scaffolded_s3.py \
    --malom /path/to/malom/db \
    --sentinel learned_ai/sentinel/checkpoints/best.pt \
    --diff 3

# Resumes from s2/best.pt automatically
# Checkpoint → learned_ai/checkpoints/scaffolded/s3/best.pt
```

If Malom DB is not yet built:
```bash
# Stage 3 still runs — SL signal will be zero, it falls back to pure A2C
.venv/bin/python scripts/train_scaffolded_s3.py --diff 3
```

---

## Integration with Existing Engine

The `ScaffoldedAgent` has a `choose_move(board)` interface identical to `HeuristicAgent`
and `LearnedAgent`.  At deployment it calls:

1. `SentinelAdvisor.advise()` — already called by the Coordinator anyway
2. `evaluate(board_after, player, strength_mode=True)` — per legal move, instantaneous static eval
3. `ScaffoldedPolicyNet.policy_logits()` — tiny MLP, <1ms

Total added latency: negligible.  The sentinel was already the bottleneck.

To wire into the game, add `ScaffoldedAgent` as an option in `ai/coordinator.py`
alongside the existing learned agent path.

---

## Success Metrics

| Stage | Target | Why |
|-------|--------|-----|
| Stage 1 | Val policy loss decreasing; model plays legal moves better than random | Confirms imitation is working |
| Stage 2 | Rolling-200 win rate ≥ 60% vs difficulty 2 in < 3,000 games | Per-move rewards should produce visible improvement within 500 games |
| Stage 2 → diff 3 | Rolling-200 win rate ≥ 60% | |
| Stage 3 | Rolling-200 win rate ≥ 40% vs difficulty 4 | Fine-tuning on top of a working Stage 2 model |

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

## Files

| File | Purpose |
|------|---------|
| `learned_ai/models/scaffolded_encoder.py` | `encode_position()` — builds (k,62) feat matrix + (23,) value input |
| `learned_ai/models/scaffolded_net.py` | `ScaffoldedPolicyNet` — per-move MLP policy + value head |
| `learned_ai/training/scaffolded_a2c.py` | `ScaffoldedStep`, `scaffolded_a2c_update()`, `scaffolded_ppo_update()` |
| `learned_ai/agents/scaffolded_agent.py` | `ScaffoldedAgent` — inference wrapper for gameplay |
| `scripts/gen_imitation_data.py` | Generate supervised dataset from heuristic self-play |
| `scripts/train_scaffolded_s1.py` | Stage 1: imitation training |
| `scripts/train_scaffolded_s2.py` | Stage 2: A2C/PPO with full scaffolded rewards |
| `scripts/train_scaffolded_s3.py` | Stage 3: Malom supervised fine-tuning |
| `tests/test_scaffolded_policy.py` | 25 unit tests (encoder shapes, net forward, A2C update, agent legality) |

### Previous architecture (kept, not used for scaffolded training)

| File | Status |
|------|--------|
| `learned_ai/models/backbone.py` — NMMNet (84→624) | Kept; used by old LearnedAgent |
| `learned_ai/models/gnn_backbone.py` — NMMGNNNet | Kept, abandoned (Stage 1 accuracy 4.6%) |
| `learned_ai/training/a2c.py` | Old A2C for NMMNet |
| `scripts/train_stage2.py` | Old Stage 2 (failed at 2% win rate) |
| `learned_ai/checkpoints/stage0/`, `stage1/`, `stage2/` | Old checkpoints |
