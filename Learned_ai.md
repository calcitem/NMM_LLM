# Learned AI — Resurrection Plan

The original `learned_ai/` attempt used REINFORCE self-play from a random initialisation and
never produced a model that could beat the heuristic engine.  This document explains why it
failed, what we now know works, and a concrete staged plan for a second attempt.

---

## Why the First Attempt Failed

| Root cause | Effect |
|------------|--------|
| Pure self-play from random init | Garbage-in / garbage-out — both players were equally uninformed, so the network never received a signal about what *good* NMM play looks like |
| Binary terminal reward only | Extremely sparse signal; a 60-ply game contributes one bit of reward information |
| No curriculum grounding | Stage 2 (vs random) was the only stable floor; stages 3–4 stalled because the heuristic opponent was orders of magnitude stronger than the learner |
| Value net trained only on self-play noise | No useful baseline; advantage estimates were noisy, destabilising policy gradients |
| Policy head replaced heuristic wholesale | No safety net — once training diverged the checkpoint was useless |

The underlying architecture (`NMMNet`, phase heads, legal-action masking) is sound and reusable.
The *training signal* was the problem.

---

## What We Now Know Works

| Component | Evidence |
|-----------|----------|
| **Value net at 80% blend** | +17.5 pp vs plain heuristic in a 40-game bench (8W/1L/31D) |
| **Malom DB** | Perfect DTM labels available for any position with ≤ 18 pieces placed; used in sentinel Stage 5 training |
| **HumanDB** | 22,895 real games, 642,703 positions in SQLite; quality labels derivable from win/loss outcome |
| **Sentinel** | Reliable *quality* classifier at the move level; advisory mode is well-calibrated; `score_adjust` (overriding alpha-beta) actively hurts — do not use it for engine steering |
| **Trajectory DB** | ~27k games indexed by move prefix; winner's path is a cheap imitation-learning signal |

---

## New Strategy: Supervised First, Self-play Second

The key change from the first attempt: **start from a grounded value function, not a random
one**.  The value net (`data/value_net.npz`) already encodes meaningful position evaluation.
Pre-training the new model's value head from this signal — before any RL — gives every
subsequent self-play update a useful baseline.

### Architecture reuse

Keep `NMMNet` (backbone + phase heads + value head) from `learned_ai/models/`.  No changes
needed.  The 84-float state encoder and 624-action space are already correct.

---

## Training Stages

### Stage 0 — Supervised Value Pre-training  *(new)*

**Goal:** Bootstrap the value head so it is not random noise from the first update.

**Method:**
1. Generate ~50k positions by running the heuristic engine (difficulty 6, vn_blend=80%) in
   self-play; record board states + the value-net score for each.
2. Supervised regression: train only the value head (freeze backbone initially) to predict the
   value-net score.
3. Unfreeze backbone and continue for a few more epochs with a small LR.

**Exit criterion:** value-head MSE stabilises (no improvement over 3 epochs).

**Why:** The value net is already +17.5 pp over the heuristic.  A pre-trained value head means
the REINFORCE baseline is informative from episode 1, not after 50k episodes of noise.

#### Stage 0 — Results

| Item | Value |
|------|-------|
| Data gen | 500 games, 18 workers, diff=5, vn_blend=80%, budget=0.1s/move |
| Positions | 28,537 (phase dist: place/early=4000, place/late=5000, midgame=16138, endgame=1916, fly=1483) |
| Phase 1 training | frozen backbone, lr=3e-3, 20 epochs — val MSE 0.384 → **0.230** |
| Phase 2 training | full network, lr=5e-4, 30 epochs — val MSE 0.295 → **0.012** |
| Checkpoint | `learned_ai/checkpoints/stage0/best.pt` |
| Notes | 0.012 val MSE on [−1,+1] scale ≈ 0.11 avg absolute error. No overfitting (train=0.002). Ready to use as `--resume` for Stage 1. |

---

### Stage 1 — Imitation Learning from Human Games  *(new)*

**Goal:** Give the policy head a reasonable prior over move selection before RL begins.

**Data:** HumanDB (30,256 games, 820,495 positions, 932,141 move records).  Label: win-rate
of each move (wins + 0.5·draws) / total, used as a per-sample weight in the CE loss.

**Method:**
- Query moves table (filter total ≥ 5) → ~820k raw (position, move) pairs.
- Apply all 8 D4 symmetry augmentations per sample (board + notation transformed together)
  so the policy sees every board orientation, not just the canonical one stored in the DB.
- Cross-entropy loss on the primary action (placement/movement slice [0:599]), weighted by
  win-rate.  Captures ([600:623]) deferred to Stage 2 (self-play value signal handles them).
- Two-phase: (1) freeze backbone, high LR; (2) full network, low LR with early stop.

**Exit criterion:** move-prediction accuracy on a held-out split stops improving.

**Why:** In the first attempt the policy started at uniform random.  After this stage it will
have a reasonable prior — play that real humans have found effective.  This is the same
approach that made AlphaGo's RL phase converge: start from supervised human imitation, not
random.

#### Stage 1 — Results

| Item | Value |
|------|-------|
| DB rows used | 21,131 (total ≥ 5) → 169,048 after 8× D4 augmentation |
| Phase distribution | opening place=32k, full place=40k, midgame=93k, endgame=1.1k, fly=1.1k |
| Phase 1 training | frozen backbone, lr=3e-3, 20 epochs — val_acc 4% → **23.5%** |
| Phase 2 training | full network, lr=5e-4, 40 epochs — val_acc 30% → **45.3%** (early stop ep 37) |
| Exit criterion | Exceeds >30% target ✓ |
| Checkpoint | `learned_ai/checkpoints/stage1/best.pt` |
| Notes | Train/val accuracy gap (52%/45%) suggests mild overfitting; val accuracy still generalising well. Captures deferred — CE trained on primary actions only. |

---

#### Pre-Stage 2 Baseline

Stage 1 checkpoint (greedy, temp=0) vs heuristic difficulty 2, vn_blend=0:
W=0, D=25, L=15 over 40 games — 0% win rate, 62.5% draw rate.
Confirms the imitation policy has learned to defend but cannot win without RL.

---

### Stage 2 — Sentinel-Filtered Self-play vs Weak Heuristic

**Goal:** Begin RL while filtering the noisiest moves so self-play does not diverge.

**Opponent:** Heuristic engine, difficulty 2–3, vn_blend=0% (weak; learner should win often
enough to get positive signal).

**Sentinel role (advisory only):**
- After each move in self-play, query the sentinel for the played move's quality score.
- If quality < 0.25 (sentinel calls it a clear blunder), *do not add that transition to the
  replay buffer*.
- This prevents the model from reinforcing obviously bad moves while it is still learning.
- Sentinel does **not** override the move selection — the policy still chooses freely.

**Algorithm:** REINFORCE with value-head baseline (same as original Stage 2/3).

**Implementation notes:**
- `override_time_budget=0.05s/move` passed to GameAI so training games run in ~3s not ~27s.
- Opponent's **first move is forced random** each game to ensure the learner sees varied
  opening positions (without this, every game diverges from the same 1–2 opponent placements).

**Exit criterion:** rolling 200-game win rate ≥ 65% vs difficulty 3.

---

### Stage 3 — Curriculum vs Heuristic + Value Net

**Goal:** Climb from weak to strong heuristic opponent.

**Opponent:** Heuristic engine, difficulty ramps 3 → 8, with vn_blend=80% at difficulties 6+.

**Difficulty ramp rule:** hold ≥ 55% win rate over a 200-game rolling window before bumping
difficulty.  Temperature resets at each bump (same as original Stage 3).

**Sentinel role:** Keep the sentinel blunder filter active (quality < 0.25 → skip transition).
As the curriculum advances and the learner strengthens, the filter will fire less often
naturally.

**Training quality note:** Stage 3+ prioritises quality over speed.  The opponent uses a full
time budget (0.3 s – 1.0 s/move), vn_blend=80% at difficulty 6+, and has access to the
fullgame DB and endgame DB.  Extended wall-clock time is acceptable.

**Exit criterion:** 55% win rate held at difficulty 8 + vn_blend=80%.

**Why difficulty 8 not 10:** the vn_blend=80% engine is already the strongest we have
evidence for.  Setting the bar at "beat difficulty 10 without value net" is a much harder
target than beating the configuration we know is good.

---

### Stage 4 — Self-play Pool  *(same as original Stage 4)*

**Goal:** Open-ended strength improvement through self-play against a pool of past checkpoints.

**Method:** Standard pool-based self-play (keep N past checkpoints; randomly sample opponents).
Remove the sentinel blunder filter here — the model should be strong enough that blunders are
rare, and filtering them would bias the replay buffer.

**Exit criterion:** Benchmark vs the Stage 0 baseline (heuristic + vn_blend=80%) shows ≥ 70%
win rate, or episode budget exhausted (70k games).

---

### Stage 5 — Malom DB Endgame Fine-tune  *(same as sentinel Stage 5)*

**Goal:** Perfect endgame play by distilling from the Malom tablebase.

**Method:**
- Sample positions with ≤ 7 pieces on board from the Malom DB.
- For each position, run a few seconds of Malom lookup to get the exact DTM (distance to mate)
  for every legal move.
- Supervised training: value head target = `tanh(DTM-normalised)`, policy target = move that
  minimises DTM (or maximises for the losing side).
- Light LR (1e-5); only a few epochs to avoid catastrophic forgetting.

**Why at the end:** fine-tuning a strong generalising model on perfect labels is much more
effective than injecting perfect-label supervision when the rest of the network is random.
This mirrors what worked in the sentinel retraining.

---

## Integration Points

The new model should *supplement* the existing engine, not replace it immediately:

| Mode | What happens |
|------|-------------|
| **Evaluation** | `bench_sentinel.py`-style A/B: new learned agent vs heuristic+vn80% |
| **Advisory** | Like sentinel — display the learned agent's top move alongside the heuristic's choice in the AI Discussion panel |
| **Hybrid** | Use the learned model's value output as an additional leaf-eval blend (similar to vn_blend) rather than replacing alpha-beta entirely |

The hybrid mode is the lowest-risk path to a playable improvement.  The learned value head's
output can be blended at, say, 20% alongside the existing value net (80%) once it reaches
parity with the value net baseline.

---

## What to Reuse vs Rewrite

| Component | Status |
|-----------|--------|
| `learned_ai/models/` — NMMNet, encoders, action space | **Keep as-is** |
| `learned_ai/agents/` — LearnedAgent, HeuristicAgent, RandomAgent | **Keep as-is** |
| `learned_ai/training/replay_buffer.py` | **Keep as-is** |
| `learned_ai/training/trainer.py` | **Extend** — add sentinel blunder filter hook |
| `learned_ai/training/self_play.py` | **Extend** — expose sentinel query point |
| `learned_ai/training/curriculum.py` | **Rewrite** — new 6-stage schedule |
| `scripts/train.py` | **Extend** — add `--stage 0` (supervised pre-train) path |
| `learned_ai/config/default_config.yaml` | **Replace** with new staged config |
| `scripts/benchmark_vs_heuristic.py` | **Keep** — already wires HeuristicAgent |

Scripts that were part of the first attempt and are now stale:
- `scripts/evaluate.py`, `scripts/evaluate_sentinel.py` — overlap with `bench_sentinel.py`;
  consolidate or remove.
- `scripts/run_self_play.py` — useful for data generation; keep but update to accept a
  `--sentinel-filter` flag.
- `scripts/human_vs_learned.py` — keep for manual testing.
- `scripts/retrain_pipeline.sh` — replace with a new shell script for the 6-stage run.

---

## Success Metrics

| Milestone | Target |
|-----------|--------|
| After Stage 0 | Value-head MSE < 0.08 on held-out positions |
| After Stage 1 | Top-1 move accuracy > 30% on held-out human games |
| After Stage 2 | ≥ 65% win rate vs heuristic difficulty 3 |
| After Stage 3 | ≥ 55% win rate vs heuristic difficulty 8 + vn80% |
| After Stage 4 | ≥ 70% win rate vs heuristic + vn80% baseline |
| After Stage 5 | Near-perfect play in positions with ≤ 7 pieces (Malom-verifiable) |

The Stage 4 target is deliberately aggressive: +17.5 pp was achieved with a simple linear
value blend.  A learned policy + value head should be able to do better, but 70% is a
meaningful bar that requires genuine tactical and strategic understanding, not just better leaf
evaluation.
