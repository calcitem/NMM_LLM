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
- If quality < 0.1 after warmup (sentinel calls it a clear blunder), *do not add that
  transition to the replay buffer*.
- Sentinel does **not** override the move selection — the policy still chooses freely.
- No filtering for the first 20% of games (warmup) — the model must accumulate enough
  transitions before blunder filtering is useful.

**Algorithm:** REINFORCE with value-head baseline (same as original Stage 2/3).

**Malom DB reward shaping (two signals, both Malom-exact, active for first 30% of games):**

1. **Move quality** — `query_move_quality(board, move)` returns a delta ∈ [−2, +2] from
   the mover's perspective.  Scaled by `malom_weight=0.3` and added to the transition reward.
   Rewards moves that directly improve the learner's own position.

2. **Trap reward** — after each learner move, `query(board)` is called on the resulting
   position from the *opponent's* perspective.  If the opponent is now in an "L" (losing)
   state, the learner's transition receives an additional `+malom_weight` bonus.  This rewards
   moves that constrain or trick the opponent into bad territory — the core strategic skill in
   NMM (cycling mill setups — oscillating a pivot piece between two 2-configs to force a capture
   every turn — forced captures, zugzwang).  The sentinel approximates
   this signal; Malom is exact.

Both signals are zero-overhead (Malom DB already loaded) and degrade gracefully if a position
is outside the DB's coverage.

**Coverage note:** Malom DB coverage drops significantly in the midgame (many pieces placed,
complex positions) — `query()` returns `None` more often there.  Once training is further
along, monitor how often each signal fires per phase to confirm the model is still receiving
useful reward in the midgame and not only in placement/endgame positions.

**Implementation notes:**
- `override_time_budget=0.05s/move` passed to GameAI so training games run in ~3s not ~27s.
- Opponent's **first move is forced random** each game to ensure the learner sees varied
  opening positions.
- Temperature = 0.5 (less random than v1), UPDATE_EVERY = 16 (larger stable batches),
  WIN_REWARD = 2.0 (strong terminal signal).

**Exit criterion:** rolling 200-game win rate ≥ 65% vs difficulty 3.

#### Stage 2 — Attempt History

**v1 (killed at game ~4372):**

| Root cause | Effect |
|------------|--------|
| Sentinel threshold 0.25 too aggressive in early training | 38% of transitions filtered → batches of 40–60, too small to learn from |
| Value head collapsed to predict loss for every position | Advantages all near zero → near-zero policy gradient |
| Temperature = 1.0 too random | Model never explored intentionally; drew many games it could have won |
| No Malom shaping | Only terminal reward: one sparse signal per game |

Win rate at game 4372: 7.5% (target 65%).  Training was not converging.

**v2 (killed at game ~200 — restarted with trap reward):**

Changes from v1: sentinel warmup (no filter for first 1000 games), Malom move-quality reward
(first 1500 games), temperature reduced to 0.5, UPDATE_EVERY=16, WIN_REWARD=2.0, advantage
normalisation guarded by `std > 1e-3`.  Win rate at game 176: 2.3% — too early to judge, but
the trap reward was identified as the missing signal and the run was restarted.

**v3 (current — in progress):**

Added Malom trap reward on top of v2: after each learner move, if `query(board)` returns "L"
from the opponent's perspective (opponent is now in a losing position), the learner's transition
receives an additional `+malom_weight` bonus.  This directly rewards the core NMM strategic
skill — creating positions where the opponent has no good response (cycling mill setups, forced
captures, zugzwang) — rather than only rewarding moves that improve the learner's own
evaluation.  Both Malom signals are exact; the sentinel approximates them.

| Parameter | v1 | v2/v3 |
|-----------|----|----|
| Temperature | 1.0 | 0.5 |
| UPDATE_EVERY | 4 | 16 |
| WIN_REWARD | 1.0 | 2.0 |
| Sentinel threshold | 0.25 | 0.1 (after warmup) |
| Sentinel warmup | none | first 1000 games |
| Malom move-quality reward | none | first 1500 games, weight=0.3 |
| Malom trap reward | none | first 1500 games, weight=0.3 |
| Checkpoint | — | `learned_ai/checkpoints/stage2/` |

Status: **in progress** — game ~80 / 5000.

---

### Stage 3 — Curriculum vs Heuristic + Value Net

**Goal:** Climb from weak to strong heuristic opponent.

**Opponent:** Heuristic engine, difficulty ramps 3 → 8, with vn_blend=80% at difficulties 6+.

**Difficulty ramp rule:** hold ≥ 55% win rate over a 200-game rolling window before bumping
difficulty.  Temperature resets at each bump (same as original Stage 3).

**Sentinel role:** Keep the sentinel blunder filter active (quality < 0.1 → skip transition).
As the curriculum advances and the learner strengthens, the filter will fire less often
naturally.

**Malom DB reward shaping:** Both Malom signals (move quality + trap reward) remain active
throughout Stage 3 with no game-count cutoff.  The model is strong enough by this stage that
the rewards reinforce genuinely good play rather than noise.  The AI does **not** receive the
raw W/L/D labels as input features — it learns from the reward signal only.

**Opponent move replay (new in Stage 3):** In every lost game, the opponent's moves are
examined post-game.  Any move where `query_move_quality >= 0` (Malom confirms it was W or D
for the opponent) is added to a supervised imitation batch.  A small CE loss
(`imitation_weight=0.1`) on these transitions teaches the model what winning play looks like
from the exact positions it failed at.  Only Malom-exact moves are used — heuristic moves that
Malom disagrees with are discarded.  This mirrors how human players study lost games to
understand the opponent's edge.

**Training quality note:** Stage 3+ prioritises quality over speed.  The opponent uses a full
time budget (0.3 s – 1.0 s/move), vn_blend=80% at difficulty 6+, and has access to the
fullgame DB and endgame DB.  Extended wall-clock time is acceptable.

**Exit criterion:** 55% win rate held at difficulty 8 + vn_blend=80%.

**Why difficulty 8 not 10:** the vn_blend=80% engine is already the strongest we have
evidence for.  Setting the bar at "beat difficulty 10 without value net" is a much harder
target than beating the configuration we know is good.

---

### Stage 4 — Self-play Pool

**Goal:** Open-ended strength improvement through self-play against a pool of past checkpoints.

**Method:** Standard pool-based self-play (keep N past checkpoints; randomly sample opponents).
Remove the sentinel blunder filter here — the model should be strong enough that blunders are
rare, and filtering them would bias the replay buffer.

**Malom DB reward shaping:** Both Malom signals remain active throughout Stage 4.  The AI
still does **not** see the raw W/L/D labels — reward shaping only.

**Exit criterion:** Benchmark vs the Stage 0 baseline (heuristic + vn_blend=80%) shows ≥ 70%
win rate, or episode budget exhausted (70k games).

---

### Stage 5 — Malom DB Full-game Supervised Distillation  *(revised)*

**Goal:** Skill refinement by distilling Malom's perfect play across the entire game, not just
endgame positions.

**Why revised from endgame-only:** Stages 2–4 gave the model Malom rewards as a training
signal throughout the game, but the model never directly *saw* Malom's W/L/D assessment of
its own moves.  Stage 5 closes that gap: every legal move in every position gets an exact Malom
label, and the model is trained supervised on the full game trajectory.

**Method:**
- Sample positions from across the full game (all piece counts, all phases) from the Malom DB.
- For each position, query Malom W/L/D for every legal move.
- Supervised training:
  - **Value head target:** W=+1.0, D=0.0, L=−1.0 (exact WDL for the side to move).
  - **Policy head target:** cross-entropy toward the distribution of Malom-winning moves
    (uniform over all "W" moves if any exist; otherwise uniform over "D" moves; otherwise "L").
- Light LR (1e-5); only a few epochs to avoid catastrophic forgetting of generalisation learned
  in Stages 2–4.

**Policy target refinement (future):** "Uniform over W moves" is the baseline target.  If
Malom has DTM (distance to mate) available, W moves can instead be weighted inversely by DTM
so faster wins receive higher probability mass.  This is not required for correctness but
improves the sharpness of the resulting policy.  Treat as a refinement once the supervised
training is otherwise stable.

**Structural feature enrichment (consider before Stage 5):** The current state encoder uses
raw piece positions (84-float flat vector).  The model has to infer mill threats and piece
relationships from data alone.  Adding a small set of pre-computed structural features — mill
threat count, open triples (two own pieces on a mill line with the third empty), mobility
(number of legal moves) — would give the model explicit pattern context before supervised
distillation begins.  The sentinel's `feature_builder.py` already computes these; the same
features can be appended to the state encoder without changing the backbone architecture.
A more powerful option is a graph-neural-network (GNN) backbone where edges are mill
adjacencies, but that requires an architecture change.  The flat feature extension is lower
risk and is the recommended first step.

**Key distinction from Stages 2–4:** The model now *sees* the Malom W/L/D labels as
supervised targets, not just as a reward shaping signal.  This is the first time perfect
knowledge is injected directly into the policy rather than learned from reward alone.

**Why at the end:** supervised distillation from perfect labels onto a strong generalising
model is much more effective than early injection when the rest of the network is random.
Catastrophic forgetting risk is low because the generalisation was learned first.

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
| After Stage 5 | Policy selects a Malom-winning move (where one exists) in ≥ 85% of full-game positions; value head WDL accuracy ≥ 80% across all phases |

The Stage 4 target is deliberately aggressive: +17.5 pp was achieved with a simple linear
value blend.  A learned policy + value head should be able to do better, but 70% is a
meaningful bar that requires genuine tactical and strategic understanding, not just better leaf
evaluation.

Stage 5 is distinct from all prior stages: it is the only stage where the model directly *sees*
perfect Malom labels rather than learning from reward alone.  Stages 2–4 built the strategic
intuition; Stage 5 sharpens it to Malom precision across the full game.
