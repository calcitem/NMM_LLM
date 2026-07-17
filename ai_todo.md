# AI Training — Diagnosis & Plan (2026-07-14)

## Current state (v2 phase specialists, all at 20k games)

| Trainer | Level (/20) | Games at level | Last-100 W/L/D | Decisive WR | Notes |
|---|---|---|---|---|---|
| `s_open_v2` | **9** | ~13k | 21/46/33 | 31.3% | Reached diff 9 @ g6879 |
| `s_mid_v2`  | **7** | ~3.4k | 19/34/47 | 35.8% | Reached diff 7 @ g16660 |
| `s_end_v2`  | **7** | ~18k | 11/25/64 | 30.6% | 64% draw rate |

All three are stalled. Advancement threshold at these levels is ~54% decisive WR (`win/(win+loss)`); current decisive WR is ~30-36%. Gap is huge — this is a capability plateau, not a "close to advancing" situation.

---

## Diagnosis: what's going wrong

### Issue 1 — Draw-heavy strategy is the local optimum

The reward structure lets draws be a comfortable equilibrium:
- `DRAW_SHORT = 0.15`, `DRAW_LONG = -0.05` — draws are near-neutral (positive if short).
- `LAMBDA = 0.5` (retro-outcome weight) halves the outcome signal.
- `_check_advance` ignores draws (`win/(win+loss)`), so a draw-heavy strategy can accumulate arbitrarily many games at a level without ever advancing.

Endgame is the worst offender: 64% draw rate over the last 100 games.

### Issue 2 — Features are heuristic-flavored; no signal for deviating

Each of the 122 per-move floats is dominated by heuristic-derived signals:
- Float 60 (`is_engine_top1`) explicitly flags the heuristic's best move.
- Floats 59, 61 encode heuristic + VN blended eval and delta.
- The 15-ply lookahead simulates *both sides using the static heuristic* — so deeper lookahead just extends "here's what heuristic-vs-heuristic play looks like from this candidate move".

The specialist can copy the heuristic (giving ~50% WR + draws) but has no path to *out-search* it. It has no information about what happens if it deviates and sticks to its own policy.

### Issue 3 — Endgame pool contains many objectively-drawn positions

End specialist starts from real endgame positions (4-11 pieces). A large fraction of these have exact Malom WDL = "D" — no winning strategy exists. Specialist can't do better than the heuristic on those. Also, Malom info is present in features but not used as a reward signal.

### Issue 4 — Retro-decay is aggressive

`LAMBDA=0.5 × 0.98^plies_remaining` — a placement 30 plies from the terminal only receives ~27% of the outcome signal. Early placements barely feel the game's outcome, so credit assignment is weak on the moves that shape the game most.

### Issue 5 — Time-budget cap flattens the ladder above L15

`_time_budget_for_level` capped at 2 s (mid/end) / 1 s (opening). From L15 onwards, all levels use the same heuristic time budget. Level difference above L15 is only depth-based, not compute-based.

---

## Batch 1 — reshape the incentive gradient (implement now)

All three trainers. These changes are coherent — they push the policy gradient away from "safety play" toward "earn wins".

- **`_check_advance` numerator counts draws at 0.5**: threshold becomes `(wins + 0.5×draws)/total ≥ level_threshold`. Level 1 threshold stays 0.51 → level 20 stays 0.60. A draw-heavy strategy stops advancing.
- **Reward weights**:
  - `DRAW_SHORT` : `0.15 → 0.0`
  - `DRAW_LONG`  : `-0.05 → -0.15`
  - `LAMBDA`     : `0.5 → 0.7`   (outcome matters more)
  - `DECAY`      : `0.98 → 0.99` (outcome reaches further back)
- **Explore bonus (Option A)**: on retro-rescoring, if a chosen move was *not* the heuristic top-1 and the game was a win, add `EXPLORE_COEF × (1 − is_top1_heuristic)` per step. `EXPLORE_COEF = 0.08`. Reshapes gradient toward creative wins.
- **Malom-agreement bonus (endgame only)**: on retro-rescoring, if the chosen move matches Malom's winning classification, add `MALOM_AGREE_BONUS = 0.10 × outcome_positive`. If it's a losing move per Malom, subtract same. Uses `enc.db_moves` already computed.

**Why together, not sequential**: they're all "reshape the loss surface" changes. Splitting them makes each individual signal too weak to move the needle in a reasonable time. The trade-off — we won't perfectly attribute impact — is worth it.

**Existing 20k-game history stays valid**: no architectural changes; checkpoints still load.

---

## Update (2026-07-14, post-Batch 1)

Batch 1 lifted all three specialists from diff 7-9 to **diff 10 / 20** — a real move but still very slow: 20k additional games each. That's the point at which Option C from Batch 2 was implemented.

### Follow-up patch — draw-rate cap in advancement (2026-07-14, mid-run)

Observed: `s_end_v2` advanced from diff 4 → 5 with a 50-game window of 10% wins, 84% draws, 6% losses. `(0.10 + 0.5×0.84) = 0.52` cleared the diff-4 threshold, but a 10% real WR is not "beating" the heuristic.

Applied to all three v2 trainers:
- **`_check_advance` now blocks advancement when `draw_rate ≥ 33%`** regardless of the weighted score. Prevents "84% draws" from ever qualifying.
- **Reward reweight**: `DRAW_SHORT: 0.00 → -0.10`, `DRAW_LONG: -0.15 → -0.25`. Draws are now unambiguously negative in both forms.

Sanity-tested — the exact case that triggered this change (10 W / 84 D / 6 L @ diff 4) now correctly returns "don't advance: draw_rate 0.84 >= 0.33".

## Batch 2 — feature/lookahead surgery (implement only if Batch 1 doesn't move the needle)

The full menu of options for the "heuristic-flavored features, can't see own deviations" issue. Ordered by cost / impact.

### Option A — Explore bonus [INCLUDED IN BATCH 1]
Small reward for non-heuristic-top1 moves that lead to wins.
- Cost: near-zero.
- Effect: policy-gradient pull toward useful deviations.
- Risk: small noise floor if heuristic usually is best.
- Status: **implemented in Batch 1**.

### Option B — Reduce heuristic-signal dominance in input
Zero out or drop `is_engine_top1` (float 60) at training time via dropout mask. Same for heuristic delta blend (float 61).
- Cost: near-zero, tweaks encoder or applies mask in training loop.
- Effect: forces specialist to learn from raw position features, not from being told the answer.
- Risk: could hurt convergence at low levels. Might need warmup schedule (drop starts 0%, ramps to 50% by L5).
- Status: hold until Batch 1 outcome.

### Option C — Model-driven lookahead for the learner side (biggest lever) ✓ IMPLEMENTED 2026-07-14
Change `LookaheadAdvisor._simulate_trajectory`: on plies where it's the learner's turn, pick the *frozen model*'s argmax instead of the heuristic top move. Opponent side stays heuristic.
- Cost: up to ~8 model forward passes per lookahead trajectory. Move encoding still dominates.
- Effect: directly addresses "can't see own deviations". Each candidate's lookahead shows "if I play this and continue with my policy, do I win?"
- Risk: non-stationarity. Mitigation: use *frozen* model snapshot (like `frozen_opp`), refreshed every N games.
- **Implementation**: `LookaheadAdvisor.set_frozen_model(model, device)`; all three v2 training scripts share `frozen_opp._model` by reference so in-place `refresh()` keeps lookahead sync'd for free. Frozen-model encoding uses `encode_position_with_lookahead(..., lookahead_advisor=None)` (zero-pads the 60-float block) — avoids infinite recursion, base 62 floats still policy-informative. Falls back to heuristic on any failure.

### Option D — Contrastive lookahead (top-K only)
Instead of full 15-ply for every candidate, do 5-ply lookahead for only the top-K policy candidates and encode the *difference* between top-1 and each non-top-1 continuation.
- Cost: fewer lookahead calls (`K × 5` vs current `k × 15`, K=3 typical).
- Effect: focuses attention on the "gap" between candidates. Reduces feature noise.
- Risk: policy needs to be reasonable first.
- Status: only as a follow-on to A, not standalone.

### Option E — Search-augmented inference (deployment)
At inference, shallow beam search or MCTS over the top-K policy moves. Uses the 1s + sentinel budget on real analysis. Not touched during training.
- Cost: implement search wrapper in `ScaffoldedAgent`. Independent of training.
- Effect: cheapest way to actually *spend* the specialist's deployment time budget. Typical Elo lift 100-200 vs raw policy.
- Risk: none — doesn't affect training.
- Status: parallel track. Implement whenever convenient.

---

## Ranked follow-through if Batch 1 alone doesn't move the specialists

1. **Batch 1** (this pass): `_check_advance` change + reward reweighting + Option A + Malom bonus (endgame).
2. **Option C** (model-driven lookahead, frozen snapshot). Highest leverage of remaining items.
3. **Option E** (search at inference). Independent, always useful.
4. **Option B** or **D** (feature dropout / contrastive lookahead). Only if 1-3 haven't cracked the plateau.

---

## Notes / caveats

- Do not lower the raw advance threshold. Lowering just lets specialists walk up a ladder they haven't earned. Fix the play, not the bar.
- All Batch 1 changes are reward/threshold — they steer training but don't change architecture. If a specialist has hit its *capacity* limit (128,64 hidden MLP), no reward change will help. Batch 2's Option C is the first change that could push through a capacity ceiling.
- Time-budget cap (`_time_budget_for_level` maxes at 2 s / 1 s) means top-of-ladder levels are only depth-differentiated, not compute-differentiated. This is intentional per the "specialist ≥ 2 × heuristic time" rule, but consider revisiting if specialists reach L15+ decisively.
