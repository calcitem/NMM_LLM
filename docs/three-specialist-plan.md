# Three-Specialist AI Plan (v2 — No Overseer)

## Goal

Remove the OverseerAdvisor meta-layer. Retrain opening / midgame / endgame specialists
independently, each acting directly at inference for its own phase. Each specialist has
access to the same information it will see at inference: sentinel, heuristics, trajectory
value net, and gap net — plus a lookahead block where it observes sentinel data across
simulated futures.

---

## Summary of changes from current architecture

| Aspect | Current | v2 |
|---|---|---|
| Inference routing | OverseerAdvisor aggregates 3 specialists | Phase router → specialist directly |
| Move feat dim | 77 (62 base + 15 lookahead) | 122 (62 base + 60 lookahead) |
| Lookahead signals | 3 per ply (h, vn, sent) | 4 per ply (h, vn, sent, gap) |
| Lookahead sentinel | disabled by default | **enabled** during training |
| Gap net | not in specialist features | lookahead signal 4 (gap_norm per ply) |
| Reward (all 3) | varied | sentinel delta + heuristic delta + mill bonus (Malom zeroed) |
| Heuristic rollouts | plain evaluate | plain evaluate, NO negamax/quiescence in rollouts |

---

## Phase 1: Feature Engineering

### 1a. Base encoding: 62 floats (unchanged)

The 62-float base encoding is kept as-is:

```
[0:58]  sentinel move features (build_move_features — unchanged)
[58]    sentinel_score: SentinelAdvisor quality for this move ∈ [0, 1]
[59]    blended_abs: 0.5 * h_abs_norm + 0.5 * vn_abs_norm ∈ [0, 1]
[60]    is_engine_top1: 1.0 if heuristic ranks this move first
[61]    blended_delta: tanh(0.5 * h_delta + 0.5 * vn_delta)
```

Gap net is **not** added to the base encoding. It lives entirely in the lookahead block
(below), which matches the user's framing: "observe sentinel data for that lookahead game"
and "investigate likely outcomes from several paths." The per-position gap signal belongs
in the trajectory observation, not in the static move encoding.

### 1b. Lookahead block: 15 → 60 floats (15 plies × 4 signals)

Add `gap_norm` as the 4th signal at each depth step, alongside sentinel:

```
Layout per depth step: [h_norm, vn_norm, sent_mean, gap_norm]
Total: 15 × 4 = 60 floats (was 5 × 3 = 15)
```

`gap_norm` at each rollout position: `(gap_net(board, current_player) + 1) / 2`,
flipped when it is the opponent's turn (opponent's blunder zone = good for learner), same
convention as `sent_mean`. Defaults to 0.5 when `gap_net=None`.

Sentinel is **enabled** during specialist training (`use_sentinel=True`). This is the
new core of the lookahead: the AI observes both sentinel quality and human-blunder density
across its planned trajectory before committing to a move. At 15 plies, the specialist
sees roughly 7–8 full moves ahead — enough to observe mill threats, captures, and
positional shifts that develop over several turns.

**Files:** `learned_ai/models/lookahead_advisor.py`
- Constructor: add `gap_net=None` param; default `ply_depth=15`
- `feat_dim = ply_depth * 4`  (was `ply_depth * 3`)
- `_record_signals()`: return `(h_norm, vn_norm, sent_mean, gap_norm)` 4-tuple
- `use_sentinel=True` by default for specialists (was `False`)

### 1c. Heuristic rollout rule

**Open question — needs user sign-off before implementation.**

The user said "give the ai the less complicated heuristics; no extended tactical search."
The current rollout move selection calls `evaluate(board, player, strength_mode=True)`,
which includes all terms: mills, mobility, blocked, `_late_game_danger`, endgame score,
cross-block, etc. "Less complicated" likely means a subset of these. Options:

- **Option A (Minimal):** mills + mobility + blocked only — drop `_late_game_danger`,
  endgame score, cross-block terms. Fast, unambiguous signal, best for sparse rollouts.
- **Option B (No late-game extras):** drop `_late_game_danger` and `endgame_score` terms
  from rollout eval but keep the positional terms. Middle ground.
- **Option C (Null change):** keep `evaluate(strength_mode=True)` as-is; "no extended
  tactical search" just means no negamax/quiescence inside lookahead (already the case).

**Recommendation: Option A** — rollouts should be cheap and directional. The sentinel and
gap signals already provide the rich tactical context; the heuristic's job in rollouts is
just to pick a plausible next move, not to be the primary quality signal.

No negamax, no quiescence search, no `tactical_move_bonus` in lookahead regardless of
which option is chosen.

### 1d. Total dimensions

```
MOVE_FEAT_DIM                  = 62   (base encoding, unchanged)
LOOKAHEAD_FEAT_DIM             = 60   (15 plies × 4 signals)
MOVE_FEAT_DIM_WITH_LOOKAHEAD   = 122  (used by specialist networks)
VALUE_INPUT_DIM                = 23   (unchanged)
```

Update `scaffolded_encoder.py` constants accordingly.

---

## Phase 2: Network Architecture

`ScaffoldedPolicyNet` takes `move_feat_dim=83` (was 77).  No architectural changes
to the MLP structure — just wider input.  Three separate checkpoint trees:

```
learned_ai/checkpoints/scaffolded/s_open_v2/
learned_ai/checkpoints/scaffolded/s_mid_v2/
learned_ai/checkpoints/scaffolded/s_end_v2/
```

Checkpoints from v1 (`s_open/`, `s_mid/`, `s_end/`) are incompatible (83 ≠ 77) so all
three specialists train from random init. The overseer checkpoints are no longer needed.

---

## Phase 3: Reward Structure (all three specialists)

Same as opening specialist v1 — proven stable:

```python
ALPHA      = 0.20   # sentinel quality delta (per step)
BETA       = 0.15   # heuristic delta (per step)
MILL_BONUS = 0.20   # per new mill closed (un-gated, all phases)
GAMMA      = 0.0    # Malom win reward — DISABLED
DELTA      = 0.0    # Malom trap reward — DISABLED
VN_BETA    = 0.0    # value-net reward — DISABLED
LAMBDA     = 0.50   # retroactive outcome weight
DECAY      = 0.98   # retro decay per ply remaining
WIN_REWARD  =  1.0
LOSS_REWARD = -1.0
DRAW_SHORT  =  0.15
DRAW_LONG   = -0.05
```

Malom DB is available for rollout termination (early-exit when a solved position is
reached) but does NOT contribute to the reward signal.

---

## Phase 4: Per-Specialist Training

### Opening specialist (`s_open_v2`)
- **Phase gate:** placement phase only (plies 0–17 or until move phase begins)
- **Opponent:** heuristic AI at difficulty 6 (same as current)
- **Book:** `BOOK_GAME_PROB = 1.0` — all games follow opening book lines
- **Script:** `scripts/train_s_open_v2.py` (copy of `train_scaffolded_opening.py`, updated dims)
- **Smoke test:** `--max-games 20` — expect no crashes, policy entropy > 0

### Midgame specialist (`s_mid_v2`)
- **Phase gate:** movement phase, both sides ≥ 6 pieces, not endgame territory
- **Opponent:** heuristic AI at difficulty 5
- **Book:** `BOOK_GAME_PROB = 0.0` (start from diverse placement outcomes)
- **Script:** `scripts/train_s_mid_v2.py` (copy of `train_scaffolded_midgame.py`, updated dims)

### Endgame specialist (`s_end_v2`)
- **Phase gate:** movement/fly phase, either side ≤ 5 pieces
- **Opponent:** heuristic AI at difficulty 4 (endgame is harder to learn; weaker opp = more wins)
- **Malom probe:** probe at each rollout position; if exact WDL found, terminate trajectory
  and assign terminal signal — but still GAMMA=0.0 (no Malom in reward)
- **Script:** `scripts/train_s_end_v2.py` (copy of `train_scaffolded_endgame.py`, updated dims)

---

## Phase 5: Inference Wiring

Remove all references to `OverseerAdvisor` from the inference path.

### Phase router (in `learned_ai/agents/` or `web/app.py`)

```python
phase = get_game_phase(board, board.turn)
piece_counts = board.piece_counts  # or similar

if phase == "place":
    specialist = open_specialist
elif phase in ("move", "fly") and min(own, opp) <= 5:
    specialist = end_specialist
else:
    specialist = mid_specialist
```

### Encoding at inference

Use `encode_position_with_lookahead()` with:
- `sentinel_advisor` = loaded SentinelAdvisor
- `value_net` = TrajectoryValueNet
- `gap_net` = GapNet
- `lookahead_advisor` = LookaheadAdvisor(use_sentinel=True, ply_depth=5)

The specialist's `policy_probs()` is called on the resulting 83-float feat matrix.

---

## Files to Change

| File | Change |
|---|---|
| `learned_ai/models/scaffolded_encoder.py` | LOOKAHEAD_FEAT_DIM=20, MOVE_FEAT_DIM_WITH_LOOKAHEAD=82; base (MOVE_FEAT_DIM) unchanged at 62 |
| `learned_ai/models/lookahead_advisor.py` | feat_dim=ply×4, add gap_norm signal, gap_net param, use_sentinel=True default |
| `scripts/train_s_open_v2.py` | New file; copy + update dims, checkpoint dir, smoke test |
| `scripts/train_s_mid_v2.py` | New file; copy + update dims, checkpoint dir |
| `scripts/train_s_end_v2.py` | New file; copy + update dims, checkpoint dir |
| `learned_ai/agents/` (phase router) | Route by phase directly to specialist; remove Overseer |
| `web/app.py` (if wired) | Remove OverseerAdvisor load; use phase router |

## Files to Retire (not delete — keep for reference)

- `learned_ai/models/overseer.py`
- `learned_ai/models/overseer_extras.py`
- `scripts/train_scaffolded_overseer.py`
- `scripts/train_scaffolded_overseer_parallel.py`

---

## Sign-off Checkpoints

- [ ] **User approves this plan** before any code changes
- [ ] Feature dimension changes (`scaffolded_encoder.py`, `lookahead_advisor.py`) reviewed
- [ ] Opening specialist smoke test (20 games) passes with `MOVE_FEAT_DIM_WITH_LOOKAHEAD=83`
- [ ] Opening specialist 500-game run: win rate > 40% vs difficulty-6 heuristic
- [ ] All three training scripts run independently without crashes
- [ ] Inference wiring tested end-to-end in the web app (each phase routes correctly)
- [ ] User approves before promoting any specialist to default in-game AI

---

## Future Exploration (if capacity + raw-board changes don't crack the plateau)

Applied 2026-07-15: policy net grew to `(512, 256, 128)` hidden and value net to `(256, 128, 64)`; value input extended by 48 raw-board one-hot floats (24 positions × 2 colors), for 80-float value input.

If a fresh run at this scale still stalls in the low-single-digit difficulty range, the next levers to try — in order of increasing effort:

### 3. Attention over legal moves (moderate ambition)

Replace the independent per-move MLP scoring with a small self-attention block across the k legal moves. Rationale: current scoring treats each move independently — the model can't reason "this move is best *given the alternatives*". Attention lets each move's logit condition on the whole candidate set (like how humans compare candidate moves side-by-side before deciding). Roughly 500k params for a modest 2-layer transformer with 4 heads.

Implementation sketch:
- Keep the current per-move 122-float encoding as input tokens.
- Add a learned board-context token (from the 80-float value input) as the first sequence position.
- Self-attention over `(1 + k)` tokens, then take output logits from each move token.
- Replace `ScaffoldedPolicyNet.policy_logits()`; value head stays MLP.

Risk: harder to train, more sample-inefficient, may overfit on small training sets. Only worth trying if bigger-MLP + raw-board didn't move the needle.

### 4. Different RL algorithm (highest ambition, last resort)

The current setup is vanilla A2C / PPO with sparse per-game outcome rewards over 15-60 ply trajectories. That's a hard credit-assignment problem no matter how big the model is. If the specialists still stall after 1-3, the problem is upstream from architecture — the algorithm can't propagate signal effectively.

Options:
- **MuZero-style**: learned world model + MCTS at inference (and possibly at training via search-based targets). Roughly matches how DeepMind's chess/go engines learned. Requires substantial re-plumbing: dynamics network, prediction network, MCTS integration. Multi-week effort.
- **AlphaZero-style search-augmented policy gradient**: run a shallow MCTS at each training-time decision, use visit counts as the policy target. Simpler than full MuZero but still a rewrite of the rollout loop.
- **Distillation from a search engine**: take the existing sentinel + endgame DB + heuristic ensemble, run deep MCTS at each position, and supervise-train the specialist to mimic the search output. Bypasses the RL credit-assignment problem entirely by turning it into supervised learning. Feasibility depends on how much compute you can spend on MCTS.

Do not undertake 4 without first exhausting 3. The A2C setup can still work if capacity + features are right; algorithmic complexity should be a last resort.

---

## Update — v3 Specialist Design (2026-07-16)

Concrete design change after diagnosing that the specialists have never actually seen the classical AI's alpha-beta search output. Approved by the user in the same conversation.

### Diagnosis (current state, pre-v3)

- **Specialist inference takes only ~1–2 s** per move, regardless of difficulty.
- **`LookaheadAdvisor` simulates 15 half-plies using `_static_best_move`** — a depth-1 static heuristic pick per side. It is NOT the alpha-beta search the classical engine runs; the specialist has never seen a real search-derived candidate ordering.
- **In training AND inference**, the specialist's per-move features (`build_move_features`) are the sentinel/heuristic-eval/counterfactual block — no alpha-beta root scores.
- **Gap-net leaf correction is applied on AI-side leaves only** (`ai/game_ai.py:1827-1840`). The Sanmill developer's critique is correct: the gap net predicts *human* blunder density, so the bonus arguably belongs on **opponent-side leaves** ("set traps"), not on our own.
- **VN and gap net short-circuit each other at the leaf.** In `ai/game_ai.py:1815-1840` the VN branch returns before the gap branch can fire, so gap correction only runs when the VN is inactive. In mid/endgame (≥10 pieces on board, VN active) we get VN only. This was under-documented and looks unintentional.

### Design change (approved)

**1. Specialist features are search-informed, not blind.**

At every position (both training and inference):

- Call `GameAI.score_root_moves(board, depth, time_budget)` at the **same per-difficulty time budget the classical engine uses in real play** (15/30/45/60 s + early-placement reductions). This is what "specialist gets the same heuristic search capability" means — literally re-use the engine's alpha-beta search output.
- Take the **top-5** search-scored candidates.
- For each of the 5 candidates, compute the standard feature block: sentinel score, VN eval, gap score, 15-ply lookahead trajectory (kept — but only for the 5 candidates, not all `k` legal moves), plus one **new** field: **`traj_freq`** — the fraction of similar historical-human games in which this move was played (from `_effective_tdb.query_all_frequencies(board)`, same source as the trajectory overlay).
- Specialist input becomes `(5, F)` where `F` = 62 base + 60 lookahead + AB score + trajectory-frequency + trajectory-rank + AB-rank ≈ ~128 floats.

**2. Specialist output = confidence to re-order.**

Rather than "score every legal move from scratch", the specialist outputs a **re-ranking distribution** over the top-5. Its target during training: pick the candidate with the highest actual outcome, given a small preference to defer to alpha-beta's #1 unless the lookahead / gap / trajectory signals contradict it strongly. This is a much smaller learning problem than picking-from-scratch and matches how humans play (compare a small set of candidates deeply).

**3. Trajectory-frequency + n-gram human-prior features are per-candidate.**

For each of the 5 candidates the specialist sees the expected *human* preference for that move — from two complementary sources:

- **Position-based**: `HumanDB.query_all_frequencies(board)` or `TrajectoryDB.query_all_frequencies(board)` returns `{notation: freq}` — the proportion of ~22 k human games where each move was played from this position. Position-only, no game history needed.
- **Sequence-based fallback**: when the current position has fewer than the trajectory DB's `min_samples` (5), `NGramOpponentModel.predict(color, game_notations)` uses the last 1-2 same-color moves as context to give a bigram/trigram probability. Requires `game_notations` (the alternating move-notation list), which the training loop can accumulate per rollout.

Per candidate the encoder attaches: **`human_freq`** (raw probability 0-1 from whichever source hit) and **`human_rank`** (normalised rank: 1.0 for the most-played, 0.5 for the 3rd, 0.0 for not-in-DB). The specialist learns to weight "play what humans commonly play" against "play what the engine says is optimal, per α-β + sentinel + gap + VN". This is the edge against average human opponents — hedge to match human expectations, but deviate when the classical/lookahead signals strongly justify it.

Signal precedence (both training and inference):
1. `HumanDB` (highest-quality — ELO/win-rate stratified)
2. `TrajectoryDB` (self-play + human) if HumanDB coverage fails
3. `NGramOpponentModel` if position coverage fails (needs `game_notations`)
4. Zero-vector fallback (feature attenuates, no bias introduced)

**4. Time budget for the specialist in real games.**

Specialist total wall time per move = classical alpha-beta search time (same per-diff cap the game already uses) + top-5 lookahead overhead (~5 × ~150 ms = <1 s) + one policy forward pass. Roughly *heuristic time + 1 s*, not 1–2 s total.

**5. In-game routing (Overseer toggle).**

Keep the current design: coordinator runs first, specialist re-scores afterwards. Do **not** skip the coordinator — its side effects (trajectory recording, sentinel calibration state) matter. If `SpecialistRouter.score_moves()` raises, **print the exception to the terminal** and fall back to the coordinator's move. Never silently swallow.

### Advancement — Sanmill-style superiority probability

Replace the current `(wins + 0.5×draws)/total ≥ threshold` + draw-cap system with the standard chess-testing formula (Sanmill `head_to_head.rs:873–1020`):

```
p   = (W + 0.5·D) / (W + D + L)
SE  = sqrt( p·(1-p) / n )
z   = (target − p) / SE
P(true score > target) = 1 − Φ(z)      # Φ = standard-normal CDF (Abramowitz–Stegun 26.2.17)
```

Rules:

- **Advance** when `P(true score > target%) ≥ 0.95`.
- **Target ramps linearly**: 55% at level 1 → 60% at level 20.
- **Time-of-flight relaxation**: after 1,000 games at a level without advancement, drop the effective target 1% per 1,000 additional games until floor = 51%. This is mathematically equivalent to "the SE is now tight enough that even a small edge is confidently better than 50%", and it matches the user's intuition that a specialist repeatedly plateauing at slightly-above-50 deserves to move on.
- **Recovery**: if `P(true score > 45%) < 0.05` (we are confidently *worse* than 45%) over the rolling window, reload `best{difficulty}.pt` — same recovery pattern as today but statistically justified.

Applies to opening, midgame, and endgame trainers identically. Draws still count as ½ but the ½ credit is now inside `p`, so a 60% draw + 20% win + 20% loss run gives `p = 0.5` and `P(true > 55%) ≈ 0` — cannot advance. That is the correct behaviour.

### Gap-net fixes (both the game engine and training)

**Game engine (`ai/game_ai.py:1815-1840`)**:

1. **Un-short-circuit VN vs gap.** Apply gap correction to the *blended* value, not skipped:

    ```python
    if self._vn_active(board):
        blended = int(blend * vn_score + (1.0 - blend) * heur)
    else:
        blended = heur
    if self.use_gap_net and self._gap_net is not None and (blend caps > 0):
        blended = human_correction(board, board.turn, blended, self._gap_net, self._weights)
    return blended
    ```

2. **Gap-net leaf-side ablation.** Run a **4-way 200-game bench** at diff 5:
    - A = current (AI-side leaves) — baseline
    - B = opponent-side leaves — the "set traps" hypothesis raised by the Sanmill developer
    - C = both sides — additive
    - D = off — control
    Score = `(W + 0.5·D)/n` vs raw heuristic at same time budget. Report all four in the plan doc regardless of outcome. If B or C beat A convincingly, update the default.

**Training (`LookaheadAdvisor._record_signals`)**: The gap_norm signal is currently computed for `current_player` and flipped when it's the opponent's turn (so the value always expresses learner-favourability). This is different from the game engine's AI-side-only application. Bring both into line with whatever wins the ablation.

### Value-net verification (already-completed audit)

Current wiring in `ai/game_ai.py`:

- **Root-move ordering blend** (`_apply_vn_blend`, line 1514): active when board has ≥10 pieces on it. Blend weight from `weights.value_net_blend / 100`. Correct.
- **Leaf blend** (line 1822-1826): symmetric, blend weight same as root. Correct in isolation, but see gap-net short-circuit fix above.
- **Late-move VN prune** (line 1973+): depth ≥3, ≥2 moves. Correct.
- **`_vn_active` gate** (line 1508): also gates via `weights.value_net_blend > 0`. Correct — passing `value_net_blend=0` correctly disables it.

VN itself is used **correctly** per the current design. The only problem is the interaction with gap net at the leaf, fixed above.

### Files to change (v3)

| File | Change |
|---|---|
| `ai/game_ai.py` (line 1815-1840) | Un-short-circuit VN vs gap: apply gap correction after VN blend, not either-or. |
| `learned_ai/models/lookahead_advisor.py` | Extend `_simulate_trajectory` to optionally take an alpha-beta scored candidate list (top-5) and a per-candidate trajectory frequency, so only those 5 get simulated. |
| `learned_ai/models/scaffolded_encoder.py` | Add a new encoder path `encode_top_k_candidates(board, player, gameai, top_k=5, ...)` that returns `(5, F)` features per candidate — reuses base 62-float row for each candidate, prepends the alpha-beta score + rank + trajectory-frequency + rank. |
| `learned_ai/models/scaffolded_net.py` | Optional: shrink `policy_hidden` since input is now (5, ~128) not (k, 122). 128k parameters is plenty for a 5-choice re-ranker. |
| `scripts/train_s_open_v2.py`, `train_s_mid_v2.py`, `train_s_end_v2.py` | 1) Replace `encode_position_with_lookahead(...)` at learner-decide sites with the new `encode_top_k_candidates(...)`. 2) Pass `GameAI` (from the opponent's factory or a new shared one) into the encoder. 3) Replace `_check_advance` with the Sanmill superiority-probability formula (imported from a new shared helper `learned_ai/training/advance_stats.py`). 4) Time-of-flight relaxation state kept per-level in the training loop. |
| `learned_ai/training/advance_stats.py` (new) | Sanmill Score% + superiority probability + Φ implementation. Shared by all three trainers. |
| `learned_ai/agents/specialist_router.py` | Update `score_moves` to build top-5-candidate features via the new encoder path. |
| `web/app.py` (line 2902-2913) | On `SpecialistRouter.score_moves` exception, `print(f"[Overseer player] specialist failed: {exc}", file=sys.stderr)` before falling back to the coordinator's move (do NOT silently `log.debug`). |
| `scripts/bench_gap_leaf.py` (new) | Four-way 200-game ablation vs raw heuristic at diff 5. Reports Score% + P(true > 50%) per config. |

### Sign-off checkpoints (v3)

- [ ] Advisor sanity check on this plan (user-directed).
- [ ] Feature encoder change reviewed on a single-position unit test: correct alpha-beta scores land in the top-5 rows, correct trajectory frequencies attached.
- [ ] Advancement helper unit test: reproduces Sanmill's reference values on a 500-game window.
- [ ] Gap-net leaf ablation bench run; results appended to this doc under a "Gap ablation" section.
- [ ] Fresh training run from scratch (all three specialists in parallel via `--batch-games`) to at least diff 5 with the new features.
- [ ] User approves before the router auto-loads a v3 checkpoint over the v2 default.

### What is explicitly NOT changing (per user)

- Bigger-MLP scaffolded net capacity (kept).
- Raw-board features in value input (kept).
- Explore bonus in `_retroactive_rescore` (kept — Option A).
- Malom-agreement bonus for endgame (kept).
- Frozen-model learner-side lookahead (kept — Option C).
- Overseer toggle in-game behaviour: coordinator runs first, specialist re-scores (kept).
- The classical heuristic AI's algorithm otherwise — only the VN/gap short-circuit is being fixed.

### Known gaps / possible improvements (deferred, 2026-07-17)

Noted during v3 wiring; leaving as-is for now per user, but worth revisiting if the specialists still stall after the current training run.

- **Gap net is not wired into either the specialist's `learner_gameai` or the opponent `GameAI` during training.** Gap net is currently used only as a *passive feature* (the `gap_norm` signal inside the 15-ply lookahead block, and as feature bits in the base 62-float per-move row). At α-β leaf time in the training-side searches — both the specialist's own top-K candidate scoring and the opponent's move choice — gap net is absent. This means:
  * The specialist's top-K candidates during training are the ones a *non-gap-aware* engine would surface, not the ones the deployed game AI (which uses gap net at AI-side leaves per V3a) would surface.  The specialist re-ranks a slightly different candidate set at inference than it saw in training.
  * The training-time opponent is weaker than the deployed classical AI, since it lacks the gap-net trap-setting bonus.
  * Fix if adopted: pass `gap_net=gap_net` to both `_GA(...)` constructions in each training script's `_rollout`.  Roughly a 3-line change per file.  Also decide which leaf-side mode (`ai_side` / `opp_side` / `both`) each side should use, which is exactly what the gap-leaf ablation bench (`scripts/bench_gap_leaf.py`) exists to answer.
- **Depends on the gap-leaf ablation outcome.** Whatever mode wins the ablation should propagate consistently: (a) the classical AI's `gap_net_leaf_mode` in real play, (b) the learner's `gap_net_leaf_mode` in training, (c) how the LookaheadAdvisor's `gap_norm` signal is flipped by `current_player == learner_color`.  All three currently follow independent conventions — worth aligning once the ablation resolves the direction.
