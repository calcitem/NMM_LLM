# HeuristicsV3a — Exploiting Human Play Patterns
## Minimum Viable Implementation Plan

**Status:** Design complete, not yet implemented  
**Goal:** Make the AI steer game positions toward squares and formations that human players
mishandle — not just play objectively well, but play in ways that actively exploit human tendencies

---

## What We're Actually Trying to Do

Right now the AI plays near-optimally from a classical game-theory standpoint. It finds strong
moves. But it treats the opponent as if they will also play near-optimally. Against a human,
that's wrong — humans have consistent weaknesses. They mismanage piece mobility in the move
phase. They overlook cross-formation threats in placement. They react to the last threat
instead of tracking the whole position.

The goal of V3 is to make the AI's search aware of these tendencies. Specifically: when the
engine is evaluating candidate lines several moves deep, it should assign extra value to
positions where a human will likely err — not just positions that are objectively better.

This is the same technique Stockfish uses with its "optimism" term. Stockfish doesn't just
find the best move; it nudges its evaluation toward positions that are not just winning, but
winning in ways that are easy to convert against a non-optimal opponent.

---

## What We Already Have That Partially Works

The **sentinel** is already doing a version of this, but only at the last step. After the
engine picks a move, the sentinel scores all candidates by move quality and can redirect the
choice if it spots a better option. This catches cases where the engine found a technically
fine move but missed a more exploitative one.

The gap is: the sentinel runs *after* the search finishes. It can only reshuffle the top
candidates. It cannot steer the search to *discover* exploitative lines in the first place.
If a juicy trap 6 moves deep only shows up when the engine values the intermediate positions
correctly, the sentinel never sees it.

The V3 leaf correction fixes this by changing what positions look valuable *during* the
search, not just at the end.

---

## The Core Bottleneck: The Value Net Has the Wrong Training Signal

The plan is to add a correction term `H` to the classical leaf score — a learned signal that
says "this position is one where humans tend to go wrong." We already have a value net (`ValueNet`
in `ai/value_net.py`) that could provide this signal.

The problem: the current value net was trained on human game **outcomes**. It predicts "did the
human win from this position?" That is correlated with position quality, but it is not the same
as "does the human blunder here?" Consider:

- A clearly winning position — humans win it 90% of the time because it's easy to close out.
  The current VN scores this high. But because it's easy, the opponent rarely blunders. There's
  nothing to exploit.
- A tricky mid-game formation with a hidden cross-threat — humans win it 55% of the time because
  they often miss the threat. The current VN scores this only slightly above average. But this
  is exactly the kind of position the AI should be steering toward.

The current VN conflates "strong position" with "human error zone." They overlap, but they're
not the same thing.

**The right signal already exists in this project.** The sentinel, when it evaluates a position,
computes an `opportunity_gap`: the difference between the best available move quality and what
the human actually played. A large gap means the human blundered. Near-zero means they found
a good move. This gap, aggregated across 642,000 positions in the human DB, is exactly the
training target we need. We just haven't used it yet.

Retraining the value net to predict `-opportunity_gap` instead of game outcome turns it from
a "win probability estimator" into a "blunder density estimator" — which is what the correction
formula actually needs.

---

## The Plan: Three Steps in Priority Order

### Stopgap (zero coding — do this now)

Raise the sentinel's `score_adjust_scale` from `0.05` to `0.15` in
`learned_ai/sentinel/config.py`. This makes the post-selection exploitation more aggressive
immediately, while the longer work runs.

This is a one-line config change. It does not require a restart of the understanding of the
system. It makes the existing sentinel push harder on move redirects when it spots an opportunity.

---

### Step 1: Retrain the Value Net on Blunder Signal (most important)

Write `scripts/build_residual_dataset.py` and run it. This script:

1. Opens the human DB (642K positions)
2. For each position, fetches all human-played moves
3. Runs `sentinel.advise()` to get the opportunity gap for each position
4. Saves `(board_features, -opportunity_gap)` pairs to a numpy file

Then change **one line** in `tools/train_value_net.py`: swap the label from game outcome
to `-opportunity_gap`. Retrain. The architecture does not change. The output file is still
`data/value_net.npz`. Nothing else in the system needs to change to pick up the new weights.

This step is mostly computation, not design. The sentinel runs in ~2ms per position, so
processing 642K positions takes roughly 20 minutes. Training is another 10–15 minutes.

After this step, `H_scaled` (the value net output scaled to heuristic units) actually means
what the correction formula assumes it means: "how much does the human tend to blunder here."

---

### Step 2: Wire in the Leaf Correction (minimum viable, ~65 lines)

Add `human_correction()` to `ai/heuristics.py` and wire it into `_negamax` in `ai/game_ai.py`.

The formula:
```
E_final = E_v2 + γ · C · (H_scaled - E_v2)
```

- `E_v2` is the classical leaf score already computed
- `H_scaled` is the value net output × 3000 (to match heuristic units)
- `γ` is a phase-specific cap: 12% in placement, 20% in move phase, 5% in fly phase
- `C` is a complexity gate that suppresses the correction in sharp/tactical positions where
  the classical eval should be trusted more than the learned signal

This is only applied when it's the AI's turn at the leaf (`board.turn == self.color`).
Opponent continuation nodes use the pure classical eval, which keeps the search tree
internally consistent. This asymmetry is intentional and is how the Stockfish optimism
term works.

A per-search leaf cache (by Zobrist hash) prevents the value net from being called thousands
of times per move. The Zobrist hash already encodes side-to-move (`SIDE_KEY` in
`game/zobrist.py`), so cache hits are always valid.

---

## What We're Deliberately Skipping

**Opponent profiles (strong_human / novice_human / perfect):** Not implementing yet. The
average_human defaults (12/20/5%) are sufficient for the first version. Add profiles later
if you want the AI to adjust its exploitation style based on the opponent's known strength.

**Tactical sharpness gate (Stage 5 in the original V3 doc):** The complexity gate `C` already
handles most of this. The sharpness gate adds an additional suppression in positions with
immediate mill threats. Skip for now — add it if you observe the correction overriding correct
tactical responses.

**The `_apply_vn_blend` root ordering update:** Do not change `_apply_vn_blend`. That function
applies corrections to full subtree scores (not leaf evals), which breaks the formula. Root
ordering is already implicitly handled because corrected leaf values propagate up through the
transposition table. Leave it alone.

---

## Implementation Notes for Claude

This section contains the exact changes needed for Steps 1 and 2. Read the sections above
first to understand why — don't implement without understanding the intent.

---

### Stopgap: raise sentinel scale

File: `learned_ai/sentinel/config.py` line ~43  
Change: `score_adjust_scale: float = 0.05` → `score_adjust_scale: float = 0.15`

---

### Step 1a: Build residual dataset

Create `scripts/build_residual_dataset.py`. Key points:

- Import `HumanDB` from `ai/human_db.py` and open `data/human_db.sqlite`
- Import `SentinelAdvisor` from `learned_ai/sentinel/infer.py`; load from `learned_ai/sentinel/best.pt`
- Import `board_to_features`, `_INPUT_DIM` from `ai/value_net.py`
- Iterate positions: query `SELECT state_key, fen FROM positions` directly on the SQLite connection
  (HumanDB doesn't expose a bulk iterator; use `db._conn.execute(...)`)
- For each position FEN, reconstruct a `BoardState` via `BoardState.from_fen(fen)` (or use the
  `fen_to_board()` helper already in `tools/train_value_net.py`)
- Get the human-played moves for that position via `db.query_moves(board)` which returns
  `list[MoveStats]` — each has `.notation`, `.wins`, `.losses`, `.total`
- For each move, build a candidate list of all legal moves, find the index of this notation,
  call `sentinel.advise(board, move_dict, all_candidates)` → `SentinelAdvice`
- Record `y = -advice.opportunity_gap` (clamp to `[-1.0, 0.0]` since humans can't consistently
  beat optimal)
- Average `y` across all human plays at that position (same deduplication logic as
  `extract_samples()` in `train_value_net.py`)
- Save: `np.savez("data/value_net_residual.npz", X=X, y=y)`

The sentinel's `advise()` call needs a `played_move` (a move dict, not just notation) and
`candidates` (all legal moves as dicts). Use `board.legal_moves()` for candidates and parse the
notation back to a move dict using the move notation format in the DB.

---

### Step 1b: Retrain value net on residual target

Modify `tools/train_value_net.py`:

1. Add `--residual` flag to argparse
2. When `--residual` is set, load `data/value_net_residual.npz` instead of calling
   `extract_samples()`
3. The training loop is otherwise identical — same network, same MSE loss, same save path
   (`data/value_net.npz`)
4. Consider training for more epochs (50–100) since the residual signal is noisier than
   outcome labels

Run: `.venv/bin/python tools/train_value_net.py --residual --epochs 80`

Back up `data/value_net.npz` before running — the new weights will overwrite it.

---

### Step 2a: Add `human_correction()` to `ai/heuristics.py`

Add these fields to `HeuristicWeights` (find the `value_net_blend` field, add after it):

```python
# V3a: asymmetric human-opponent correction
vnet_blend_place: int = 12   # % correction cap in placement phase
vnet_blend_move:  int = 20   # % correction cap in move phase
vnet_blend_fly:   int = 5    # % correction cap in fly phase
vnet_gate_place:  int = 200  # complexity gate denominator for placement
vnet_gate_move:   int = 500  # complexity gate denominator for move/fly
vnet_gate_fly:    int = 500
```

Add this function after `evaluate_v2()`:

```python
def human_correction(
    board: "BoardState",
    color: str,
    e_v2: int,
    value_net,
    weights: "HeuristicWeights | None" = None,
    *,
    _profile_cap: float = 1.0,
) -> int:
    """Apply asymmetric blunder-zone correction to an already-computed E_v2 score.

    Only called from _negamax when board.turn == self.color (the AI's side).
    Returns e_v2 unchanged when value_net is None or all blend caps are 0.

    Formula: E_v2 + γ · C · (H_scaled - E_v2)
      γ = min(phase_cap / 100, _profile_cap)
      C = 1 / (1 + |E_v2 - H_scaled| / gate_denom)   — suppressed in sharp positions
      H_scaled = value_net.predict(board, color) * 3000
    """
    if value_net is None:
        return e_v2

    w = weights if weights is not None else DEFAULT_WEIGHTS
    phase = get_game_phase(board, color)

    _blend_map = {"place": w.vnet_blend_place, "move": w.vnet_blend_move, "fly": w.vnet_blend_fly}
    cap_pct = _blend_map[phase]
    if cap_pct <= 0:
        return e_v2

    gamma = min(cap_pct / 100.0, _profile_cap)
    if gamma <= 0:
        return e_v2

    h_scaled = int(value_net.predict(board, color) * 3000)

    _gate_map = {"place": w.vnet_gate_place, "move": w.vnet_gate_move, "fly": w.vnet_gate_fly}
    gate_denom = max(1, _gate_map[phase])

    C = 1.0 / (1.0 + abs(e_v2 - h_scaled) / gate_denom)

    # Fly phase: near-solvable; human model is least informative here
    if phase == "fly":
        C = min(C, 0.3)

    return e_v2 + int(gamma * C * (h_scaled - e_v2))
```

---

### Step 2b: Add leaf cache to `GameAI.__init__` (`ai/game_ai.py`)

Find `__init__` (around line 475), add alongside `self._tt`:

```python
self._vn_leaf_cache: dict[int, int] = {}   # hash_key → corrected heur; cleared per get_move()
```

Find `get_move()` or `_iterative_deepen()` — wherever `self._tt.clear()` is called at the
start of a new move search. Add `self._vn_leaf_cache.clear()` on the same line or immediately
after.

---

### Step 2c: Replace B-73 leaf block in `_negamax` (`ai/game_ai.py`)

Find lines 1810–1816 (the B-73 block):

```python
# B-73: blend in value network score when loaded and blend > 0
if self._value_net is not None and self._weights.value_net_blend > 0:
    vn_raw = self._value_net.predict(board, board.turn)  # (-1, 1)
    vn_score = int(vn_raw * _VN_SCALE)
    blend = self._weights.value_net_blend / 100.0
    return int(blend * vn_score + (1.0 - blend) * heur)
return heur
```

Replace with:

```python
# V3a: asymmetric human blunder-zone correction (applied only on AI's side)
if (board.turn == self.color
        and self._value_net is not None
        and (self._weights.vnet_blend_move > 0
             or self._weights.vnet_blend_place > 0
             or self._weights.vnet_blend_fly > 0)):
    _cached = self._vn_leaf_cache.get(board.hash_key)
    if _cached is not None:
        return _cached
    from .heuristics import human_correction
    corrected = human_correction(board, board.turn, heur, self._value_net, self._weights)
    self._vn_leaf_cache[board.hash_key] = corrected
    return corrected
return heur
```

Note: the old `value_net_blend` field (B-73 symmetric blend) is now inactive. Set it to 0 in
settings if it was previously non-zero. The new per-phase fields `vnet_blend_place/move/fly`
in `HeuristicWeights` control the correction.

---

## Verification Tests After Implementation

1. **Zero-impact default test:** With all `vnet_blend_*` fields at 0, `human_correction()`
   must return `e_v2` unchanged. The AI should play identically to V2 baseline.

2. **Think-time benchmark:** Run 10 moves with V3a enabled and compare to V2. V3a with the
   leaf cache should add less than 20% to average think time. If it's higher, check that
   `_vn_leaf_cache.clear()` is being called and not just between games.

3. **Blunder exploitation test:** After enabling V3a, play a game and check that the AI
   is choosing lines into positions flagged as high-opportunity-gap by the sentinel's
   post-move analysis. If it's steering correctly, the sentinel should report fewer
   "missed opportunity" flags per game (because the search already found the exploitative line).

---

## The Complexity Gate: How It Behaves

The gate `C` suppresses the correction when the classical eval and the value net strongly
disagree. This is important: in sharp tactical positions (forced captures, immediate mill
threats), the classical eval is reliable and the learned signal may be noisy. We don't want
the correction overriding a forced capture.

Gate behaviour in move/fly phase (denominator D=500):

| Disagreement between E_v2 and H | Gate C | Effect |
|---|---|---|
| 0 (full agreement) | 1.00 | Full correction applied |
| 250 | 0.67 | Mild suppression |
| 500 | 0.50 | Half correction |
| 1500 | 0.25 | Sharp position — quarter correction |
| 3000 | 0.14 | Forced line — almost no correction |

In placement phase (denominator D=200), the gate tightens faster because placement scores are
numerically smaller (±100–250 vs ±1000–2000 in move phase). This keeps the correction
proportional to the scale of the eval in each phase.

---

## What This Looks Like in Practice

Before V3a: the engine finds the theoretically strongest continuation. Against a strong player
this is correct. Against a human it often means the engine enters a quiet, slightly-better
endgame that a human can navigate.

After V3a: the engine still avoids blunders. But when two lines are similarly valued by the
classical eval, the one that leads to a formation humans routinely mishandle gets a small boost.
Over time, this should manifest as the AI preferring:
- Cross-threats over simple direct mills (humans miss cross-threats more)
- Move-phase formations where mobility asymmetry develops gradually (humans underestimate mobility
  restriction until it's too late)
- Placement sequences that create convergent threats the human can't block simultaneously

The γ caps (12/20/5%) keep the correction small enough that it never overrides a genuine tactical
advantage. It's a tiebreaker, not a replacement.
