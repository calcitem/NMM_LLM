# HeuristicsV4 — Search Performance Improvements
## Plan: Rust Engine Optimisations + Python Fallback Acceleration

**Status:** Planning — no code changes made yet  
**Goal:** Reduce node count in the Rust search, improve placement-phase move ordering, and accelerate the Python fallback path when the Rust extension is unavailable

---

## Background

A profiling session (July 2026) produced two outcomes:

1. **A critical regression was identified and fixed** (`_time_ms` bug — see V4-A). This was the proximate cause of "1 ply in 30–60 seconds" behaviour.

2. **A comprehensive investigation of NMM search optimisations** was completed, cross-referencing Sanmill (the leading open-source NMM engine), MTD(f) literature, and our own profile data. The remaining items in this document are the prioritised findings.

---

## What the Rust Engine Already Has

As a reference baseline, the Rust search (`native/nmm_core/src/search.rs`) already implements:

- Iterative deepening with aspiration windows (±50)
- PVS (Principal Variation Search) — null window on non-first moves, re-search on fail
- LMR (Late Move Reduction) — late non-tactical moves searched at `depth-1` first
- Killer moves (2 per ply, MAX_PLY = 64)
- History heuristic (25×24 table)
- Countermove table
- Transposition table (Zobrist hashing, EXACT/LOWER/UPPER bounds, mate-score normalisation)
- Null-move pruning (R=2, skipped in fly phase and with own ≤ 3 pieces)
- SE-11b opponent extension (+1 ply for high-frequency trajectory moves at first opponent ply)
- FGOP pruning (frequency-gated opponent pruning at depth ≤ 5)
- Qsearch with forced captures, mill closures, reachable two-config creators, and forced blocks (QS_FORCING_CAP = 6 extra forcing plies)
- B-64 dead/near-dead placement penalty at root
- FullGame DB probe, EndgameSolvedDB probe (O(1) WDL for endgame positions)

**This is already more comprehensive than Sanmill's search in most respects.** The items below are additive refinements, not fixes to a weak engine.

---

## V4-A: `_time_ms` Regression Fix (Deployed)

**Status: Already fixed. Documented here for posterity.**

### The Bug

When `_override_time_budget` is `None` (all normal gameplay), Python computed:

```python
_time_ms = int(min(_time_cap, 3600.0) * 1000)   # _time_cap = inf
# → int(min(inf, 3600.0) * 1000) = 3,600,000 ms
```

This passed a 1-hour budget to Rust. Rust set its internal `Instant` deadline to `now + 1 hour` and ran iterative deepening until it completed every depth — for depth 12 that takes many minutes.

`force_stop()` only sets Python flags (`_deadline = 0.0`, `_force_stop = True`). It has **no path to interrupt the blocked Rust thread** in `asyncio.to_thread`. The user's "force move" button would set those flags, but Rust never checked them.

### The Fix

```python
_time_ms: "int | None" = (
    int(min(_time_cap, 3600.0) * 1000) if self._override_time_budget is not None
    else None
)
```

When `time_limit_ms=None`, Rust uses `min(300_000, max_depth² × 300ms)`. For depth 12: `min(300_000, 43_200) = 43.2s`. With iterative deepening the engine returns its best-so-far result if the next depth overruns — so in practice it completes several depths quickly and returns well before the cap.

### Files Changed

- `ai/game_ai.py` — `_time_ms` computation (~line 1339)

---

## V4-B: MTD(f) in Rust Iterative Deepening (High Priority)

### What and Why

MTD(f) (Memory-enhanced Test Driver) is an alpha-beta variant that uses zero-window `(f-1, f)` searches repeatedly until the exact minimax value is found. Zero-window searches hit cutoffs sooner than wide-window searches — every non-PV node fails high or low immediately — so fewer nodes are expanded per depth.

Sanmill implements MTD(f) alongside PVS. In Chinook (checkers, similar branching factor to NMM) MTD(f) "outperformed all other search algorithms" — typically 15–35% node reduction over plain alpha-beta.

MTD(f) requires two things, both of which we already have:

1. **A transposition table** — to reuse bounds between the repeated zero-window passes at the same depth. Without a TT, MTD(f) re-expands the whole tree every pass and is slower than plain alpha-beta.
2. **A good initial f-guess** — to minimise the number of passes needed. We use the previous depth's score (already available from iterative deepening).

### Where It Applies

The `iterative_deepening` function in `search.rs` (line ~600) loops over depths `d = 1..=max_depth`. For each depth it calls `searcher.root(board, d, a_init, b_init)` with aspiration windows.

MTD(f) replaces the aspiration window retry with a convergence loop per depth:

```
mtdf(board, depth, f):
    lower = -INF
    upper = +INF
    while lower < upper:
        beta = max(f, lower + 1)
        f = root(board, depth, beta - 1, beta)   // zero-window search
        if f < beta:
            upper = f
        else:
            lower = f
    return f
```

The TT from depth `d-1` ensures most nodes resolve immediately on the first pass. Two to three passes is typical; a near-exact f-guess from the previous iteration often converges in one.

### Scope Constraint: `root_scored` vs `root`

There are two root functions:

- `root()` — returns the single best move. Used by `iterative_deepening` → `py_get_best_move` / `py_search_stats`. This is the natural target for MTD(f).
- `root_scored()` — returns an exact score for every move. Used by `iterative_deepening_scored` → `py_search_root_scored` → main gameplay. Each move gets an independent `(-INF, +INF)` window so we get a full ranking.

**MTD(f) applies cleanly to `root()`/`iterative_deepening`.** For `root_scored`, MTD(f) does not directly help (we cannot prune between root moves since we need all scores). However, the TT is shared, so running an MTD(f) pass first would warm the TT entries for the subsequent `root_scored` pass — yielding indirect benefit through better cutoffs inside each move's subtree.

### Implementation Plan

1. Add `fn mtdf()` to the `Searcher` impl in `search.rs`:
   - Inputs: `board`, `depth`, `f: i64`
   - Loop calling `self.root(board, depth, beta-1, beta)` until `lower >= upper`
   - Return the converged minimax value and the best move from the last pass

2. In `iterative_deepening`, replace the aspiration window block with:
   - Depth 1: full-window root call (no useful f yet)
   - Depth 2+: `mtdf(board, d, last_score)` where `last_score` is the previous depth's value

3. In `iterative_deepening_scored`: optionally run a single MTD(f) pass first (to warm TT) then call `root_scored`. The TT warm-up cost is low since MTD(f) at the current depth will re-expand fewer nodes than a cold root search.

### Expected Gain

- 15–35% node reduction in the `iterative_deepening` path
- Indirect 5–15% reduction in `root_scored` through TT warm-up
- Allows one additional depth iteration within the same time budget at depth 10+

### Files

- `native/nmm_core/src/search.rs` — add `mtdf()`, update `iterative_deepening`, optionally update `iterative_deepening_scored`
- Rebuild the extension after changes: `scripts/build_rust.sh`

---

## V4-C: Star Square Placement Bonus in Rust Move Ordering (Medium Priority)

### What and Why

Sanmill's `movepick.cpp` scores candidate moves in placement phase with `RATING_STAR_SQUARE` for moves to high-connectivity squares. These are squares that belong to multiple mill lines — placing there gives maximum flexibility to close mills in several directions and cannot easily be blocked by a single opponent response.

Our current `ordered_moves()` in `search.rs` (line ~199) scores:
- Captures: `-2000`
- Mill-forming moves: `-1000`
- Killers (slot 0/1): `-600 / -500`
- Countermove: `-450`
- History heuristic: subtracted

During placement, when `mv.from = None`, we score moves only by mill formation and history. There is no positional bonus for placing on strategically strong squares. This means placement move ordering is weaker than it could be, leading to more re-searches and worse alpha-beta efficiency in the first 9–18 plies of the game.

### The Star Squares

In standard NMM, the high-connectivity squares are those where 3 or more mill lines intersect or where adjacency degree is 3. These are the midpoints of each ring's edges:

| Ring   | Squares              | Adjacency | Mill memberships |
|--------|---------------------|-----------|-----------------|
| Outer  | a4, d1, d7, g4      | 3         | 2               |
| Middle | b4, d2, d6, f4      | 3         | 2               |
| Inner  | c4, d3, d5, e4      | 3         | 2               |

These 12 squares have adjacency degree 3 (vs 2 for corner squares) and each belongs to exactly 2 mill lines, giving the most threatening placements per move.

### Implementation Plan

1. In `movegen.rs` or `search.rs`, add a precomputed const:
   ```rust
   /// Squares with adjacency degree 3 — ring-edge midpoints.
   const STAR_SQUARES: u32 = (1 << a4_idx) | (1 << d1_idx) | ... ; // all 12 squares
   ```
   (Use the same square index scheme as `POSITIONS` in `game/board.py`.)

2. In `ordered_moves()`, add to the sort key:
   ```rust
   // Placement only: star square bonus (between killers and history).
   if mv.from.is_none()
       && (1u32 << mv.to as u32) & STAR_SQUARES != 0
       && !move_forms_mill(board, color, mv.from, mv.to)  // already scored higher
   {
       s -= 300;  // prioritise star placements like a killer move
   }
   ```
   Only applies in placement phase (from=None) when the move does not already form a mill (those are already at -1000).

3. The exact penalty value (300) is a starting point — tune if placement move ordering metrics change.

### Expected Gain

- Better alpha-beta efficiency in placement phase (game turns 1–18)
- Fewer nodes expanded before finding the best placement sequence
- Should have no effect on move phase (from ≠ None, gate inactive)

### Files

- `native/nmm_core/src/search.rs` — `ordered_moves()` method
- `native/nmm_core/src/movegen.rs` (or `types.rs`) — `STAR_SQUARES` const

---

## V4-D: Python Fallback — Use Rust `legal_moves` (Medium Priority)

### Context

The Python `_negamax` in `ai/game_ai.py` is the fallback search, only used when the compiled `nmm_core` extension is unavailable (e.g., first-run without building, or a broken extension). Normal gameplay always uses Rust via `_choose_rust_scored`.

Despite being a fallback, it is worth improving because:
- The profiling session showed depth-4 taking 90 seconds in Python vs ~0.22s in Rust
- Calling `get_all_legal_moves(board)` was the second largest cost (10.5s + 4.9s = 15.4s out of 90s)
- `native_core.legal_moves(board)` already exists and is a thin PyO3 wrapper around Rust's bitboard movegen — **no new Rust code required**

### Current Code (two call sites)

```python
# ai/game_ai.py ~line 1812 (depth==0 branch):
_q_moves = get_all_legal_moves(board)

# ai/game_ai.py ~line 1861 (main search):
moves = get_all_legal_moves(board)
```

### Proposed Change

```python
# Top of _negamax or module level:
from .native_core import legal_moves as _rust_legal_moves, RUST_AVAILABLE

# Replace both call sites:
moves = (_rust_legal_moves(board) if RUST_AVAILABLE else None) or get_all_legal_moves(board)
```

`native_core.legal_moves()` returns `None` when Rust is absent (not an empty list — see `ai/native_core.py:70`), so `or get_all_legal_moves(board)` is the correct fallback idiom already used elsewhere.

### Same fix for `_qsearch`

The qsearch also calls `get_all_legal_moves(board)` — apply the same pattern there.

### Expected Gain

- Eliminates ~15.4s out of 90s at depth 4 (≈17% of total Python fallback time)
- Each call: Python dict scanning (~61µs) → Rust bitboard + PyO3 overhead (~5–10µs)

### Files

- `ai/game_ai.py` — `_negamax` (~lines 1812, 1861) and `_qsearch`

---

## V4-E: Python Fallback — Use Rust `evaluate_base` at Leaf Nodes (Medium Priority)

### Context

The largest single cost in the Python fallback profile was `evaluate_v2` at 21s out of 90s (depth 4, 118K calls).

The Rust `py_evaluate` function (`lib.rs:176`) calls `evaluate_v2` in Rust — the same function, already ported to Rust, verified against the Python implementation. `native_core.evaluate_base(board)` (`native_core.py:103`) wraps it with `board_to_bits` conversion.

### Current Code

```python
# ai/game_ai.py ~line 1818:
if self.use_v2_heuristics:
    heur = evaluate_v2(board, board.turn, weights=self._weights, _ply=ply)
```

### Proposed Change

When Rust is available and `use_v2_heuristics` is True, delegate to Rust:

```python
if self.use_v2_heuristics:
    if RUST_AVAILABLE:
        from .native_core import evaluate_base as _rust_eval
        heur = _rust_eval(board) or 0
    else:
        heur = evaluate_v2(board, board.turn, weights=self._weights, _ply=ply)
```

**Caveat:** `native_core.evaluate_base()` uses the default `EvalScale` (all weights at 100%) and does not accept `HeuristicWeights`. If personality weights diverge significantly from defaults, this produces slightly different scores to the Python fallback. However:
- The Python fallback already produces different scores to the Rust search (they are not guaranteed to match)
- Personality differences are minor (±20% on individual terms)
- The fallback path is only active when the Rust extension is unavailable, in which case personality accuracy matters less than speed

If exact weight parity is required, this item should wait until `py_evaluate_v2` is extended to accept an `EvalScale` argument.

### Expected Gain

- Eliminates ~21s out of 90s at depth 4 (≈23% of total Python fallback time)
- Combined with V4-D: ~38% reduction → depth-4 from 90s to ~55s

### Files

- `ai/game_ai.py` — leaf evaluation block in `_negamax` (~line 1817)
- `ai/native_core.py` — `evaluate_base()` already exists, no changes needed

---

## V4-F: Missing `_V2_MV_SHIFT` in Rust evaluate_v2 (Low Priority — Correctness)

### What

The Python `evaluate_v2` in `ai/heuristics.py` has a move-phase shiftable 2-config term (`_V2_MV_SHIFT = 6`) added in V3a. The Rust `evaluate_v2` in `native/nmm_core/src/heuristics.rs` was written before this term was added and does not include it.

The Rust `evaluate_v2` weights (from the doc comment at line ~367) are:
```
Place: piece(1) mob(1) blocked(8) mill(30) threat(15)
Move:  piece(12) mob(1) opp_blocked(48) mill(30) threat(18) zugzwang(600)
Fly:   piece(2) mill(32) threat(80) surplus(900)
```

The missing term: `+6 × (own_shift - opp_shift)` in move phase, where `own_shift` counts 2-configs where one of the pair is adjacent to the closing square (the "shiftable" configuration).

### Impact

This means the Rust search at depth ≥ 2 has a slightly weaker gradient toward shiftable plans than the Python evaluate_v2. The effect is small (weight 6 vs mill-threat weight 18) but it causes a minor inconsistency between the two evaluators that is worth closing.

### Implementation Plan

In `native/nmm_core/src/heuristics.rs`, add to the move-phase score:

1. In the mill scan loop (`for mm in MILL_MASKS`), after counting `own_thr`/`opp_thr`, detect shiftable 2-configs:
   - For each own 2-config (2 own pieces + 1 empty closing square):
     - Check no external own piece is adjacent to the closing square (it would already be closeable)
     - Check at least one of the two own pieces is adjacent to the closing square
     - Check that the potentially shifting piece has a free neighbour other than the closing square
   - Count as `own_shift`; do the same for `opp_shift`

2. Add to the move-phase return:
   ```rust
   + 6 * (own_shift - opp_shift)
   ```

3. Update the weight comment in the doc string to include `shift(6)`.

### Files

- `native/nmm_core/src/heuristics.rs` — mill scan loop in `evaluate_v2()` (~line 418)

---

## V4-G: TT Prefetching in Rust (Low Priority — Minor)

### What

Sanmill's `search.cpp` prefetches TT entries for all children before recursing into them. Modern CPUs can hide the cache miss latency (~100ns for a last-level cache miss) behind computation if the prefetch is issued early enough.

In `negamax()`, this would look like:

```rust
for mv in moves.iter() {
    let nb = make_move(board, mv);
    let nb_key = self.zobrist.hash(&nb);
    // Issue prefetch before recursing — CPU fetches cache line while we prepare.
    self.tt.prefetch(nb_key);
    // ... then recurse
}
```

`TranspositionTable::prefetch()` would emit an `_mm_prefetch` intrinsic (x86) or `__builtin_prefetch` (ARM).

### Expected Gain

2–5% reduction in search time in the move phase where TT hit rates are high. Not worth pursuing until V4-B and V4-C are implemented.

### Files

- `native/nmm_core/src/hash.rs` — add `prefetch()` method to `TranspositionTable`
- `native/nmm_core/src/search.rs` — add prefetch calls in `negamax()` and `qsearch()`

---

## V4-H: Full Bitboard Threading in Python Fallback (Low Priority — Large Refactor)

### What

The remaining 55s at depth 4 after V4-D and V4-E is dominated by two costs:

| Cost | Time | Root cause |
|------|------|------------|
| `_negamax` recursion | 11s | Python function call overhead |
| `board.apply_move` | 4.6s | Creates 4 new Python dicts per call |

`apply_move` cannot be eliminated without threading bitboard state `(white, black, wp, bp, stm)` through `_negamax` instead of Python `BoardState` objects. This would require:

1. Adding `py_apply_move(white, black, wp, bp, stm, from_idx, to_idx, cap_idx)` → `(white, black, wp, bp, stm)` to the Rust extension
2. Rewriting `_negamax` to accept and return bitboard tuples, calling `py_apply_move` for each child
3. Replacing all `board.positions`, `board.turn`, `board.pieces_placed` references with tuple accessors

This is a significant refactor (~300 lines of changes to `_negamax` and `_qsearch`). The Python fallback is used infrequently enough that the gain is not worth the complexity. Defer unless the Python fallback becomes a primary code path.

### Alternative

If the Python fallback performance becomes critical, the correct fix is to ensure the Rust extension builds reliably on all platforms (fix build scripts, add pre-built wheel) so the fallback is never hit in practice.

---

## Priority Summary

| Item | Priority | Effort | Benefit | Status |
|------|----------|--------|---------|--------|
| V4-A: `_time_ms` fix | Critical | Done | Restores correct Rust depth | **Deployed** |
| V4-B: MTD(f) in Rust | High | Medium (Rust) | 15–35% node reduction in iterative deepen | **Done** |
| V4-C: Star square ordering | Medium | Low (Rust) | Better placement-phase pruning | **Done** |
| V4-D: Rust `legal_moves` in fallback | Medium | Low (Python) | 17% faster Python fallback | Planned |
| V4-E: Rust `evaluate_base` in fallback | Medium | Low (Python) | 23% faster Python fallback | Planned |
| V4-F: `_V2_MV_SHIFT` in Rust heuristics | Low | Low (Rust) | Evaluator consistency | **Removed** — 25% overhead for weight-6 term |
| V4-G: TT prefetching | Low | Low (Rust) | 2–5% cache improvement | Deferred |
| V4-H: Bitboard threading in fallback | Low | High (Rust + Python) | Faster Python fallback | Deferred |

---

## Notes on Algorithm Choice

**Why not MCTS?**  
Sanmill includes an MCTS implementation. MCTS explores faster per node but requires many more nodes for accuracy at tactical depths (≥ 6). For exact best-move computation in NMM where tactical threats resolve in 2–4 moves, alpha-beta with MTD(f) is strictly better. MCTS is useful for analysis at shallower depth with very limited time.

**Why not something other than negamax?**  
For NMM's game tree size (~10^10 game states, branching factor 7–15), alpha-beta is optimal for exact minimax computation. The existing endgame DB and FullGame DB already shortcut large portions of the tree. MTD(f) is the best-known refinement of alpha-beta for this scale; nothing fundamentally better exists for exact search.

**NMM branching factor:**  
- Placement phase: ~10–14 (place × optional capture)
- Move phase: ~7–12 (slide adjacent × optional capture)
- With good move ordering: effective branching factor ~3–5 (alpha-beta halves the exponent)
- At depth 12 with effective b=4: ~4^6 = 4096 nodes in best case vs 13^12 ≈ 23 billion naive
