# HeuristicsV2 — Rust Leaf Evaluator

**Implementation:** `native/nmm_core/src/heuristics.rs` — `evaluate_v2()`  
**Language:** Rust, called from Python via PyO3 FFI  
**Status:** Live — default search evaluator as of July 2026

---

## Overview

`evaluate_v2` is the stripped-down leaf evaluator used at every node inside the Rust negamax search. It was designed around one constraint: **speed over completeness**. The full Python `evaluate()` has ~20 terms including fork threats, position values, cycling mills, encirclement, and wrapping pressure. V2 keeps only the terms that measurably affect play quality per microsecond spent — the rest are left to the root-only `tactical_move_bonus` which scores candidate moves once before the search begins.

Two passes over the board:

1. **O(24) piece scan** — mobility and blocked count for both sides via set-bit iteration over occupied squares
2. **O(16) mill scan** — closed mills and two-configs (mill threats) for both sides in a single loop over `MILL_MASKS`

No helper function calls at leaf depth. All 5 terms are computed inline.

---

## evaluate_v2 Formula

### Inputs

| Param | Type | Description |
|-------|------|-------------|
| `board` | `&Board` | Current position (bitboard representation) |
| `color` | `Color` | Side to evaluate from |
| `scale` | `EvalScale` | Personality scale factors (see below) |

### Terminal checks (before any arithmetic)

```
winner exists  → ±INF
own pieces < 3 (post-placement) → -INF
opp pieces < 3 (post-placement) → +INF
own mobility == 0 (Move phase)  → -INF  (blockade)
```

### Phase-specific formula

**Placement phase** (pieces still being placed):

```
score = (own_p − opp_p)
      + mob_w   × (own_mob − opp_mob) / 100
      + block_w × 8 × (opp_blocked − own_blocked) / 100
      + mill_w  × 30 × (own_mills − opp_mills) / 100
      + mill_w  × 15 × (own_thr − opp_thr) / 100
```

**Move phase** (all pieces placed, both sides have > 3):

```
score = 12 × (own_p − opp_p)
      + mob_w   × (own_mob − opp_mob) / 100
      + block_w × 48 × opp_blocked / 100
      + mill_w  × 30 × (own_mills − opp_mills) / 100
      + mill_w  × 18 × (own_thr − opp_thr) / 100
      + 600 × max(0, 3 − opp_mob)          ← zugzwang bonus
```

**Fly phase** (one side reduced to 3 pieces):

```
own_surp = max(0, own_thr − 1)
opp_surp = max(0, opp_thr − 1)

score = 2 × (own_p − opp_p)
      + mill_w × 32  × (own_mills − opp_mills) / 100
      + mill_w × 80  × (own_thr   − opp_thr)   / 100
      + mill_w × 900 × (own_surp  − opp_surp)   / 100
```

### Term glossary

| Symbol | Meaning |
|--------|---------|
| `own_p / opp_p` | Pieces on board |
| `own_mob / opp_mob` | Legal moves available (fly phase: capped at 5) |
| `own_blocked / opp_blocked` | Pieces with zero legal moves |
| `own_mills / opp_mills` | Closed mills (all 3 squares occupied) |
| `own_thr / opp_thr` | Two-configs with one empty closing square (mill threats) |
| `own_surp / opp_surp` | Surplus threats (fly: second simultaneous threat) |

### What was cut vs evaluate_base

| Term | In `evaluate_base` | In `evaluate_v2` |
|------|--------------------|------------------|
| Mill count | ✅ | ✅ |
| Mobility | ✅ | ✅ |
| Blocked pieces | ✅ | ✅ |
| Mill threats | ✅ | ✅ |
| Piece count | ✅ | ✅ |
| Zugzwang bonus | ✅ | ✅ |
| Two-configs (raw count) | ✅ | ❌ (subsumed by mill threats) |
| Double mills | ✅ | ❌ |
| Fork threats | ✅ | ❌ |
| Position value (cardinal/cross bonus) | ✅ | ❌ |
| Mill cycle ready | ✅ | ❌ |
| Encirclement | ✅ | ❌ |
| Squeeze count | ✅ | ❌ |
| Mill wrapping pressure | ✅ | ❌ |
| Fly asymmetry | ✅ | ❌ |
| Open mill domination | ✅ | ❌ |

The cut terms are either: (a) handled at root by `tactical_move_bonus`, (b) only meaningful at depth 1-2 where the root adjustments already cover them, or (c) measurably slow without affecting win-rate.

---

## EvalScale — Personality Weights

```rust
#[derive(Copy, Clone)]
pub struct EvalScale {
    pub mill:  i32,   // mill_count_scale,  default 100
    pub mob:   i32,   // mobility_scale,    default 100
    pub block: i32,   // blocked_scale,     default 100
}
```

Values are integer percentages. `100` = unchanged baseline. Passed from Python personality JSON → `HeuristicWeights` → `py_search_root_scored` FFI → `EvalScale` inside the Rust searcher, applied at every leaf call.

| Personality | mill | mob | block | Effect |
|-------------|------|-----|-------|--------|
| Balanced | 100 | 100 | 100 | Baseline |
| Aggressive | 180 | 50 | 80 | Values mills 1.8×; ignores mobility |
| Defensive | 75 | 200 | 250 | Heavily rewards restricting opponent movement |
| Positional | 80 | 300 | 150 | Maximises own legal moves at every node |
| Scholar | 130 | 190 | 95 | Mill-oriented with strong mobility awareness |
| Chaos | 50 | 50 | 50 | Half-strength on all terms; erratic play |

---

## FGOP — Frequency-Gated Opponent Pruning

Added alongside EvalScale. Prunes opponent moves that are both statically poor and structurally rare, reducing the effective branching factor at opponent plies.

**Fires when:** opponent to move AND `depth ≤ 5` AND not the first move AND not a tactical move (capture or mill closure).

**Dual gate:**

1. **Eval gate (Gate 1):** `opp_static_eval < best_opp_static_seen − 150` — this opponent move scores at least 150 below the best opponent reply found so far.
2. **Structural gate (Gate 2 — `is_structurally_rare`):** The move is leaving a square that belongs to an own two-config, AND the destination does not complete any mill. O(6) check using `SQUARE_MILLS[sq]` (each square in at most 3 mills).

Both gates must pass to prune. Tactical moves (captures, mill closures) are always searched regardless.

---

## Search Infrastructure

`evaluate_v2` is the leaf evaluator inside a Rust negamax with the following infrastructure:

| Feature | Detail |
|---------|--------|
| Iterative deepening | SMP across threads; aspiration windows (±175) |
| Transposition table | 64-bit Zobrist; persistent across turns |
| Move ordering | Killers (2/depth), history heuristic, preferred-root hints |
| Late-move reductions | LMR fires from move index ≥ 3 |
| Null-move pruning | R=2, skipped in placement/endgame/zugzwang |
| Quiescence search | Captures only; `evaluate_v2` as stand-pat |
| DB probing | FullgameDB + EndgameSolvedDB (mmap) between TT and leaf |
| FGOP | See above |

---

## Ply Depth → Time Budget

Time budget formula: `min(120 s, 0.065 × 1.66^depth)`  
The budget is a wall-clock cap; iterative deepening completes whichever ply it can within the window.

| Ply | Time budget |
|-----|-------------|
|  1  | 108 ms      |
|  2  | 179 ms      |
|  3  | 297 ms      |
|  4  | 494 ms      |
|  5  | 819 ms      |
|  6  | 1.4 s       |
|  7  | 2.3 s       |
|  8  | 3.7 s       |
|  9  | 6.2 s       |
| 10  | 10.3 s      |
| 11  | 17.1 s      |
| 12  | 28.5 s      |
| 13  | 47.2 s      |
| 14  | 1.3 min     |
| 15  | 2.0 min     |

Tournament difficulty mapping (default min=5 / max=16 settings):

| Difficulty | Max ply | Budget |
|------------|---------|--------|
| 3 | 8 | 3.7 s |
| 4 | 10 | 10.3 s |
| 5 | 11 | 17.1 s |
| 6 | 13 | 47.2 s |
| 7 | 14 | 1.3 min |
| 8 | 16 | 120 s (cap) |

---

## Phase 1 — Mate-Distance Fix

**Change:** Terminal scoring changed from `INF - depth` (remaining depth) to `INF - ply` (ply from root). TT adjustment functions `score_to_tt` / `score_from_tt` added so mate-in-N scores are stored as absolute distances, not search-relative.

**Why it mattered:** `INF - depth` is remaining depth, so a mate-in-3 found at depth 7 scored `INF - 4` and a mate-in-3 found at depth 3 scored `INF - 0`. The search had no preference for the shorter path. With `INF - ply`, mate-in-3 always scores `INF - 3` regardless of where in the tree it was found. The AI now aggressively pursues the fastest win.

**Benchmark — 10-game post-Phase-1 assessment (V2+Phase1 vs V1):**

| | V2+Phase1 wins | Draws | V1 wins |
|-|----------------|-------|---------|
| Count | 7 | 1 | 2 |
| % | 70% | 10% | 20% |

Draw rate dropped from 37% (pre-Phase-1, 100-game run) to 10%. The fix converts middlegame positions that previously drifted into draws into decisive wins by pursuing the shortest forcing sequence.

---

## Phase 2 — Qsearch Forcing Extension

**Change:** Quiescence search extended beyond captures and mill closures to three categories of forcing moves:

1. **Reachable two-config creator** — a move that gives own side a mill threat where the closing square is reachable next move (still has pieces to place, or own piece adjacent and not mill-locked, or ≤3 pieces in fly phase). Opponent is forced to block.
2. **Forced block** — blocking an opponent's reachable two-config. The search must explore the block because the alternative (opponent closes the mill next move) is already in scope.
3. **Blocker displacement** — own piece that was blocking an opponent mill moves away, forcing the opponent to respond to the newly opened threat.

Cap: `QS_FORCING_CAP = 6` extra plies on forcing lines. Tactical moves (captures, mill closures) do not count against the cap. The base qsearch tactical extensions are unaffected.

**Concrete depth gain:** At difficulty 3 (ply 8 main search budget), forcing lines search to ply 14. At difficulty 6 (ply 13), forcing lines reach ply 19.

**Implementation:** `creates_reachable_two_config()` in `search.rs`, O(6) per move via `SQUARE_MILLS[to_sq]`. Counter in global `FORCING_EXT_COUNT` (AtomicU64). Test tool: `tools/test_forcing_ext.py`.

**Test — 100 AI-vs-AI games at difficulty 3 (`tools/test_forcing_ext.py --games 100 --diff 3`):**

| | Value |
|-|-------|
| Games with ≥1 extension | 100 / 100 (100%) |
| Total extensions fired | 9,849,060,851 |
| Average per game | 98.5 million |
| Range (min / max) | 38.4M / 564.3M |

Extension fires on every game. Variation reflects game length and tactical density — draw games and longer battles fire more than quick decisive games.

*Earlier 20-game run (same conditions): avg 88.6M/game, range 46.8M–311.8M. The 100-game average is a more stable baseline.*

---

## Benchmark: V2 vs V1 (100 games, July 2026)

*This run was completed before Phase 1 (mate-distance fix). It establishes the baseline. Phase 1 results are in the section above.*

**V2:** Rust search + `evaluate_v2` + `EvalScale` + FGOP  
**V1:** Python negamax + full `evaluate()` (old heuristics, ~16 terms)  
**Conditions:** 3.0 s/move budget, balanced personality, colours alternated each game

At 3 s, V2 typically reaches **ply 11** while V1 reaches **ply 5–6**.

| | V2 wins | Draws | V1 wins |
|-|---------|-------|---------|
| Count | 46 | 37 | 17 |
| % | 46% | 37% | 17% |

**V2 score: 64.5 / 100 (64.5%)**  
**Decisive games only: V2 46 – V1 17 (73%)**

The high draw rate (37%) reflects Nine Men's Morris's natural drawing tendency when both engines share the same opening book and trajectory hints. In decisive games the depth advantage is conclusive. V1 remains competitive in positionally balanced middlegames where the extra plies don't find a forcing sequence.

Total match time: 187.8 minutes (~1.9 min/game average).
