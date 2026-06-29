# HeuristicsV2 Plan — Bare-Bones Deep Search

**Target:** 12–20 ply iterative-deepening negamax with a stage-relevant O(n) leaf evaluator.  
**Repository:** `benmarkbrandwood-blip/NMM_LLM`  
**Files touched:** `ai/heuristics.py`, `ai/game_ai.py`, `ai/transposition_table.py`

---

## Table of Contents

1. [Goals and constraints](#1-goals-and-constraints)
2. [Stage 0 — Scaffold v2 alongside v1 (zero game-change)](#2-stage-0--scaffold-v2-alongside-v1)
3. [Stage 1 — Write `evaluate_v2()`](#3-stage-1--write-evaluate_v2)
4. [Stage 2 — Wire v2 behind a feature flag](#4-stage-2--wire-v2-behind-a-feature-flag)
5. [Stage 3 — Transposition table upgrade](#5-stage-3--transposition-table-upgrade)
6. [Stage 4 — Aspiration windows](#6-stage-4--aspiration-windows)
7. [Stage 5 — Null-move pruning](#7-stage-5--null-move-pruning)
8. [Stage 6 — Late-move reductions (LMR)](#8-stage-6--late-move-reductions-lmr)
9. [Stage 7 — Expanded endgame DB probing](#9-stage-7--expanded-endgame-db-probing)
10. [Stage 8 — Trajectory DB: deeper human-move search](#10-stage-8--trajectory-db-deeper-human-move-search)
11. [Stage 9 — Additional improvements](#11-stage-9--additional-improvements)
12. [Stage 10 — Validation and old-code removal](#12-stage-10--validation-and-old-code-removal)
13. [Weight reference table](#13-weight-reference-table)

---

## 1  Goals and constraints

| Goal | Detail |
|------|--------|
| Leaf evaluation ≤ 10 μs | No function calls that loop over MILLS more than once; no graph traversal |
| Drop `tactical_move_bonus` from leaf | Root-level bonus retained in `_root_search` only |
| Drop `two_config` counting from leaf | Too cheap to measure but conceptually noisy at depth |
| Mobility weight reduced | Movement phase: ×1 instead of ×8 (directional signal, not dominant) |
| Keep `tactical_move_bonus` at root | Still called in `_root_search` unchanged |
| Keep all endgame DBs | `EndgameSolvedDB`, `FullGameDB`, Malom — probed at every depth |
| Keep humanDB trajectory search | `HumanDB.query_line()` path extended to depth 6 (was 3) |
| Old code deleted only in Stage 10 | Backup tag `v1-heuristics` created before any deletion |

---

## 2  Stage 0 — Scaffold v2 alongside v1

### 2.1  Create git backup tag

```bash
git tag v1-heuristics HEAD
git push origin v1-heuristics
```

### 2.2  Add flag to `game_ai.py` — **`GameAI.__init__`**

**Insertion point:** after line  
```python
        self._weights: HeuristicWeights = weights if weights is not None else DEFAULT_WEIGHTS
```

Add:
```python
        # V2 heuristics flag — set True to use bare-bones evaluate_v2() instead of evaluate().
        # All search improvements (aspiration, null-move, LMR) activate automatically when True.
        self.use_v2_heuristics: bool = False
```

### 2.3  Add import in `game_ai.py`

Extend the existing import from `heuristics.py`:
```python
# EXISTING:
from .heuristics import (INF, evaluate, clear_eval_cache, HeuristicWeights,
                          DEFAULT_WEIGHTS, tactical_move_bonus, _sealed_two_configs,
                          _dual_connected_mill_alert, _closeable_mills)
# ADD:
from .heuristics import evaluate_v2  # noqa: F401  — Stage 1 adds this symbol
```

---

## 3  Stage 1 — Write `evaluate_v2()`

### 3.1  New file section in `ai/heuristics.py`

**Insertion point:** at the very bottom of `ai/heuristics.py`, after the last function definition.

```python
# ══════════════════════════════════════════════════════════════════════════════
# HEURISTICS V2 — Bare-bones stage-relevant leaf evaluator
# ══════════════════════════════════════════════════════════════════════════════
#
# Design goals:
#   - Single pass over MILLS (O(16)) for all mill/threat counts
#   - Single pass over POSITIONS (O(24)) for piece/mobility/blocked counts
#   - No graph traversal, no simulation, no multi-step lookahead inside leaf
#   - Stage-gated: only compute what matters for the current game phase
#   - Symmetric: score = self_features - opp_features  (opponent model built-in)
#
# Removed vs evaluate():
#   - tactical_move_bonus (root-only, not called here)
#   - two_config counting
#   - assembly gradients (all _free_piece_assembly, _one_config_approach etc.)
#   - convergence cluster / bipartite matching
#   - placement chain scan
#   - _encirclement / wrap weights
#   - All B-XX placement bonuses (moved to root tactical_move_bonus)
#
# Kept vs evaluate():
#   - Piece count differential
#   - Mobility count (movement phase only, weight x1)
#   - Blocked opponent pieces (movement phase, weight x48)
#   - Mill count (all phases)
#   - Mill threats / closeable mills (all phases)
#   - Blocked-piece count (Sanmill-style) in placement and movement
#   - Mill-type count (Sanmill-style: pieces in hand / on board / removable proxy)
#   - Cycle-ready mills (movement + fly phases only; not placement)
#   - Fork threats (movement + fly phases only; not placement)
#   - Near-blocked (squeeze) count (movement phase only; not placement)
#   - Simple positional value by connectivity (placement phase)
#   - Near-zugzwang inline bonus (movement phase)
#   - Fly surplus / win config (fly phase)
# ══════════════════════════════════════════════════════════════════════════════


def _v2_scan_board(board, color: str) -> tuple:
    """Single O(24) pass: piece count, mobility, blocked counts, pieces in hand.

    Returns
        (own_pieces, opp_pieces, own_mob, opp_mob, own_blocked, opp_blocked, own_hand, opp_hand)

    Sanmill-style board terms retained here:
      - pieces in hand
      - pieces on board
      - blocked pieces

    Mobility = sum of free adjacent squares across all pieces of a side.
    Blocked = pieces with zero adjacent empty squares.
    """
    opp = "B" if color == "W" else "W"
    own_p = opp_p = own_mob = opp_mob = own_blocked = opp_blocked = 0
    for pos in POSITIONS:
        owner = board.positions[pos]
        if not owner:
            continue
        free = sum(1 for nb in ADJACENCY[pos] if not board.positions[nb])
        if owner == color:
            own_p += 1
            own_mob += free
            if free == 0:
                own_blocked += 1
        else:
            opp_p += 1
            opp_mob += free
            if free == 0:
                opp_blocked += 1
    own_hand = max(0, 9 - board.pieces_placed.get(color, 0))
    opp_hand = max(0, 9 - board.pieces_placed.get(opp, 0))
    return own_p, opp_p, own_mob, opp_mob, own_blocked, opp_blocked, own_hand, opp_hand


def _v2_scan_mills(board, color: str) -> tuple:
    """Single O(16) pass over MILLS: mill count, threat count, removable proxy.

    Returns (own_mills, opp_mills, own_thr, opp_thr, own_rem_proxy, opp_rem_proxy).

    Sanmill refers to three main scoring components: pieces in hand, pieces on board,
    and pieces that can be removed.  In NMM_LLM v2 we approximate the third term with
    closeable mill threats, because each immediate closeable mill is a near-removal.
    """
    opp = "B" if color == "W" else "W"
    own_mills = opp_mills = own_thr = opp_thr = 0
    for mill in MILLS:
        vals = [board.positions[p] for p in mill]
        c = vals.count(color)
        o = vals.count(opp)
        e = vals.count("")
        if c == 3:
            own_mills += 1
        elif o == 3:
            opp_mills += 1
        elif c == 2 and e == 1:
            own_thr += 1
        elif o == 2 and e == 1:
            opp_thr += 1
    own_rem_proxy = own_thr + own_mills
    opp_rem_proxy = opp_thr + opp_mills
    return own_mills, opp_mills, own_thr, opp_thr, own_rem_proxy, opp_rem_proxy


def _v2_cycle_ready(board, color: str) -> int:
    """Count mills where one piece has an adjacent empty square outside the mill.

    A cycle-ready mill can open/close on consecutive turns, producing repeated
    captures.  O(mills x adjacency) -- typically 16 x 2 = 32 checks.
    """
    count = 0
    for mill in MILLS:
        vals = [board.positions[p] for p in mill]
        if vals.count(color) != 3:
            continue
        mill_set = set(mill)
        if any(
            not board.positions[nb]
            for p in mill
            for nb in ADJACENCY[p]
            if nb not in mill_set
        ):
            count += 1
    return count


def _v2_fork_threats(board, color: str) -> int:
    """Count squares where placing/moving would close 2+ mills simultaneously.

    These are 'diamond' or 'triple-junction' squares -- unblockable fork threats.
    O(positions x mills).
    """
    count = 0
    for pos in POSITIONS:
        if board.positions[pos]:
            continue
        closes = sum(
            1 for mill in MILLS
            if pos in mill
            and [board.positions[p] for p in mill].count(color) == 2
        )
        if closes >= 2:
            count += 1
    return count


def _v2_squeeze_count(board, opp: str) -> int:
    """Count opponent pieces that have exactly one free neighbour (nearly blocked).

    Herding these pieces to zero moves produces zugzwang.  O(opp_pieces x adj).
    """
    count = 0
    for pos in POSITIONS:
        if board.positions[pos] != opp:
            continue
        free = sum(1 for nb in ADJACENCY[pos] if not board.positions[nb])
        if free == 1:
            count += 1
    return count


def _v2_position_value(board, color: str) -> int:
    """Sum positional value of all own pieces.

    Cardinal squares (4 neighbours): 3 pts each -- highest mobility potential.
    Cross squares (3 neighbours): 2 pts -- moderate potential.
    Corner squares (2 neighbours): 1 pt -- limited mobility.

    O(own_pieces) after ADJACENCY lookup (constant per square).
    """
    total = 0
    for pos in POSITIONS:
        if board.positions[pos] != color:
            continue
        n = len(ADJACENCY[pos])
        total += 3 if n == 4 else (2 if n == 3 else 1)
    return total


def _v2_win_config(board, opp: str) -> int:
    """1 if opponent has exactly 3 pieces (is in fly phase or about to lose), else 0."""
    return 1 if board.pieces_on_board[opp] <= 3 else 0


# -- V2 weight constants — tunable, stage-specific ----------------------------
# All weights intentionally kept as module-level constants for easy tuning.
# Increase MILL weight or FORK weight to observe directional AI behaviour changes.

# Placement phase
_V2_PL_HAND    =  1    # Sanmill-style pieces in hand differential
_V2_PL_PIECE   =  1    # Sanmill-style pieces on board differential
_V2_PL_MOB     =  1    # positional mobility signal
_V2_PL_BLOCKED =  8    # Sanmill-style blocked-piece differential
_V2_PL_MILL    = 30    # closed mills dominate
_V2_PL_THREAT  = 15    # 2-config closeable threat
_V2_PL_REM     = 10    # Sanmill-style removable proxy = mills + closeable mills
_V2_PL_POS     =  2    # positional connectivity bonus

# Movement phase
_V2_MV_PIECE   = 12    # material advantage scales
_V2_MV_MOB     =  1    # directional signal only (was x8, reduced to keep evaluation fast)
_V2_MV_BLOCKED = 48    # Sanmill-style fully blocked opponent piece = near-won
_V2_MV_MILL    = 30
_V2_MV_THREAT  = 18
_V2_MV_CYCLE   = 22    # cycle-ready mills
_V2_MV_FORK    = 14    # fork threats (diamond squares)
_V2_MV_SQUEEZE = 30    # opponent near-blocked (1 free neighbour)
_V2_MV_ZUGZ    = 600   # near-zugzwang: each step below opp_mob=3

# Fly phase
_V2_FLY_PIECE  =  2
_V2_FLY_MILL   = 32
_V2_FLY_THREAT = 80    # every 2-config is a near-immediate mill
_V2_FLY_CYCLE  = 80    # cycling mills are the primary winning pattern
_V2_FLY_FORK   = 55    # unblockable diamond forks
_V2_FLY_WIN    = 1190  # opponent in fly (<=3 pieces) = almost won
_V2_FLY_SURP   = 900   # fly surplus: (own_thr-1) - (opp_thr-1)


def evaluate_v2(
    board,
    color: str,
    endgame_state=None,   # retained for API compatibility; not used in v2
    weights=None,          # retained for API compatibility; ignored in v2
    *,
    _ply: int = 0,
) -> int:
    """Bare-bones stage-relevant leaf evaluator.

    Returns score in the same integer scale as evaluate() so all TT entries,
    alpha-beta windows, and sentinel comparisons remain valid.

    Score = own_features - opp_features (opponent model built-in).
    Terminal positions return +/-INF as in v1.
    """
    from game.rules import is_terminal, get_game_phase

    terminal, winner = is_terminal(board)
    if terminal:
        return -(INF - _ply) if winner != color else (INF - _ply)

    opp = "B" if color == "W" else "W"
    phase = get_game_phase(board, color)

    # -- Single-pass board scan (O(24)) ----------------------------------------
    own_p, opp_p, own_mob, opp_mob, own_blocked, opp_blocked, own_hand, opp_hand = _v2_scan_board(board, color)

    # -- Single-pass mill scan (O(16)) -----------------------------------------
    own_mills, opp_mills, own_thr, opp_thr, own_rem_proxy, opp_rem_proxy = _v2_scan_mills(board, color)

    # -- Stage-gated features --------------------------------------------------

    if phase == "place":
        own_pos = _v2_position_value(board, color)
        opp_pos = _v2_position_value(board, opp)
        score = (
            _V2_PL_HAND    * (own_hand - opp_hand)
            + _V2_PL_PIECE   * (own_p    - opp_p)
            + _V2_PL_MOB     * (own_mob  - opp_mob)
            + _V2_PL_BLOCKED * (opp_blocked - own_blocked)
            + _V2_PL_MILL    * (own_mills - opp_mills)
            + _V2_PL_THREAT  * (own_thr  - opp_thr)
            + _V2_PL_REM     * (own_rem_proxy - opp_rem_proxy)
            + _V2_PL_POS     * (own_pos  - opp_pos)
        )

    elif phase == "move":
        own_cycle  = _v2_cycle_ready(board, color)
        opp_cycle  = _v2_cycle_ready(board, opp)
        own_fork   = _v2_fork_threats(board, color)
        opp_fork   = _v2_fork_threats(board, opp)
        own_sqz    = _v2_squeeze_count(board, opp)    # opponent near-blocked = good
        opp_sqz    = _v2_squeeze_count(board, color)  # own near-blocked = bad
        score = (
            _V2_MV_PIECE   * (own_p     - opp_p)
            + _V2_MV_MOB     * (own_mob   - opp_mob)
            + _V2_MV_BLOCKED * opp_blocked
            + _V2_MV_MILL    * (own_mills  - opp_mills)
            + _V2_MV_THREAT  * (own_thr   - opp_thr)
            + _V2_MV_CYCLE   * (own_cycle  - opp_cycle)
            + _V2_MV_FORK    * (own_fork   - opp_fork)
            + _V2_MV_SQUEEZE * (own_sqz    - opp_sqz)
        )
        # Inline domination bonus: no function call needed
        if own_p >= 6 and opp_p <= 4:
            score += 80 * max(0, own_mills - opp_p)
        # Near-zugzwang bonus: each step below opp_mob = 3
        if opp_mob < 3:
            score += _V2_MV_ZUGZ * (3 - opp_mob)

    else:  # fly
        own_cycle = _v2_cycle_ready(board, color)
        opp_cycle = _v2_cycle_ready(board, opp)
        own_fork  = _v2_fork_threats(board, color)
        opp_fork  = _v2_fork_threats(board, opp)
        win_cfg   = _v2_win_config(board, opp)
        own_surp  = max(0, own_thr - 1)
        opp_surp  = max(0, opp_thr - 1)
        score = (
            _V2_FLY_PIECE  * (own_p    - opp_p)
            + _V2_FLY_MILL   * (own_mills - opp_mills)
            + _V2_FLY_THREAT * (own_thr   - opp_thr)
            + _V2_FLY_CYCLE  * (own_cycle  - opp_cycle)
            + _V2_FLY_FORK   * (own_fork   - opp_fork)
            + _V2_FLY_WIN    * win_cfg
            + _V2_FLY_SURP   * (own_surp  - opp_surp)
        )

    return score
```

---

## 4  Stage 2 — Wire v2 behind a feature flag

### 4.1  `_negamax` leaf evaluation switch

**File:** `ai/game_ai.py`  
**Insertion point:** in `_negamax()`, replace the single `evaluate()` call at the leaf (depth == 0 or quiescence leaf).

Find:
```python
        if depth == 0:
            return evaluate(board, board.turn, endgame_state, self._active_weights())
```

Replace with:
```python
        if depth == 0:
            if self.use_v2_heuristics:
                return evaluate_v2(board, board.turn, _ply=ply)
            return evaluate(board, board.turn, endgame_state, self._active_weights())
```

### 4.2  `_root_search` — suppress `tactical_move_bonus` when using v2

**File:** `ai/game_ai.py`  
**Insertion point:** in `_root_search()`, find:
```python
            if abs(score_raw) < INF // 2:
                score = score_raw + tactical_move_bonus(board, nb, self.color, self._active_weights(), self._opp_last_weak)
            else:
                score = score_raw
```

Replace with:
```python
            if abs(score_raw) < INF // 2 and not self.use_v2_heuristics:
                score = score_raw + tactical_move_bonus(
                    board, nb, self.color, self._active_weights(), self._opp_last_weak
                )
            else:
                score = score_raw
```

### 4.3  `_populate_thinking` guard

**File:** `ai/game_ai.py`  
In `_populate_thinking()`, wrap the `tactical_move_bonus` call:
```python
        try:
            if self.use_v2_heuristics:
                self.last_thinking = ""
                return
            after = board.apply_move(move)
            bd = tactical_move_bonus(...)
            ...
```

### 4.4  Activate in test harness

In any test script or `web/app.py` initialisation:
```python
ai = GameAI(color="B", difficulty=7)
ai.use_v2_heuristics = True   # switch on v2
```

---

## 5  Stage 3 — Transposition table upgrade

The current TT (`transposition_table.py`) uses a 2^18 (262 144) slot table with depth-preferred replacement. For 16-ply search this is too small — collisions degrade to ~40% hit-rate at depth 12+.

### 5.1  Resize to 2^21 and add always-replace slot

**File:** `ai/transposition_table.py`  
**Modification:** replace the two constants at the top.

```python
# EXISTING (remove):
_TABLE_SIZE = 1 << 18
_MASK       = _TABLE_SIZE - 1

# REPLACE WITH:
# 2^21 = 2 097 152 slots — ~450 MB worst case but typically < 100 MB in practice
# because Python dicts only fill touched slots.  Two-tier: depth-preferred primary
# + always-replace secondary (classic "two deep" TT scheme).
_TABLE_SIZE = 1 << 21
_MASK       = _TABLE_SIZE - 1
_ALT_OFFSET = _TABLE_SIZE >> 1   # secondary slot offset within the same array
```

### 5.2  Add two-tier replacement policy

In `TranspositionTable.store()`:

```python
    def store(
        self,
        hash_key: int,
        depth: int,
        score: int,
        flag: int,
        from_sq: str | None,
        to_sq: str,
    ) -> None:
        """Two-deep replacement: depth-preferred primary + always-replace secondary."""
        idx = hash_key & _MASK
        existing = self._table[idx]
        if existing is None or depth >= existing[1]:
            # Primary slot: prefer deeper searches
            self._table[idx] = (hash_key, depth, score, flag, from_sq, to_sq)
        else:
            # Secondary slot (always-replace): always store so recent positions
            # are accessible even if a deeper entry holds the primary slot.
            alt_idx = (hash_key ^ _ALT_OFFSET) & _MASK
            self._table[alt_idx] = (hash_key, depth, score, flag, from_sq, to_sq)
```

### 5.3  Update `lookup()` to check both slots

```python
    def lookup(self, hash_key: int):
        """Check primary slot first, then secondary (always-replace) slot."""
        idx = hash_key & _MASK
        entry = self._table[idx]
        if entry is not None and entry[0] == hash_key:
            return entry[1:]
        # Secondary slot
        alt_idx = (hash_key ^ _ALT_OFFSET) & _MASK
        entry = self._table[alt_idx]
        if entry is not None and entry[0] == hash_key:
            return entry[1:]
        return None
```

### 5.4  Do NOT clear TT between iterative deepening iterations

**File:** `ai/game_ai.py` in `_iterative_deepen()`:

Find the per-iteration TT clear (if present) and remove it. The TT should only be cleared at the start of `choose_move()` (which already calls `self._tt.clear()`). Keeping entries across iterations is the primary speedup for deep search.

Currently `choose_move()` does:
```python
        self._tt.clear()
```
This is correct. **Do not add any additional `_tt.clear()` calls inside `_iterative_deepen()`'s loop.**

---

## 6  Stage 4 — Aspiration windows

Aspiration windows reduce the effective branching factor at deeper plies by first trying a narrow `[alpha-delta, beta+delta]` window. A fail-high or fail-low triggers a full re-search.

### 6.1  Add aspiration window logic in `_iterative_deepen()`

**File:** `ai/game_ai.py`  
**Insertion point:** inside the `_iterative_deepen()` loop body, replacing the bare `_root_search` call.

Find the inner loop structure (pseudocode):
```python
        for depth in range(1, max_depth + 1, ...):
            ...
            best_move, score = self._root_search(board, depth, ...)
```

Replace with:
```python
        # Aspiration window — only activate with v2 heuristics and depth >= 5
        _ASPIRATION_DELTA = 50   # initial half-window (in heuristic score units)
        prev_score: int | None = None

        for depth in range(1, max_depth + 1):
            if self.use_v2_heuristics and depth >= 5 and prev_score is not None:
                delta = _ASPIRATION_DELTA
                lo = prev_score - delta
                hi = prev_score + delta
                while True:
                    try:
                        best_move, score = self._root_search(
                            board, depth, top_n=top_n, moves=moves,
                            alpha=lo, beta=hi,
                        )
                    except _SearchAbort:
                        break
                    if score <= lo:
                        lo -= delta * 2    # fail low — widen downward
                        delta *= 2
                    elif score >= hi:
                        hi += delta * 2    # fail high — widen upward
                        delta *= 2
                    else:
                        break              # score inside window — done
                    if lo <= -INF + 100 and hi >= INF - 100:
                        # Full window — run as normal
                        best_move, score = self._root_search(
                            board, depth, top_n=top_n, moves=moves
                        )
                        break
            else:
                try:
                    best_move, score = self._root_search(
                        board, depth, top_n=top_n, moves=moves
                    )
                except _SearchAbort:
                    break
            prev_score = score
```

---

## 7  Stage 5 — Null-move pruning

A null move (pass) gives a rough beta upper bound. If even with no move the position exceeds beta, we can prune. Particularly effective in NMM where zugzwang is common — add a zugzwang guard.

### 7.1  Add null-move in `_negamax()`

**File:** `ai/game_ai.py`  
**Insertion point:** inside `_negamax()`, **after** the TT lookup and **before** the move generation loop.

```python
        # -- Null-move pruning (V2 mode only) ------------------------------------
        # Skip when: depth < 3, in endgame (pieces <= 5 total), zugzwang risk
        # (own_mob < 3), or already in a null-move subtree (ply is odd heuristic).
        # R = 2 reduction (standard for NMM's branching factor).
        _NULL_R = 2
        if (self.use_v2_heuristics
                and depth >= 3
                and not terminal
                and abs(alpha) < INF // 2
                and abs(beta) < INF // 2):
            # Zugzwang guard: count own free moves; skip null-move if very restricted
            _own_mob = sum(
                1 for pos in POSITIONS
                if board.positions[pos] == board.turn
                for nb in ADJACENCY[pos]
                if not board.positions[nb]
            )
            _total_pieces = sum(board.pieces_on_board.values())
            _all_placed = (board.pieces_placed.get("W", 0) >= 9
                           and board.pieces_placed.get("B", 0) >= 9)
            _in_endgame = _all_placed and _total_pieces <= 7
            if _own_mob >= 3 and not _in_endgame:
                # Make null move: swap turn without placing/moving
                null_board = board.swap_turn()   # see Section 7.2 below
                if null_board is not None:
                    null_score = -self._negamax(
                        null_board, depth - 1 - _NULL_R,
                        -beta, -beta + 1,
                        endgame_state, 0, 0, ply + 1,
                    )
                    if null_score >= beta:
                        return beta   # fail-hard cutoff
```

### 7.2  Add `BoardState.swap_turn()` helper

**File:** `game/board.py`  
Add a method to `BoardState`:

```python
    def swap_turn(self) -> "BoardState":
        """Return a copy with the turn flipped (for null-move pruning).

        WARNING: Only valid for movement and fly phases.  Do not call during
        placement (piece counts change on placement -- null move is undefined).
        """
        import copy
        nb = copy.copy(self)
        nb.turn = "B" if self.turn == "W" else "W"
        return nb
```

---

## 8  Stage 6 — Late-move reductions (LMR)

After trying the first few moves at a node fully, reduce the depth for later moves that are unlikely to be best. Re-search at full depth if the reduced search raises alpha.

### 8.1  Add LMR in `_negamax()` move loop

**Insertion point:** inside the move loop in `_negamax()`, after the first 2 moves have been tried.

The current loop structure (simplified):
```python
        for move in moves:
            nb = board.apply_move(move)
            score = -self._negamax(nb, depth - 1, -beta, -alpha, ...)
            if score > best:
                best = score
                ...
            if best >= beta:
                break
```

Replace with:
```python
        _lmr_move_idx = 0
        for move in moves:
            nb = board.apply_move(move)
            _lmr_move_idx += 1

            # LMR: reduce depth for quiet moves after the 3rd candidate.
            # Conditions: v2 mode, depth >= 3, not a capture, not a mill close,
            # not a forced block, and not in fly phase (too few moves).
            _do_lmr = (
                self.use_v2_heuristics
                and depth >= 3
                and _lmr_move_idx > 2
                and not move.get("capture")
                and not _is_mill_close(board, move)   # helper -- see Section 8.2
                and get_game_phase(board, board.turn) != "fly"
            )
            _reduction = 2 if _lmr_move_idx > 5 else 1

            if _do_lmr:
                # Search at reduced depth
                score = -self._negamax(
                    nb, depth - 1 - _reduction, -(alpha + 1), -alpha,
                    endgame_state, ext_budget, opp_plies_left, ply + 1,
                )
                # Re-search at full depth if the reduced search improved alpha
                if score > alpha:
                    score = -self._negamax(
                        nb, depth - 1, -beta, -alpha,
                        endgame_state, ext_budget, opp_plies_left, ply + 1,
                    )
            else:
                score = -self._negamax(
                    nb, depth - 1, -beta, -alpha,
                    endgame_state, ext_budget, opp_plies_left, ply + 1,
                )

            if score > best_score:
                best_score = score
                best_move_inner = move
            if score > alpha:
                alpha = score
            if alpha >= beta:
                self._store_killer(depth, move.get("from"), move["to"])
                self._history[(move.get("from"), move["to"])] = (
                    self._history.get((move.get("from"), move["to"]), 0) + depth * depth
                )
                break
```

### 8.2  Add `_is_mill_close()` helper in `game_ai.py`

```python
def _is_mill_close(board: "BoardState", move: dict) -> bool:
    """True if this move closes a mill (used as LMR exemption guard)."""
    color = board.turn
    to = move["to"]
    for mill in MILLS:
        if to not in mill:
            continue
        vals = [board.positions[p] for p in mill]
        if vals.count(color) == 2 and vals.count("") == 1:
            return True
    return False
```

---

## 9  Stage 7 — Expanded endgame DB probing

The current probe in `_negamax()` only fires when **both** sides have <= 3 pieces. The `EndgameSolvedDB` already loads all `endgame_N_M.wdl` files (3v3, 4v3, 4v4, 5v3, 5v4, 5v5, 6v3, 6v4, 6v5, 6v6, 7v3, etc.) whenever they are present in the configured `db_dir`.

### 9.1  Broaden the probe guard in `_negamax()`

**File:** `ai/game_ai.py`  
Find in `_negamax()`:
```python
        if (self._endgame_solved_db is not None
                and board.pieces_placed.get("W", 0) >= 9
                and board.pieces_placed.get("B", 0) >= 9
                and board.pieces_on_board.get("W", 0) + board.pieces_on_board.get("B", 0) <= 6):
```

Replace with:
```python
        if (self._endgame_solved_db is not None
                and self._endgame_solved_db.is_available()
                and board.pieces_placed.get("W", 0) >= 9
                and board.pieces_placed.get("B", 0) >= 9):
            # Probe all available tables -- EndgameSolvedDB.query() returns None
            # when the (nW, nB) combination has no loaded table, so the guard
            # inside query() is sufficient.  No piece-count cap here.
```

### 9.2  Also probe in `choose_move()` for all available table sizes

Find in `choose_move()`:
```python
            if (board.pieces_placed.get("W", 0) >= 9
                    and board.pieces_placed.get("B", 0) >= 9
                    and _w_on <= 3 and _b_on <= 3
                    and _w_on + _b_on <= 6):
```

Replace with:
```python
            if (board.pieces_placed.get("W", 0) >= 9
                    and board.pieces_placed.get("B", 0) >= 9):
                # Let EndgameSolvedDB.query() handle piece-count validity;
                # it returns None when no table is available for the position.
```

This single change makes the AI use the 7v3, 6v4, 5v5, etc. tables wherever they exist, which completely bypasses heuristic search for those positions.

---

## 10  Stage 8 — Trajectory DB: deeper human-move search

The human trajectory DB (`HumanDB`) is already queried at root for move ordering. The SE-11b/11c extension probes 2 opponent plies deep. Extending this to depth 6 keeps the AI aligned with likely human continuations further into the game.

### 10.1  Increase `_MAX_OPP_PLIES` to 6

**File:** `ai/game_ai.py`  
Find:
```python
_MAX_OPP_PLIES = 2
```

Replace with:
```python
# When v2 heuristics are active, deeper trajectory extension is cheap because
# the leaf evaluator is much faster.  V1 keeps the original 2-ply limit.
_MAX_OPP_PLIES    = 2   # v1 default
_MAX_OPP_PLIES_V2 = 6   # v2: deeper human-move alignment
```

### 10.2  Use the correct constant in `_root_search` and `score_root_moves`

In `_root_search()`, find:
```python
                score_raw = -self._negamax(nb, depth - 1, -beta, -alpha_raw, None, depth // 2, _MAX_OPP_PLIES, 1)
```

Replace with:
```python
                _opp_plies = _MAX_OPP_PLIES_V2 if self.use_v2_heuristics else _MAX_OPP_PLIES
                score_raw = -self._negamax(nb, depth - 1, -beta, -alpha_raw, None, depth // 2, _opp_plies, 1)
```

Same substitution in `score_root_moves()`.

---

## 11  Stage 9 — Additional improvements

### 11.1  Internal iterative deepening (IID) at deep nodes

When a node has no TT best move and depth >= 5, run a reduced-depth search internally to get a move ordering hint before the full search.

**Insertion point:** in `_negamax()`, after TT lookup returns no hit and before move generation:

```python
        # IID: no TT hit at deep nodes -- run a quick shallow search for ordering
        _iid_best_move: str | None = None
        if (self.use_v2_heuristics
                and depth >= 5
                and tt_entry is None):
            try:
                _iid_score = -self._negamax(
                    board, depth - 3, -beta, -alpha,
                    endgame_state, 0, 0, ply
                )
                # The recursive call will have stored a TT entry -- look it up
                _iid_entry = self._tt.lookup(board.hash_key)
                if _iid_entry:
                    _iid_best_move = _iid_entry[3]  # to_sq
            except (_SearchAbort, Exception):
                pass
```

Then in move ordering, promote `_iid_best_move` to the front alongside TT best move.

### 11.2  SEE-style static capture ordering

In `_order_moves()`, when a capture is available, order captures before quiet moves. The current ordering already puts mill-closers first; add explicit capture promotion within the quiet/history bucket:

```python
        # Within P2 (quiet moves), sub-sort by: captures first, then history score.
        captures = [m for m in p2 if m.get("capture")]
        quiets   = [m for m in p2 if not m.get("capture")]
        if history and quiets:
            quiets.sort(key=lambda m: history.get((m.get("from"), m["to"]), 0), reverse=True)
        p2 = captures + quiets
```

### 11.3  Futility pruning at depth 1

At depth 1 (one ply from leaf), if the static evaluation is already so far below alpha that no single move can recover, skip full evaluation for quiet moves.

**Insertion point:** in `_negamax()`, near the depth-0 leaf return:

```python
        # Futility pruning at depth 1 (v2 only)
        _FUTILITY_MARGIN = 120   # tune: ~one mill threat's worth
        if (self.use_v2_heuristics
                and depth == 1
                and not terminal
                and abs(alpha) < INF // 2):
            static = evaluate_v2(board, board.turn, _ply=ply)
            if static + _FUTILITY_MARGIN <= alpha:
                return static   # no legal move can raise alpha sufficiently
```

### 11.4  Persist TT across `choose_move()` calls (pondering)

The current implementation calls `self._tt.clear()` at the start of every `choose_move()`. For the deep-search case, consider **not** clearing the TT and instead using age-based eviction:

```python
        # V2: do not clear TT between turns -- aged entries still help.
        # V1: always clear (original behaviour preserved).
        if not self.use_v2_heuristics:
            self._tt.clear()
        # else: TT persists; depth-preferred replacement handles staleness.
```

**Note:** This makes the AI state-dependent across turns. Only enable when `use_v2_heuristics = True` and after extensive testing.

### 11.5  Quiescence search depth increase

The codebase has a quiescence framework (SE-9). Verify that `qsearch` activates when `use_v2_heuristics = True`. The simplified leaf evaluator makes quiescence much cheaper, so the budget can be increased from depth 2 to depth 4:

```python
        _QSEARCH_DEPTH = 4 if self.use_v2_heuristics else 2
```

---

## 12  Stage 10 — Validation and old-code removal

### 12.1  Validation checklist

Run before removing any v1 code:

- [ ] AI self-play: 200 games v2 vs v1 at difficulty 7. Win rate within +-15% of 50/50 (expected: v2 slightly better due to depth).
- [ ] Tactical test suite: all `tests/test_tactics.py` puzzles solved by v2 at difficulty 6.
- [ ] Timing benchmark: `python -m cProfile -o v2.prof` shows `evaluate_v2` < 5% of total `_negamax` time.
- [ ] Endgame probe test: v2 at difficulty 5 correctly plays all 3v3 positions from `data/endgame_3_3.wdl`.
- [ ] Trajectory test: humanDB `query_line()` called at depth 6; verify no timeout regression.
- [ ] Aspiration window test: confirm no score oscillation (v2 at depth 14, 100 positions).
- [ ] Null-move test: confirm no horizon collapse in zugzwang positions.
- [ ] LMR test: verify re-search fires correctly on 10 sample positions (add debug print temporarily).

### 12.2  Old-code removal sequence

Only after all validation passes:

```bash
# 1. Remove evaluate() and all v1 helpers from heuristics.py
#    (keep _sealed_two_configs, _dual_connected_mill_alert, _closeable_mills
#     because they are still used in choose_move() filter logic)

# 2. Remove tactical_move_bonus() from heuristics.py and game_ai.py imports
#    (it is no longer called anywhere with use_v2_heuristics = True)
#    NOTE: _populate_thinking() already has a guard -- remove the guard and
#    replace with a simple self.last_thinking = "" or a v2-specific label.

# 3. Remove the use_v2_heuristics flag and make v2 the only code path.

# 4. Remove the HeuristicWeights class if weights are no longer used.
#    Keep DEFAULT_WEIGHTS if any opening adherence / book logic still references it.
```

Remove imports from `game_ai.py`:
```python
# DELETE after Stage 10:
# from .heuristics import (evaluate, clear_eval_cache, HeuristicWeights,
#                           DEFAULT_WEIGHTS, tactical_move_bonus, _sealed_two_configs,
#                           _dual_connected_mill_alert, _closeable_mills)
# ADD:
from .heuristics import (evaluate_v2, INF, _sealed_two_configs,
                          _dual_connected_mill_alert, _closeable_mills)
```

---

## 13  Weight reference table

| Term | Phase | Weight | Notes |
|------|-------|--------|-------|
| `_V2_PL_HAND` | place | 1 | Sanmill-style pieces in hand differential |
| `_V2_PL_PIECE` | place | 1 | Sanmill-style pieces on board differential |
| `_V2_PL_MOB` | place | 1 | Connectivity signal |
| `_V2_PL_BLOCKED` | place | 8 | Sanmill-style blocked-piece differential |
| `_V2_PL_MILL` | place | 30 | Closed mills |
| `_V2_PL_THREAT` | place | 15 | Closeable 2-config |
| `_V2_PL_REM` | place | 10 | Sanmill-style removable proxy = mills + closeable mills |
| `_V2_PL_POS` | place | 2 | Cardinal=3, cross=2, corner=1 |
| `_V2_MV_PIECE` | move | 12 | Material advantage |
| `_V2_MV_MOB` | move | 1 | Directional only (was x8) |
| `_V2_MV_BLOCKED` | move | 48 | Sanmill-style fully blocked opp piece |
| `_V2_MV_MILL` | move | 30 | Closed mills |
| `_V2_MV_THREAT` | move | 18 | Closeable threats |
| `_V2_MV_CYCLE` | move | 22 | Cycle-ready mills |
| `_V2_MV_FORK` | move | 14 | Diamond fork squares |
| `_V2_MV_SQUEEZE` | move | 30 | Near-blocked opp (1 free nb) |
| `_V2_MV_ZUGZ` | move | 600 | Per step below opp_mob=3 |
| `_V2_FLY_PIECE` | fly | 2 | Material (fewer pieces) |
| `_V2_FLY_MILL` | fly | 32 | Closed mills |
| `_V2_FLY_THREAT` | fly | 80 | Every 2-config is near-win |
| `_V2_FLY_CYCLE` | fly | 80 | Cycling mills dominate |
| `_V2_FLY_FORK` | fly | 55 | Unblockable forks |
| `_V2_FLY_WIN` | fly | 1190 | Opponent at <=3 pieces |
| `_V2_FLY_SURP` | fly | 900 | Fly threat surplus |

**Tuning note:** Placement now explicitly carries Sanmill-style terms for pieces in hand, pieces on board, blocked pieces, and a removable proxy. Start by tuning `_V2_PL_BLOCKED` and `_V2_PL_REM`, then `_V2_MV_CYCLE` and `_V2_MV_FORK` in 50-game self-play batches. The `_V2_FLY_*` weights are less sensitive because the endgame DB takes over for positions it covers.

---

*End of HeuristicsV2-plan.md*
