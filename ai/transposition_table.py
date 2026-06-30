"""
ai/transposition_table.py — Fixed-size Zobrist-keyed transposition table.

Each slot stores (hash_key, depth, score, flag, from_sq, to_sq).
Collision policy: depth-preferred replacement (only overwrite if the new entry
searched at least as deep as the stored entry).

Flag values
-----------
EXACT       — the stored score is the exact minimax value for this position
LOWER_BOUND — the search failed high (beta cutoff); stored score is a lower bound
UPPER_BOUND — the search failed low (all moves were bad); stored score is an upper bound

Usage in _negamax
-----------------
    entry = tt.lookup(board.hash_key)
    if entry:
        depth, score, flag, from_sq, to_sq = entry
        if depth >= remaining_depth:
            if flag == EXACT:           return score
            if flag == LOWER_BOUND and score >= beta:  return score
            if flag == UPPER_BOUND and score <= alpha: return score
        # Use (from_sq, to_sq) to order the best move first regardless.

    ... search ...

    tt.store(board.hash_key, remaining_depth, value, flag, best_from, best_to)
"""
from __future__ import annotations

# Flag constants
EXACT       = 0
LOWER_BOUND = 1
UPPER_BOUND = 2

# Table size must be a power of two (bitmask index).
# 2**21 = 2 097 152 slots — two-tier: depth-preferred primary + always-replace secondary.
# Secondary slot offset (_ALT_OFFSET) is half the table size; both slots are in the same
# array so the list stays a single allocation.  Typical memory: < 100 MB (only touched
# slots allocate tuple objects); worst-case ~450 MB if every slot is filled.
#
# NOTE: this resize applies to both v1 and v2 searches — both benefit.
_TABLE_SIZE = 1 << 21
_MASK       = _TABLE_SIZE - 1
_ALT_OFFSET = _TABLE_SIZE >> 1   # secondary-slot XOR offset within the same array


class TranspositionTable:
    """Fixed-size two-tier transposition table (depth-preferred + always-replace)."""

    __slots__ = ("_table",)

    def __init__(self) -> None:
        self._table: list = [None] * _TABLE_SIZE

    def clear(self) -> None:
        """Reset all slots.  Called at the start of each choose_move() call."""
        self._table = [None] * _TABLE_SIZE

    def lookup(self, hash_key: int):
        """Return (depth, score, flag, from_sq, to_sq) or None on miss.

        Checks primary (depth-preferred) slot first, then secondary (always-replace).
        """
        idx = hash_key & _MASK
        entry = self._table[idx]
        if entry is not None and entry[0] == hash_key:
            return entry[1:]
        alt_idx = (hash_key ^ _ALT_OFFSET) & _MASK
        entry = self._table[alt_idx]
        if entry is not None and entry[0] == hash_key:
            return entry[1:]
        return None

    def store(
        self,
        hash_key: int,
        depth: int,
        score: int,
        flag: int,
        from_sq: str | None,   # None for placement moves
        to_sq: str,
    ) -> None:
        """Two-deep replacement: depth-preferred primary + always-replace secondary."""
        idx = hash_key & _MASK
        existing = self._table[idx]
        if existing is None or depth >= existing[1]:
            self._table[idx] = (hash_key, depth, score, flag, from_sq, to_sq)
        else:
            alt_idx = (hash_key ^ _ALT_OFFSET) & _MASK
            self._table[alt_idx] = (hash_key, depth, score, flag, from_sq, to_sq)
