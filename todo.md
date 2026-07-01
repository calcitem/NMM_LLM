# Search Performance TODO

## Switch board.positions from dict to list

**Why:** `apply_move` calls `dict(self.positions)` on every search node to create a
24-entry copy. Python list copy is 2–3× faster than dict copy for the same number of
elements. With ~70K+ `apply_move` calls per second of search time, this is a
meaningful bottleneck.

**What changes:**
- `board.py`: change `positions: Dict[str, str]` to `positions: List[str]`, indexed by
  `SQ_INDEX[pos]` (the existing Zobrist square index, 0–23). Add a module-level
  `POSITIONS_LIST = list(POSITIONS)` for iteration.
- `ADJACENCY`: change values from `List[str]` to `List[int]` (index-based). Or keep
  str-based adjacency but add an index-based variant `ADJACENCY_IDX`.
- `apply_move`: `new_pos = list(self.positions)` instead of `dict(self.positions)`.
  Position updates become `new_pos[SQ_INDEX[sq]] = value`.
- All callers that do `board.positions[sq]` become `board.positions[SQ_INDEX[sq]]`
  or use a new accessor helper.

**Scope:** Touches board.py, rules.py, heuristics.py, game_ai.py, and any tool/test
that reads `board.positions[pos]` directly. Estimate ~20 files.

**Risk:** High refactor surface — every position lookup in the codebase changes form.
Needs a full test run after. Do not mix with other changes.

**Expected gain:** ~2–3× speedup on `apply_move` (173ms → ~70ms in 5s search profile).
Likely adds another ply at L8 on top of the move-gen and eval improvements already made.
