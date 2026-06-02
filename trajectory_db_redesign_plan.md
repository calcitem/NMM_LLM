# TrajectoryDB Redesign: Board-State-First Architecture

## Claude Implementation Brief — NMM_LLM Project

**Repository:** `https://github.com/benmarkbrandwood-blip/NMM_LLM`  
**Scope:** Redesign `ai/trajectory_db.py` and all integration points.

---

## Design Decisions (Confirmed)

| Decision | Choice |
|----------|--------|
| Position strength evaluator | **Extend `evaluate()`** with `strength_mode` flag — no new file |
| Per-move blame/reward storage | **Compute inline** at `_index_game()` time — no `data/games_enriched/` |
| Hard move bans | **Removed** — soft scoring only |
| Rollout order | **Core-first**: board-state keying ships first; follow-ons after |

---

## What's Wrong Now

The current `TrajectoryDB` indexes games by pipe-joined move-notation prefixes at checkpoint depths `(4, 6, 8, …, 48)`. The same board position reached by two different legal move sequences produces two completely separate DB entries. Learned knowledge doesn't transfer across transpositions.

Additionally, every move in a losing game is penalised equally — opening moves that were objectively fine are blamed as much as the actual blunder.

---

## Target Architecture

**Primary key:** canonical board-state key (D4-normalised position + turn + phase + placed counts).  
Two boards reached by different paths that look identical on the board share one DB bucket.

**Per-entry stats:** win/loss/draw counts (split by source: AI-vs-AI vs human-involved), plus blame and reward sums computed inline from position-strength deltas.

**Query interface:** takes a `BoardState` object, returns `{notation: float}` score-delta dict.  
No hard bans. Soft scoring only.

---

## Blocker: `BoardState.from_fen_string()` Does Not Exist

Every move record in the JSONL files already has `board_fen_before` in the format:

```
<24-char position string>|<turn W/B>|<W_placed>|<B_placed>
```

This is produced by `BoardState.to_fen_string()`. But there is no `from_fen_string()` / `from_fen()` parser. **This must be implemented before Phase 1 can begin.**

---

## Phase 1 (Core): Board-State Keying

**Goal:** Ship the fundamental swap — sequence-prefix keys → canonical board-state keys — with the simplest possible stats (win/loss/draw counts only, no blame/reward yet). All callers updated. Tests pass.

### Step 1a: `game/board.py` — Add `from_fen_string()`

`BoardState` is a dataclass with fields: `positions`, `turn`, `pieces_on_board`, `pieces_placed`, `pieces_captured`, `hash_key`. The FEN string stores position, turn, and placed counts; the rest are derived.

```python
@classmethod
def from_fen_string(cls, fen: str) -> "BoardState":
    """Parse a FEN string produced by to_fen_string().
    Format: '<24 chars>|<turn>|<W_placed>|<B_placed>'
    """
    from game.board import hash_board  # Zobrist hash function
    pos_str, turn, w_placed_s, b_placed_s = fen.split("|")
    w_placed, b_placed = int(w_placed_s), int(b_placed_s)
    positions = {POSITIONS[i]: (pos_str[i] if pos_str[i] != "." else "") for i in range(24)}
    w_on = sum(1 for v in positions.values() if v == "W")
    b_on = sum(1 for v in positions.values() if v == "B")
    board = cls(
        positions=positions,
        turn=turn,
        pieces_on_board={"W": w_on, "B": b_on},
        pieces_placed={"W": w_placed, "B": b_placed},
        # pieces captured *by* color = opponent pieces placed − opponent pieces still on board
        pieces_captured={"W": b_placed - b_on, "B": w_placed - w_on},
        hash_key=0,
    )
    board = board.__class__(  # reuse same fields, set hash
        **{**board.__dict__, "hash_key": hash_board(board)}
    )
    return board
```

Simpler alternative if dataclass supports `replace()`:
```python
    import dataclasses
    board_no_hash = cls(positions=positions, turn=turn,
                        pieces_on_board={"W": w_on, "B": b_on},
                        pieces_placed={"W": w_placed, "B": b_placed},
                        pieces_captured={"W": b_placed - b_on, "B": w_placed - w_on},
                        hash_key=0)
    return dataclasses.replace(board_no_hash, hash_key=hash_board(board_no_hash))
```

Verify with a round-trip test: `BoardState.from_fen_string(b.to_fen_string())` reproduces `b` field-for-field.

### Step 1b: `ai/trajectory_db.py` — Add `make_board_state_key()`

`get_game_phase` lives in `game.rules` (it's already imported by `ai/heuristics.py` from there).

```python
import math
from ai.board_symmetry import (
    canonical_board_str as _canonical_board_str,
    transform_notation as _transform_notation,
    SYM_INVERSE as _SYM_INVERSE,
)
from game.board import POSITIONS
from game.rules import get_game_phase

def make_board_state_key(board: "BoardState") -> tuple[str, int]:
    """Return (canonical_state_key, sym_idx) for this board under D4 symmetry.

    sym_idx must be retained by callers so that notation strings can be
    transformed consistently: stored notations are in canonical space;
    query results are mapped back to actual-game notation via SYM_INVERSE[sym_idx].
    """
    board24 = "".join(board.positions.get(p, "") or "." for p in POSITIONS)
    canon, sym_idx = _canonical_board_str(board24)
    phase = get_game_phase(board, board.turn)
    placed_w = board.pieces_placed.get("W", 0)
    placed_b = board.pieces_placed.get("B", 0)
    state_key = f"{canon}|{board.turn}|{phase}|{placed_w}|{placed_b}"
    return state_key, sym_idx
```

### Step 1c: Rewrite `_index_game()`

Replace the checkpoint-depth loop entirely. **Critical:** the move notation must be transformed into canonical (D4) space using the same `sym_idx` as the board key — otherwise two transposed boards with the same canonical `state_key` will fragment their move stats under different notation strings.

```python
def _index_game(self, record: dict) -> None:
    if record.get("adaptive_softened"):
        return
    winner = record.get("winner")
    moves = record.get("moves", [])
    if not moves:
        return

    source_type = record.get("source_type")
    if source_type is None:
        if record.get("self_play") or (
            record.get("white_difficulty") and record.get("black_difficulty")
            and not record.get("human_color")
        ):
            source_type = "ai_vs_ai"
        else:
            source_type = "human_involved"
    is_ai = (source_type == "ai_vs_ai")

    self._game_count += 1

    for move in moves:
        notation = _norm(move.get("notation", ""))
        fen = move.get("board_fen_before", "")
        if not notation or not fen:
            continue
        try:
            board = BoardState.from_fen_string(fen)
        except Exception:
            continue

        state_key, sym_idx = make_board_state_key(board)
        # Transform notation into canonical space (same D4 transform as the board key).
        canon_notation = _transform_notation(notation, sym_idx)
        if canon_notation is None:
            continue

        color = move.get("color", "W")

        bucket = self._index.setdefault(state_key, {})
        entry = bucket.setdefault(canon_notation, {
            "wins_ai": 0, "losses_ai": 0, "draws_ai": 0,
            "wins_human": 0, "losses_human": 0, "draws_human": 0,
            "total": 0,
            # blame_sum / reward_sum are always 0 until Phase 3 adds inline enrichment.
            "reward_sum": 0.0, "blame_sum": 0.0,
        })
        entry["total"] += 1

        if winner == color:
            entry["wins_ai" if is_ai else "wins_human"] += 1
        elif winner is not None and winner != color:
            entry["losses_ai" if is_ai else "losses_human"] += 1
        else:
            entry["draws_ai" if is_ai else "draws_human"] += 1
```

### Step 1d: Rewrite `query()`, `query_opponent_loss()`, `query_all_frequencies()`

All three now take `board: BoardState` as their first argument instead of `move_notations: list[str]`.

**`query()`:**

Query time: compute `sym_idx` from the live board, look up the canonical bucket, then map each stored `canon_notation` back to the actual game notation via the inverse D4 transform.

```python
def query(
    self,
    board: "BoardState",
    current_color: str,
    min_samples: int = 3,
    prefer_ai: bool = False,
) -> dict[str, float]:
    """Return score-delta dict for candidate next moves from this board state.

    Score in [-0.5, +0.5]:
      +0.5 = historically always wins for current_color
      -0.5 = historically always loses

    Confidence-weighted: low-sample positions return smaller deltas.
    Returns {} when no data or fewer than min_samples total moves at this state.
    """
    state_key, sym_idx = make_board_state_key(board)
    candidates = self._index.get(state_key)
    if not candidates:
        return {}

    inv = _SYM_INVERSE[sym_idx]
    result: dict[str, float] = {}
    for canon_notation, stats in candidates.items():
        total = stats["total"]
        if total < min_samples:
            continue

        # Map canonical notation back to actual-game notation.
        actual_notation = _transform_notation(canon_notation, inv)
        if actual_notation is None:
            continue

        if prefer_ai:
            wins = stats["wins_ai"]   + 0.5 * stats["wins_human"]
            draws = stats["draws_ai"] + 0.5 * stats["draws_human"]
            eff  = max(1, stats["wins_ai"] + stats["losses_ai"] + stats["draws_ai"]
                       + 0.5 * (stats["wins_human"] + stats["losses_human"] + stats["draws_human"]))
        else:
            wins  = stats["wins_ai"]  + stats["wins_human"]
            draws = stats["draws_ai"] + stats["draws_human"]
            eff   = max(1, total)

        win_rate = (wins + 0.4 * draws) / eff
        raw = win_rate - 0.5

        # Blend blame/reward signal (always 0 until Phase 3; harmless here).
        avg_blame  = stats["blame_sum"]  / total
        avg_reward = stats["reward_sum"] / total
        adjusted = raw - avg_blame * 0.4 + avg_reward * 0.3

        # Confidence: shrinks delta when sample count is low (reaches 1.0 at ~20 samples).
        confidence = min(1.0, math.log(total + 1) / math.log(20))

        result[actual_notation] = max(-0.5, min(0.5, adjusted * confidence))

    return result
```

**`query_opponent_loss()`:** Same pattern — take `board: BoardState`, call `make_board_state_key(board)`, apply inverse transform on results, score by opponent-loss rate.

**`query_all_frequencies()`:**

```python
def query_all_frequencies(
    self,
    board: "BoardState",
    min_samples: int = 5,
) -> dict[str, float]:
    state_key, sym_idx = make_board_state_key(board)
    candidates = self._index.get(state_key)
    if not candidates:
        return {}
    total_all = sum(c["total"] for c in candidates.values())
    if total_all < min_samples:
        return {}
    inv = _SYM_INVERSE[sym_idx]
    result = {}
    for canon_n, c in candidates.items():
        if c["total"] == 0:
            continue
        actual_n = _transform_notation(canon_n, inv)
        if actual_n:
            result[actual_n] = c["total"] / total_all
    return result
```

### Step 1e: Remove the ban system

Delete `_bans`, `mark_bad_move()`, `load_bad_moves()`, `save_bad_move()`, and all `bad_moves_path` parameters. Remove the `load()` call to `load_bad_moves()`. Remove ban-related logic from `query()`.

Any callers that currently call `trajectory_db.save_bad_move(...)` or `mark_bad_move(...)` must have those calls removed (check `ai/coordinator.py`).

### Step 1f: Update callers

**`ai/move_guidance.py`:**
- `build_trajectory_hints()`: change `trajectory_db.query(notations, board.turn)` → `trajectory_db.query(board, board.turn)`
- `query_opponent_loss()`: change to `trajectory_db.query_opponent_loss(board, opp_color)`
- Remove `game_notations` from trajectory calls (no longer needed for keying)
- Keep `game_notations` parameter for SE-11 compatibility if still needed elsewhere

**`ai/coordinator.py`:**
- `on_game_end()`: no change needed — `add_game(record)` still works
- Remove any `save_bad_move` / `mark_bad_move` calls
- Remove `bad_moves_path` from `TrajectoryDB.load()` calls

**`ai/game_ai.py`:**
- SE-11 `query_all_frequencies()` calls: replace `self._trajectory_db.query_all_frequencies(game_notations + [root_mn], ...)` with `self._trajectory_db.query_all_frequencies(nb, ...)` where `nb` is the board state *after* applying the root move. (`nb` is already computed as `board.apply_move(move)` in those loops.)

### Step 1g: Backward compatibility during transition

When loading old (pre-redesign) JSONL files that lack `board_fen_before` on moves, `_index_game()` silently skips those moves (the `if not fen: continue` guard handles this). The DB will be sparse initially and fill as new games are played.

---

## Phase 2 (Follow-on): Extend `evaluate()` with Strength Mode

**File:** `ai/heuristics.py`

Add `strength_mode: bool = False` to `evaluate()`:

```python
def evaluate(
    board: BoardState,
    color: str,
    weights: dict | None = None,
    endgame_state: int | None = None,
    strength_mode: bool = False,
) -> float:
    ...
    raw = <existing score computation>
    if strength_mode:
        # Return tanh-normalised [-1, +1] from color's perspective
        scale = {"place": 12.0, "move": 18.0, "fly": 25.0}.get(phase, 15.0)
        return math.tanh(raw / scale)
    return raw
```

This is the signal used by the GUI strength graph and by the inline blame/reward computation in Phase 3.

**Tests:** verify `evaluate(board, "W", strength_mode=True)` returns a value in `[-1, 1]` on a range of board states; symmetry: `evaluate(b, "W", strength_mode=True) == -evaluate(b, "B", strength_mode=True)` on a neutral board (approximately).

---

## Phase 3 (Follow-on): Inline Blame/Reward in `_index_game()`

Once `evaluate(strength_mode=True)` exists, extend `_index_game()` to compute per-ply blame/reward at index time.

**Algorithm:**

For each game, after computing the per-ply strength trace:
1. Build loser's strength trace across all plies.
2. Find the turning-point ply: the loser's ply where strength was ≥ −0.15 and dropped ≥ 0.12 in the next 2 loser-plies, and did not recover above −0.10 afterwards.
3. Assign blame weights: 1.0 at turning point, 0.6 one loser-move before, 0.3 two before, 0.2 one after (if not yet in forced-defense). Reduce to 0.1 if the move was forced (legal_move_count ≤ 1). No blame before the window.
4. Assign reward weights to winner moves: `min(1.0, strength_delta * 4.0)` when delta > 0.05; minimum 0.4 for the last 4 winner moves.
5. Store accumulated `blame_sum` and `reward_sum` in the DB entry.

**`_index_game()` extension (sketch):**

```python
# After the existing per-move loop, add a second pass for blame/reward:
def _compute_blame_reward(moves, winner):
    """Return (blame_weights[], reward_weights[]) indexed by move list position."""
    ...  # implement turning-point logic described above

blame_wts, reward_wts = _compute_blame_reward(valid_moves, winner)
for i, (move, blame, reward) in enumerate(zip(valid_moves, blame_wts, reward_wts)):
    state_key = move["_state_key"]   # cached from first pass
    notation  = move["_notation"]
    entry = self._index[state_key][notation]
    entry["blame_sum"]  += blame
    entry["reward_sum"] += reward
```

To avoid replaying all FEN positions twice, cache `(state_key, notation)` tuples during the first pass before the blame/reward pass.

---

## Phase 4 (Follow-on): `query_line()` — Multi-Ply Guidance

Add a new `query_line()` method once Phase 1 is stable:

```python
def query_line(
    self,
    board: "BoardState",
    k: int = 4,
    min_samples: int = 3,
) -> list[tuple[str, float]]:
    """Top-k historically likely next moves, sorted by score descending.
    Used for root move ordering and SE-11b depth extension."""
    scores = self.query(board, board.turn, min_samples=min_samples)
    filtered = [(n, s) for n, s in scores.items()]
    filtered.sort(key=lambda x: x[1], reverse=True)
    return filtered[:k]
```

Use in `ai/game_ai.py` to promote trajectory-top moves to the front of the root move list in `_root_search()`.

---

## Phase 5 (Follow-on): GUI Position Strength Graph

**File:** locate the web handler that computes graph data (check `web/` directory).

Replace current strength calculation with `evaluate(board, color, strength_mode=True)`.  
Apply 2-ply moving-average smoothing before sending to frontend.  
Return both White and Black traces.  
After game ends, compute the turning-point ply and return it as `turning_point_ply` in the graph payload for a visual blunder marker.

---

## Phase 6 (Follow-on): Tests

**File:** `tests/test_trajectory_db_v2.py`

| Test | What it checks |
|------|----------------|
| `test_transpositions_share_bucket` | Same board via two paths → one DB entry |
| `test_early_moves_not_penalised` | Blame at ply 1–8 is < 0.1 when turning point is ply 20 |
| `test_divergent_game_penalises_losing_branch` | Two games share 7 moves then diverge; losing branch move gets negative delta |
| `test_forced_move_low_blame` | legal_move_count == 1 → blame_weight ≤ 0.1 |
| `test_low_sample_reduced_influence` | Same win rate with 2 vs 30 samples: high-sample gives delta closer to ±0.5 |
| `test_source_type_separation` | AI-vs-AI and human wins stored in separate counters |

---

## File-by-File Summary

| File | Action | Phase |
|------|--------|-------|
| `game/board.py` | **Edit** — add `from_fen_string()` | 1 (blocker) |
| `ai/trajectory_db.py` | **Rewrite** — board-state keys, new query interface, remove bans | 1 |
| `ai/move_guidance.py` | **Edit** — pass `board` not `notations` to all trajectory calls | 1 |
| `ai/coordinator.py` | **Edit** — remove ban calls, keep `add_game()` as-is | 1 |
| `ai/game_ai.py` | **Edit** — update `query_all_frequencies()` to pass `nb` (board after move) | 1 |
| `ai/heuristics.py` | **Edit** — add `strength_mode` flag to `evaluate()` | 2 |
| `ai/trajectory_db.py` | **Edit** — inline blame/reward in `_index_game()` | 3 |
| `ai/trajectory_db.py` | **Edit** — add `query_line()` | 4 |
| `web/` handler | **Edit** — use evaluate(strength_mode=True) for graph | 5 |
| `tests/test_trajectory_db_v2.py` | **Create** — 6 validation tests | 6 |

**Do not create:**
- `ai/position_strength.py` — replaced by `evaluate(strength_mode=True)`
- `ai/blame_analyzer.py` — logic lives in `_index_game()`
- `scripts/enrich_games.py` — no disk enrichment files
- `data/games_enriched/` — no enriched game directory

---

## Implementation Order

```
Session 1:  Phase 1 (core) — from_fen_string(), make_board_state_key(),
            rewrite _index_game() / query() / query_opponent_loss() /
            query_all_frequencies(), remove bans, update all callers.
            Run full test suite. Ship.

Session 2:  Phase 2 — evaluate(strength_mode=True).
            Phase 3 — inline blame/reward in _index_game().

Session 3:  Phase 4 — query_line() + move ordering in game_ai.
            Phase 5 — GUI graph update.
            Phase 6 — tests.
```

---

## Notes for Implementation

- `canonical_board_str()` and `transform_notation()` and `SYM_INVERSE` all exist in `ai/board_symmetry.py`.
- `get_game_phase()` is in `game/rules.py` — import from there, not from `ai/heuristics.py`.
- `POSITIONS` (ordered list of 24 square names) is in `game/board.py` — import it for `make_board_state_key()`.
- All 317 current JSONL game files have `board_fen_before` populated on every move — full coverage confirmed.
- `source_type` field does not exist in current JSONL records — derive it from `self_play` + `human_color` as shown in Step 1c.
- Keep `game_notations` passed through `choose_move()` kwargs for now — SE-11 depth extension may still want recent move context as a secondary tie-breaker even after Phase 1 ships.
- Heuristic versioning (`HEURISTIC_VERSION = 1`): when `evaluate()` weights change significantly, increment this constant and add a guard in `_index_game()` to skip records annotated with an older version.

**Sparse index during warmup:** The board-state-keyed DB will return `{}` for most positions until duplicate board states accumulate across games. With `min_samples=3`, the AI loses trajectory guidance initially — this is expected and correct. The DB rebuilds itself over subsequent games. If you want trajectory guidance immediately after the switch, lower `min_samples=1` temporarily; the confidence weighting (`log(total+1)/log(20)`) keeps low-sample influence small automatically.
