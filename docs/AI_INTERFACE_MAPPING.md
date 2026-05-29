# NMM Heuristic AI — Interface Mapping

This document maps every integration point that the learned-AI subsystem must hook into. It is the result of a full repository inspection performed before writing any learning code.

## Entry function (heuristic AI)

- **Class:** `GameAI`
- **File:** `ai/game_ai.py`
- **Constructor signature:**
  ```python
  GameAI(
      color: str = "B",                # "W" or "B"
      difficulty: int = 3,             # 1..10
      blunder_probability: float = 0.0,
      weights: HeuristicWeights | None = None,
      use_mcts: bool = False,
      value_net=None,
      fullgame_db=None,
      endgame_solved_db=None,
  )
  ```
- **Primary move-selection method:**
  ```python
  GameAI.choose_move(
      board: BoardState,
      recognition=None,
      endgame_state=None,
      trajectory_hints=None,
      top_n: int = 1,
      fast_early_game: bool = False,
      force_book_early: bool = False,
      fullgame_db=None,
  ) -> dict
  ```
  Returns a complete move dict (see *Move representation* below). Called from `main.py` (line 192) via `game_ai.choose_move(board)` and from `ai/coordinator.py` via `Coordinator.deliberate(board)`.

The learned agent (`learned_ai.agents.learned_agent.LearnedAgent`) exposes the same surface:
- Constructor with `color: str` (W/B).
- `choose_move(board, **kwargs) -> dict` returning the same move dict shape and ignoring unused kwargs so it is a drop-in replacement.

## Board state object

- **Class:** `BoardState` (dataclass, immutable)
- **File:** `game/board.py`
- **Fields:**
  - `positions: Dict[str, str]` — 24 entries keyed by algebraic position name (`"a1".."g7"`), value is `"W"`, `"B"`, or `""`.
  - `turn: str` — side to move (`"W"` or `"B"`).
  - `pieces_on_board: Dict[str, int]` — `{"W": n, "B": n}`.
  - `pieces_placed: Dict[str, int]` — cumulative placements per side (caps at 9).
  - `pieces_captured: Dict[str, int]` — count of opponent pieces this side has captured.
  - `hash_key: int` — Zobrist hash, maintained incrementally.

The 24 position names are defined in `game.board.POSITIONS` and are ordered:
- Outer ring: `a7 d7 g7 g4 g1 d1 a1 a4`
- Middle ring: `b6 d6 f6 f4 f2 d2 b2 b4`
- Inner ring: `c5 d5 e5 e4 e3 d3 c3 c4`

The state encoder uses this exact ordering for its 24×3 one-hot block so that index `i` always refers to `POSITIONS[i]`.

Adjacency and mill geometry come from `game.board.ADJACENCY` and `game.board.MILLS` (16 mills total: 12 ring sides + 4 cross-ring lines).

## Move representation

A "move" is a plain `dict` with three keys:
```python
{"from": Optional[str], "to": str, "capture": Optional[str]}
```
- **Placement:** `from is None`, `to` is the destination square.
- **Movement / fly:** `from` is the source square, `to` is the destination.
- **Capture:** if the placement/movement closes a mill, `capture` is the opponent square removed; otherwise `None`.

A complete game-loop legal move always includes the capture (or `None`) — see `game.rules.get_all_legal_moves` which expands mill-forming partials into one move per legal capture target.

## Phase detection

- **Function:** `game.rules.get_game_phase(board: BoardState, color: str) -> str`
- Returns one of: `"place"`, `"move"`, `"fly"`.
  - `"place"` while `board.pieces_placed[color] < 9`.
  - `"fly"` when `pieces_placed[color] == 9` and `pieces_on_board[color] <= 3`.
  - `"move"` otherwise.

The learned-AI subsystem layers a finer 5-phase classification (IDs 0..4) on top of this:

| ID | Name              | Trigger                                                        |
|----|-------------------|----------------------------------------------------------------|
| 0  | opening_placement | side-to-move is still placing AND `pieces_placed[stm] < 4`     |
| 1  | full_placement    | side-to-move is still placing AND `4 <= pieces_placed[stm] <= 9` |
| 2  | midgame           | both sides done placing AND both `pieces_on_board >= 4`        |
| 3  | endgame           | both done placing AND stm has 3-4 pieces AND not yet flying    |
| 4  | flying            | side-to-move `can_fly` (placed all 9 and has exactly 3 pieces) |

Phase classification lives in `learned_ai/models/state_encoder.py::detect_phase`.

## Side-to-move encoding

- The current side is read from `board.turn` (`"W"` or `"B"`).
- The state vector encodes it as a single float: `0.0` for white, `1.0` for black.
- This intentionally avoids per-color models — the network conditions on the bit and learns symmetric play through experience.

## Legal-move generation (single source of truth)

- **Function:** `game.rules.get_all_legal_moves(board: BoardState) -> List[dict]`
- Returns *complete* moves (with capture expanded). The learned agent uses this directly and never duplicates rules logic.
- Auxiliary helpers used internally by encoders:
  - `BoardState.legal_placements(color) -> List[str]`
  - `BoardState.legal_moves(color) -> List[Tuple[str, str]]`
  - `BoardState.legal_captures(color) -> List[str]`
  - `game.rules.does_form_mill(board, move) -> bool`

## Existing tests

Located in `tests/`:
- `test_board.py` — board, mills, adjacency, FEN, notation.
- `test_ai.py` — heuristic AI behaviour.
- `test_blocking.py`, `test_tactics.py`, `test_b39_b44.py`, `test_search_enhancements.py`, `test_stage{12,3,4,5,6}.py`, `test_transposition_table.py`, `test_endgame_solved_db.py`, `test_fullgame_db.py`, `test_build_endgame_db.py`.

Tests are runnable directly with `python -m tests.test_<name>` because each test file prepends the repo root to `sys.path` and uses `unittest`. The new `tests/test_*.py` files follow that same convention.

## Existing AI / LLM integration points

- `ai/coordinator.py::Coordinator` — orchestrates heuristic AI, LLM commentary, opening/endgame recognisers, and memory. Calls `game_ai.choose_move(board)` internally.
- `ai/mills_llm.py` — Ollama-based LLM commentary; not touched by the learned-AI subsystem.
- `ai/value_net.py` — small numpy-based value network used by MCTS leaf evaluation. Distinct from the PyTorch network introduced in `learned_ai/models/backbone.py`.
- `ai/mcts.py`, `ai/transposition_table.py`, `ai/board_symmetry.py`, `ai/heuristics.py` — minimax/MCTS scaffolding.

## How the learned AI slots in

`main.py` and `web/app.py` instantiate `GameAI` for the AI side. The migration path (see `docs/MIGRATION_GUIDE.md`) adds an `NMM_AI_ENGINE` environment variable to select between `"heuristic"` (default, unchanged behaviour) and `"learned"` (which constructs a `LearnedAgent` loaded from `NMM_LEARNED_CHECKPOINT`). The learned agent's `choose_move(board)` returns the same move-dict shape, so callers — including `Coordinator.deliberate` — do not need to change.
