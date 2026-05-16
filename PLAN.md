# Nine Men's Morris — Project Plan

## Stage Summary

| Stage | Title                        | Status      |
|-------|------------------------------|-------------|
| 1     | Core Game Engine             | ✅ Complete |
| 2     | Classical AI (Minimax)       | ✅ Complete |
| 3     | Memory & LLM Layer           | ✅ Complete |
| 4     | Opening Book                 | ✅ Complete |
| 5     | Web GUI                      | ✅ Complete |
| 5.5   | Install & Run Scripts        | ✅ Complete |
| 6     | Self-Play Training Loop      | 🔄 Current  |
| 7     | Heuristic Parameter Evolution| ⬜ Planned  |
| 8     | Adaptive Difficulty          | ⬜ Planned  |
| 9     | Tournament / Match Mode      | ⬜ Planned  |
| 10    | Advanced Search (MCTS / NN)  | ⬜ Stretch  |

---

## Completed Stages

### Stage 1 — Core Game Engine ✅

Full Nine Men's Morris rules in pure Python with no external dependencies.

**Delivered:**
- `game/board.py` — Immutable `BoardState` dataclass; `apply_move()` returns a new state (enables safe undo and MCTS branching).
- `game/rules.py` — `get_all_legal_moves()`, `is_terminal()`, `does_form_mill()`, phase detection (placement / movement / flying).
- `game/game_engine.py` — Mutable `GameEngine` wrapper; records every move into `game_record` with FEN, notation, turn metadata.
- `game/notation.py` — Algebraic notation helpers.
- 57 tests pass.

---

### Stage 2 — Classical AI (Minimax) ✅

Negamax with alpha-beta pruning, iterative deepening, and blunder injection.

**Delivered:**
- `ai/game_ai.py` — `GameAI` with `choose_move()`, `score_move()`, `position_eval()`.
  - Difficulties 1–8 map to fixed depths (2–9 ply).
  - Difficulties 9–10 use iterative deepening with 20 s and 45 s time budgets.
  - `blunder_probability` selects a worst-quartile move intentionally (teaching aid).
- `ai/heuristics.py` — Phase-aware static evaluation:
  - Mill count, blocked pieces, piece count, two-configurations.
  - Double-mill pivots, win configuration (opponent in fly phase).
  - Mobility difference, immediate mill threats, positional value (cross/cardinal nodes score 3; corner nodes score 2).
  - tanh normalisation with per-phase scale (`place=120`, `move=180`, `fly=280`).
- `ai/endgame_recognizer.py` — Detects midgame → endgame → deep-endgame transitions, mill-cycle patterns, and zugzwang risk; boosts search depth in critical positions.
- 74 tests pass.

---

### Stage 3 — Memory & LLM Layer ✅

Local Ollama LLM (MillsAI) that comments on moves, chats with the player, and accumulates game history.

**Delivered:**
- `ai/memory_manager.py` — ChromaDB vector store for bad-move memory and strategy snippets; JSONL game log in `data/games/`; session narratives in `data/session_memory/`.
- `ai/mills_llm.py` — Ollama interface:
  - `ask_for_move_opinion()` — strict `MOVE: / REASON:` format enforcement; auto-retry on parse failure.
  - `evaluate_human_move()` — comments when score drops > threshold (capped per game).
  - `player_chat()` — multi-turn in-game conversation with context history.
  - `summarise_session()`, `name_novel_opening()`, `debrief_game()`.
  - Reads last 10 games before each new game.
- `ai/coordinator.py` — Orchestrates GameAI + MillsLLM:
  - `deliberate()` — GameAI picks best move, LLM recommends, coordinator adopts LLM move if it beats engine score + bonus threshold.
  - `react_to_human_move()` — scores move, emits poor-move comment if warranted.
  - `on_game_start()` / `on_game_end()` — lifecycle with opening book integration.
- 34 tests pass (includes stages 1–2).

---

### Stage 4 — Opening Book ✅

Curated opening library with UCB1-scored selection and D4 symmetry recognition.

**Delivered:**
- `ai/opening_book.py` — `OpeningBook` (read from `data/openings/book_openings.json`; writes to `data/openings/openings.json`):
  - UCB1 selection: `score + C * sqrt(log(N) / (n_i + 1))`, exploration rate 0.25.
  - Per-opening win/loss/draw stats split by human/AI side.
  - Novel opening auto-save with LLM-generated name.
- `ai/opening_recognizer.py` — Real-time recognition during placement phase:
  - Full D4 dihedral group (4 rotations × 4 reflections, 8 symmetries).
  - Provides `book_move`, `strategic_notes`, `common_blunders` context to coordinator.
  - Detects deviations; records branches.
- `tools/import_openings.py` — Imported curated openings from `strategy_book.txt`.
- `tools/teach_opening.py`, `tools/list_openings.py` — Maintenance tools.

---

### Stage 5 — Web GUI ✅

Full browser-based interface over FastAPI + WebSockets, with no page reloads.

**Delivered:**
- `web/app.py` — FastAPI server with `/ws` WebSocket endpoint:
  - Messages: `new_game`, `move`, `capture`, `undo`, `player_message` (client→server).
  - Messages: `state`, `capture_required`, `thinking`, `ai_move`, `commentary`, `game_over`, `error` (server→client).
  - Board state history for undo; `projected_board` preview during mill-capture sequence.
- `web/static/board.js` — Pure-JS SVG board:
  - Three concentric squares + 4 cross connections; coordinate labels (a–g, 1–7).
  - Layer order: bg → lines → labels → nodes → **pieces → hints** (hints above pieces so capture rings intercept clicks).
  - Colour-coded hints: green = legal placements, yellow = selectable, red = capturable.
  - Mill flash on capture.
- `web/static/game.js` — Game logic and UI:
  - Real-time eval history SVG graph (no external library).
  - Player chat with MillsAI; sent as `player_message` WebSocket messages.
  - Undo button; disabled when no history.
  - Info panel: pieces placed and pieces taken (not on-board count).
  - Settings: colour (White / Black / Random), opponent (AI / Human), difficulty 1–10, LLM toggle.
- `web/static/style.css` — Dark wooden theme (`--bg: #1a1510`), responsive two-column layout.
- `web/templates/index.html` — Jinja2 template wiring everything together.

---

### Stage 5.5 — Install & Run Scripts ✅

One-command setup and launch on Linux / macOS / WSL2.

**Delivered:**
- `install.sh` — Creates `.venv`, installs Python requirements, installs Ollama, starts Ollama service, pulls configured LLM model.
- `run_nmm.sh` — Starts Ollama if needed, launches `uvicorn`, opens browser (`xdg-open` / `open` / `wslview`), handles port conflict.
- `README.md` — Full project documentation.

---

## Current Capabilities

The game as shipped today can:

- Play Nine Men's Morris at 10 difficulty levels in a browser.
- Let the player choose colour or pick randomly; play Human vs AI, Human vs Human.
- Provide LLM commentary from a locally running Ollama model (no cloud, no cost after setup).
- Chat with the player during the game; store conversations in game records.
- Show a live eval-history graph updated after every move.
- Recognise named openings (D4 symmetry-aware) and steer toward statistically good ones via UCB1.
- Detect endgame phases, zugzwang, and mill-cycle patterns; announce transitions.
- Undo moves.
- Record every game in structured JSONL; MillsAI reads the last 10 before each new game.
- Score-normalise position evaluation using per-phase tanh scaling so the graph is meaningful across all phases.

---

## Planned Stages

---

### Stage 6 — Self-Play Training Loop 🔄

**Goal:** Populate the opening book with real win-rate data and enrich LLM game history without requiring a human player.

**Script:** `tools/self_play.py`

**Modes:**
| Flag | Description |
|------|-------------|
| (default) | Full LLM mode — coordinator deliberates for White, comments on Black's moves |
| `--no-llm` | Fast mode — no LLM calls, two raw GameAI instances; recommended for bulk runs |
| `--swap` | Alternate which engine plays White each game to reduce first-mover bias |
| `--blunder P` | White makes a random blunder with probability P (generates varied game data) |
| `--summary` | Ask LLM for a batch summary after all games complete |

**What improves from self-play:**
1. **Opening book win rates** — UCB1 scores updated after every game; future game starts favour statistically stronger openings.
2. **Novel opening discovery** — Sequences that don't match the book are saved as "learned" openings and named by the LLM (or auto-named in fast mode).
3. **LLM game context** — All games land in `data/games/`; MillsAI reads them before web games, giving richer positional commentary.
4. **Pattern analysis** — `MemoryManager.analyse_patterns()` distils placement and weakness patterns into the coordinator's narrative-memory prompt.

**What does NOT improve from self-play (yet):**
- Minimax heuristic weights (fixed — see Stage 7).
- Search depth strategy (fixed by difficulty level).

**Recommended usage:**
```bash
# Quick opening-book warm-up (fast, ~5 min):
python tools/self_play.py --no-llm --games 100 --white 6 --black 6 --swap

# Overnight deep training run:
python tools/self_play.py --no-llm --games 500 --white 8 --black 6 --swap --blunder 0.05

# LLM-enriched run (slow, generates commentary + novel opening names):
python tools/self_play.py --games 20 --white 7 --black 5 --summary
```

---

### Stage 7 — Heuristic Parameter Evolution ⬜

**Goal:** Automatically tune the weights in `ai/heuristics.py` to maximise win rate in self-play.

**Approach:** Simple (1+1) evolution / hill-climbing:
1. Start from the current weight vector as the baseline.
2. Perturb weights by Gaussian noise.
3. Play N self-play games (candidate vs baseline).
4. If candidate win rate > 50% + margin, promote to new baseline.
5. Repeat until convergence or time budget exhausted.

**Deliverables:**
- `tools/evolve_weights.py` — Evolution driver; saves weight checkpoints to `data/weights/`.
- Serialisable `HeuristicWeights` dataclass (extracted from `heuristics.py`).
- `ai/heuristics.py` refactored to accept a weights parameter.
- Best weights auto-loaded at game start if `data/weights/best.json` exists.

**Risk:** Overfitting to self-play (the engine plays itself, not humans). Mitigate by evaluating against fixed reference engines (difficulty 4 and difficulty 8).

---

### Stage 8 — Adaptive Difficulty ⬜

**Goal:** Keep human games competitive by auto-adjusting difficulty to match the player's skill.

**Approach:**
- Track win/loss/draw history for the current player (stored in `data/player_profile.json`).
- Estimate player Elo from recent outcomes (K=32, initial=1000; AI difficulties map to approximate Elo).
- After each game, nudge difficulty ±1 toward the target win-rate band (40–60%).
- Expose current estimated Elo in the UI info panel.

**Deliverables:**
- `ai/player_profile.py` — Profile manager with Elo estimation.
- `web/app.py` — Reads profile on `new_game`, writes outcome on `game_over`.
- `web/templates/index.html` — Estimated Elo and adaptive mode toggle in settings.

---

### Stage 9 — Tournament / Match Mode ⬜

**Goal:** Let users run head-to-head matches between named difficulty configs and view results.

**Approach:**
- New `/tournament` endpoint in `web/app.py`.
- Match config: White difficulty, Black difficulty, number of games, colour swap.
- Runs via `self_play.py` as a subprocess (non-blocking, streamed progress).
- Results stored in `data/tournaments/` and displayed in a new browser tab.

**Deliverables:**
- `web/templates/tournament.html` — Match config form + live results table.
- `web/app.py` — `/tournament/start`, `/tournament/stream` (SSE), `/tournament/results/{id}`.
- Results include per-opening breakdown, average game length, eval trajectory summary.

---

### Stage 10 — Advanced Search (MCTS / Neural Evaluation) ⬜  *(Stretch)*

**Goal:** Replace or augment negamax with Monte Carlo Tree Search, optionally with a learned value function.

**Approach:**
- Implement `ai/mcts.py` — UCT-based MCTS with the existing `heuristics.evaluate()` as rollout heuristic.
- Self-play generates (state, outcome) pairs for supervised training of a small MLP value network.
- Value network replaces rollout at MCTS leaves (AlphaZero-lite).
- MCTS and negamax can be toggled per-difficulty slot.

**Note:** This stage requires significant compute (GPU recommended for training). Designed to run offline on the self-play records accumulated in Stages 6–7.

---

## Architecture Principles

- **Immutable board state** — `BoardState.apply_move()` always returns a new object. The engine holds the mutable current state. This enables safe undo, MCTS branching, and self-play without deep-copy overhead.
- **Coordinator owns the narrative** — All commentary and LLM calls flow through `Coordinator`. `GameAI` is pure search; `MillsLLM` is pure text generation. Neither knows about the other.
- **No cloud dependency** — All LLM inference runs locally via Ollama. No API keys, no cost after initial model pull.
- **Progressive enhancement** — Every stage adds capability without breaking the previous one. Fast mode (no LLM, no opening book) always works as a fallback.
