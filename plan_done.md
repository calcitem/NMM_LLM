# Nine Men's Morris — Completed Work

_This file contains finished stages and resolved bugs. New completed items are appended here; active work lives in `plan_todo.md`._

---

# Nine Men's Morris — Project Plan

## Stage Summary

| Stage | Title | Status |
| - | - | - |
| 1 | Core Game Engine | ✅ Complete |
| 2 | Classical AI (Minimax) | ✅ Complete |
| 3 | Memory & LLM Layer | ✅ Complete |
| 4 | Opening Book | ✅ Complete |
| 5 | Web GUI | ✅ Complete |
| 5.5 | Install & Run Scripts | ✅ Complete |
| 5.6 | In-Game Hint System | ✅ Complete |
| 5.7 | Force Move + Thinking Timer | ✅ Complete |
| 5.8 | Enhanced LLM Commentary | ✅ Complete |
| 5.9 | Move Replay Viewer | ✅ Complete |
| 5.10 | Position Setup / Editor | ✅ Complete |
| 5.11 | Bug Fixes & Hardening | ✅ Complete |
| 5.12 | AI Tactical Imperatives | ✅ Complete |
| 5.13 | AI Settings & Weight Tuning UI | ✅ Complete |
| 5.14 | Opening Replay on GUI | ✅ Complete |
| 5.15 | User Guide & README Expansion | ✅ Complete |
| 5.16 | Starting Play Variants & Opening Database | ✅ Complete |
| 5.17 | Game Trajectory Memory & Winner-Aware Learning | ✅ Complete |
| 5.18 | Per-Personality Saved Settings | ✅ Complete |
| 5.23 | UI Layout & Default Preference Fixes | ✅ Complete |
| 5.24 | Endgame Position Memory | ✅ Complete |
| 5.25 | AI Personality Selector in Header | ✅ Complete |
| 5.26 | Setup Position Button in Header | ✅ Complete |
| 5.27 | Game Counter & Automatic Library Consolidation | ✅ Complete |
| 6 | Self-Play Training Loop | ✅ Complete |
| 7 | Heuristic Parameter Evolution | ✅ Complete |
| 8 | Adaptive Difficulty | ✅ Complete |
| 9 | Tournament / Match Mode | ✅ Complete |
| 10 | Player Profiles & Persistent Stats | ✅ Complete |
| 11 | Endgame Self-Play | ✅ Complete |
| 12 | Advanced Search (MCTS / NN) | ✅ Complete |

---

## Completed Stages

### Stage 1 — Core Game Engine ✅

Full Nine Men's Morris rules in pure Python with no external dependencies.

**Delivered:**

- `game/board.py` — Immutable `BoardState` dataclass; `apply\\\\\\\_move()` returns a new state (enables safe undo and MCTS branching).

- `game/rules.py` — `get\\\\\\\_all\\\\\\\_legal\\\\\\\_moves()`, `is\\\\\\\_terminal()`, `does\\\\\\\_form\\\\\\\_mill()`, phase detection (placement / movement / flying).

- `game/game\\\\\\\_engine.py` — Mutable `GameEngine` wrapper; records every move into `game\\\\\\\_record` with FEN, notation, turn metadata.

- `game/notation.py` — Algebraic notation helpers.

- 57 tests pass.

### Stage 2 — Classical AI (Minimax) ✅

Negamax with alpha-beta pruning, iterative deepening, and blunder injection.

**Delivered:**

- `ai/game\\\\\\\_ai.py` — `GameAI` with `choose\\\\\\\_move()`, `score\\\\\\\_move()`, `position\\\\\\\_eval()`.

  - Difficulties 1–8 map to fixed depths (2–9 ply).

  - Difficulties 9–10 use iterative deepening with 20 s and 45 s time budgets.

  - `blunder\\\\\\\_probability` selects a worst-quartile move intentionally (teaching aid).

- `ai/heuristics.py` — Phase-aware static evaluation:

  - Mill count, blocked pieces, piece count, two-configurations.

  - Double-mill pivots, win configuration (opponent in fly phase).

  - Mobility difference, immediate mill threats, positional value (cross/cardinal nodes score 3; corner nodes score 2).

  - tanh normalisation with per-phase scale (`place=120`, `move=180`, `fly=280`).

- `ai/endgame\\\\\\\_recognizer.py` — Detects midgame → endgame → deep-endgame transitions, mill-cycle patterns, and zugzwang risk; boosts search depth in critical positions.

- 74 tests pass.

### Stage 3 — Memory & LLM Layer ✅

Local Ollama LLM (MillsAI) that comments on moves, chats with the player, and accumulates game history.

**Delivered:**

- `ai/memory\\\\\\\_manager.py` — ChromaDB vector store for bad-move memory and strategy snippets; JSONL game log in `data/games/`; session narratives in `data/session\\\\\\\_memory/`.

- `ai/mills\\\\\\\_llm.py` — Ollama interface:

  - `ask\\\\\\\_for\\\\\\\_move\\\\\\\_opinion()` — strict `MOVE: / REASON:` format enforcement; auto-retry on parse failure.

  - `evaluate\\\\\\\_human\\\\\\\_move()` — comments when score drops \> threshold (capped per game).

  - `player\\\\\\\_chat()` — multi-turn in-game conversation with context history.

  - `summarise\\\\\\\_session()`, `name\\\\\\\_novel\\\\\\\_opening()`, `debrief\\\\\\\_game()`.

  - Reads last 10 games before each new game.

- `ai/coordinator.py` — Orchestrates GameAI + MillsLLM:

  - `deliberate()` — GameAI picks best move, LLM recommends, coordinator adopts LLM move if it beats engine score + bonus threshold.

  - `react\\\\\\\_to\\\\\\\_human\\\\\\\_move()` — scores move, emits poor-move comment if warranted.

  - `on\\\\\\\_game\\\\\\\_start()` / `on\\\\\\\_game\\\\\\\_end()` — lifecycle with opening book integration.

- 34 tests pass (includes stages 1–2).

### Stage 4 — Opening Book ✅

Curated opening library with UCB1-scored selection and D4 symmetry recognition.

**Delivered:**

- `ai/opening\\\\\\\_book.py` — `OpeningBook` (read from `data/openings/book\\\\\\\_openings.json`; writes to `data/openings/openings.json`):

  - UCB1 selection: `score + C \\\\\\\* sqrt(log(N) / (n\\\\\\\_i + 1))`, exploration rate 0.25.

  - Per-opening win/loss/draw stats split by human/AI side.

  - Novel opening auto-save with LLM-generated name.

- `ai/opening\\\\\\\_recognizer.py` — Real-time recognition during placement phase:

  - Full D4 dihedral group (4 rotations × 4 reflections, 8 symmetries).

  - Provides `book\\\\\\\_move`, `strategic\\\\\\\_notes`, `common\\\\\\\_blunders` context to coordinator.

  - Detects deviations; records branches.

- `tools/import\\\\\\\_openings.py` — Imported curated openings from `strategy\\\\\\\_book.txt`.

- `tools/teach\\\\\\\_opening.py`, `tools/list\\\\\\\_openings.py` — Maintenance tools.

### Stage 5 — Web GUI ✅

Full browser-based interface over FastAPI + WebSockets, with no page reloads.

**Delivered:**

- `web/app.py` — FastAPI server with `/ws` WebSocket endpoint:

  - Messages: `new\\\\\\\_game`, `move`, `capture`, `undo`, `player\\\\\\\_message` (client→server).

  - Messages: `state`, `capture\\\\\\\_required`, `thinking`, `ai\\\\\\\_move`, `commentary`, `game\\\\\\\_over`, `error` (server→client).

  - Board state history for undo; `projected\\\\\\\_board` preview during mill-capture sequence.

- `web/static/board.js` — Pure-JS SVG board:

  - Three concentric squares + 4 cross connections; coordinate labels (a–g, 1–7).

  - Layer order: bg → lines → labels → nodes → **pieces → hints** (hints above pieces so capture rings intercept clicks).

  - Colour-coded hints: green = legal placements, yellow = selectable, red = capturable.

  - Mill flash on capture.

- `web/static/game.js` — Game logic and UI:

  - Real-time eval history SVG graph (no external library).

  - Player chat with MillsAI; sent as `player\\\\\\\_message` WebSocket messages.

  - Undo button; disabled when no history.

  - Info panel: pieces placed and pieces taken (not on-board count).

  - Settings: colour (White / Black / Random), opponent (AI / Human), difficulty 1–10, LLM toggle.

- `web/static/style.css` — Dark wooden theme (`--bg: \\\\\\\#1a1510`), responsive two-column layout.

- `web/templates/index.html` — Jinja2 template wiring everything together.

### Stage 5.5 — Install & Run Scripts ✅

One-command setup and launch on Linux / macOS / WSL2.

**Delivered:**

- `install.sh` — Creates `.venv`, installs Python requirements, installs Ollama, starts Ollama service, pulls configured LLM model.

- `run\\\\\\\_nmm.sh` — Starts Ollama if needed, launches `uvicorn`, opens browser (`xdg-open` / `open` / `wslview`), handles port conflict.

- `README.md` — Full project documentation.

### Stage 5.6 — In-Game Hint System ✅

**Goal:** Let the human player request a hint at any point during their turn, getting both a visual board highlight and a plain-English explanation from MillsAI.

**Delivered:**

- `web/app.py` — `hint\\\\\\\_request` handler; `Session.hints\\\\\\\_used` counter (cap 3 per game).

- `web/static/game.js` — Hint button wiring, `hint` message handler.

- `web/static/board.js` — `showHint(from, to)` method with 4 s timed fade.

- `web/static/style.css` — `\\\\\\\#btn-hint` styles.

### Stage 5.7 — Force Move + Thinking Time Indicator ✅

**Goal:** Let the player interrupt a slow AI search and see how long the AI has been thinking.

**Delivered:**

- `ai/game\\\\\\\_ai.py` — `force\\\\\\\_stop()` sets `self.\\\\\\\_deadline = 0`; `\\\\\\\_score\\\\\\\_all()` catches `\\\\\\\_SearchAbort` and returns partial results so the best move found so far is still returned; `choose\\\\\\\_move()` resets deadline at start.

- `web/app.py` — AI turn runs as a background `asyncio.Task` so `force\\\\\\\_move` WebSocket messages can be received concurrently; `\\\\\\\_expected\\\\\\\_think\\\\\\\_seconds()` computes a rough budget by difficulty; `thinking` message now includes `expected\\\\\\\_seconds`.

- `web/static/game.js` — `startThinkingTimer()` / `stopThinkingTimer()` update the status bar with elapsed time every 200 ms; Force Move button appears while AI thinks (animated border) and disappears when `state` arrives.

- `web/templates/index.html` — `\\\\\\\<button id="btn-force-move"\\\\\\\>` in the bottom bar, hidden by default.

- `web/static/style.css` — Force Move button with pulsing gold border animation.

### Stage 5.8 — Enhanced LLM Commentary ✅

**Goal:** MillsAI comments on more than just blunders — it now reacts to mills, strong moves, and asks periodic strategic questions.

**Delivered:**

- `ai/mills\\\\\\\_llm.py` — Three new prompt templates: `\\\\\\\_POSITIVE\\\\\\\_COMMENT\\\\\\\_SYSTEM`, `\\\\\\\_MILL\\\\\\\_COMMENT\\\\\\\_SYSTEM`, `\\\\\\\_POSITION\\\\\\\_QUESTION\\\\\\\_SYSTEM`; three new methods: `comment\\\\\\\_on\\\\\\\_good\\\\\\\_move()`, `comment\\\\\\\_on\\\\\\\_mill()`, `ask\\\\\\\_strategic\\\\\\\_question()`.

- `ai/coordinator.py` — `react\\\\\\\_to\\\\\\\_human\\\\\\\_move()` now has four commentary paths in priority order:

  1. Mill/capture comment (always, when gap ≥ 2 turns).

  2. Poor-move warning (capped at `max\\\\\\\_poor\\\\\\\_move\\\\\\\_comments`).

  3. Positive comment on strong moves (score ≥ 0.75).

  4. Periodic strategic question every 8 human turns.

- `\\\\\\\_human\\\\\\\_turn\\\\\\\_num` counter and `\\\\\\\_can\\\\\\\_comment\\\\\\\_general()` helper added to `Coordinator`.

## Current Capabilities

The game as shipped today can:

- Play Nine Men's Morris at 10 difficulty levels in a browser.

- Let the player choose colour or pick randomly; play Human vs AI, Human vs Human.

- Provide LLM commentary from a locally running Ollama model (no cloud, no cost after setup).

- Chat with the player during the game; store conversations in game records.

- Show a live eval-history graph updated after every move.

- Recognise named openings (D4 symmetry-aware) and steer toward statistically good ones via UCB1, with room to extend toward richer starting-play variant tracking.

- Detect endgame phases, zugzwang, and mill-cycle patterns; announce transitions.

- Undo moves; offer / accept draw; Force Capture toggle.

- Record every game in structured JSONL; MillsAI reads the last 10 before each new game.

- Score-normalise position evaluation using per-phase tanh scaling.

- Interrupt the AI search mid-think via Force Move and see elapsed thinking time.

- LLM comments on poor moves, good moves, mills, and asks periodic strategic questions.

## Active / Near-Term Stages

### Stage 5.11 — Bug Fixes & Hardening ✅

**Goal:** Resolve a set of confirmed bugs affecting game stability, self-test reliability, and human-vs-AI playability.

#### Bug 5.11-A — Self-Test Cannot Locate Ollama

**Symptom:** Running the self-test reports that Ollama is unavailable even when `ollama serve` is running.

**Root cause (suspected):** The self-test probes a hardcoded host/port or constructs the Ollama URL independently of the runtime config, so it misses the actual running process.

**Fix:**

- `tests/self\\\\\\\_test.py` — Replace any inline URL construction with a call to the shared `OllamaClient` factory used by `mills\\\\\\\_llm.py`. Both the test and the game must hit the same endpoint.

- Add a connectivity pre-check: `GET http://\\\\\\\<host\\\\\\\>:\\\\\\\<port\\\\\\\>/api/tags`. If the request succeeds, mark Ollama as reachable; if it fails, emit a clear diagnostic (`"Ollama unreachable at \\\\\\\<url\\\\\\\> — is 'ollama serve' running?"`), then skip LLM tests gracefully rather than hard-failing the entire suite.

- Expose `OLLAMA\\\\\\\_HOST` and `OLLAMA\\\\\\\_PORT` as environment variables (defaulting to `localhost` / `11434`) so the test and game can both be overridden without code changes.

#### Bug 5.11-B — Games Do Not Finish at High Difficulty

**Symptom:** At difficulty settings 8–10, games run indefinitely and never reach a terminal state.

**Root cause (suspected):** The endgame recogniser's transition threshold and/or `is\\\\\\\_terminal()` may not fire correctly when the piece count drops slowly; iterative deepening may also loop under its time budget without committing the final best move on timeout.

**Fix:**

- `ai/endgame\\\\\\\_recognizer.py` — Lower the endgame detection trigger: **endgame fires when either player has ≤5 pieces OR total pieces ≤ 12**; deep-endgame remains at ≤8 total. This causes deeper search and more decisive evaluation earlier, and catches the 7v5 / 6v5 asymmetric positions the current threshold misses.

- `game/rules.py` — Audit `is\\\\\\\_terminal()`: ensure it correctly returns `True` when a side drops to 2 pieces or has no legal moves; add a unit test for each terminal condition.

- `ai/game\\\\\\\_ai.py` — Guarantee `choose\\\\\\\_move()` always returns the best move found so far, even if the iterative-deepening loop is still running when the deadline fires. The current `\\\\\\\_SearchAbort` path must reach the return statement on all code paths.

- `tools/self\\\\\\\_play.py` — Add a hard per-game move-count cap (e.g. 300 moves) with a `draw\\\\\\\_by\\\\\\\_repetition` result to prevent infinite self-play games as a safety net.

#### Bug 5.11-C — AI Exceeds Thinking-Time Budget in Human vs AI

**Symptom:** During a Human vs AI game, the AI regularly exceeds its allotted thinking time by minutes — or appears to think indefinitely — forcing the human to press Force Move every turn.

**Root cause:** The iterative-deepening loop in `choose\\\\\\\_move()` checks `time.time() \\\\\\\> self.\\\\\\\_deadline` only at the top of each depth iteration, not inside the inner alpha-beta search. A single deep call tree can therefore run far past the deadline before the check fires.

**Fix:**

- `ai/game\\\\\\\_ai.py` — Add a time-check call inside `\\\\\\\_negamax()` every N nodes (e.g. every 2 048 leaf evaluations). If `time.time() \\\\\\\> self.\\\\\\\_deadline`, raise `\\\\\\\_SearchAbort` immediately rather than waiting for the outer loop to notice.

- `web/app.py` — The `asyncio.Task` running the AI must be cancelled if it has not completed within `expected\\\\\\\_seconds + grace\\\\\\\_period` (suggest grace = 5 s). On cancellation, call `game\\\\\\\_ai.force\\\\\\\_stop()` and then collect the best partial result via the existing `\\\\\\\_SearchAbort` path. This ensures the AI move is **always delivered automatically** when the timer expires, with no player intervention required.

- `web/static/game.js` — Remove any UX that implies the player must press Force Move to proceed. Force Move remains available as an early-interrupt option, but the timer expiry must trigger the move automatically from the server.

- Update `startThinkingTimer()` to show a countdown (time remaining) rather than elapsed time, so the player can see when the forced move will fire.

#### Bug 5.11-D — AI Resignation at Dominant Human Position

**Goal:** If the human's normalised position strength stays above **0.95 for 3 consecutive AI turns**, the AI offers to concede the game.

**Implementation:**

- `ai/coordinator.py` — Add `\\\\\\\_dominant\\\\\\\_turn\\\\\\\_streak: int = 0`. After each AI move, evaluate the position from the human's perspective. If the human's tanh-normalised score exceeds 0.95, increment the counter; otherwise reset to 0. When the counter reaches 3, call `offer\\\\\\\_defeat()`.

- `web/app.py` — Handle the `offer\\\\\\\_defeat` signal: send a `game\\\\\\\_over` WebSocket message with `result: "ai\\\\\\\_resignation"` and a MillsAI farewell comment.

- `web/static/game.js` — Display the resignation as a distinct outcome in the result overlay (different copy and colour from checkmate / draw).

**Deliverables:**

- `ai/game\\\\\\\_ai.py` — Deadline-aware node counter in `\\\\\\\_negamax()`.

- `ai/game\\\\\\\_ai.py` — Guaranteed partial-result return on `\\\\\\\_SearchAbort`.

- `ai/endgame\\\\\\\_recognizer.py` — Lowered detection threshold (12 pieces).

- `game/rules.py` — Audited `is\\\\\\\_terminal()`; new terminal-condition tests.

- `ai/coordinator.py` — `\\\\\\\_dominant\\\\\\\_turn\\\\\\\_streak` + `offer\\\\\\\_defeat()`.

- `web/app.py` — Auto-force-move on deadline; `ai\\\\\\\_resignation` handler.

- `web/static/game.js` — Countdown timer; auto-move on expiry; resignation UI.

- `tests/self\\\\\\\_test.py` — Shared Ollama URL; graceful LLM-skip; terminal-condition tests.

- `tools/self\\\\\\\_play.py` — Hard move-count cap.

### Stage 5.12 — AI Tactical Imperatives ✅

**Goal:** Make the AI decisively exploit tactical patterns — mills, double mills, feeder structures, and diamond formations — rather than defaulting to passive positional play.

#### Background

The current heuristic scores individual features (mill count, blocked pieces, mobility) but does not encode the **urgency hierarchy** a strong human player follows. The AI sometimes allows opponents to close mills, misses double-mill cycles, and over-values remote positional gains over immediate tactical threats.

#### Tactical Priority Hierarchy

The AI must evaluate moves in this priority order before falling back to positional scoring:

1. **Close an open mill now** — if the AI can form a mill this move, it must do so unless a higher-priority threat exists; eg it can move into an open mill so that the opponent cannot close it.

2. **Close a double mill / cycling mill** — two mills sharing a pivot piece that can be cycled each turn (remove → replace → remove) score significantly higher than a single mill.

3. **Block an opponent mill about to close** — if the opponent can form a mill next move, blocking or disabling scores higher than any non-urgent alternative. However, making another mill available so the opponent must disable one is also a valid response.

4. **Remove opponent pieces enabling future mills** — when making a capture, prefer opponent pieces whose removal:

   - Opens a line for an AI mill in 1–2 moves.

   - Disrupts an opponent feeder mill (a mill with a neighbouring piece that can slide back to re-form the mill immediately after capture).

   - Removes a pivot of an opponent double mill or diamond.

5. **Long-term positioning** — all positional gains are secondary to the above.

#### Specific Patterns to Recognise

**Feeder mill:** A mill where the captured piece has a neighbour that can step back into the mill position next turn, recreating the mill immediately. Removing such an opponent piece gives at least two free captures in a row.

**Diamond:** Four pieces on the four corners of a diamond (e.g. `a4`, `b3`, `c4`, `b5` or classical `a4, b2, c4, b6`) where any adjacent pair can form a mill. Dismantling opponent diamonds by targeted capture is a high-value tactic.

**Cycling double mill:** Two mills sharing a pivot piece. Each turn the pivot slides off one mill line (removing an opponent piece), then slides back, reactivating the other mill. The AI must recognise and build these structures aggressively; recognising and disrupting opponent structures is equally important.

**Mill wrapping / parallel mill pressure:** Recognise attempts to build a second mill parallel or adjacent to an already-formed mill so that the owner of the first mill becomes positionally constrained and cannot freely break and remake it. In practical terms, this includes patterns like the illustrated parallel build where one side threatens to create a neighbouring line while the existing mill holder is effectively immobilised by ownership of the current mill. The AI should score these patterns in both directions: build them when advantageous, and break them early when the opponent is trying to establish them.

#### Implementation

- `ai/heuristics.py` — Add a `tactical\\\\\\\_urgency\\\\\\\_bonus` evaluation layer:

  - `WEIGHT\\\\\\\_CLOSE\\\\\\\_MILL` (default 500) — bonus for a move that closes a mill.

  - `WEIGHT\\\\\\\_CLOSE\\\\\\\_DOUBLE\\\\\\\_MILL` (default 800) — additional bonus for a cycling double-mill closure.

  - `WEIGHT\\\\\\\_BLOCK\\\\\\\_OPPONENT\\\\\\\_MILL` (default 400) — bonus for a move that prevents the opponent forming a mill next turn.

  - `WEIGHT\\\\\\\_CAPTURE\\\\\\\_DISRUPT\\\\\\\_FEEDER` (default 300) — bonus when the captured piece was a feeder-mill participant.

  - `WEIGHT\\\\\\\_CAPTURE\\\\\\\_DISRUPT\\\\\\\_DIAMOND` (default 250) — bonus when the captured piece was a diamond corner.

  - `WEIGHT\\\\\\\_LONG\\\\\\\_TERM\\\\\\\_POSITION` (default 60) — multiplier on existing positional score; kept intentionally low relative to the above.

  - Add `WEIGHT\\\\\\\_STOP\\\\\\\_OPPONENT\\\\\\\_MILL` (default 450) — penalty applied to any move that leaves an opponent mill threat open when a blocking move was available.

  - All weights exposed to the Settings page (Stage 5.13).

- `ai/heuristics.py` — Add helper functions:

  - `detect\\\\\\\_feeder\\\\\\\_mills(board, colour)` — returns list of mill positions where the removed piece has a re-entry neighbour.

  - `detect\\\\\\\_diamonds(board, colour)` — returns list of diamond corner sets.

  - `detect\\\\\\\_double\\\\\\\_mills(board, colour)` — returns list of pivot positions shared by two mills.

  - `opponent\\\\\\\_mills\\\\\\\_in\\\\\\\_n\\\\\\\_moves(board, colour, n)` — returns moves within which opponent can form a mill (n ≤ 2).

- `ai/coordinator.py` — Before calling `GameAI.choose\\\\\\\_move()`, run a tactical pre-screen:

  - If any legal move closes an open mill → bias the move scorer toward that move (inject urgency weight).

  - If no urgency is detected (no immediate mill threats either side) → allow the AI to favour long-term positional play.

  - "No urgency" is defined as: no AI mill closable this turn, no opponent mill closable next turn, no opponent double-mill in progress.

**Deliverables:**

- `ai/heuristics.py` — Tactical weights, helper functions, urgency layer.

- `ai/coordinator.py` — Tactical pre-screen before move selection.

- `tests/test\\\\\\\_tactics.py` — At least 10 unit tests covering: mill closure priority, double-mill detection, feeder-mill capture preference, diamond dismantling, and correct pass-through when no urgency.

### Stage 5.13 — AI Settings & Weight Tuning UI ⬜

**Goal:** Expose all AI heuristic weights and behaviour settings on an in-game Settings page with sliders, a default-reset button, and persistent storage between sessions. Make several AI settings; one for a more aggressive ‘personality’, one for a defensive / blocking player, one who sticks to opening plays, one who moves all over the board up to their 6th placement unless they have to block a cardinal mill. One who plays only book opening plays where posssible. One who follows no rules; add random moves in. Each will try develop their other sweights but will not be able to change the dominant ones. These personalities can play in tournaments with the human and play each other in the self play games.

**User flow:**

1. Open the **Settings** tab (already present in the sidebar).

2. A new **AI Tuning** section appears below the existing difficulty / colour / opponent selectors.

3. Each weight has a labelled slider (min / max / default clearly shown), a live numeric readout, and a tooltip explaining what the weight controls.

4. **Reset to Defaults** button restores all sliders to their designed values.

5. **Save Settings** button persists the current values; they are automatically applied to all subsequent games in the session.

6. Values are sent to the server on `new\\\\\\\_game` as part of the game config payload.

**Weights exposed on the Settings page:**

| Parameter | Default | Range | Description |
| - | - | - | - |
| Mill closure urgency | 500 | 100–1000 | How strongly the AI prioritises closing its own mills |
| Double mill urgency | 800 | 200–1500 | Extra bias toward cycling double-mill formations |
| Block opponent mill | 450 | 100–900 | How aggressively the AI blocks opponent mills |
| Feeder-mill capture | 300 | 50–600 | Preference for capturing pieces that feed opponent mills |
| Diamond dismantling | 250 | 50–500 | Preference for breaking opponent diamond structures |
| Stop opponent mills | 450 | 100–900 | Penalty for ignoring an available opponent-mill block |
| Long-term position | 60 | 10–200 | Multiplier on all positional (non-tactical) scoring |
| Mill count weight | (current) | 0–300 | Value of each mill in static eval |
| Mobility weight | (current) | 0–400 | Value of each additional legal move |
| Blocked pieces weight | (current) | 0–500 | Penalty for each blocked own piece; points for blocked opponent pieces |
| Moving all over the board up to 6th piece; eg d6, e5, f4, g1) – all places that might be a dual potential mill | 100 | 0-500 |  |
| Blocking cardinal mills | 400 | 0-500 |  |
| Random moves | 0 | 0-500 |  |


**Implementation:**

- `web/static/game.js` — Build `SettingsPanel` class; render sliders from a `WEIGHT\\\\\\\_DEFAULTS` map; `saveSettings()` stores current slider values in a JS object (not `localStorage`); `resetSettings()` restores from `WEIGHT\\\\\\\_DEFAULTS`.

- `web/static/style.css` — Slider styling to match the dark wooden theme; tooltip `\\\\\\\[data-tooltip\\\\\\\]` attribute CSS.

- `web/app.py` — Accept `ai\\\\\\\_weights` dict in the `new\\\\\\\_game` message; pass to `GameAI` and `Coordinator` constructors.

- `ai/heuristics.py` — `HeuristicWeights` dataclass accepted by `evaluate()`; defaults match the table above.

- `ai/game\\\\\\\_ai.py` — Pass `HeuristicWeights` through to heuristic calls.

**Deliverables:**

- `web/static/game.js` — `SettingsPanel` with sliders, reset, and save.

- `web/static/style.css` — Slider and tooltip styles.

- `web/app.py` — `ai\\\\\\\_weights` handling in `new\\\\\\\_game`.

- `ai/heuristics.py` — `HeuristicWeights` dataclass.

- `ai/game\\\\\\\_ai.py` — Weight injection.

### Stage 5.14 — Opening Replay on GUI ⬜ - human note – this is done

**Goal:** Let the player watch a named opening played out move-by-move on the live board, driven by either two AI instances or a forced move sequence. Useful for learning opening theory.

**User flow:**

1. Open the **Openings** panel (new tab or section in the sidebar).

2. A dropdown lists all openings in `data/openings/book\\\\\\\_openings.json`, with win-rate stats.

3. Click **Replay Opening** — the board resets and the opening moves are played out automatically at a configurable speed (0.5 s – 3 s per move).

4. After the last recorded opening move, the game either:

   - Continues as a normal AI vs AI game (Auto-continue mode), or

   - Pauses for the human to take over (Practice mode).

5. The current move number and opening name are shown in the status bar during replay.

**Implementation:**

- `web/app.py` — New `replay\\\\\\\_opening` WebSocket message: `\\\\\\\{ type: "replay\\\\\\\_opening", opening\\\\\\\_id: str, speed\\\\\\\_ms: int, continue\\\\\\\_mode: "auto" | "practice" \\\\\\\}`. Server streams moves from the opening sequence as `ai\\\\\\\_move` messages with a configurable delay.

- `web/static/game.js` — `OpeningsPanel` class: fetches opening list on load via `/api/openings`; renders dropdown + replay controls; sends `replay\\\\\\\_opening` message; disables board interaction during replay.

- `web/app.py` — New `/api/openings` GET endpoint returning the opening list with names and stats.

- `web/static/style.css` — Opening panel and replay progress styles.

- `web/templates/index.html` — Openings tab in the sidebar.

**Deliverables:**

- `web/app.py` — `replay\\\\\\\_opening` handler; `/api/openings` endpoint.

- `web/static/game.js` — `OpeningsPanel`, replay state machine.

- `web/templates/index.html` — Openings tab.

- `web/static/style.css` — Replay UI styles.

### Stage 5.15 — User Guide & README Expansion 🔄

**Goal:** Provide complete end-user and developer documentation covering installation, all tools, CLI flags, game settings, and the self-test system.

**README sections to add / rewrite:**

#### Installation

```
git clone \\\\\\\<repo\\\\\\\>      
cd nine-mens-morris      
bash install.sh
```

`install.sh` creates a Python virtual environment, installs all dependencies, installs Ollama (if not present), starts the Ollama service, and pulls the configured LLM model. Run once; subsequent launches use `run\\\\\\\_nmm.sh`.

#### Running the Game

```
bash run\\\\\\\_nmm.sh
```

Opens the game in your browser at `http://localhost:8000`. The script:

- Checks whether Ollama is running and starts it if needed.

- Launches the FastAPI server via `uvicorn`.

- Opens the browser automatically (`xdg-open` / `open` / `wslview`).

- Handles port conflicts gracefully.

To run without the LLM (faster startup, no Ollama required):

```
bash run\\\\\\\_nmm.sh --no-llm
```

#### CLI Flags

| Flag | Effect |
| - | - |
| `--no-llm` | Disables all Ollama calls. The AI plays using minimax only; commentary and chat are unavailable. Faster and works without Ollama. |
| `--port N` | Run the server on port N (default 8000). |
| `--host H` | Bind to host H (default 127.0.0.1). |
| `--debug` | Enable FastAPI debug mode with auto-reload. |


#### Game Settings (In-Browser)

| Setting | Description |
| - | - |
| Colour | Play as White, Black, or Random. |
| Opponent | Human vs AI, Human vs Human. |
| Difficulty | 1–10 (1 = weakest, 10 = strongest). Levels 1–8 are fixed-depth; 9–10 use iterative deepening with time budgets. |
| LLM Toggle | Enable / disable MillsAI commentary and chat mid-game (requires Ollama). |
| Show Moves | Toggle move indicators on the board (default: ON). |
| AI Tuning | Sliders for all AI heuristic weights (see Stage 5.13). |


#### In-Game Controls

| Control | Description |
| - | - |
| Force Move | Immediately ends the AI's thinking and plays the best move found so far. The AI timer also fires this automatically at deadline. |
| Hint | Highlights the AI-recommended move for the human. Capped at 3 hints per game. |
| Undo | Steps back one half-move. |
| Chat | Send a message to MillsAI for live commentary and advice. |


#### Tools

All tools live in the `tools/` directory and run in the virtual environment:

```
\\\\\\\# Activate the virtual environment first:      
source .venv/bin/activate
```

| Tool | Command | Description |
| - | - | - |
| Self-test | `python tools/self\\\\\\\_test.py` | Runs the full test suite. Reports engine correctness, LLM connectivity (skipped gracefully if Ollama is offline), and opening-book integrity. |
| Self-play | `python tools/self\\\\\\\_play.py --no-llm --games 100 --white 6 --black 6 --swap` | Run AI vs AI games to warm up the opening book. See Stage 6 for all flags. |
| List openings | `python tools/list\\\\\\\_openings.py` | Print all openings in the book with win/loss/draw stats. |
| Teach opening | `python tools/teach\\\\\\\_opening.py` | Interactively add a named opening sequence to the book. |
| Import openings | `python tools/import\\\\\\\_openings.py` | Bulk-import openings from `strategy\\\\\\\_book.txt`. |


#### Self-Test Details

`python tools/self\\\\\\\_test.py` runs the following checks:

1. **Core engine tests** — all 57 board/rules/notation tests.

2. **AI tests** — all 74 minimax and heuristic tests.

3. **LLM connectivity** — probes `http://\\\\\\\<OLLAMA\\\\\\\_HOST\\\\\\\>:\\\\\\\<OLLAMA\\\\\\\_PORT\\\\\\\>/api/tags`. If reachable, runs a short move-opinion request; if unreachable, prints a diagnostic and skips LLM tests (no hard failure).

4. **Opening book integrity** — verifies JSON schema and that all recorded moves are legal.

5. **Self-play smoke test** — plays one 30-move `--no-llm` game and checks it reaches a valid terminal state.

6. **EndgameDB integration test** — after the smoke-test game, verifies `EndgameDB.query()` returns at least one hint for the final endgame position (≤11 pieces). Confirms the hint dict contains valid board notations with float deltas in the range \[−0.5, 0.5\]. Passes even if no historical games exist yet (empty result accepted). Run with: `python tools/self\_test.py --endgame`.

Set `OLLAMA\\\\\\\_HOST` / `OLLAMA\\\\\\\_PORT` environment variables to override the default `localhost:11434` if Ollama is running on a non-standard address.

**Deliverables:**

- `README.md` — All sections above, fully written.

- `docs/USER\\\\\\\_GUIDE.md` — Expanded standalone guide with screenshots placeholder, troubleshooting FAQ.

- `tools/self\\\\\\\_test.py` — Updated to use shared Ollama URL config (see Stage 5.11-A).

### Stage 5.18 — Per-Personality Saved Settings ⬜ -- Human note, this has been done

**Goal:** Every named personality (and "Custom") has its own persistent settings file. Editing weights and clicking Save stores them to that personality's file; loading a personality restores from its file.

**User flow:**

1. Select a personality from the AI Tuning panel.

2. Move any sliders; click **Save Settings** — values are written to `data/personalities/\<name\>.json`.

3. Next session: selecting the same personality auto-loads the saved file, not the coded defaults.

4. **Custom** personality works the same way — independent file from any named preset.

5. Reset button restores the *original* coded defaults (not the saved file).

**Implementation:**

- `web/app.py` — `GET /api/personalities/\<name\>` returns the saved personality JSON (or built-in defaults if no file exists). `POST /api/personalities/\<name\>` writes the JSON to `data/personalities/\<name\>.json`.

- `web/static/game.js` — On personality selection: `fetch('/api/personalities/\<name\>')` and apply the returned weights to all sliders. On Save Settings: `fetch POST /api/personalities/\<personality\>` with current slider values.

- `data/personalities/` directory (auto-created on first save).

**Deliverables:**

- `web/app.py` — GET/POST `/api/personalities/\<name\>` endpoints.

- `web/static/game.js` — Personality load/save wiring.

- `data/personalities/` directory support.

### Stage 5.23 — UI Layout & Default Preference Fixes ✅

**Goal:** Small UI improvements requested after playtest.

**Items:**

1. **Move list shown by default** — The moves panel should be visible on page load without the player having to press a button. The current "Moves" toggle button moves to the left sidebar near the Settings button.

2. **Start New Game to top-centre** — Relocate the "Start New Game" button to the top-centre of the page so it is easily accessible during replay and between games.

**Implementation:**

- `web/static/game.js` — Set moves panel initial state to `visible`; wire Moves toggle button in left sidebar.

- `web/templates/index.html` — Move Moves button to left sidebar; move Start New Game button to top-centre header.

- `web/static/style.css` — Layout adjustments for new button positions.

### Stage 5.19 — Commentary Feed Improvements ⬜ - human note, this has been done

**Goal:** New commentary messages appear at the top (most-recent-first). The feed is split into two sections: AI internal monologue (AI vs AI strategy discussion) and Player Chat (AI ↔ human conversation).

**Implementation:**

- `web/static/game.js` — `addCommentaryLine()` prepends to `commentary-feed` instead of appending (`insertBefore(el, feed.firstChild)`). Separate containers: `\#commentary-ai-feed` and `\#commentary-chat-feed`.

- `web/templates/index.html` — Two labelled sub-sections inside the commentary panel.

- `web/static/style.css` — Divider styling between the two feeds.

### Bug 5.11-H — Free Pieces Not Assembling Into Mills ✅

**Symptom:** In the move phase, isolated AI pieces with no nearby allies don't move toward forming mills unless there is an immediate threat. The AI leaves "stranded" pieces that never contribute to mill formations.

**What should happen:** Any piece not contributing to a current or developing mill should move toward the nearest group of same-color pieces that could form a mill, unless it is needed to block a specific opponent threat. "Free piece assembly" is the book's term for this — pieces that are not blocking anything should gather into productive formations.

**Fix:**

- `ai/heuristics.py` — Add `\_free\_piece\_assembly(board, color)`: for each own piece not participating in any 2-config or closed mill, measure its distance (in adjacency hops) to the nearest same-color piece that IS in a 2-config. Sum reciprocals (closer = higher score) to reward pieces gathering toward productive formations. Weight ~40 in move phase.

- `ai/heuristics.py` — Add `\_path\_to\_mill(board, color)`: count own pieces that are 2 adjacency hops from an empty slot in an existing own 2-config (enabling closure in 2 moves). Weight ~60 in move phase. This extends `setup\_mill` semantics into the movement phase and catches pieces 2 moves away from joining a forming mill.

### Bug 5.11-E — AI Not Moving Toward Mill Closure or Block in Move Phase ✅

**Symptom:** During the move phase the AI sometimes fails to move a piece toward an open mill line (to form or complete it), or fails to move to block an opponent's developing mill. The issue can occur at any position — the examples below are illustrative only, not exhaustive.

**Root cause (suspected):** The `\_mill\_threats` term only rewards mills closeable in **one** adjacency move from the current position. Pieces that are two hops away from joining a mill line contribute nothing. When positional value of a cross-node (c3 = 3-connection, high `\_position\_value`) competes with a blocking move that only triggers `block\_opponent\_mill` at moderate weight, the positional score can win incorrectly.

**Which sliders currently cover this:**

- `close\_mill` (500) — bonus for closing a mill *this* move (correct, fires when mill closes)

- `block\_opponent\_mill` (400) — bonus for neutralising an opponent's immediately closeable mill (should fire for d3→d2, investigate why it doesn't dominate)

- `setup\_mill` (100) — bonus for new 2-configs, but **placement phase only**; does not help in move phase

**Fix:**

- `ai/heuristics.py` — Extend `setup\_mill` logic into move phase: add a `\_two\_configs\_gained\_move` delta that rewards gaining a new own 2-config (two pieces + one empty slot in a mill line) during the move phase, weighted at ~120. This captures c5→c4 creating c3-c4-c5 with c3 empty = new 2-config.

- `ai/heuristics.py` — Add a `\_mill\_approach\_bonus`: in move phase, reward moves that bring an own piece adjacent to the empty slot of an existing own 2-config (enabling closure next turn). Weight ~80.

- Investigate why `block\_opponent\_mill` (400) does not override c3 positional score: add debug logging to `tactical\_move\_bonus` for the specific position.

### Bug 5.11-F — AI Resign Does Not Trigger LLM Debrief ✅

**Symptom:** When the AI resigns (via the `offer\_defeat` path added in Stage 5.11-D), the MillsAI post-game debrief does not run. The game ends but no summary or debrief commentary is shown.

**Root cause:** The `ai\_resignation` path in `web/app.py` sends `game\_over` without calling `coordinator.on\_game\_end()`, which is where `mills\_llm.debrief\_game()` is triggered.

**Fix:**

- `web/app.py` — In the `offer\_defeat` / `ai\_resignation` handler, call `await asyncio.to\_thread(session.coordinator.on\_game\_end, board, result="ai\_resignation")` before sending the `game\_over` WebSocket message so the debrief runs just as it does for normal termination.

### Bug 5.11-G — Self-Play Crashes in Parallel Mode ✅

**Symptom:** `tools/self\_play.py` crashes when run with parallel game flags, especially with `--parallel N \> 1`.

**Root cause (suspected):** Multiple game threads sharing mutable state (opening book file I/O, ChromaDB writes, or the trajectory DB index) without locking.

**Fix:**

- `tools/self\_play.py` — Add a `threading.Lock` around all `book.save\_opening()`, `memory.save\_game\_record()`, and `trajectory\_db.add\_game()` calls in parallel worker threads.

- `ai/opening\_book.py` — Make `save\_opening()` and `merge\_duplicates()` thread-safe (file-level lock or in-memory lock around JSON read-modify-write).

- Reproduce the crash deterministically with `--parallel 2 --games 4 --no-llm`, then confirm the fix.

### Stage 5.21 — Bad Move Button Fix ✅

**Goal:** After pressing "Bad Move", the AI must not replay the same bad move in its next attempt. Currently the ban is saved to `bad\_moves.json` but the in-memory TrajectoryDB in the running server instance is not queried for bans when the coordinator re-runs `deliberate()`.

**Root cause:** The coordinator queries the TrajectoryDB via `trajectory\_hints` before scoring, but the ban only applies as a −0.5 override *within* the TrajectoryDB. If the AI's root search at depth ≥5 finds the banned move optimal through pure alpha-beta, the trajectory hint penalty (scaled from −0.5) may not be large enough to override the heuristic score.

**Fix:**

- `web/app.py` `bad\_move` handler — after restoring engine state, pass `banned\_moves: set\[str\]` to the coordinator so it can be injected as a hard exclusion (not just a score penalty).

- `ai/coordinator.py` — Accept `banned\_moves` in `deliberate()`; filter them from `get\_all\_legal\_moves()` result before scoring.

- `ai/game\_ai.py` — Accept `excluded\_moves: set\[str\]` in `choose\_move()`; skip any move whose notation matches.

### Stage 5.22 — Self-Play Book Variety ✅

**Goal:** Self-play games should start from different opening positions, not all converge on the single highest-UCB1 opening. Each game should force a different book start.

**Implementation:**

- `tools/self\_play.py` — Before each game, call `book.select\_opening(ai\_color='W', exploration\_rate=1.5)` (high exploration) and lock the first 4 placement moves to that opening's sequence. Both AIs follow the forced start, then play freely.

- Or: keep a round-robin index over all openings for the session and cycle through them.

### Stage 5.16 — Starting Play Variants & Opening Database ✅

**Goal:** Extend opening recognition into a richer, staged starting-play system that identifies early deviations, stores named variant lines, and lays the groundwork for a searchable opening database the AI can consult by structure, move sequence, and outcome.

#### Recognition Windows

The opening system should no longer treat the full placement phase as a single recognition bucket. Instead, it should recognise three linked stages:

1. **Early starting play recognition (first 6–8 moves)** — detect broad intent, shape families, and early forcing motifs before enough pieces exist for a full named opening match.

2. **Mid-placement recognition (12 pieces placed total)** — detect stronger structural commitments once both sides have enough material on the board for variant branching to become meaningful.

3. **Final placement recognition (end of placement)** — lock in the final named opening or variation once the full placement sequence is known.

This allows the AI to reason about likely continuations earlier, not just after a full placement line is complete.

#### Starting Play Variant Structure

Store recognised starting-play sequences in a dedicated variant structure rather than only as flat opening strings. Each variant record should include:

- `variant\\\_id` — stable identifier.

- `name` — human-readable opening / variation name.

- `stage` — `early`, `mid\\\_placement`, or `final\\\_placement`.

- `move\\\_sequence` — canonical move list.

- `normalised\\\_move\\\_sequence` — symmetry-normalised sequence for D4-equivalent matching.

- `board\\\_signatures` — board snapshots or hashes at 6–8 moves, 12 placed, and end of placement.

- `parent\\\_variant\\\_id` — link to the broader family this line belongs to.

- `tags` — keywords such as `double-mill`, `diamond`, `wrap-threat`, `defensive`, `aggressive`, `outer-square`, `inner-square`, `anti-wrap`.

- `outcomes` — win/loss/draw stats by side, difficulty, and follow-up branch.

- `strategic\\\_notes` — human-readable explanation of the plan.

- `recommended\\\_continuations` — best next moves by stage and resulting branch.

#### Mill Wrapping in Opening Recognition

Starting-play recognition should explicitly tag early structures that indicate a likely future **mill wrapping** attempt — that is, building a parallel mill beside an existing or likely mill so the first mill becomes awkward or immobilised for its owner. These structures should become searchable tags and also feed the tactical evaluator so the AI can prefer anti-wrap or pro-wrap continuations earlier in the game.

#### Searchable Opening Database Direction

The long-term goal is a searchable database so the AI can learn which moves perform best from any recognised starting-play branch and so human players can study openings by name, theme, and result. The database should support:

- Search by opening name or alias.

- Search by move prefix.

- Search by board pattern / symmetry-normalised position.

- Search by tags such as `double-mill`, `mill-wrap`, `diamond`, `feeder`, `defensive`, `aggressive`.

- Search by outcome statistics (best win rate for White / Black, strongest reply, most common deviation).

- Search by stage (`early`, `mid\\\_placement`, `final\\\_placement`).

#### Implementation

- `ai/opening\\\_recognizer.py` — Refactor recognition into a staged pipeline: `recognise\\\_early\\\_starting\\\_play()`, `recognise\\\_mid\\\_placement\\\_variant()`, and `recognise\\\_final\\\_placement\\\_variant()`.

- `ai/opening\\\_book.py` — Add `StartingPlayVariant` dataclass / schema with parent-child relationships and tag support.

- `data/openings/starting\\\_play\\\_variants.json` — New canonical store for staged move sequences, tags, notes, and outcome stats.

- `ai/coordinator.py` — Inject recognised stage + variant context into both minimax move ordering and MillsAI prompts.

- `tools/list\\\_openings.py` — Extend to list staged variants, tags, aliases, and branch statistics.

- `tools/teach\\\_opening.py` — Extend to add or edit staged variants rather than only final openings.

- `tools/import\\\_openings.py` — Import broader starting-play families and preserve move-sequence ancestry.

- `web/app.py` / `web/static/game.js` — Surface recognised starting-play family, current branch, and tags in the GUI info panel during placement.

#### Deliverables

- `ai/opening\\\_recognizer.py` — Three-stage recognition pipeline.

- `ai/opening\\\_book.py` — `StartingPlayVariant` structure, tag indexing, branch statistics.

- `data/openings/starting\\\_play\\\_variants.json` — Initial staged opening / variation database.

- `tools/list\\\_openings.py` — Tag-aware opening browser output.

- `tools/teach\\\_opening.py` — Variant authoring support.

- `tests/test\\\_opening\\\_variants.py` — Tests for 6–8 move recognition, 12-piece recognition, final placement recognition, symmetry normalisation, and tag persistence.

#### Long-Term Follow-On

Once enough self-play and human-play data exists, promote the variant store into a proper searchable database layer (SQLite or equivalent) so the AI can retrieve best continuations by opening family, branch, and outcome history rather than relying only on static JSON files. That future database should remain compatible with the `StartingPlayVariant` structure introduced here.

### Stage 5.17 — Game Trajectory Memory & Winner-Aware Learning ✅

**Goal:** Give the AI a persistent, full-game memory so it can learn which moves historically correlated with wins — covering the entire game, not just the opening placement phase.

**Problem addressed:** The opening book previously stored games with `side='both'` regardless of who won, meaning the AI could follow a losing side's moves when targeting a book opening. The AI had no memory of games beyond the placement phase.

**Delivered:**

- `ai/trajectory\_db.py` — `TrajectoryDB` class. Indexes every saved game JSONL file by move-sequence prefix at checkpoint depths 4–48. `query(notations, color)` returns a `\{notation: float\}` score-delta dict centred on 0 (+0.5 = 100 % win rate for that colour at that branch, −0.5 = 100 % loss). Normalises `×`/`x` notation variants.

- `ai/game\_ai.py` — Added `\_move\_notation(move)`, `\_apply\_trajectory\_hints(scored, hints)`, and `trajectory\_hints` parameter to `choose\_move()` and `\_iterative\_deepen()`. Trajectory bonuses are scaled to ±`opening\_adherence`% × 3000 so they complement but don't overwhelm opening-book bonuses in the early game.

- `ai/coordinator.py` — Accepts `trajectory\_db` in constructor. In `deliberate()`, queries the DB with the current game's move-notation prefix and passes hints to `choose\_move()`. In `on\_game\_end()`, calls `trajectory\_db.add\_game()` so every completed game (including human wins) is immediately indexed for future play.

- `ai/opening\_book.py` — `select\_opening()` now filters to `side in (ai\_color, "both")` so the AI only targets openings where its colour plays the winning side.

- `ai/coordinator.py` (on\_game\_start) — Guards `\_target\_opening` with a side check so a stale or 'both' entry is only accepted when the AI colour matches.

- `tools/import\_book\_games.py` — Sets `side = winner` ('W'/'B') for games with a known winner; 'both' for draws/unknown. Cleans stale `book-\*` openings from `openings.json` before each re-import. Added to `requirements.txt`.

- `web/app.py` — Instantiates `\_trajectory\_db` at startup (reloaded once from disk), passes it to every `Coordinator`. After each game the coordinator's `on\_game\_end` call keeps it live without a full reload.

**How winner-aware learning works:**

1. Book games are stored with `side='W'` or `side='B'` for clear winners. `select\_opening()` only offers W-winning openings when the AI is W (and vice versa), so it always follows the winning side's placement moves.

2. As the game progresses into the movement phase, `TrajectoryDB` takes over. It finds the longest-matching prefix from all 116 indexed games (book + self-play) and returns per-move win-rate deltas. Moves that historically won get a positive bonus; moves that historically lost get a penalty.

3. If the opponent follows a known losing trajectory, the AI naturally plays the historical winning counter-moves because those are the next moves in the indexed game branches with high win rates.

4. Every completed game — including games where the human wins — is immediately added to the trajectory index, so the AI can attempt those same winning moves in future play.

### Stage 5.24 — Endgame Position Memory ✅

**Goal:** Give the AI a position-based endgame database so it can learn from historical endgame positions (≤11 total pieces, post-placement) independent of how those positions were reached — complementing TrajectoryDB's move-prefix index with exact board-state matching.

**Problem addressed:** TrajectoryDB is prefix-based and only matches when the current game's move sequence appears verbatim in historical games. Endgame positions can be reached via many different routes; prefix matching often finds no match by the time both sides have ≤6 pieces. The AI had no endgame-specific move guidance.

**Delivered:**

- `ai/endgame\_db.py` — `EndgameDB` class. Scans all JSONL game files; for each position with placement complete and ≤11 total pieces, indexes `board\_string|turn → \{notation: \{W, B, D, total\}\}`. `query(board, color)` returns a score-delta dict (same ±0.5 scale as TrajectoryDB) for candidate moves from the exact current position. Position key: the 24-char board layout + whose turn it is (order-independent of how the position was reached).

- `ai/coordinator.py` — Accepts `endgame\_db` in constructor. In `deliberate()`, when `endgame\_state.active`, queries `EndgameDB` and merges hints into `trajectory\_hints` (averaged when both sources fire). In `on\_game\_end()`, calls `endgame\_db.add\_game()` so every completed game immediately updates the index.

- `web/app.py` — Instantiates `\_endgame\_db` at startup alongside TrajectoryDB; passes it to every `Coordinator`.

- `tools/self\_play.py` — Loads `EndgameDB` once at startup; calls `endgame\_db.add\_game()` after each sequential fast-mode game. Added 50-half-move no-capture draw rule to prevent endgame stalling (analogous to chess 50-move rule). Now initialises from prior data so the first few games of a run can already benefit from historical endgame patterns.

**How position-based endgame learning works:**

1. The first time the AI encounters a position it has seen before in historical games, `EndgameDB.query()` returns positive deltas for historically winning moves and negative deltas for losing ones.

2. These are merged with any `TrajectoryDB` hints: averaged when both fire, used alone when only one matches.

3. The merged hints are passed to `choose\_move()` as `trajectory\_hints`, where they are scaled by `opening\_adherence` and applied on top of the minimax score — nudging the AI toward historically effective continuations without overriding sound tactical play.

4. As more games are played (self-play or human vs AI), new endgame positions accumulate automatically via `add\_game()`, so the database grows richer with every game.

**Known gaps:** Positional symmetry (rotation/reflection) is not exploited; two structurally equivalent positions in different board orientations will not match. Future work could add a symmetry-normalised key for deeper equivalence.

#### Bug 5.24-A — Self-Play Stalling After Move ~36 (Slow, Not Infinite)

**Symptom:** When running `python tools/self\_play.py --games 40 --white 5 --black 5 --blunder 0.1 -v --no-llm`, games become very slow after move 36 (the start of the movement phase endgame). The script does not hang indefinitely — it eventually resolves via repetition detection — but individual moves can take 3–8 seconds at difficulty 5 in deep endgame positions.

**Root cause:** At difficulty 5, the search uses iterative deepening with a time budget. In the fly phase (3v3), the branching factor is ~54 legal moves per side, which combined with a 2s budget and deep search causes each move to take several seconds. The `\_REPEAT\_DRAW = 3` repetition check and the new 100-half-move no-capture rule both fire correctly, but only after many slow moves.

**Mitigation applied (Stage 5.24):** Added a 100-half-move no-capture draw rule as an analogue of the chess 50-move rule. This bounds game length in cyclic endgame positions without stopping decisive games prematurely.

**Remaining fix:** Reduce self-play time budget at difficulty 5 when `fast\_early\_game=True` is set (or add a dedicated `fast\_self\_play\_cap` to `\_run\_fast\_game`). Target: ≤1s per move in self-play mode regardless of difficulty.

#### Bug 5.24-B — Ollama / LLM Not Working in Self-Play

**Symptom:** Running self-play with LLM mode (`python tools/self\_play.py --games N --white 5 --black 5`) fails or stalls. LLM commentary in the web interface is also non-functional.

**Root cause (suspected):** Ollama service is not running or the configured model has not been pulled. This is an environment/installation issue rather than a code bug.

**Fix:** Before using LLM mode, run:

```
ollama serve &  
ollama pull llama3.1:8b   \# or the model configured in data/settings.json
```

If Ollama is already running but the model is missing, only the pull is needed. The `install.sh` script should handle this automatically on first run.

**Code-side investigation needed:** Add a pre-flight check in `tools/self\_play.py` and `web/app.py` that pings the Ollama endpoint and logs a clear warning (not a crash) when it is unreachable.

### Stage 5.25 — AI Personality Selector in Header ✅

**Goal:** Add a compact personality picker to the top header bar, immediately to the right of the "New Game" button, so the player can switch AI personality with a single click before or between games. Random should be the default.

**User flow:**

1. A `\<select\>` or row of small toggle buttons appears in the header: `Random | Balanced | Aggressive | Defensive | Positional | Scholar | Chaos`.

2. Selecting a personality immediately updates `sel-game-personality` in the Settings panel so both controls stay in sync.

3. Default selection on page load: **Random** — AI personality changes each new game, as it does today.

4. When a new game starts, the selected personality propagates through the existing `personality` field in the `/new\_game` request body.

**Implementation notes:**

- Add a `\<select id="hdr-personality"\>` to the `\<header\>` element in `index.html`, mirroring the options in `\#sel-game-personality` (Settings panel).

- In `game.js`, sync both selects bidirectionally: any change in either updates the other.

- Style to match other header buttons: `border: 1px solid var(--border); background: none; color: var(--text-dim); padding: 4px 8px; border-radius: 4px; font-size: .85rem;`.

- No server changes required — personality already travels in the new-game payload.

### Stage 5.26 — Setup Position Button in Header ✅

**Goal:** Add a "Setup" button to the header bar next to the "Openings" toggle, giving fast access to the position-setup editor without needing to open Settings.

**User flow:**

1. A `\<button id="toggle-setup"\>Setup\</button\>` appears in the header between "Openings" and "New Game".

2. Clicking it opens the position-setup editor (shows `\#setup-panel`, hides `\#settings-panel`) exactly as if the player had opened Settings and clicked "Setup Position…".

3. Clicking again closes the setup panel.

4. The button follows the same `.btn-active` active-state styling as the other header toggles.

**Implementation notes:**

- Add button to `\<header\>` in `index.html`: `\<button id="toggle-setup"\>Setup\</button\>`.

- In `game.js`, add a click handler that mirrors the existing `\#btn-setup-toggle` logic (show `setup-panel`, hide `settings-panel`, toggle board into setup mode).

- Keep `\#btn-setup-toggle` in the Settings panel for users who navigate that way.

- CSS: same rule block as `\#toggle-moves`, `\#toggle-settings`, `\#toggle-openings`.

### Stage 5.27 — Game Counter & Automatic Library Consolidation ⬜

**Goal:** Every time 50 new opening or endgame positions have been added to the library (via self-play or human games), automatically trigger a consolidation pass: run `tools/name\_openings.py` to name any un-named novel openings, rebuild the TrajectoryDB and EndgameDB indexes, and log a summary.

**Motivation:** As self-play runs accumulate, the games directory fills with JSONL files and the TrajectoryDB/EndgameDB in-memory indexes grow stale relative to what's on disk. Un-named openings also accumulate (`needs\_llm\_name=True` in opening records). A periodic housekeeping step keeps the library clean and ensures the AI is always learning from the most current data.

**Trigger logic:**

- Maintain a `data/game\_count.json` file with `\{"total": N, "last\_consolidated": M\}`.

- After each game is written to `data/games/`, increment `total`.

- When `total - last\_consolidated \>= 50`, run consolidation and update `last\_consolidated`.

- Consolidation runs in a background thread (non-blocking).

**Consolidation steps:**

1. Call `tools/name\_openings.py` (or its library equivalent) to rename any openings with `needs\_llm\_name=True`. Skip if LLM is unavailable.

2. Reload `TrajectoryDB` and `EndgameDB` from disk (picks up all new games).

3. Log a summary line: `"Consolidated: N games, M openings named, P trajectory entries, Q endgame positions"`.

**Implementation:**

- `web/app.py` — `\_increment\_game\_counter()` called from `\_game\_over` and `on\_game\_end`. If threshold crossed, spawn `\_consolidate\_library()` as `asyncio.create\_task`.

- `tools/self\_play.py` — After each batch of 50 fast games, call consolidation (already loads EndgameDB; extend to reload TrajectoryDB too).

- `data/game\_count.json` — Persistent counter file.

#### Bug 9-A — Bad Move Ban Should Be Position-Specific ✅ Fixed

**Symptom:** Bad-move button was banning a notation for any position in the game (and persisting it to `bad\_moves.json` for all future games), which was too broad — the move may be valid in a different board configuration.

**Fix (implemented):**

1. `GameAI.banned\_game\_moves: set\[str\]` replaced with `\_pos\_bans: dict\[str, set\[str\]\]` — a dict mapping `board\_fen → set\[notation\]`.

2. `ban\_move(notation, board\_fen)` stores the ban under the exact position FEN at the time the move was made.

3. `choose\_move()` looks up `board.to\_fen\_string()` on each call — if any piece moves or is captured the FEN changes and the ban no longer applies.

4. Removed `\_trajectory\_db.save\_bad\_move()` call — bans are now session-local and position-specific rather than permanent global notation bans.

### Bug UI-A — AI Discussion Panel Hidden / No Scrollbar ✅

**Symptom:** The "AI Discussion" (AI ↔ AI commentary) panel was not visible and neither panel scrolled.

**Root cause:** CSS selector `\#commentary-col \> .commentary-panel` targeted DIRECT children of `\#commentary-col`, but the panels are inside `\#chat-view` (a child of `\#commentary-col`), so the `display:flex`, `min-height:0`, and `overflow:hidden` rules never applied. Without a bounded parent height the `overflow-y:auto` scroll on the inner divs had nothing to trigger against.

**Fix:** Changed selector to `\#chat-view \> .commentary-panel`, which correctly matches the panels as direct children of `\#chat-view`.

### Bug UI-B — Profile Tab Does Not Close When Clicked Again ✅

**Symptom:** Clicking the active "Profile" tab a second time did not dismiss the profile view or return to chat.

**Fix:** `web/static/game.js` — profile tab click handler now calls `\_switchLeftTab("chat")` when the profile view is already visible, toggling back to the chat.


---

## Completed Bugs & Enhancements

### Bug B-0 — New Openings Not Appearing in Openings Panel ✅

**Symptom:** Games played in recent sessions are not generating new entries in the Openings panel. All five games played after the last `learned\_openings.json` update have `recognised\_opening\_id: null` and `opening\_recognition\_status: null` in their saved game records.

**Root cause:** Opening recognition only runs when `use\_llm=True` (i.e. when the `Coordinator` is instantiated). The `Coordinator` owns the `OpeningRecognizer`. When LLM is disabled, `session.coordinator` is `None` and the no-LLM game-over path (`elif not session.vs\_human and session.game\_ai`) saves the game record to `TrajectoryDB`/`EndgameDB` **without ever running opening recognition**. As a result:

- Novel game sequences are never detected.

- Nothing is written to `learned\_openings.json`.

- The `/api/openings` endpoint (which reads `learned\_openings.json` on every call) has nothing new to return.

**Confirmed by inspection:**

- `data/openings/learned\_openings.json` — 54 entries, last modified 14:05.

- Five game records created after 14:05 all show `"opening\_recognition\_status": null`.

- The `OpeningBook.\_index` contains 65 entries (11 book + 54 learned) and the API correctly returns all 65 — but no new entries have been added for days of play.

**Secondary issue:** The Openings panel is populated once on page load (`DOMContentLoaded`). Even if a novel opening IS saved during a session (LLM path), the panel list does not refresh until the user reloads the page. The `name\_opening\_prompt` WS message is sent but the frontend only handles it as a naming prompt, not as a signal to refresh the dropdown.

**Fix:**

**Part 1 — Run opening recognition without LLM:**

- `web/app.py` — In `\_start\_game()` (the `new\_game` handler), always instantiate an `OpeningRecognizer` attached to the shared `OpeningBook`, even when `use\_llm=False`. Store it as `session.opening\_recognizer`.

- `web/app.py` — In the AI-move handler, forward each placement move to `session.opening\_recognizer.step(move)` so recognition progresses throughout placement phase.

- `web/app.py` — In the no-LLM game-over path (`elif not session.vs\_human`), after saving the trajectory/endgame record, run a lightweight novel-opening save:

- ```
if session.opening\_recognizer:  
    final = session.opening\_recognizer.get\_recognition()  
    if final and final.status in ("novel", "inactive"):  
        placement\_moves = \[m\["notation"\] for m in record.get("moves", \[\])  
                           if m.get("type") == "place"\]  
        if len(placement\_moves) \>= 6:  
            first3 = "-".join(placement\_moves\[:3\])  
            auto\_name = f"Novel — \{first3\}"  
            book = session.opening\_recognizer.book  
            book.save\_novel\_opening(  
                line\_moves=placement\_moves,  
                name=auto\_name,  
                needs\_llm\_name=True,  \# flag for future LLM naming pass  
                seed\_source="learned",  
            )
```

- The `needs\_llm\_name=True` flag allows `tools/name\_openings.py` to retroactively name these entries the next time it is run.

**Part 2 — Refresh the Openings panel when a new opening is saved:**

- `web/app.py` — After any novel opening is saved (both LLM and no-LLM paths), send a `"openings\_updated"` WebSocket message: `\{"type": "openings\_updated"\}`.

- `web/static/game.js` — Handle `"openings\_updated"`: call `\_loadOpenings()` to re-fetch `/api/openings` and rebuild the dropdown.

**Part 3 — Record opening status in no-LLM game records:**

- `web/app.py` — In the no-LLM game-over path, if `session.opening\_recognizer` is set, populate the game record fields before saving:

- ```
record\["recognised\_opening\_id"\]   = final.opening\_id if final else None  
record\["recognised\_opening\_name"\] = final.opening\_name if final else None  
record\["opening\_recognition\_status"\] = final.status if final else None
```

- This makes game records searchable and consistent regardless of whether LLM was on.

**Part 4 — Wire up the naming prompt in the frontend (MISSING):**

The server already sends `\{"type": "name\_opening\_prompt", "opening\_id": ..., "auto\_name": ..., "moves": \[...\]\}` after a novel opening is saved. This message is currently **not handled anywhere in game.js** — no case in `handleMessage()`, no UI element, nothing. The user never sees the prompt.

- `web/static/game.js` — Add `"name\_opening\_prompt"` case in `handleMessage()`:

  - Show a small inline dialog or a prompt in the commentary feed:

    - Text: `"New opening sequence detected! AI suggested name: "\<auto\_name\>". Accept or rename?"`

    - Two inputs: a text field pre-filled with `auto\_name`, and `\[Save\]` / `\[Skip\]` buttons.

  - On `\[Save\]`: send `\{"type": "rename\_opening", "opening\_id": ..., "name": \<userText\>\}` over `ws`.

  - On `\[Skip\]`: do nothing (the opening is already saved with the auto-name and `needs\_llm\_name: true`).

  - After Save: call `\_loadOpenings()` to refresh the dropdown so the new entry appears immediately.

- `web/static/game.js` — Add `"rename\_opening\_ack"` case in `handleMessage()`:

  - Call `\_loadOpenings()` to refresh the opening dropdown.

  - Show a brief confirmation in the commentary feed: `"Opening saved as '\<name\>'."`.

- A lightweight approach (no modal overlay needed) is to inject the prompt into the `\#commentary-ai` feed as a special styled entry with the two buttons inline, so it appears alongside other game commentary without interrupting play.

**Example of intended experience:**

After the game `1.d7 f4 2.d6 d5 3.b4 d2 4.b6 b2 5.f6×d2 a4 ...` ends, if the opening recognizer finds no match, the user should see in the commentary:

```
\[Opening\] New sequence detected: "Novel — d7-f4-d6" · Rename: \[\_\_\_\_\_\_\_\_\_\_\_\] \[Save\] \[Skip\]
```

**Deliverables:**

- `web/app.py` — `session.opening\_recognizer` in no-LLM games; lightweight novel-save in no-LLM game-over path; `openings\_updated` WS message.

- `web/static/game.js` — `openings\_updated` handler; `name\_opening\_prompt` handler with inline naming UI; `rename\_opening\_ack` handler.

- `web/static/style.css` — Styles for the inline naming prompt entry in the commentary feed.


### Enhancement B-6 — Trajectory: Exploit Opponent's Losing Lines ✅

**Goal:** When the current game's move sequence matches a historical game in which the **opponent's side eventually lost**, the AI should recognise this and apply a stronger, more deliberate follow-through on the winning continuation — not just a statistical hint, but an active exploitation of a known losing pattern the opponent is re-entering.

**Concept:**

The current `TrajectoryDB.query()` scores each candidate move by win-rate for `current\_color` across all matching historical games at the deepest matching prefix. This is symmetric: a move that historically won for White also gets penalised when queried from Black's side. The result is a statistical hint that nudges the AI toward historically better moves.

What is missing is a **pattern-recognition layer**: when the opponent's most recent moves exactly match the opening moves of a historical game that ended in their defeat, the AI should:

1. **Recognise the match** — "the human is heading into the sequence from Game X, which Black lost."

2. **Commit more strongly to the winning line** — increase the trajectory hint multiplier for moves that appeared in the winning side of that specific game, beyond the standard win-rate delta.

3. **Comment** — tell the player (via commentary) that this pattern was seen before and went badly for their side.

**Why the current implementation doesn't fully cover this:**

`TrajectoryDB` aggregates win/loss counts across *all* matching games at a given depth. If 10 games matched this prefix and White won 7 of them, White's move from those games gets roughly `+0.2` delta. But if the opponent's *exact move sequence* so far appears in 3 of those 10 games and White won all 3 of those, that signal (100% win rate on the exact sub-trajectory) is diluted into the aggregate `+0.2`. The AI cannot distinguish "this specific sequence is catastrophic for Black" from "White generally does well from here."

**Fix — Part 1: Opponent-loss trajectory query**

- `ai/trajectory\_db.py` — Add `query\_opponent\_loss(notations, opponent\_color)` method that queries the same index but scores moves by: `(loss\_by\_opponent / total)` rather than `(win\_by\_current / total)`. Returns `\{notation: float\}` deltas in `\[0, +0.5\]` where `+0.5` means the opponent lost 100% of the time from this position after this move. Uses the same D4 canonical lookup and prefix-depth fallback.

- ```
\# loss\_rate = how often opponent\_color lost when this move was played  
loss\_rate = (stats\[opp\_color\_loses\] / total)  
delta = loss\_rate - 0.5   \# range \[-0.5, +0.5\]; +0.5 = opponent always loses here
```

- Where `opp\_color\_loses` is: `stats\["W"\]` when `opponent\_color == "B"` (White wins = Black loses), or `stats\["B"\]` when `opponent\_color == "W"`.

- `ai/coordinator.py` — In `deliberate()`, after the standard `trajectory\_db.query()` call, call `trajectory\_db.query\_opponent\_loss(game\_moves, opp\_color)`. Merge the result into `trajectory\_hints` with a `loss\_exploitation\_multiplier` (default 1.5×): moves that appear in the opponent-loss query get their hint bonus scaled up by this factor.

- ```
exploit\_hints = self.\_trajectory\_db.query\_opponent\_loss(notations, opp\_color)  
for notation, delta in exploit\_hints.items():  
    if notation in trajectory\_hints:  
        trajectory\_hints\[notation\] = trajectory\_hints\[notation\] \* 1.5 + delta \* 0.5  
    else:  
        trajectory\_hints\[notation\] = delta \* 0.5
```

**Fix — Part 2: Commentary trigger**

- `ai/coordinator.py` — When `query\_opponent\_loss` returns at least one move with delta ≥ 0.3 (opponent lost ≥ 80% of the time from this position), set a flag `\_opponent\_in\_losing\_line = True` and record the approximate game depth at which this was detected.

- In `react\_to\_human\_move()`: if `\_opponent\_in\_losing\_line` was just set (first detection), emit a commentary line via `MillsAI` (or a direct string if LLM is off):

> - "I recognise this pattern — your position has entered a line where \[color\] has historically struggled. I'll press the advantage."

- Clear the flag after the commentary fires (so it only triggers once per recognised sequence, not every move).

**Fix — Part 3: Opening panel integration**

- In the Openings panel, add a visual indicator when the current game is tracking a historical losing line for the human player — a small "⚠ Danger Zone" label next to the matching opening name if available.

**Configuration:**

`loss\_exploitation\_multiplier` is exposed as a UI slider in the AI Tuning panel (range 0–3, default 1.5, step 0.1) under the existing **Behaviour** group alongside `make\_mistakes` and `opening\_adherence`. It follows the same path as all other tuning weights:

1. `WEIGHT\_DEFAULTS` in `game.js` declares the slider (key `"loss\_exploit"`, range 0–300, default 150, displayed as a percentage internally scaled to a float: `150 / 100 = 1.5×`).

2. `\_getWeights()` includes it in the `ai\_weights` dict sent with each `new\_game` request.

3. `HeuristicWeights.loss\_exploit: int = 150` stores it (same integer-percentage convention as `opening\_adherence`).

4. The coordinator reads `self.game\_ai.\_weights.loss\_exploit / 100.0` as the float multiplier when merging the exploit hints.

Personality presets in `PERSONALITY\_PRESETS` (game.js) set the value per personality:

| Personality | `loss\_exploit` | Multiplier |
| - | :-: | - |
| Aggressive | 200 | 2.0× |
| Defensive | 100 | 1.0× |
| Positional | 160 | 1.6× |
| Scholar | 180 | 1.8× |
| Balanced | 150 | 1.5× |
| Chaos | 50 | 0.5× |


Setting `loss\_exploit = 0` disables opponent-loss exploitation entirely and falls back to standard win-rate hints only.

**Example scenario:**

The human plays `1.d7 f4 2.d6 d5 3.b4 d2 4.b6 b2 5.f6×d2...`. TrajectoryDB finds that in a historical game (`learned\_openings.json` entry) this line led to Black losing at move 14. The standard hint would give White's move 5 response a `+0.2` delta. With opponent-loss exploitation, it gets `+0.35` delta (boosted by 1.5×), and the coordinator comments that Black entered a historically losing pattern.

**Deliverables:**

- `ai/trajectory\_db.py` — `query\_opponent\_loss(notations, opponent\_color)` method.

- `ai/coordinator.py` — Exploit hint merge in `deliberate()`; `\_opponent\_in\_losing\_line` flag; commentary in `react\_to\_human\_move()`.

- `ai/heuristics.py` — `loss\_exploitation\_multiplier` field in `HeuristicWeights`.

- `web/static/game.js` — Optional: Openings panel "Danger Zone" indicator when matching a losing line.


### Enhancement B-7 — Mill Abandonment: Locked Mill Escape and Redirected-Pin Creation ✅

**Goal:** Teach the AI two related tactics that both involve deliberately moving away from an established or partially-built mill:

1. **Locked mill escape** — recognise when a closed mill's cycling value is zero because every exit square is occupied by opponent pieces, and treat the pieces as "stranded capital" that should be redeployed elsewhere.

2. **Redirected-pin creation** — recognise that moving a piece away from a stalled mill attempt (where an opponent piece is already pinning the formation) can create a second simultaneous threat on that same blocker, freezing it in place while the AI operates freely elsewhere.

These are unified by the same underlying idea: a piece committed to a mill that cannot progress is worth less than a piece that creates new threats.


#### Form 1 — Locked Mill (Example 2)

```
1.f4 d6  2.f2 d2  3.f6×d6 d6  4.d7 e4  5.e3 g4  6.g1
```

After move 6 White has closed the f6-f4-f2 mill, but:

- f6 can only exit to d6 — occupied by Black.

- f4 can only exit to g4 or e4 — both occupied by Black.

- f2 can only exit to d2 — occupied by Black.

Every exit square from every piece in the mill is blocked. The mill **cannot cycle** — White can never open it to force a capture and re-close it. It contributes exactly zero dynamic value. White's other pieces (d7, e3, g1) are outside the locked formation and could form their own mill or 2-configs if the locked pieces were freed.

**Detection:** `\_is\_mill\_locked(board, color, mill)` — returns True when for every piece `p` in `mill`, every neighbour of `p` that is not another piece in the same mill is occupied by the opponent. Equivalently: the mill has no legal "open" move (no piece can slide out).

**Heuristic signal:**

- In `evaluate()`, add a move-phase penalty for each own locked mill: `−locked\_mill\_penalty × count`. Suggested default weight: 80 per locked mill. This makes the static position score reflect the immobility cost.

- In `tactical\_move\_bonus()`, add a bonus for moves that: (a) move a piece OUT of a locked mill, AND (b) the destination square is in a new 2-config or adjacent to a 2-config piece elsewhere (i.e., the piece contributes immediately to a new formation). Bonus: `locked\_mill\_escape` weight (default 160).

- The bonus must be gated: do not fire if breaking the mill gives the opponent an immediate capture (i.e., the moved piece was the only blocker of an opponent 2-config or 3-config).

**What this does NOT mean:** A mill with at least one free exit is NOT locked — the AI should keep it and try to cycle. Only zero-exit mills trigger this logic.


#### Form 2 — Redirected-Pin Creation (Example 1)

```
1.f4 d6  2.g1 e4  3.g4 g7  4.d7 d1  5.e5 f2
```

White has g1, g4 — a 2-config in the g7-g4-g1 mill. Black's g7 is pinning it (if g7 moves, White completes the mill). White also has d7 (adjacent to g7 via the outer top edge). The move **d7 → a7** does the following:

- d7 was on the a7-d7-g7 outer top mill line (where White had only d7, so no threat there yet).

- After d7→a7, White now has a7 and the square d7 is empty.

- Black's g7 is now simultaneously blocking **two** White formations:

  1. g7-g4-g1 (White has g4, g1; needs g7)

  2. a7-d7-g7 (White has a7; d7 is empty and one step from being filled; g7 is the shared closing square)

Black's g7 **cannot move** without surrendering at least one of these threats. If g7 → d7, the g7-g4-g1 threat is real (White has g4, g1 and g7 is now empty, reachable by a7→d7 or another piece). If g7 stays, White freely develops elsewhere.

The user's clarifying note: this would NOT apply if Black had a piece at d5, because d5 blocks the d7-d6-d5 cross line which doesn't directly affect the g7 pin — but if Black had a piece at d7 already (instead of g7 being the blocker), then d7 would BE a blocking piece and the situation is different.

**Detection:** `\_creates\_redirected\_pin(board, color, from\_sq, to\_sq)` — after the hypothetical move `from\_sq→to\_sq`:

1. Find all opponent pieces `b` such that `b` is the sole blocker of a White 2-config (White has 2 pieces in a mill line, `b` occupies the 3rd square).

2. For each such blocker `b`, check if `b` is also the **sole blocker or only missing piece** of a SECOND White 2-config or new potential mill line created by the move.

3. If yes: this move has created a "double-pin" on `b` — it must simultaneously guard two White threats.

**Heuristic signal:**

- In `tactical\_move\_bonus()`, add a bonus when `\_creates\_redirected\_pin` fires. Bonus: `redirected\_pin` weight (default 140, in the Tactical group alongside `mill\_wrapping`).

- The bonus should scale with the quality of the second threat: a clean 2-config (one empty square away from completing) scores the full bonus; a more distant formation scores half.

- Cap at 1 per move (even if the move creates double-pins on multiple pieces, award once).

**Relationship to mill\_wrapping:** Mill wrapping occupies exit squares of *opponent* closed mills. Redirected-pin creation forces an opponent piece to guard *own* forming mills. They are complementary: wrapping restricts the opponent from cycling; pin creation restricts the opponent from moving their blockers. Both should scale together, so expose `redirected\_pin` weight in the AI Tuning panel and include it in mill-wrapping-oriented personality presets (higher in Positional and Defensive, lower in Chaos).


#### Shared conditions where neither signal should fire

- **Fly phase**: pieces can jump anywhere; adjacency locks and pin relationships dissolve. Neither signal applies.

- **Placement phase**: pieces are being placed, not moved; locked-mill escape is irrelevant. Redirected-pin creation could theoretically apply during placement (placing a piece to create a double-pin on an opponent blocker) but is deferred to a future pass.

- **The escape is clearly losing**: if moving a piece out of a locked mill immediately allows the opponent to complete a mill next turn (the escaped piece was a blocker), the locked\_mill\_escape bonus must not fire.


**Deliverables:**

- `ai/heuristics.py` — `\_is\_mill\_locked(board, color, mill)` helper; `locked\_mill\_penalty` term in `evaluate()` move-phase; `locked\_mill\_escape` bonus and `redirected\_pin` bonus in `tactical\_move\_bonus()`.

- `ai/heuristics.py` — `\_creates\_redirected\_pin(board, color, from\_sq, to\_sq)` helper.

- `ai/heuristics.py` — `HeuristicWeights`: add `locked\_mill\_penalty: int = 80`, `locked\_mill\_escape: int = 160`, `redirected\_pin: int = 140`.

- `web/static/game.js` — `WEIGHT\_DEFAULTS`: add sliders for `locked\_mill\_escape` and `redirected\_pin` in the Tactical group; update `PERSONALITY\_PRESETS` with appropriate values.


### Enhancement B-8 — Forked Mill Blocking: Choose the Mill with Less Cycling Freedom ✅

**Goal:** When the opponent threatens to complete two mills simultaneously (a fork), teach the AI to block the mill that leaves the opponent with *more* cycling freedom — surrendering the mill whose pieces are more constrained — rather than always blocking the cardinal mill or defaulting to whichever scoring branch comes first.

**Core rule:** Block the fork arm whose closure would give the opponent the *highest* cycling freedom score. The surrendered arm should be the one with the *lowest* cycling freedom (i.e., the pieces would be nearly locked in place after closure, limiting the opponent's ability to harvest captures).

**Cardinal exception:** The existing `cardinal\_block` preference (block fork arms on cardinal squares) is *overridden* when an **own piece is already adjacent to a key exit square** of the cardinal arm. In that case the own adjacent piece functionally reduces the cardinal mill's cycling freedom; blocking the non-cardinal arm is better.


#### Key Concept — Cycling Freedom

A closed mill's **cycling freedom** is the count of empty exit squares reachable by any piece in that mill (squares that are not occupied by own pieces and are not the other mill pieces themselves). A mill with high cycling freedom can be repeatedly opened and re-closed to force opponent captures. A mill with zero exits is a locked mill (see B-7).

```
def \_mill\_cycling\_freedom(board, color, mill):  
    """Count of empty non-mill exit squares reachable from pieces in \`mill\`."""  
    mill\_set = set(mill)  
    count = 0  
    for sq in mill:  
        for nb in ADJACENCY\[sq\]:  
            if nb not in mill\_set and board.get(nb) is None:  
                count += 1  
    return count
```


#### Example 1 — Cardinal Exception (`1.f4 d6 2.g1 d1 3.b4 e3 4.g4`)

White threatens two mills simultaneously:

- **g7-g4-g1** — needs g7. If closed: only exit is g7→d7 (1 free exit at the time of analysis). Low cycling freedom.

- **g4-f4-e4** (cardinal arm, f4 is a cardinal square) — needs e4. If closed: exits are g4→g7, f4→f6, f4→f2, e4→e5 (4 free exits). High cycling freedom.

Black has a piece at e3. e3 is adjacent to e4 (an exit square of the cardinal arm g4-f4-e4). This own piece materially constrains the cardinal mill's practical cycling — if White closes g4-f4-e4 and tries to exit via e4→e5, Black at e3 cannot fully be ignored, and the e3 adjacency reduces effective cycling.

**Without the cardinal exception (naive rule):** block e4 (cardinal arm), give White g7-g4-g1. **With the cardinal exception:** Black's e3 is adjacent to e4 → override cardinal preference → block g7 instead → White gets the g4-f4-e4 cardinal mill but e3 constrains its cycling exit.


#### Example 2 — Standard Case (`1.d6 d2 2.f4 b4 3.c4 e4 4.d7 d5 5.f6`)

White threatens two mills simultaneously:

- **b6-d6-f6** — needs b6. If closed: d6 is surrounded by d7 (White), f6 (White), b6 (White), d5 (Black) — the only exit from d6 is back into the mill or toward d5 which is occupied. Very low cycling freedom (near-locked).

- **f6-f4-f2** — needs f2. If closed: f4 can exit to g4 (if empty) or elsewhere. Meaningfully higher cycling freedom.

**Decision:** block f2 (the arm with higher cycling freedom), surrender b6-d6-f6 to White. White's b6-d6-f6 mill will have near-zero cycling value — d6's neighbourhood is already crowded — so this concession costs little.


#### Detection and Signal

**Fork detection:** already partially present via `\_fork\_in\_n()` from B-4. Extend or create a specific helper:

```
def \_opponent\_fork\_arms(board, color):  
    """Return list of (mill, closing\_square) pairs where opponent  
    has 2 pieces in the mill and closing\_square is empty."""  
    opp = "B" if color == "W" else "W"  
    arms = \[\]  
    for mill in MILLS:  
        opp\_count = sum(1 for sq in mill if board.get(sq) == opp)  
        empty = \[sq for sq in mill if board.get(sq) is None\]  
        if opp\_count == 2 and len(empty) == 1:  
            arms.append((mill, empty\[0\]))  
    return arms
```

When 2+ arms exist (a fork), score each arm's closure by `\_mill\_cycling\_freedom(board, opp, mill)` evaluated *after* hypothetically placing the opponent piece at `closing\_square`.

**Cardinal exception check:**

```
def \_own\_piece\_adjacent\_to\_exit(board, color, mill):  
    """True if own piece occupies a neighbour of any non-mill exit square of mill."""  
    mill\_set = set(mill)  
    for sq in mill:  
        for nb in ADJACENCY\[sq\]:  
            if nb not in mill\_set and board.get(nb) is None:  
                \# nb is an exit square; check if own piece is a neighbour of nb  
                for nb2 in ADJACENCY\[nb\]:  
                    if nb2 not in mill\_set and board.get(nb2) == color:  
                        return True  
    return False
```

If `\_own\_piece\_adjacent\_to\_exit` returns True for the cardinal arm, remove the cardinal priority override for that arm and use pure cycling-freedom comparison.

**Heuristic signal:** In `tactical\_move\_bonus()`, when a move blocks the closing square of a fork arm that has the *higher* cycling freedom (or the non-cardinal arm when the cardinal exception fires), add a bonus scaled by the freedom differential:

```
freedom\_diff = high\_freedom - low\_freedom  
bonus = block\_cycling\_priority \* (1 + freedom\_diff \* 0.1)
```

New weight: `block\_cycling\_priority: int = 120` (in the Defensive group in AI Tuning).

**Gate:** only fire in placement phase and movement phase; not fly phase (cycling freedom analysis assumes adjacency constraints hold).


**Deliverables:**

- `ai/heuristics.py` — `\_mill\_cycling\_freedom(board, color, mill)` helper.

- `ai/heuristics.py` — `\_opponent\_fork\_arms(board, color)` helper.

- `ai/heuristics.py` — `\_own\_piece\_adjacent\_to\_exit(board, color, mill)` helper (cardinal exception check).

- `ai/heuristics.py` — `HeuristicWeights`: add `block\_cycling\_priority: int = 120`.

- `ai/heuristics.py` — `tactical\_move\_bonus()`: integrate fork-arm cycling comparison, apply cardinal exception, emit `block\_cycling\_priority` bonus.

- `web/static/game.js` — `WEIGHT\_DEFAULTS`: add `block\_cycling\_priority` slider in the Defensive group; update `PERSONALITY\_PRESETS` (higher for Defensive/Positional, lower for Aggressive/Chaos).


### Enhancement B-1 — Force Capture Button: Restrict to 4-Piece Gate ✅

**Goal:** The Force Capture button should only be enabled when the **human player** has exactly 4 pieces on the board — the condition where the AI's fly-sacrifice hesitation actually applies. Enabling it in all other situations is misleading and creates noise.

**Background:** The AI's `\_fly\_asym` heuristic penalises captures that would reduce the opponent to 3 pieces (triggering fly phase) unless `force\_aggressive=True`. This hesitation is only meaningful when the *opponent* (from the AI's perspective) is at 4 pieces. Since the human is the AI's opponent, the button only makes strategic sense when `humanPieces === 4`.

**Implementation:**

- `web/static/game.js` — In the `state` handler, update the Force Capture button's disabled logic:

- ```
// Current:  
$("btn-force-cap").disabled = (phase === "idle" || phase === "game\_over");  
  
// New:  
const humanPieces = Object.values(msg.board || \{\})  
  .filter(c =\> c === humanColor).length;  
$("btn-force-cap").disabled = (  
  phase === "idle" || phase === "game\_over" || humanPieces !== 4  
);
```

- The `humanColor` is the colour chosen by the player, already available as the `human\_color` field in the state message (or derivable from `gameState.turn` / `gameState.is\_human\_turn`).

- When `forceAggressive` is toggled on but `humanPieces` later moves away from 4 (e.g. after a capture), auto-disable the button and send `force\_aggressive: false` to keep server state consistent.

**Deliverables:** `web/static/game.js` — gated enable/disable logic.


### Enhancement B-2 — Placement Busy Scan: Defer Mill to Execute Chain ✅

**Goal:** When a level-4 forcing chain exists (last piece closes a mill), the AI should prefer executing the full chain — even when an immediate mill closure is available earlier — rather than taking the early mill and abandoning the superior sequence.

**Background:** `\_placement\_chain\_scan()` is currently gated by `\_closeable\_mills(before, color) == 0` (line 1439 of `heuristics.py`). This means the scan never fires when the AI already has a closeable mill on the board. In the following game the AI places piece 7 to close a mill instead of building the chain that closes with piece 9:

```
1.d6 d2  2.f4 b4  3.c4 e4  4.d1 d5  5.a7 g4  
6.d3  
   e5  
7.c5  ← AI should place here and keep building...  
   c3  
8.e3  b2  
9.f2  b6×d6          ← piece 9 closes mill, captures
```

or the similar variant:

```
1.d6 a4  2.f4 c4  3.b4 d5  4.d3 d2  
5.d1  c5   6.e5  c3×f4   7.a1 g1   8.b6 b2   9.f6×d2
```

**Root cause:** The gate condition `\_closeable\_mills(before, color) == 0` blocks the scan when the AI can already close a mill this turn. If `close\_mill` bonus (500) wins over the deferred-chain approach, the AI forgets the chain. The game examples show chains where piece 9 — not piece 7 — closes the decisive mill; pieces 6–8 are setup, not tactical.

**Fix:**

- `ai/heuristics.py` — Relax the gate: allow the chain scan to fire when `\_closeable\_mills(before, color) \> 0` **if** the scan returns level 4. Compare the chain bonus against `close\_mill` weight to decide whether to defer:

- ```
if (get\_game\_phase(before, color) == "place" and mills\_delta == 0):  
    if \_closeable\_mills(before, color) == 0:  
        chain = \_placement\_chain\_scan(after, color)  
    else:  
        \# Only run the expensive scan if a level-4 chain is plausible  
        chain = \_placement\_chain\_scan(after, color) if our\_rem \>= 2 else 0  
        if chain \< 4:  
            chain = 0  \# don't override close\_mill for weaker chains
```

- When `chain == 4` and a mill was available but not taken, apply a "defer mill" bonus of `weights.placement\_busy\_scan × 3` (level-4 bonus). This should exceed `close\_mill` (500) at default weight 120 × 3 = 360 — still lower, so increase the level-4 multiplier or add a separate `defer\_for\_chain: int = 180` weight that is added to the level-4 chain bonus when the AI deliberately skips an available mill.

- `ai/heuristics.py` — Fix the level-3 continuation condition at line 1189:

- ```
\# Current: our\_rem \>= 3  
elif t2 and opp\_rem \>= 2 and our\_rem \>= 2:   \# was \>= 3, now \>= 2
```

- This allows the scan to detect 4-level chains when the AI has exactly 2 pieces left to place (pieces 8 and 9).

- Add a `defer\_for\_chain: int = 180` weight field to `HeuristicWeights`. Default 180 added to the level-4 busy-chain bonus specifically when a mill was available but the chain involves a better late-closing mill.

**Scan description for reference (what `\_placement\_chain\_scan` can detect):**

The function performs a 4-half-move lookahead interleaving AI placements and forced opponent responses. At each step it identifies "threats" — squares the AI is 1 piece away from filling to close a mill — and checks whether the opponent can block all of them in a single placement. When the opponent cannot cover all threats (level 3: fork) or when the final AI placement closes a mill (level 4), it returns the quality level. Currently the function reliably finds:

- Level 1: a single immediate mill threat (opponent must respond or lose a piece next turn)

- Level 2: the threat survives one opponent block (still present after forced response)

- Level 3: a fork — two simultaneous threats the opponent cannot both cover with one piece

- Level 4: a complete forcing chain where piece 9 closes a mill

The missing behaviour is simply that the gate `\_closeable\_mills == 0` prevents levels 3 and 4 from registering when the AI already has a mill available at an earlier step.

**Deliverables:** `ai/heuristics.py` — relaxed gate, `our\_rem \>= 2` fix, `defer\_for\_chain` weight.


### Enhancement B-3 — Ring Crowding: Cardinal Position Preference ✅

**Goal:** When the opponent has concentrated 3+ pieces on a single ring (outer, middle, or inner), the AI should prefer **cardinal cross-node positions** adjacent to that ring — the connector squares between rings — rather than continuing to fill other positions on the same ring. This makes the AI start "surrounding" the opponent's ring cluster rather than passively matching it.

**Background:** The existing `ring\_crowding\_penalty` (weight 150) penalises the *AI* for putting 6+ own pieces on a single ring. It does not reward exploiting an opponent who over-commits to one ring. Cardinal squares (d7, g4, d1, a4, d6, f4, d2, b4, d5, e4, d3, c4) connect rings via cross-lines; controlling them constrains where the opponent's ring pieces can exit.

**Example (from user):**

```
1.d7 g4  
2.a4 d1   ← d1 blocks opponent from placing a 3rd outer-ring piece; AI should follow  
           with cross-node occupancy e.g. e4, c4, f4 to wrap the opponent's outer  
           ring structure rather than continue on a7/g7/g1.
```

**Fix:**

- `ai/heuristics.py` — Add `\_opponent\_ring\_concentration(board, color)`: for each ring, count opponent pieces; return a dict `\{ring\_id: count\}`. If any ring has ≥ 3 opponent pieces, flag it.

- `ai/heuristics.py` — Add a placement-phase bonus in `tactical\_move\_bonus()` for placing on a **cardinal square adjacent to a concentrated opponent ring**. The "cardinal squares adjacent to a ring" are the cross-line connector squares that can attack two squares on that ring simultaneously.

  - Outer ring connectors: `d7` (connects to `a7` and `g7`), `g4`, `d1`, `a4`.

  - Middle ring connectors: `d6`, `f4`, `d2`, `b4`.

  - Inner ring connectors: `d5`, `e4`, `d3`, `c4`.

- When `opponent\_ring\_concentration\[outer\_ring\] \>= 3`, bonus for placing on `d6, b4, f4, d2` (the middle-ring connectors that threaten the outer ring).

- Link to the existing `cardinal\_block` weight (currently 200) as the scaling factor: `cardinal\_block × 0.5` per concentrated-ring cardinal.

- This should have D4 symmetry: the same logic applies to any rotation of the described pattern.

**Deliverables:** `ai/heuristics.py` — `\_opponent\_ring\_concentration()`, cardinal-adjacent bonus in `tactical\_move\_bonus()`.


### Enhancement B-4 — Fork Mill Anticipation (Pre-emptive Mill Wrapping) ✅

**Goal:** The AI should detect when the opponent is **2–3 placements away** from completing a fork mill (two intersecting or shared-pivot mills simultaneously threatening two different mill closures), and preemptively block the anticipated convergence point — even before the second 2-config is formed.

**Background:** The existing `block\_opponent\_mill` (weight 400) only blocks mills the opponent can close **this turn** (1-step threat). The existing `stop\_opponent\_mills` (weight 450) penalises opponent 2-configs. Neither looks ahead far enough to block a fork before it materialises.

**Example (user):**

```
1.d7 f4  
2.a4      ← White placed d7 and a4; both are part of the outer-ring square.  
           At Black's move 2, the AI can predict White may place at a7 or g7 next  
           to complete a fork: a7-d7 (top cross) AND a7-a4-a1 (left column).  
           The anticipatory block is d1 — this occupies the bottom of the left column  
           before White can set up the fork.
```

This is conceptually a form of **mill wrapping** applied prospectively: the AI wraps (blocks the approach squares of) mills that don't yet exist. Link the bonus weight to the `mill\_wrapping` slider.

**Recognition criteria:** A "fork in N moves" exists when:

1. The opponent has 2 pieces in one mill line (2-config, needs 1 more).

2. The opponent has 1 piece in a *different* mill line that shares a square or pivot with the first.

3. There exists a single empty square that, once occupied by the opponent, would simultaneously: (a) complete the first mill OR (b) form a second 2-config in the shared line.

This is a "1.5-move fork threat" — not yet two 2-configs, but 1 placement would create two.

**Fix:**

- `ai/heuristics.py` — Add `\_fork\_in\_n(board, opp, n)` that returns the set of squares the opponent could place on within `n` moves to create a fork (two simultaneous mill threats). `n=1` is already covered by `stop\_opponent\_mills`; this function handles `n=2` and `n=3`.

- `ai/heuristics.py` `tactical\_move\_bonus()` — Add placement and move-phase bonus for placements that block a `\_fork\_in\_n(board, opp, 2)` square. Scale by `weights.mill\_wrapping × 0.6` (default 150 × 0.6 = 90 bonus per blocked anticipatory fork square).

- In the **move phase**, the same logic applies: moving to a square that blocks an anticipated fork earns the same bonus.

- Cap the bonus at 1 per move (avoid double-counting when multiple fork threats are blocked).

**Link to existing slider:** The `mill\_wrapping` slider (default 150, range 0–500) in the AI Tuning panel scales this bonus. Increasing `mill\_wrapping` makes the AI more proactively defensive; decreasing it makes the AI ignore anticipated forks and play purely tactically.

**Deliverables:** `ai/heuristics.py` — `\_fork\_in\_n()`, anticipatory bonus in `tactical\_move\_bonus()`.


### Enhancement B-5 — Free-Piece Assembly: Extended Range (Step 3–4) ✅

**Goal:** Extend the free-piece assembly incentive from a 2-step range to a 4-step range, with appropriately diminishing weights, so that genuinely isolated pieces — several moves away from any formation — still receive a pull toward productive groupings.

**Background:** The current implementation in `heuristics.py`:

- `\_free\_piece\_assembly()` (step-1, weight ×65): pieces directly adjacent to a 2-config piece.

- `\_assembly\_reach\_count()` (step-2, weight ×22): pieces 2 adjacency hops from a 2-config piece.

Any piece 3 or more steps away earns nothing and may stay isolated indefinitely. In a 4-piece endgame or after heavy captures, a piece might need to travel all the way around a ring to meet its allies — 3 or even 4 adjacency hops. Without an incentive, it stays where it is and never contributes.

**Benefits (as identified by user):**

- Avoids herding: a distant piece converging on the group forces the opponent to keep removing it (no time to prepare multiple mills).

- Reduces odds of losing to blockade: the opponent must spread attention to intercept the converging piece.

- Increases fly-phase probability: gathering all 3 pieces into a coherent group before they drop below 3 is essential.

**Fix:**

- `ai/heuristics.py` — Add `\_assembly\_step3\_count(board, color)` and `\_assembly\_step4\_count(board, color)` using the same "free piece not in any 2-config or closed mill" definition as the existing helpers, but measuring 3 and 4 adjacency hops to the nearest 2-config piece.

- Suggested weights: step-3: ×10, step-4: ×4. The gradient (65 / 22 / 10 / 4) creates a smooth pull: the closer a free piece gets to a formation, the more valuable its position becomes.

- `ai/heuristics.py` `evaluate()` — Add both new terms to the move-phase positional sum (alongside the existing ×65 and ×22 terms). Do NOT apply in fly phase (pieces can jump anywhere).

- Ensure adjacency-hop calculation uses the same path the engine already computes for mobility (no new BFS needed if the graph is already available).

**Note on "assembly across the board":** A piece may travel from one side of the outer ring to the other via up to 6 hops, but the 4-step cap still rewards progress. Once the piece comes within 4 hops it is on the radar; within 2 it is strongly pulled. The cap is intentional — a piece 5+ steps away provides no positional benefit and the AI should not sacrifice other priorities to drag it across.

**Deliverables:** `ai/heuristics.py` — `\_assembly\_step3\_count()`, `\_assembly\_step4\_count()`, new terms in `evaluate()` move-phase block.



---

## Stages 5.9, 5.10 and 6–12 (Complete)

### Stage 5.9 — Move Replay Viewer ⬜

**Goal:** Let the player step forward and backward through any moves of the game (mid-game or post-game) directly in the browser, with a "Back to Live" button to resume the active game.

**User flow:**

1. **During or after the game**, a Replay button is available in the top-left area (near Settings). Clicking it enters replay mode at the current position.

2. **◀ Prev** and **Next ▶** arrow buttons step one half-move at a time; current ply shown (e.g. "Move 7 / 24").

3. The moves list highlights the current ply.

4. A **"▶ Back to Live"** button returns the board to the current live position and re-enables play (if game still in progress).

5. **Start New Game** button moves to top-centre of the page for easy access mid-replay.

6. MillsAI commentary feed is unaffected (read-only during replay).

**Implementation sketch:**

*Server side:* No changes — full move list already in `state` message `moves` array.

*Client side:*

- Keep a `replayMoves\[\]` array (populated from every `state` message, not only final).

- Replay mode available any time (not only on `phase === "game\_over"`).

- `board.renderFromFen(fen)` — new method on `Board`.

- On "Back to Live": restore the board from the latest `state` message and resume normal play.

**Deliverables:**

- `web/static/board.js` — `renderFromFen(fen)` method.

- `web/static/game.js` — Replay state machine, prev/next, back-to-live handlers.

- `web/templates/index.html` — Replay button top-left; Start New Game top-centre.

- `web/static/style.css` — Replay button and Back-to-Live button styles.

### Stage 5.10 — Position Setup / Editor ⬜

**Goal:** Let the player drag pieces onto the board to set up any legal mid-game position before starting play.

**Rules enforced during setup:**

- Maximum 9 White and 9 Black pieces.

- If a player has a mill on the board at the start, the opponent loses one piece from their starting count.

- Minimum 3 pieces per side to start in move/fly phase.

**User flow:**

1. Toggle a **Setup Position** button in the Settings panel.

2. A piece palette appears (W piece, B piece, eraser); click any node to cycle: empty → W → B → empty.

3. A phase selector (`place` / `move`) and a turn selector (W / B) are shown.

4. **"Start from here"** validates the position and starts the game.

**Deliverables:**

- `game/board.py` — `BoardState.from\\\\\\\_positions()` class method.

- `web/app.py` — `setup\\\\\\\_game` handler.

- `web/static/game.js` — Setup mode state machine, palette, validation.

- `web/templates/index.html` — Setup toggle and palette UI.

- `web/static/style.css` — Palette styles.

### Stage 6 — Self-Play Training Loop ✅

**Goal:** Populate the opening book with real win-rate data and enrich LLM game history without requiring a human player.

**Script:** `tools/self\\\\\\\_play.py`

**Modes:**

| Flag | Description |
| - | - |
| (default) | Full LLM mode — coordinator deliberates for White, comments on Black's moves |
| `--no-llm` | Fast mode — no LLM calls; two raw `GameAI` instances; recommended for bulk runs |
| `--swap` | Alternate which engine plays White each game to reduce first-mover bias |
| `--blunder P` | White makes a random blunder with probability P (generates varied game data) |
| `--summary` | Ask LLM for a batch summary after all games complete |
| `--games N` | Number of games to play |
| `--white D` | Difficulty for White engine (1–10) |
| `--black D` | Difficulty for Black engine (1–10) |


**What improves from self-play:**

1. **Opening book win rates** — UCB1 scores updated after every game.

2. **Novel opening discovery** — new sequences saved and named, then attached to the staged starting-play variant structure where possible.

3. **LLM game context** — games land in `data/games/`; MillsAI reads them before web games.

4. **Pattern analysis** — `MemoryManager.analyse\\\\\\\_patterns()` distils placement patterns into the coordinator's narrative-memory prompt.

**Recommended usage:**

```
\\\\\\\# Quick opening-book warm-up (~5 min):      
python tools/self\\\\\\\_play.py --no-llm --games 100 --white 6 --black 6 --swap      
      
\\\\\\\# Overnight deep training run:      
python tools/self\\\\\\\_play.py --no-llm --games 500 --white 8 --black 6 --swap --blunder 0.05      
      
\\\\\\\# LLM-enriched run (slow, generates commentary + novel opening names):      
python tools/self\\\\\\\_play.py --games 20 --white 7 --black 5 --summary
```

### Stage 7 — Heuristic Parameter Evolution ✅

**Goal:** Automatically tune the weights in `ai/heuristics.py` to maximise win rate in self-play.

**Delivered (2026-05-18):**

- `tools/evolve\_weights.py` — (1+1) evolution strategy driver. Each generation: mutates baseline weights by Gaussian noise (σ = 12% of each weight by default), evaluates candidate vs baseline in N symmetric games (colours swapped), promotes if candidate win rate ≥ threshold (default 55%). Saves `data/weights/best.json` and numbered checkpoints on every promotion. Logs all generations to `data/weights/evolution\_log.jsonl`.

- `web/app.py` — Loads `data/weights/best.json` at server startup. Exposes evolved weights via `GET /api/weights` (merged with user-saved settings; user overrides take priority). In `new\_game`, applies evolved weights as the default base before personality/slider overrides.

**Usage:**

```
\# Quick 20-generation run, 4 parallel workers  
python tools/evolve\_weights.py --generations 20 --parallel 4  
  
\# Continue hill-climbing from the best weights found so far  
python tools/evolve\_weights.py --generations 50 --from-best --parallel 4  
  
\# Strong evaluation (difficulty 6, 30 games/generation)  
python tools/evolve\_weights.py --difficulty 6 --games-per-gen 30 --parallel 6
```

**Flags:** `--generations N`, `--games-per-gen G`, `--difficulty D`, `--sigma S`, `--threshold T`, `--parallel N`, `--from-best`, `--seed S`

**Weight priority (lowest → highest):** evolved best.json → user-saved settings.json → per-game personality sliders

**Anti-overfitting note:** Evaluations are symmetric (colour swap) to avoid first-mover bias. Future work: periodically test promoted weights against a fixed reference engine (difficulty 4 or 8) to catch overfitting to the current baseline's style.

### Stage 8 — Adaptive Difficulty ✅

**Goal:** Keep human games competitive by auto-adjusting difficulty to match the player’s skill in session.

**Delivered (session-only, 2026-05-18):**

- `web/app.py` — `AdaptiveTracker` class. Persists across `new\_game` messages on the same WebSocket connection. After 3 consecutive losses, automatically drops difficulty by 1 and adds 15% extra blunder rate (capped at 35%). After 3 consecutive wins at a softened level, restores difficulty by 1. After 3 consecutive wins at the player’s chosen level, suggests trying the next difficulty. If the player manually changes difficulty, all streaks reset.

- `web/static/game.js` — Handles `adaptive` field in `game\_over` message: shows commentary for soften/restore/suggest events, updates `\#adaptive-badge` pill near the status bar.

- `web/static/style.css` — `.adaptive-badge` and `.adaptive-softened` (amber pulsing) styles.

**Library protection:** Adaptive games are flagged with `self\_play: false` and the adaptive blunder boost only affects live blunder injection, not heuristic weights. The trajectory and endgame DBs index moves by board state and outcome — a softened loss still records what happened; it does not teach the AI to play badly (the AI’s own move search is unchanged; only blunder injection probability increases). See Stage 8 Library Protection note below.

**Future extensions:**

- Persistent player profile (across sessions) — see Stage 10.

- Elo estimation from recent outcomes.

- Per-personality adaptive ranges (Scholar adjusts slower than Chaos).

#### Stage 8 Library Protection Note

When adaptive difficulty lowers the AI's level and increases blunder rate, there is a risk that the resulting games (lower-quality moves, intentional blunders) are stored in `data/games/` and used to train TrajectoryDB and EndgameDB. This would teach the AI that bad moves are "good" in those positions.

**Current protection:** TrajectoryDB and EndgameDB only weight moves by the game *outcome* (win/loss/draw), not by which moves were played. A blundered move that leads to a loss is recorded as a losing move — so blunders naturally accumulate negative weight, not positive. The AI's own search is minimax and remains unaffected by blunder injection.

**Additional protection (to implement):** Tag adaptive-softened games in the JSONL record with `"adaptive\_softened": true`. TrajectoryDB and EndgameDB loaders should skip or down-weight these records so they don't pollute the opening/endgame library with beginner-level play patterns. See Bug 8-A below.

#### Bug 8-A — Adaptive Games Polluting TrajectoryDB / EndgameDB ✅ Fixed

**Symptom:** After many adaptive-softened games, TrajectoryDB move hints degrade as the AI "learns" from its own intentional blunders.

**Fix (implemented):**

1. `\_game\_over()` in `web/app.py` tags the JSONL record with `"adaptive\_softened": true` when `adaptive.extra\_blunder \> 0`.

2. Both `TrajectoryDB.\_index\_game()` and `EndgameDB.\_index\_game()` skip records where `adaptive\_softened == true`.

### Stage 9 — Tournament / Match Mode ✅

**Goal:** Let players compete through a gauntlet of 6 AI personalities and earn a rank.

**Delivered:**

- `web/app.py` — `TournamentState` class; `\_PERSONALITY\_WEIGHTS` dict (6 presets); `\_after\_game\_end()` closure in `ws\_endpoint`; `tournament\_start` WebSocket handler.

- Qualification gate: player must complete 3 normal games before tournament unlocks.

- Tournament roster (weakest → strongest): Chaos (diff 2, Elo 720) → Aggressive (3, 850) → Scholar (3, 900) → Balanced (4, 960) → Defensive (4, 1020) → Positional (5, 1080).

- Colours alternate: W/B/W/B/W/B for fairness.

- K=32 Elo update after each game; rank labels: Apprentice / Beginner / Intermediate / Advanced / Master.

- Tournament games use server-authoritative personality weights (no user-slider override).

- `web/templates/index.html` — `\#toggle-tournament` header button (disabled until 3 games played); `\#tournament-panel` sidebar with scoreboard table and final rank display.

- `web/static/game.js` — `sessionGames` counter; tournament unlock logic; handles `tournament\_init`, `tournament\_next`, `tournament\_update`, `tournament\_complete` WS messages; auto-sends `new\_game` with `tournament\_game: true` for each round.

### Stage 10 — Player Profiles & Persistent Stats ⬜

**Goal:** Allow players to sign in with a username, persist their stats and adaptive difficulty across sessions, and participate in tournaments against the AI personalities at appropriate difficulty levels.

**User flow:**

1. On first visit, player enters a name. Profile saved to `data/players/\<name\>.json`.

2. Stats shown in a small profile panel: games played, W/L/D record, current adaptive difficulty, Elo estimate, favourite personality (most played), best win streak.

3. Adaptive difficulty initialised from profile history rather than starting fresh each session.

4. **Tournament mode**: player joins a round-robin against all 6 personalities. Difficulty for each personality matchup is set automatically from the player's adaptive level. Results recorded to profile.

5. Leaderboard: `data/leaderboard.json` aggregates all player profiles. Viewable via `/leaderboard` page.

**Implementation sketch:**

- `ai/player\_profile.py` — `PlayerProfile` dataclass; load/save JSON; Elo update (`K=32`, initial `1000`; difficulty maps to Elo: `diff 1≈600, 3≈900, 5≈1100, 7≈1300, 10≈1600`).

- `web/app.py` — `GET/POST /api/profile/\<name\>`. `new\_game` accepts optional `player\_name`; loads profile and initialises `AdaptiveTracker` from stored state.

- `web/templates/index.html` — Player name input in Settings panel; profile stats row in Game Info panel.

- `web/static/game.js` — Store player name in `localStorage`; send with each `new\_game`; display Elo in info panel.

- Tournament: player clicks "Enter Tournament" → server queues 6 games (one per personality) at the player's current Elo-mapped difficulty → results panel shows standings.

### Stage 11 — Endgame Self-Play ✅

**Goal:** Rapidly build up EndgameDB with high-quality position data by running self-play games that start directly from generated endgame positions, bypassing the placement and mid-game phases entirely.

**Motivation:** Full-game self-play generates relatively few endgame positions per game (only the last dozen or so half-moves tend to fall below the 11-piece threshold). Dedicated endgame self-play can generate hundreds of tagged endgame positions per minute, making EndgameDB much more useful for move guidance.

**Approach:**

- `tools/endgame\_play.py` — new tool that generates random valid endgame starting positions (total pieces ≤11, both sides ≥3) and plays them out with the same headless engine used by `self\_play.py`.

- Positions should be plausible (not degenerate): both sides have pieces spread across rings, no side already in a lost position.

- Each completed game is saved to `data/games/` in the standard JSONL format so EndgameDB picks it up on the next server start (or incremental reload).

- Supports `--parallel N` and a `--positions FILE` flag to seed from a JSON list of specific FEN strings (useful for practising known problem positions).

**Deliverables:**

- `tools/endgame\_play.py` — position generator + headless self-play loop.

- Optional: `--seed-from-games` flag that extracts real endgame positions from existing `data/games/` JSONL files rather than generating random ones — ensures positions are always reachable.

- README section under **Self-Play Training** explaining endgame self-play and its effect on EndgameDB.

**Examples (planned):**

```
\# 500 random endgame positions, 4 workers, difficulty 5  
python tools/endgame\_play.py --positions 500 --parallel 4 --difficulty 5  
  
\# Replay real endgame positions extracted from existing game records  
python tools/endgame\_play.py --seed-from-games --positions 200 --parallel 4
```

### Stage 12 — Advanced Search (MCTS / Neural Evaluation) ✅ *(Stretch)*

**Goal:** Replace or augment negamax with Monte Carlo Tree Search, optionally with a learned value function.

**Approach:**

- `ai/mcts.py` — UCT-based MCTS with `heuristics.evaluate()` as rollout heuristic.

- Self-play generates (state, outcome) pairs for supervised training of a small MLP value network.

- Value network replaces rollout at MCTS leaves (AlphaZero-lite).

- MCTS and negamax can be toggled per-difficulty slot.

**Note:** Requires significant compute (GPU recommended). Designed to run offline on self-play records from Stages 6–7.

