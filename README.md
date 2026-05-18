# Nine Men's Morris — AI-Powered Web Game

A browser-based Nine Men's Morris game with a classical minimax engine, an Ollama-powered LLM commentary system, a curated opening book, trajectory-based and endgame position learning, and a fully tunable AI personality system.

![board](https://img.shields.io/badge/game-Nine%20Men's%20Morris-c8a96e?style=flat-square)
![python](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square)
![license](https://img.shields.io/badge/license-MIT-green?style=flat-square)

---

## Quick Start

```bash
git clone <repo-url>
cd NMM_ollama
./install.sh      # one-time setup: venv + Ollama + model
./run_nmm.sh      # starts server and opens browser
```

`install.sh` will:
- Create a Python venv at `.venv/`
- Install all Python dependencies
- Install [Ollama](https://ollama.com) if not already present
- Pull the configured LLM model (`llama3.1:8b` by default)

`run_nmm.sh` will:
- Start Ollama if it isn't running
- Launch the FastAPI server (`uvicorn`)
- Open `http://127.0.0.1:8000` in your browser automatically

---

## Requirements

- **Python 3.10+**
- **Linux / macOS** (WSL2 supported for Windows)
- **curl** (for Ollama install)
- ~5 GB disk for the default LLM model

---

## Features

### Game engine
- Full Nine Men's Morris rules: placement, movement, flying (3-piece phase), mill detection, and captures
- 10 difficulty levels — levels 1–8 use fixed minimax depth (2–9 ply); levels 9–10 use iterative deepening (20 s / 45 s time budget)
- Human vs AI, Human vs Human (local pass-and-play), or random colour selection
- Draw by threefold repetition, 50-move rule, and mutual agreement (Offer Draw unlocks after 40 post-placement half-moves)
- Undo rewinds both your move and the AI's response

### AI engine
- **Negamax + alpha-beta** with phase-aware heuristics:
  - Closed mills, blocked pieces, piece count, two-configurations, double-mill pivots
  - Mobility and immediate mill threats (phase-weighted)
  - Mill-cycle readiness (feeder mills), fork threats, herding and encirclement
  - Cross/cardinal node positional bonus (3-neighbour midpoint nodes score higher than 2-neighbour corners)
  - Fly-phase asymmetry bonus (prefer reaching 3 pieces before opponent in 4v4 endgame)
- **Tactical urgency layer** — delta-based bonuses applied at move-selection level (outside negamax to avoid sign-inversion):
  - Closing a mill; building or disrupting cycling mill setups
  - Blocking an opponent's immediately closeable mill; dismantling opponent two-configurations
  - Creating feeder diamond structures (four pieces adjacent to one key square, forming two simultaneous mill threats)
  - Mill wrapping — occupying exit squares of opponent closed mills so their pivot has nowhere useful to slide
  - Controlling cardinal squares; early-game scatter placement
- **Deadline-aware search** — checks the clock every 4 096 nodes; always returns the best partial result on timeout
- **Auto-force-move** — when the AI exceeds its expected thinking time the browser countdown fires `force_move` automatically; a server-side safety net fires 5 s later if the client message is lost
- **AI resignation** — if the human's position strength exceeds 0.95 (tanh-normalised) for 3 consecutive AI turns, the AI concedes with a farewell message
- **Force Capture** toggle — makes the AI capture aggressively, disabling the fly-sacrifice heuristic

### Opening book
- Curated opening lines with UCB1-scored selection; learns win/loss/draw outcomes per opening
- **Opening Recogniser** — detects rotated and mirrored variants via full D4 dihedral symmetry (4 rotations × 4 reflections)
- Novel openings discovered during play are saved with `needs_llm_name=True` and named on the next LLM-enabled run
- Opening Explorer panel (header → **Openings**) — browse and step-replay any named opening line

### Starting play detection
During the placement phase, the Game Info panel shows the **opening family** as it emerges — Outer Square, Diamond, Cardinal Cross, and other named configurations. White and Black families are shown independently when they differ.

### Trajectory-based learning (TrajectoryDB)
- All completed games are indexed by move prefix at server startup
- The AI consults win-rate statistics for the current move sequence when choosing openings, and avoids move sequences associated with losses
- Bad moves flagged by the player are recorded in `data/bad_moves.json` and excluded from future trajectory suggestions for the duration of the session

### Endgame learning (EndgameDB)
- Positions from completed games are stored and indexed by piece configuration
- Extra search depth is added automatically when the current position matches a known endgame pattern
- **Endgame Recogniser** detects named endgame phases (active / deep), zugzwang risk, and mill-cycle patterns

### LLM commentary (MillsAI)
- Consults a locally running Ollama model for move opinions, position commentary, and post-game session summaries
- Reads the last 10 games (with full move sequences) before each new game for richer context
- Remembers bad moves via ChromaDB vector store
- Comments on mill formations, strong moves (score ≥ 0.75), and poor human moves (capped to avoid spam)
- Asks periodic strategic questions to invite the player to think ahead
- **Player chat** — type a message at any point and MillsAI responds in context
- All LLM move recommendations are validated against the legal move list
- MillsAI can be disabled per game via the Settings panel checkbox

### Adaptive difficulty (session-only)
- After 3 consecutive losses, difficulty is automatically softened by one level with an on-screen notification
- Once softened, difficulty is gradually restored as the player wins
- After 3 consecutive wins, the UI suggests trying a harder difficulty

### AI Tuning and Personalities
- **13 configurable weight sliders** accessible via the **AI Tuning** header button (panel stays open during play); settings persist across sessions via **Save settings**:

  | Group | Slider | Default | What it rewards |
  |-------|--------|---------|-----------------|
  | Tactical | Mill closure urgency | 500 | Closing one of the AI's own mills this move |
  | Tactical | Cycling mill setup | 800 | Building two 2-configs whose empty closing squares are adjacent — a single pivot piece shuttles between them, forcing a capture every two turns |
  | Tactical | Block immediate mill threat | 400 | Neutralising an opponent 2-config closeable next turn |
  | Tactical | Disrupt opponent 2-configs | 450 | Breaking up any opponent two-piece mill setup |
  | Tactical | Feeder diamond creation | 300 | Building a diamond/fork structure — four pieces adjacent to one key empty square, forming two simultaneous mill threats |
  | Tactical | Mill wrapping | 250 | Occupying exit squares around opponent closed mills so their pivot cannot slide usefully |
  | Tactical | Block cardinal mills | 400 | Occupying or evicting pieces from cross-node (midpoint) squares |
  | Tactical | Early spread placement | 100 | Placing pieces away from existing own pieces in the first 6 placements |
  | Positional | Positional weight % | 100 | Overall multiplier on all non-tactical positional scoring |
  | Positional | Mill count weight % | 100 | How much each closed mill contributes to static evaluation |
  | Positional | Mobility weight % | 100 | How much having more legal moves than the opponent is valued |
  | Positional | Blocked pieces weight % | 100 | Bonus for having opponent pieces with no legal moves |
  | Behaviour | Make mistakes % | 0 | Probability (%) of playing a deliberately bad move each turn |

- **7 personality presets** — select from the header dropdown or Settings panel to pre-fill all sliders; dragging any slider switches to Custom. Per-personality settings are saved to `data/personalities/`:

  | Preset | Character |
  |--------|-----------|
  | **Random** | A different personality is chosen randomly at the start of each game |
  | **Balanced** | All defaults |
  | **Aggressive — The Crusher** | Hunts mills and cycling setups relentlessly, ignores wrapping defence, clusters pieces |
  | **Defensive — The Blocker** | Smothers every opponent threat, builds resilient diamond structures, wraps opponent mills |
  | **Positional — The Strategist** | Spreads across cross nodes, builds long-term cycling structures, thinks ahead |
  | **Scholar — The Bookworm** | Methodical opening placement, balanced diamond and wrapping awareness, solid all-round |
  | **Chaos — The Trickster** | Scatters randomly, ignores strategy, 45 % blunder rate |

### Web interface
- SVG board with coordinate labels (a–g, 1–7)
- Dark wood theme; three-column layout: MillsAI Chat | Board | Side panel
- Real-time eval graph (bottom bar) showing White/Black position strength across all moves
- **Countdown timer** in the status bar counts down remaining expected think time; fires Force Move automatically at zero
- Colour-coded move hints: green = legal placements, yellow = selectable pieces, red = capturable pieces
- Optimistic board rendering — your move appears instantly before the server confirms
- Mill highlight on capture; Hint system (3 per game) with LLM explanation
- Commentary feed with speaker labels (GameAI / MillsAI / Game)
- **AI resignation overlay** — distinct result screen when the AI concedes
- Move Replay viewer — step through the completed game forward and backward after it ends

---

## Controls

### Header bar

| Control | Action |
|---------|--------|
| **Moves** | Toggle the move list in the side panel |
| **Openings** | Toggle the Opening Explorer (browse and replay named openings) |
| **Setup** | Toggle the Position Setup editor (place pieces on the board manually) |
| **► New Game** | Start a new game with current settings |
| **Personality dropdown** | Switch AI personality; changes take effect from the next game |
| **Settings** | Toggle the New Game / settings panel |
| **AI Tuning** | Toggle the weight sliders panel (stays open during play) |

### Bottom bar

| Button | When visible | Action |
|--------|-------------|--------|
| **Force Move** (gold pulse) | While AI is thinking | Interrupt AI search immediately; AI plays the best move found so far |
| **Bad Move** | After AI plays a move | Flag the AI's last move as bad, undo it, and have the AI try a different move; the flagged move is banned for the rest of the game |
| **Force Capture** | Always | Toggle: forces the AI to capture aggressively, disabling fly-sacrifice strategy |
| **Offer Draw** | After 40 post-placement half-moves | Offer a draw; AI may accept or decline |
| **Hint (3)** | Human's turn | Request a hint; MillsAI explains the suggested move (3 hints per game) |
| **← Undo** | After a move is made | Rewind the last human move and the AI's response |

### Settings panel (right side)

| Control | Description |
|---------|-------------|
| **Play as** | White, Black, or Random |
| **Opponent** | AI or Human (local pass-and-play) |
| **AI Personality** | Per-game personality override; "Use current AI Tuning sliders" applies your custom weights |
| **AI Difficulty** | 1 (Beginner) – 10 (Maximum 45 s) |
| **MillsAI commentary** | Enable/disable LLM commentary for this game |
| **New Game** | Start the game |
| **Setup Position…** | Open the Position Setup editor |

### Position Setup editor

Click any board node to cycle it through empty → White → Black → empty. Set the game phase, whose turn it is, then click **Start from Here** to begin a game from that position. Useful for practising specific endgames or problem positions.

### Replay controls (post-game)

After a game ends the Replay panel activates. Use ⏮ / ◀ / ▶ / ⏭ to step through all moves, or **↩ Back to Live** to return to the live board view.

### Opening Explorer

Select any named opening from the dropdown to see its win/loss/draw record, then click **Replay Opening** to watch it played out at configurable speed. Choose **Practice — I play on** to continue from the end of the opening, or **Watch — AI continues** to observe both sides.

---

## Self-Play Training

```bash
python tools/self_play.py --no-llm --games 100 --white 6 --black 6 --swap --parallel 4
```

Self-play games are saved to `data/games/` and are read by the AI before each future web game, enriching its opening book win rates and LLM context.

| Flag | Description |
|------|-------------|
| `--games N` | Number of games to play |
| `--white D` / `--black D` | AI difficulty 1–10 per side |
| `--blunder P` | Blunder probability for White (0.0–1.0) |
| `--swap` | Alternate which side plays White each game |
| `--parallel N` | Run N games simultaneously (fast/no-llm mode only) |
| `--no-llm` | Skip all LLM calls — fast mode |
| `--personalities LIST` | Comma-separated list of personalities to randomly mix (e.g. `aggressive,defensive,positional`) |
| `--white-personality NAME` | Fix White to one personality (disables random mixing) |
| `--black-personality NAME` | Fix Black to one personality (disables random mixing) |
| `--name-openings` | Use LLM to name novel openings discovered during the run |
| `--summary` | Ask LLM for a batch summary after all games finish |
| `-v` / `--verbose` | Print a live board view after each move |

**Examples:**

```bash
# Fast parallel run at equal strength
python tools/self_play.py --games 40 --no-llm --parallel 4

# Mixed personalities to reduce draws
python tools/self_play.py --games 20 --no-llm --personalities aggressive,defensive,positional

# One game with verbose board output and LLM commentary
python tools/self_play.py --games 1 --white 5 --black 1 -v --white-personality scholar
```

### Weight Evolution

```bash
python tools/evolve_weights.py --generations 20 --parallel 4
```

Runs a (1+1) evolution strategy: each generation mutates the current best heuristic weights by Gaussian noise, plays the candidate against the baseline, and promotes the candidate if its win rate reaches ≥ 55 %. Best weights are saved to `data/weights/best.json` and loaded automatically on the next server restart.

| Flag | Description |
|------|-------------|
| `--generations N` | Number of evolution generations (default: 20) |
| `--games-per-gen G` | Games played per generation to measure win rate |
| `--difficulty D` | AI difficulty used for both sides |
| `--parallel N` | Run N games simultaneously per generation |
| `--from-best` | Seed the starting weights from `data/weights/best.json` |

**Examples:**

```bash
# Quick 20-generation run with 4 parallel games
python tools/evolve_weights.py --generations 20 --parallel 4

# Longer run continuing from the current best weights
python tools/evolve_weights.py --generations 50 --from-best --parallel 4
```

---

## Board Coordinate System

```
a7 ——— d7 ——— g7
|       |       |
|  b6 — d6 — f6  |
|  |    |    |  |
|  |  c5-d5-e5  |
a4-b4-c4    e4-f4-g4
|  |  c3-d3-e3  |
|  |    |    |  |
|  b2 — d2 — f2  |
|       |       |
a1 ——— d1 ——— g1
```

24 valid positions on three concentric squares connected by cross-lines.  
**Cross/cardinal nodes** (midpoints of each side, 3 neighbours): `a4 d7 g4 d1 b4 d6 f4 d2 c4 d5 e4 d3`  
**Corner nodes** (corners of squares, 2 neighbours): all remaining 12 positions.

---

## Configuration

Edit `data/settings.json` to change the Ollama model, URL, and LLM behaviour thresholds.

| Key | Default | Description |
|-----|---------|-------------|
| `ollama_model` | `llama3.1:8b` | Ollama model to use |
| `ollama_url` | `http://localhost:11434` | Ollama server address |
| `poor_move_threshold` | `0.3` | Score drop that triggers an LLM comment on a human move |
| `max_poor_move_comments_per_game` | `5` | Cap on poor-move LLM comments per game |
| `endgame_active_threshold` | `11` | Total pieces on board to enter endgame mode |
| `endgame_deep_threshold` | `8` | Total pieces to enter deep-endgame mode |

### Changing the LLM model

```bash
ollama pull mistral        # or any other Ollama model
```

Then update `data/settings.json`:
```json
{ "ollama_model": "mistral" }
```

The game uses the new model from the next game start.

---

## Project Structure

```
NMM_ollama/
├── game/                        # Core engine: board, rules, game engine
├── ai/
│   ├── game_ai.py               # Negamax + alpha-beta, blunder mode, weights
│   ├── heuristics.py            # Phase-aware evaluation + HeuristicWeights dataclass
│   ├── mills_llm.py             # Ollama LLM interface
│   ├── coordinator.py           # AI deliberation, commentary, resignation tracking
│   ├── opening_book.py          # Opening library + UCB1 selection
│   ├── opening_recognizer.py    # D4 symmetry-aware opening recognition
│   ├── endgame_recognizer.py    # Phase detection, zugzwang, mill-cycle patterns
│   ├── endgame_db.py            # Endgame position database (learned from games)
│   ├── trajectory_db.py         # Move-prefix win-rate index (learned from games)
│   ├── starting_play.py         # Opening family detection (Outer Square, Diamond, etc.)
│   ├── memory_manager.py        # Game record persistence and pattern analysis
│   └── debriefer.py             # Post-game session summary
├── web/
│   ├── app.py                   # FastAPI + WebSocket server, session management, adaptive difficulty
│   ├── static/
│   │   ├── game.js              # Game controller, personality presets, weight sliders, replay
│   │   ├── board.js             # SVG board renderer
│   │   └── style.css            # Dark wood theme
│   └── templates/index.html
├── tools/
│   ├── self_play.py             # AI vs AI training loop
│   ├── evolve_weights.py        # (1+1) evolution strategy to tune heuristic weights
│   ├── import_openings.py       # Import openings from strategy book text file
│   ├── import_book_games.py     # Import games into opening book
│   ├── name_openings.py         # LLM-name novel openings
│   ├── list_openings.py         # Print opening book summary
│   ├── teach_opening.py         # Interactively teach a new opening line
│   └── debrief.py               # Run post-session debrief manually
├── data/
│   ├── settings.json            # Runtime configuration
│   ├── openings/                # Opening book JSON (openings.json, book_openings.json)
│   ├── personalities/           # Saved per-personality weight files
│   ├── games/                   # Game records (JSONL, one file per game)
│   ├── bad_moves.json           # Player-flagged bad AI moves
│   ├── chroma/                  # ChromaDB vector store (LLM memory)
│   └── session_memory/          # LLM session narrative files
├── tests/                       # unittest test suite (160+ tests)
├── main.py                      # CLI harness for quick engine testing
├── install.sh                   # One-time installer
├── run_nmm.sh                   # Launch script
└── requirements.txt
```

---

## Running Tests

```bash
source .venv/bin/activate
python -m unittest discover tests/ -v
```

---

## License

MIT
