# Nine Men's Morris — AI-Powered Web Game

A browser-based Nine Men's Morris game with a classical minimax engine and an Ollama-powered LLM commentary system that learns from every game.

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
- Full Nine Men's Morris rules: placement, movement, flying, mill detection, captures
- 10 difficulty levels — level 1–8 use fixed minimax depth (2–9 ply); levels 9–10 use iterative deepening (20 s / 45 s time budget)
- Undo (rewinds both your move and the AI's response)
- Human vs AI, Human vs Human, or Random colour selection
- Draw by threefold repetition (both players oscillate), 50-move rule, and mutual agreement
- **Force Capture** toggle — makes the AI capture aggressively, disabling the fly-sacrifice strategy

### AI
- **GameAI** — negamax + alpha-beta with phase-aware heuristics:
  - Closed mills, blocked pieces, piece count, two-configurations, double-mill pivots, win configuration
  - Mobility difference and immediate mill threats (phase-weighted)
  - Cross/cardinal node positional bonus (3-neighbour nodes score higher than 2-neighbour corners)
  - Fly-phase asymmetry bonus (prefer reaching 3 pieces before opponent in 4v4 endgame)
  - tanh normalisation with per-phase scale (not a hard clamp)
- **Force Move** — interrupt the AI's current search at any time; it plays the best move found so far
- **Opening Book** — curated opening lines with UCB1-scored selection; learns win/loss/draw per opening; names novel openings via LLM
- **Opening Recogniser** — detects rotated and mirrored opening variants (full D4 dihedral group: 4 rotations × 4 reflections)
- **Endgame Recogniser** — detects named endgame phases, zugzwang risk, and mill-cycle patterns

### LLM commentary (MillsAI)
- Consults a locally running Ollama model for move opinions, position commentary, and post-game session summaries
- Reads the last 10 games (with full move sequences) before each new game
- Remembers bad moves via ChromaDB vector store
- Comments on **mill formations**, **strong moves** (score ≥ 0.75), and **poor moves** (capped to avoid spam)
- Asks periodic strategic questions to invite the player to think ahead
- **Player chat** — type a message at any point and MillsAI responds in context; conversations saved to game files
- All move recommendations are validated against the legal move list (LLM cannot suggest an illegal move)

### Web interface
- SVG board with coordinate labels (a–g, 1–7)
- Real-time game strength graph — shows White/Black advantage across all moves
- **Thinking time indicator** — status bar shows elapsed time and expected max wait while AI computes
- **Force Move button** (animated gold pulse) — visible while AI is thinking; interrupts search immediately
- Colour-coded hints: green = legal placements, yellow = selectable pieces, red = capturable pieces
- Optimistic board rendering — your move appears instantly before the server confirms
- Mill highlight on capture; **Hint** system (3 per game) with LLM explanation
- Commentary feed with speaker labels
- Settings: colour, opponent, difficulty 1–10, LLM toggle

---

## Board Coordinate System

```
a7 ——— d7 ——— g7
|       |       |
|  b6 — d6 — f6  |
|  |    |    |  |
|  |  c5-d5-e5  |  |
a4-b4-c4    e4-f4-g4
|  |  c3-d3-e3  |  |
|  |    |    |  |
|  b2 — d2 — f2  |
|       |       |
a1 ——— d1 ——— g1
```

24 valid positions on three concentric squares connected by cross-lines.  
**Cross/cardinal nodes** (midpoints of each side, 3 neighbours): `d7 g4 d1 a4 d6 f4 d2 b4 d5 e4 d3 c4`  
**Corner nodes** (corners of squares, 2 neighbours): all remaining 12 positions.

---

## Configuration

Edit `data/settings.json` to change the Ollama model, URL, difficulty defaults, and LLM behaviour thresholds.

| Key | Default | Description |
|-----|---------|-------------|
| `ollama_model` | `llama3.1:8b` | Ollama model to use |
| `ollama_url` | `http://localhost:11434` | Ollama server address |
| `difficulty` | `4` | Default AI difficulty (1–10) |
| `poor_move_threshold` | `0.3` | Score drop that triggers LLM comment |
| `max_poor_move_comments_per_game` | `5` | Cap on LLM poor-move comments |
| `endgame_active_threshold` | `11` | Total pieces on board to enter endgame mode |

---

## Project Structure

```
NMM_ollama/
├── game/               # Core engine: board, rules, game engine
├── ai/                 # GameAI, MillsLLM, Coordinator, heuristics
│   ├── game_ai.py      # Negamax + alpha-beta, blunder mode
│   ├── heuristics.py   # Phase-aware static evaluation
│   ├── mills_llm.py    # Ollama LLM interface
│   ├── coordinator.py  # AI deliberation + commentary pipeline
│   ├── opening_book.py # Opening library + UCB1 selection
│   └── opening_recognizer.py  # D4 symmetry-aware recognition
├── web/                # FastAPI server + browser frontend
│   ├── app.py          # WebSocket server, session management
│   ├── static/         # board.js, game.js, style.css
│   └── templates/      # index.html
├── data/
│   ├── settings.json   # Runtime configuration
│   ├── openings/       # Opening book JSON
│   ├── games/          # Game records (JSONL)
│   └── session_memory/ # LLM session narratives
├── tests/              # pytest test suite
├── install.sh          # One-time installer
├── run_nmm.sh          # Launch script
└── requirements.txt
```

---

## Running Tests

```bash
source .venv/bin/activate
pytest tests/ -v
```

---

## Changing the LLM Model

```bash
ollama pull mistral        # or any other Ollama model
```

Then update `data/settings.json`:
```json
{ "ollama_model": "mistral" }
```

The game will use the new model from the next game start.

---

## License

MIT
