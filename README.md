# Nine Men's Morris — AI-Powered Web Game

A browser-based Nine Men's Morris game with a classical minimax engine, an Ollama-powered LLM commentary system, a curated opening book, trajectory-based and endgame position learning, a fully tunable AI personality system, adaptive difficulty, and a 6-opponent Tournament Mode with Elo tracking. An optional self-learning neural AI (PyTorch, self-play RL) is available as a drop-in alternative engine — see [Learned (Neural) AI](#learned-neural-ai).

![board](https://img.shields.io/badge/game-Nine%20Men's%20Morris-c8a96e?style=flat-square) ![python](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square) ![license](https://img.shields.io/badge/license-MIT-green?style=flat-square)


## Quick Start

### Linux / macOS

```
git clone \<repo-url\>  
cd NMM\_LLM  
./install.sh      \# one-time setup: venv + Ollama + model  
./run\_nmm.sh      \# starts server and opens browser
```

### Windows

```
git clone \<repo-url\>  
cd NMM\_LLM  
install.bat       :: one-time setup: venv + Ollama + model  
run\_nmm.bat       :: starts server and opens browser
```

`install.bat` is the recommended entry point on Windows — you can also double-click it from Explorer. It just launches `install.ps1` with `-ExecutionPolicy Bypass` so PowerShell's default execution policy doesn't block the script. If you'd rather call PowerShell directly:

```
powershell -ExecutionPolicy Bypass -File .\\install.ps1
```

Optional flags (work on either `install.bat` or `install.ps1`):

| Flag | Effect |
| - | - |
| `/noollama` (`-NoOllama`) | Skip Ollama install and model download |
| `/yes` (`-Yes`) | Non-interactive; auto-installs Ollama |
| `/model NAME` (`-Model NAME`) | Override the LLM model (e.g. `mistral:7b`) |


The installer will:

- Create a Python venv at `.venv/`

- Install all Python dependencies

- Install [Ollama](https://ollama.com/) if not already present (optional on Windows)

- Pull the configured LLM model (`llama3.1:8b` by default)

The launcher (`run\_nmm.sh` / `run\_nmm.bat`) will:

- Start Ollama if it isn't running

- Launch the FastAPI server (`uvicorn`)

- Open `http://127.0.0.1:8000` in your browser automatically

**Optional — neural AI training** (PyTorch, only needed if you want to train the learned engine):

```bash
# Linux / macOS
source .venv/bin/activate
pip install -r requirements_learned_ai.txt

# Windows
.venv\Scripts\activate
pip install -r requirements_learned_ai.txt
```

See [Learned (Neural) AI](#learned-neural-ai) for the full training walkthrough.


## Requirements

- **Python 3.10+** — on Windows, install from [https://www.python.org/downloads/](https://www.python.org/downloads/) and tick **"Add python.exe to PATH"** during setup. The Microsoft Store stub is detected and skipped automatically.

- **Linux / macOS / Windows 10+** (WSL2 also works)

- **curl** on Linux/macOS (for Ollama install). Built into Windows 10+.

- ~5 GB disk for the default LLM model

- **Windows only — if `chromadb` fails to install**, install the free [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) (check "Desktop development with C++") and re-run `install.bat`.


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

  - Mobility and immediate mill threats (phase-weighted); fly-phase mobility capped at 5 to prevent fly-entry from looking artificially bad (B-63)

  - Mill-cycle readiness (feeder mills), fork threats, herding and encirclement

  - Cycling mill and fork threat weights increase sharply in fly phase (×80/×55) to reflect the urgency of dual-threat structures

  - Cross/cardinal node positional bonus (3-neighbour midpoint nodes score higher than 2-neighbour corners)

  - Fly-phase asymmetry bonus (prefer reaching 3 pieces before opponent in 4v4 endgame)

  - **Sealed 2-config detection** (B-59): 2-configs the opponent cannot contest score ~4× higher and receive elevated move-ordering priority, propagating forced-mill sequences through negamax

- **Tactical urgency layer** — delta-based bonuses applied at move-selection level (outside negamax to avoid sign-inversion):

  - Closing a mill; building or disrupting cycling mill setups

  - Blocking an opponent's immediately closeable mill; dismantling opponent two-configurations

  - **Dual-connected mill block** (B-55): extra urgency when blocking the closing square of a 2-config that would share a square with an already-closed opponent mill — two interconnected cycling mills are nearly unbeatable

  - Creating feeder diamond structures (four pieces adjacent to one key square, forming two simultaneous mill threats)

  - Mill wrapping — occupying exit squares of opponent closed mills so their pivot has nowhere useful to slide

  - Controlling cardinal squares; early-game scatter placement

  - **Cycling-capture unblock penalty** (B-60): penalises captures that leave an own cycling piece about to unblock an opponent mill on its next oscillation

  - **Dead/near-dead placement penalty** (B-64): penalises placing a piece with 0 or 1 free adjacent neighbours (permanently immobile or easily trapped), suppressed when the placement closes a mill

- **Deadline-aware search** — checks the clock every 4 096 nodes; always returns the best partial result on timeout

- **Auto-force-move** — when the AI exceeds its expected thinking time the browser countdown fires `force\_move` automatically; a server-side safety net fires 5 s later if the client message is lost

- **AI resignation** — if the human's position strength exceeds 0.95 (tanh-normalised) for 3 consecutive AI turns, the AI concedes with a farewell message

- **Force Capture** toggle — makes the AI capture aggressively, disabling the fly-sacrifice heuristic

### Opening book

- Curated opening lines with UCB1-scored selection; learns win/loss/draw outcomes per opening

- **Opening Recogniser** — detects rotated and mirrored variants via full D4 dihedral symmetry (4 rotations × 4 reflections)

- Novel openings discovered during play are saved with `needs\_llm\_name=True` and named on the next LLM-enabled run

- Opening Explorer panel (header → **Openings**) — browse and step-replay any named opening line

### Starting play detection

During the placement phase, the Game Info panel shows the **opening family** as it emerges — Outer Square, Diamond, Cardinal Cross, and other named configurations. White and Black families are shown independently when they differ.

### Trajectory-based learning (TrajectoryDB)

- All completed games are indexed by move prefix at server startup

- The AI consults win-rate statistics for the current move sequence when choosing openings, and avoids move sequences associated with losses

- Adaptive-softened games (where the AI was deliberately weakened) are excluded from the index so intentional blunders don't pollute the library

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

- After 3 consecutive losses, difficulty drops by one level and the AI makes additional deliberate mistakes; an amber badge appears near the status bar

- Difficulty is gradually restored as the player wins — one level per 3-win streak

- After 3 consecutive wins at base difficulty, the UI suggests trying a harder level

- Manually changing the difficulty in Settings resets the adaptive state

- Games played under adaptive softening are tagged and excluded from the TrajectoryDB and EndgameDB so the library is never trained on intentional blunders

### Tournament Mode

Play a 6-game gauntlet against all AI personalities in order from weakest to strongest. After completing 3 qualifying games in a session, the **Tournament** header button unlocks.

| Round | Personality | Difficulty | AI Elo |
| - | - | - | - |
| 1 | Chaos — The Trickster | 2 | 720 |
| 2 | Aggressive — The Crusher | 3 | 850 |
| 3 | Scholar — The Bookworm | 3 | 900 |
| 4 | Balanced | 4 | 960 |
| 5 | Defensive — The Blocker | 4 | 1020 |
| 6 | Positional — The Strategist | 5 | 1080 |


- Colours alternate (White / Black / White / …) for fairness across rounds

- Score: **2 pts** for a win, **1 pt** for a draw, **0 pts** for a loss (max 12)

- Player Elo is updated after each game using K=32; displayed live in the Tournament panel

- Final rank: Apprentice / Beginner / Intermediate / Advanced / Master (based on total points)

- The next round starts automatically after each game completes; the scoreboard updates in real time

- Tournament AI uses server-authoritative personality weights — slider settings do not affect tournament games

### AI Tuning and Personalities

- **13 configurable weight sliders** accessible via the **AI Tuning** header button (panel stays open during play); settings persist across sessions via **Save settings**:

| Group | Slider | Default | What it rewards |
| - | - | - | - |
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
| - | - |
| **Random** | A different personality is chosen randomly at the start of each game |
| **Balanced** | All defaults |
| **Aggressive — The Crusher** | Hunts mills and cycling setups relentlessly, ignores wrapping defence, clusters pieces |
| **Defensive — The Blocker** | Smothers every opponent threat, builds resilient diamond structures, wraps opponent mills |
| **Positional — The Strategist** | Spreads across cross nodes, builds long-term cycling structures, thinks ahead |
| **Scholar — The Bookworm** | Methodical opening placement, balanced diamond and wrapping awareness, solid all-round |
| **Chaos — The Trickster** | Scatters randomly, ignores strategy, 45 % blunder rate |


### Personality Profiles

Each preset ships with a distinct weight configuration saved to `data/personalities/`. The table below shows how each personality differs from the defaults:

| Personality | Style | Notable weights | Blunder rate |
| - | - | - | - |
| **Balanced** | All-round; follows the opening book closely | All weights near default | 0 % |
| **Aggressive — The Crusher** | Mill hunter; closes mills at any cost and seizes cardinal squares | `close\_mill` 900, `cardinal\_block` 500, `mill\_count\_scale` 180 %; low `mill\_wrapping` (50) | 0 % |
| **Defensive — The Blocker** | Prioritises neutralising every opponent threat over building its own | `block\_opponent\_mill` 850, `stop\_opponent\_mills` 825, `mill\_wrapping` 450, `blocked\_scale` 355 % | 0 % |
| **Positional — The Strategist** | Spreads across cross nodes, plans long cycling chains, anticipates forks | `cardinal\_block` 475, `defer\_for\_chain` 475, `redirected\_pin` 230, `mobility\_scale` 250 % | 0 % |
| **Scholar — The Bookworm** | Methodical opening adherence; balanced diamond and wrapping awareness | `close\_mill` 775, `opening\_adherence` 100, `long\_term\_position` 175 % | 0 % |
| **Chaos — The Trickster** | Scatters randomly, ignores cardinal squares entirely, exploits losing lines opportunistically | `scatter\_placement` 325, `cardinal\_block` 0, `make\_mistakes` 30 % | 30 % |


Key differences to look for in play:

- **Aggressive** closes mills early and often and clusters around the centre — effective against passive opponents but weak to wrapping.

- **Defensive** rarely opens a cycling mill itself; instead it dismantles yours before it forms, making it frustrating to attack.

- **Positional** is the strongest long-term planner: it defers immediate mill closures when a deeper chain is available and redirects pins to double-block the opponent.

- **Scholar** follows the opening book most faithfully (`opening\_adherence` 100) and transitions smoothly into the midgame.

- **Chaos** plays almost randomly with a 30 % deliberate blunder rate — good for experimenting and debugging; starts every game without any opening book guidance (`opening\_adherence` 0).

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


## Controls

### Header bar

| Control | Action |
| - | - |
| **Moves** | Toggle the move list in the side panel |
| **Openings** | Toggle the Opening Explorer (browse and replay named openings) |
| **Setup** | Toggle the Position Setup editor (place pieces on the board manually) |
| **🏆 Tournament** | Toggle the Tournament panel (unlocks after 3 qualifying games) |
| **► New Game** | Start a new game with current settings |
| **Personality dropdown** | Switch AI personality; changes take effect from the next game |
| **Settings** | Toggle the New Game / settings panel |
| **AI Tuning** | Toggle the weight sliders panel (stays open during play) |


### Bottom bar

| Button | When visible | Action |
| - | - | - |
| **Force Move** (gold pulse) | While AI is thinking | Interrupt AI search immediately; AI plays the best move found so far |
| **Bad Move** | After AI plays a move | Flag the AI's last move as bad, undo it, and have the AI try a different move; the ban is **position-specific** — if any piece moves or is captured afterward, the same move becomes legal again from the new board state |
| **Force Capture** | Always | Toggle: forces the AI to capture aggressively, disabling fly-sacrifice strategy |
| **Offer Draw** | After 40 post-placement half-moves | Offer a draw; AI may accept or decline |
| **Hint (3)** | Human's turn | Request a hint; MillsAI explains the suggested move (3 hints per game) |
| **← Undo** | After a move is made | Rewind the last human move and the AI's response |


### Settings panel (right side)

| Control | Description |
| - | - |
| **Play as** | White, Black, or Random |
| **Opponent** | AI or Human (local pass-and-play) |
| **AI Personality** | Per-game personality override; "Use current AI Tuning sliders" applies your custom weights |
| **AI Difficulty** | 1 (Beginner) – 10 (Maximum 45 s) |
| **MillsAI commentary** | Enable/disable LLM commentary for this game |
| **New Game** | Start the game |
| **Setup Position…** | Open the Position Setup editor |


### Position Setup editor

Click **Setup** in the header bar (or **Setup Position…** in the Settings panel) to open the position editor. The board becomes interactive in edit mode:

- Click any node to cycle it through **empty → White → Black → empty**.

- Use the phase selector to specify whether the game is in the placement, movement, or fly phase.

- Use the turn selector to set whose move it is.

- Click **Start from Here** to begin a game from the custom position.

The position editor is useful for practising specific endgames, reproducing puzzle positions, or testing how the AI handles unusual configurations.

### Replay controls (post-game)

After a game ends the Replay panel activates. Use ⏮ / ◀ / ▶ / ⏭ to step through all moves, or **↩ Back to Live** to return to the live board view.

### Opening Explorer

Select any named opening from the dropdown to see its win/loss/draw record, then click **Replay Opening** to watch it played out at configurable speed. Choose **Practice — I play on** to continue from the end of the opening, or **Watch — AI continues** to observe both sides.

### Named Openings

When the AI plays a placement sequence it hasn't seen before, the opening is saved automatically with `needs\_llm\_name=True`. Names are assigned in two ways:

- **During self-play** — pass `--name-openings` to `self\_play.py` and MillsAI names each novel opening at the end of the run.

- **On demand** — run `python tools/name\_openings.py` to batch-name all un-named openings in one pass (requires Ollama).

After a game where a novel opening was played, MillsAI proposes a name for it. You can confirm or edit that name directly in the GUI's naming prompt before it is written to `data/openings/learned\_openings.json`.

The opening book uses **UCB1 selection** to balance exploration and exploitation when choosing an opening at game start:

```
score = win\_rate + C × √(ln(total\_plays) / (plays\_this\_opening + 1))   C = 0.25
```

Openings are filtered to the AI's side: White-winning lines are only offered when the AI plays White, and vice versa.

**Browsing openings in the GUI**

Click **Openings** in the header to open the Opening Explorer panel. Each named opening is listed with its name, move count, and win/loss/draw record. Click any entry to step-replay that opening on the board. The **Replay Opening** button plays the moves at configurable speed; **Practice — I play on** lets you continue from the end of the opening as the human; **Watch — AI continues** starts an AI-vs-AI game from the final position of that opening line.

**Importing openings from a strategy book**

```
\# Validate and import opening lines from a JSON-formatted book file  
python tools/import\_openings.py --input raw\_openings.json --validate \\  
    --output data/openings/book\_openings.json  
  
\# Dry-run (shows what would be imported, no changes written)  
python tools/import\_openings.py --input raw\_openings.json --dry-run  
  
\# Merge new lines into an existing book file  
python tools/import\_openings.py --input raw\_openings.json --merge \\  
    --output data/openings/book\_openings.json
```

**Other useful commands:**

```
\# List all openings with win/loss/draw stats, sorted by win rate  
python tools/list\_openings.py  
  
\# Import curated game records from the strategy book (seeds win/loss stats)  
python tools/import\_book\_games.py  
  
\# Name all un-named openings via LLM  
python tools/name\_openings.py
```


## Training Tools

The `tools/` directory contains scripts for building and improving the AI's knowledge bases. They work independently of the web server and can be run while the server is stopped.

| Tool | Purpose |
| - | - |
| `self\_play.py` | Run AI-vs-AI full games to populate the trajectory DB, endgame DB, and opening book win rates |
| `endgame\_play.py` | Generate endgame-only positions and play them out — much faster than full games for building the EndgameDB |
| `evolve\_weights.py` | Era-aware (1+1)-ES to hill-climb the global `HeuristicWeights`; saves result to `data/weights/best.json` |
| `evolve\_weights\_v2.py` | Per-personality era-aware (1+1)-ES; evolves each personality’s weight overrides and saves back to `data/personalities/*.json` |
| `build\_fullgame\_db.py` | Frequency-seeded BFS builder: scans human JSONL games, expands around common positions, writes a sorted binary `.bin` file with win/loss/draw outcomes |
| `fullgame\_db.py` | Read-only query interface for the full-game position DB (used by `GameAI` at move-selection time) |
| `build\_endgame\_db.py` | Offline retrograde solver: builds an exact WDL table for all 3v3 fly-phase positions using D4 symmetry (~8× speedup); writes `data/endgame/endgame_3_3.wdl` |
| `train\_value\_net.py` | Train a small MLP value estimator from saved game records |
| `import\_openings.py` | Validate and import curated opening lines from a JSON book file |
| `import\_book\_games.py` | Seed opening win/loss statistics from annotated book game records |
| `name\_openings.py` | Batch-name all un-named openings via the local Ollama LLM |
| `list\_openings.py` | Print the opening book sorted by win rate |
| `purge\_ai\_learning.py` | Remove AI self-play data while preserving book-imported content |


**Recommended workflow for a fresh install:**

```
\# 1. Import book openings to seed the opening book  
python tools/import\_book\_games.py  
  
\# 2. Run self-play to build trajectory and endgame databases  
python tools/self\_play.py --games 100 --no-llm --parallel 4  
  
\# 3. (Optional) Evolve weights to find a stronger evaluation function  
python tools/evolve\_weights.py --generations 20 --parallel 4  
  
\# 4. (Optional) Train a value network once you have 200+ games  
python tools/train\_value\_net.py
```


## Self-Play Training

```
python tools/self\_play.py --no-llm --games 100 --white 6 --black 6 --swap --parallel 4
```

Self-play games are saved to `data/games/` and are read by the AI before each future web game, enriching its opening book win rates and LLM context.

| Flag | Description |
| - | - |
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

```
\# Fast parallel run at equal strength  
python tools/self\_play.py --games 40 --no-llm --parallel 4  
  
\# Mixed personalities to reduce draws  
python tools/self\_play.py --games 20 --no-llm --personalities aggressive,defensive,positional  
  
\# One game with verbose board output and LLM commentary  
python tools/self\_play.py --games 1 --white 5 --black 1 -v --white-personality scholar
```

### Endgame Self-Play

```
python tools/endgame\_play.py --positions 200 --parallel 4
```

Full-game self-play produces only a handful of endgame positions per game. This tool generates (or extracts) endgame starting positions directly and plays them out, building up `EndgameDB` much faster. Each completed game is saved to `data/games/` in the standard JSONL format and indexed by the server on the next restart.

| Flag | Description |
| - | - |
| `--positions N` | Number of endgame positions to play (default: 100) |
| `--difficulty D` | AI difficulty for both sides (default: 5) |
| `--parallel N` | Run N games simultaneously |
| `--min-pieces` / `--max-pieces` | Total piece count range (default: 6–11) |
| `--personalities LIST` | Comma-separated personality pool (default: all except Chaos) |
| `--seed-from-games` | Seed from real positions extracted from `data/games/` rather than random generation |


**Examples:**

```
\# 500 random endgame positions, 4 parallel workers, difficulty 5  
python tools/endgame\_play.py --positions 500 --parallel 4 --difficulty 5  
  
\# Replay real endgame positions from existing game records  
python tools/endgame\_play.py --seed-from-games --positions 300 --parallel 4  
  
\# Narrow to 6–8 piece positions with mixed personalities  
python tools/endgame\_play.py --positions 100 --min-pieces 6 --max-pieces 8 --personalities balanced,positional,defensive
```

### Purging AI-Generated Learning Data

If the AI accumulates bad self-play data that degrades its play, you can revert to only the clean, book-imported data while keeping the capability for future AI learning:

```
\# Preview what would be removed (no changes made)  
python tools/purge\_ai\_learning.py --dry-run  
  
\# Run the purge (prompts for confirmation, backs up everything first)  
python tools/purge\_ai\_learning.py  
  
\# Skip the confirmation prompt  
python tools/purge\_ai\_learning.py --yes
```

**What is removed:**

- Openings with `seed\_source='learned'` and no `source\_reference` (AI self-generated openings)

- All self-play JSONL game files (`human\_color == 'self\_play'`)

**What is kept:**

- Openings imported from the strategy book (`seed\_source='book'`)

- Openings imported from book games (`seed\_source='learned'` with a `source\_reference`)

- Human vs AI game records

- `bad\_moves.json`, ChromaDB vector memory, player profiles, settings

A full backup is written to `data/backups/\<timestamp\>/` before any changes. The TrajectoryDB and EndgameDB rebuild automatically from the remaining human game records on the next server start.


### Weight Evolution

```
python tools/evolve\_weights.py --generations 20 --parallel 4
```

Runs a (1+1) evolution strategy: each generation mutates the current best heuristic weights by Gaussian noise, plays the candidate against the baseline, and promotes the candidate if its win rate reaches ≥ 55 %. Best weights are saved to `data/weights/best.json` and loaded automatically on the next server restart.

| Flag | Description |
| - | - |
| `--generations N` | Number of evolution generations (default: 20) |
| `--games-per-gen G` | Games played per generation to measure win rate |
| `--difficulty D` | AI difficulty used for both sides |
| `--parallel N` | Run N games simultaneously per generation |
| `--from-best` | Seed the starting weights from `data/weights/best.json` |


**Examples:**

```
\# Quick 20-generation run with 4 parallel games  
python tools/evolve\_weights.py --generations 20 --parallel 4  
  
\# Longer run continuing from the current best weights  
python tools/evolve\_weights.py --generations 50 --from-best --parallel 4
```

### Per-Personality Weight Evolution

```
python tools/evolve\_weights\_v2.py --generations 30 --parallel 4
```

Evolves each personality's weight overrides independently using the same era-aware (1+1)-ES algorithm. Unlike `evolve_weights.py`, this script targets the per-personality style files rather than the global baseline.

**Architecture**: personalities are thin overrides on top of `best.json`. Only the fields already present in each personality file are mutated — the rest inherit from `best.json` as they would at runtime. `make_mistakes` and `opening_adherence` are never mutated so each personality retains its character.

Results are saved to `data/personalities/{name}.json` on every promotion. Logs and checkpoints go to `data/weights/personalities/`.

| Flag | Description |
| - | - |
| `--personalities LIST` | Comma-separated personalities to train (default: all except `custom`) |
| `--skip LIST` | Personalities to skip (default: `custom`) |
| `--generations N` | Generations per personality (default: 30) |
| `--games-per-gen G` | Games per evaluation, rounded to even (default: 20) |
| `--difficulty D` | Search difficulty for both sides (default: 5) |
| `--parallel N` | Parallel game workers (default: 4) |
| `--era-size N` | Generations per era for sigma adaptation (default: 5) |
| `--bias-strength F` | Fraction of era directional bias added to mutations (default: 0.3) |

**Examples:**

```
\# Train all personalities, 30 gens each  
python tools/evolve\_weights\_v2.py --generations 30 --parallel 4  
  
\# Train specific personalities only  
python tools/evolve\_weights\_v2.py --personalities aggressive,defensive --generations 50  
  
\# Ben's full run  
python tools/evolve\_weights\_v2.py --generations 100 --parallel 8 --games-per-gen 32 \  
    --difficulty 7 --era-size 10 --bias-strength 0.3 --era-top-k 3
```

Restart the web server after the run to pick up updated personality files.

### Full-Game Position Database

```
python tools/build\_fullgame\_db.py
```

Scans human-played JSONL game records, BFS-expands around frequently-visited positions, back-propagates win/loss/draw outcomes, and writes a sorted binary `.bin` file. Uses D4 board symmetry so each equivalence class is stored once. Output is consulted at move-selection time via `ai/fullgame_db.py` using O(log N) binary search.

Positions are stored in a temporary SQLite file during the build so that large
expansions never require holding all data in RAM simultaneously. Only a small
key-set (~9 bytes per position) is kept in memory. The temp DB is deleted
automatically once the binary output is written.

| Flag | Default | Description |
| - | - | - |
| `--expand-from-games DIR` | `data/games` | Directory of human JSONL game records |
| `--output PATH` | `data/fullgame.bin` | Output binary file |
| `--temp-db PATH` | `<output>.tmp.db` | Path for the temporary SQLite build DB — point at a large/fast drive for big builds |
| `--max-db-gb GB` | `10.0` | Stop BFS and write partial results if the temp DB grows beyond this. Raise to e.g. `100` or `1000` for very large builds on a big drive |
| `--min-seed-frequency N` | `2` | Only positions seen ≥ N times in human games seed the BFS |
| `--expand-depth D` | `4` | BFS depth for late-game / end-of-placement seeds |
| `--early-expand-depth D` | `2×expand-depth` | BFS depth for early-game seeds (tapers linearly to `--expand-depth`) |
| `--max-expand-positions N` | unlimited | Hard cap on total positions expanded |
| `--max-gb GB` | `6.0` | Secondary RAM guard: abort BFS if process RSS exceeds this |
| `--passes N` | `6` | Backpropagation passes for win/loss labelling |
| `--dry-run` | — | Build from a tiny synthetic game set; no disk write |

```
\# Default build (scan data/games, write data/fullgame.bin):  
python tools/build\_fullgame\_db.py  
  
\# Smaller DB — higher seed threshold, shallower expansion:  
python tools/build\_fullgame\_db.py --min-seed-frequency 5 --expand-depth 2  
  
\# Very large build on a 2 TB drive, temp DB allowed up to 500 GB:  
python tools/build\_fullgame\_db.py \\  
    --temp-db /mnt/bigdrive/nmm_build.tmp.db \\  
    --max-db-gb 500 --expand-depth 10  
  
\# Dry run to verify the pipeline:  
python tools/build\_fullgame\_db.py --dry-run
```

### Retrograde Endgame Database

```
python tools/build\_endgame\_db.py --nW 3 --nB 3
```

Retrograde solver that produces an exact WDL (Win/Draw/Loss from side-to-move) table for all fly-phase positions with a given piece count. Uses D4 board symmetry (~8× speedup) and writes directly to a memory-mapped binary file — large tables never require full RAM allocation.

Output is written to `data/endgame/endgame_<nW>_<nB>.wdl` and consulted by `GameAI` at search time.

| Flag | Default | Description |
| - | - | - |
| `--nW N --nB N` | — | Build a single table for this piece count |
| `--build-all` | — | Build all tables in dependency order |
| `--max-sum N` | `11` | Largest `nW+nB` to build when using `--build-all` |
| `--skip-existing` | — | Skip tables whose `.wdl` file already exists |
| `--out-dir PATH` | `data/endgame` | Output directory |
| `--quiet` | — | Suppress per-pass logging |

**Table sizes by `--max-sum`:**

| `--max-sum` | Tables | Largest single table | Total disk |
| - | - | - | - |
| 6 | 3v3 only | 1.3 MB | ~1.3 MB |
| 7 | + 4v3, 3v4 | ~5 MB each | ~11 MB |
| 8 | + 5v3, 4v4, 3v5 | ~20–70 MB | ~200 MB |
| 9 | + 6v3, 5v4, 4v5, 3v6 | ~82–330 MB | ~1.5 GB |

```
\# Build just the 3v3 base table (~30 min):  
python tools/build\_endgame\_db.py --nW 3 --nB 3  
  
\# Build all tables up to 7-piece total, skipping existing:  
python tools/build\_endgame\_db.py --build-all --max-sum 7 --skip-existing  
  
\# Build to a custom location:  
python tools/build\_endgame\_db.py --build-all --max-sum 8 --out-dir /mnt/fast/endgame
```

### Value Network Training

```
python tools/train\_value\_net.py
```

Trains a small MLP (79 inputs → 128 → 64 → 1 output) on completed game records. Every board position in every saved game is labelled with the final outcome (win/loss/draw from that colour's perspective) and used as a training sample.

- **Input**: 24 board positions × 3 channels (own/opponent/empty) + 7 scalar metadata = 79 features, encoded from the current player's perspective so the same weights handle both colours.

- **Output**: `tanh` scalar in (−1, 1) — positive means the current player is likely to win.

- **Training**: mini-batch SGD with MSE loss; runs entirely on CPU in pure numpy (no framework needed).

- **Output file**: `data/value\_net.npz`, loaded automatically by MCTS at server start.

- When a value net is present, it **replaces** the heuristic evaluator at MCTS leaf nodes — faster and stronger than the hand-tuned formula after sufficient training data.

- Inference is ~0.1 ms per position; no GPU required.

| Flag | Description |
| - | - |
| `--games-dir PATH` | Source directory for JSONL game files (default: `data/games`) |
| `--output PATH` | Where to write the trained weights (default: `data/value\_net.npz`) |
| `--epochs N` | Training epochs (default: 50) |
| `--batch-size N` | Mini-batch size (default: 256) |


Recommended: accumulate at least 200 self-play games before training for useful signal.


## Learned (Neural) AI

In addition to the classical minimax engine, the repo ships an **opt-in** neural
AI under `learned_ai/` — a PyTorch policy/value network trained by self-play
reinforcement learning. It plugs into the game through the same
`choose_move(board)` contract as the heuristic engine, so it is a drop-in
replacement selectable by an environment variable. The default run path is
unchanged and needs no PyTorch.

```
# Default: heuristic engine (unchanged behaviour)
python main.py

# Use the trained neural engine instead
NMM_AI_ENGINE=learned python main.py
```

If `NMM_AI_ENGINE=learned` but the checkpoint is missing or PyTorch is not
installed, the game prints a warning and falls back to the heuristic engine so
play is never blocked.

### Step 1 — Install learning dependencies

These are only needed for training or running the neural engine. The base game
does not require them.

**Linux / macOS** (run inside the activated venv):

```bash
source .venv/bin/activate          # activate venv created by install.sh
pip install -r requirements_learned_ai.txt
```

**Windows** (run inside the activated venv):

```bat
.venv\Scripts\activate             :: activate venv created by install.bat
pip install -r requirements_learned_ai.txt
```

CPU-only PyTorch (smaller download, sufficient for smoke tests and light training):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements_learned_ai.txt
```

`requirements_learned_ai.txt` adds: `torch>=2.0`, `numpy`, `pyyaml`,
`jsonlines`, `tqdm`.

### Step 2 — Smoke test

Verifies encoders, model routing, self-play loop, and checkpoint round-trips.
All 37 tests should pass in under a minute.

```bash
python scripts/smoke_test.py
```

Then run a tiny end-to-end training pass (no useful model produced — just proves
the pipeline runs without crashing):

```bash
python scripts/train.py --config learned_ai/config/smoke_test_config.yaml
```

### Step 3 — Train stage by stage

The curriculum advances through four stages: random → heuristic → self-play →
(optional) human data. You can start from a specific stage or let it auto-advance:

```bash
# Full run from stage 1 (slow — may take hours/days depending on hardware)
python scripts/train.py --config learned_ai/config/default_config.yaml

# Jump directly to a stage
python scripts/train.py --config learned_ai/config/default_config.yaml --stage 2
python scripts/train.py --config learned_ai/config/default_config.yaml --stage 3
python scripts/train.py --config learned_ai/config/default_config.yaml --stage 4
```

| Stage | Opponent | Target |
| - | - | - |
| 1 | self (sanity) | completes without crashes |
| 2 | random | win rate climbs toward **70%+** |
| 3 | heuristic | steady improvement; heavy early losses are normal |
| 4 | self-play | strength improves open-endedly |

### Step 4 — Resume from a checkpoint

Checkpoints embed their architecture, so you do not need to re-specify hidden
sizes:

```bash
# Resume from the latest checkpoint
python scripts/train.py --resume learned_ai/checkpoints/latest.pt

# Resume from a specific checkpoint
python scripts/train.py \
  --config learned_ai/config/default_config.yaml \
  --resume learned_ai/checkpoints/ckpt-010000.pt
```

### Step 5 — Benchmark vs heuristic

```bash
python scripts/benchmark_vs_heuristic.py \
  --checkpoint learned_ai/checkpoints/latest.pt --games 100
```

Arbitrary head-to-head (e.g. learned vs random):

```bash
python scripts/evaluate.py --agent1 learned --agent2 random \
  --games 100 --agent1-checkpoint learned_ai/checkpoints/latest.pt
```

### Step 6 — Play against the trained AI

```bash
# Play as Black against the learned engine
python scripts/human_vs_learned.py \
  --checkpoint learned_ai/checkpoints/latest.pt --side black

# Play as White
python scripts/human_vs_learned.py \
  --checkpoint learned_ai/checkpoints/latest.pt --side white
```

### Monitor training

Metrics are JSON-Lines, one object per policy update:

```bash
tail -f learned_ai/logs/metrics.jsonl
```

Each line includes episode count, stage, win/loss/draw totals, temperature,
`policy_loss`, `value_loss`, `entropy`, and `mean_reward`. Illegal move attempts
should always be 0 (action masking guarantees this).

### Full documentation

- [`docs/LEARNED_AI_ARCHITECTURE.md`](docs/LEARNED_AI_ARCHITECTURE.md) — state/action
  encoding, the shared-backbone + 5-phase-head network, training algorithm.
- [`docs/TRAINING_GUIDE.md`](docs/TRAINING_GUIDE.md) — detailed training reference with
  expected win rates per stage, hyperparameter guide, and troubleshooting.
- [`docs/MIGRATION_GUIDE.md`](docs/MIGRATION_GUIDE.md) — switching engines, A/B
  testing, and instant rollback to the heuristic engine.
- [`docs/AI_INTERFACE_MAPPING.md`](docs/AI_INTERFACE_MAPPING.md) — the exact engine
  interface the learned AI implements.


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


## Configuration

Edit `data/settings.json` to change the Ollama model, URL, and LLM behaviour thresholds.

| Key | Default | Description |
| - | - | - |
| `ollama\_model` | `llama3.1:8b` | Ollama model to use |
| `ollama\_url` | `http://localhost:11434` | Ollama server address |
| `poor\_move\_threshold` | `0.3` | Score drop that triggers an LLM comment on a human move |
| `max\_poor\_move\_comments\_per\_game` | `5` | Cap on poor-move LLM comments per game |
| `endgame\_active\_threshold` | `11` | Total pieces on board to enter endgame mode |
| `endgame\_deep\_threshold` | `8` | Total pieces to enter deep-endgame mode |
| `endgame\_solved\_dir` | `data/endgame` | Directory containing the retrograde WDL file (`endgame_3_3.wdl`); set to empty string to disable |
| `fullgame\_db\_path` | *(unset)* | Path to the built SQLite fullgame position DB; leave unset to skip |


### Changing the LLM model

```
ollama pull mistral        \# or any other Ollama model
```

Then update `data/settings.json`:

```
\{ "ollama\_model": "mistral" \}
```

The game uses the new model from the next game start.


## AI Slider Weights Reference

The **AI Tuning** panel exposes the most user-visible weights. All fields correspond directly to `HeuristicWeights` in `ai/heuristics.py`. The complete dataclass has ~40 fields; the table below covers the ~13 shown in the UI plus the most impactful positional scalers.

### Mill control

| Field | Default | What it controls |
| - | - | - |
| `close\_mill` | 500 | Delta bonus per mill the AI closes this move |
| `cycling\_mill` | 300 | Bonus for building a cycling-mill setup (two 2-configs whose closing squares are adjacent; capped at 1 per move) |
| `block\_opponent\_mill` | 400 | Bonus per opponent 2-config that is closeable next turn and gets neutralised |
| `stop\_opponent\_mills` | 450 | Bonus per opponent two-piece setup (any 2-config) dismantled this move |
| `mill\_wrapping` | 150 | Bonus per own piece that surrounds an opponent closed mill, cutting off the pivot's useful slides |


### Mobility and space

| Field | Default | What it controls |
| - | - | - |
| `cardinal\_block` | 200 | Bonus for occupying or evicting pieces from cross-node (midpoint) squares that have 3 neighbours |
| `feeder\_diamond` | 200 | Bonus for creating a fork structure: four pieces adjacent to one key square, threatening two mills at once |
| `scatter\_placement` | 75 | Bonus for placing away from own existing pieces in the first 6 placements (prevents clustering) |
| `mobility\_scale` | 100 | % multiplier on the mobility component of the static evaluator (how much having more legal moves than the opponent is worth) |
| `blocked\_scale` | 100 | % multiplier on the blocked-pieces component (bonus for leaving opponent pieces with no legal moves) |


### Placement and structure

| Field | Default | What it controls |
| - | - | - |
| `setup\_mill` | 100 | Bonus per new two-config gained in a single placement move |
| `mill\_count\_scale` | 100 | % multiplier on the mill-count component of the static evaluator |
| `long\_term\_position` | 100 | Overall % multiplier on the entire positional base score (all non-tactical terms) |


### Endgame and behaviour

| Field | Default | What it controls |
| - | - | - |
| `mill\_opening` | 200 | Bonus for opening a cycling-ready mill (sliding out of a closed mill to enable the next capture cycle) |
| `make\_mistakes` | 0 | Probability (%) that the AI plays a deliberately bad move on any given turn |


> **Tip:** Increasing `cycling\_mill` and `mill\_wrapping` together produces a slow, suffocating style. Maximising `close\_mill` and `cardinal\_block` produces an aggressive attacking style. Setting `make\_mistakes` to 10–20 % creates a forgiving training partner.


## Project Structure

```
NMM\_ollama/  
├── game/                        \# Core engine: board, rules, game engine  
├── ai/  
│   ├── game\_ai.py               \# Negamax + alpha-beta, blunder mode, weights  
│   ├── heuristics.py            \# Phase-aware evaluation + HeuristicWeights dataclass  
│   ├── mills\_llm.py             \# Ollama LLM interface  
│   ├── coordinator.py           \# AI deliberation, commentary, resignation tracking  
│   ├── opening\_book.py          \# Opening library + UCB1 selection  
│   ├── opening\_recognizer.py    \# D4 symmetry-aware opening recognition  
│   ├── endgame\_recognizer.py    \# Phase detection, zugzwang, mill-cycle patterns  
│   ├── endgame\_db.py            \# Endgame position database (learned from games)  
│   ├── trajectory\_db.py         \# Move-prefix win-rate index (learned from games)  
│   ├── fullgame\_db.py           \# Read-only query interface for the full-game position DB  
│   ├── endgame\_solved\_db.py     \# Exact WDL table for 3v3 fly-phase positions (retrograde, ~1.3 MB)  
│   ├── starting\_play.py         \# Opening family detection (Outer Square, Diamond, etc.)  
│   ├── memory\_manager.py        \# Game record persistence and pattern analysis  
│   └── debriefer.py             \# Post-game session summary  
├── web/  
│   ├── app.py                   \# FastAPI + WebSocket server, session management, adaptive difficulty, tournament  
│   ├── static/  
│   │   ├── game.js              \# Game controller, personality presets, weight sliders, replay, tournament  
│   │   ├── board.js             \# SVG board renderer  
│   │   └── style.css            \# Dark wood theme  
│   └── templates/index.html  
├── tools/  
│   ├── self\_play.py             \# AI vs AI training loop (full games)  
│   ├── endgame\_play.py          \# Endgame self-play for rapid EndgameDB enrichment  
│   ├── evolve\_weights.py        \# Era-aware (1+1)-ES to tune global heuristic weights  
│   ├── evolve\_weights\_v2.py     \# Per-personality era-aware (1+1)-ES  
│   ├── build\_fullgame\_db.py     \# Build bounded SQLite position DB with win/loss outcomes  
│   ├── build\_endgame\_db.py      \# Retrograde solver: exact WDL for all 3v3 fly-phase positions (D4 sym, ~8× faster)  
│   ├── fullgame\_db.py           \# Query interface for the full-game position DB  
│   ├── import\_openings.py       \# Import openings from strategy book text file  
│   ├── import\_book\_games.py     \# Import games into opening book  
│   ├── name\_openings.py         \# LLM-name novel openings  
│   ├── list\_openings.py         \# Print opening book summary  
│   ├── teach\_opening.py         \# Interactively teach a new opening line  
│   ├── debrief.py               \# Run post-session debrief manually  
│   └── purge\_ai\_learning.py     \# Remove AI self-play data; revert to book-only openings  
├── data/  
│   ├── settings.json            \# Runtime configuration  
│   ├── openings/                \# Opening book JSON (openings.json, book\_openings.json)  
│   ├── personalities/           \# Saved per-personality weight files  
│   ├── games/                   \# Game records (JSONL, one file per game)  
│   ├── weights/                 \# Evolved heuristic weights (best.json + checkpoints)  
│   ├── chroma/                  \# ChromaDB vector store (LLM memory)  
│   └── session\_memory/          \# LLM session narrative files  
├── tests/                       \# unittest test suite (160+ tests)  
├── main.py                      \# CLI harness for quick engine testing  
├── install.sh                   \# One-time installer (Linux/macOS)  
├── run\_nmm.sh                   \# Launch script (Linux/macOS)  
├── install.bat                  \# One-time installer (Windows — calls install.ps1)  
├── install.ps1                  \# PowerShell installer with optional Ollama choice  
├── run\_nmm.bat                  \# Launch script (Windows)  
└── requirements.txt
```


## Running Tests

```
source .venv/bin/activate  
python -m unittest discover tests/ -v
```


## License

MIT

