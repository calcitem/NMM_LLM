# Human Game DB Integration Plan

**Goal:** Import 3,536+ PlayOK human vs human games, use them to improve TrajectoryDB
guidance, retrain the sentinel, retrain the value network, and add a new lightweight
behaviour-cloning policy net that encourages human-like play as a fourth AI overlay.

---

## Background and design rationale

The current AI stack trains on `data/games/` which is predominantly AI self-play generated
by the heuristic engine. That data has systematic bias: the AI plays in tight heuristic
patterns, rarely explores strategically creative lines, and never makes the kinds of
positional sacrifices that human experts make. Sentinel and the value network learned
those same biases from their training data.

3,536 human vs human PlayOK games (GameType `70,0`, standard Nine Men's Morris,
5-minute time control, ELO range ~1100–1400) are available in
`~/playok_archive/games/` organised as `year/month/player/mlXXXXXXXX.txt`.

**Why four separate uses of the human data:**

| System | What it currently learns | What human data adds |
|---|---|---|
| TrajectoryDB | Which moves won in AI games | Which moves humans win with — opens up non-heuristic lines |
| Sentinel | Move quality from AI patterns + Malom labels | Move quality from human strategic intent + Malom labels |
| Value network | Position value from AI game outcomes | Position value from human game outcomes — teaches human strategic evaluation |
| Human policy net | (does not exist) | Behaviour-cloned next-move distribution — explicit human-style prior |

**Will this beat the heuristic AI?** The value network + sentinel retrained on human
data will likely make the AI more strategically coherent and harder to exploit positionally,
while keeping the heuristic's tactical sharpness via the negamax search. The human policy
net is not a replacement for the heuristic; it is an overlay that can push move selection
toward human-like patterns at the cost of some tactical precision. At high value_net_blend
settings + sentinel active, the combined system may outperform the pure heuristic on
strategic positions; it will not outperform it at forced-sequence tactics.

---

## Phase 0 — Verify PlayOK position numbering

**Why first:** The PlayOK 1–24 position numbering must be mapped to the project's
named positions (a7, d7, …) before any parsing can be correct. The standard NMM
board layout is well-known but must be confirmed against actual game data by checking
that all `from-to` moves in sample games are legal on the reconstructed board.

**Standard mapping to verify** (most likely layout, clockwise from top-left per ring):

```
Outer ring:   1=a7  2=d7  3=g7  4=g4  5=g1  6=d1  7=a1  8=a4
Middle ring:  9=b6 10=d6 11=f6 12=f4 13=f2 14=d2 15=b2 16=b4
Inner ring:  17=c5 18=d5 19=e5 20=e4 21=e3 22=d3 23=c3 24=c4
```

**Verification method:**

1. Write a small script that replays 20 sample games using the candidate mapping.
2. For each move, apply it to a `BoardState` and check `is_legal()`.
3. If any game has an illegal move, the mapping is wrong — adjust and retry.
4. A correct mapping will produce zero illegal moves across the sample.

This is a one-time step. Once the mapping is confirmed it is hard-coded into the
importer and does not change.

---

## Phase 1 — PlayOK import pipeline

**New tool:** `tools/import_playok.py`

**Input:** `~/playok_archive/games/` directory tree  
**Output:** JSONL files in `data/human_games/` (one file per PlayOK game, named
`human_<game_id>.jsonl` e.g. `human_ml11756018.jsonl`)

### 1.1 Parser

Read each `.txt` file and extract:

- Game ID from filename (`ml11756018`)
- Headers: White player, Black player, Date, Result, WhiteElo, BlackElo
- Move list: split placement phase (bare numbers) from movement phase (`from-to` / `from-toxcap`)

**Placement phase** (turns 1–9, both players place each turn):
- Input: `1. 16 17 2. 20 19 ...`
- White plays first number, Black plays second
- Output: `{"type":"place","from":null,"to":"b4","capture":null}`

**Movement phase** (turn 10+):
- Input: `10. 15-3 23-24` or `11. 14-6x11 15-14`
- `15-3` → `{"type":"move","from":"b2","to":"c5"}`
- `14-6x11` → `{"type":"move","from":"d2","to":"d1","capture":"f6"}`

**Draws / resignations:** Result `1/2-1/2` → `winner: null`. Result `1-0` / `0-1` → `winner: "W"` / `"B"`.

### 1.2 FEN reconstruction

For each move, apply the move to a running `BoardState` starting from the empty
board. Record `board_fen_before` on every move using `board.to_fen_string()`.
This is required for sentinel training (features are built from FEN).

### 1.3 Output JSONL schema

Match the existing project JSONL format as closely as possible so all existing
tooling (TrajectoryDB, sentinel dataset, value net trainer) can read human games
without modification:

```json
{
  "session_id": "ml11756018",
  "source": "playok",
  "date": "2026-01-04",
  "white_player": "nadannmalsehn",
  "black_player": "binger",
  "white_elo": 1133,
  "black_elo": 1384,
  "human_color": null,
  "winner": "B",
  "draw_reason": null,
  "moves": [
    {
      "turn": 1, "color": "W", "type": "place",
      "from": null, "to": "b4", "capture": null,
      "notation": "b4",
      "board_fen_before": "........................|W|0|0"
    },
    ...
  ]
}
```

The `source: "playok"` field distinguishes human games from AI self-play in any
future filtering step.

### 1.4 Deduplication

Maintain a manifest file at `data/human_games/imported.json` — a dict mapping
game ID → import timestamp. Before importing any file, check the manifest. Skip
already-imported IDs. This supports incremental updates.

### 1.5 Incremental update

Running `import_playok.py` a second time on the same or expanded archive will:
1. Walk all `.txt` files in the archive tree
2. Skip any whose game ID appears in `imported.json`
3. Import and append only new games
4. Update the manifest

**CLI:**
```
python tools/import_playok.py \
    --archive ~/playok_archive/games \
    --output data/human_games \
    [--dry-run]          # report counts without writing
    [--validate-only]    # check legality of all moves, report errors
    [--limit N]          # import at most N new games (testing)
```

### 1.6 Tools page

Add an "Import PlayOK Games" section to `web/templates/tools.html` with:
- Archive path input (pre-filled from settings.json `playok_archive_path`)
- Dry-run checkbox
- Run button wired to `ws/tools`
- Status card showing: games in human_games/, earliest/latest date, player count

---

## Phase 2 — TrajectoryDB multi-source support

**Current state:** `TrajectoryDB.__init__` takes a single `games_dir` path and
uses `rglob("*.jsonl")` on it.

**Change:** Accept an optional second directory `human_games_dir`. Index both
directories. Apply a configurable reward multiplier to human game entries (default
`1.5×`) so the trajectory guidance favours human-game lines when both AI and human
games share the same canonical prefix.

**Why a multiplier rather than human-only:** The AI self-play games contain many
more positions (thousands of games vs 3,536 human games). Without upweighting,
human lines would be drowned out in common positions. The multiplier gives human
patterns priority without discarding the AI game coverage for rare positions.

**app.py change:** Pass `_human_games_dir` alongside `_trajectory_db_dir` at startup.
Read `human_games_dir` from `settings.json` (default `data/human_games`).

**No change to the query interface** — existing callers are unaffected.

---

## Phase 3 — Sentinel retraining on human games

**Current state:** Sentinel trains via `scripts/train_sentinel.py --game-dir data/games`.
It reads JSONL files, replays positions, generates one training example per legal move,
and labels each with Malom DB WDL when available (else heuristic fallback).

**Change:** No code changes required to the sentinel itself. Human games are in the
same JSONL format. Simply pass `--game-dir data/human_games` (or a combined dir).

**Recommended training strategy:**

1. **Human-only first pass:** Train fresh from `data/human_games/` with Malom labels.
   This removes all AI self-play bias. Expected: 3,536 games × ~25 positions × ~12
   candidates ≈ ~1M training examples. More than enough for the model architecture
   (input_dim=58, hidden=[128,64,32]).

2. **Mixed fine-tune (optional):** After convergence on human games, fine-tune for
   a few epochs on a small AI self-play sample to recover rare endgame position coverage.

3. **Assessment:** Compare the retrained sentinel's per-move WDL prediction accuracy
   against the Malom DB on a held-out set of human games. The current checkpoint was
   trained on AI data; human data should significantly improve the calibration.

**Add to Tools page:** "Train Sentinel on Human Games" button using existing
`train_sentinel.py` with `--game-dir data/human_games` pre-filled.

---

## Phase 4 — Value network retraining on human games

**Current state:** `tools/train_value_net.py --games-dir data/games` trains a 79-input
numpy MLP to predict game outcome (W/D/L) from board features. Trained on AI self-play,
it reflects heuristic-AI strategic values.

**Change:** Point `--games-dir` at `data/human_games`. No other code changes needed —
the training script already accepts the argument and reads JSONL.

**Why this matters most:** The value network feeds directly into the negamax leaf
evaluator when `value_net_blend > 0`. A value network that learned from human games
will steer the search tree toward positions humans consider strategically strong, not
just positions the heuristic weights score highly. This is the most impactful single
change for making the AI "play more like a human".

**Recommended training strategy:**

1. Retrain from scratch on human games only (outcome label: +1 W win, 0 draw, -1 W loss
   from White's perspective, as currently implemented).
2. After retraining, run a gauntlet evaluation (existing `evolve_weights_v2.py`) to
   confirm the human-trained value network + heuristic blend beats the current baseline.
3. A recommended starting `value_net_blend` weight of 30–50 is suggested in the Tools
   page hint text.

**Add to Tools page:** "Train Value Net on Human Games" button with `--games-dir`
pre-filled to `data/human_games`.

---

## Phase 5 — Human policy network (behaviour cloning)

This is the new learned AI component. It is a fourth overlay alongside sentinel,
trajectory DB, and the value network.

### 5.1 What it does

Given a board position and whose turn it is, output a probability distribution over
all legal moves. Trained by supervised imitation of human move choices (behaviour
cloning). At inference time, the move with the highest policy probability can replace
or blend with the heuristic's top candidate.

This is explicitly **not** a replacement for the heuristic engine. It is an overlay
that can encourage human-like move selection when activated.

### 5.2 Architecture

- **Input:** Same 79-dim board feature vector as the value network (`board_to_features`)
  plus a move-specific encoding (24-dim from-position one-hot + 24-dim to-position
  one-hot + 1 capture flag = 49 dims) → total 128-dim concatenated input
- **Output:** Single scalar logit per (position, move) pair; softmax across all legal
  moves gives the policy distribution
- **Hidden layers:** [256, 128] with ReLU, dropout 0.2 — small enough to run at
  inference speed without slowing move selection

This is effectively the same architecture as the sentinel but with a different
training objective: cross-entropy against the human's actual chosen move rather
than WDL quality regression.

### 5.3 Training

- **Labels:** For each position in each human game, the human's actual move is the
  positive example. All other legal moves are negatives. Loss is cross-entropy.
- **Quality filtering:** Optionally exclude positions where the human's move was
  a clear blunder according to Malom DB (WDL flips from W→L or D→L). This prevents
  the policy from learning human mistakes as well as human strengths.
- **Data:** ~3,536 games × ~50 positions per game (placement + movement) ≈ ~175k
  training positions
- **Tool:** `scripts/train_human_policy.py` (new script)
- **Checkpoint:** `learned_ai/human_policy/checkpoints/best.pt`

### 5.4 Integration into GameAI

New `GameAI` attribute: `self.human_policy = None`  
New method: `_apply_human_policy_guidance(board, move, moves)` — analogous to
`_apply_sentinel_intervention`.

When `human_policy` is set and `use_human_policy=True`:
- Score all legal moves with the policy net
- Apply a configurable top-K filter: if the heuristic's top move is outside the
  policy's top-K human moves, replace it with the policy's top-1
- K is difficulty-scaled: K=1 at difficulty 10 (always pick most human-like),
  K=5 at difficulty 5 (human guidance only when heuristic agrees), disabled at diff ≤ 3

### 5.5 GUI controls

- "Human Policy" chip in the Scores bar overlay row (alongside Sentinel, DB Lines)
  — shows the policy's top-move probabilities as overlay arrows on the board
- Settings checkbox: "Use human-style play" — when checked, attaches the policy to
  the AI at game start (analogous to existing sentinel checkbox)
- Settings panel note: "Makes the AI play more human-like patterns; may reduce
  tactical precision at lower difficulty"

---

## Phase 6 — Assessment framework

Before and after each training step, measure:

1. **Sentinel calibration:** On held-out human games with Malom labels, compute
   move quality prediction accuracy (rank correlation between sentinel score and
   Malom WDL outcome). Compare human-trained vs AI-trained checkpoint.

2. **Value network win prediction:** On held-out human games, measure how well the
   retrained value net predicts the actual game winner from the board position at
   move 10, 20, 30. Compare to AI-trained baseline.

3. **Human policy top-1 accuracy:** On held-out human games, what fraction of the
   time does the policy's top-1 prediction match the human's actual move? This is the
   canonical behaviour-cloning metric. Expect 25–40% (above random ~8% for typical
   branching factor).

4. **Head-to-head gauntlet:** Human-trained AI (value net blend + human policy) vs
   current heuristic baseline, 200 games at difficulty 5. Win rate target: ≥ 50%
   indicates the human training does not hurt and may help.

---

## Phase 7 — Tools page integration

Add a new "Human Games" section to `tools.html` above the existing Database Status:

- **Import PlayOK Games** — archive path, dry-run, Run button
- **Human DB Status card** — game count, date range, player count, duplicate skips
- **Train Sentinel on Human Games** — shortcut button, pre-fills `--game-dir data/human_games`
- **Train Value Net on Human Games** — shortcut button, pre-fills `--games-dir data/human_games`
- **Train Human Policy** — epochs, batch size, quality-filter checkbox, Run button
- **Human Policy Status card** — loaded/not loaded, checkpoint date, top-1 accuracy if known

Settings: add `playok_archive_path` to `settings.json` and the DB Settings section
in the Tools page.

---

## Implementation order

| Step | Deliverable | Depends on |
|---|---|---|
| 0 | Position mapping verified | Nothing |
| 1 | `tools/import_playok.py` working, games in `data/human_games/` | Step 0 |
| 2 | TrajectoryDB reads both directories | Step 1 |
| 3 | Sentinel retrained on human games | Step 1 |
| 4 | Value network retrained on human games | Step 1 |
| 5 | Human policy net trained | Step 1, Step 4 (shared feature code) |
| 6 | Assessment runs | Steps 3–5 |
| 7 | Tools page + GUI controls | Steps 1–5 |

Steps 2–5 are independent of each other once Step 1 is complete and can proceed
in parallel.

---

## Files to create / modify

| File | Action |
|---|---|
| `tools/import_playok.py` | Create — PlayOK parser and JSONL writer |
| `data/human_games/` | Create directory, `.gitignore` the JSONL files |
| `data/human_games/imported.json` | Create — deduplication manifest |
| `ai/trajectory_db.py` | Modify — accept optional second games dir with multiplier |
| `web/app.py` | Modify — pass human_games_dir to TrajectoryDB; add playok_archive_path to settings |
| `scripts/train_human_policy.py` | Create — behaviour cloning training script |
| `learned_ai/human_policy/` | Create — model, config, checkpoint dir |
| `ai/game_ai.py` | Modify — add human_policy attribute and _apply_human_policy_guidance |
| `web/static/game.js` | Modify — Human Policy chip + use_human_policy setting |
| `web/templates/index.html` | Modify — Human Policy overlay chip and settings checkbox |
| `web/templates/tools.html` | Modify — Human Games section |
| `web/static/tools.js` | Modify — import/train handlers for human games |
| `data/settings.json` | Modify — add playok_archive_path |

No changes to `scripts/train_sentinel.py`, `tools/train_value_net.py`, or the
sentinel model architecture — they already support `--game-dir` and accept the
human-game JSONL format as-is.
