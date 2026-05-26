# Nine Men's Morris — Active Backlog

*New items go here. When an item is completed, move it to `plan\\\_done.md`.*

---

## Implementation Roadmap

Active implementation order agreed 2026-05-25. Two parallel tracks:

### Track 1 — Heuristic / Phase-Control ✅ COMPLETE (2026-05-25)

| # | Item | Key outcome | Status |
|---|------|-------------|--------|
| 1 | **B-46** | Add `placement_index` parameter — prerequisite for B-28/B-47 | ✅ |
| 2 | **B-28** | Late-placement capitalisation scaling (validates B-34/B-35 game examples) | ✅ |
| 3 | **B-29** | Busy-chain priority: suppress `close_mill` bonus when level-4 chain confirmed | ✅ |
| 4 | **B-37** | Opponent chain detection (mirror `_placement_chain_scan` for opponent) | ✅ |
| 5 | **B-36** | Unguarded cardinal mill alert | ✅ |
| 6 | **B-22** | Regression test + emergency block fix (investigate FEN at move 32 first) | ✅ |
| 7 | **B-47** | White / Black asymmetric placement weights | ✅ |
| 8 | **B-33** | Dead-block quality / forcing response value | ✅ |
| 9 | **B-30** | Dual-mill oscillation preservation | ✅ |

### Track 2 — DB / Infrastructure (independent; can run in parallel)

| # | Item | Key outcome | Status |
|---|------|-------------|--------|
| A | **B-26** (DB) | Wire FullGameDB into server startup | ✅ 2026-05-26 |
| B | **B-27** (DB) | Binary format for fullgame DB | ✅ 2026-05-26 |
| C | **B-23** (DB) | Syzygy-style endgame DB builder | ✅ 2026-05-26 |

### Track 3 — Search Stack (independent; implement after Track 1 is stable)

~~SE-1~~ ✅ → ~~SE-2~~ ✅ → ~~SE-3~~ ✅ (transposition table, killer heuristic, history heuristic)

### Phase strategy guide

`data/phase_strategy.md` — compact phased playing guide for LLM context injection.
Feed the relevant phase section(s) to MillsLLM at each game stage.

---

## New Bug & Enhancement Items

### Bug B-26 — FullGameDB is never loaded by the server ⬜

**Symptom:** Even if `data/fullgame.sqlite` exists (built via `tools/build_fullgame_db.py`), the web server ignores it entirely. The game plays as if the DB were absent regardless of its size or content.

**Root cause:** `web/app.py` never constructs a `FullGameDB` instance. `GameAI` supports an optional `fullgame_db=` parameter and will query it when provided, but nothing in the server startup passes one in. Compare with `TrajectoryDB` and `EndgameDB`, which are loaded at startup and threaded through to every `GameAI` instance.

Additionally, if the DB was built to a non-default location (e.g. another drive via `--db-dir`), there is no setting in `data/settings.json` to tell the server where to find it.

**Fix:**

1. **`web/app.py`** — load `FullGameDB` at startup alongside the other databases:
   ```python
   _fullgame_db_path = _ROOT / "data" / "fullgame.sqlite"
   # override from settings if present
   _fullgame_db = FullGameDB(_fullgame_db_path) if _fullgame_db_path.exists() else None
   ```

2. **`data/settings.json`** — add optional `fullgame_db_path` key. If set, the server uses that path instead of the default. Example:
   ```json
   { "fullgame_db_path": "/mnt/windows/NMM_DB/fullgame.sqlite" }
   ```

3. **`web/app.py`** — pass `_fullgame_db` through to every `GameAI` constructor call (same pattern as `trajectory_db` and `endgame_db` are passed today).

4. **`web/app.py`** — add `fullgame_db_path` to the `/api/settings` GET response so the UI can display the configured path and whether the DB loaded successfully.

**Files:**

- `web/app.py` — startup loading, settings read, GameAI construction sites
- `data/settings.json` — add `fullgame_db_path` (optional, defaults to `data/fullgame.sqlite`)
- `README.md` — add `fullgame_db_path` to the Configuration table

### Enhancement B-23 — Endgame position database builder (direct-index format) ⬜

**Goal:** A new script `tools/build_endgame_db.py` that builds a complete, solved endgame database for movement and fly phase positions (≤ 12 pieces on the board) using a **Syzygy-style direct-index array** — not SQLite. A new read interface `ai/endgame_solved_db.py` queries it.

**Rationale:** The full-game DB (`build_fullgame_db.py`) must be bounded because the full game tree is ~10¹⁰ positions. The endgame space is small enough to solve *completely* for the most useful piece counts, and a direct-index format is dramatically better than SQLite for this case:

| Total pieces | Positions (with D4 reduction) | WDL file size | Feasibility |
|---|---|---|---|
| ≤ 6 | ~330 K | ~83 KB | trivially complete |
| ≤ 8 | ~13 M | ~3 MB | completely solvable in minutes |
| ≤ 10 | ~124 M | ~31 MB | solvable in hours |
| ≤ 12 | ~575 M | ~144 MB | feasible, near-complete |

Only movement and fly phase positions are stored (no placement phase).

**Storage format — Syzygy-style direct-index array:**

Rather than a key-value store, every possible piece configuration is assigned a sequential integer ID derived from combinatorial indexing:

```
position_id = combinatorial_index(white_squares, black_squares, turn)
```

Outcomes are stored as two compact flat arrays:
- **WDL file** (`endgame_wdl.bin`): 2 bits per position — Win/Draw/Loss for the side to move. ≤8 pieces = ~3 MB; fits entirely in RAM.
- **DTZ file** (`endgame_dtz.bin`): 1 byte per position — depth to terminal (distance-to-zero). Same size as WDL × 4.

Query is O(1): compute the ID (arithmetic only, no search), read one array element. No B-tree, no binary search, no I/O once loaded into RAM.

This is the same principle used by Syzygy and Nalimov chess tablebases.

**Key differences from `build_fullgame_db.py`:**

- Constrained to `pieces_on_board["W"] + pieces_on_board["B"] <= max_pieces` (default 10; flag `--max-pieces N`)
- Retrograde analysis from terminal positions outward — produces complete, exact, optimal solutions
- Output: two compact binary files (`endgame_wdl.bin`, `endgame_dtz.bin`) — not SQLite
- New `ai/endgame_solved_db.py` query module; `GameAI` consults it before the fullgame DB and before negamax when piece count is in range

**Output location — any drive supported:**

- Default: `<project>/data/endgame_wdl.bin` + `<project>/data/endgame_dtz.bin` (absolute, not relative to cwd)
- `--db-dir <path>`: write both files into any directory, including another drive
- `--output-dir <path>`: alias for `--db-dir`
- Pre-flight write check before build starts
- Resolved paths printed at startup

```bash
# Default location
python tools/build_endgame_db.py --max-pieces 10

# Another drive
python tools/build_endgame_db.py --db-dir /mnt/external --max-pieces 10
python tools/build_endgame_db.py --db-dir D:/databases  --max-pieces 10
```

**Flags:**

| Flag | Default | Description |
|---|---|---|
| `--db-dir PATH` | `<project>/data/` | Directory to write `endgame_wdl.bin` and `endgame_dtz.bin` into |
| `--max-pieces N` | 10 | Only solve positions with ≤ N total pieces |
| `--dry-run` | — | Validate pipeline without writing |
| `--wdl-only` | — | Build WDL table only, skip DTZ (faster, smaller) |

**Server wiring (same pattern as B-26 for fullgame DB):**

1. **`web/app.py`** — load at startup from configurable path:
   ```python
   _eg_dir = Path(settings.get("endgame_solved_dir") or (_ROOT / "data"))
   _endgame_solved_db = EndgameSolvedDB(_eg_dir)
   ```

2. **`data/settings.json`** — add optional `endgame_solved_dir` key for non-default locations:
   ```json
   { "endgame_solved_dir": "/mnt/external" }
   ```

3. **`web/app.py`** — pass to `GameAI`; endgame DB consulted before negamax and before the fullgame DB when piece count is within its solved range.

4. **`README.md`** — add `endgame_solved_dir` to the Configuration table.

**Files:**

- `tools/build_endgame_db.py` (new)
- `ai/endgame_solved_db.py` (new — O(1) query interface for the binary format)
- `ai/game_ai.py` (consult endgame DB when piece count ≤ threshold)
- `web/app.py` (startup loading + pass to GameAI — same pattern as B-26)
- `data/settings.json` (add `endgame_solved_dir`)
- `README.md` (Configuration table)

---

### Enhancement B-27 — Replace fullgame DB SQLite format with memory-mapped sorted binary ⬜

**Goal:** Replace the SQLite storage in `build_fullgame_db.py` and `ai/fullgame_db.py` with a memory-mapped sorted binary file. This gives 2–5× faster queries and ~40% smaller files, which matters once the DB grows beyond available RAM.

**Why SQLite degrades at scale:**

SQLite uses a B-tree index on the 9-byte key. For a DB that fits in RAM, it is fine. Once the file exceeds available RAM (easily reached at 500M+ positions, which is 10–30 GB), every query causes random page faults — typically 3–4 B-tree node reads, each potentially a disk seek. At game speed (multiple queries per move decision) this becomes noticeable.

**New format — memory-mapped sorted binary:**

All records are written sorted by key and stored as fixed-size structs:

```
Record (32 bytes):
  [9 bytes]  canonical position key
  [1 byte]   outcome  (0=unknown, 1=W win, 2=draw, 3=B win)
  [2 bytes]  depth to terminal (0–65535)
  [4 bytes]  best move (from/to/capture packed as 3× 5-bit position indices + flags)
  [16 bytes] top-4 child moves with outcome flags (4 bytes each)
```

Query = binary search on the key field: ~23 comparisons for 10M records, ~27 for 100M. The file is memory-mapped so the OS pages in only the needed portions; sequential access during build is highly cache-friendly.

**Build pipeline change:**

1. Enumerate positions as now, writing records to a temp file (unsorted)
2. External sort the temp file by key (can be done in chunks, resumable)
3. Write final sorted binary — this is the queryable DB
4. Optional: write a sparse index (~1 entry per 1000 records, ~few MB) that loads into RAM to narrow binary searches to a 1000-record window in one step

**Migration:**

- `tools/build_fullgame_db.py` gains `--format [sqlite|binary]` flag; default remains `sqlite` until the binary format is validated, then switches
- `ai/fullgame_db.py` gains a format-detection branch: opens binary if `.bin` extension or magic bytes match, otherwise falls back to SQLite
- Existing SQLite DBs remain readable; no forced rebuild

**Files:**

- `tools/build_fullgame_db.py` — add binary output path + external sort step
- `ai/fullgame_db.py` — add binary query path (mmap + binary search)

---

### Enhancement B-24 — GUI settings for position DB usage ⬜

**Goal:** Add controls to the Settings and AI Tuning panels so the player can see which position databases are active and tune how strongly they influence the AI's play.

**Rationale:** The fullgame DB and endgame DB change AI behaviour in ways that aren't currently visible or configurable in the GUI. A player should be able to turn DB lookup on/off and understand what is affecting the AI's moves.

**Proposed controls (Settings panel or AI Tuning panel):**

| Control | Type | Description |
|---|---|---|
| Use FullGame DB | Checkbox | Enable/disable `data/fullgame.sqlite` lookup (greyed out if file absent) |
| Use Endgame DB | Checkbox | Enable/disable `data/endgame_solved.sqlite` lookup (greyed out if absent) |
| DB influence | Slider 0–100 % | How much a DB result overrides the heuristic score (0 = heuristic only, 100 = DB always wins) |
| DB status line | Read-only | Shows e.g. "FullGame: 500K positions · Endgame: 13M positions (complete ≤8)" or "No DBs found" |

**Behaviour:**

- If both DBs are enabled and a position exists in both, the endgame DB takes priority (it is exact)
- DB influence slider feeds into `ai/fullgame_db.py`'s `score_delta()` blend factor
- Checkbox state is persisted to `data/settings.json` alongside other AI settings
- DB file presence is checked at server start; the UI greys out absent DBs automatically

**Files:**

- `web/templates/index.html` — new controls in Settings or AI Tuning panel
- `web/static/game.js` — load/save DB toggle state; send with game start message
- `web/static/style.css` — DB status line styling
- `web/app.py` — expose `/api/db_status` endpoint; pass DB toggle flags to `GameAI`
- `ai/game_ai.py` / `ai/fullgame_db.py` — honour the toggle and blend factor at runtime

---

### Enhancement B-25 — Tools management page ⬜

**Goal:** A new web page (`/tools`) that lets the user inspect the state of all AI knowledge bases and trigger training tools from the browser, without needing a terminal.

**Access:** A **Tools** button added to the NMM game page header bar opens `/tools` in a new browser tab.

**Page sections:**

**1. Database Status**

A summary card per database showing:
- File size and last-modified date
- Position count (queried from the DB)
- Coverage note (e.g. "complete ≤ 8 pieces", "bounded at 500K", "not built")
- **Rebuild** button (runs the builder script as a background subprocess; streams output to the page via WebSocket or SSE)

**2. Heuristic Weight Evolution**

- Current `best.json` — lists all weight values vs defaults, highlights changed fields
- Per-personality summary — shows which personalities have been evolved and when
- **Run evolve_weights.py** and **Run evolve_weights_v2.py** buttons with configurable `--generations` and `--parallel` inputs
- Live log output streamed to the page while running

**3. Self-Play & Training**

- Game count and date range from `data/games/`
- Trajectory DB size and top-N most-played opening prefixes
- Endgame DB position count
- **Run self-play** button (`--games`, `--parallel`, `--difficulty` inputs)
- **Train value net** button (`--epochs` input); shows current `data/value_net.npz` metadata if present

**4. Opening Book**

- Total named openings, un-named openings, total play count
- **Name openings** button (runs `tools/name_openings.py`; requires Ollama)
- **Purge AI learning** button (with confirmation dialog; runs `tools/purge_ai_learning.py`)

**Implementation notes:**

- FastAPI route `GET /tools` serves a new `tools.html` template (standalone page, not the game SPA)
- A `/ws/tools` WebSocket (or `/tools/stream` SSE endpoint) streams subprocess stdout line-by-line to the browser
- Only one tool subprocess runs at a time; the UI disables all run buttons while a task is active
- All destructive actions (purge, rebuild DB) show a confirmation dialog before running
- The page is read-only safe to leave open — status cards auto-refresh every 30 s

**Files:**

- `web/templates/tools.html` (new)
- `web/static/tools.js` (new)
- `web/static/tools.css` (new, or extend `style.css`)
- `web/app.py` — `/tools` route, `/api/tool_status`, `/ws/tools` WebSocket
- `web/templates/index.html` — add **Tools** button to header bar

---

### Bug B-21 — Windows installer: improve model pull failure guidance ⬜

**Symptom:** After Ollama installs and the service starts, the installer attempts to pull the default model (`llama3.1:8b`, ~5 GB). If the pull fails or the user cancels, the only feedback is a terse warning: *"Model pull failed. Run manually later: ollama pull \<model\>"*. The user is not told about smaller alternatives, how to change the model, or what config file to update.

**Root cause:** `install.ps1` step 8 (`ollama pull`) has no post-failure guidance block. The completion banner also doesn't remind the user to pull the model if Ollama was installed but the pull was skipped.

**Fix — `install.ps1`:**

1. After a failed `ollama pull`, print a help block:

```
[NMM] Model pull failed or was skipped.
[NMM] To pull the default model later:
[NMM]   ollama pull llama3.1:8b          (~5 GB — best quality)
[NMM]
[NMM] Lighter alternatives that work well:
[NMM]   ollama pull llama3.2:3b          (~2 GB)
[NMM]   ollama pull phi3:mini            (~2.3 GB)
[NMM]   ollama pull gemma2:2b            (~1.6 GB)
[NMM]
[NMM] Then update data\settings.json:
[NMM]   "ollama_model": "llama3.2:3b"    (or whichever model you pulled)
[NMM]
[NMM] The game works without a model -- LLM commentary will be disabled.
```

2. In the "Installation complete!" banner, if Ollama was installed but the model was not pulled, repeat the short version of the above (pull command + settings file reminder) so the user knows what to do next.

**Files:**

- `install.ps1` — step 8 failure block + completion banner
- `install.bat` — if it surfaces any model-related messages, mirror the same guidance

### Bug B-17 — GUI text contrast too dim ⬜

**Symptom:** Many GUI labels, board coordinates, and control text are hard to read. The stylesheet applies `--text-dim: \\\#8a7a60` widely across buttons, labels, move numbers, eval-graph labels, slider labels, replay controls, and form rows.

**Root cause:** `web/static/style.css` uses `var(--text-dim)` for most functional labels, not just decorative chrome. The palette prioritises atmosphere over readability.

**Fix:**

- `web/static/style.css` — raise `--text-dim` to approximately `\\\#b7a78c`, or split it into:

  - `--text-muted` for secondary/decorative chrome only,

  - `--text-label` for all functional gameplay labels.

- Increase board coordinate / grid label contrast specifically.

- Audit all `var(--text-dim)` uses and promote critical gameplay labels (move numbers, eval labels, phase indicators) to `var(--text)` or the new `--text-label`.

- Verify tournament draw rows, AI tuning labels, and replay text remain readable.

**Files:**

- `web/static/style.css`

- `web/static/board.js` if board coordinate text is rendered separately

### Bug B-18 — Clarify and document Bad Move button policy ⬜

**Symptom:** Get rid of the bad move button and the illegal moves it stored. Replace with force blunder button that makes the ai take a different move.

### Enhancement B-20 — Reward long-game trajectory lines in opening + midgame ⬜

**Goal:** Give extra weight to moves from previously played games that lasted at least ~25 moves. These games often contain sound opening and midgame decisions even if they eventually lost in the endgame.

**Rationale:** The current trajectory system assigns win/loss deltas per move-prefix but does not distinguish "game lost tactically in move 10" from "game lost in a close endgame at move 40". A game that reached move 40 before losing contains valuable opening and midgame lines that should still be exploitable.

**Recommended change:**

- In `TrajectoryDB`, track per stored line:

  - total game length (half-moves),

  - deepest phase reached (placement / move / fly),

  - whether the loss, if any, occurred only in the endgame (fly phase or late move phase).

- Add a `survival\\\_value` weighting path:

  - boosts moves appearing in games that survived beyond a configurable threshold (default ~25 moves),

  - stronger effect in placement + move phase,

  - zero or negligible effect in fly phase (endgame quality is independent).

- Blend with existing win/loss delta, not replacing it.

**Files:**

- `ai/trajectory\\\_db.py`

- `ai/coordinator.py`

- `AI\\\_INTERNALS.md` (update trajectory section)

### Tactic B-22 — Emergency one-move mill denial must outrank speculative improvement ⬜

**Symptom:** In the move phase the AI sometimes ignores a direct block of an opponent one-move mill closure and plays a weaker positional improvement instead.

**Example game (failure at move 32):**

```
1.f4 b4       2.d2 d6       3.d5 e4       4.d7 d3    
5.e5 c5       6.f2 b2       7.f6xb4 b4    8.g7 a7    
9.c4 a1      10.d2-d1 b4-a4xc4  11.f4-g4 e4-f4    
12.d1-g1xf4 a4-b4   13.g4-f4xa7 d6-b6xd7    
14.f6-d6 d3-c3   15.g7-d7xc5 a1-a4   16.d7-a7 c3-c4xa7    
17.g1-g4 c4-c5   18.e5-e4xa4 b4-a4   19.g4-g7 a4-b4xd6    
20.g7-d7 b6-d6   21.d7-a7 d6-b6xf4   22.d5-d6 b4-a4    
23.e4-e3 a4-b4xd6   24.e3-c4 c5-d5   25.c4-g7 d5-d6    
26.a7-f6 b4-c4   27.g7-b4 d6-d5   28.f6-d7 b6-d6    
29.f2-a7 c4-c5   30.a7-e5 c5-c4   31.b4-a7 c4-b4    
32.e5-e4  ← AI should have moved a piece to b6 to block Black's mill; instead plays e5-e4, losing the game
```

**Expected at move 32:** White (AI) should move a piece to `b6` to prevent Black from closing a mill. White has pieces at `g7` and `d6`; Black is at `a7`, meaning there is a mill threat. The AI plays `e5-e4` instead, which is a speculative positional move.

**Additional note from human:** The position includes White at `g7`, White at `d6`, and Black at `a7`. White can play `d5`, forcing Black to respond at `d7`. `d7` has no use except blocking a mill at `d5-d6-d7`, but White's piece remains mobile. Forcing the opponent onto a passive blocking square like this should be rewarded. Additionally, the AI should *anticipate* this forcing opportunity and place at `d5` before Black can, so that it forces the response rather than reacting to it.

**Investigation required:**

1. Reproduce the exact FEN at move 32 and call `\\\_immediate\\\_mill\\\_threats(board)` directly — confirm whether `b6` is detected.

2. Check adjacency rules: can White's available pieces legally reach `b6` in move phase?

3. If the threat is detected but the blocked-move list excludes the blocking piece, inspect adjacency check in `\\\_immediate\\\_mill\\\_threats`.

4. If detection is correct but heuristic still picks `e5-e4`, the forcing-sequence bonus may be outweighing the block.

**Fix:**

- Confirm `\\\_immediate\\\_mill\\\_threats` detection at this FEN.

- If missed: fix the detection logic.

- Add a `forced\\\_block\\\_priority` flag that gates speculative setup bonuses when an immediate mill threat exists.

- Add regression test for this exact FEN position.

**Files:**

- `ai/game\\\_ai.py` — `\\\_immediate\\\_mill\\\_threats()`

- `ai/heuristics.py` — `tactical\\\_move\\\_bonus()`

- `tests/` — regression test from FEN at move 32

### Tactic B-33 — Reward forcing placements that compel opponent onto low-utility squares ⬜

**Goal:** The AI should prefer placements that force an opponent defensive response onto squares of low strategic value — locking the opponent's piece to a passive role while the AI retains high-value positions.

**Example game (forcing d5 → d7 response):**

```
1.d6 d2    2.f4 b4    3.f6 f2    4.b6xf2 f2    
5.d3 a7    6.e4 b2xe4   7.g7 e3    8.d5 d7
```

At turn 8, White places at `d5`. This forces Black to block at `d7`, which has no independent value except preventing the mill at `d5-d6-d7`. White's `d5` piece remains mobile. `d7` for Black is a passive blocking piece with low reuse value.

**Human note:** The AI should anticipate this kind of forcing move and take `d5` proactively before Black does, so it is White making the forcing placement rather than reacting. When White is the forcer, Black is left with a passive response; when Black takes `d5` first, White is the one forced to react.

**Current related mechanisms:**

- `\\\_placement\\\_chain\\\_scan()` — rewards forcing chains.

- `placement\\\_busy\\\_scan` — base weight per chain level.

- `fork\\\_anticipation` (B-4) — rewards blocking opponent fork-in-2 squares.

**Gap:** The current scan rewards forcing chains from the AI's own perspective but does not evaluate the *quality of the forced response* — whether the block square is low-mobility, low-mill-participation, or a dead-end.

**Fix:**

- Extend `\\\_placement\\\_chain\\\_scan()` to score the utility of the **opponent's forced response square**:

  - opponent forced to a corner node (2 connections): high forcing quality → higher bonus.

  - opponent forced to a cardinal/cross node: low forcing quality → no extra bonus.

- Add a `dead\\\_block\\\_bonus` weight (default ~60) per forcing step where the opponent's forced response lands on a low-mobility square.

- Add a proactive anticipation signal: bonus for placing on a square that denies the opponent a future forcing position before they can use it.

**Files:**

- `ai/heuristics.py` — `\\\_placement\\\_chain\\\_scan()`, `tactical\\\_move\\\_bonus()`

### Bug B-34 — Placement 9 should avoid sterile forks with no nearby feeder support ⬜ *(game test case; implementation covered by B-28)*

**Symptom:** On the last placement, the AI sometimes creates a nominal fork or 2-config that has no nearby feeder pieces and confers no forcing continuation. This reduces the AI's mobility and immediately gives the opponent initiative.

**Example game (bad last placement — White plays g1):**

```
1.d6 d2    2.b4 f4    3.g4 a4    4.f6 d7    
5.e4 c4    6.d3 e5    7.d1 a7    8.b6xa7 b2    
9.g1  ← White places last piece at g1    
10.d6-d5 d7-d6    
11.e4-e3  ← game already losing from turn 9
```

White's last placement at `g1` on turn 9 reduces the mobility of adjacent pieces, creates no immediate threat, and allows Black to form a 2-config with a nearby piece enabling an immediate mill. White should instead have placed at `a1` or `g7` to form a 2-config that forces Black to react. The game is losing from turn 10 because of the bad placement in turn 9.

**Root cause hypothesis:** Bonuses for cross-node occupancy or setup creation may overvalue latent structure on the last placement without requiring nearby feeder pieces or a practical continuation.

**Fix:**

- Add a **late-placement quality gate** for placements 8–9:

  - a newly created 2-config must have at least one friendly feeder piece within 2 adjacency steps, OR

  - the placement must close a mill or block an immediate opponent threat.

- If neither condition holds: apply a `sterile\\\_fork\\\_penalty` (default ~100) on the last placement.

- Scale `setup\\\_mill` bonus down ~40% on placement 9 unless the setup is immediately actionable.

**Files:**

- `ai/heuristics.py` — `tactical\\\_move\\\_bonus()`, late-placement window checks

### Bug B-35 — Final placements: prefer dual-purpose block-and-build over passive 2-config ⬜ *(game test case; implementation covered by B-28)*

**Symptom:** On the last placement the AI creates a 2-piece setup that ignores an opponent mobile mill, when a dual-purpose square would both block and create own pressure.

**Example game (Black's 9th placement should go to a4, not e5):**

```
1.d6 d2    2.f4 b4    3.g7 g4    4.d7 d5    
5.a7xd5 d5   6.f6 f2    7.b6xd5 d5   8.c4 b2xc4    
9.d3 e5  ← Black's last placement — passive 2-config that ignores White's mobile mill    
10.a7-a4 g4-g1
```

At turn 9, Black places at `e5` creating a 2-piece. This ignores White's mill structure on line 7, which is now free to open — no Black piece contests it. Placing at `a4` instead would both block the cardinal mill line `a4-b4-c4` and create a 2-config approach from the inner ring, placing Black on a winning trajectory. The `e5` placement leaves White with initiative.

**Goal:** Reward final placements that simultaneously block opponent activity **and** create own pressure.

**Fix:**

- Add a `dual\\\_purpose\\\_final\\\_bonus` (default ~150) for a placement that:

  1. blocks or contests an opponent active mill line or mobile mill pivot, AND

  2. simultaneously creates a new own 2-config or advances an existing one.

- Weight this bonus higher on placements 8–9.

- Give blocking open cardinal mill lines on last placements higher priority than speculative setup.

**Files:**

- `ai/heuristics.py`

### Bug B-36 — Increase imperative to block unguarded cardinal mill lines during placement ✅ 2026-05-26

**Symptom:** The AI fails to block opponent cardinal mill lines (e.g. `a4-b4-c4`) when they are completely unguarded by any own piece and about to become a permanent structural threat.

**Example game (repeated failure to block a4-b4-c4):**

```
1.f4 b4    2.d2 d6    3.d5 d3    4.d7 c4    
5.g4 a4xg4   6.e5  ← White never blocks the a4-b4-c4 cardinal line
```

From turn 5, White never places at `a4` or any square contesting the `a4-b4-c4` line. Black already has pieces at `b4` and `c4`, and `a4` is the closing square. White has no contesting piece on that line at all. By turn 6 it is impossible to block without sacrificing another threat. White is on a losing trajectory from turn 5 because of this failure.

**Additional example — failure in a different opening:**

```
1.f4 b4    2.d2 d6    3.d5 d3    4.c4 e4    
5.d7 g4    6.b6  ← White fails to contest Black's cardinal line development
```

Black builds a forcing chain from turn 4 onwards partly using cardinal squares, and White does not disrupt it.

**Current logic:**

- `cardinal\\\_block` (weight 200) rewards placing on cross/cardinal nodes generally.

- B-3 ring-cardinal preference rewards blocking ring concentrations.

- `fork\\\_anticipation` (B-4) looks 2 moves ahead for fork squares.

**Gap:** The existing cardinal logic rewards general cross-node control but does not specifically detect when the opponent has already formed a **nearly-complete cardinal mill skeleton with no contesting pieces**. This is treated as a generic mill threat at the normal urgency level, rather than a structural emergency.

**Fix:**

- Add `\\\_unguarded\\\_cardinal\\\_mill\\\_alert(board, opp)`: returns closing squares for opponent cardinal mills where both other squares are occupied by the opponent and no own piece is adjacent.

- In `evaluate()` placement phase: add penalty ~250–350 per detected unguarded cardinal skeleton.

- In `tactical\\\_move\\\_bonus()`: add a large bonus (~300) for placing directly on the closing square of such a mill.

- Give this alert higher priority than speculative setup on any placement turn.

**Files:**

- `ai/heuristics.py`

- `tests/` — test for the `1.f4 b4 2.d2 d6 3.d5 d3 4.d7 c4 5.g4 a4xg4 6.e5` failure

### Enhancement B-37 — Opponent placement\_busy\_scan: detect and disrupt opponent forcing chains ⬜

**Goal:** Mirror `\\\_placement\\\_chain\\\_scan()` to also detect when the **opponent** is building a forcing-chain capability, and disrupt it before it becomes unavoidable.

**Current state:** `\\\_placement\\\_chain\\\_scan()` and `placement\\\_busy\\\_scan` only assess the AI's own forcing potential. No opponent-side mirroring is documented in `AI\\\_INTERNALS.md`.

**Example game — Black acquires forcing-chain capability (first variation):**

```
1.f4 b4    2.d2 d6    3.d5 d3    4.c4 e4    
5.d7 g4    6.b6  ← Black is now on a winning forcing trajectory
```

By turn 6, Black has `b4`, `d6`, `d3`, `e4`, `g4` placed and is ready to begin a level-3 or level-4 forcing chain. Unless White immediately creates a mill by placing at `f2` (closing the `f2-f4-f6` mill and forcing Black to react instead), Black wins. A secondary option is to disrupt one of Black's linked 2-config pairs.

**Example game — Black's forcing chain plays out (second variation):**

```
1.f4 b4    2.d2 d6    3.d5 d3    4.c4 e4    
5.d7 g4    6.b6 a1    7.a4 c3
```

At turn 7, Black places at `c3`, forcing White to block the `c3-d3-e3` mill threat. White must block but Black simultaneously develops another 2-config. The chain continues.

**Full forcing sequence demonstrating loss:**

```
1.f4 b4    2.d2 d6    3.d5 d3    4.c4 e4    
5.d7 g4    6.b6 a1    7.a4 c3    8.d1 g1    9.g7 a7
```

Black creates a dual 2-config setup in turn 8. White cannot block both. Black closes a mill on turn 9 and removes the blocking piece, leaving the mill mobile and White in a losing position.

**Fix:**

- Add `\\\_opp\\\_chain\\\_level = \\\_placement\\\_chain\\\_scan(board, opp)` alongside the existing own-chain scan in `tactical\\\_move\\\_bonus()`.

- Penalise moves that leave `\\\_opp\\\_chain\\\_level \\\>= 2` without reducing it.

- When `\\\_opp\\\_chain\\\_level \\\>= 3`: treat as emergency — reward any move that reduces the opponent chain level or breaks one of their linked 2-config pairs.

- When `\\\_opp\\\_chain\\\_level == 4`: treat as equivalent urgency to a direct mill threat.

- Flag via coordinator commentary when opponent-chain disruption was the AI's primary motive.

**Files:**

- `ai/heuristics.py` — `tactical\\\_move\\\_bonus()`, add opponent chain evaluation

- `ai/coordinator.py` — commentary flag for disruption play

- `AI\\\_INTERNALS.md` — update placement scan section to document opponent mirroring

### Enhancement B-28 — Shift placement priorities: build until 7–8, capitalise by 8–9 ⬜

**Goal:** The AI should spend early placements (1–6) building pressure structure, but in the final 1–2 placements strongly prefer **capitalising** on those structures rather than continuing to create opportunities

1.d6 d2

2.f4 b4

3.c4 e4

4.d3 d5

5.a4 g4

6.d7 b2

7.f2 b6xf4

8.f4 e5

9.f6xe5 e5

**Symptom:** Multiple observed games show the AI continuing to build abstract 2-config setups on the last placement while ignoring direct threats, opponent forcing chains, or obvious conversion opportunities.

**Pattern across supplied game examples:**

- Turn 9 sterile fork instead of actionable 2-config (B-34 game above).

- Turn 9 passive 2-config instead of dual-purpose block (B-35 game above).

- Turn 6 speculative setup instead of forcing play or opponent disruption (B-37 games above).

**Fix — phase-sensitive placement weighting:**

- Add a `placement\\\_index` parameter to `tactical\\\_move\\\_bonus()` (partially available via `pieces\\\_placed`).

- Scale setup-building bonuses by a `late\\\_placement\\\_multiplier`:

  - placements 1–6: `multiplier = 1.0`

  - placement 7: `multiplier = 0.8`

  - placement 8: `multiplier = 0.5`

  - placement 9: `multiplier = 0.25` (setup reward nearly suppressed)

- On the same scale, increase reward for:

  - direct mill closure: `close\\\_mill` bonus ×1.5 on placement 9,

  - blocking opponent mill: `block\\\_opponent\\\_mill` ×1.5 on placements 8–9,

  - dual-purpose block-and-build (B-25 `dual\\\_purpose\\\_final\\\_bonus`),

  - opponent forcing-chain disruption (B-27 `\\\_opp\\\_chain\\\_level` penalty).

**Files:**

- `ai/heuristics.py` — `tactical\\\_move\\\_bonus()`, `HeuristicWeights`

- `AI\\\_INTERNALS.md` — update placement busy-scan and tactical bonus documentation

## Additions for `plan\\\_todo.md`

### Tactic B-29 — `search\\\_ahead\\\_busy` / placement busy-chain must outrank immediate mill closure when the chain is forced ⬜

**Symptom:** In some placement positions, `search\\\_ahead\\\_busy` / `placement\\\_busy\\\_scan` identifies a forcing chain that is effectively winning, but the AI still prefers an immediate mill on another line. In the example below, Black (AI) closes a mill on line 5 instead, and White then reaches two open mills versus Black’s restricted one.

**Primary example (keep move record):**

```
1.d6 d2    
2.f4 b4    
3.d3 d5    
4.c4 e4    
5.a4 g4    
6.d7 b6    
7.b2 d1    
8.f2 e5    
9.e3 c5xd7    
10.a4-a1 d1-g1
```

**Observed issue:** the AI takes the immediate mill instead of following the forcing busy-chain line.

**Requested behaviour:** when `search\\\_ahead\\\_busy` finds a forced chain / winning busy-sequence, it should take priority over a merely good local mill closure.

**Agreed implementation:**

- `ai/heuristics.py` — when `_placement_chain_scan()` returns level 4 (forced chain that closes a mill on the final piece), **suppress the `close_mill_contribution` from `tactical_move_bonus()`** for moves that are NOT the chain continuation. A hard override, not merely a bonus boost: the chain result replaces the local mill score rather than adding to it.

- `ai/heuristics.py` — increase `defer\\\_for\\\_chain` further so that level-4 chains are reliably preferred over level-2/3 chains when no suppress applies.

- Add a `busy_chain_priority` weight (default ~400): the flat bonus added to any move that enters or continues a confirmed level-4 chain. Tunable separately from `defer_for_chain`.

- `ai/heuristics.py` — extend `placement\\\_busy\\\_scan` / `search\\\_ahead\\\_busy` depth to **5 plies ahead** in placement phase when the chain remains forcing.

- Add a separate slider / heuristic weight for this priority, e.g. `busy\\\_chain\\\_priority`, so it can be tuned independently from ordinary defer-for-chain behaviour.

- In move selection, allow the chain result to **outrank immediate close-mill bonus** when the chain leads to dual open mills, convergent 2-configs, or other near-forced win structures.

**Potential new slider:**

- `busy\\\_chain\\\_priority` — extra bonus when a busy-chain line is evaluated as forced / dominant.

**Related note:** this overlaps with the existing need for stronger opponent-side busy-chain detection, but this item is about making the AI trust its own winning chain enough to skip the tempting local mill.

### Tactic B-30 — Preserve and prefer 5-ply busy-chain lines that lead to dual-mill oscillation ⬜

**Human-provided winning continuation (keep move record):**

```
1.d6 d2    
2.f4 b4    
3.c4 e4    
4.d3 d5    
5.a4 g4    
6.d7 e5    
7.e3 c3    
8.c5 b2    
9.b6 f2xf4
```

**Claim:** this line produces a dual-mill structure, with two 2-piece groups sharing a third piece that can oscillate between closures, leading to victory.

**Goal:** the busy-chain search should explicitly recognise and reward this class of outcome.

**Fix:**

- `ai/heuristics.py` — extend the terminal reward logic of `search\\\_ahead\\\_busy` / `\\\_placement\\\_chain\\\_scan` so that it recognises:

  - dual open mills,

  - shared-pivot oscillating mill closures,

  - two 2-configs sharing a pivot piece,

  - forced defender dead-squares.

- Increase `defer\\\_for\\\_chain` enough that these lines beat ordinary local mill closure unless the local mill itself wins immediately.

- Add a regression test from the supplied move sequence.

### Bug B-31 — Opening play should still be recorded when the AI resigns ⬜

**Symptom:** opening play / opening sequence is not being recorded properly when the AI resigns.

**Why it matters:** even when the AI resigns, the opening sequence is still valuable training data. Those games are especially useful because they often show where an opening or early-midgame trajectory failed.

**Requested behaviour:** always store the opening sequence even if the game terminates by AI resignation.

**Fix:**

- `web/app.py` — verify the resignation path persists the game record and opening line before any early return or overlay-only exit.

- `ai/opening\\\_book.py` / training pipeline — ensure resignation games still contribute opening statistics and opening-sequence storage.

- `ai/memory\\\_manager.py` — confirm resignation-tagged games are not filtered out from opening extraction.

- Add a regression test: AI resigns after a legal opening, and that opening sequence is still present in the stored game record / opening data.

**Note:** this is slightly different from the LLM debrief-on-resign bug. That debrief path may already be fixed, but opening persistence still needs separate verification.

### Enhancement B-32 — Increase AI reasoning / commentary transparency for chosen moves ⬜

**Goal:** increase how much the AI reports about *why* it chose a move, including which function, tactical detector, or weighting was the main driver.

**Requested behaviour:** commentary/debug output should say more than just the move choice; it should identify the dominant reason, such as:

- immediate mill closure,

- mandatory block,

- busy-chain / search-ahead win,

- fork prevention,

- convergence disruption,

- cardinal-lane block,

- mobility squeeze,

- trajectory exploit,

- endgame DB recognition,

- opening-book adherence.

**Fix:**

- `ai/game\\\_ai.py` — capture a structured explanation object for the selected move, listing the top scoring features / bonuses / blockers.

- `ai/coordinator.py` — expose those reasons in commentary, debug logs, and optional dev overlays.

- `web/static/game.js` — display a richer “AI thought process” summary when debugging or commentary mode is enabled.

- Include the strongest positive driver and strongest defensive driver, not just a generic explanation.

**Nice-to-have:** add a toggle for “verbose AI reasoning” so the default UI is not overwhelmed.

### Suggested umbrella note

These new reports point to one broader evaluator problem:

**The AI still undervalues long forcing busy-chain wins compared with short-term local gains, and it does not surface that reasoning clearly enough for debugging.**

A coherent implementation order would be:

1. strengthen `search\\\_ahead\\\_busy` / `placement\\\_busy\\\_scan` depth and winning-line recognition,

2. add a dedicated slider or explicit priority term for forced busy-chains,

3. add regression tests from the two supplied move records,

4. ensure resignation games still persist opening lines,

5. improve reasoning transparency so future misses are easier to diagnose.

## Thematic note — placement-phase root causes (updated 2026-05-25)

The B-22 through B-37 cluster around three confirmed core weaknesses:

**Weakness 1 — Late placement overvalues speculative structure.**
The fix is `placement_index` scaling (B-46 / B-28): setup-building bonuses taper from
1.0× at placement 1 to 0.25× at placement 9, while blocking and converting bonuses rise.

**Weakness 2 — Opponent forcing potential is not mirrored.**
`_placement_chain_scan` is one-sided (AI initiative only). B-37 mirrors it for the
opponent so the AI can choose between "build my chain" and "break their chain first."

**Weakness 3 — Tactical priority ladder exists in ordering but not in scoring.**
`_order_moves()` has a clean P0/P1/P2 hierarchy but `tactical_move_bonus()` is
fully additive — speculative bonuses can still outscore an emergency block or a
confirmed winning chain. B-29 fixes the chain case; B-22 investigates the block case.

**Implementation order:** see the Roadmap section at the top of this file.

## Note for Claude — hidden heuristic weights are being evolved but are not visible in the GUI sliders

The current heuristic-weight evolution script is **not limited to the visible slider set**. In `ai/heuristics.py`, `HeuristicWeights` currently defines 36 fields, and `tools/evolve\\\_weights.py` mutates every field except `make\\\_mistakes` and `opening\\\_adherence` via `tunable\\\_fields()` and `\\\_FIXED\\\_FIELDS = \\\{"make\\\_mistakes", "opening\\\_adherence"\\\}`.\[cite:2\]

This means the overnight tuning run is currently adjusting **34 heuristic weights**, while the web GUI appears to expose only about **22 slider-backed weights**.\[cite:2\] In practice, the slider panel is behind the dataclass: the evolution script is tuning more weights than the UI currently shows.\[cite:2\]

### Hidden / non-slider heuristic weights currently being tuned

These `HeuristicWeights` fields appear to be evolved by `tools/evolve\\\_weights.py` but are not currently visible in the GUI slider set:\[cite:2\]

- `capture\\\_disrupt\\\_diamond`

- `capture\\\_disrupt\\\_feeder`

- `convergence\\\_block`

- `convergence\\\_disrupt`

- `convergence\\\_penalty`

- `cross\\\_feed\\\_mobility`

- `herding\\\_squeeze`

- `locked\\\_mill\\\_penalty`

- `mill\\\_trap\\\_build`

- `mobility\\\_reduction`

- `own\\\_convergence`

- `placement\\\_busy\\\_scan`

- `ring\\\_crowding\\\_penalty`

- `sacrifice\\\_viable`

### Why this matters

This creates a **visibility and reproducibility mismatch**:\[cite:2\]

- `best.json` may contain evolved values that cannot be fully reviewed or manually adjusted from the current slider UI.\[cite:2\]

- Claude or a human operator may think a tuning pass only affected slider-visible weights, when in fact several advanced/internal heuristics were also changed.\[cite:2\]

- Manual tuning in the web panel cannot currently achieve parity with the full evolved heuristic set.\[cite:2\]

### Recommendation

Please review slider/UI parity with `HeuristicWeights` and decide one of the following:\[cite:2\]

1. **Expose all tunable heuristic weights** in the GUI, possibly with an “advanced” section for the less user-facing ones.\[cite:2\]

2. **Document the hidden/internal weights explicitly** in `AI\\\_INTERNALS.md` and the tuning workflow so Claude knows that evolution affects more than the visible sliders.\[cite:2\]

3. **Restrict `evolve\\\_weights.py` to slider-visible weights only** if human-manageable tuning and UI parity are more important than full-parameter search.\[cite:2\]

### Suggested implementation note

If keeping the current architecture, the safest short-term fix is to:

- keep evolving all heuristic fields for strength,

- but add an **Advanced Heuristics** slider section or a debug export/import panel,

- and ensure `best.json` can be loaded, inspected, and edited without losing the hidden fields.\[cite:2\]

Suggested bug / enhancement title:

### Enhancement — GUI slider set is missing evolved heuristic weights ⬜

**Symptom:** `tools/evolve\\\_weights.py` tunes more heuristic fields than the web slider panel exposes.\[cite:2\]

**Root cause:** `tunable\\\_fields()` automatically includes all `HeuristicWeights` dataclass fields except `make\\\_mistakes` and `opening\\\_adherence`, but the frontend slider definition has not been kept in sync with the expanded dataclass.\[cite:2\]

**Fix:** Bring the frontend slider list into sync with `HeuristicWeights`, or explicitly split the dataclass into “UI-exposed” and “internal-only” weights and document that distinction clearly.\[cite:2\]  
  
Additional TODO items for Claude

## Evolve weights v2 — cross-personality master tuning

### Task

- [ ] Extend `tools/evolve\_weights\_v2.py` so it can evolve **one additional Master personality's weight set** while evaluating it against the other personalities, rather than only tuning a single generic weight profile.

### Recommendation

- Add a mode such as `--target-personality \<name\>` that selects one Master personality as the mutable candidate.

- Keep the other personalities fixed during each evaluation batch, and rotate opponents across the other available personalities so the candidate is not overfitting to one mirror matchup.

- Save outputs separately per personality, e.g. `data/weights/master\_\<name\>\_best.json`.

- Log which opponent personalities were faced in each generation / era.

- If `evolve\_weights\_v2.py` already supports multiple personalities internally, verify only the target personality is mutated and the others are loaded read-only.

### Why

- This should let one Master personality improve while still being stress-tested against the broader personality pool instead of only self-play against itself.

- It should reduce overfitting to one style and produce a more robust personality-specific profile.


## Tactical bug — black failed to close its own mill and missed white's immediate threat

### Game sequence

Keep this exact game in the notes and use it as a regression test:

```
1.d6 d2  
2.f4 b4  
3.c4 e4  
4.d3 d5  
5.a4 d7  
6.d1 e5  
7.e3 c3  
8.c5 a7  
9.g7 b6  
10.d1-g1 b4-b2
```

### Reported issue

At Black's last move, the AI played `b4-b2`. That appears wrong for two separate reasons:

1. Black should have seen the imminent White mill threat on the g-file / g-line, where White can play `f4-g4`.

2. Black already had a mill available to close on the b-line by moving `d2-b2`, and that move also appears to support a stronger dual-threat structure involving shared pressure from `b6` and the `d5-d6-d7` group.

So the AI seems to have chosen a non-converting move when it had a direct conversion available.

### What to check in logic

- [ ] Reconstruct the exact board position after move 10 and verify whether `d2-b2` is legal and recognized by the move generator.

- [ ] Check whether the search/evaluator is undervaluing **closing an immediately available mill** compared with positional reshuffling.

- [ ] Check whether the evaluator is underweighting **opponent immediate mill threats** in the move phase.

- [ ] Check whether the AI is correctly recognizing the **dual-purpose value** of `d2-b2`: immediate own mill plus improved follow-up pressure.

- [ ] Check whether move ordering or pruning caused the converting move to be searched too late or too shallowly.

- [ ] Check whether this is a heuristic-weight issue versus a logic bug in threat detection / candidate scoring.

### Likely failure modes

Possible causes include:

- immediate mill-closing bonus too low,

- immediate threat-blocking bonus too low,

- dual-threat continuation bonus too low,

- bug in move legality / adjacency / line detection,

- quiescence / tactical extension not applied to forcing move lines,

- search cutoff or move ordering causing the decisive move to be insufficiently explored.

### Suggested fix

- Add or increase a **must-convert immediate mill** priority in move phase when a legal closing move exists and does not walk into a clearly superior tactical refutation.

- Add or increase a **must-respect opponent immediate mill threat** priority.

- Add a bonus for **dual-purpose tactical moves** that both close a mill now and preserve / create a second forcing threat.

- If the move is being missed due to search depth rather than static eval, add a tactical extension for:

  - own immediate mill-closing moves,

  - opponent immediate mill threats,

  - dual-threat creation after a mill closure.

### Classification guidance

If `d2-b2` is being generated and evaluated but still loses narrowly to `b4-b2`, this is probably a **slider / weight calibration** problem. If `d2-b2` is not being surfaced correctly as an immediate mill-closing tactical move, this is more likely a **logic bug** in threat / line recognition or move generation.

### Add as regression test

- [ ] Add a regression test for the exact above sequence and assert that Black strongly prefers `d2-b2` (or at minimum ranks it above `b4-b2`) in the reconstructed position.

- [ ] Log the top candidate list and tactical feature scores for that position during test/debug mode so the cause can be diagnosed quickly.

  


### Enhancement B-46 — Add `placement_index` parameter to `tactical_move_bonus()` ⬜

**Goal:** Replace the 8+ scattered `pieces_placed < 6`, `pieces_placed >= 6`,
`pieces_placed < 9` checks in `tactical_move_bonus()` with a single
`placement_index = before.pieces_placed.get(color, 0)` value (0–8) and a
`late_placement_multiplier` curve. Prerequisite for B-28 and B-47.

**Multiplier curve (for setup-building bonuses):**

| placement_index | multiplier |
|---|---|
| 0–5 | 1.0 |
| 6 | 0.8 |
| 7 | 0.5 |
| 8 | 0.25 |

The multiplier applies to: `setup_mill`, `scatter_placement`, `feeder_diamond`,
`fork_anticipation`, and any other speculative structure bonus.
It does NOT apply to: `close_mill`, `block_opponent_mill`, `stop_opponent_mills`,
`convergence_block`, or emergency-block contributions — those are not setup bonuses.

**Files:**

- `ai/heuristics.py` — `tactical_move_bonus()`: add `placement_index` variable;
  apply `late_placement_multiplier` to the relevant bonus terms; remove the
  existing scattered threshold checks and replace with the single variable.

---

### Enhancement B-47 — White / Black asymmetric placement weights ⬜

**Goal:** Add side-specific weight fields to `HeuristicWeights` so White and Black
can have different strategic emphasis during the placement phase. Both sides share
the same HeuristicWeights instance today; this adds parallel fields for the asymmetric
bonuses, leaving the symmetric fields unchanged.

**New weight fields (add to `HeuristicWeights`):**

```python
# White early-placement cardinal emphasis (placements 1–4)
white_cardinal_early: int = 260    # replaces cardinal_block for White in placement_index 0–4
                                   # (base cardinal_block 200 + 30% boost)

# Black early-placement reactive blocking emphasis (placements 1–4)
black_convergence_early: int = 325  # replaces convergence_block for Black in index 0–4
black_fork_anticipation_early: int = 117  # replaces fork_anticipation for Black in index 0–4

# Black late-placement last-move advantage (placements 7–8)
black_dual_threat_late: int = 240   # replaces dual_threat_placement for Black in index 6–7

# White late-placement two-independent requirement (placements 7–8)
white_independent_mills: int = 280  # bonus for White having 2 independent 2-configs
                                    # at placement_index 7–8
```

**Application logic in `tactical_move_bonus()`:**
```python
# Early cardinal bonus (placement_index < 5)
cardinal_w = (weights.white_cardinal_early if color == "W"
              else weights.cardinal_block)

# Early convergence/fork blocking (placement_index < 5)
conv_w = (weights.black_convergence_early if color == "B"
          else weights.convergence_block)
```

**Why separate rather than scale:** White and Black have genuinely different strategic
goals in the same phase (White: proactive cardinal claim; Black: reactive disruption).
A shared weight tuned for one side is wrong for the other.

**Files:**

- `ai/heuristics.py` — `HeuristicWeights` dataclass: add 5 new fields above.
- `ai/heuristics.py` — `tactical_move_bonus()`: use the asymmetric fields when
  `placement_index` is in the relevant window.
- `tools/evolve_weights.py` — new fields are automatically included via
  `tunable_fields()`; no change needed unless `_FIXED_FIELDS` override is wanted.

---

## Search & Evaluation Enhancements

### TIER 1 — Core Search Stack (implement together)

### SE-1 — Transposition Table + Zobrist Hashing ✅ 2026-05-26

**Why:** The same board position is reached via many different move sequences (transpositions). Without a TT, `\\\_negamax` re-evaluates every transposed position from scratch. A TT keyed by a Zobrist hash stores `(depth, score, flag, best\\\_move)` per position, allowing the search to skip re-evaluation and use the stored best move for immediate ordering at that node. Expected gain: ~2× effective search depth in endgame; very large node savings throughout the move phase.

**NMM specifics:** Only 73 random 64-bit keys needed (24 squares × 3 states + 1 side-to-move bit). XOR-updated incrementally on each `apply\\\_move`.

**Critical implementation note:** Use a fixed-size `list` (pre-allocated, indexed by `hash % TABLE\\\_SIZE`) with depth-preferred replacement — **not** a Python `dict`. At high difficulty levels Python dict overhead would consume much of the gain.

**Deliverables:**

- `ai/transposition\\\_table.py` — new `TranspositionTable` class; `hash\\\_board()`, `lookup()`, `store()`

- `ai/game\\\_ai.py` — probe TT at top of `\\\_negamax`; store on exit; use hash-move as first candidate in ordering; reset between `choose\\\_move` calls

### SE-2 — Killer Heuristic (2 killers per depth) ✅ 2026-05-26

**Why:** A move that causes a beta cutoff at depth `d` in one branch is statistically likely to cause a cutoff in sibling branches at the same depth. Storing two such "killer" moves per depth and trying them before the unsorted remainder (but after captures/mill-closures) reduces node count by 20–30%. Zero change to evaluation quality; the implementation is ~15 lines.

Gains compound with SE-1: the TT provides a hash-move to try first at each node, killers then cover the next-most-likely cutoff movers.

**Deliverables:**

- `ai/game\\\_ai.py` — `self.\\\_killers` list (2 per depth up to depth 32); `\\\_store\\\_killer()`; insert killer-match tier between priority-1 and priority-2 in `\\\_order\\\_moves`; reset killers at start of each `choose\\\_move`

### SE-3 — History Heuristic ✅ 2026-05-26

**Why:** Maintains a global `hist\\\[(from\\\_sq, to\\\_sq)\\\]` table incremented by `depth²` whenever a move causes a beta cutoff. Used as a sort key within the priority-2 bucket of `\\\_order\\\_moves`. Unlike killers (depth-local), history is global across all positions, making the two techniques complementary.

**Largest gain in fly phase** where the existing sort leaves ~50 of 54 moves unordered. Together SE-1 + SE-2 + SE-3 should lift effective depth by 1.5–2 ply within the same time budget.

**Deliverables:**

- `ai/game\\\_ai.py` — `self.\\\_history` dict; increment on beta cutoff; use as tiebreaker in `\\\_order\\\_moves` priority-2 bucket; reset between `choose\\\_move` calls (or age between iterations)

### TIER 2 — High Value, after Tier 1

### SE-4 — Endgame Tablebase Query Inside Search ✅ 2026-05-26

**Why:** Currently `EndgameDB` is consulted only at root level in `choose\\\_move`. Querying it inside `\\\_negamax` at every node where `total\\\_pieces ≤ 8` returns `±INF` for known positions without any further search. This converts the lower search tree from estimated heuristic values to **exact outcomes** — a qualitative improvement, not just a speedup. The infrastructure already exists; this is approximately 10 lines of change.

**Deliverables:**

- `ai/game\\\_ai.py` — add `EndgameDB` lookup at top of `\\\_negamax` when `total\\\_pieces \\\<= 8`; return `outcome \\\* (INF - depth)` so fastest wins are scored first

### SE-5 — Principal Variation Search (PVS / NegaScout) ✅ 2026-05-26

**Why:** PVS assumes the first move explored is best (valid after good ordering from SE-1–3). All subsequent siblings are searched with a cheap zero-window `(alpha, alpha+1)` scout; only if the scout fails high is a full re-search triggered. With good ordering, the majority of siblings never need re-searching. ~10% additional node reduction on top of Tier-1 gains.

**Deliverables:**

- `ai/game\\\_ai.py` — replace inner loop in `\\\_negamax` with PVS scheme: first move at full window, siblings at zero-window with re-search on fail-high

### SE-6 — Late Move Reductions (LMR) ⬜ ★ Medium Impact

**Why:** Reduces search depth by 1 ply for moves sorted toward the end of the move list (assumed inferior after good ordering). **Largest proportional gain in fly phase** where branching factor reaches ~54 and the existing sort leaves most moves unordered.

**Guards (never reduce):**

- Mill-closing moves (priority-0)

- Opponent-mill-blocking moves (priority-1)

- Any move at depth \< 3 or root level (`\\\_score\\\_all`)

- Moves during iterative deepening at depth ≤ 2

**Rule:** reduce last 60% of sorted moves by 1 ply at depth ≥ 4; re-search at full depth if reduced score exceeds alpha.

**Deliverables:**

- `ai/game\\\_ai.py` — LMR applied after priority-0/1/killer ordering in `\\\_negamax`; conditional re-search on fail-high

### SE-7 — Aspiration Windows in Iterative Deepening ⬜ ★ Medium Impact

**Why:** Currently each iterative-deepening iteration restarts with `alpha = −INF, beta = +INF`. Using `\\\[prev\\\_score − 175, prev\\\_score + 175\\\]` for depth `d+1` produces more early cutoffs since most moves are outside the window. Fail-high or fail-low triggers a re-search at full window — rare in the positionally stable mid-game common in NMM.

**Deliverables:**

- `ai/game\\\_ai.py` — aspiration window around `prev\\\_score` in `\\\_iterative\\\_deepen`; window margin ~175 score units; widen and re-search on fail

### TIER 3 — Solid, Secondary Priority

### SE-8 — Search Extensions for Critical Positions ⬜ ★ Medium Impact

**Why:** +1 depth at nodes containing: forced mill closure (own or opponent); opponent has 2+ immediate mill threats (fork); position is 4v4 fly-phase; EndgameDB confirms a critical pattern. Root-level depth bonuses already exist in `choose\\\_move` — extend the same logic into internal `\\\_negamax` nodes. Cap total extensions at `depth / 2` per line to prevent blowup.

**Deliverables:**

- `ai/game\\\_ai.py` — extension check at top of `\\\_negamax` using existing tactical detection helpers; max-extension cap per line

### SE-9 — Quiescence Search (Capture Extension at Depth 0) ⬜ ★ Medium Impact

**Why:** Eliminates the horizon effect in 4v4 endgame and fly-phase transitions. At `depth == 0`, if a mill closure (capture) is immediately available, extend 1–2 plies searching only capture sequences before returning static evaluation. Cap at 2–3 extra plies to avoid cycling in repetitive mill positions.

**Deliverables:**

- `ai/game\\\_ai.py` — `\\\_negamax\\\_q()` quiescence search called at `depth == 0` when mill-closing moves exist; depth cap via `\\\_qsearch\\\_remaining` counter

### SE-10 — Proactive Fly-Fork Anticipation (Move Phase) ⬜ ★ Medium Impact

**Why:** The existing `fly\\\_fork\\\_bonus` fires reactively. The documented gap in `AI\\\_INTERNALS.md` is that the AI does not pre-plan the sequence of moves that *creates* the fork. Extend `\\\_fork\\\_in\\\_n(board, opp, n=2)` (already used in placement-phase, Enhancement B-4) to the move phase: scan forward up to 3 half-moves for forcing lines that result in 2+ simultaneous 2-configs.

**Deliverables:**

- `ai/heuristics.py` — `\\\_move\\\_phase\\\_fork\\\_anticipation(board, color, depth=3)`; bonus `fork\\\_depth × 80` added to root move score

### SE-11 — Opponent Likelihood Weighting (Asymmetric Depth via TrajectoryDB) ⬜ ★ Medium Impact

**Why:** Standard alpha-beta allocates equal depth to all opponent responses regardless of how likely they are. Using the existing `TrajectoryDB`, empirical move frequency at the current game prefix can drive +1 extension for high-frequency opponent moves and −1 LMR for rare ones. Analogous to LMR but data-driven on actual opponent behaviour rather than sort position.

**Deliverables:**

- `ai/trajectory\\\_db.py` — `query\\\_move\\\_frequency(prefix, notation)` method returning normalised frequency `\\\[0.0, 1.0\\\]`

- `ai/game\\\_ai.py` — apply frequency-based depth delta at opponent nodes inside `\\\_negamax`

### TIER 4 — Infrastructure / Long-Term

### SE-12 — Incremental Evaluation Cache (Zobrist-Keyed Sub-Functions) ⬜

**Why:** Heavy heuristic sub-calls (`\\\_convergence\\\_cluster\\\_count`, `\\\_mill\\\_wrapping`, `\\\_free\\\_piece\\\_assembly`, `\\\_assembly\\\_reach\\\_count`) recompute from scratch every leaf call. With Zobrist hashing already in place (SE-1), a secondary cache keyed by board hash stores sub-function results and invalidates on state change. Requires SE-1.

**Deliverables:**

- `ai/heuristics.py` — result cache dict keyed by Zobrist hash for top-cost sub-functions; invalidate on apply\_move

### SE-13 — N-Gram Opponent Move Predictor ⬜

**Why:** Complements TrajectoryDB (which tracks win/loss rates) with a pure move-frequency bigram/trigram model: given the last N moves, predict opponent's next move distribution. Feeds into SE-11 with richer per-sequence predictions. Lower priority since TrajectoryDB already covers this partially.

**Deliverables:**

- `ai/ngram\\\_opponent\\\_model.py` — new `NGramOpponentModel` class; `update()` called after each game; `predict()` returns probability dict; trained incrementally from `data/games/` JSONL records

## Architecture Principles

- **Immutable board state** — `BoardState.apply\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_move()` always returns a new object. Enables safe undo, MCTS branching, and self-play without deep-copy overhead.

- **Coordinator owns the narrative** — All commentary and LLM calls flow through `Coordinator`. `GameAI` is pure search; `MillsLLM` is pure text generation. Neither knows about the other.

- **No cloud dependency** — All LLM inference runs locally via Ollama. No API keys, no cost after initial model pull.

- **Progressive enhancement** — Every stage adds capability without breaking the previous one. Fast mode (`--no-llm`, no opening book) always works as a fallback.

- **Weight-injectable heuristics** — All evaluation weights are injectable via `HeuristicWeights`. The Settings page, evolution driver, and self-play all use the same injection point.

- **Tactical before positional** — The AI urgency hierarchy (close mill → block mill → disrupt structures → position) is a first-class design constraint, not an afterthought.

- **Staged opening memory** — Starting play is recognised in phases (early, 12-piece mid-placement, final placement), with move-sequence ancestry and searchable tags preserved so both the engine and the study tools can reason over opening families rather than only isolated final lines.



### Tactic B-39 — Avoid creating disrupted 2-configs next to an opponent blocker ✅ 2026-05-26

**Symptom:** the AI sometimes places a piece beside its own piece when that adjacent pair is already pinned or spoiled by an opponent piece on the far side, creating a disrupted 2-config that looks like structure but is usually strategically bad.

**Example:** opponent at `d7`, own piece at `d6`, placing at `d5` creates a disrupted vertical 2-config unless it is serving another tactical purpose. In general this should be treated as a bad move, not as ordinary 2-config progress.

**Requested behaviour:** in placement and early movement phases, penalise creating own adjacent pairs where the third square relationship is already spoiled by an opponent blocker, unless the move also serves a stronger tactical purpose such as an emergency block, immediate mill, dual-purpose tactic, or a verified busy-chain continuation.

**Suggested implementation:**

- `ai/heuristics.py` — add `disrupted_two_config_penalty` for patterns where `(own, own, opp)` or the equivalent blocked adjacency makes the 2-config non-convertible or strategically cramped.

- `ai/heuristics.py` — add helper such as `\_is_disrupted_two_config(board, move, side) -> bool`.

- `ai/game_ai.py` — ensure B-38 tactical ladder exceptions can override this penalty only when a stronger tactical reason is present.

- `AI_INTERNALS.md` — document the disrupted-2-config rule and the example `d7` / `d6` / `d5`.

**Files:**

- `ai/heuristics.py`

- `ai/game_ai.py`

- `AI_INTERNALS.md`

### Tactic B-40 — Preserve and cycle shared-piece dual mills until endgame ✅ 2026-05-26

**Goal:** when the AI has a true dual-mill configuration (two 2-configs or mill closures sharing a pivot piece), it should preserve and use that structure repeatedly instead of abandoning it for random reshuffling, at least until the endgame transition or until a clearly superior tactical conversion exists.

**Requested behaviour:** if the AI has a shared-piece dual mill config it can legally cycle, it should keep cycling the two mills to extract captures and maintain pressure until at least endgame is reached, unless doing so walks into a stronger refutation or exact endgame DB guidance says otherwise.

**Suggested implementation:**

- `ai/heuristics.py` — increase `shared_pivot_detection` and `mobile_mill_value` when the oscillation is legal and repeatable.

- `ai/game_ai.py` — add a persistence bonus for preserving a working dual-mill cycle instead of breaking it voluntarily.

- `AI_INTERNALS.md` — document the rule that shared-piece dual mills are a maintain-and-harvest structure, not just a one-turn tactic.

> **Tie-in (B-30, B-37):** this generalises the dual-mill oscillation requirement from the specific busy-chain example in B-30 into a standing policy for mid/late movement phases.

**Files:**

- `ai/heuristics.py`

- `ai/game_ai.py`

- `AI_INTERNALS.md`

### Tactic B-41 — When opening a mill, move the safest re-closing piece ✅ 2026-05-26

**Symptom:** when the AI opens one of its own mills to keep mobility, it may move the wrong piece and leave itself unable to close the mill again next turn because the re-closing route is blocked or immediately contestable.

**Requested behaviour:** if the AI opens a mill, it should prefer moving the mill piece whose departure still leaves a clean, re-closeable mill on the next turn — specifically, the moved piece should be the one with no adjacent blocker or opposing stopper that can prevent re-closure on the next move.

**Suggested implementation:**

- `ai/heuristics.py` — add `safe_mill_open_bonus` and `unsafe_mill_open_penalty`.

- `ai/game_ai.py` — when candidate moves open an existing mill, evaluate which moved square preserves the highest probability of immediate legal re-closure.

- `AI_INTERNALS.md` — document that opening a mill is not neutral; the preferred departure square is the one least vulnerable to immediate interruption.

**Files:**

- `ai/heuristics.py`

- `ai/game_ai.py`

- `AI_INTERNALS.md`

### Bug B-42 — Endgame move choice is drifting instead of closing available mills ✅ 2026-05-26

**Symptom:** in endgame positions the AI is sometimes moving pieces around the board seemingly at random instead of closing an available mill.

**Why this matters:** late movement and fly phases are the least tolerant of wasted tempi. If a legal mill closure exists and is not tactically refuted, failing to take it can throw away a won or drawable endgame.

**Requested behaviour:** diagnose why the AI is drifting in endgame instead of converting available mills. Exact endgame DB hits should dominate. On DB miss, heuristic fallback should strongly prefer immediate mill closure over random relocation.

**Suggested checks:**

- verify exact endgame DB lookup is firing when the position is within solved range

- verify `\_move_closes_mill()` is still recognised in late movement and fly phases

- verify phase-specific fallback is not underweighting `mill_closure_rate` or `immediate_mill_bonus` in endgame

- verify move ordering and pruning are not hiding immediate conversions

> **Tie-in (B-23, B-37, B-38):** exact endgame DB should have first priority, and on miss the late-movement/fly phase plus tactical ladder should make immediate mill closure the default conversion choice.

**Files:**

- `ai/game_ai.py`

- `ai/heuristics.py`

- `ai/endgame_solved_db.py`

- `AI_INTERNALS.md`

### Tactic B-43 — If both sides threaten immediate mills, prefer the mill closure that removes the opponent threat ✅ 2026-05-26

**Scenario:** the opponent has a ready-to-close mill (an open 2-config plus an adjacent feeder piece that can close next move), and the AI also has a ready-to-close mill.

**Requested behaviour:** if the AI can close its own mill in a way that also removes the opponent's immediate mill threat, it should do that. In other words, when both sides have immediate conversions available, prefer the AI mill closure that simultaneously neutralises the opponent threat by capture or by structural disruption.

**Suggested implementation:**

- `ai/game_ai.py` — extend tactical ladder Priority 5 dual-purpose logic so it explicitly recognises "close own mill and remove opponent immediate mill threat" as a premium tactical class.

- `ai/heuristics.py` — add or raise `dual_purpose_bonus` where the removal target is the opponent's feeder or blocking piece needed for next-turn mill closure.

- add regression tests where both sides have open 2-config plus feeder-piece threats and assert the AI chooses the converting-and-neutralising line.

**Files:**

- `ai/game_ai.py`

- `ai/heuristics.py`

- `AI_INTERNALS.md`

### Tactic B-44 — Mill captures should preferentially remove the piece the opponent most needs to replace ✅ 2026-05-26

**Goal:** when making a mill, the AI should not capture randomly or merely by static value. It should preferentially remove the opponent piece whose loss forces the opponent to spend a move or placement replacing a critical blocker, feeder, or anti-mill piece.

**Requested behaviour:** mill captures should strongly prefer pieces that:

- are currently stopping one of the AI's 2-configs from becoming a mill on the next move or next placement

- are the feeder piece the opponent needs to complete its own 2-config

- are the only practical blocker preventing the AI from re-closing a mobile mill

- force the opponent to spend its next action repairing structure rather than progressing its own plan

**Suggested implementation:**

- `ai/heuristics.py` — add `capture_replacement_pressure_bonus` and `capture_feeder_removal_bonus`.

- `ai/game_ai.py` — in capture selection, rank removable opponent pieces by replacement burden and threat interruption, not only by local mobility or raw piece centrality.

- `AI_INTERNALS.md` — document that best capture is often the piece the opponent most urgently needs back, not merely the most central one.

**Files:**

- `ai/heuristics.py`

- `ai/game_ai.py`

- `AI_INTERNALS.md`

### Bug B-48 — Endgame DB WIN lookup returns `moves[0]` instead of the best winning move ✅ 2026-05-26

**Symptom:** when `choose_move()` gets `WDL_WIN` back from the retrograde endgame DB, it returns `moves[0]` — the first legal move in generation order — rather than the specific move that leads to a position labelled `WDL_LOSS` for the opponent.

**Why this matters:** in a won 3v3 fly endgame the AI knows it is winning but may still take dozens of extra moves to convert, or even accidentally step into a drawn line, if `moves[0]` happens to move to an opponent WIN or DRAW position.

**Correct fix:** iterate over all legal fly moves from the current position; for each, compute the successor position and query `EndgameSolvedDB` with the side-to-move flipped; return the first move whose successor decodes to `WDL_LOSS` for the opponent. Fall through to the existing heuristic search only if no such move is found (should not happen in a correctly solved DB).

**Sketch:**

```python
# In choose_move(), replace the WDL_WIN early-return block:
for move in moves:
    succ_board = apply_move(board, move)
    succ_wdl = _esdb.query(succ_board)   # now queried from opponent's perspective
    if succ_wdl == "L":                  # opponent is in a LOSS → this is our best move
        self.last_thinking = "endgame DB (win)"
        return move
# no exact winner found — fall through to heuristic
```

Note: `EndgameSolvedDB.query()` already reads the `turn` field from `board.current_turn`, so after `apply_move` the successor board has the opponent's turn set automatically.

**Files:**

- `ai/game_ai.py` — `choose_move()` endgame DB WIN block

- `ai/endgame_solved_db.py` — may need `query()` to accept an optional explicit turn override

- `tests/test_game_ai.py` — add test: construct a known-won 3v3 position, assert `choose_move` picks the converting move

### Enhancement B-45 — Replace automatic AI resignation with an offer of defeat ⬜

**Goal:** change the current automatic AI resignation into an offer of defeat that the human player can accept or decline, allowing lost positions to be played out when desired.

**Requested behaviour:** when the AI reaches the current resignation threshold, it should present an offer of defeat instead of ending the game immediately. The human can grant the defeat or continue playing the position out.

**Why this matters:**

- some users want to verify the win on the board

- some positions are useful for debugging endgame behaviour even if the AI believes it is lost

- resignation offers are less disruptive to training-data capture and post-game analysis than forced auto-termination

**Suggested implementation:**

- `web/app.py` — replace the immediate resignation branch with an offer state stored in the game/session model

- `web/static/game.js` — show a UI prompt with accept-decline controls

- if declined, continue the game normally and log that the AI offered defeat but play continued

- ensure opening and game records are still persisted regardless of whether the offer is accepted or declined

> **Tie-in (B-31):** resignation-related game records and opening sequences must still be saved when an offer of defeat is made, accepted, or declined.

**Files:**

- `web/app.py`

- `web/static/game.js`

- `web/templates/index.html`

- `AI_INTERNALS.md`

### Bug B-49 — `endgame_solved_dir` relative path not resolved from project root ✅ 2026-05-26

**Symptom:** Server logs `EndgameSolvedDB: not found at data/endgame` even though the setting `"endgame_solved_dir": "data/endgame"` is present in `data/settings.json`. The DB file lives at `/mnt/windows/NMM_DB/endgame_3_3.wdl` (external drive).

**Root cause:** `web/app.py` does `Path(_load_settings().get("endgame_solved_dir") or ...)`. When the value is the relative string `"data/endgame"`, Python resolves it from the process working directory, not from `_ROOT` (the project root). If the server is launched from a different directory the path silently misses.

**Fix:**

```python
raw = _load_settings().get("endgame_solved_dir") or ""
_esdb_dir = Path(raw) if (raw and Path(raw).is_absolute()) else (_ROOT / (raw or "data/endgame"))
```

Apply the same pattern to `fullgame_db_path` for consistency.

**Immediate workaround:** set `"endgame_solved_dir": "/mnt/windows/NMM_DB"` in `data/settings.json` (absolute path bypasses the resolution issue).

**Files:**

- `web/app.py` — path resolution for `endgame_solved_dir` and `fullgame_db_path`

- `data/settings.json` — update default/example to absolute path or document that relative paths are relative to project root

### Bug B-50 — AI does not block opponent double 2-config (fork) during placement phase ✅ 2026-05-26

**Symptom:** In AI-vs-AI play (and against human) both players keep building 2-configs during the placement phase but neither issues mandatory blocking moves when the opponent has formed two simultaneous 2-configs (a fork). The result is that the opponent reaches turn 9 with a pre-formed fork and closes a mill unopposed.

**Root cause:** `_immediate_mill_threats()` in `ai/game_ai.py` only fires for move-phase and fly-phase positions:

```python
opp_in_fly = opp_placed >= 9 and board.pieces_on_board[opp] <= 3
...
if opp_in_fly:
    threats.add(empty)
elif opp_placed >= 9:   # move phase: adjacent piece needed
    ...
# placement phase (opp_placed < 9): returns empty set always
```

Because it always returns `{}` during placement, the mandatory-block filter at lines 394–398 (`if threats: moves = blocking`) never restricts candidates. `_order_moves` does promote 2-config-closing squares into the P1 bucket during placement (no phase guard there), so *ordering* works, but it only affects which move is tried first — it doesn't prevent the AI from choosing a development move over a block if the search prefers it.

The heuristic weights (`stop_opponent_mills=400 > setup_mill=100`) should in theory prefer blocking, but this only holds when the eval can see the fork being exploited within the current search depth. In the early placement phase at depth 5–7, the converging fork often resolves outside the horizon.

**Proposed fix (no new functions):**

Extend `_immediate_mill_threats()` to also fire during the **placement phase** when the opponent holds **≥2 simultaneous 2-configs** (a fork — cannot be blocked by a single piece):

```python
# Placement phase: flag any 2-config closing square when opp has ≥2 simultaneous 2-configs.
if opp_placed < 9:
    two_config_empties = [
        next(p for p in mill if board.positions[p] == "")
        for mill in MILLS
        if [board.positions[p] for p in mill].count(opp) == 2
        and [board.positions[p] for p in mill].count("") == 1
    ]
    if len(two_config_empties) >= 2:
        threats.update(two_config_empties)
```

**Why ≥2:** a single 2-config does not need mandatory blocking — the AI should weigh it against development via normal search. Only when the opponent holds two (possibly sharing a pivot piece), blocking becomes obligatory since one response cannot neutralise both threats.

**Why not a new function:** this is a targeted extension of existing `_immediate_mill_threats()` logic; the mandatory-block filter already uses its output correctly.

**Edge case:** shared-pivot forks (opponent has three pieces forming two incomplete mills) count as two 2-configs and are correctly caught because both empty closing squares appear in `two_config_empties`.

**Files:**

- `ai/game_ai.py` — extend `_immediate_mill_threats()` placement-phase branch (~8 lines)

**Tests needed:**

- Board with opponent having exactly two 2-configs during placement → `_immediate_mill_threats()` returns both closing squares
- Board with opponent having one 2-config during placement → returns empty set
- Board with opponent having two 2-configs but one closing square is occupied → only unoccupied closing square returned (or empty, depending on actual count)

### B-51 — Early-Endgame DB: expand retrograde solver beyond 3v3 ⬜ ★ High Impact

**Goal:** Build a family of WDL tables covering piece counts from 4v3 through 7v4 (and symmetric reverses). These cover the critical **early endgame transition** — positions where one or both sides have just lost pieces but haven't reached fly phase yet.

**Table sizes (2 bits/position, white_rank × black_rank × turn encoding):**

| nW | nB | Positions | MB |
|----|----|-----------|----|
| 4 | 3 | 24,227,280 | 6.1 |
| 3 | 4 | 24,227,280 | 6.1 |
| 5 | 3 | 82,372,752 | 20.6 |
| 3 | 5 | 82,372,752 | 20.6 |
| 4 | 4 | 102,965,940 | 25.7 |
| 5 | 4 | 329,491,008 | 82.4 |
| 4 | 5 | 329,491,008 | 82.4 |
| 6 | 3 | 219,660,672 | 54.9 |
| 3 | 6 | 219,660,672 | 54.9 |
| 7 | 3 | 470,701,440 | 117.7 |
| 3 | 7 | 470,701,440 | 117.7 |
| 6 | 4 | 823,727,520 | 205.9 |
| 4 | 6 | 823,727,520 | 205.9 |
| 7 | 4 | 1,647,455,040 | 411.9 |
| 4 | 7 | 1,647,455,040 | 411.9 |
| **All** | | | **~1,825 MB** |

**Practical tiers:**
- **Tier 1 — Recommended:** 4v3, 3v4, 5v3, 3v5, 4v4 → ~79 MB total. Covers the most common early-endgame transitions.
- **Tier 2 — Optional:** add 5v4, 4v5 → ~244 MB total. Significant, but fits comfortably on an external drive.
- **Tier 3 — Large/optional:** 6v3, 3v6, 7v3, 3v7 (118–220 MB each pair); 6v4, 7v4 (200–412 MB each). Only worthwhile if Tier 1+2 improve play measurably.

**Key algorithm changes vs the existing 3v3 builder:**

1. **Mixed fly/move successor generation:** a side with exactly 3 pieces flies (any empty square); a side with ≥4 pieces moves along adjacency edges. The successor generator must branch on `len(mover) == 3` per turn.

2. **Cross-table captures:** a capture in an nW v nB position leaves (nW) v (nB-1) or (nW-1) v nB. These successor positions live in a *different* (smaller) table. The retrograde solver must load the relevant already-solved tables as read-only references. A capture that reduces a side to 2 pieces is an immediate WIN (game over rule).

3. **Build order (dependency chain):** each table depends on both smaller tables from captures:
   ```
   3v3 (done) ──┬── 4v3 ──┬── 5v3, 4v4
                └── 3v4 ──┘   │
                               └── 5v4, 6v3, 4v4 ...
   ```
   Concretely: solve in order of (nW + nB) ascending; within the same total, order doesn't matter.

4. **File naming:** `endgame_{nW}_{nB}.wdl` alongside the existing `endgame_3_3.wdl`. Same 2-bit-packed format with the existing `get_wdl`/`set_wdl` helpers.

5. **Query integration:** extend `EndgameSolvedDB` to load all available `endgame_{nW}_{nB}.wdl` files at startup; `query()` dispatches to the correct table by `(len(w_pieces), len(b_pieces))` — no API change for callers.

**Feasibility:** Yes, up to Tier 2. The algorithm rework (adjacency vs fly per side, cross-table lookups) is non-trivial — estimate 1–2 days of implementation. The retrograde passes for Tier 1 should complete in minutes; Tier 2 in under an hour on modern hardware.

**Files:**

- `tools/build_endgame_db.py` — rewrite to accept `--nW` and `--nB` args; mixed fly/move successor generator; cross-table reference loading; output `endgame_{nW}_{nB}.wdl`
- `ai/endgame_solved_db.py` — extend `EndgameSolvedDB.__init__` to load all available tables; extend `query()` to dispatch by piece count; remove hardcoded 3v3 guard

