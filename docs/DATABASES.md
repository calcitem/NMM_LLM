# NMM Databases Reference

Every database the AI reads or writes, what it stores, how it is built, and when the engine consults it.

---

## 1. TrajectoryDB (`ai/trajectory_db.py`)

**File:** `data/trajectory_db.json` (auto-created on first run)

**What it stores:** Win-rate statistics indexed by move-notation prefix. Every completed game contributes positive deltas to the winner's move sequences and negative deltas to the loser's. A special delta of exactly `−1.0` is a hard ban (written by the Bad Move button) and causes the move to receive `−INF+1` regardless of other signals.

**How it is built:** Updated automatically at the end of every game played through the web server. Also populated by `tools/self_play.py` and `tools/endgame_play.py`.

**D4 symmetry:** All prefixes are stored in canonical (lex-minimum) D4 form. Queries search all 8 D4 equivalents and merge statistics, then inverse-transform move notations back to the actual board orientation. This multiplies effective sample size by up to 8× with no extra games.

**When consulted:** After opening recognition diverges (`choose_move()` in `ai/game_ai.py`). Statistical hints in `[−0.5, +0.5]` are added to each candidate's negamax score before final selection.

---

## 2. EndgameDB (`ai/endgame_db.py`)

**File:** `data/endgame_db.json` (auto-created)

**What it stores:** Board-position snapshots from completed games, indexed by a 24-character board string. For each stored position it records win/loss/draw outcomes and piece configurations.

**How it is built:** Updated automatically at game end. Populated faster with `tools/endgame_play.py --seed-from-games`.

**D4 symmetry:** Positions are stored in canonical D4 form; queries search all 8 equivalents.

**When consulted:** When `EndgameRecognizer` marks the game active (≤ 11 total pieces), the DB provides extra search depth for known positions.

---

## 3. EndgameSolvedDB — Retrograde WDL (`ai/endgame_solved_db.py`)

**File:** `data/endgame/endgame_3_3.wdl` (and additional tables per piece count)

**What it stores:** Mathematically exact Win/Draw/Loss values for every legal fly-phase position. Each value is 2 bits; the 3v3 table is ~1.3 MB. Built once offline by retrograde analysis — never written at runtime.

**How it is built:** `tools/build_endgame_db.py`. The retrograde solver uses D4 symmetry (~8× speedup), starts from terminal positions, and propagates backward. `--build-all` builds all tables in dependency order up to `--max-sum`.

**Encoding:** Combinatorial number system — `pos_id = combo_rank(white_squares) × C(21,3) × 2 + combo_rank(black_squares) × 2 + turn_bit`. Table size for 3v3: 5,383,840 entries.

**When consulted:** In `choose_move()` when both sides have exactly 3 pieces and all 9 pieces have been placed. Returns `"W"` / `"L"` / `"D"` / `None`. Consulted before the FullGameDB and negamax search.

**Known limitation (B-85):** WIN/LOSS counts are not symmetric (~5:1 ratio). The table still returns results but some 3v3 WDL values may be incorrect due to a canonicalization issue in the offline solver. Tracked in `plan_todo.md`.

---

## 4. FullGameDB (`ai/fullgame_db.py`)

**File:** `data/fullgame.bin` (binary, sorted)

**What it stores:** Position keys (~9 bytes each) with WDL outcomes for positions seen in human games, BFS-expanded around frequently-visited positions to the configured depth. Stored sorted for O(log N) binary search at runtime.

**How it is built:** `tools/build_fullgame_db.py`. Scans JSONL game records, BFS-expands around seeds, back-propagates win/loss outcomes, writes sorted binary output. Uses a temporary SQLite DB during the build to avoid holding all data in RAM.

**D4 symmetry:** Each position is stored in canonical D4 form (one equivalence class stored once, ~8× compression).

**When consulted:** In `choose_move()` after the EndgameSolvedDB check, before negamax. If the current position is in the DB and a clear WDL signal exists, the best-labelled move is preferred.

**Build options:**

| Flag | Default | Effect |
|------|---------|--------|
| `--min-seed-frequency N` | 2 | Only positions seen ≥ N times seed the BFS |
| `--expand-depth D` | 4 | BFS expansion depth from seeds |
| `--max-db-gb GB` | 10 | Stop BFS if temp SQLite DB exceeds this size |
| `--max-gb GB` | 6 | Stop BFS if process RAM exceeds this |

---

## 5. Opening Book (`ai/opening_book.py`)

**Files:** `data/openings/learned_openings.json`, `data/openings/book_openings.json`

**What it stores:** Named opening lines with per-line win/loss/draw statistics and UCB1 scores. Each line is a sequence of placement move notations. Lines can be imported from a strategy book, learned from played games, or named by the LLM.

**How it is built:** Seeded by `tools/import_openings.py` and `tools/import_book_games.py`. Updated automatically at game end when the played sequence matches or extends a known line. Novel sequences are saved with `needs_llm_name=True` and named by `tools/name_openings.py` or during self-play with `--name-openings`.

**Selection:** UCB1 with temperature sampling (`temperature = 0.18`) — best lines are most likely but under-explored lines get real play time. Filtered to the AI's side (White-winning lines only offered when AI plays White, etc.).

**D4 symmetry:** The `OpeningRecognizer` scans all 8 D4 variants of the current board to match book openings regardless of rotation or reflection.

---

## 6. ChromaDB — LLM Vector Memory

**Directory:** `data/chroma/`

**What it stores:** Vectorised summaries of past game positions and strategic insights from `MillsLLM`. Used for nearest-neighbour retrieval of relevant strategic context when the LLM is generating commentary or move suggestions.

**How it is built:** Written during LLM-enabled games via `ai/memory_manager.py`. Read back at game start and during deliberation via `_strategy_context()` in `ai/mills_llm.py`.

**When consulted:** During LLM deliberation (`Coordinator.deliberate()`). The top-K nearest strategic memories are prepended to the LLM prompt as context.

---

## 7. Malom Ultra-Strong Solved Database (external, training only)

**Directory:** configured in `configs/sentinel_default.yaml` → `external_db_path`
**Format:** Binary `.sec2` sector files (498 sectors total), one per `(W_pieces, B_pieces, W_flying, B_flying)` combination. A `.secval` file stores virtual win/loss thresholds.

**What it stores:** Perfect-play WDL for every legal Nine Men's Morris position. Built by Gévay and Danner (GPL-3, `ggevay/malom`). Covers the full game — not just endgame positions.

**How it is used:** Read-only at sentinel training time by `learned_ai/sentinel/db_teacher.py`. For each legal move in each training position, the DB returns exact WDL (`"win"` / `"draw"` / `"loss"` from the mover's perspective). These become quality labels (`1.0` / `0.5` / `0.0`) for the sentinel's BCE training objective.

**Hashing:** The board→index hash uses D4 × ring-swap 16-symmetry canonical form + combinadic ranking, ported from [Sanmill](https://github.com/calcitem/Sanmill). 210,140 entries validated in `std_3_3_0_0.sec2`.

**Not used at runtime:** The Malom DB is only consulted during sentinel training and evaluation (`scripts/evaluate_sentinel.py`). The game AI never queries it during play.

---

## 8. Value Network (`data/value_net.npz`)

**Format:** NumPy `.npz` archive containing MLP weights (3-layer: 79 → 128 → 64 → 1).

**What it stores:** A position evaluator trained on game outcomes. Input is 24 board positions × 3 one-hot channels + 7 scalar metadata = 79 features, encoded from the moving player's perspective. Output is a `tanh` scalar in (−1, 1): positive = current player likely wins.

**How it is built:** `tools/train_value_net.py`. Reads all JSONL files from one or more `--games-dir` directories. Labels each position with `+1.0` (mover won), `−1.0` (mover lost), or `0.0` (draw/unknown). Trains with mini-batch SGD (MSE loss, pure numpy, no GPU required). Saves to `--output` (default `data/value_net.npz`). Training ~200+ games gives useful signal; retraining after each major batch takes under a minute.

**Current status:** Dormant infrastructure. The value net trains and saves correctly, but the production game path does not currently load it by default — gameplay is unaffected whether or not `data/value_net.npz` exists. The hook is in `ai/mcts.py` and `ai/game_ai.py` (`value_net` parameter); the **Value network blend %** slider in AI Tuning controls the blend weight (0 = heuristic only, 100 = value net only) when wired in.

**Target performance:** Val loss < 0.55 (baseline random). A well-trained value net on 1000+ varied games reaches ~0.45–0.50 val loss.

---

## 9. Sentinel Checkpoint (`learned_ai/sentinel/checkpoints/`)

**Files:** `best.pt` (lowest val loss epoch), `latest.pt` (most recent epoch)

**Format:** PyTorch checkpoint dict — `{state_dict, optimizer, config, epoch, best_val}`.

**What it stores:** Trained weights for the sentinel move-quality scorer (58 → 128 → 64 → 32 → 1 sigmoid MLP). Each forward pass scores one candidate move in [0, 1].

**How it is built:** `scripts/train_sentinel.py`. Reads JSONL game records, generates one `MoveExample` per legal move per position (not just the played move), labels with Malom DB WDL when available, trains with BCE loss. `best.pt` is selected automatically.

**To retrain from scratch:** Delete both `best.pt` and `latest.pt` and rerun training.

**When consulted:** At runtime in `ai/game_ai.py` when a sentinel is attached (`set_sentinel()`). Runs one batched forward pass over all candidate moves before the engine commits to a choice. Also consulted by `scripts/evaluate_sentinel.py` and `scripts/sentinel_review.py`.

**Validated performance** (3.6 M examples, 83% Malom DB labelled, 50 epochs):
- Val BCE loss: 0.0355
- Move-quality accuracy vs Malom DB: 99.6%
- Winning-trajectory accuracy: 100%, mean score 0.891 ± 0.109
- Losing-trajectory accuracy: 99.2%, mean score 0.460 ± 0.176
- Game-level trajectory polarity: 90%

---

## Summary

| Database | File/Dir | Written by | Read by | Purpose |
|----------|----------|-----------|---------|---------|
| TrajectoryDB | `data/trajectory_db.json` | web server, self-play | `game_ai.py` | Move-prefix win-rate hints |
| EndgameDB | `data/endgame_db.json` | web server, self-play | `game_ai.py` | Learned endgame positions |
| EndgameSolvedDB | `data/endgame/*.wdl` | `build_endgame_db.py` | `game_ai.py` | Exact retrograde WDL (3v3+) |
| FullGameDB | `data/fullgame.bin` | `build_fullgame_db.py` | `game_ai.py` | BFS-expanded position WDL |
| Opening Book | `data/openings/*.json` | web server, tools | `opening_book.py` | UCB1-selected opening lines |
| ChromaDB | `data/chroma/` | `memory_manager.py` | `mills_llm.py` | LLM strategic vector memory |
| Malom DB | (external, user-configured) | Gévay/Danner | `db_teacher.py` | Perfect-play WDL labels (training only) |
| Value Network | `data/value_net.npz` | `train_value_net.py` | `ai/mcts.py` (dormant) | MLP position evaluator |
| Sentinel | `learned_ai/sentinel/checkpoints/` | `train_sentinel.py` | `game_ai.py` | Move-quality scorer |
