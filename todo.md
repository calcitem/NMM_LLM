# Rust-native AI move pipeline — consolidated plan

This document tracks the remaining work to make Rust the primary end-to-end
search engine, retaining every Python behaviour that matters and adding the
search techniques Rust is missing today.

Four tracks, in recommended order:

- **Track A — Correctness.** ✅ DONE. `py_search_root_scored` exposes per-move scores; `_choose_rust_scored` applies opening/trajectory bonuses on top. Rust is now the primary path for all `top_n==1` moves.
- **Track B — Search strength.** ✅ DONE. PVS, LMR+aspiration windows, killers/history, quiescence search, and null-move pruning all implemented in `search.rs`.
- **Track C — Integration.** Rust-native SE-11b frequency extension, DB probes
  inside the search, persistent TT across turns.
- **Track D — Cleanup.** Route the last Python helpers through the Rust
  equivalents that already exist.

**Known regression from Track A/B (2026-07-02):** `test_completed_ponder_ai_has_populated_tt`
— Rust search handles move selection but doesn't populate the Python `_tt` object.
Ponder pre-computation (B-94) no longer seeds the TT used by the live search.
Not a crash; revisit when touching ponder or when T-C4 (persistent Rust TT) is implemented.

## Motivation (Track A)

`_choose_rust` is fast (`R:OK` ~ms), but any active Python-side hint currently
forces fallback to `_iterative_deepen` (`P:HINTS-ACTIVE`). The four Python
pipelines are:

1. **Opening book adjustments** — `_apply_opening_adjustments` adds bonuses to
   `recognition.book_move` and penalties to `recognition.common_blunders`.
2. **TrajectoryDB / HumanDB score deltas** — `_apply_trajectory_hints` applies
   `trajectory_hints: dict[notation, float]` as additive root-move deltas.
3. **TrajectoryDB winner-follow ordering** — `self._trajectory_line: list[(notation, score)]`
   promotes top-3 human-preferred moves to the front of the root move list
   (search efficiency, not selection).
4. **Root move filters** — mandatory-block (`_immediate_mill_threats`), sentinel
   bans, dead-square filter, DB rescue moves. These shrink the legal root set.

All four apply **at the root only**. Rust knows nothing about any of them and
does its own root loop, so we've been routing to Python whenever any is active.
That loses the Rust speedup exactly when the AI is following mastered play — the
worst tradeoff.

## Design (Track A)

**Rust returns per-move scores at root; Python applies hint semantics on top.**

- Rust adds a new PyO3 entry point `py_search_root_scored` that returns
  `(nodes, depth_reached, Vec<(from|None, to, cap|None, score)>)` sorted best-first.
- Python takes that list, intersects with any active filter, adds any active
  bonuses/penalties, and picks the best remaining move.
- All bonus/filter code stays in Python (small, well-tested, close to the
  authoritative implementations). Rust just does the tree search.

Score compatibility: Rust `evaluate_v2` and Python `evaluate_v2` use the same
integer scale, so Python's bonus deltas plug into Rust scores directly.

---

## Track A — Correctness

### M1 — Rust: expose scored root moves

**File:** `native/nmm_core/src/search.rs`

- Add `pub struct RootMoveScore { pub mv: Move, pub score: i64 }`.
- Add `pub struct SearchResultScored { pub scored_moves: Vec<RootMoveScore>,
  pub nodes: u64, pub depth_reached: u8 }`.
- Add `Searcher::root_scored(&mut self, board, depth) -> Vec<RootMoveScore>`
  mirroring `root()` but:
  - Searches every root move with `(-INF, INF)` window (no alpha-beta pruning
    at root) so every returned score is exact.
  - Preserves the B-64 dead/near-dead placement penalty exactly as today.
  - Returns the full list; caller sorts.
- Add `pub fn iterative_deepening_scored(board, max_depth, time_limit_ms)`:
  - At each depth, call `root_scored`. Keep the last completed depth's result.
  - On abort mid-depth, return the previous completed depth's result.
  - Sort `scored_moves` descending by score before returning.

**File:** `native/nmm_core/src/lib.rs`

- Add `#[pyfunction] py_search_root_scored(...) -> (u64, u8, Vec<(Option<u8>, u8, Option<u8>, i64)>)`.
- Register in the `nmm_core` module.

**Test:** empty-board depth-3, assert 24 scored moves, all scores in a plausible range.

### M2 — Python: route Rust output through hint pipelines

**File:** `ai/game_ai.py`

- Rename `_choose_rust` → `_choose_rust_scored`. Signature accepts
  `recognition`, `trajectory_hints`, and `moves` (filtered list). Behaviour:
  1. Call `py_search_root_scored`. On any exception, log traceback + return None.
  2. Convert returned tuples into move-dict form with `.score`.
  3. **Filter** to intersection with passed-in `moves` list. Match by
     `(from, to, capture)` triple. Empty filter → return None.
  4. **Bonuses** via existing `_apply_opening_adjustments(scored, recognition, board)`
     and `_apply_trajectory_hints(scored, trajectory_hints)` (already operate on
     the `_score_all` shape — mostly wiring).
  5. Sort by adjusted score, pick top. Set `self.last_depth_reached`.
  6. Emit `R:OK ... adjusted={n_bonuses}` on terminal.

- Update the two `choose_move` call sites (early-game + main path) to always call
  `_choose_rust_scored(..., recognition, trajectory_hints, moves)` when `top_n == 1`.
- Remove `_use_rust` / `_python_hints_active` block + `P:HINTS-ACTIVE` print.

**Test:** `tests/test_rust_hint_integration.py`
- Mandatory-block position + filtered `moves` → block move returned.
- `recognition.book_move="d2"` + `opening_adherence=100` → `d2` returned.
- `trajectory_hints={"a1": 5000}` on a losing move → `a1` returned.
- Hint-free position → same move as current `py_search_stats` (score parity).

### M3 — Ordering hints from Python to Rust

No correctness change — internal alpha-beta pruning boost when the
TrajectoryDB winner-follow line is available.

**Rust:** extend `py_search_root_scored` with optional
`preferred_root: Vec<(Option<u8>, u8, Option<u8>)>`. `Searcher::root_scored`
promotes matching moves to the front (stable sort).

**Python:** in `_choose_rust_scored`, translate `self._trajectory_line[:3]`
notations to `(from, to, capture)` triples and pass through.

### M4 — Enable Rust for `top_n > 1` (self-play noise selection)

Rust now returns a scored ranked list — self-play noise selection can use it.

- `_choose_rust_scored` returns the sorted list on request.
- `choose_move` `top_n > 1` branch pulls top-N from Rust; falls back to
  `_iterative_deepen` on empty/error.

### M5 — Remove the hints-active guard + verification

Only after M2/M3 verified in-game.

- Delete `_hint_reasons` / `_python_hints_active` / `_use_rust` block.
- Delete `P:HINTS-ACTIVE` print (`R:OK` / `R:FAIL` stay).
- Verify a full game with `opening_adherence=100`, `chaos` personality, live
  `trajectory_db`: Rust-only path, book moves followed, blunders fire,
  trajectory-line moves preferred.

---

## Track B — Search strength (port Python search techniques to Rust)

Python's `_negamax` is smarter per-node than Rust's; that's why Python matches
Rust's tactical depth despite being ~8× slower per node. Porting these five
techniques to Rust would give it Python's tactical strength at Rust's speed —
worth another 2-3 plies at the same time budget.

### T-B1 — PVS (Principal Variation Search)

**File:** `native/nmm_core/src/search.rs::negamax`

- After the first (best-ordered) child, search subsequent children with a
  zero-width null window `(-alpha-1, -alpha)`. On fail-high, re-search with
  full `(-beta, -alpha)`.
- No behaviour change; strict node reduction. Combines with LMR below.

### T-B2 — LMR (Late Move Reduction) + aspiration windows

**LMR:** In `negamax`, after the first N moves (typically N=3) and depth ≥ 3,
search non-tactical moves (no capture, no mill formation, not extending) at
`depth - 1 - reduction` where reduction ~= `f(move_index, depth)`. Re-search at
full depth on fail-high. Estimated 30-40% node reduction.

**Aspiration windows:** In `iterative_deepening`, seed `alpha/beta` around the
previous depth's score `± MARGIN` instead of `-INF/INF`. On fail-low/high,
widen and re-search. Small gain (~5-10%), compounding with PVS.

### T-B3 — Killer moves + history heuristic in move ordering

**Killer moves:** add `killers: [[Option<Move>; 2]; MAX_PLY]` to `Searcher`.
On beta cutoff for a non-capture move, store it at `killers[ply]`. In
`ordered_moves`, promote killer moves after captures/mill-forming, before
the rest.

**History heuristic:** add `history: [[i32; 24]; 24]` indexed by
`(from_or_25, to)`. On beta cutoff, add `depth * depth` to the entry.
Use as a tiebreaker score in `ordered_moves`.

**Counter-move heuristic** (optional): remember, per opponent move, which
reply caused a cutoff last time; promote that reply.

Improves alpha-beta pruning by 20-30% cumulatively.

### T-B4 — Quiescence search

**File:** `native/nmm_core/src/search.rs`

- Add `qsearch(&mut self, board, alpha, beta) -> i64`. At `depth == 0`,
  instead of returning `evaluate_v2`, return `qsearch(alpha, beta)`.
- Qsearch flow: `stand_pat = evaluate_v2(...)`; if `stand_pat >= beta` return
  `beta` (stand-pat cutoff); tighten `alpha = max(alpha, stand_pat)`. Then
  search only capture / mill-forming moves recursively.
- Prevents horizon-effect tactical blunders where the AI stops right before
  a losing capture-exchange.

### T-B5 — Null-move pruning

**File:** `native/nmm_core/src/search.rs::negamax`

- Before the child loop, if depth ≥ 3, position is not in check-analog
  (opponent has no immediate mill threat that would blockade), and it's not
  fly phase (zugzwang risk):
  - Make a "null move" (swap side-to-move, no piece movement).
  - Search at `depth - 1 - R` (R=2 typical) with window `(-beta, -beta+1)`.
  - If the returned score ≥ beta, return beta (position too good to bother
    searching normally).
- Big node reduction in tactical positions, but guard against zugzwang or
  mate-search inaccuracy in near-terminal positions.

---

## Track C — Integration (Python-only features living inside the search)

### T-C1 — SE-11b: opponent-frequency depth extension

Python's `_negamax` extends depth by +1 at the first opponent ply for
high-frequency human moves (`trajectory_db.query_all_frequencies`). Rust has
no equivalent.

**Rust:** extend `py_search_root_scored` with
`opp_ext_moves: Vec<(Option<u8>, u8, Option<u8>)>`. Inside `negamax`, when at
`opp_ply_from_root == 1` and the move matches the list, add +1 to the child's
search depth. Cap total extensions per node at 1.

**Python:** query trajectory frequencies at the root (once per turn), select
moves above the frequency threshold, pass to Rust.

**SE-11c value-net re-ordering** stays Python-only (PyTorch). Python can
pre-order the first-opp-ply candidates with VN and pass them via
`preferred_root` (M3).

### T-C2 — FullGame DB mmap probe in Rust

Python's `_negamax` at line 1669 probes the fullgame DB at internal nodes
(gated by `_db_active_this_move`). Currently the key is generated in Rust
(`py_db_key`) but the read is Python.

**Rust:**
- Add `mmap2 = "0.9"` crate to `nmm_core/Cargo.toml`.
- New `db_probe::FullgameDbHandle` that mmaps the binary file at creation.
- Expose a construction path: `iterative_deepening_scored` accepts an optional
  `&FullgameDbHandle`. Path passed in from Python as bytes.
- Inside `negamax`, when a probe would fire, look up the WDL byte directly and
  bypass leaf eval on a hit.
- WDL byte format documented in `docs/HumanDB.md` and `endgame_db.py`.

**Python:** pass `_fullgame_db.path` (or an mmap handle) when constructing the
Rust search context.

Every DB hit inside the search skips a Python round-trip. Impact scales with
DB coverage in the search tree — could be very large in mid/endgame positions.

### T-C3 — Endgame DB mmap probe in Rust ✅ DONE

`EndgameSolvedDbHandle` PyO3 class holds `Arc<HashMap<(u8,u8), Mmap>>`;
opened once in `game_ai._choose_rust_scored`, passed to `py_search_root_scored`.
In `negamax`: guard on `white_placed >= 9 && black_placed >= 9` + piece range
3..=7, then direct O(1) combinatorial-rank lookup via `probe_endgame_solved`.
Returns `INF-depth` / `-(INF-depth)` / `0`.

### T-C4 — Persistent Rust TT across turns

Currently the Rust `Searcher` creates a fresh TT for every call. Python's
Ponder → live-search → next-turn chain could reuse cached entries.

**Rust:**
- Move `TranspositionTable` behind a module-level `Mutex<Arc<TranspositionTable>>`
  or expose PyO3 accessors to construct + hold a handle in Python.
- Design: Python holds a `TtHandle` PyO3 object; passes it to
  `py_search_root_scored`. Rust reuses it. On new-game, Python drops the
  handle and creates a new one.
- Locking overhead is only at TT access time; keep it small (per-entry mutex
  or lock-free with `AtomicU64` slots).

Warm TT between moves = free depth. Also enables ponder-to-live handoff
(B-94 equivalent, but native).

---

## Track D — Cleanup: wire Python to existing Rust primitives

### T-D1 — `_immediate_mill_threats` via Rust

Python `_immediate_mill_threats` in `ai/game_ai.py` scans 16 mills on every
turn. Rust already has `tactics::immediate_mill_threats` (exposed as
`py_immediate_threats`). Swap the Python impl to call the Rust function via
`native_core`; keep the Python fallback path.

Trivial diff, per-turn microseconds saved, but keeps the codebase consistent.

### T-D2 — Skip move-dict allocations in tight loops

Rust returns `(from_idx, to_idx, cap_idx)` tuples; Python wraps each in a
dict. Where callers iterate the result immediately (root filter, hint
application) they can operate on tuples directly and only build dicts for the
one move that ends up chosen. Small win, medium refactor churn.

---

---

## Track E — Multi-core parallelism (Lazy SMP)

Modern-engine standard: N threads run iterative deepening on the same root,
sharing one transposition table. Threads "help" each other via TT hits — one
thread's completed subtree becomes another's cache. No coordination beyond TT
reads/writes.

**Thread policy: default to `max(1, num_cpus / 2)`** — leaves headroom for the
OS, the Python process, background pondering, and thermal/battery margin on
laptops. User-configurable via `settings.json:"search_threads"` and a UI slider.

Depends on **T-C4 (persistent TT)** as a prerequisite — Lazy SMP needs an
atomic-friendly TT.

### T-E1 — Atomicize TT slots ✅ DONE

Xor-key trick (Stockfish-style): two `AtomicU64` per slot `(key^data, data)`.
Read verifies `xor_key ^ data == key`; torn reads fail the check = miss.
`TranspositionTable` is `Sync` via `AtomicU64`; shared across threads as `Arc<TT>`.

### T-E2 — Lazy SMP driver in Rust ✅ DONE (needs fix — see T-E2b)

`iterative_deepening_scored_smp` spawns `n_threads-1` helpers sharing `Arc<TT>`.
Current stagger `start_depth = 1 + (i % 2)` causes helpers to race main thread
in sync — zero TT warming benefit. Benchmarks: 2 threads = 0% speedup, 4 threads
= 15% slowdown (contention). Fix tracked as **T-E2b**.

### T-E2b — Fix Lazy SMP: mid-depth stagger ✅ DONE (Approach A)

**Root cause:** helpers and main all start at depth 1-2 and complete each ply in
roughly the same time, so TT entries arrive too late to benefit the main thread.

**Implemented:** `start_depth = ((max_depth * i) / n_threads).max(1)` in
`iterative_deepening_scored_smp`. For 4 threads at depth 12: helpers start at
depths 3, 6, 9 — pre-warming deep nodes before main arrives.

**Remaining optional improvements** (Approach B and shared stop flag):

#### Step 0 — Shared stop flag (optional)

Add `stop: Arc<AtomicBool>` to `Searcher`. In the `nodes & 2047 == 0` branches of
`negamax` and `qsearch`, check it:
```rust
if self.stop.load(Ordering::Relaxed) { self.aborted = true; return ABORT_SCORE; }
```
Thread through `new_searcher`. Lets main signal helpers to exit immediately after
time expires (avoids join blocking past the budget).

#### Approach A — Mid-depth stagger (implement first, low risk)

Replace helper `start_depth` calculation in `iterative_deepening_scored_smp`:
```rust
// Old:
let start_depth = 1u8 + (i % 2) as u8;
// New — spread helpers evenly across [1..max_depth]:
let start_depth = ((max_depth as usize * i) / n_threads).max(1) as u8;
```
For `n_threads=4, max_depth=12`: helpers start at depths 3, 6, 9. Helper 3 begins
near `max_depth`, pre-warming the deepest nodes before main arrives. Keep
`drop(helpers)` + let deadline bound them. Signal stop flag before returning so
helpers exit promptly instead of burning their remaining budget.

After implementing A, benchmark. If speedup is confirmed, ship. Proceed to B only
if A is insufficient.

#### Approach B — Root move partitioning (join-based, deeper change)

Divide root moves among threads. Each thread does full iterative deepening on its
assigned slice, writing TT entries. Main collects all thread results and assembles
the final ranked list.

1. **Add `root_scored_subset(&mut self, moves: &[Move], depth, alpha, beta)`** on
   `Searcher` — mirrors `root_scored` but iterates over a pre-computed move slice
   instead of calling `ordered_moves`. Identical B-64 penalty and extension logic.

2. **Restructure `iterative_deepening_scored_smp`**:
   - Compute `ordered_moves` once on the calling thread (canonical order).
   - Distribute moves round-robin to `n_threads` threads (not contiguous slices —
     round-robin balances strong/weak moves across threads).
   - Create shared `stop: Arc<AtomicBool>`.
   - Spawn `n_threads - 1` helpers; each runs `iterative_deepening` (all depths
     1..=max_depth) on its subset, writing to the shared TT.
   - Main thread runs `iterative_deepening_scored` on the full move list (TT hits
     from helpers give it free depth on helper-explored lines).
   - **After main completes**: set `stop = true`, then `join()` all helpers.
   - **Merge**: for each root move, take the score from whichever thread reached the
     highest `depth_reached`. Do NOT compare raw scores across different depths —
     only substitute helper's score if helper's `depth_reached > main's depth_reached`.

3. Return merged result sorted descending by adjusted score.

**Risk:** merge-depth hazard — a helper's shallower score for move X must not
override main's deeper score. The per-move depth check in step 2 mitigates this.

### T-E3 — Python: expose threads parameter ✅ DONE

`py_search_root_scored` accepts `threads: Option<usize>`. `game_ai.search_threads`
set from `settings.json:"search_threads"` via `_apply_search_depth`. Passed as
`threads=_threads` (None when 1 = single-threaded default).

### T-E4 — Parallel ponder on top-N predicted replies ✅ DONE

**File:** `ai/ponder.py` — `PonderManager` rewritten for N branches.
- `N_PONDER_BRANCHES = 2` (top-2 predicted opponent replies pondered in parallel)
- Each branch is a `_Branch` dataclass with its own `GameAI` + daemon thread
- `start()` predicts top-N moves with trajectory/fullgame/ngram scoring, spawns N threads
- `stop()` force-stops all branch AIs, joins all threads (0.5s each)
- `get_result(board)` checks all branches for hash match, returns first hit
- Tests in `tests/test_ponder.py` updated to use multi-branch API (`pm._branches[0]`)

**Realistic speedup (Lazy SMP alone, on top of Tracks A-D):**

| Machine | Threads | Speedup vs. 1-thread | Effective plies over baseline (11) |
|---|---|---|---|
| 4-core desktop | 2 | ~1.7× | +0.7 ply |
| 8-core desktop | 4 | ~2.5× | +1.2 plies |
| 16-core desktop | 8 | ~4× | +2 plies |

Stacked with Tracks A-D, an 8-core desktop lands at **~18-20 effective plies**
at the same time budget — approximately double today's tactical horizon.

**Risks:**
- **Nondeterminism.** Same position + same time budget can produce different
  moves across runs. Fine for a game AI; complicates reproducibility in tests.
  Mitigate by allowing `threads=1` for deterministic test runs.
- **Thermal / battery.** All allocated cores at 100% for the search budget.
  Half-cores default is the mitigation.
- **TT contention.** Atomicised TT slots are cheap under low contention but
  measurable at 16+ threads. Xor-key layout scales better than sharded mutex.
- **Debug complexity.** Threading bugs are harder to reproduce. Standard Rust
  `Send/Sync` catches most, but log per-thread stats (nodes, depth) so
  regressions are visible.

---

## Non-goals

- **Value-net inference in Rust.** PyTorch → ONNX → tract/candle is a big
  engineering effort for ~5-10ms per turn saved. Skip.
- **Sentinel advisor in Rust.** Same reason.
- **Opening book recognizer in Rust.** Per-turn only; recognition tree
  changes often; Python is a better home.
- **TrajectoryDB SQLite queries in Rust.** Per-turn; marshalling result rows
  back to Python eats the gain.
- **MCTS in Rust.** Rarely used, orthogonal to the negamax path.
- **Porting `_apply_opening_adjustments` / `_apply_trajectory_hints` /
  `tactical_move_bonus` / sentinel bans / dead-square filter into Rust.**
  All are per-turn root operations; Python is fine and keeps the semantics
  close to their authoritative data structures.

## Risks

- **Score-scale drift** (Track A). If Python and Rust `evaluate_v2` weights
  diverge, Python bonuses land on the wrong scale. Add a parity test:
  fixed positions get identical scores from both.
- **Root full-window cost** (M1). If profiling shows > 20% slowdown from
  disabling root alpha-beta, fall back to a two-pass approach: full
  alpha-beta first, then re-score non-best moves at a widened window
  centred on the best score ± max-bonus-size.
- **Time-budget starvation** (M1). Per-move exact scoring shallows iterative
  deepening by ~1 ply on the same budget. Track B (PVS/LMR/killers/history/
  quiescence/null-move) more than makes this back once landed.
- **Zugzwang null-move blindspot** (T-B5). Standard risk; mitigated by
  disabling null-move in fly phase and when the side to move has very few
  pieces or no non-mill moves.
- **TT staleness** (T-C4). Persistent TT across turns must invalidate on
  new-game and, for correctness, use full Zobrist keys (not truncated).
  Verify by parity test after multi-game runs.
- **DB file format churn** (T-C2/T-C3). If the binary layout ever changes,
  Rust and Python both need updating. Keep a single format-version constant
  and check it at handle-open time.

## Ordering summary

- Ship **Track A** first (M1 → M5). Correctness fix.
- Then **Track B**, in order T-B1 → T-B5. Each measurable in isolation via
  nodes/ply on the bench harness.
- Then **Track C**, in order T-C1 → T-C4. T-C4 (persistent TT) is a
  prerequisite for Track E.
- Then **Track E** (T-E1 → T-E4). Multi-core parallelism. Half-cores default.
- **Track D** is opportunistic — pick up when touching related files.

Expected cumulative headline: **11 plies today → 15-17 plies after A+B+C+D →
18-20 plies after Track E on an 8-core desktop.**

---

## HeuristicsV2 — Stage Completion Status

See `docs/HeuristicsV2-plan.md` for full spec.

| Stage | Description | Status |
|-------|-------------|--------|
| 0 | Scaffold `evaluate_v2` alongside v1; `use_v2_heuristics` flag | ✅ Done |
| 1 | Write `evaluate_v2()` in `ai/heuristics.py` | ✅ Done |
| 2 | Wire v2 at leaf in `_negamax`; suppress `tactical_move_bonus` in v2 mode | ✅ Done (guards added 2026-07-02) |
| 3 | TT upgrade to 2^21 with two-tier replacement | ✅ Done |
| 4 | Aspiration windows | ✅ Done (SE-7) |
| 5 | Null-move pruning | ✅ Done |
| 6 | LMR (late-move reductions) | ✅ Done (SE-6) |
| 7 | Broadened endgame DB probe (no piece-count cap) | ✅ Done |
| 8 | HumanDB trajectory depth 6 via `_opp_plies_budget` | ✅ Done |
| 9 | IID, SEE capture ordering, futility pruning, TT persist | ⏸ Deferred |
| 10 | Validation checklist + v1 code removal | ⏸ Pending |

### Rust `evaluate_v2` alignment gap

Rust `heuristics.rs::evaluate_v2` is missing vs Python v2:
- **Place:** pieces-in-hand diff (`_V2_PL_HAND`), REM proxy (`_V2_PL_REM`), positional value (`_V2_PL_POS`)
- **Move:** cycle-ready mills (×22), fork threats (×14), squeeze count (×30), domination bonus
- **Fly:** cycle-ready mills (×80), fork threats (×55), win-config bonus (×1190)

These are intentional simplifications for Rust speed. The Rust evaluator is the primary search; Python v2 is the fallback and reaches the same decisions via deeper ply. If alignment is needed, extend Rust `evaluate_v2` to include the missing terms.
