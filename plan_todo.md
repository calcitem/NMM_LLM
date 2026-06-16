# Nine Men's Morris — Active Backlog

*New items go here. When an item is completed, move it to `plan\_done.md`.*


## Implementation Roadmap

Track 1 (heuristic/phase-control), SE-1 through SE-9, SE-14, and the B-55–B-64 tactical cluster are complete (2026-05-28). **Active priorities:**

| Priority | Item | Key outcome |
| - | - | - |
| ★★★ | **B-86** ✅ | `\_closeable\_mills` in-mill exclusion bug → broken sealed-2-config detection |
| ★★★ | **B-87** ✅ | `setup\_mill\_bonus` fires for non-closeable 2-configs → `d2→d1` style blunders |
| ★★ | **B-88** ✅ | Vacate-threat penalty: 1-step lookahead for opponent stepping onto vacated square |
| ★★★ | **B-89** ✅ | Dead-placement B-64 fires on valid mill-contributor pieces (immobile but closeable 2-config) |
| ★★ | **B-90** ✅ | Feeder bonus fires for locked pieces; no cascade mill signal — spurious xe4 over xd5/xg7 |
| ★★★ | **B-82** ✅ | Mill-close suppressed by multi-threat filter (two bugs) |
|  | **B-83** ✅ | Fly-phase forked 2-config: AI should prefer d1→f6 style fork over d1→d7 |
|  | **B-84** ✅ | Mill assembly from 3 separate same-colour pieces on same ring |
| ★★ | **B-58** | Multiple LLM provider support (Claude, OpenAI, Perplexity, Null) |
|  | **SE-10** ✅ | Proactive own fork setup bonus (move phase) |
| ★ | **B-53** | ChromaDB embedding dimension mismatch when `ollama\_model` changes |
| ★ | **B-54** | Inject `phase\_strategy.md` into LLM prompts by game phase |
| ★ | **B-24** | GUI settings for FullGame / Endgame DB toggles and blend factor |
|  | **B-73** ✅ | Wire value network into negamax leaf evaluation |
|  | **B-74** ✅ | Cross-mill cycling fork: static eval bonus for two-mill pivot positions |
|  | **SE-11** ✅ | Opponent likelihood weighting + VN reordering (SE-11b/11c) via TrajectoryDB |
|  | **B-75** ✅ | Background pondering during opponent's turn |
|  | **B-51** | Build extended endgame WDL tables (code ready — run `--build-all`) |
|  | **SE-12** ✅ | Incremental evaluation cache (Zobrist-keyed) |
|  | **SE-13** ✅ | N-gram opponent move predictor |


### Recently completed (2026-05-28)

B-55, B-59, B-60, B-61, B-62, B-63, B-64, B-56 (D4 endgame symmetry), B-57 (mmap DB writes), SE-14 (FullGameDB inside `\_negamax`). Detailed specs remain below for reference; implementations are in `ai/heuristics.py`, `ai/game\_ai.py`, `tools/build\_endgame\_db.py`.


## Active Bug Fixes

### Bug B-82 — Mill-close suppressed by multi-threat filter ✅ 2026-06-02

**Observed**: Level-10 AI plays `a7→a4` (a blocking move) at move 11 in the game below instead of `g4→g7` (closes mill `a7-d7-g7` and captures a Black piece).

```
1.b2 f4 / 2.b6 b4 / 3.d6 f6 / 4.f2 d2 / 5.d7 d5 / 6.d3 e4  
7.g4 c5 / 8.e5 c4 / 9.a4 c3xd3 / 10.a4-a7 c3-d3 / 11.a7-a4 ← WRONG
```

**Root cause — two co-located bugs**:

1. **`\_immediate\_mill\_threats` returns 3 threats** (`\{a4, c3, d1\}`), which triggers the mandatory-block filter in `choose\_move`. The B-66 carveout (`threats.clear()` when White can close a mill) fires only when `len(threats) == 1`. With 3 threats, the carveout is skipped and `moves` is hard-restricted to blocking moves (`a7→a4`, `b6→a4`, etc.), excluding `g4→g7`.

2. **`\_pinned\_move\_squares` false positive for g4**: Mill `(g4-f4-e4)` has Black at f4 and e4, White at g4. The function pins g4 because "Black piece f4 is adjacent to g4". But f4 is **inside the mill** — it cannot slide to g4 and close the mill (moving f4 to g4 empties f4, leaving `(g4-f4-e4) = B-empty-B` — not a closed mill). No external Black piece is adjacent to g4. Even if Bug 1 is fixed (all moves allowed), Bug 2 would filter out `g4→g7` via the pin rule.

**Fix — implemented**:

- `\_pinned\_move\_squares`: add `mill\_set = set(mill)` and require adjacent Black piece `nb not in mill\_set` — so only external pieces (capable of sliding in and completing the mill) trigger the pin.

- Blocking filter in `choose\_move`: when `threats` is non-empty and own player is in move phase and can close a mill, add own mill-closing destinations to the `blocking` set (conservative: still forces AI to either block a threat or close own mill; search ranks which is best).

- Added 1-ply pin exemption for mill-closing moves (parallel to existing 2-ply exemption).

**Regression test**: `tests/test\_b82.py` — move-11 board asserts AI chooses a mill-closing move.


### Bug B-85 — Endgame DB WIN/LOSS counts violate NMM symmetry ⬜ ★★

**Symptom:** `tests/test\_build\_endgame\_db.py::TestSolver3v3Properties::test\_win\_equals\_loss\_by\_symmetry` fails:

```
AssertionError: 4464320 != 911296 : WIN and LOSS counts should be equal by NMM symmetry
```

Total table entries: 5,383,840. WIN = 4,464,320; LOSS = 911,296; DRAW = 8,224. By NMM symmetry (swapping White and Black sides), WIN and LOSS counts must be equal — every mover-wins position has a color-swapped mover-loses counterpart. The ~5:1 ratio indicates a systematic error in the offline retrograde solver.

**Root cause (unknown — investigation needed):**

1. **D4 fill pass interacts with colour-swap symmetry**: canonical form is lex-min over D4 transforms of `(w\_mask, b\_mask)`. Some D4 transforms swap which bitmask is "lower" but do NOT swap White/Black semantics. The fill pass copies canonical WDL to all equivalents, which is correct. However, if the canonical form of a position P is NOT also canonical for the colour-swapped position P' = (b\_mask, w\_mask, flipped\_turn), the LOSS entry for P' might be overwritten by a WIN entry during the fill pass.

2. **Propagation asymmetry**: a position is WIN if any successor is LOSS; LOSS if all successors are WIN. If the fly-move generation canonicalises successors inconsistently (e.g., resolves ties differently for W-to-move vs B-to-move positions), some LOSS entries may not be reachable.

3. **Terminal detection**: `\_closes\_mill(piece\_mask, to\_idx)` is used to detect immediate mill closures. If this function has a false negative for certain board orientations, some terminal WINs are missed and become UNKNOWN → DRAW after convergence.

**Likely fix:** Add a diagnostic pass in `tools/build\_endgame\_db.py` that counts how many (WIN, P) positions have their colour-swap P' = (b, w, 1-turn) also labelled WIN (expected: all of them should be WIN too, since WIN means the mover wins regardless of colour). Any P where P' is labelled LOSS or DRAW indicates a mislabelled canonical entry. Fix the canonical resolution to ensure colour-swap pairs are consistently handled.

**Impact:** The runtime `EndgameSolvedDB.query()` returns incorrect WDL for some 3v3 positions. Since B-48 (best-move selection) is also unresolved, the AI currently treats any WIN result as "try to win" without checking specific successors — the WDL mislabelling degrades quality of 3v3 fly play but does not cause crashes.

**Files:**

- `tools/build\_endgame\_db.py` — investigate and fix propagation / canonical fill logic

- `tests/test\_build\_endgame\_db.py` — existing symmetry test already captures the failure (pre-existing; not introduced by recent changes)


### Enhancement B-83 — Fly-phase forked 2-config preference ✅ 2026-06-03

**Observed**: In fly-vs-fly position (White: d6, f4, d1 / Black: c3, d2, a7, White to move), the AI plays `d1→d7` (one 2-config) instead of `d1→f6` (two 2-configs: closing at b6 AND f2 — an unblockable fork). Static eval correctly scores `d1→f6` at +2527 vs `d1→d7` at +1286. Bug may appear at specific search depths due to alpha-beta cutoff or ordering.

**Suggested investigation**: Run `\_root\_search` on this position at depth 3–7 and print per-depth best move. If the fork is only missed at certain depths, the issue is aspiration-window or cutoff related. If missed at all depths, there may be a fly-phase surplus calculation bug interacting with Black's responses.

**Note**: Static eval already includes `900 \* (own\_surplus - opp\_surplus)` for fly-vs-fly, which correctly flags d1→f6 as superior. The issue is search-depth dependent.


### Enhancement B-84 — Mill assembly from 3 separate same-colour pieces (cold convergence) ✅ 2026-06-03

**Observed**: The AI sometimes fails to converge 3 separate pieces (especially on the same ring) toward a mill, even when the path is straightforward and no tactical urgency exists. The existing `\_free\_piece\_assembly` / `\_assembly\_reach\_count` heuristics reward APPROACH but may under-reward direct ring-completion moves.

**Investigation needed**: Profile a specific game position where this occurs. Check whether the assembly heuristic weights are outbid by tactical bonuses or convergence-penalty terms. May require increasing move-phase `assembly\_reach\_count` weights or adding a "ring completion" bonus for owning 2 of 3 squares on any ring with the third reachable in 1 move.


### Bug B-86 — `\_closeable\_mills` does not exclude in-mill pieces ✅ 2026-06-03 ★★★

**Symptom:** `\_closeable\_mills(board, color)` counts a mill as closeable when the only "supporting" piece adjacent to the empty closing square is already inside the mill itself. Moving that piece to the closing square would vacate the mill, not close it. Meanwhile `\_mill\_threats` correctly excludes in-mill pieces using `if nb not in mill\_set`.

**Root cause:** `\_closeable\_mills` (line ~1575 in `ai/heuristics.py`) does not build `mill\_set = set(mill)` and therefore counts in-mill neighbors when checking reachability. `\_mill\_threats` has the correct guard.

**Downstream effects:**

- `\_sealed\_two\_configs(board, color)` uses `\_closeable\_mills(board, opp) \> 0` as an early exit guard. Because the guard fires erroneously (sees false-positive closeable mills for the opponent), `\_sealed\_two\_configs` returns 0 in positions where the opponent has no real closing threat. This means the P0.5 move-ordering tier (sealed-2-config-creating moves) is suppressed when it should fire.

- Call sites at lines ~1555, 2105, 2234, 2496: audit each to confirm the in-mill exclusion is needed; some tactical bonuses may have been implicitly calibrated around the buggy count — run self-play A/B before promoting.

**Fix:** In `\_closeable\_mills`, after `mill\_set = set(mill)`, filter: `any(board.positions\[nb\] == color for nb in ADJACENCY\[empty\] if nb not in mill\_set)`.

**Second test case (game 2026-06-03):** Black AI level 7 places c5 on move 9 (last placement) instead of a1.

```
1.d6 d2 / 2.f4 b4 / 3.f6 b6 / 4.f2xb6 b6 / 5.d3 a7  
6.c5 b2xc5 / 7.c4 d7 / 8.g7 g4 / 9.c3 c5 ← WRONG (should be a1)
```

After White places c3, White has c3+c4 in mill c3-c4-c5. The bug falsely flags this mill as closeable (c4 is adjacent to closing square c5, but c4 is inside the mill). Black's c5 placement gets P1 block priority and a `blocked` bonus. Better move: a1 (creates genuine 2-config a1-a4-a7, closeable via b4→a4, winning in one more move).

**Status:** Fixed 2026-06-03. `\_closeable\_mills` now builds `mill\_set = set(mill)` and excludes in-mill pieces from adjacency check (same pattern as `\_mill\_threats`). 88/88 tests pass.

**Files:** `ai/heuristics.py` — `\_closeable\_mills` function.


### Bug B-87 — `setup\_mill\_bonus` fires for non-closeable 2-configs in move phase ✅ 2026-06-03 ★★★

**Symptom:** In the game below, Black AI (level 7) plays `d2→d1` on move 10 instead of the better `c4→c3`.

```
1.d6 d2 / 2.f4 a7 / 3.f6 b6 / 4.f2xb6 b6 / 5.b4 d7  
6.g7 g1 / 7.e4 g4 / 8.a4 c4 / 9.e3 e5 / 10.e3-d3 d2-d1 ← WRONG
```

**Why `d2→d1` is bad:** vacates cardinal node d2, enabling White d3→d2 → b4→b2 forming mill b2-d2-f2 and capturing a Black piece.

**Root cause:** `d2→d1` creates 2-config g1-d1-a1 (Black has g1 and d1; a1 is the closing square). But a1's neighbors are a4=White and d1=Black(in-mill) — no reachable Black piece outside the mill can close to a1, so this 2-config is **not closeable**. Nevertheless `tactical\_move\_bonus()` computes `setup\_mill\_bonus = int(100 × 1.3) × 1 = +130` using `\_two\_configs` delta (which counts all 2-configs regardless of reachability). The negamax penalty for the resulting White mill combination is only ~–41 at depth 4, so net score for `d2→d1` ≈ +89 vs. ~+20 for `c4→c3` → AI incorrectly picks `d2→d1`.

**Fix:** In the move-phase branch of `setup\_mill\_bonus` (lines ~2139–2143), replace the `\_two\_configs` delta with a `\_closeable\_mills` delta: award setup\_mill\_bonus only when the move gains a **closeable** 2-config, not a structurally inert one.

```
\# Before (buggy):  
two\_cfg\_gained = max(0, \_two\_configs(after, color) - \_two\_configs(before, color))  
setup\_mill\_bonus = int(weights.setup\_mill \* 1.3) \* two\_cfg\_gained  
  
\# After (fixed):  
closeable\_gained = max(0, \_closeable\_mills(after, color) - \_closeable\_mills(before, color))  
setup\_mill\_bonus = int(weights.setup\_mill \* 1.3) \* closeable\_gained
```

Note: fix B-86 first (or apply its in-mill exclusion to `\_closeable\_mills`) so the count is accurate before wiring it here.

**Files:** `ai/heuristics.py` — `tactical\_move\_bonus()`, move-phase setup\_mill\_bonus block.


### Enhancement B-88 — Vacate-threat penalty (1-step consolidation lookahead) ✅ 2026-06-03 ★★


### Bug B-89 — B-64 dead-placement penalty fires on valid mill-contributor pieces ✅ 2026-06-03 ★★★

**Symptom:** In the game below, Black AI level 7 places e5 (last placement) instead of g1.

```
1.d6 d2 / 2.f4 d7 / 3.f6 b6 / 4.f2xb6 b6 / 5.b4 d5  
6.d1 a4 / 7.e3 g4 / 8.a7 c3 / 9.c5 e5 ← WRONG (should be g1)
```

After White places c5, Black placing g1 would create closeable 2-config g7-g4-g1 (closeable by moving d7→g7 in move phase), enabling the mill and a cycling fork (g1↔d1 oscillating with d1-d2-d3 and g7-g4-g1). Black places e5 instead, which creates a dead pattern (c5-d5-e5 with White c5).

**Root cause:** B-64 dead/near-dead penalty fires when `free\_nb == 0`. g1's neighbors are g4=Black and d1=White — both occupied → `free\_nb = 0` → penalty = -1500. But g1 creates a closeable 2-config (g7-g4-g1, closeable via d7→g7 where d7 is Black and not in the mill). The piece doesn't need to move itself — another Black piece closes the mill. B-64 incorrectly treats "immobile piece" as "dead piece."

**Fix:** In B-64 block, add `\_creates\_closeable = \_closeable\_mills(after, color) \> \_closeable\_mills(before, color)`. Exempt the penalty when either `\_is\_pivot\_blocker` OR `\_creates\_closeable`. After fix: g1 scores +47 (setup bonus, no dead penalty); e5 scores -178 (disrupted 2-config penalty).

**Status:** Fixed 2026-06-03. Also: LMR test `test\_lmr\_block\_guard\_not\_reduced` updated — the old position (W: a7/a4/c5) had White winning immediately by closing a1-a4-a7 and reducing Black to 2 pieces; test updated to a position where blocking is genuinely best (W: b6/g7/d3).

**Files:** `ai/heuristics.py` — B-64 block in `tactical\_move\_bonus()`. `tests/test\_search\_enhancements.py` — LMR test position updated.

**Motivation:** `consolidation\_penalty` currently checks: after my move, does the opponent's 2-config count increase immediately? For `d2→d1` this fires 0 — White needs to move d3→d2 first, so the new 2-config doesn't appear until the next ply. There is no signal for "vacating square X lets an adjacent opponent piece step onto X and form a closeable 2-config."

**Logic:** In `tactical\_move\_bonus()`, when a move vacates `from\_sq`:

1. For each opponent piece `opp\_piece` adjacent to `from\_sq`:

   - Simulate `opp\_piece → from\_sq`

   - Check if the resulting board has a new closeable White 2-config that passed through `from\_sq`

   - If yes, apply penalty ~`weights.consolidation\_penalty \* 3` (severity is higher than the standard 1-step check because the vacate enables an immediate forced response)

**Effect for `d2→d1`:** d3 is adjacent to d2; simulating d3→d2 creates closeable b2-d2-f2; penalty fires ~–300. Combined with B-87 dropping setup\_mill\_bonus to 0, net score for `d2→d1` becomes ≈ –341 vs. +20 for `c4→c3` — unambiguous.

**Files:** `ai/heuristics.py` — `tactical\_move\_bonus()`, near `consolidation\_penalty\_val`.


### Enhancement B-90 — Cascade mill perception: feeder guard + cascade bonus ✅ 2026-06-03

**Symptom:** In the game below, Black AI (post-B-82 position) closes g1-d1-a1 and captures e4 instead of the better xd5 or xg7 which enable cascade mills.

```
\[Black: f6, g1, d1, a1 closed on this turn / White: c5, e4, f4 with mills e4-f4-g4 and f2-f4-f6\]  
Cascade plan: xd5 → Black can close d3-d2-d1 immediately (d5 was the only blocker)  
              xg7 → Black can close g1-g4-g7 immediately (d7→g7 gives 2-config g1-g4-g7)  
              xe4 (wrong): e4 is LOCKED (all neighbors occupied) — feeder bonus was spuriously applied
```

**Root cause — two issues:**

1. **Spurious `capture\_feeder\_bonus` for locked pieces:** `capture\_feeder\_bonus` (+300) fired for xe4 because e4 is adjacent to the closed mill f6-f4-f2 (via f4). But e4 was LOCKED — all neighbors occupied (e5=Black, e3=Black, f4=White), so it could never have "fed" f4 into a mill. A locked piece cannot move, so it has no feeder value. This inflated xe4's tactical score by +300.

2. **No cascade mill signal:** xd5 and xg7 each enable an immediate own-mill closure on the very next move (the capture removes the sole blocker of a closeable own 2-config). This was worth +0 tactically — the cascade value only appears at depth ≥2 in negamax, and may be cut by alpha-beta before being counted.

**Fix — two-part:**

1. **Feeder guard:** Before the feeder detection loop, compute `\_cap\_free\_nb = sum(1 for nb in ADJACENCY\[captured\_pos\] if before.positions\[nb\] == "")`. If `\_cap\_free\_nb == 0` (piece was locked), skip feeder bonus entirely.

2. **Cascade mill bonus** (`+weights.close\_mill // 2 ≈ +250`): After a mill-closing capture, check each mill containing `captured\_pos`. If the post-capture board has own 2 + 1 empty in that mill AND the empty square is reachable (phase-aware), award the cascade bonus.

**Diagnostic results (after fix):**

```
xd5: combined = +1606  ← WINS (cascade bonus fires, d3-d2-d1 closeable)  
xg7: combined = +1308  (cascade bonus fires, g1-g4-g7 closeable)  
xe4: combined = +1123  (feeder bonus removed — was +1423)
```

**Status:** Fixed 2026-06-03.

**Files:** `ai/heuristics.py` — `tactical\_move\_bonus()`: `capture\_feeder\_bonus` block (locked guard), new `cascade\_mill\_bonus` block after `safe\_capture\_bonus`.


## DB / Infrastructure

### Bug B-26 — FullGameDB is never loaded by the server ✅ 2026-05-26

*(Archived — see plan\_done.md)*

### Enhancement B-23 — Endgame position database builder ✅ 2026-05-26

*(Archived — see plan\_done.md)*

### Enhancement B-27 — Make binary format the default fullgame DB output ✅ 2026-05-26

*(Archived — see plan\_done.md)*

### Enhancement B-52 — FullGameDB: Frequency-Weighted Build from Human-Played Games ✅ 2026-05-26

*(Archived — see plan\_done.md)*

### Enhancement B-24 — GUI settings for position DB usage ⬜

**Goal:** Add controls to the Settings and AI Tuning panels so the player can see which position databases are active and tune how strongly they influence the AI's play.

**Proposed controls (Settings panel or AI Tuning panel):**

| Control | Type | Description |
| - | - | - |
| Use FullGame DB | Checkbox | Enable/disable `data/fullgame.sqlite` lookup (greyed out if file absent) |
| Use Endgame DB | Checkbox | Enable/disable `data/endgame\_solved.sqlite` lookup (greyed out if absent) |
| DB influence | Slider 0–100 % | How much a DB result overrides the heuristic score (0 = heuristic only, 100 = DB always wins) |
| DB status line | Read-only | Shows e.g. "FullGame: 500K positions · Endgame: 13M positions (complete ≤8)" or "No DBs found" |


**Behaviour:**

- If both DBs are enabled and a position exists in both, the endgame DB takes priority (it is exact)

- DB influence slider feeds into `ai/fullgame\_db.py`'s `score\_delta()` blend factor

- Checkbox state is persisted to `data/settings.json` alongside other AI settings

- DB file presence is checked at server start; the UI greys out absent DBs automatically

**Files:**

- `web/templates/index.html` — new controls in Settings or AI Tuning panel

- `web/static/game.js` — load/save DB toggle state; send with game start message

- `web/static/style.css` — DB status line styling

- `web/app.py` — expose `/api/db\_status` endpoint; pass DB toggle flags to `GameAI`

- `ai/game\_ai.py` / `ai/fullgame\_db.py` — honour the toggle and blend factor at runtime


### Enhancement SE-14 — DB-Guided Horizon Search ✅ 2026-05-26

*(Archived — see plan\_done.md)*

### Enhancement B-25 — Tools management page ✅ 2026-05-27

*(Archived — see plan\_done.md)*


## Bug Reports

### Bug B-53 — ChromaDB embedding dimension mismatch when ollama\_model changes ⬜

**Symptom:** `Error: Collection expecting embedding with dimension of 4096, got 2048`. Occurs when `ollama\_model` in settings.json is changed (e.g. from `llama3.1:8b` → `gemma:2b`).

**Root cause:** `MemoryManager` uses the main LLM model for embeddings. When the user switches models, the embedding dimensionality changes but the existing ChromaDB collections still expect the old dimensions.

**Recommended fix:** Add `ollama\_embed\_model` to settings.json (default `nomic-embed-text`). `MemoryManager` always uses this fixed model for embeddings, independent of the main LLM.

**Files:**

- `data/settings.json` — add `ollama\_embed\_model` key (optional, default `nomic-embed-text`)

- `ai/memory\_manager.py` — use `ollama\_embed\_model` for embeddings instead of `ollama\_model`

- `web/app.py` — pass `ollama\_embed\_model` setting to `MemoryManager`


### Bug B-54 — LLM phase strategy guide never fed to MillsLLM ⬜

**Symptom:** `data/phase\_strategy.md` exists (179 lines, phase-segmented NMM tactics guide) but is never injected into the LLM prompt.

**Fix:** In `Coordinator`, detect the current game phase (placement / move / fly) and inject the relevant section(s) from `phase\_strategy.md` into the system prompt. The file is already segmented by phase (Phase A = placement 1–6, Phase B = placement 7–9, Phase C = movement, Phase D = fly).

**Files:**

- `ai/coordinator.py` — load `phase\_strategy.md` once at init; add `\_get\_phase\_context(board)` helper

- `ai/mills\_llm.py` — accept optional `phase\_context: str` parameter and prepend to system prompt


### Enhancement B-65 — Wire opening guidance + trajectory hints into no-LLM path ✅ 2026-05-29 ★★

*(Implemented: `ai/move\_guidance.py` shared helpers; `\_nollm\_choose\_move()` in `web/app.py`; coordinator + `mills\_llm` trajectory context string.)*

**Context:** Session interrupted (credits). Implementation was in progress. User confirmed the fix is needed for both no-LLM and coordinator/LLM paths. The learning infrastructure (277 games, 2107 trajectory entries, 21 novel openings) is populated correctly but never consulted during no-LLM move selection.

**Root cause:** `\_ai\_turn()` in `web/app.py` (line ~1181) calls `choose\_move(board)` with no `trajectory\_hints`, no `recognition`, and no `force\_book\_early` in the no-LLM branch. The coordinator path does all of this but was also reportedly not working during LLM games — verify trajectory hint plumbing shifts the chosen move before closing.

**Fix (no-LLM branch in `\_ai\_turn()`):**

1. Add `\_target\_opening: Optional\[Opening\]` to `Session.\_\_init\_\_`

2. At game start (line ~1475): after `session.opening\_recognizer = OpeningRecognizer(...)`, call `\_nollm\_book.select\_opening(ai\_color=game\_ai.color)` and store on session; validate `side in (ai\_color, "both")`

3. In `\_ai\_turn()` else-branch: query `\_trajectory\_db` using `session.engine.game\_record\["moves"\]` notations (both `query()` and `query\_opponent\_loss()` with `loss\_exploit` weight); synthesise `RecognitionResult` from `\_target\_opening` when recognition is inactive/novel in place phase (check `book\_move in legal\_dests` first); compute `force\_book\_early` (`ai\_placements \< 2` and phase == "place"); pass all to `choose\_move`

4. Update `session.opening\_recognizer` after AI move (currently only updated for human moves at line 1841)

5. Verify: `choose\_move(board, trajectory\_hints=\{"c4": 0.5\})` must prefer c4 over baseline call — if not, bug is inside `\_apply\_trajectory\_hints`

**Imports needed:**

- `RecognitionResult, INACTIVE\_RESULT` from `ai.opening\_recognizer` (add to existing import)

- No new imports for `get\_game\_phase` (already imported) or `get\_all\_legal\_moves` (already imported)

**Advisor notes (pre-implementation):**

- Check `book\_move in legal\_dests` before synthesising RecognitionResult (missing in initial draft)

- `ply = len(notations)` — do not double-assign from placement count

- Optionally extract shared helper `\_build\_ai\_guidance(board, moves, target, recognizer, trajectory\_db, ai\_color)` shared by both coordinator and no-LLM paths to avoid future drift

**LLM / external AI integration (forward-compatible design):**

The trajectory hints currently reach `choose\_move()` as a numeric dict but are never shown to the LLM opinion layer. To keep B-58 (Claude/OpenAI/external providers) unblocked:

- Format the trajectory query result as a human-readable context string, e.g.:

- ```
"Trajectory DB (depth 6, 14 games): c4 +0.50, g4 +0.50, d5 +0.50, d7 -0.10"
```

- Pass this string alongside the trajectory hints dict into `Coordinator.deliberate()` so `mills\_llm.ask\_for\_move\_opinion()` (or any future external AI call) receives it as part of its prompt context — it already receives `move\_history` and `recognition`; trajectory context slots in the same way

- In the no-LLM path, surface it in the server log so it is visible during debugging

- When B-58 lands and an external AI (Claude, OpenAI etc.) is wired up, it should receive this context string so it can factor historical win-rate data into its move suggestion — the game AI's numeric hint already nudges search, but the LLM can reason about *why* certain lines win more often and offer a higher-level suggestion that the engine then validates

**Coordinator change (small, can be done now or with B-58):** In `deliberate()`, after building `trajectory\_hints`, produce `\_traj\_context\_str` and pass it to `ask\_for\_move\_opinion(trajectory\_context=\_traj\_context\_str)`. `MillsLLM.ask\_for\_move\_opinion()` currently ignores unknown kwargs — add the parameter and append the string to the system prompt when non-empty.

**Files:**

- `web/app.py` — Session class, game-start block, `\_ai\_turn()`

- `ai/coordinator.py` — build `\_traj\_context\_str`; pass to LLM; optional shared helper

- `ai/mills\_llm.py` — accept `trajectory\_context` in `ask\_for\_move\_opinion()`; inject into prompt


### Bug B-66 — Black plays passive block instead of closing own mill (move-phase) ✅ 2026-05-29 ★★

**Game:** 1.d2 d6 / 2.d7 g4 / 3.d1 d3 / 4.b4 a1 / 5.a4 c4 / 6.f6 g1 / 7.g7 a7 / 8.f4 f2 / 9.d5 c5 / 10.d2-b2 **d3-c3×** (was d6-b6)

**Symptom (reported):** Black played d6-b6 instead of closing a mill.

**Root cause (investigation):** `d3-d2` does **not** close a mill on this board. The correct mill is **c3-c4-c5** via `d3-c3` (+ capture). Tactical scoring already preferred `d3-c3` (+865). `choose\_move` picked `d6-b6` because `\_immediate\_mill\_threats()` returned `\{b6\}` (White threatens b2-b4-b6), and mandatory-block filtering restricted candidates to moves landing on `b6` only. The placement-phase carveout (“close own mill instead of block”) did not apply in move phase.

**Fix:** Extend the single-threat carveout to move phase in `\_immediate\_mill\_threats()`: when exactly one opponent closing square is threatened and STM can close an own mill, clear the threat set so `choose\_move` may close with capture.

**Files changed:**

- `ai/game\_ai.py` — `\_stm\_can\_close\_mill()`, move-phase carveout in `\_immediate\_mill\_threats()`

- `tests/test\_b66.py` — regression replay + `choose\_move` assertion

- `tests/test\_blocking.py` — move-phase carveout unit test


### Bug B-64 — AI places pieces with 0 or 1 free neighbours (dead/near-dead placement) ✅ 2026-05-28

*(Implemented: `dead\_placement\_penalty` / `near\_dead\_placement\_penalty` in `HeuristicWeights` + `tactical\_move\_bonus()`.)*

**Symptom:** White AI (balanced personality) places at b2 (1 free neighbour) on turn 6 and d1 (0 free neighbours) on turn 8, yielding a piece permanently trapped from birth. No penalty exists for creating a piece with no future movement options, so the tactical score prefers positionally strong but mobility-dead squares.

**Root cause:** `tactical\_move\_bonus()` has no term that penalises placing a piece onto a square where it will have zero or near-zero free adjacent squares after placement. `\_NEAR\_BLOCKED\_WEIGHTS\["place"\] = 0` means the static evaluator also ignores this.

**Fix:**

- Add `dead\_placement\_penalty: int = 600` and `near\_dead\_placement\_penalty: int = 150` to `HeuristicWeights`.

- In `tactical\_move\_bonus()`, when `\_is\_placement and mills\_delta == 0`, find the placed square, count `free\_nb = sum(1 for nb in ADJACENCY\[sq\] if after.positions\[nb\] == "")`. Apply:

  - `free\_nb == 0` → penalty = `dead\_placement\_penalty` (piece is permanently immobile)

  - `free\_nb == 1` → penalty = `near\_dead\_placement\_penalty \* (placement\_index / 8)` (scales with how far into placement we are — early game has fewer pieces and more mobility options later)

- Skip penalty when `mills\_delta \> 0` (piece is in a just-closed mill — it has value regardless of mobility).

- Add entry `("Dead/near-dead placement (B-64)", -placement\_mobility\_penalty)` to `\_contributions`.

- Update all personality JSONs to include new fields.

**Files:**

- `ai/heuristics.py` — `HeuristicWeights`, `tactical\_move\_bonus()`

- `data/personalities/\*.json` — add `dead\_placement\_penalty`, `near\_dead\_placement\_penalty`

- `tests/test\_tactics.py` — regression: at placement 8, d5 scores higher than d1; at placement 6, d2 scores higher than b2


### Bug B-55 — AI allows opponent to build two interconnected cardinal ring mills ✅ 2026-05-28 ★

*(Implemented: `\_dual\_connected\_mill\_alert()` + dual-connected block bonus in `ai/heuristics.py`; P1 ordering in `ai/game\_ai.py`.)*

**Symptom:** The AI (Black) fails to block White from establishing two cardinal mills through the middle ring in the same game. Once White has two such mills, Black is in a near-losing position because White can oscillate both independently.

**Game example 1:**

```
1.d6 d2  
2.f4 b4  
3.f6 d7  
4.f2xd7 d7
```

At turn 4, White plays f2xd7 (closes f2-f4-f6 mill, captures d7). Black re-places at d7 instead of b6. White is set up for b6-d6-f6. Black must place at b6 to block.

**Game example 2:**

```
1.d6 d2  
2.f4 b4  
3.f6 f2  
4.b6xf2 g7
```

White plays b6xf2 (closes b6-d6-f6, captures f2). Black plays g7 instead of f2. White can now set up f2-f4-f6.

**Pattern:** White gains two middle-ring cardinal mills sharing the f6 corner, giving White a highly mobile dual-mill oscillation structure.

**Fix:**

- Add `\_dual\_cardinal\_mill\_alert(board, opp\_color)` in `ai/heuristics.py`: returns True if opponent has 1 closed mill AND a 2-config in a second mill sharing a square with the first.

- Apply a block-bonus (~400) to any move that prevents the second such mill from forming.

- Urgency: equivalent to blocking a direct mill closure (P1 priority in `\_order\_moves`).

**Files:**

- `ai/heuristics.py` — `\_dual\_cardinal\_mill\_alert()`, apply in `tactical\_move\_bonus()`

- `tests/` — regression tests for both game sequences


### Bug B-67 — Copy button omits placement moves for setup-position games ⬜

*(Renamed from B-56 to avoid collision with D4 endgame symmetry B-56.)*

**Symptom:** When using the "Copy" button to export a game position, the copied output only includes move-phase moves, not the placement moves that led to the current position.

**Fix:** Ensure the copy/export function includes all placement moves in the notation output, followed by movement phase moves.

**Files:**

- `web/static/game.js` — copy button handler: include placement moves in exported notation

- `web/app.py` — `/api/copy\_game` or equivalent endpoint: return full game history


### Bug B-68 — Opening book bonus overrides B-64 dead-placement penalty ✅ 2026-05-31

**Symptom (game 1):** In an AI-vs-AI game (both Scholar personality), White places at g7 on turn 8 despite g7 having 0–1 free neighbours. The B-64 penalty (-1500 or -400) is overwhelmed by the opening-book bonus applied in `\_apply\_opening\_adjustments()`:

- Scholar adherence = 75 → `book\_bonus\_abs = int(3000 \* 75/100) = 2250`

- Net for dead square with book bonus: -1500 + 2250 = +750 (net positive; beats other moves)

- Also possible: blunder penalties on other moves widen the gap further

**Symptom (game 2):** Forced block at a dead square (human vs balanced AI, turn 7: g7). Opponent threatens a7-d7-g7; g7 is the only block AND has 0 free neighbours. The AI correctly blocks but `\_populate\_thinking` labels the chosen move "Dead/near-dead placement (B-64)" — misleading, since the move was mandatory, not a strategic error.

**Root cause (game 1):** `\_apply\_opening\_adjustments()` adds `book\_bonus\_abs` unconditionally, even when the book-recommended square is a dead/near-dead placement. B-64's penalty was sized to beat tactical noise but not opening-book bonuses.

**Fix options (pick one or both):**

1. **Skip book bonus for dead squares (recommended):** In `\_apply\_opening\_adjustments()`, before applying `book\_bonus\_abs`, check whether the destination is a dead/near-dead placement (0 or 1 free neighbours after placing). If so, skip (or halve) the book bonus so B-64 can override. Only applies during placement phase.

2. **Scale B-64 penalty above book bonus:** Raise `dead\_placement\_penalty` from 1500 → 3500 and `near\_dead\_placement\_penalty` from 400 → 2500 so they exceed even 100%-adherence book bonuses (3000). Risk: may over-penalise correct early placement on slightly constrained squares.

**Fix (game 2 label):** In `\_populate\_thinking()` (in `ai/game\_ai.py`), detect when the chosen move is a mandatory block (`is\_forced\_block`): when the chosen move's `to` square was in the `\_immediate\_mill\_threats()` set, label it "Forced block (dead square — unavoidable)" instead of "Dead/near-dead placement (B-64)".

**Files:**

- `ai/game\_ai.py` — `\_apply\_opening\_adjustments()`: skip book bonus when dest is dead during placement; `\_populate\_thinking()`: detect forced block → update label

- `ai/heuristics.py` — optionally raise penalty constants if fix-option 2 chosen


### Bug B-69 — Deep search overrides B-64 dead-placement penalty ✅ 2026-05-31

**Symptom:** AI (difficulty 4, 6-second budget) still plays to squares with 0 free neighbours during placement. Confirmed live in two games: Black plays to a4 (a1=B, a7=W, b4=W → 0 free) and b6 (b4=W, d6=W → 0 free). B-64 root penalty (-1500) is overcome by iterative deepening reaching depth 7-8+ where convergence-disruption bonuses (+325) and chain-disruption (+240) offset the penalty.

**Root cause:** B-64 is a root-only tactical bonus applied once in `\_score\_all`. The negamax search at deeper plies evaluates the resulting position without "remembering" the permanent immobility. Horizon effect: at depth 7-8, b6 appears favorable because it disrupts White's convergence cluster.

**Fix:** Hard-filter dead placements from the move list BEFORE any search — same pattern as mandatory-block filter (lines 509-513). Add `\_is\_dead\_placement(board, move)` module-level helper that returns True when a placement has 0 free neighbours and doesn't close a mill. Apply the filter during placement phase only; preserve mill-closing exemption.

**Safety fallback:** `if non\_dead: moves = non\_dead` — if ALL remaining moves are dead (e.g., the only mandatory block is a dead square), the filter no-ops.

**Files:**

- `ai/game\_ai.py` — add `\_is\_dead\_placement()` helper; add filter block after mandatory-block in `choose\_move`

- `tests/test\_b69.py` — 7 regression tests (unit tests for helper + integration tests for the a4-game and b6-last-piece positions)


### Bug B-70 — AI vacates sole blocker of opponent 2-config (movement-phase pin) ✅ 2026-05-31

**Symptom:** White (balanced, difficulty 4) plays a4→a1 on move 11, vacating the sole blocker of Black's a4-b4-c4 2-config. Black immediately slides a7→a4, closes the mill, and captures b6. Creates a double cycling mill at lines 4 and 6.

**Game record:** 1.d1 d6 / 2.f2 b4 / 3.f4 f6 / 4.b6 d3 / 5.d5 g4 / 6.a4 a7 / 7.g7 e5 / 8.e3 c5 / 9.d2 c4 / 10.d2-b2 d3-d2 / **11.a4-a1** ← failing move

**Root cause:** `\_immediate\_mill\_threats()` only flags EMPTY squares where the opponent can slide to close a mill. When White's piece IS at the blocking square (a4=W), the function returns \[\] — the threat is latent (only materialises after White vacates). Mandatory-block filter never fires. The deep search (difficulty 4, depth 7-8+) finds a speculative counter-attack (a1-d1-g1) that evaluates falsely positive at the leaf node (horizon effect).

**Fix:** Add `\_pinned\_move\_squares(board, color)` — analogous to `\_pinned\_fly\_squares` but with an additional adjacency check: opponent must have a piece adjacent to the pinned square to slide in immediately (move-phase only; fly phase can jump from anywhere). Apply as a hard filter in `choose\_move` after the fly-phase pin rule.

**Files:**

- `ai/game\_ai.py` — `\_pinned\_move\_squares()` helper (lines ~109-126); movement-phase pin filter in `choose\_move` (after fly-phase pin block)

- `tests/test\_b70.py` — 13 regression tests (unit tests for helper; integration tests for game-record position and minimal 3-piece case)


### Bug B-71 — AI captures suboptimal opponent piece after closing a mill ✅ 2026-05-31

**Symptom:** Black AI closes a mill and removes White's f4 instead of f2. White can re-form a mill in 2 moves via g4-f4 + d6-f6, meaning Black's mill advantage is quickly neutralised. Removing f2 instead would prevent White from doing this before Black can open its own mill.

**Game record:** 1.d6 d2 / 2.c4 b4 / 3.g4 d5 / 4.e3 d1 / 5.d3 c3 / 6.f4 e4 / 7.f6 b6 / 8.f2xb6 b6 / 9.b2 a7 / 10.d6-d7 b4-a4 / 11.f6-d6 d1-a1×f4 ← should capture f2

**Root cause:** Capture selection (`\_best\_capture()` in `ai/game\_ai.py`) evaluates captured pieces by their positional value but does not model the opponent's "time to re-form a mill" after the capture. A piece adjacent to two existing 2-configs is harder to replace; a piece with both adjacent mill slots occupied by own pieces is nearly irreplaceable.

**Proposed fix:** In `\_best\_capture()`, add a "mill re-formation speed" heuristic: for each capturable opponent piece, count the minimum moves needed for the opponent to replace that piece in a mill (e.g., how many pieces adjacent to that square, how many 2-configs the opponent has that the piece is NOT part of). Prefer captures that maximise this re-formation time. This is a scoring delta on top of the existing positional-value score.

**Files:**

- `ai/game\_ai.py` — `\_best\_capture()`: add mill re-formation delay heuristic

- `tests/` — regression test: in the f4/f2 position, `\_best\_capture()` must prefer f2 over f4


### Enhancement B-72 — Pure AI button: disable personality weight customisations ✅ 2026-05-31

**Request:** A GUI button that resets all personality sliders and weight adjustments to the bare evolved weights (best.json), with no user customisations on top. Useful for diagnosing whether a bug is caused by personality overlays or the base AI.

**Context:** "balanced" personality already maps to empty `\{\}` overlay → base evolved weights. The button would clear any slider-adjusted `ai\_weights` in `data/settings.json` back to `\{\}`, equivalent to selecting "balanced" with all sliders at default.

**Proposed implementation:**

- Add a "Pure AI" button in the AI Tuning panel (web UI)

- On click: POST `/api/reset\_weights` which sets `ai\_weights: \{\}` in settings.json and reloads

- Or: add a "Reset to defaults" action to the existing personality selector

**Files:**

- `web/templates/index.html` — add Pure AI / Reset button in AI Tuning panel

- `web/static/game.js` — click handler: POST reset; reload slider display

- `web/app.py` — `/api/reset\_weights` endpoint


### Bug B-21 — Windows installer: improve model pull failure guidance ⬜

**Symptom:** After a failed `ollama pull`, the only feedback is a terse warning with no alternatives or guidance about how to change the model.

**Fix — `install.ps1`:** After a failed pull, print a help block listing lighter alternatives and instructions for updating `data/settings.json`. In the "Installation complete!" banner, if the model was not pulled, repeat the short version.

**Files:**

- `install.ps1` — step 8 failure block + completion banner

- `install.bat` — mirror the same guidance if applicable


### Bug B-17 — GUI text contrast too dim ⬜

**Symptom:** Many GUI labels, board coordinates, and control text are hard to read. `--text-dim: \#8a7a60` is used widely.

**Fix:**

- `web/static/style.css` — raise `--text-dim` to approximately `\#b7a78c`, or split into `--text-muted` (decorative) and `--text-label` (functional).

- Increase board coordinate / grid label contrast.

- Audit all `var(--text-dim)` uses and promote critical gameplay labels to `var(--text)` or the new `--text-label`.

**Files:**

- `web/static/style.css`

- `web/static/board.js` if board coordinate text is rendered separately


### Enhancement B-18 — Remove Bad Move button; add Force Move button for AI ⬜

**Goal:** Remove the Bad Move button and all related code. Replace with a **Force Move** button that lets the human player specify the next AI move.

**Bad Move removal scope:**

- `web/static/game.js`, `web/app.py`, `web/templates/index.html`, `web/static/style.css`

- `ai/game\_ai.py` — remove bad\_moves avoidance logic

- `data/bad\_moves.json` — delete file

**Force Move button spec:**

- Visible only when it is the AI's turn

- Opens a modal: "Enter square to move to (and from, if move phase)"

- Validates against `get\_all\_legal\_moves(board)`; rejects illegal moves

- Sends to server as override via `/api/force\_ai\_move`

**Files:**

- `web/app.py` — new `/api/force\_ai\_move` endpoint

- `web/static/game.js` — Force Move button + modal

- `web/templates/index.html` — Force Move button element



### Bug B-31 — Opening play should still be recorded when the AI resigns ⬜

**Symptom:** Opening sequence is not being recorded properly when the AI resigns.

**Fix:**

- `web/app.py` — verify the resignation path persists the game record and opening line before any early return.

- `ai/opening\_book.py` / training pipeline — ensure resignation games still contribute opening statistics.

- Add a regression test: AI resigns after a legal opening, and that opening sequence is still present in the stored game record.

**Files:**

- `web/app.py`

- `ai/opening\_book.py`

- `ai/memory\_manager.py`


### Enhancement B-32 — Increase AI reasoning / commentary transparency ⬜

**Goal:** Commentary/debug output should identify the dominant reason for the AI's move choice (immediate mill closure, mandatory block, busy-chain win, fork prevention, convergence disruption, cardinal-lane block, mobility squeeze, trajectory exploit, endgame DB recognition, opening-book adherence).

**Fix:**

- `ai/game\_ai.py` — capture a structured explanation object for the selected move listing top scoring features / bonuses / blockers.

- `ai/coordinator.py` — expose those reasons in commentary, debug logs, and optional dev overlays.

- `web/static/game.js` — display a richer "AI thought process" summary when commentary mode is enabled.

**Files:**

- `ai/game\_ai.py`

- `ai/coordinator.py`

- `web/static/game.js`


### Bug B-34 — Placement 9 should avoid sterile forks with no nearby feeder support ⬜ *(implementation covered by B-28)*

**Symptom:** On the last placement, the AI sometimes creates a nominal fork or 2-config that has no nearby feeder pieces and confers no forcing continuation.

**Example game (bad last placement — White plays g1):**

```
1.d6 d2    2.b4 f4    3.g4 a4    4.f6 d7      
5.e4 c4    6.d3 e5    7.d1 a7    8.b6xa7 b2      
9.g1  ← White places last piece at g1    
```

White's last placement at `g1` reduces mobility, creates no immediate threat, and allows Black to form a 2-config for an immediate mill.

**Fix:**

- Add a **late-placement quality gate** for placements 8–9: a newly created 2-config must have at least one friendly feeder piece within 2 adjacency steps, OR close a mill or block an immediate opponent threat.

- If neither: apply a `sterile\_fork\_penalty` (default ~100) on the last placement.

- Scale `setup\_mill` bonus down ~40% on placement 9 unless the setup is immediately actionable.

**Files:**

- `ai/heuristics.py` — `tactical\_move\_bonus()`, late-placement window checks


### Bug B-35 — Final placements: prefer dual-purpose block-and-build over passive 2-config ✅ 2026-05-28 *(implemented via B-28 `dual\_purpose\_final\_bonus` in `ai/heuristics.py`)*

**Symptom:** On the last placement the AI creates a 2-piece setup that ignores an opponent mobile mill, when a dual-purpose square would both block and create own pressure.

**Example game:**

```
1.d6 d2    2.f4 b4    3.g7 g4    4.d7 d5      
5.a7xd5 d5   6.f6 f2    7.b6xd5 d5   8.c4 b2xc4      
9.d3 e5  ← Black's last placement — passive 2-config
```

Placing at `a4` instead would both block `a4-b4-c4` and create a 2-config approach.

**Fix:**

- Add a `dual\_purpose\_final\_bonus` (~150) for a placement that simultaneously blocks an opponent active mill line AND creates a new own 2-config.

- Weight this bonus higher on placements 8–9.

**Files:**

- `ai/heuristics.py`



### Tactical bug — Black failed to close its own mill and missed White's immediate threat ✅ 2026-05-31

**Root cause:** Resolved by B-66's move-phase carveout. `\_immediate\_mill\_threats()` returns `\{g4\}` (White threatens g1-g4-g7 via f4), but the single-threat + `\_stm\_can\_close\_mill` guard clears it — Black has b4+b6 with b2 empty and d2 adjacent to b2. Unrestricted `choose\_move` then scores d2-b2 (closes b2-b4-b6 + capture) far above b4-b2 (no mill, no capture).

**Regression tests added:** `tests/test\_b66.py::TestMillCloseVsPassiveSlide` (2 tests).


### Bug B-59 — AI misses forced mills in move phase (sealed 2-config) ✅ 2026-05-28 ★★

*(Implemented: `\_sealed\_two\_configs()`, static eval term, P0.5 move ordering, `sealed\_setup\_bonus`.)*

**Symptom:** In the move phase, the AI fails to recognise and pursue a forced mill where both empty closing squares are accessible only to its own pieces (no opponent can reach either square to block). The AI drifts to known-good oscillation or cardinal-node mobility moves instead.

**Motivating game** (White = Scholar, move phase begins after 9 placements each):

```
After placement: White on a7,g7,g4,a4,d3,c4,e4,f2,b6 / Black on d7,d6,b4,a1,g1,f6,d2,b2,d5  
10.g4-f4 g1-g4 / 11.d1-g1 d2-d1 / 12.f4-e4 b2-d2
```

After move 12, White has **d3, c4, e4** on the board. The inner ring bottom side **e3-d3-c3** is a forced two-move mill:

- `c3` is adjacent only to `d3` (White) and `c4` (White) — no opponent piece can reach it.

- `e3` is adjacent only to `e4` (White) and `d3` (White) — no opponent piece can reach it. Path A: `c4→c3`, then `e4→e3` (no Black piece can prevent e3). Path B: `e4→e3`, then `c4→c3`. White instead plays `f4→e4` (cardinal oscillation), missing the forced mill entirely.

**Root cause (three layered failures):**

1. **Static eval weight too small.** `\_WEIGHTS\["move"\] = (30, 48, 12, 5, 50, 0)` — `two\_cfg` weight = **5**. A normal 2-config contributes 5 to static eval; far too small to signal a forced win through deep negamax.

2. **`setup\_mill` bonus is root-only and undifferentiated.** `tactical\_move\_bonus` adds `int(weights.setup\_mill \* 1.3) \* two\_cfg\_gained` (Scholar: `195`) to the root move score. This cannot propagate sealed mill urgency into the alpha-beta tree, and treats a sealed 2-config the same as an easily-blocked one.

3. **Move ordering ignores sealed 2-configs.** `\_order\_moves` puts sealed-2-config-creating moves in bucket P2 (history-sorted), behind direct mill closes (P0) and opponent-mill blocks (P1). The sealed threats never surface early enough to guide search efficiently.

**Fix — three coordinated changes:**

**A. `\_sealed\_two\_configs(board, color) -\> int` in `ai/heuristics.py`**

A "sealed" 2-config is a 2-config whose empty closing square satisfies:

1. No opponent piece is adjacent to the closing square (`all(board.positions\[nb\] != opponent for nb in ADJACENCY\[closing\_sq\])`).

2. Guard: `\_closeable\_mills(board, opponent) == 0` — opponent cannot immediately close a mill of its own (which would let it capture the piece sealing the closing square before we act).

```
def \_sealed\_two\_configs(board: BoardState, color: str) -\> int:  
    opponent = "B" if color == "W" else "W"  
    if \_closeable\_mills(board, opponent) \> 0:  
        return 0   \# guard: opponent can punish immediately  
    count = 0  
    for mill in MILLS:  
        own   = sum(1 for p in mill if board.positions\[p\] == color)  
        empty = \[p for p in mill if board.positions\[p\] == ""\]  
        if own == 2 and len(empty) == 1:  
            closing = empty\[0\]  
            if all(board.positions\[nb\] != opponent for nb in ADJACENCY\[closing\]):  
                count += 1  
    return count
```

**B. Boost sealed threat in static eval**

In `evaluate()`, add a `sealed\_two\_cfg` term to the move-phase call beside the existing `two\_cfg` term:

```
sealed\_w = \_sealed\_two\_configs(board, "W")  
sealed\_b = \_sealed\_two\_configs(board, "B")  
sealed\_score = (sealed\_w - sealed\_b) \* SEALED\_TWO\_CFG\_WEIGHT   \# target weight: 18–22
```

`SEALED\_TWO\_CFG\_WEIGHT` should be a constant ~18 (3-4× the regular `two\_cfg` weight of 5) so the term propagates clearly through even 2–3 plies of negamax.

**C. Elevate sealed-2-config-creating moves in `\_order\_moves` in `ai/game\_ai.py`**

After the existing P0 (direct mill close or fork) bucket, add a new **P0.5** bucket for moves that create a new sealed 2-config:

```
\# Compute post-move sealed count for each candidate (lightweight: only call for move-phase)  
if board.phase == "move":  
    sealed\_before = \_sealed\_two\_configs(board, color)  
    sealed\_creates = \{  
        m for m in moves  
        if \_sealed\_two\_configs(board.apply\_move(\{"from": m\[0\], "to": m\[1\], "capture": None\}), color) \> sealed\_before  
    \}  
else:  
    sealed\_creates = set()  
  
\# Priority buckets  
p0  = \[m for m in moves if m\[1\] in close or \_is\_fork(m)\]  
p05 = \[m for m in moves if m not in p0 and m in sealed\_creates\]   \# NEW  
p1  = \[m for m in moves if m not in p0 and m not in p05 and (m\[1\] in block or \_is\_squeeze(m))\]  
p2  = \[m for m in moves if m not in p0 and m not in p05 and m not in p1\]
```

**D. `sealed\_setup\_bonus` in `tactical\_move\_bonus`**

Add a large root-level bonus for moves that create a new sealed 2-config:

```
sealed\_after  = \_sealed\_two\_configs(new\_board, color)  
sealed\_before = \_sealed\_two\_configs(board, color)  
sealed\_gained = max(0, sealed\_after - sealed\_before)  
sealed\_setup\_bonus = int(weights.close\_mill \* 0.75) \* sealed\_gained   \# ~243 for Scholar
```

This root-only bonus supplements the static eval term. Together they ensure the AI both evaluates sealed positions correctly in the tree and selects them decisively at the root.

**Regression test (required):**

```
\# After move 12 (f4-e4 played by White, b2-d2 by Black), reconstruct board.  
\# White: a7,g7,g4(moved to f4),a4,d3,c4,e4,f2,b6 minus piece at f4 plus at e4...  
\# Exact FEN: build from game trace.  
\# Assert: AI (White, difficulty ≥ 3) at this position selects c4→c3 or e4→e3, NOT f4→e4 or e4→e5.
```

**Files:**

- `ai/heuristics.py` — `\_sealed\_two\_configs()`, `SEALED\_TWO\_CFG\_WEIGHT` constant, `evaluate()` sealed term, `tactical\_move\_bonus()` sealed\_setup\_bonus

- `ai/game\_ai.py` — `\_order\_moves()` P0.5 bucket for sealed-2-config-creating moves

- `tests/test\_heuristics.py` — unit test for `\_sealed\_two\_configs` on the move-12 position

- `tests/test\_game\_ai.py` — regression test: move selection asserts c4→c3 or e4→e3


### Bug B-60 — Cycling-capture unblock: AI ignores opponent threats enabled by vacating the mill ✅ 2026-05-28 ★★

*(Implemented: `cycling\_capture\_unblock` penalty in `tactical\_move\_bonus()`.)*

**Symptom:** When the AI closes a cycling mill (a mill it intends to oscillate by repeatedly moving the same piece in and out), it selects the highest-value capture by standard heuristics. It does not consider that its *next* oscillation move will vacate a square currently blocking an opponent 2-config, enabling an immediate opponent mill.

**Motivating game** (continuation of the B-59 game, White = Scholar):

```
17.e4-e5 b4-b2 / 18.b6-b4xa7 f6-f4 / 19.e5-e4 d5-c5 / 20.a4-a7xg4 a1-a4  
21.g7-g4 d1-a1 / 22.g4-g7xf4 a1-d1 / 23.g7-g4 d6-b6
```

At **move 22**, White closes mill `g4-g7-g1` (g4→g7 + capture). White is cycling this mill via g7↔g4. After the capture:

- White has **a7** on board (from the cycling mill on line 7: `a1-a4-a7`).

- Black has **a1** and **a4** on board.

- On White's next oscillation, White will move **g7→g4**, vacating a7. Black immediately closes **a1-a4-a7** — a full mill.

- White should capture **a1** (removes one leg of Black's pending mill), but instead captures **f4** (a cardinal node, scored higher by `capture\_feeder\_bonus` / `capture\_diamond\_bonus`).

**Root cause:**

The capture-selection heuristics in `tactical\_move\_bonus` are fully reactive — they score the opponent piece being removed, not the opponent threat that will be unblocked next turn. None of the five existing capture bonuses (`capture\_feeder\_bonus`, `capture\_diamond\_bonus`, `safe\_capture\_bonus`, `capture\_creates\_diamond\_bonus`, `capture\_activates\_feeder\_bonus`) model the constraint:

> *"This cycling mill will oscillate. When I vacate the oscillation square next turn, does the un-captured opponent piece complete a mill?"*

**Fix — `\_cycling\_capture\_unblock\_penalty` in `tactical\_move\_bonus`:**

When a move closes a mill on a **cycling-ready** line (i.e., the piece that was just moved can oscillate back next turn — it has an adjacent empty square), evaluate each legal capture against the following check:

```
def \_cycling\_unblock\_penalty(board\_after\_capture: BoardState,  
                              color: str,  
                              cycling\_sq: str) -\> int:  
    """  
    Score penalty: if the cycling piece at cycling\_sq moves away next turn,  
    does the resulting board give the opponent an immediate mill closure?  
    """  
    opponent = "B" if color == "W" else "W"  
    \# Simulate vacating the cycling square  
    sim = dict(board\_after\_capture.positions)  
    sim\[cycling\_sq\] = ""  
    sim\_board = board\_after\_capture  \# lightweight: only need positions for mill check  
    \# Check every mill containing cycling\_sq  
    for mill in MILLS:  
        if cycling\_sq not in mill:  
            continue  
        opp\_count = sum(1 for p in mill if (sim\[p\] == opponent or (p == cycling\_sq and sim\[p\] == "")))  
        \# Recount with cycling\_sq empty  
        opp\_in\_mill = sum(1 for p in mill if p != cycling\_sq and board\_after\_capture.positions\[p\] == opponent)  
        empty\_in\_mill = sum(1 for p in mill if p != cycling\_sq and board\_after\_capture.positions\[p\] == "")  
        if opp\_in\_mill == 2 and empty\_in\_mill == 0:  
            \# cycling\_sq is the only non-opponent square; vacating it hands opponent a mill  
            return weights.cycling\_mill   \# recycle the personality's cycling weight as a penalty scale  
    return 0
```

For each candidate capture `cap`, compute `\_cycling\_unblock\_penalty(board\_after\_cap, color, cycling\_sq)` and subtract it from the move score. This makes capturing the piece that contributes to the blocked mill worth more than capturing a high-value but irrelevant piece.

**New personality weight:** `cycling\_capture\_unblock` (default 180). Subtract this from any capture that leaves an opponent 2-config with a closing square that is the next vacated square.

**`cycling\_capture\_unblock` in `data/personalities/\*.json`:** add to all personality files with value 180 (tunable).

**Regression test (required):**

```
\# Reconstruct board at move 22 (White to move, g4→g7 closes cycling mill).  
\# White pieces: a7, g1, g4, e4, c4, d3, a4, b4, b6  
\# Black pieces: a1, a4(moved from), d1, c5, f4, b2, d6, d2 — adjust from exact trace  
\# Assert: after White plays g4→g7 (closing cycling mill on g-column),  
\#   the AI selects capture a1 (not f4 or other cardinal node).
```

**Files:**

- `ai/heuristics.py` — `\_cycling\_unblock\_penalty()` helper, `tactical\_move\_bonus()` apply penalty per candidate capture

- `ai/game\_ai.py` — no structural change needed (penalty applied inside existing capture loop)

- `game/rules.py` or `ai/heuristics.py` — `\_is\_cycling\_ready(board, sq) -\> bool` helper (piece at sq has ≥1 adjacent empty square)

- `data/personalities/aggressive.json`, `balanced.json`, `defensive.json`, `positional.json`, `scholar.json`, `custom.json` — add `cycling\_capture\_unblock: 180`

- `tests/test\_heuristics.py` — unit test for penalty function on the move-22 board

- `tests/test\_game\_ai.py` — regression test: assert capture = a1 at move 22


### Bug B-61 — Cycling capture receives zero close\_mill bonus (gross vs net mill delta) ✅ 2026-05-28 ★★

*(Implemented: gross `mills\_delta = len(\_after\_closed\_set - \_before\_closed\_set)` in `tactical\_move\_bonus()`.)*

**Symptom:** During the move phase, the AI refuses to execute winning cycling-mill captures. When a move simultaneously opens one closed mill and closes another (the cycling pattern), `tactical\_move\_bonus()` computes `mills\_delta = max(0, \_closed\_mills(after) - \_closed\_mills(before)) = 0` because the net change is zero. The move receives no `close\_mill\_contribution` bonus (~500 pts) despite enabling a capture. The resulting ~400-point scoring gap causes the AI to prefer idle positional shuffles indefinitely.

**Motivating example (game — turn 17):**

Position: White \{a7, d6, f2, a4, d1, b6, a1, d5\} (8 pieces, closed mill: a7-a4-a1)  
Black: \{b4, g1, d2, b2\} (4 pieces, no 2-configs)

Correct move `a7→d7 xb4`:

- Opens `a7-a4-a1` (a7 leaves) and closes `d5-d6-d7` (a7 arrives at d7)

- `\_closed\_mills(before) = 1`, `\_closed\_mills(after) = 1` → `mills\_delta = max(0, 0) = 0` → **no close\_mill bonus**

- Black drops to 3 pieces and enters fly phase — a near-winning position for White

White instead repeats `f2↔f4` indefinitely; the game never finishes.

**Root cause — line 1855 of `ai/heuristics.py`:**

```
mills\_delta = max(0, \_closed\_mills(after) - \_closed\_mills(before))   \# net delta
```

A cycling move scores net 0 regardless of the capture it enables.

**Fix — compute gross newly-closed mills:**

```
before\_closed\_set = \{  
    tuple(sorted(m)) for m in MILLS  
    if all(board.positions\[p\] == color for p in m)  
\}  
after\_closed\_set = \{  
    tuple(sorted(m)) for m in MILLS  
    if all(new\_board.positions\[p\] == color for p in m)  
\}  
mills\_delta = len(after\_closed\_set - before\_closed\_set)   \# replaces net-delta line
```

This mirrors the existing `mill\_opened` variable (line 1924) which already counts gross opened mills; the symmetry makes the fix self-consistent.

**Secondary fix — `\_mill\_threats()` phantom 2-config overcount:**

`f2→f4` is also inflated by `herding\_coverage` (+40) because f4 is adjacent to f2, which `\_mill\_threats()` counts as the closing square for Black's "2-config" f2-d2-b2. This 2-config is a phantom: moving d2→f2 leaves d2 empty while b2 is already in the line — the mill still cannot close. Fix in `\_mill\_threats()` (line ~464): when testing if any friendly piece is adjacent to a closing square, exclude pieces that are already inside the same mill being counted. This prevents `\_closed\_mills(before) - \_closed\_mills(after)` from going wrong — but more directly, it prevents `herding\_coverage` from rewarding adjacency to impossible-to-close mills.

**Files:**

- `ai/heuristics.py` — `tactical\_move\_bonus()`: replace net `mills\_delta` with gross `mills\_delta = len(after\_closed\_set - before\_closed\_set)`; `\_mill\_threats()`: exclude pieces already in the same counted mill from the adjacency check

**Tests (still open):**

- Unit test: construct the turn-17 board; assert `tactical\_move\_bonus(board, "W", move\_a7\_d7\_xb4)` ≥ 450 (close\_mill\_contribution fires)

- Regression: AI (White, any difficulty) at the turn-17 position selects `a7→d7`, not `f2→f4`


### Bug B-62 — `own\_convergence` suppresses execution of cycling mills that share a pivot ✅ 2026-05-28 ★★

*(Implemented: `convergence\_restoration = weights.own\_convergence \* mills\_delta` in `tactical\_move\_bonus()`.)*

**Symptom:** When White's two active 2-configs share a pivot piece (e.g., d6 is pivot for both `b6-d6-f6` and `d5-d6-d7`), the `own\_convergence` bonus (+250) rewards White for keeping d6 as a convergence pivot. When White cycles a mill and a piece lands at d7 (closing `d5-d6-d7`), the 2-config becomes a closed mill — d6 is no longer a 2-config pivot. The convergence bonus drops from +250 to 0. This ~250-point static-eval loss partially offsets the close\_mill bonus restored by B-61, and can still tip the scale against the cycling capture in some positions.

**Root cause:**

`evaluate()` computes `own\_convergence` as a static term counting 2-config pairs sharing a pivot or closing square. After a 2-config closes into a mill (even when closing creates a capture), the convergence term drops. The evaluator treats a mill closure as a structural loss.

**Fix — neutralize the convergence loss when a mill was just closed:**

In `tactical\_move\_bonus()`, after the gross `mills\_gross\_closed` computation from B-61, add a restoration term:

```
if mills\_gross\_closed \> 0:  
    \# Closing a mill is structurally good. Counteract the static-eval drop  
    \# in own\_convergence that fires when the closed 2-config loses its pivot status.  
    score += weights.own\_convergence \* mills\_gross\_closed
```

This is applied once per newly-closed mill, matching the scale of the convergence bonus that was suppressed. It does not affect moves where no mill was closed.

**Note:** Implement B-61 first and re-test. B-62 is only needed if a scoring gap \> 150 pts persists after B-61. The gap at turn 17 is approximately -250 from own\_convergence on top of the -408 from the net-delta bug; after B-61 restores ~500 pts, the residual gap is ~250-408+500 ≈ +92 in favour of the correct move — meaning B-62 may be unnecessary. Verify empirically before implementing.

**Files:**

- `ai/heuristics.py` — `tactical\_move\_bonus()`: add `own\_convergence \* mills\_gross\_closed` restoration when `mills\_gross\_closed \> 0`

**Tests (still open):**

- Unit test: own\_convergence restoration fires correctly when move closes a mill that shared a pivot in a convergence pair

- Regression: turn-17 move selection is correct after B-61; add B-62 only if gap persists


### Bug B-63 — Fly-entry position undervalued: opponent mobility over-counted on entering fly phase ✅ 2026-05-28 ★ Medium Priority

*(Implemented: `\_FLY\_MOBILITY\_CAP = 5` in `\_mobility()`.)*

**Symptom:** Immediately after White captures an opponent piece that drops Black to 3 pieces (fly phase), `\_mobility(B)` returns the number of empty squares on the board (~13–15). This makes Black appear highly mobile. In `evaluate()`, the `(own\_mob - opp\_mob) × 8` term becomes strongly negative (`8 × (4 - 13) = -72`), penalising the position White just achieved. The AI systematically undervalues captures that send the opponent into fly phase — the very captures that are winning.

**Root cause — `\_mobility()` line ~451 of `ai/heuristics.py`:**

```
if board.pieces\_on\_board\[color\] \<= 3:  
    return len(\[p for p in POSITIONS if board.positions\[p\] == ""\])
```

Fly phase mobility = empty squares ≈ 13–15. Normal move-phase mobility ≈ 3–6. The differential swings by ~70 points in the wrong direction on the ply where White makes its best move.

**Fix — cap fly-phase mobility in the mobility differential:**

```
if board.pieces\_on\_board\[color\] \<= 3:  
    return 5   \# constant cap; fly pieces can jump anywhere, so raw mobility is misleading
```

A constant of 5 (matching typical move-phase values) prevents fly-entry from appearing worse than the position before capture. An alternative is to use a separate `fly\_mob\_weight` field in `HeuristicWeights` (initially 0) and multiply the fly mobility by that instead of `mob\_weight`, suppressing the fly mobility contribution entirely.

The simple cap is preferred: fewer fields, same effect, easier to reason about.

**Impact:** Prevents the ~50–80 point penalty that discourages capturing moves that send the opponent into fly. After B-61 and B-62, this is unlikely to be the deciding factor — but it causes systematic mis-scoring of whole game sub-trees whenever the fly transition is in the search horizon.

**Files:**

- `ai/heuristics.py` — `\_mobility()`: return a capped constant (5) for fly-phase pieces instead of empty-square count

**Tests:**

- Unit test: `\_mobility(board, "B")` returns ≤ 5 when Black has 3 pieces, regardless of board fill

- Regression: AI (White) at a position one ply before sending Black into fly correctly prefers the capturing move


### Note — GUI slider set is missing evolved heuristic weights

**Symptom:** `tools/evolve\_weights.py` tunes more heuristic fields than the web slider panel exposes. `HeuristicWeights` has 36 fields; the GUI exposes ~22.

**Hidden weights currently tuned but not in GUI:** `capture\_disrupt\_diamond`, `capture\_disrupt\_feeder`, `convergence\_block`, `convergence\_disrupt`, `convergence\_penalty`, `cross\_feed\_mobility`, `herding\_squeeze`, `locked\_mill\_penalty`, `mill\_trap\_build`, `mobility\_reduction`, `own\_convergence`, `placement\_busy\_scan`, `ring\_crowding\_penalty`, `sacrifice\_viable`.

**Fix:** Bring the frontend slider list into sync with `HeuristicWeights`, or explicitly split the dataclass into "UI-exposed" and "internal-only" weights.


### Evolve weights v2 — cross-personality master tuning ✅ DONE

**Implemented 2026-05-30:**

- `--gauntlet` mode: tunes `best.json` against all personalities as opponents (averaged win rate, threshold 0.52)

- `--gauntlet-threshold` arg (default 0.52 vs 0.55 per-personality)

- `--subset-size N`: rotate random subset of N tunable fields per era (0 = all 53 fields)

- Sigma fix: 0 promotions in era → sigma × 1.50 (UP) instead of shrinking (escape local minima)

- Per-era subset logged; `\[STUCK→↑σ\]` note on boundary log line

- `web/app.py`: `\_maybe\_auto\_evolve()` triggers gauntlet after N human games (0 = disabled)

- `/api/auto\_evolve` GET/POST endpoints; `auto\_evolve\_after\_games` stored in settings.json

- Tools page: gauntlet checkbox, subset-size input, auto-evolve N input + save button


## Search & Evaluation Enhancements (SE-1 through SE-9 complete ✅)

### TIER 3 — Solid, Secondary Priority

### SE-10 — Proactive Own Fork Setup (Move Phase) ✅ 2026-06-03

**Why:** The existing `fly\_fork\_bonus` fires reactively. Extend `\_fork\_in\_n(board, opp, n=2)` (already used in placement-phase, Enhancement B-4) to the move phase: scan forward up to 3 half-moves for forcing lines that result in 2+ simultaneous 2-configs.

**Deliverables:**

- `ai/heuristics.py` — `\_move\_phase\_fork\_anticipation(board, color, depth=3)`; bonus `fork\_depth × 80` added to root move score


### SE-11 / SE-11b / SE-11c — Opponent Likelihood Weighting + VN Reordering ✅ 2026-06-01

**SE-11 (original):** +1 extension for high-frequency opponent moves at first opponent ply using `query\_all\_frequencies`.

**SE-11b:** Shared `\_move\_path\_buf` push/pop path buffer through `\_negamax` recursion; query gated to first opponent ply only (307µs/call too expensive at deeper plies).

**SE-11c:** At first opponent ply, VN re-sorts the LMR tail (last 60% of ordered moves) by `value\_net.predict()` score descending so the strongest opponent moves are searched first. Gated to `depth \>= 3` and VN available.

**`\_MAX\_OPP\_PLIES = 2`** constant controls reach; path tracking preserved for future extension.

**Files changed:** `ai/game\_ai.py`


### Enhancement B-73 — Wire Value Network into Negamax Leaf Evaluation ✅ 2026-05-31

**Goal:** Make the trained `data/value\_net.npz` actually affect game play. Currently `ValueNet` trains fine and the hook exists in `game\_ai.py` / `mcts.py`, but `app.py` never loads the file or passes it to `GameAI`, so it has zero effect.

**What to do:**

1. In `web/app.py` at server startup, attempt to load `ValueNet.load\_if\_exists(ROOT / "data" / "value\_net.npz")` and store as `\_value\_net` (similar to how `\_fullgame\_db` and `\_endgame\_solved\_db` are loaded).

2. Pass `value\_net=\_value\_net` wherever `GameAI(...)` is constructed — in `\_make\_game\_ai\_for\_personality()`, the handoff branch, and the AvsA block.

3. In `ai/game\_ai.py` `\_negamax()`, at leaf nodes (depth == 0 or quiescence cutoff), if `self.\_value\_net` is not None, blend the value net output with the heuristic score:

   - Get `vn\_score = self.\_value\_net.predict(board, color) \* SCALE` (scale to match heuristic range, e.g. × 500)

   - Return `blend \* vn\_score + (1 - blend) \* heuristic\_score` where `blend` starts at 0.25 and can be tuned

   - Only activate after the placement phase (reduces noise in phase where network has less data)

4. Add `value\_net\_blend: float = 0.25` to `HeuristicWeights` so it's tunable from the AI Tuning panel or via evolution.

5. Add a `vn-status` status line to the game header or AI Tuning panel showing "Value net: loaded (33 KB)" or "Value net: not found".

**Note:** The value network quality depends entirely on how many self-play games have been run and trained on. With \<200 games expect noise; with 2000+ expect genuine signal. Start with `blend=0.1` and raise only after verifying it doesn't hurt play.

**Files:**

- `web/app.py` — load `\_value\_net` at startup; pass to `GameAI`

- `ai/game\_ai.py` — blend `\_value\_net.predict()` at leaf nodes in `\_negamax`

- `ai/heuristics.py` — add `value\_net\_blend` field to `HeuristicWeights`

- `web/templates/index.html` / `web/static/game.js` — optional status display


### TIER 4 — Infrastructure / Long-Term

### Enhancement B-74 — Cross-Mill Cycling Fork Static Bonus ✅ 2026-06-01

**Problem:** When White holds a completed mill M1 and a near-complete mill M2 where a piece of M1 is one step from M2's closing square, the AI undervalues the position. The existing `\_mill\_cycle\_ready` in `evaluate()` treats this the same as any cycle-ready mill. The more powerful "two-mill fork" (alternating captures every two turns, opponent cannot simultaneously block both) only received a delta bonus in `score\_move()` via `cycling\_close\_bonus` and `cycling\_gain` — never in the leaf static eval.

**Fix:** `\_cross\_mill\_cycling(board, color)` counts (closed mill, near-mill) pairs where a piece in the closed mill is adjacent to the empty closing square of the near-mill. Wired into `evaluate()` in move and fly phases with `cross\_mill\_cycling` weight (default 300).

**Key scenario:** After 11.f4→f6×d5 in the user's demonstration game:

- White has closed mill b6-d6-f6

- White has near-mill c5-?-e5 (d5 just captured, now empty)

- d6 (in the closed mill) is adjacent to d5 (the closing square)

- `\_cross\_mill\_cycling` = 1 for White; Black cannot block both mills simultaneously

**Files changed:**

- `ai/heuristics.py` — `cross\_mill\_cycling` field in `HeuristicWeights`, `\_cross\_mill\_cycling()` function, wired into `evaluate()`

- `web/app.py` — all 4 `HeuristicWeights(...)` constructors include `cross\_mill\_cycling`

- `data/weights/best.json` — `"cross\_mill\_cycling": 300`

- `tests/test\_b74.py` — unit test with the post-move-11 board


### B-75 — Background Pondering During Opponent's Turn ✅ 2026-06-01

**What:** After the AI makes a move, predict the most likely opponent reply and begin a full-depth search of the response in a daemon thread. If the human plays the predicted move, the cached result is used immediately (ponder hit); otherwise it is discarded.

**Prediction:** `\_order\_moves` priority ordering of opponent moves, optionally refined by VN re-score of top-3 candidates.

**Shadow AI:** Identical config to main AI (`difficulty`, `weights`, `value\_net`, `fullgame\_db`, `endgame\_solved\_db`, `neural\_evaluator`), fresh transposition table (avoids TT contamination).

**Integration:** `PonderManager.start()` called after each AI move (difficulty ≥ 3, no coordinator, no vs\_human). `stop()` + `get\_result()` called at start of the next AI turn. Cancelled on any human capture action.

**Files changed:**

- `ai/ponder.py` — new `PonderManager` class

- `web/app.py` — `Session.ponder\_manager`; start/stop hooks in `\_ai\_turn` and move/capture handlers


### SE-12 — Incremental Evaluation Cache (Zobrist-Keyed Sub-Functions) ✅ 2026-06-10

**Why:** Heavy heuristic sub-calls recompute from scratch every leaf call. With Zobrist hashing already in place (SE-1), a secondary cache keyed by board hash stores sub-function results. Requires SE-1.

**Deliverables:**

- `ai/heuristics.py` — result cache dict keyed by Zobrist hash for top-cost sub-functions; invalidate on `apply\_move`


### SE-13 — N-Gram Opponent Move Predictor ✅ 2026-06-10

**Why:** Complements TrajectoryDB (win/loss rates) with a pure move-frequency bigram/trigram model. Feeds into SE-11 with richer per-sequence predictions.

**Deliverables:**

- `ai/ngram\_opponent\_model.py` — new `NGramOpponentModel` class; `update()` called after each game; `predict()` returns probability dict; trained incrementally from `data/games/` JSONL records


### SE-14 — DB-Guided Horizon Search (FullGameDB + Negamax Hybrid) ✅ 2026-05-26

*(Archived — see plan\_done.md. Implemented in `ai/game\_ai.py` `\_negamax` FullGameDB probe.)*


## B-51 — Early-Endgame DB: expand retrograde solver beyond 3v3 ✅ 2026-05-28 ★ (code complete; tables need building)

**Status:** Builder (`tools/build\_endgame\_db.py`) supports `--nW`/`--nB`, mixed fly/move successors, mmap writes, and `--build-all`. Runtime query dispatch in `ai/endgame\_solved\_db.py` loads all `endgame\_\{nW\}\_\{nB\}.wdl` files. **Remaining work is operational:** run builds to populate tables beyond 3v3.

**Goal:** Build a family of WDL tables covering piece counts from 4v3 through 7v4 (and symmetric reverses). These cover the critical **early endgame transition** — positions where one or both sides have just lost pieces but haven't reached fly phase yet.

**Table sizes (2 bits/position, white\_rank × black\_rank × turn encoding):**

| nW | nB | Positions | MB |
| - | - | - | - |
| 4 | 3 | 24,227,280 | 6.1 |
| 3 | 4 | 24,227,280 | 6.1 |
| 5 | 3 | 82,372,752 | 20.6 |
| 3 | 5 | 82,372,752 | 20.6 |
| 4 | 4 | 102,965,940 | 25.7 |
| 5 | 4 | 329,491,008 | 82.4 |
| 4 | 5 | 329,491,008 | 82.4 |
| **Tier 1 total** |  |  | **~79 MB** |


**Practical tiers:**

- **Tier 1 — Recommended:** 4v3, 3v4, 5v3, 3v5, 4v4 → ~79 MB total.

- **Tier 2 — Optional:** add 5v4, 4v5 → ~244 MB total.

- **Tier 3 — Large/optional:** 6v3, 3v6, 7v3, 3v7, 6v4, 7v4.

**Key algorithm changes vs the existing 3v3 builder:**

1. **Mixed fly/move successor generation:** a side with exactly 3 pieces flies; a side with ≥4 moves along adjacency edges.

2. **Cross-table captures:** a capture in nWvnB leaves nWv(nB-1) or (nW-1)vnB — successor lives in a different already-solved table.

3. **Build order:** each table depends on both smaller tables from captures. Solve in order of (nW + nB) ascending.

4. **File naming:** `endgame\_\{nW\}\_\{nB\}.wdl` alongside `endgame\_3\_3.wdl`.

5. **Query integration:** extend `EndgameSolvedDB` to load all available files; `query()` dispatches by `(len(w\_pieces), len(b\_pieces))`.

**Files:**

- `tools/build\_endgame\_db.py` — rewrite to accept `--nW` and `--nB` args; mixed fly/move successor generator; cross-table reference loading

- `ai/endgame\_solved\_db.py` — extend `EndgameSolvedDB.\_\_init\_\_` to load all available tables; extend `query()` to dispatch by piece count

**Prerequisite:** B-57 (direct-to-disk mmap writing) must land first — the 5v4 and 4v5 tables (82 MB each) will exceed practical `bytearray` size without it.


## B-57 — Direct-to-disk binary writing for endgame DB and fullgame DB ✅ 2026-05-28 ★

**Goal:** Replace the in-memory `bytearray` (endgame) and large intermediate structures (fullgame) with direct writes to a pre-allocated binary file using Python's `mmap`. All solve and fill passes operate on the memory-mapped file handle instead of RAM. A final label pass then edits the file in-place to mark winning and losing trajectory positions.

**Why this matters:**

- 3v3 is 1.3 MB — RAM is not the constraint today.

- B-51 tables (5v4, 4v5) reach 82–330 MB each; 6v5 exceeds 1 GB. These will not fit in a Python `bytearray` on most build machines.

- The fill pass (non-canonical → canonical propagation) and label pass (WIN/LOSS annotation) are sequential scans with local reads — exactly the workload OS page caching excels at with mmap.


### Endgame DB (`tools/build\_endgame\_db.py`)

**Current flow:**

1. `table = bytearray(n\_bytes)` — entire table allocated in RAM as UNKNOWN.

2. Solve passes iterate `canonical\_ids`, set WIN/LOSS/DRAW in `table`.

3. Fill pass iterates all positions, copies from canonical entry.

4. Caller does `wdl\_path.write\_bytes(bytes(table))`.

**Proposed flow:**

1. Pre-allocate the `.wdl` file on disk: `path.write\_bytes(bytes(n\_bytes))` (zeros = UNKNOWN = valid starting state).

2. Open the file and `mmap.mmap(f.fileno(), n\_bytes)` — the OS manages which pages are in RAM.

3. All solve passes, fill pass, and DRAW marking operate on the mmap handle via the existing `get\_wdl`/`set\_wdl` API — **no API change required**.

4. `table.flush(); table.close()` — file is already on disk; no bulk write at the end.

5. **Label pass** (new, optional): after the file is fully solved, re-open it and mark a chosen subset of positions with a high-contrast sentinel (e.g. reuse `WDL\_WIN`/`WDL\_LOSS` is already the label — but if a richer annotation is needed, extend to 3 bits per entry or a sidecar index file mapping `pos\_id → outcome + move`).

**Key change in `solve\_table`:**

```
\# Before  
table = bytearray(n\_bytes)  
  
\# After  
out\_path.write\_bytes(bytes(n\_bytes))          \# pre-allocate (UNKNOWN = 0x00)  
f = open(out\_path, "r+b")  
table = mmap.mmap(f.fileno(), n\_bytes)  
try:  
    \_solve\_passes(table, ...)  
    \_fill\_pass(table, ...)  
    table.flush()  
finally:  
    table.close()  
    f.close()
```

`solve\_table` returns `None` (file is already written); callers that need the bytes for sub-table loading use `open(path, "rb").read()` or a second mmap.


### Fullgame DB (`tools/build\_fullgame\_db.py`)

**Done (by B-52):** SQLite was already replaced with a sorted binary `.bin` format (36-byte records, mmap'd read-only at query time). A `--max-gb` guard prevents OOM during BFS. No further changes needed for B-57.

### Files changed

- `tools/build\_endgame\_db.py` — replaced `bytearray` with mmap; `solve\_table` writes to `out\_path`, returns None ✅

- `ai/endgame\_solved\_db.py` — no change needed; `get\_wdl`/`set\_wdl` work on any `\[\]`-indexable type ✅

- `tools/build\_fullgame\_db.py` — binary format already done (B-52); no change ✅

- `ai/fullgame\_db.py` — binary reader already in place (B-52); no change ✅

**Prerequisite B-57 → B-51 satisfied.** B-51 (expand retrograde solver beyond 3v3) can now proceed.



## Enhancement B-58 — Multiple LLM Provider Support ⬜ ★★

**Goal:** Replace the Ollama-only LLM integration with a pluggable provider abstraction. Allow the user to choose between Ollama (local), Claude (Anthropic), ChatGPT (OpenAI), Perplexity, or no LLM at all — without changing any game logic.


### Design constraints

- API keys **never** go in `data/settings.json` (git-tracked). Use env vars only: `ANTHROPIC\_API\_KEY`, `OPENAI\_API\_KEY`, `PERPLEXITY\_API\_KEY`.

- Embedding provider is **separate** from the chat provider. Default: ChromaDB `DefaultEmbeddingFunction` (local, no key required). `ollama\_embed\_model` setting (B-53) remains available if user wants Ollama embeddings.

- No streaming for v1 — all calls are blocking `chat(system, messages) -\> str`.

- On any error, return `""` (empty string) — matches current `\_chat()` behaviour.

- Graceful fallback: if a selected provider's package is not installed, log a warning and fall back to Null provider rather than crashing.


### New module: `ai/llm\_provider.py`

```
class BaseLLMProvider(ABC):  
    @abstractmethod  
    def chat(self, system: str, messages: list\[dict\]) -\> str: ...  
    def available(self) -\> bool: return True  
  
class OllamaProvider(BaseLLMProvider):     \# existing behaviour, wraps requests  
class ClaudeProvider(BaseLLMProvider):     \# anthropic SDK \>= 0.20  
class OpenAIProvider(BaseLLMProvider):     \# openai SDK \>= 1.0  
class PerplexityProvider(BaseLLMProvider): \# openai-compatible base URL  
class NullProvider(BaseLLMProvider):       \# always returns ""  
  
def make\_provider(settings: dict) -\> BaseLLMProvider:  
    """Factory — reads settings\['llm\_provider'\] and env vars."""
```

`messages` format matches the OpenAI schema: `\[\{"role": "user"/"assistant", "content": "..."\}\]`.

`OllamaProvider` translates this internally to the Ollama `/api/chat` format (already what `MillsLLM.\_chat` does).


### Provider defaults

| Provider | Default model setting key | Default model |
| - | - | - |
| `ollama` | `ollama\_model` | *(existing setting)* |
| `claude` | `claude\_model` | `claude-sonnet-4-6` |
| `openai` | `openai\_model` | `gpt-4o-mini` |
| `perplexity` | `perplexity\_model` | `sonar` |
| `null` | — | — |



### `data/settings.json` additions

```
"llm\_provider": "ollama",  
"claude\_model": "claude-sonnet-4-6",  
"openai\_model": "gpt-4o-mini",  
"perplexity\_model": "sonar",  
"embed\_provider": "default"
```

`embed\_provider` values: `"default"` (ChromaDB `DefaultEmbeddingFunction`), `"ollama"` (uses `ollama\_embed\_model`).


### Changes to existing files

**`ai/mills\_llm.py`**

- `\_\_init\_\_` replaces `(url, model)` params with `(provider: BaseLLMProvider)`.

- Remove `\_chat()` helper (logic moves to `OllamaProvider.chat()`).

- `get\_move\_commentary()` / `get\_strategy\_commentary()` call `self.\_provider.chat(system, messages)`.

**`ai/memory\_manager.py`**

- Accept `embed\_provider: str` param (default `"default"`).

- If `embed\_provider == "ollama"`, use `OllamaEmbeddingFunction(model=ollama\_embed\_model)`.

- Otherwise use `DefaultEmbeddingFunction()`.

**`web/app.py`**

- Call `make\_provider(settings)` once at startup.

- Pass `provider` to `MillsLLM(provider)` and `embed\_provider` to `MemoryManager`.

**`requirements.txt`**

- Add optional deps with comments:

- ```
\# Optional: anthropic\>=0.20   \# for Claude provider  
\# Optional: openai\>=1.0       \# for OpenAI / Perplexity provider
```


### Settings UI additions

In the Settings panel:

| Control | Type | Notes |
| - | - | - |
| LLM Provider | Dropdown | `ollama / claude / openai / perplexity / none` |
| Model | Text input | Shows the active model setting for the chosen provider |
| API key status | Read-only badge | `✓ key found` (green) or `✗ key missing` (amber) — reads from `/api/provider\_status` |
| Embed provider | Dropdown | `default (local) / ollama` |


New endpoint: `GET /api/provider\_status` → `\{ "provider": "claude", "model": "claude-sonnet-4-6", "key\_present": true, "embed\_provider": "default" \}`.

When `key\_present` is false, the badge is amber with text "Set `ANTHROPIC\_API\_KEY` env var". Commentary is silently disabled (Null fallback) — the game still works.


### Files

- `ai/llm\_provider.py` — new; `BaseLLMProvider`, five concrete classes, `make\_provider()`

- `ai/mills\_llm.py` — constructor change + remove `\_chat()` helper

- `ai/memory\_manager.py` — `embed\_provider` param

- `web/app.py` — factory call, new `/api/provider\_status` endpoint

- `data/settings.json` — four new keys

- `web/templates/index.html` — provider dropdown, model field, key-status badge, embed dropdown

- `web/static/game.js` — load/save provider settings; call `/api/provider\_status` on settings open

- `requirements.txt` — optional dep comments


### B-76 — train\_value\_net.py: use AI vs AI games for assessment ⬜ ★

**Problem:** `train\_value\_net.py` currently draws training/assessment data only from human game records. Human games have uneven quality and sparse tactical situations. AI vs AI games at high difficulty generate much denser tactical content and more reliable signal for value-net calibration.

**Goal:** Extend `train\_value\_net.py` to also load AI vs AI game logs (from `data/logs/` or a dedicated directory) and include those positions in the assessment/training pipeline alongside human games.

**Scope:**

- Detect AI vs AI game files (e.g. by header metadata or filename pattern).

- Feed their positions into the same feature extraction + label pipeline already used for human games.

- No change to the net architecture or training loop — purely a data source extension.


### B-77 — Multi-step mill setup detection (2-ply pin rule) ✅ 2026-06-01 ★★

**Problem:** The move-phase pin rule (B-70) blocks moves where the AI's own piece is the *sole blocker* of an opponent 2-config AND an opponent piece is *adjacent* to the pinned square. This looks exactly 1 ply ahead. A 2-step setup (e.g. Black d2→d1 then a4→a1) is not detected, so the AI vacates g1 allowing Black to form the a1-d1-g1 mill two moves later.

**Example game (observed 2026-06-01):** Turn 18 — White played g1→g4. Black subsequently formed a1-d1-g1 over the next 2 moves.

**Goal:** Extend the pin-detection logic to look 2 plies ahead: after removing the AI's piece from square S, simulate all opponent moves from the resulting position and check whether any opponent move then creates a new direct mill threat (i.e. completes a 2-config where S is now undefended). Flag S as a 2-ply pin and hard-filter or penalise moves that vacate it.

**Files:**

- `ai/game\_ai.py` — `\_pinned\_move\_squares()` or new `\_pinned\_move\_squares\_2ply()` helper


### B-78 — Trajectory DB turn-4 interference investigation ✅ 2026-06-01 ★

**Problem:** In the same game as B-77, White played b2 (placement) on turn 4 instead of completing the mill at g7 (a7 and d7 already placed). This is a B-22-class error but occurs during the placement phase. Suspected cause: trajectory DB promoting b2 as a frequent human move, overriding the tactical mill-close priority.

**Goal:** Audit whether the trajectory DB or FullGameDB hint is interfering with mill-close detection during placement. Add a diagnostic log entry when a DB-suggested move ranks lower than a mill-closing move, to make future regressions visible.

**Files:**

- `ai/game\_ai.py` — SE-14 FullGameDB probe and `\_score\_all` DB hint blending

- `ai/trajectory\_db.py` — `score\_delta` call site in `\_root\_search`


### B-79 — Dead placement filter regression at a7 ✅ 2026-06-01 ★★

**Problem:** B-69 hard-filters placements on squares with 0 free neighbours (unless they close a mill). In an AI vs AI level-3 game, the AI placed on a7 despite a7's neighbours being d7=W and a4=B (0 free adjacent squares), meaning a7 was dead on arrival.

**Observed:** Game notation provided 2026-06-01. B-69 filter (`\_is\_dead\_placement`) did not prevent this.

**Likely causes to investigate:**

1. `board.phase != "place"` test is wrong at the relevant turn (e.g. phase already flipped to "move").

2. The placement move dict has a `"from"` key in some code path, causing the `if mv.get("from") is None` guard to pass it through as a movement rather than a placement.

3. A mandatory-block or mill-close exemption is incorrectly firing.

4. `\_is\_dead\_placement` neighbour lookup uses a stale board state.

**Goal:** Write a regression test reproducing the exact board position, confirm the filter fires correctly, and fix whichever guard is leaking.

**Files:**

- `ai/game\_ai.py` — `\_is\_dead\_placement()` (~~line 278), placement filter (~~line 546-549)

- `tests/test\_blocking.py` or new `tests/test\_b79.py`


### SE-15 — Audit SE-11b/c effectiveness and effective search depth ⬜ ★

**Problem:** In AI vs AI games at difficulty 5–6, moves appear to be made very quickly, suggesting the search may not be reaching the expected effective depth. SE-11b/c (trajectory path buffer + VN reordering at opponent plies) was intended to prune bad human trajectories and deepen effective search. The user requests confirmation that this pruning is actually firing and improving depth.

**Goal:**

1. Add a debug/diagnostic mode that logs: effective depth reached per move, TT hit rate, SE-11 prune count (how many opponent moves were dropped by VN reordering).

2. Verify that `\_MAX\_OPP\_PLIES = 2` is correct and that the VN reordering is actually reordering (not a no-op due to all moves scoring 0).

3. If SE-11 is not firing effectively, investigate and fix.

**Files:**

- `ai/game\_ai.py` — `\_negamax` SE-11b/c block, `\_opp\_plies\_left` usage


### B-81 — Independent dual 2-config threat detection ("keep-busy fork") ✅ 2026-06-01 ★★★

**Problem:** The AI doesn't penalise (or value) positions where one side has two 2-configs whose closing squares are **non-adjacent**. A standard fork where one move can block both threats (closing squares are adjacent or identical) is less dangerous than an independent fork where no single move defends both. The AI treats them identically, so it plays passively during placement while the opponent constructs a structural advantage that becomes a cycling win in move phase.

**Observed pattern (multiple games, 2026-06-01):**

- Turn 6 Black plays f2 → creates (f6,f4,f2) close at f6 AND (f2,d2,b2) close at b2 simultaneously.

- f6 and b2 are non-adjacent; one White placement cannot block both.

- After White blocks one, Black closes the other, captures, and cycles.

**Goal:**

1. New heuristic term `\_independent\_threat\_pairs(board, color)`: count pairs of 2-configs where the two empty (closing) squares are **not adjacent and not the same**. Each such pair is a "free" threat — one move blocks at most one.

2. Weight this term heavily in `evaluate()` (e.g. 2–3× the base 2-config weight). This makes the search avoid letting the opponent build independent forks, and seek to build them for itself.

3. Write tests: verify the term scores correctly on the example positions.

**Files:**

- `ai/heuristics.py` — new `\_independent\_threat\_pairs()`, wired into `evaluate()`

- `tests/test\_blocking.py` or new test file


## Architecture Principles

- **Immutable board state** — `BoardState.apply\_move()` always returns a new object.

- **Coordinator owns the narrative** — All commentary and LLM calls flow through `Coordinator`. `GameAI` is pure search.

- **No cloud dependency** — LLM inference runs locally via Ollama by default; B-58 will add optional cloud providers.

- **Progressive enhancement** — Every stage adds capability without breaking the previous one.

- **Weight-injectable heuristics** — All evaluation weights injectable via `HeuristicWeights`.

- **Tactical before positional** — AI urgency hierarchy (close mill → block mill → disrupt structures → position) is a first-class design constraint.

- **Staged opening memory** — Starting play recognised in phases; move-sequence ancestry and searchable tags preserved.


## Thematic note — placement-phase root causes

The B-22 through B-37 cluster around three confirmed core weaknesses:

**Weakness 1 — Late placement overvalues speculative structure.** Fixed via B-46/B-28: setup-building bonuses taper from 1.0× at placement 1 to 0.25× at placement 9.

**Weakness 2 — Opponent forcing potential is not mirrored.** Fixed via B-37: `\_placement\_chain\_scan` mirrored for the opponent.

**Weakness 3 — Tactical priority ladder exists in ordering but not in scoring.** `\_order\_moves()` has a clean P0/P1/P2 hierarchy but `tactical\_move\_bonus()` is fully additive — speculative bonuses can still outscore emergency blocks. B-29 fixes the chain case; B-22 investigates the block case.



## Game Problem Catalogue (2026-06-09)

The following bugs were recorded from live game observations. Games are expressed in the same half-move notation as the rest of the backlog. All game records use White as the AI under examination unless otherwise noted.


### Bug B-91 — Malom perfect DB does not override bad last placement ⬜ ★★★ [partial]

**Symptom:** White AI level 9 (Malom perfect DB enabled) places its 9th piece at a4 instead of blocking the Black triangle at c5/d6/e4.

**Game record:**

```
1.f4 b4 / 2.d2 d6 / 3.b6 e4 / 4.c4 d3 / 5.d7 g4 / 6.e3 d1 / 7.g1 c5 / 8.a7 g7 / 9.a4 b2 / 10.d2-f2
```

White's 9th piece is a4. With Black already holding c5, d6, e4 (inner cardinal triangle), Black can form multiple closeable 2-configs from the move phase. White should have placed at a square that blocks c5-d6 or e4-c4 convergence (e.g. e5, d5, or f6 depending on what remains available).

**Expected behaviour:** With Malom perfect DB active, placement moves should be scored by WDL outcome at the resulting position — a4 should rank below a blocking placement if the DB entries confirm a different move is winning/drawing.

**Root cause (suspected):**

1. The Malom DB probe in `_negamax` / SE-14 is called at leaf nodes but placement-phase root scoring still relies entirely on `tactical_move_bonus` + opening book. If the DB is only queried mid-search rather than as a root override, a sufficiently large tactical bonus (opening book adherence, feeder bonuses) can mask a DB-indicated loss.

2. The placement dead-filter (B-69) only filters 0-free-neighbour squares. a4 has at least one free neighbour so the filter does not remove it.

**Proposed fix:** At root selection in `choose_move()` (or `_root_search()`), after negamax returns, check if the Malom/endgame DB returns a WDL result for the top candidate's resulting position. If the DB says LOSS and a different candidate is WIN or DRAW, override the negamax choice with the DB-indicated best move (same pattern as the current single-move WDL carveout, but applied across the full root candidate list).

**Files:**

- `ai/game_ai.py` — root WDL override pass after `_root_search` completes
- `ai/endgame_solved_db.py` — confirm `query()` is callable on placement-phase positions

**Implementation notes (2026-06-09):** The root cause was two separate issues:
1. The dead-placement filter (B-69) was removing b2 and f6 from the candidate list because they had 0 free neighbors, even though they created closeable 2-configs via external-piece moves (d2→f2 closes b2-d2-f2). Fixed: added "setup-rescue" exemption that re-admits dead placements that gain a new closeable 2-config via `apply_move + _closeable_mills`.
2. The Malom DB root override cannot be tested (no .sec2 sector files on this machine). That part remains unimplemented.
After the partial fix, the AI picks e5 instead of a4 in the B-91 position (e5 is tactically sound; a4 was creating a dead-end 2-config). The full DB override (Malom required) remains a future task.


### Bug B-92 — AI misses mill relay (advance-to-relay opportunity) ✅ ★★

**Symptom:** Black AI (move phase, turn 14) plays c4-c3 instead of a7-d7, which would set up the g1-g4-g7 cycling mill via d7→g7.

**Game record:**

```
1.d6 d2 / 2.f4 b4 / 3.f6 b6 / 4.f2xb6 b6 / 5.e5 d5 / 6.e4 g4 / 7.e3xg4 d3 / 8.d1 g4 / 9.b2 a4 / 10.d1-a1 b4-c4 / 11.b2-b4 a4-a7 / 12.a1-a4 d2-d1 / 13.f2-d2 d1-g1 / 14.d2-f2xd3 c4-c3
```

After turn 14, Black plays c4-c3. Better: a7-d7, which establishes Black control of the d-column and sets up g1-g4-g7 (Black has g1 and g4; moving d7→g7 closes the mill). c4-c3 creates a 2-config (c3-c4-c5) but c5 is unoccupied by Black — no immediate progression.

**Root cause (suspected):** The relay-setup value of a7-d7 depends on a 3-ply chain (a7→d7; then d7→g7 closes mill). The `_sealed_two_configs` heuristic only rewards moves that *create* a sealed config in one step. A move that positions a piece to *enable* a future sealed config in one more step earns no bonus. The trajectory DB may also not have enough data on this specific position.

**Proposed fix:** In `_order_moves` or `tactical_move_bonus`, add a 2-ply relay bonus: for each candidate move, simulate the resulting board and count sealed 2-configs that would become closeable in one further step (i.e. the piece moved is now adjacent to an empty closing square of an existing 2-config). Weight approximately 0.5× the `sealed_setup_bonus`.

**Files:**

- `ai/heuristics.py` — `tactical_move_bonus()`: relay setup bonus (2-ply sealed approach)
- `ai/game_ai.py` — optional: P0.25 ordering tier for relay-setup moves


### Enhancement B-93 — Ponder: use trajectory DB to predict opponent moves ✅ 2026-06-10

**Symptom:** `PonderManager` predicts the opponent's reply using `_order_moves` priority ordering (mill closes > blocks > history) optionally refined by VN re-score of the top 3 candidates. It does not query the trajectory DB or FullGameDB, both of which contain frequency-weighted human move data.

**Current behaviour:** `ai/ponder.py` calls `_order_moves(board, opp_color)` and takes the first result as the predicted move. On a ponder hit, the cached move is returned immediately with `elapsed=0.00s`.

**Proposed fix:**

1. In `PonderManager._ponder_thread()`, after `_order_moves` produces the candidate list, query `trajectory_db.query(move_history)` and blend the frequency weights into the candidate ordering (same logic as SE-11).
2. If FullGameDB is available, also query it for the post-AI-move position and bias toward high-frequency human continuations.
3. The predicted opponent move becomes the highest-scoring candidate after blending rather than the raw priority-order first element.

**Impact:** Better ponder hit rate → more AI turns served from cache → lower latency.

**Files:**

- `ai/ponder.py` — `_ponder_thread()`: add trajectory DB + FullGameDB query for opponent prediction
- `web/app.py` — pass `_trajectory_db` and `_fullgame_db` to `PonderManager` constructor


### Enhancement B-94 — Ponder hit: search deeper with remaining think time ✅ 2026-06-10

**Symptom:** When the human plays the pondered move (ponder hit), the AI returns the cached move immediately with `elapsed=0.00s`. The remaining think-time budget is wasted.

**Proposed behaviour:** On a ponder hit, the pre-computed root move is the starting best move. Use the remaining time budget (configured think time minus the ~0 ms elapsed) to run a deeper iterative deepening search from the same position (the cached result seeds the aspiration window and TT from the ponder search). Return the deepest-confirmed best move when time expires, or the cached result if time is negligible.

**Implementation sketch:**

1. `PonderManager.get_result()` returns `(best_move, ponder_board, ponder_tt)` — the cached TT table from the ponder search.
2. In `_ai_turn()`, on ponder hit: call `game_ai._root_search(ponder_board, remaining_time, seed_move=best_move, seed_tt=ponder_tt)` instead of returning immediately.
3. `_root_search` starts aspiration window around the cached score; iterative deepening continues from the depth already reached in the ponder search.

**Files:**

- `ai/ponder.py` — expose TT state from completed ponder search
- `ai/game_ai.py` — `_root_search()`: accept `seed_move` / `seed_tt` to resume from ponder depth
- `web/app.py` — `_ai_turn()`: pass remaining budget on ponder hit


### Enhancement B-95 — Disable complex heuristics at difficulty 1–2 ✅ ★

**Goal:** At difficulty 1 and 2, the AI should make noticeably simpler, more beginner-like decisions. Several heuristics (defer-for-chain, sealed 2-config detection, fork anticipation, convergence block) make the AI play too well even at low difficulty because they fire unconditionally regardless of the difficulty setting.

**Proposed changes:**

- At difficulty ≤ 2: disable `_move_phase_fork_anticipation`, `_sealed_two_configs` bonus in `tactical_move_bonus`, `defer_for_chain` heuristic, and `_dual_connected_mill_alert`.
- At difficulty ≤ 1: also disable `_pinned_move_squares` (the AI should occasionally blunder into traps).
- The `difficulty` value is already available on the `GameAI` instance — add an `_is_beginner` property (`difficulty <= 2`) and gate the relevant blocks.

**Files:**

- `ai/game_ai.py` — `_is_beginner` guard in `choose_move` and `_order_moves`
- `ai/heuristics.py` — `tactical_move_bonus()`: gate fork anticipation and sealed setup bonus on `difficulty`


### Enhancement B-96 — Log to terminal when sentinel intervenes on an AI move ✅ ★

**Symptom:** When the sentinel changes (intervenes on) the GameAI's chosen move, there is no terminal output. The intervention is only visible in the browser AI Discussion box.

**Proposed fix:** In `ai/game_ai.py`, in `_sentinel_score_adjust()` and `_sentinel_reconsider()`, when an intervention fires (i.e. `new_move != move` or the llm/rank1 branch executes), emit a `print()` or `logger.info()` line to stdout:

```
[Sentinel] intervened: engine intended <orig_notation> → redirected to <new_notation> (type: score_adjust|llm_override|sentinel_best, gap: {gap:.2f})
```

This makes sentinel activity visible when monitoring the server process, and aids debugging of B-91 through B-98 game problems.

**Files:**

- `ai/game_ai.py` — `_sentinel_score_adjust()` and `_sentinel_reconsider()`: add logger.info on intervention


### Bug B-97 — Mill blocker abandonment (three game instances) ✅ [partial] ★★★

Three separate games exhibit the same pattern: the AI vacates a square that is the sole blocker of an opponent mill line, either abandoning an explicit block or disarming its own mill threat.

**Instance (a) — Turn 16, Level 3, d1→g1 abandons block on a1-d1-g1:**

```
1.d6 d2 / 2.f4 a7 / 3.f6 b6 / 4.f2xa7 g1 / 5.d3 d5 / 6.a4 g4 / 7.g7 a7 / 8.b4 c4 / 9.e5 e3 / 10.f4-e4 g4-f4 / 11.g7-g4 d2-b2 / 12.f2-d2 g1-d1 / 13.a4-a1 a7-a4 / 14.g4-g7 a4-a7 / 15.g7-d7 a7-a4 / 16.d7-a7 d1-g1
```

At turn 16, Black plays d1-g1. Black has a1 on board; White plays a7 (turn 16). With d1 vacated, the a1-d1-g1 mill is now open to any d-column piece. Black should maintain d1 as the blocking anchor and find a different move.

**Instance (b) — Turn 15, f2→f4 disarms own mill threat:**

```
1.f4 b4 / 2.d2 d6 / 3.b6 f2 / 4.d1 d3 / 5.c4 a4 / 6.c5 c3 / 7.e3 g4 / 8.d7 g1 / 9.g7 e5 / 10.d7-a7 e5-d5 / 11.a7-d7 a4-a7 / 12.d2-b2 f2-d2 / 13.e3-e4 d3-e3 / 14.f4-f2 d6-f6 / 15.f2-f4 d2-d3xe4
```

White plays f2-f4 at turn 15. White had just played f4-f2 (turn 14) to set up f2-d2-b2 (White holds f2 and d2; b2 is the closing square). f2-f4 vacates f2, dissolving this 2-config. Black takes d2-d3 and captures e4 with tempo. White should have preserved f2 and closed the f2-d2-b2 mill instead.

**Instance (c) — Turn 10, d2→d3 abandons cardinal control:**

```
1.d6 d2 / 2.f4 b4 / 3.f6 b6 / 4.f2xb6 b6 / 5.e5 b2xe5 / 6.c5 d5 / 7.c3 c4 / 8.a4 g4 / 9.a1 a7 / 10.a1-d1 d2-d3
```

White plays a1-d1 at turn 10 (creates 2-config a1-d1-g1). Black responds d2-d3. d2 was a cardinal node and a key feeder for the b2-d2-f2 line. d3 is a weaker square with fewer strategic connections. The move gains nothing while surrendering a key central node.

**Root cause (common thread):** B-70 (`_pinned_move_squares`) only detects 1-ply pins where an opponent piece is *adjacent* to the blocking square. Instances (a) and (c) involve longer setup chains (opponent needs 2+ moves to exploit the vacated square). Instance (b) involves the AI voluntarily undoing its own setup, which no pin rule addresses.

**Proposed fixes:**

1. **Instances (a) and (c):** Extend B-77 (2-ply pin rule) to also cover 2-move opponent setup chains. The `_pinned_move_squares_2ply()` helper should detect when vacating S allows an opponent piece to move to an adjacent square S' where S' then becomes the blocker-needed square for an existing opponent 2-config.

2. **Instance (b):** Add a "2-config dissolution penalty" in `tactical_move_bonus`: when a move vacates a square that was part of the AI's own closeable 2-config (and no mill is being closed), apply a penalty proportional to the number of closeable 2-configs lost (`closeable_before - closeable_after`). This is symmetric to the `setup_mill_bonus` and should be sized similarly (~`weights.setup_mill`).

**Files:**

- `ai/game_ai.py` — `_pinned_move_squares_2ply()` extension (B-77 follow-on)
- `ai/heuristics.py` — `tactical_move_bonus()`: 2-config dissolution penalty


### Bug B-98 — Mill patience: AI closes mill prematurely without net subsequent benefit ✅ ★★

**Symptom:** Black AI (turn 12) plays d2→d1, creating a non-closeable 2-config (g1-d1-a1 — the closing square a1 has no reachable Black piece outside the mill), instead of d5→c5 which maintains cardinal control and patience.

**Game record:**

```
1.d6 d2 / 2.f4 b4 / 3.f6 b6 / 4.f2xb6 b6 / 5.e5 d5 / 6.e4 g4 / 7.e3xg4 d3 / 8.d1 g4 / 9.b2 a4 / 10.d1-a1 b4-c4 / 11.b2-b4 a4-a7 / 12.a1-a4 d2-d1
```

After White plays a1-a4 (turn 12), Black has g4 and d3 on board with d5 also available. Black plays d2-d1, creating g1-d1-a1. But a1's only neighbours are a4=White and d1=Black(in-mill) — the 2-config is permanently non-closeable. Better: d5-c5 (cardinal node, adjacent to c4=Black; creates closeable c3-c4-c5 via the c-column; maintains central positional control).

**Root cause:** The `setup_mill_bonus` fix from B-87 correctly uses `_closeable_mills` delta to avoid rewarding non-closeable 2-configs. However, the broader heuristic still awards positional bonuses for moves that land at well-connected squares (d1 is a junction square with high `cardinal_bonus`). When a player has few pieces on board and d1 is "free", the cardinal node bonus plus any assembly reward may still net more than d5-c5 (which involves a less dramatic static improvement).

Additionally, the trajectory DB may not have enough data on this specific 12-move position to down-weight d2-d1 in favour of d5-c5.

**Proposed fix:**

1. In `tactical_move_bonus()`, add a "cardinal patience penalty": when the AI creates a non-closeable 2-config by a move to a cardinal/junction node (d1, d7, a4, g4, etc.), apply a small penalty (~50–80 pts) to represent the opportunity cost of locking a feeder piece in an un-closeable configuration. This is distinct from the dead-placement penalty — the piece is mobile, but structurally committed to a dead end.

2. Extend `_closeable_mills` analysis to also check whether a 2-config whose closing square has both neighbours occupied will become closeable in ≤2 moves (i.e. one of the blocking pieces will need to move). If no such path exists, apply the patience penalty.

**Files:**

- `ai/heuristics.py` — `tactical_move_bonus()`: non-closeable 2-config penalty
- `ai/heuristics.py` — optional `_will_become_closeable(board, color, mill, horizon=2)` helper


### Enhancement B-99 — HumanDB Malom DTW overlay on the main game board ⬜ ★

**Description:** The HumanDB already stores `malom_dtw` (Malom depth-to-win for the current position) and `malom_dtw_after` (DTW for the position after each candidate move). Add a GUI toggle near the existing Sentinel/Malom overlay to display these values on the board.

**What to show:** For each square that is a valid next move from the current position, overlay the `malom_dtw_after` value as a badge on the target square — showing how many moves perfect play needs to win or lose from that resulting position. This lets the player compare "shortest path to forced win" (Malom) against "historically played moves" (human win%) in a single view.

**Requires:** `data/human_db.sqlite` built with `--malom-db` (so `malom_dtw_after` columns are populated). When the field is NULL (DB not annotated), the overlay shows nothing for that square.

**GUI placement:** New checkbox "DTW" in the board overlay control group (alongside Sentinel and Malom DB toggles). Enabled only when HumanDB is loaded and `malom_dtw_after` data is present.

**API change:** `GET /api/explorer/position` already returns `malom_dtw_after` per move. For the main game board, the existing `/api/hint` or a new `/api/board_dtw?fen=…` endpoint returns `{notation: dtw_after}` dict for the current position.

**Files:**
- `web/app.py` — add `/api/board_dtw` endpoint querying `_human_db.query_moves(board)`
- `web/static/game.js` — DTW overlay layer (badge rendering alongside Sentinel badges)
- `web/templates/index.html` — DTW toggle checkbox near Sentinel toggle
