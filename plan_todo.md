# Nine Men's Morris — Active Backlog

*New items go here. When an item is completed, move it to `plan_done.md`.*

---

## Implementation Roadmap

Track 1 (heuristic/phase-control) and SE-1 through SE-9 complete. Active priorities:

| Priority | Item | Key outcome |
|----------|------|-------------|
| ★★ | **B-61** | Cycling capture blind spot: close_mill bonus = 0 when mill opens+closes simultaneously ✅ 2026-05-28 |
| ★★ | **B-62** | own_convergence suppresses cycling mill closure (pivot piece leaves 2-config when mill closes) ✅ 2026-05-28 |
| ★★ | **B-59** | Sealed 2-config detection in move phase (forced mills, all rings) ✅ 2026-05-28 |
| ★★ | **B-60** | Cycling-capture unblock awareness (avoid enabling opponent mill on vacated square) ✅ 2026-05-28 |
| ★★ | **B-55** | Block opponent dual cardinal mill (placement phase) ✅ 2026-05-28 |
| ★ | **B-63** | Fly-entry position undervalued: opponent mobility inflated after entering fly phase ✅ 2026-05-28 |
| ★★ | **B-64** | Dead/near-dead placement penalty: pieces placed with 0 or 1 free adjacent squares are strategically trapped ✅ 2026-05-28 |
| ★★ | **SE-10** | Proactive fly-fork anticipation (move phase) |
| ★★ | **B-58** | Multiple LLM provider support (Claude, OpenAI, Perplexity, Null) |
| ★ | **B-51** | Expand retrograde solver beyond 3v3 |
| ★ | **B-57** | Direct-to-disk binary writing for endgame DB and fullgame DB (mmap; no large in-memory arrays) ✅ 2026-05-28 |
| ★ | **B-56** | Add D4 board symmetry to endgame database (8x size/speed reduction) ✅ 2026-05-28 |
| ★ | **SE-10** | Proactive fly-fork anticipation (move phase) |
| | **SE-11** | Opponent likelihood weighting via TrajectoryDB |
| | **SE-12** | Incremental evaluation cache (Zobrist-keyed) |
| | **SE-13** | N-gram opponent move predictor |

---

## DB / Infrastructure

### Bug B-26 — FullGameDB is never loaded by the server ✅ 2026-05-26
*(Archived — see plan_done.md)*

### Enhancement B-23 — Endgame position database builder ✅ 2026-05-26
*(Archived — see plan_done.md)*

### Enhancement B-27 — Make binary format the default fullgame DB output ✅ 2026-05-26
*(Archived — see plan_done.md)*

### Enhancement B-52 — FullGameDB: Frequency-Weighted Build from Human-Played Games ✅ 2026-05-26
*(Archived — see plan_done.md)*

### Enhancement B-24 — GUI settings for position DB usage ⬜

**Goal:** Add controls to the Settings and AI Tuning panels so the player can see which position databases are active and tune how strongly they influence the AI's play.

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

### Enhancement SE-14 — DB-Guided Horizon Search ✅ 2026-05-26
*(Archived — see plan_done.md)*

### Enhancement B-25 — Tools management page ✅ 2026-05-27
*(Archived — see plan_done.md)*

---

## Bug Reports

### Bug B-53 — ChromaDB embedding dimension mismatch when ollama_model changes ⬜

**Symptom:** `Error: Collection expecting embedding with dimension of 4096, got 2048`. Occurs when `ollama_model` in settings.json is changed (e.g. from `llama3.1:8b` → `gemma:2b`).

**Root cause:** `MemoryManager` uses the main LLM model for embeddings. When the user switches models, the embedding dimensionality changes but the existing ChromaDB collections still expect the old dimensions.

**Recommended fix:** Add `ollama_embed_model` to settings.json (default `nomic-embed-text`). `MemoryManager` always uses this fixed model for embeddings, independent of the main LLM.

**Files:**
- `data/settings.json` — add `ollama_embed_model` key (optional, default `nomic-embed-text`)
- `ai/memory_manager.py` — use `ollama_embed_model` for embeddings instead of `ollama_model`
- `web/app.py` — pass `ollama_embed_model` setting to `MemoryManager`

---

### Bug B-54 — LLM phase strategy guide never fed to MillsLLM ⬜

**Symptom:** `data/phase_strategy.md` exists (179 lines, phase-segmented NMM tactics guide) but is never injected into the LLM prompt.

**Fix:** In `Coordinator`, detect the current game phase (placement / move / fly) and inject the relevant section(s) from `phase_strategy.md` into the system prompt. The file is already segmented by phase (Phase A = placement 1–6, Phase B = placement 7–9, Phase C = movement, Phase D = fly).

**Files:**
- `ai/coordinator.py` — load `phase_strategy.md` once at init; add `_get_phase_context(board)` helper
- `ai/mills_llm.py` — accept optional `phase_context: str` parameter and prepend to system prompt

---

### Bug B-64 — AI places pieces with 0 or 1 free neighbours (dead/near-dead placement) ✅ 2026-05-28

**Symptom:** White AI (balanced personality) places at b2 (1 free neighbour) on turn 6 and d1 (0 free neighbours) on turn 8, yielding a piece permanently trapped from birth. No penalty exists for creating a piece with no future movement options, so the tactical score prefers positionally strong but mobility-dead squares.

**Root cause:** `tactical_move_bonus()` has no term that penalises placing a piece onto a square where it will have zero or near-zero free adjacent squares after placement. `_NEAR_BLOCKED_WEIGHTS["place"] = 0` means the static evaluator also ignores this.

**Fix:**
- Add `dead_placement_penalty: int = 600` and `near_dead_placement_penalty: int = 150` to `HeuristicWeights`.
- In `tactical_move_bonus()`, when `_is_placement and mills_delta == 0`, find the placed square, count `free_nb = sum(1 for nb in ADJACENCY[sq] if after.positions[nb] == "")`. Apply:
  - `free_nb == 0` → penalty = `dead_placement_penalty` (piece is permanently immobile)
  - `free_nb == 1` → penalty = `near_dead_placement_penalty * (placement_index / 8)` (scales with how far into placement we are — early game has fewer pieces and more mobility options later)
- Skip penalty when `mills_delta > 0` (piece is in a just-closed mill — it has value regardless of mobility).
- Add entry `("Dead/near-dead placement (B-64)", -placement_mobility_penalty)` to `_contributions`.
- Update all personality JSONs to include new fields.

**Files:**
- `ai/heuristics.py` — `HeuristicWeights`, `tactical_move_bonus()`
- `data/personalities/*.json` — add `dead_placement_penalty`, `near_dead_placement_penalty`
- `tests/test_tactics.py` — regression: at placement 8, d5 scores higher than d1; at placement 6, d2 scores higher than b2

---

### Bug B-55 — AI allows opponent to build two interconnected cardinal ring mills ✅ 2026-05-28 ★

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
- Add `_dual_cardinal_mill_alert(board, opp_color)` in `ai/heuristics.py`: returns True if opponent has 1 closed mill AND a 2-config in a second mill sharing a square with the first.
- Apply a block-bonus (~400) to any move that prevents the second such mill from forming.
- Urgency: equivalent to blocking a direct mill closure (P1 priority in `_order_moves`).

**Files:**
- `ai/heuristics.py` — `_dual_cardinal_mill_alert()`, apply in `tactical_move_bonus()`
- `tests/` — regression tests for both game sequences

---

### Bug B-56 — Copy button omits placement moves for setup-position games ⬜

**Symptom:** When using the "Copy" button to export a game position, the copied output only includes move-phase moves, not the placement moves that led to the current position.

**Fix:** Ensure the copy/export function includes all placement moves in the notation output, followed by movement phase moves.

**Files:**
- `web/static/game.js` — copy button handler: include placement moves in exported notation
- `web/app.py` — `/api/copy_game` or equivalent endpoint: return full game history

---

### Bug B-21 — Windows installer: improve model pull failure guidance ⬜

**Symptom:** After a failed `ollama pull`, the only feedback is a terse warning with no alternatives or guidance about how to change the model.

**Fix — `install.ps1`:** After a failed pull, print a help block listing lighter alternatives and instructions for updating `data/settings.json`. In the "Installation complete!" banner, if the model was not pulled, repeat the short version.

**Files:**
- `install.ps1` — step 8 failure block + completion banner
- `install.bat` — mirror the same guidance if applicable

---

### Bug B-17 — GUI text contrast too dim ⬜

**Symptom:** Many GUI labels, board coordinates, and control text are hard to read. `--text-dim: #8a7a60` is used widely.

**Fix:**
- `web/static/style.css` — raise `--text-dim` to approximately `#b7a78c`, or split into `--text-muted` (decorative) and `--text-label` (functional).
- Increase board coordinate / grid label contrast.
- Audit all `var(--text-dim)` uses and promote critical gameplay labels to `var(--text)` or the new `--text-label`.

**Files:**
- `web/static/style.css`
- `web/static/board.js` if board coordinate text is rendered separately

---

### Enhancement B-18 — Remove Bad Move button; add Force Move button for AI ⬜

**Goal:** Remove the Bad Move button and all related code. Replace with a **Force Move** button that lets the human player specify the next AI move.

**Bad Move removal scope:**
- `web/static/game.js`, `web/app.py`, `web/templates/index.html`, `web/static/style.css`
- `ai/game_ai.py` — remove bad_moves avoidance logic
- `data/bad_moves.json` — delete file

**Force Move button spec:**
- Visible only when it is the AI's turn
- Opens a modal: "Enter square to move to (and from, if move phase)"
- Validates against `get_all_legal_moves(board)`; rejects illegal moves
- Sends to server as override via `/api/force_ai_move`

**Files:**
- `web/app.py` — new `/api/force_ai_move` endpoint
- `web/static/game.js` — Force Move button + modal
- `web/templates/index.html` — Force Move button element

---

### Enhancement B-20 — Reward long-game trajectory lines in opening + midgame ⬜

**Goal:** Give extra weight to moves from previously played games that lasted at least ~25 moves.

**Recommended change:** In `TrajectoryDB`, track per stored line: total game length, deepest phase reached, whether loss occurred only in endgame. Add `survival_value` weighting: boosts moves from games that survived beyond ~25 moves; stronger in placement + move phase; zero in fly phase.

**Files:**
- `ai/trajectory_db.py`
- `ai/coordinator.py`
- `AI_INTERNALS.md` (update trajectory section)

---

### Bug B-31 — Opening play should still be recorded when the AI resigns ⬜

**Symptom:** Opening sequence is not being recorded properly when the AI resigns.

**Fix:**
- `web/app.py` — verify the resignation path persists the game record and opening line before any early return.
- `ai/opening_book.py` / training pipeline — ensure resignation games still contribute opening statistics.
- Add a regression test: AI resigns after a legal opening, and that opening sequence is still present in the stored game record.

**Files:**
- `web/app.py`
- `ai/opening_book.py`
- `ai/memory_manager.py`

---

### Enhancement B-32 — Increase AI reasoning / commentary transparency ⬜

**Goal:** Commentary/debug output should identify the dominant reason for the AI's move choice (immediate mill closure, mandatory block, busy-chain win, fork prevention, convergence disruption, cardinal-lane block, mobility squeeze, trajectory exploit, endgame DB recognition, opening-book adherence).

**Fix:**
- `ai/game_ai.py` — capture a structured explanation object for the selected move listing top scoring features / bonuses / blockers.
- `ai/coordinator.py` — expose those reasons in commentary, debug logs, and optional dev overlays.
- `web/static/game.js` — display a richer "AI thought process" summary when commentary mode is enabled.

**Files:**
- `ai/game_ai.py`
- `ai/coordinator.py`
- `web/static/game.js`

---

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
- If neither: apply a `sterile_fork_penalty` (default ~100) on the last placement.
- Scale `setup_mill` bonus down ~40% on placement 9 unless the setup is immediately actionable.

**Files:**
- `ai/heuristics.py` — `tactical_move_bonus()`, late-placement window checks

---

### Bug B-35 — Final placements: prefer dual-purpose block-and-build over passive 2-config ⬜ *(implementation covered by B-28)*

**Symptom:** On the last placement the AI creates a 2-piece setup that ignores an opponent mobile mill, when a dual-purpose square would both block and create own pressure.

**Example game:**
```
1.d6 d2    2.f4 b4    3.g7 g4    4.d7 d5    
5.a7xd5 d5   6.f6 f2    7.b6xd5 d5   8.c4 b2xc4    
9.d3 e5  ← Black's last placement — passive 2-config
```
Placing at `a4` instead would both block `a4-b4-c4` and create a 2-config approach.

**Fix:**
- Add a `dual_purpose_final_bonus` (~150) for a placement that simultaneously blocks an opponent active mill line AND creates a new own 2-config.
- Weight this bonus higher on placements 8–9.

**Files:**
- `ai/heuristics.py`

---

### Enhancement B-45 — Replace automatic AI resignation with an offer of defeat ⬜

**Goal:** Change automatic AI resignation into an offer of defeat that the human player can accept or decline.

**Suggested implementation:**
- `web/app.py` — replace the immediate resignation branch with an offer state stored in the session model
- `web/static/game.js` — show a UI prompt with accept-decline controls
- Ensure opening and game records are still persisted regardless of outcome

**Files:**
- `web/app.py`
- `web/static/game.js`
- `web/templates/index.html`
- `AI_INTERNALS.md`

---

### Tactical bug — Black failed to close its own mill and missed White's immediate threat

**Game sequence (regression test needed):**
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

**Reported issue:** At Black's move 10, the AI played `b4-b2`. It should have either:
1. Closed its own mill via `d2-b2` (closing the b-line mill with `b6`).
2. Blocked White's imminent mill threat (`f4-g4`).

**What to check:**
- [ ] Reconstruct position after move 10; verify `d2-b2` is legal and recognized by move generator.
- [ ] Check whether immediate mill-closing bonus is insufficient vs positional reshuffling.
- [ ] Check whether opponent immediate mill threats are underweighted in move phase.
- [ ] Check whether dual-purpose value of `d2-b2` is recognised.
- [ ] Add regression test asserting Black strongly prefers `d2-b2` over `b4-b2`.

---

### Bug B-59 — AI misses forced mills in move phase (sealed 2-config) ✅ 2026-05-28 ★★

**Symptom:** In the move phase, the AI fails to recognise and pursue a forced mill where both empty closing squares are accessible only to its own pieces (no opponent can reach either square to block). The AI drifts to known-good oscillation or cardinal-node mobility moves instead.

**Motivating game** (White = Scholar, move phase begins after 9 placements each):
```
After placement: White on a7,g7,g4,a4,d3,c4,e4,f2,b6 / Black on d7,d6,b4,a1,g1,f6,d2,b2,d5
10.g4-f4 g1-g4 / 11.d1-g1 d2-d1 / 12.f4-e4 b2-d2
```
After move 12, White has **d3, c4, e4** on the board. The inner ring bottom side **e3-d3-c3** is a forced two-move mill:
- `c3` is adjacent only to `d3` (White) and `c4` (White) — no opponent piece can reach it.
- `e3` is adjacent only to `e4` (White) and `d3` (White) — no opponent piece can reach it.
Path A: `c4→c3`, then `e4→e3` (no Black piece can prevent e3). Path B: `e4→e3`, then `c4→c3`.
White instead plays `f4→e4` (cardinal oscillation), missing the forced mill entirely.

**Root cause (three layered failures):**

1. **Static eval weight too small.** `_WEIGHTS["move"] = (30, 48, 12, 5, 50, 0)` — `two_cfg` weight = **5**. A normal 2-config contributes 5 to static eval; far too small to signal a forced win through deep negamax.

2. **`setup_mill` bonus is root-only and undifferentiated.** `tactical_move_bonus` adds `int(weights.setup_mill * 1.3) * two_cfg_gained` (Scholar: `195`) to the root move score. This cannot propagate sealed mill urgency into the alpha-beta tree, and treats a sealed 2-config the same as an easily-blocked one.

3. **Move ordering ignores sealed 2-configs.** `_order_moves` puts sealed-2-config-creating moves in bucket P2 (history-sorted), behind direct mill closes (P0) and opponent-mill blocks (P1). The sealed threats never surface early enough to guide search efficiently.

**Fix — three coordinated changes:**

**A. `_sealed_two_configs(board, color) -> int` in `ai/heuristics.py`**

A "sealed" 2-config is a 2-config whose empty closing square satisfies:
1. No opponent piece is adjacent to the closing square (`all(board.positions[nb] != opponent for nb in ADJACENCY[closing_sq])`).
2. Guard: `_closeable_mills(board, opponent) == 0` — opponent cannot immediately close a mill of its own (which would let it capture the piece sealing the closing square before we act).

```python
def _sealed_two_configs(board: BoardState, color: str) -> int:
    opponent = "B" if color == "W" else "W"
    if _closeable_mills(board, opponent) > 0:
        return 0   # guard: opponent can punish immediately
    count = 0
    for mill in MILLS:
        own   = sum(1 for p in mill if board.positions[p] == color)
        empty = [p for p in mill if board.positions[p] == ""]
        if own == 2 and len(empty) == 1:
            closing = empty[0]
            if all(board.positions[nb] != opponent for nb in ADJACENCY[closing]):
                count += 1
    return count
```

**B. Boost sealed threat in static eval**

In `evaluate()`, add a `sealed_two_cfg` term to the move-phase call beside the existing `two_cfg` term:

```python
sealed_w = _sealed_two_configs(board, "W")
sealed_b = _sealed_two_configs(board, "B")
sealed_score = (sealed_w - sealed_b) * SEALED_TWO_CFG_WEIGHT   # target weight: 18–22
```

`SEALED_TWO_CFG_WEIGHT` should be a constant ~18 (3-4× the regular `two_cfg` weight of 5) so the term propagates clearly through even 2–3 plies of negamax.

**C. Elevate sealed-2-config-creating moves in `_order_moves` in `ai/game_ai.py`**

After the existing P0 (direct mill close or fork) bucket, add a new **P0.5** bucket for moves that create a new sealed 2-config:

```python
# Compute post-move sealed count for each candidate (lightweight: only call for move-phase)
if board.phase == "move":
    sealed_before = _sealed_two_configs(board, color)
    sealed_creates = {
        m for m in moves
        if _sealed_two_configs(board.apply_move({"from": m[0], "to": m[1], "capture": None}), color) > sealed_before
    }
else:
    sealed_creates = set()

# Priority buckets
p0  = [m for m in moves if m[1] in close or _is_fork(m)]
p05 = [m for m in moves if m not in p0 and m in sealed_creates]   # NEW
p1  = [m for m in moves if m not in p0 and m not in p05 and (m[1] in block or _is_squeeze(m))]
p2  = [m for m in moves if m not in p0 and m not in p05 and m not in p1]
```

**D. `sealed_setup_bonus` in `tactical_move_bonus`**

Add a large root-level bonus for moves that create a new sealed 2-config:

```python
sealed_after  = _sealed_two_configs(new_board, color)
sealed_before = _sealed_two_configs(board, color)
sealed_gained = max(0, sealed_after - sealed_before)
sealed_setup_bonus = int(weights.close_mill * 0.75) * sealed_gained   # ~243 for Scholar
```

This root-only bonus supplements the static eval term. Together they ensure the AI both evaluates sealed positions correctly in the tree and selects them decisively at the root.

**Regression test (required):**

```python
# After move 12 (f4-e4 played by White, b2-d2 by Black), reconstruct board.
# White: a7,g7,g4(moved to f4),a4,d3,c4,e4,f2,b6 minus piece at f4 plus at e4...
# Exact FEN: build from game trace.
# Assert: AI (White, difficulty ≥ 3) at this position selects c4→c3 or e4→e3, NOT f4→e4 or e4→e5.
```

**Files:**
- `ai/heuristics.py` — `_sealed_two_configs()`, `SEALED_TWO_CFG_WEIGHT` constant, `evaluate()` sealed term, `tactical_move_bonus()` sealed_setup_bonus
- `ai/game_ai.py` — `_order_moves()` P0.5 bucket for sealed-2-config-creating moves
- `tests/test_heuristics.py` — unit test for `_sealed_two_configs` on the move-12 position
- `tests/test_game_ai.py` — regression test: move selection asserts c4→c3 or e4→e3

---

### Bug B-60 — Cycling-capture unblock: AI ignores opponent threats enabled by vacating the mill ✅ 2026-05-28 ★★

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
- White should capture **a1** (removes one leg of Black's pending mill), but instead captures **f4** (a cardinal node, scored higher by `capture_feeder_bonus` / `capture_diamond_bonus`).

**Root cause:**

The capture-selection heuristics in `tactical_move_bonus` are fully reactive — they score the opponent piece being removed, not the opponent threat that will be unblocked next turn. None of the five existing capture bonuses (`capture_feeder_bonus`, `capture_diamond_bonus`, `safe_capture_bonus`, `capture_creates_diamond_bonus`, `capture_activates_feeder_bonus`) model the constraint:

> *"This cycling mill will oscillate. When I vacate the oscillation square next turn, does the un-captured opponent piece complete a mill?"*

**Fix — `_cycling_capture_unblock_penalty` in `tactical_move_bonus`:**

When a move closes a mill on a **cycling-ready** line (i.e., the piece that was just moved can oscillate back next turn — it has an adjacent empty square), evaluate each legal capture against the following check:

```python
def _cycling_unblock_penalty(board_after_capture: BoardState,
                              color: str,
                              cycling_sq: str) -> int:
    """
    Score penalty: if the cycling piece at cycling_sq moves away next turn,
    does the resulting board give the opponent an immediate mill closure?
    """
    opponent = "B" if color == "W" else "W"
    # Simulate vacating the cycling square
    sim = dict(board_after_capture.positions)
    sim[cycling_sq] = ""
    sim_board = board_after_capture  # lightweight: only need positions for mill check
    # Check every mill containing cycling_sq
    for mill in MILLS:
        if cycling_sq not in mill:
            continue
        opp_count = sum(1 for p in mill if (sim[p] == opponent or (p == cycling_sq and sim[p] == "")))
        # Recount with cycling_sq empty
        opp_in_mill = sum(1 for p in mill if p != cycling_sq and board_after_capture.positions[p] == opponent)
        empty_in_mill = sum(1 for p in mill if p != cycling_sq and board_after_capture.positions[p] == "")
        if opp_in_mill == 2 and empty_in_mill == 0:
            # cycling_sq is the only non-opponent square; vacating it hands opponent a mill
            return weights.cycling_mill   # recycle the personality's cycling weight as a penalty scale
    return 0
```

For each candidate capture `cap`, compute `_cycling_unblock_penalty(board_after_cap, color, cycling_sq)` and subtract it from the move score. This makes capturing the piece that contributes to the blocked mill worth more than capturing a high-value but irrelevant piece.

**New personality weight:** `cycling_capture_unblock` (default 180). Subtract this from any capture that leaves an opponent 2-config with a closing square that is the next vacated square.

**`cycling_capture_unblock` in `data/personalities/*.json`:** add to all personality files with value 180 (tunable).

**Regression test (required):**

```python
# Reconstruct board at move 22 (White to move, g4→g7 closes cycling mill).
# White pieces: a7, g1, g4, e4, c4, d3, a4, b4, b6
# Black pieces: a1, a4(moved from), d1, c5, f4, b2, d6, d2 — adjust from exact trace
# Assert: after White plays g4→g7 (closing cycling mill on g-column),
#   the AI selects capture a1 (not f4 or other cardinal node).
```

**Files:**
- `ai/heuristics.py` — `_cycling_unblock_penalty()` helper, `tactical_move_bonus()` apply penalty per candidate capture
- `ai/game_ai.py` — no structural change needed (penalty applied inside existing capture loop)
- `game/rules.py` or `ai/heuristics.py` — `_is_cycling_ready(board, sq) -> bool` helper (piece at sq has ≥1 adjacent empty square)
- `data/personalities/aggressive.json`, `balanced.json`, `defensive.json`, `positional.json`, `scholar.json`, `custom.json` — add `cycling_capture_unblock: 180`
- `tests/test_heuristics.py` — unit test for penalty function on the move-22 board
- `tests/test_game_ai.py` — regression test: assert capture = a1 at move 22

---

### Bug B-61 — Cycling capture receives zero close_mill bonus (gross vs net mill delta) ⬜ ★★ High Priority

**Symptom:** During the move phase, the AI refuses to execute winning cycling-mill captures. When a move simultaneously opens one closed mill and closes another (the cycling pattern), `tactical_move_bonus()` computes `mills_delta = max(0, _closed_mills(after) - _closed_mills(before)) = 0` because the net change is zero. The move receives no `close_mill_contribution` bonus (~500 pts) despite enabling a capture. The resulting ~400-point scoring gap causes the AI to prefer idle positional shuffles indefinitely.

**Motivating example (game — turn 17):**

Position: White {a7, d6, f2, a4, d1, b6, a1, d5} (8 pieces, closed mill: a7-a4-a1)  
Black: {b4, g1, d2, b2} (4 pieces, no 2-configs)

Correct move `a7→d7 xb4`:
- Opens `a7-a4-a1` (a7 leaves) and closes `d5-d6-d7` (a7 arrives at d7)
- `_closed_mills(before) = 1`, `_closed_mills(after) = 1` → `mills_delta = max(0, 0) = 0` → **no close_mill bonus**
- Black drops to 3 pieces and enters fly phase — a near-winning position for White

White instead repeats `f2↔f4` indefinitely; the game never finishes.

**Root cause — line 1855 of `ai/heuristics.py`:**

```python
mills_delta = max(0, _closed_mills(after) - _closed_mills(before))   # net delta
```

A cycling move scores net 0 regardless of the capture it enables.

**Fix — compute gross newly-closed mills:**

```python
before_closed_set = {
    tuple(sorted(m)) for m in MILLS
    if all(board.positions[p] == color for p in m)
}
after_closed_set = {
    tuple(sorted(m)) for m in MILLS
    if all(new_board.positions[p] == color for p in m)
}
mills_delta = len(after_closed_set - before_closed_set)   # replaces net-delta line
```

This mirrors the existing `mill_opened` variable (line 1924) which already counts gross opened mills; the symmetry makes the fix self-consistent.

**Secondary fix — `_mill_threats()` phantom 2-config overcount:**

`f2→f4` is also inflated by `herding_coverage` (+40) because f4 is adjacent to f2, which `_mill_threats()` counts as the closing square for Black's "2-config" f2-d2-b2. This 2-config is a phantom: moving d2→f2 leaves d2 empty while b2 is already in the line — the mill still cannot close. Fix in `_mill_threats()` (line ~464): when testing if any friendly piece is adjacent to a closing square, exclude pieces that are already inside the same mill being counted. This prevents `_closed_mills(before) - _closed_mills(after)` from going wrong — but more directly, it prevents `herding_coverage` from rewarding adjacency to impossible-to-close mills.

**Files:**
- `ai/heuristics.py` — `tactical_move_bonus()`: replace net `mills_delta` with gross `mills_delta = len(after_closed_set - before_closed_set)`; `_mill_threats()`: exclude pieces already in the same counted mill from the adjacency check

**Tests:**
- Unit test: construct the turn-17 board; assert `tactical_move_bonus(board, "W", move_a7_d7_xb4)` ≥ 450 (close_mill_contribution fires)
- Regression: AI (White, any difficulty) at the turn-17 position selects `a7→d7`, not `f2→f4`

---

### Bug B-62 — `own_convergence` suppresses execution of cycling mills that share a pivot ⬜ ★★ High Priority

**Symptom:** When White's two active 2-configs share a pivot piece (e.g., d6 is pivot for both `b6-d6-f6` and `d5-d6-d7`), the `own_convergence` bonus (+250) rewards White for keeping d6 as a convergence pivot. When White cycles a mill and a piece lands at d7 (closing `d5-d6-d7`), the 2-config becomes a closed mill — d6 is no longer a 2-config pivot. The convergence bonus drops from +250 to 0. This ~250-point static-eval loss partially offsets the close_mill bonus restored by B-61, and can still tip the scale against the cycling capture in some positions.

**Root cause:**

`evaluate()` computes `own_convergence` as a static term counting 2-config pairs sharing a pivot or closing square. After a 2-config closes into a mill (even when closing creates a capture), the convergence term drops. The evaluator treats a mill closure as a structural loss.

**Fix — neutralize the convergence loss when a mill was just closed:**

In `tactical_move_bonus()`, after the gross `mills_gross_closed` computation from B-61, add a restoration term:

```python
if mills_gross_closed > 0:
    # Closing a mill is structurally good. Counteract the static-eval drop
    # in own_convergence that fires when the closed 2-config loses its pivot status.
    score += weights.own_convergence * mills_gross_closed
```

This is applied once per newly-closed mill, matching the scale of the convergence bonus that was suppressed. It does not affect moves where no mill was closed.

**Note:** Implement B-61 first and re-test. B-62 is only needed if a scoring gap > 150 pts persists after B-61. The gap at turn 17 is approximately -250 from own_convergence on top of the -408 from the net-delta bug; after B-61 restores ~500 pts, the residual gap is ~250-408+500 ≈ +92 in favour of the correct move — meaning B-62 may be unnecessary. Verify empirically before implementing.

**Files:**
- `ai/heuristics.py` — `tactical_move_bonus()`: add `own_convergence * mills_gross_closed` restoration when `mills_gross_closed > 0`

**Tests:**
- Unit test: own_convergence restoration fires correctly when move closes a mill that shared a pivot in a convergence pair
- Regression: turn-17 move selection is correct after B-61; add B-62 only if gap persists

---

### Bug B-63 — Fly-entry position undervalued: opponent mobility over-counted on entering fly phase ⬜ ★ Medium Priority

**Symptom:** Immediately after White captures an opponent piece that drops Black to 3 pieces (fly phase), `_mobility(B)` returns the number of empty squares on the board (~13–15). This makes Black appear highly mobile. In `evaluate()`, the `(own_mob - opp_mob) × 8` term becomes strongly negative (`8 × (4 - 13) = -72`), penalising the position White just achieved. The AI systematically undervalues captures that send the opponent into fly phase — the very captures that are winning.

**Root cause — `_mobility()` line ~451 of `ai/heuristics.py`:**

```python
if board.pieces_on_board[color] <= 3:
    return len([p for p in POSITIONS if board.positions[p] == ""])
```

Fly phase mobility = empty squares ≈ 13–15. Normal move-phase mobility ≈ 3–6. The differential swings by ~70 points in the wrong direction on the ply where White makes its best move.

**Fix — cap fly-phase mobility in the mobility differential:**

```python
if board.pieces_on_board[color] <= 3:
    return 5   # constant cap; fly pieces can jump anywhere, so raw mobility is misleading
```

A constant of 5 (matching typical move-phase values) prevents fly-entry from appearing worse than the position before capture. An alternative is to use a separate `fly_mob_weight` field in `HeuristicWeights` (initially 0) and multiply the fly mobility by that instead of `mob_weight`, suppressing the fly mobility contribution entirely.

The simple cap is preferred: fewer fields, same effect, easier to reason about.

**Impact:** Prevents the ~50–80 point penalty that discourages capturing moves that send the opponent into fly. After B-61 and B-62, this is unlikely to be the deciding factor — but it causes systematic mis-scoring of whole game sub-trees whenever the fly transition is in the search horizon.

**Files:**
- `ai/heuristics.py` — `_mobility()`: return a capped constant (5) for fly-phase pieces instead of empty-square count

**Tests:**
- Unit test: `_mobility(board, "B")` returns ≤ 5 when Black has 3 pieces, regardless of board fill
- Regression: AI (White) at a position one ply before sending Black into fly correctly prefers the capturing move

---

### Note — GUI slider set is missing evolved heuristic weights

**Symptom:** `tools/evolve_weights.py` tunes more heuristic fields than the web slider panel exposes. `HeuristicWeights` has 36 fields; the GUI exposes ~22.

**Hidden weights currently tuned but not in GUI:** `capture_disrupt_diamond`, `capture_disrupt_feeder`, `convergence_block`, `convergence_disrupt`, `convergence_penalty`, `cross_feed_mobility`, `herding_squeeze`, `locked_mill_penalty`, `mill_trap_build`, `mobility_reduction`, `own_convergence`, `placement_busy_scan`, `ring_crowding_penalty`, `sacrifice_viable`.

**Fix:** Bring the frontend slider list into sync with `HeuristicWeights`, or explicitly split the dataclass into "UI-exposed" and "internal-only" weights.

---

### Evolve weights v2 — cross-personality master tuning

**Task:**
- [ ] Extend `tools/evolve_weights_v2.py` so it can evolve **one additional Master personality's weight set** while evaluating it against the other personalities.

**Recommendation:**
- Add `--target-personality <name>` mode that selects one Master personality as the mutable candidate.
- Keep other personalities fixed during each evaluation batch; rotate opponents so candidate isn't overfitting.
- Save outputs separately per personality: `data/weights/master_<name>_best.json`.
- Log which opponent personalities were faced in each generation.

---

## Search & Evaluation Enhancements (SE-1 through SE-9 complete ✅)

### TIER 3 — Solid, Secondary Priority

### SE-10 — Proactive Fly-Fork Anticipation (Move Phase) ⬜ ★ Medium Impact

**Why:** The existing `fly_fork_bonus` fires reactively. Extend `_fork_in_n(board, opp, n=2)` (already used in placement-phase, Enhancement B-4) to the move phase: scan forward up to 3 half-moves for forcing lines that result in 2+ simultaneous 2-configs.

**Deliverables:**
- `ai/heuristics.py` — `_move_phase_fork_anticipation(board, color, depth=3)`; bonus `fork_depth × 80` added to root move score

---

### SE-11 — Opponent Likelihood Weighting (Asymmetric Depth via TrajectoryDB) ⬜ ★ Medium Impact

**Why:** Standard alpha-beta allocates equal depth to all opponent responses regardless of how likely they are. Use existing `TrajectoryDB` move frequency to drive +1 extension for high-frequency opponent moves and −1 LMR for rare ones.

**Deliverables:**
- `ai/trajectory_db.py` — `query_move_frequency(prefix, notation)` method returning normalised frequency [0.0, 1.0]
- `ai/game_ai.py` — apply frequency-based depth delta at opponent nodes inside `_negamax`

---

### TIER 4 — Infrastructure / Long-Term

### SE-12 — Incremental Evaluation Cache (Zobrist-Keyed Sub-Functions) ⬜

**Why:** Heavy heuristic sub-calls recompute from scratch every leaf call. With Zobrist hashing already in place (SE-1), a secondary cache keyed by board hash stores sub-function results. Requires SE-1.

**Deliverables:**
- `ai/heuristics.py` — result cache dict keyed by Zobrist hash for top-cost sub-functions; invalidate on `apply_move`

---

### SE-13 — N-Gram Opponent Move Predictor ⬜

**Why:** Complements TrajectoryDB (win/loss rates) with a pure move-frequency bigram/trigram model. Feeds into SE-11 with richer per-sequence predictions.

**Deliverables:**
- `ai/ngram_opponent_model.py` — new `NGramOpponentModel` class; `update()` called after each game; `predict()` returns probability dict; trained incrementally from `data/games/` JSONL records

---

### SE-14 — DB-Guided Horizon Search (FullGameDB + Negamax Hybrid) ⬜ ★ High Impact

**Why:** Currently `_negamax` rebuilds the full search tree from scratch on every move decision — even for positions already exactly solved in `FullGameDB`. Moving the DB lookup inside `_negamax` lets the search consume DB coverage as perfect depth-∞ oracle calls. When the DB covers the first K plies of the game tree, the AI only spends its time budget searching from the *frontier* of known territory.

**How it works:**
1. At every internal `_negamax` node (not just root), query `FullGameDB` for the current position.
2. If an exact outcome is found (`outcome ∈ {WIN, LOSS, DRAW}`) → return `±(INF − depth)` immediately.
3. If the DB knows a best move but no definitive outcome → promote that move to the front of the move list (same as TT-best-move promotion from SE-1), then continue normal search.
4. If no DB match → continue normal negamax.

**Legal-move safety:** Validate DB best move against `get_all_legal_moves(board)` before promoting; fall through silently on mismatch.

**Evaluation order inside `_negamax`:** terminal check → SE-4 endgame probe → SE-14 FullGameDB probe → SE-8 extension → depth-0 / SE-9 quiescence → TT probe → search loop.

**Build prerequisite:** B-52 (frequency-weighted build from human games) ensures the DB is dense in positions that actually occur, maximising SE-14's hit rate. SE-14 degrades gracefully to full negamax when the DB is absent or the position is not covered.

**Deliverables:**
- `ai/game_ai.py` — DB probe at top of `_negamax`, after SE-4, before SE-8; exact outcomes short-circuit search; best-move hints promote front-of-list (with legality check); guarded by `self._fullgame_db is not None`
- `ai/fullgame_db.py` — `best_move_validated(board)` helper that maps canonical move back to actual orientation AND verifies against legal moves

---

## B-51 — Early-Endgame DB: expand retrograde solver beyond 3v3 ⬜ ★ High Impact

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
| **Tier 1 total** | | | **~79 MB** |

**Practical tiers:**
- **Tier 1 — Recommended:** 4v3, 3v4, 5v3, 3v5, 4v4 → ~79 MB total.
- **Tier 2 — Optional:** add 5v4, 4v5 → ~244 MB total.
- **Tier 3 — Large/optional:** 6v3, 3v6, 7v3, 3v7, 6v4, 7v4.

**Key algorithm changes vs the existing 3v3 builder:**

1. **Mixed fly/move successor generation:** a side with exactly 3 pieces flies; a side with ≥4 moves along adjacency edges.
2. **Cross-table captures:** a capture in nWvnB leaves nWv(nB-1) or (nW-1)vnB — successor lives in a different already-solved table.
3. **Build order:** each table depends on both smaller tables from captures. Solve in order of (nW + nB) ascending.
4. **File naming:** `endgame_{nW}_{nB}.wdl` alongside `endgame_3_3.wdl`.
5. **Query integration:** extend `EndgameSolvedDB` to load all available files; `query()` dispatches by `(len(w_pieces), len(b_pieces))`.

**Files:**
- `tools/build_endgame_db.py` — rewrite to accept `--nW` and `--nB` args; mixed fly/move successor generator; cross-table reference loading
- `ai/endgame_solved_db.py` — extend `EndgameSolvedDB.__init__` to load all available tables; extend `query()` to dispatch by piece count

**Prerequisite:** B-57 (direct-to-disk mmap writing) must land first — the 5v4 and 4v5 tables (82 MB each) will exceed practical `bytearray` size without it.

---

## B-57 — Direct-to-disk binary writing for endgame DB and fullgame DB ✅ 2026-05-28 ★

**Goal:** Replace the in-memory `bytearray` (endgame) and large intermediate structures (fullgame) with direct writes to a pre-allocated binary file using Python's `mmap`. All solve and fill passes operate on the memory-mapped file handle instead of RAM. A final label pass then edits the file in-place to mark winning and losing trajectory positions.

**Why this matters:**
- 3v3 is 1.3 MB — RAM is not the constraint today.
- B-51 tables (5v4, 4v5) reach 82–330 MB each; 6v5 exceeds 1 GB. These will not fit in a Python `bytearray` on most build machines.
- The fill pass (non-canonical → canonical propagation) and label pass (WIN/LOSS annotation) are sequential scans with local reads — exactly the workload OS page caching excels at with mmap.

---

### Endgame DB (`tools/build_endgame_db.py`)

**Current flow:**
1. `table = bytearray(n_bytes)` — entire table allocated in RAM as UNKNOWN.
2. Solve passes iterate `canonical_ids`, set WIN/LOSS/DRAW in `table`.
3. Fill pass iterates all positions, copies from canonical entry.
4. Caller does `wdl_path.write_bytes(bytes(table))`.

**Proposed flow:**
1. Pre-allocate the `.wdl` file on disk: `path.write_bytes(bytes(n_bytes))` (zeros = UNKNOWN = valid starting state).
2. Open the file and `mmap.mmap(f.fileno(), n_bytes)` — the OS manages which pages are in RAM.
3. All solve passes, fill pass, and DRAW marking operate on the mmap handle via the existing `get_wdl`/`set_wdl` API — **no API change required**.
4. `table.flush(); table.close()` — file is already on disk; no bulk write at the end.
5. **Label pass** (new, optional): after the file is fully solved, re-open it and mark a chosen subset of positions with a high-contrast sentinel (e.g. reuse `WDL_WIN`/`WDL_LOSS` is already the label — but if a richer annotation is needed, extend to 3 bits per entry or a sidecar index file mapping `pos_id → outcome + move`).

**Key change in `solve_table`:**
```python
# Before
table = bytearray(n_bytes)

# After
out_path.write_bytes(bytes(n_bytes))          # pre-allocate (UNKNOWN = 0x00)
f = open(out_path, "r+b")
table = mmap.mmap(f.fileno(), n_bytes)
try:
    _solve_passes(table, ...)
    _fill_pass(table, ...)
    table.flush()
finally:
    table.close()
    f.close()
```

`solve_table` returns `None` (file is already written); callers that need the bytes for sub-table loading use `open(path, "rb").read()` or a second mmap.

---

### Fullgame DB (`tools/build_fullgame_db.py`)

**Done (by B-52):** SQLite was already replaced with a sorted binary `.bin` format (36-byte records, mmap'd read-only at query time). A `--max-gb` guard prevents OOM during BFS. No further changes needed for B-57.

### Files changed
- `tools/build_endgame_db.py` — replaced `bytearray` with mmap; `solve_table` writes to `out_path`, returns None ✅
- `ai/endgame_solved_db.py` — no change needed; `get_wdl`/`set_wdl` work on any `[]`-indexable type ✅
- `tools/build_fullgame_db.py` — binary format already done (B-52); no change ✅
- `ai/fullgame_db.py` — binary reader already in place (B-52); no change ✅

**Prerequisite B-57 → B-51 satisfied.** B-51 (expand retrograde solver beyond 3v3) can now proceed.

---

---

## Enhancement B-58 — Multiple LLM Provider Support ⬜ ★★

**Goal:** Replace the Ollama-only LLM integration with a pluggable provider abstraction. Allow the user to choose between Ollama (local), Claude (Anthropic), ChatGPT (OpenAI), Perplexity, or no LLM at all — without changing any game logic.

---

### Design constraints

- API keys **never** go in `data/settings.json` (git-tracked). Use env vars only: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `PERPLEXITY_API_KEY`.
- Embedding provider is **separate** from the chat provider. Default: ChromaDB `DefaultEmbeddingFunction` (local, no key required). `ollama_embed_model` setting (B-53) remains available if user wants Ollama embeddings.
- No streaming for v1 — all calls are blocking `chat(system, messages) -> str`.
- On any error, return `""` (empty string) — matches current `_chat()` behaviour.
- Graceful fallback: if a selected provider's package is not installed, log a warning and fall back to Null provider rather than crashing.

---

### New module: `ai/llm_provider.py`

```python
class BaseLLMProvider(ABC):
    @abstractmethod
    def chat(self, system: str, messages: list[dict]) -> str: ...
    def available(self) -> bool: return True

class OllamaProvider(BaseLLMProvider):     # existing behaviour, wraps requests
class ClaudeProvider(BaseLLMProvider):     # anthropic SDK >= 0.20
class OpenAIProvider(BaseLLMProvider):     # openai SDK >= 1.0
class PerplexityProvider(BaseLLMProvider): # openai-compatible base URL
class NullProvider(BaseLLMProvider):       # always returns ""

def make_provider(settings: dict) -> BaseLLMProvider:
    """Factory — reads settings['llm_provider'] and env vars."""
```

`messages` format matches the OpenAI schema: `[{"role": "user"/"assistant", "content": "..."}]`.

`OllamaProvider` translates this internally to the Ollama `/api/chat` format (already what `MillsLLM._chat` does).

---

### Provider defaults

| Provider | Default model setting key | Default model |
|----------|--------------------------|---------------|
| `ollama` | `ollama_model` | *(existing setting)* |
| `claude` | `claude_model` | `claude-sonnet-4-6` |
| `openai` | `openai_model` | `gpt-4o-mini` |
| `perplexity` | `perplexity_model` | `sonar` |
| `null` | — | — |

---

### `data/settings.json` additions

```json
"llm_provider": "ollama",
"claude_model": "claude-sonnet-4-6",
"openai_model": "gpt-4o-mini",
"perplexity_model": "sonar",
"embed_provider": "default"
```

`embed_provider` values: `"default"` (ChromaDB `DefaultEmbeddingFunction`), `"ollama"` (uses `ollama_embed_model`).

---

### Changes to existing files

**`ai/mills_llm.py`**
- `__init__` replaces `(url, model)` params with `(provider: BaseLLMProvider)`.
- Remove `_chat()` helper (logic moves to `OllamaProvider.chat()`).
- `get_move_commentary()` / `get_strategy_commentary()` call `self._provider.chat(system, messages)`.

**`ai/memory_manager.py`**
- Accept `embed_provider: str` param (default `"default"`).
- If `embed_provider == "ollama"`, use `OllamaEmbeddingFunction(model=ollama_embed_model)`.
- Otherwise use `DefaultEmbeddingFunction()`.

**`web/app.py`**
- Call `make_provider(settings)` once at startup.
- Pass `provider` to `MillsLLM(provider)` and `embed_provider` to `MemoryManager`.

**`requirements.txt`**
- Add optional deps with comments:
  ```
  # Optional: anthropic>=0.20   # for Claude provider
  # Optional: openai>=1.0       # for OpenAI / Perplexity provider
  ```

---

### Settings UI additions

In the Settings panel:

| Control | Type | Notes |
|---------|------|-------|
| LLM Provider | Dropdown | `ollama / claude / openai / perplexity / none` |
| Model | Text input | Shows the active model setting for the chosen provider |
| API key status | Read-only badge | `✓ key found` (green) or `✗ key missing` (amber) — reads from `/api/provider_status` |
| Embed provider | Dropdown | `default (local) / ollama` |

New endpoint: `GET /api/provider_status` → `{ "provider": "claude", "model": "claude-sonnet-4-6", "key_present": true, "embed_provider": "default" }`.

When `key_present` is false, the badge is amber with text "Set `ANTHROPIC_API_KEY` env var". Commentary is silently disabled (Null fallback) — the game still works.

---

### Files

- `ai/llm_provider.py` — new; `BaseLLMProvider`, five concrete classes, `make_provider()`
- `ai/mills_llm.py` — constructor change + remove `_chat()` helper
- `ai/memory_manager.py` — `embed_provider` param
- `web/app.py` — factory call, new `/api/provider_status` endpoint
- `data/settings.json` — four new keys
- `web/templates/index.html` — provider dropdown, model field, key-status badge, embed dropdown
- `web/static/game.js` — load/save provider settings; call `/api/provider_status` on settings open
- `requirements.txt` — optional dep comments

---

## Architecture Principles

- **Immutable board state** — `BoardState.apply_move()` always returns a new object.
- **Coordinator owns the narrative** — All commentary and LLM calls flow through `Coordinator`. `GameAI` is pure search.
- **No cloud dependency** — All LLM inference runs locally via Ollama.
- **Progressive enhancement** — Every stage adds capability without breaking the previous one.
- **Weight-injectable heuristics** — All evaluation weights injectable via `HeuristicWeights`.
- **Tactical before positional** — AI urgency hierarchy (close mill → block mill → disrupt structures → position) is a first-class design constraint.
- **Staged opening memory** — Starting play recognised in phases; move-sequence ancestry and searchable tags preserved.

---

## Thematic note — placement-phase root causes

The B-22 through B-37 cluster around three confirmed core weaknesses:

**Weakness 1 — Late placement overvalues speculative structure.**
Fixed via B-46/B-28: setup-building bonuses taper from 1.0× at placement 1 to 0.25× at placement 9.

**Weakness 2 — Opponent forcing potential is not mirrored.**
Fixed via B-37: `_placement_chain_scan` mirrored for the opponent.

**Weakness 3 — Tactical priority ladder exists in ordering but not in scoring.**
`_order_moves()` has a clean P0/P1/P2 hierarchy but `tactical_move_bonus()` is fully additive — speculative bonuses can still outscore emergency blocks. B-29 fixes the chain case; B-22 investigates the block case.
