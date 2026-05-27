# Nine Men's Morris — Active Backlog

*New items go here. When an item is completed, move it to `plan_done.md`.*

---

## Implementation Roadmap

Track 1 (heuristic/phase-control) and SE-1 through SE-9 complete. Active priorities:

| Priority | Item | Key outcome |
|----------|------|-------------|
| ★★ | **B-55** | Block opponent dual cardinal mill (placement phase) |
| ★★ | **SE-10** | Proactive fly-fork anticipation (move phase) |
| ★ | **B-51** | Expand retrograde solver beyond 3v3 |
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

### Bug B-55 — AI allows opponent to build two interconnected cardinal ring mills ⬜ ★ High Priority

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
