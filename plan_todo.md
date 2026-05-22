# Nine Men's Morris — Active Backlog

_New items go here. When an item is completed, move it to `plan_done.md`._

---

## Deferred Ideas

### Opening variety — alternative approach (logged, not implemented)

The current solution forces the book move for the first 2 AI placements and uses temperature-weighted UCB sampling in `select_opening()`.

An alternative considered but not implemented: **force difficulty level 1–3 for the first 4–6 moves**, then restore the configured difficulty for the rest of the game. This would make the early-game search shallower so the opening-book bonus reliably dominates over positional heuristics without requiring any explicit forcing logic.

This was not implemented because:
1. The explicit `force_book_early` path is more surgical and doesn't affect the quality of tactical play on moves 3–9.
2. Reduced difficulty for the first 6 moves would also suppress mandatory-block detection (`_immediate_mill_threats`), potentially causing the AI to miss obvious defensive plays in the early placement phase.
3. The first-two-placement force already covers the observable symptom (d7 always first).

If the forcing approach proves insufficient after extended play, revisit this option as a fallback.

---

## Search & Evaluation Enhancements

### TIER 1 — Core Search Stack (implement together)

---

### SE-1 — Transposition Table + Zobrist Hashing ⬜ ★ Highest Impact

**Why:** The same board position is reached via many different move sequences (transpositions). Without a TT, `_negamax` re-evaluates every transposed position from scratch. A TT keyed by a Zobrist hash stores `(depth, score, flag, best_move)` per position, allowing the search to skip re-evaluation and use the stored best move for immediate ordering at that node. Expected gain: ~2× effective search depth in endgame; very large node savings throughout the move phase.

**NMM specifics:** Only 73 random 64-bit keys needed (24 squares × 3 states + 1 side-to-move bit). XOR-updated incrementally on each `apply_move`.

**Critical implementation note:** Use a fixed-size `list` (pre-allocated, indexed by `hash % TABLE_SIZE`) with depth-preferred replacement — **not** a Python `dict`. At high difficulty levels Python dict overhead would consume much of the gain.

**Deliverables:**
- `ai/transposition_table.py` — new `TranspositionTable` class; `hash_board()`, `lookup()`, `store()`
- `ai/game_ai.py` — probe TT at top of `_negamax`; store on exit; use hash-move as first candidate in ordering; reset between `choose_move` calls

---

### SE-2 — Killer Heuristic (2 killers per depth) ⬜ ★ High Impact

**Why:** A move that causes a beta cutoff at depth `d` in one branch is statistically likely to cause a cutoff in sibling branches at the same depth. Storing two such "killer" moves per depth and trying them before the unsorted remainder (but after captures/mill-closures) reduces node count by 20–30%. Zero change to evaluation quality; the implementation is ~15 lines.

Gains compound with SE-1: the TT provides a hash-move to try first at each node, killers then cover the next-most-likely cutoff movers.

**Deliverables:**
- `ai/game_ai.py` — `self._killers` list (2 per depth up to depth 32); `_store_killer()`; insert killer-match tier between priority-1 and priority-2 in `_order_moves`; reset killers at start of each `choose_move`

---

### SE-3 — History Heuristic ⬜ ★ High Impact

**Why:** Maintains a global `hist[(from_sq, to_sq)]` table incremented by `depth²` whenever a move causes a beta cutoff. Used as a sort key within the priority-2 bucket of `_order_moves`. Unlike killers (depth-local), history is global across all positions, making the two techniques complementary.

**Largest gain in fly phase** where the existing sort leaves ~50 of 54 moves unordered. Together SE-1 + SE-2 + SE-3 should lift effective depth by 1.5–2 ply within the same time budget.

**Deliverables:**
- `ai/game_ai.py` — `self._history` dict; increment on beta cutoff; use as tiebreaker in `_order_moves` priority-2 bucket; reset between `choose_move` calls (or age between iterations)

---

### TIER 2 — High Value, after Tier 1

---

### SE-4 — Endgame Tablebase Query Inside Search ⬜ ★ High Impact (underrated)

**Why:** Currently `EndgameDB` is consulted only at root level in `choose_move`. Querying it inside `_negamax` at every node where `total_pieces ≤ 8` returns `±INF` for known positions without any further search. This converts the lower search tree from estimated heuristic values to **exact outcomes** — a qualitative improvement, not just a speedup. The infrastructure already exists; this is approximately 10 lines of change.

**Deliverables:**
- `ai/game_ai.py` — add `EndgameDB` lookup at top of `_negamax` when `total_pieces <= 8`; return `outcome * (INF - depth)` so fastest wins are scored first

---

### SE-5 — Principal Variation Search (PVS / NegaScout) ⬜ ★ Medium–High Impact

**Why:** PVS assumes the first move explored is best (valid after good ordering from SE-1–3). All subsequent siblings are searched with a cheap zero-window `(alpha, alpha+1)` scout; only if the scout fails high is a full re-search triggered. With good ordering, the majority of siblings never need re-searching. ~10% additional node reduction on top of Tier-1 gains.

**Deliverables:**
- `ai/game_ai.py` — replace inner loop in `_negamax` with PVS scheme: first move at full window, siblings at zero-window with re-search on fail-high

---

### SE-6 — Late Move Reductions (LMR) ⬜ ★ Medium Impact

**Why:** Reduces search depth by 1 ply for moves sorted toward the end of the move list (assumed inferior after good ordering). **Largest proportional gain in fly phase** where branching factor reaches ~54 and the existing sort leaves most moves unordered.

**Guards (never reduce):**
- Mill-closing moves (priority-0)
- Opponent-mill-blocking moves (priority-1)
- Any move at depth < 3 or root level (`_score_all`)
- Moves during iterative deepening at depth ≤ 2

**Rule:** reduce last 60% of sorted moves by 1 ply at depth ≥ 4; re-search at full depth if reduced score exceeds alpha.

**Deliverables:**
- `ai/game_ai.py` — LMR applied after priority-0/1/killer ordering in `_negamax`; conditional re-search on fail-high

---

### SE-7 — Aspiration Windows in Iterative Deepening ⬜ ★ Medium Impact

**Why:** Currently each iterative-deepening iteration restarts with `alpha = −INF, beta = +INF`. Using `[prev_score − 175, prev_score + 175]` for depth `d+1` produces more early cutoffs since most moves are outside the window. Fail-high or fail-low triggers a re-search at full window — rare in the positionally stable mid-game common in NMM.

**Deliverables:**
- `ai/game_ai.py` — aspiration window around `prev_score` in `_iterative_deepen`; window margin ~175 score units; widen and re-search on fail

---

### TIER 3 — Solid, Secondary Priority

---

### SE-8 — Search Extensions for Critical Positions ⬜ ★ Medium Impact

**Why:** +1 depth at nodes containing: forced mill closure (own or opponent); opponent has 2+ immediate mill threats (fork); position is 4v4 fly-phase; EndgameDB confirms a critical pattern. Root-level depth bonuses already exist in `choose_move` — extend the same logic into internal `_negamax` nodes. Cap total extensions at `depth / 2` per line to prevent blowup.

**Deliverables:**
- `ai/game_ai.py` — extension check at top of `_negamax` using existing tactical detection helpers; max-extension cap per line

---

### SE-9 — Quiescence Search (Capture Extension at Depth 0) ⬜ ★ Medium Impact

**Why:** Eliminates the horizon effect in 4v4 endgame and fly-phase transitions. At `depth == 0`, if a mill closure (capture) is immediately available, extend 1–2 plies searching only capture sequences before returning static evaluation. Cap at 2–3 extra plies to avoid cycling in repetitive mill positions.

**Deliverables:**
- `ai/game_ai.py` — `_negamax_q()` quiescence search called at `depth == 0` when mill-closing moves exist; depth cap via `_qsearch_remaining` counter

---

### SE-10 — Proactive Fly-Fork Anticipation (Move Phase) ⬜ ★ Medium Impact

**Why:** The existing `fly_fork_bonus` fires reactively. The documented gap in `AI_INTERNALS.md` is that the AI does not pre-plan the sequence of moves that *creates* the fork. Extend `_fork_in_n(board, opp, n=2)` (already used in placement-phase, Enhancement B-4) to the move phase: scan forward up to 3 half-moves for forcing lines that result in 2+ simultaneous 2-configs.

**Deliverables:**
- `ai/heuristics.py` — `_move_phase_fork_anticipation(board, color, depth=3)`; bonus `fork_depth × 80` added to root move score

---

### SE-11 — Opponent Likelihood Weighting (Asymmetric Depth via TrajectoryDB) ⬜ ★ Medium Impact

**Why:** Standard alpha-beta allocates equal depth to all opponent responses regardless of how likely they are. Using the existing `TrajectoryDB`, empirical move frequency at the current game prefix can drive +1 extension for high-frequency opponent moves and −1 LMR for rare ones. Analogous to LMR but data-driven on actual opponent behaviour rather than sort position.

**Deliverables:**
- `ai/trajectory_db.py` — `query_move_frequency(prefix, notation)` method returning normalised frequency `[0.0, 1.0]`
- `ai/game_ai.py` — apply frequency-based depth delta at opponent nodes inside `_negamax`

---

### TIER 4 — Infrastructure / Long-Term

---

### SE-12 — Incremental Evaluation Cache (Zobrist-Keyed Sub-Functions) ⬜

**Why:** Heavy heuristic sub-calls (`_convergence_cluster_count`, `_mill_wrapping`, `_free_piece_assembly`, `_assembly_reach_count`) recompute from scratch every leaf call. With Zobrist hashing already in place (SE-1), a secondary cache keyed by board hash stores sub-function results and invalidates on state change. Requires SE-1.

**Deliverables:**
- `ai/heuristics.py` — result cache dict keyed by Zobrist hash for top-cost sub-functions; invalidate on apply_move

---

### SE-13 — N-Gram Opponent Move Predictor ⬜

**Why:** Complements TrajectoryDB (which tracks win/loss rates) with a pure move-frequency bigram/trigram model: given the last N moves, predict opponent's next move distribution. Feeds into SE-11 with richer per-sequence predictions. Lower priority since TrajectoryDB already covers this partially.

**Deliverables:**
- `ai/ngram_opponent_model.py` — new `NGramOpponentModel` class; `update()` called after each game; `predict()` returns probability dict; trained incrementally from `data/games/` JSONL records

---

## Architecture Principles

- **Immutable board state** — `BoardState.apply\\\\\\\_move()` always returns a new object. Enables safe undo, MCTS branching, and self-play without deep-copy overhead.

- **Coordinator owns the narrative** — All commentary and LLM calls flow through `Coordinator`. `GameAI` is pure search; `MillsLLM` is pure text generation. Neither knows about the other.

- **No cloud dependency** — All LLM inference runs locally via Ollama. No API keys, no cost after initial model pull.

- **Progressive enhancement** — Every stage adds capability without breaking the previous one. Fast mode (`--no-llm`, no opening book) always works as a fallback.

- **Weight-injectable heuristics** — All evaluation weights are injectable via `HeuristicWeights`. The Settings page, evolution driver, and self-play all use the same injection point.

- **Tactical before positional** — The AI urgency hierarchy (close mill → block mill → disrupt structures → position) is a first-class design constraint, not an afterthought.

- **Staged opening memory** — Starting play is recognised in phases (early, 12-piece mid-placement, final placement), with move-sequence ancestry and searchable tags preserved so both the engine and the study tools can reason over opening families rather than only isolated final lines.

