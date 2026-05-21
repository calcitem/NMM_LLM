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

## Open Bugs & Enhancements

### Stage 5.20 — Position Strength Late-Game Fix ⬜ - human note, prioritise this

**Goal:** When a player is down to 3–4 pieces and the opponent has 6–7 pieces with 3 open mills or a double-parallel mill, the position-strength eval should reflect the losing side's danger (not give false hope from mobility).

**Root cause:** The tanh normalisation uses a flat scale per phase; a 3-piece player who can fly anywhere scores high mobility, which inflates their eval beyond what the real material+threat situation warrants.

**Fix:**

- `ai/heuristics.py` — Add a late-game danger penalty: when one side has ≤4 pieces and the opponent has ≥6 pieces with ≥2 open mills, apply a large negative adjustment (e.g. `−800`) to the weaker side's score before tanh normalisation.

- `ai/heuristics.py` — Reduce `TANH\_SCALE` for the fly phase from 280 to ~180 so extreme positions are less compressed near ±1.

### Tactic 5.12-A — 6v4 Piece Sacrifice to Reach Winning Endgame ⬜ - human note - prioritise

**Goal:** When the AI has 6 pieces and the opponent has 4, and the standard 6v3 zugzwang pattern is achievable, the AI should deliberately sacrifice 3 of its own pieces to reach 3v3 — unless the 6v3 domination pattern is reachable first.

**Rationale:** A 6v4 position is roughly level (neither side dominates). The well-known 3v3 flying endgame, however, is a forced win for the side with superior mill structure. Reaching 3v3 by voluntary piece reduction (trading pieces rather than protecting them) puts the AI into a position it can play from a known winning blueprint.

**Conditions to trigger:**

1. AI has exactly 6 pieces, opponent has exactly 4 pieces (6v4).

2. 6v3 domination (three AI mills, opponent cannot cover all) is NOT immediately available.

3. The AI can engineer a sequence of piece trades (deliberately moving into captures) to reach 3v3 within 3–4 moves.

4. After reduction, the AI's resulting 3-piece position forms or imminently forms a mill, giving it the better 3v3 structure.

**Check endgame patterns are stored:**

- Verify `data/endgame/` (or the EndgameDB) contains 3v3 flying winning patterns. If not, supply them or run `tools/endgame\_play.py` seeded from 3v3 positions.

**Implementation:**

- `ai/heuristics.py` — Add `\_is\_6v4\_sacrifice\_position(board, color)`: returns True when the conditions above hold.

- `ai/heuristics.py` `evaluate()` — When in 6v4 and sacrifice is viable: add a bonus (~250) to moves that reduce own piece count toward 3v3 *if* the resulting 3-piece arrangement contains at least one closed mill or 2-config.

- `ai/coordinator.py` — In `\_tactical\_situation()`: flag `"6v4\_sacrifice\_viable": True` and emit a commentary hint when triggered.

- `tests/test\_tactics.py` — Add tests: 6v4 sacrifice bonus fires when structure is good; does not fire when 6v3 is available; does not fire when the post-trade 3-piece arrangement is weak.

### Bug UI-C — AI Double-Mill Prevention Weakness ⬜ - human note – prioritise this

**Symptom:** The AI fails to prevent an opponent from setting up a double mill (two open mills that share a pivot piece). Once the opponent has two 2-configs whose closing squares share a common piece, the AI cannot block both in one move. Example: White moves between d6 and f6 creating lines 6 and f simultaneously — no single black move blocks both.

**Root cause:** The existing `stop\_opponent\_mills` bonus (weight 450) dismantles individual 2-configs but does not specifically detect or penalise the opponent's *double-mill convergence* — the moment two opponent 2-configs share a closing square or pivot. The AI waits until the mills are closeable before treating them as urgent, by which point the fork is already established.

**Fix:**

- `ai/heuristics.py` — Add `\_double\_mill\_convergence(board, opp)`: count positions where the opponent has two or more 2-configs that share a common empty closing square or a common own-piece pivot. This is the precursor to a fork that cannot be blocked.

- `ai/heuristics.py` `evaluate()` — Add term: penalise `\_double\_mill\_convergence(board, opp)` with weight ~180 in move phase (slightly above `stop\_opponent\_mills` to make disrupting convergence the priority over dismantling isolated 2-configs).

- `ai/heuristics.py` `tactical\_move\_bonus()` — Add a bonus for moves that *reduce* the opponent's double-mill convergence count (break one of the two 2-configs that share a square/pivot). Weight ~220, applied on top of `stop\_opponent\_mills`.

### Bug UI-D — Tournament Mode: Show AI Personality Per Game ⬜

**Symptom:** In tournament mode the round results only show "White wins" / "Black wins" without indicating which AI personality was playing each side, making it hard to compare personality performance.

**Fix:**

- `web/app.py` — Include `white\_personality` and `black\_personality` in the `tournament\_game\_result` WebSocket message payload.

- `web/static/game.js` — Display personality names in the tournament results table alongside the win/loss result for each game.

### Bug UI-E — User Guide: Missing Sections ⬜

**Goal:** Stage 5.15 (User Guide) is in progress. The following sections are missing and must be added:

1. **Named Openings** — how the opening book works, how openings get named, how to browse them.

2. **Game Setup / Position Editor** — how to use the Setup Position button to create custom starting positions.

3. **AI Slider Weights** — what each weight slider in the AI Tuning panel controls and how they interact.

4. **Personality Profiles** — description of each built-in personality and what makes them distinct.

5. **Training Tools** — brief guide to `self\_play.py`, `evolve\_weights.py`, `train\_value\_net.py`, and `import\_book\_games.py`.

**Deliverable:** README.md (or separate GUIDE.md) updated with these five sections.


---

## Enhancement Backlog

### Bug B-9 — Mills LLM Commentary: Wrong Line Names and Factual Errors ⬜

**Symptom:** The LLM makes factually wrong commentary because it cannot read the board correctly. Examples observed:

- *"Are you planning to challenge White's central dominance with a mill on the c-line?"* — when the c-line (c3-c4-c5) is packed with immobile pieces and no mill there is possible.

- *"Black forms a mill on the e-line and captures White's piece, gaining significant advantage."* — when the mill was actually formed on line 1 (g1-d1-a1), not the e-line.

**Root cause:** `MillsLLM` prompts pass only `board.to\_display\_grid()` (the raw ASCII board) plus the move-notation history. The LLM receives no structured summary of: which mills are currently closed, which 2-configs (one-away threats) exist, piece counts, current phase, or canonical mill line names. Without this, the LLM must infer all tactical state from the ASCII grid — and it guesses wrong.

**The 16 legal mills in NMM notation:**

| Name | Squares |
| - | - |
| Outer top | a7-d7-g7 |
| Outer right | g7-g4-g1 |
| Outer bottom | g1-d1-a1 |
| Outer left | a1-a4-a7 |
| Middle top | b6-d6-f6 |
| Middle right | f6-f4-f2 |
| Middle bottom | f2-d2-b2 |
| Middle left | b2-b4-b6 |
| Inner top | c5-d5-e5 |
| Inner right | e5-e4-e3 |
| Inner bottom | e3-d3-c3 |
| Inner left | c3-c4-c5 |
| d-column top | d7-d6-d5 |
| g-row | g4-f4-e4 |
| d-column bottom | d1-d2-d3 |
| a-row | a4-b4-c4 |


**Fix:** Add a `\_board\_summary(board)` helper (in `ai/mills\_llm.py` or a shared utility) that computes and formats:

1. **Phase** — `placement / move / fly`

2. **Piece counts** — `White: N on board (M in hand)`, `Black: N on board (M in hand)`

3. **Closed mills** — list each closed mill by name and squares, e.g. `White: Outer bottom (g1-d1-a1)`

4. **Two-piece threats (2-configs)** — list each by name and which square closes it, e.g. `White: d-column bottom (d1-d2 — closes at d3)`

5. **Mobility** — count of legal moves per side (optional; gives the LLM a quick dominance signal)

Inject this summary into every `MillsLLM` prompt that calls `board.to\_display\_grid()`, replacing or augmenting it with a `POSITION SUMMARY:` block.

**Affected methods in `ai/mills\_llm.py`:**

- `ask\_for\_move\_opinion()` — primary deliberation prompt; most important

- `evaluate\_human\_move()` — poor-move commentary

- `comment\_on\_mill()` — mill formation commentary

- `ask\_strategic\_question()` — strategic position question

- `comment\_on\_good\_move()` — positive commentary

- `generate\_question\_for\_human()` — question generation

- `player\_chat()` — in-game chat

**Also check `deliberate()` in `ai/coordinator.py`:** The `react\_to\_human\_move()` call chain is where mill commentary fires — confirm it also passes the enriched board summary.

**Deliverables:**

- `ai/mills\_llm.py` — `\_board\_summary(board)` helper; inject into all prompt-building sites

- (optional) `game/board.py` or `ai/mills\_llm.py` — `MILL\_NAMES` dict mapping each mill tuple to a human-readable name


### Bug B-10 — AI Allows Opponent to Consolidate Three Scattered Pieces into a Line ⬜

**Symptom:** The AI makes a move that frees a square the opponent needs to complete a line of three pieces that were previously unconnected. Example from a recorded game:

```
18.a4-a7   b4-a4
```

White moves `a4-a7`, vacating `a4`. Black immediately plays `b4-a4`, consolidating three pieces on line 1 (`a1-a4-a7`... eventually) while White cannot contest `g4`. The AI should have foreseen that vacating `a4` gifted Black exactly the landing square needed to form a three-piece alignment.

**Root cause:** The heuristic `evaluate()` does not check, for each candidate move, whether the resulting empty square(s) allow the opponent to unite two previously disconnected groups into a contiguous line of three. The danger is invisible to the search unless look-ahead depth happens to reach the opponent's consolidating reply.

**Fix:**

1. Add a helper `\_opponent\_line\_consolidation\_threat(board, move, side)` in `ai/heuristics.py`:

   - After applying `move`, enumerate all groups of opponent pieces that share a mill line with an empty square adjacent to both groups.

   - If any such empty square is the one just vacated (or any other square newly freed by the move), return a penalty proportional to how many mill lines that consolidation would enable.

2. Subtract the penalty from `evaluate()` for the moving side, making the AI less likely to expose consolidation squares when the opponent has pieces waiting on both sides of that square.

3. This check is distinct from the existing 2-config threat detection: it fires when the opponent has *two separate pieces on a line with one empty square between/at the end*, not when they already have two in a mill triplet.

**Affected files:**

- `ai/heuristics.py` — new helper + penalty injected into `evaluate()`

- `ai/game\_ai.py` — no changes needed (penalty surfaces naturally through `evaluate()`)


### Bug B-11 — AI Session Summary Contains Fabricated Move Numbers and Piece Counts ⬜

**Symptom:** The LLM session summary invents events that did not occur. Observed examples from a recorded game:

- References to "move 23" when the game ended at move 20.

- Claims "the AI had 2 pieces left" when it resigned with many pieces on the board.

- The summary gives no indication of *why* the GameAI chose its moves (the LLM does not know what the classical engine was calculating).

**Root cause (two separate issues):**

1. **Fabricated summary facts** — The LLM receives only the move-notation list and the ASCII board; it has no reliable anchor for the game length, resignation trigger, or final piece counts, so it hallucinates plausible-sounding endings.

2. **GameAI "thinking" is invisible to the LLM** — The classical engine's best-score line, the dominant heuristic term that drove its decision, and its tactical intent are never reported, so the LLM cannot accurately narrate *why* a move was made.

**Fix — Part A: Ground the session summary**

In `ai/coordinator.py` `generate\_session\_summary()`, inject a structured `GAME FACTS` block into the prompt:

```
GAME FACTS (authoritative — do not contradict):  
  Total half-moves: \{N\}  
  Termination: \{resignation | piece-loss | no-legal-moves\}  
  Final piece counts: White \{W\}, Black \{B\}  
  Winner: \{White | Black | Draw\}
```

This block must be derived from `game\_record`, not inferred from the board ASCII.

**Fix — Part B: AI "thinking" trace**

Add a `thinking` field to the dict returned by `GameAI.choose\_move()`:

```
\{  
  "move": \<move\>,  
  "thinking": "Blocked opponent mill threat on outer-bottom; mobility +2"  
\}
```

`thinking` is a one- or two-phrase plain-English label identifying the 1–2 dominant heuristic contributions to the chosen move's score (e.g. *"closed mill on d-column bottom"*, *"blocked 2-config on inner-left"*, *"improved mobility +3"*).

Surface `thinking` in the UI:

- In the **AI Discussion** panel, add a **"Show AI reasoning"** toggle (checkbox, off by default).

- When toggled on, each AI move in the discussion feed is annotated with the thinking string below the move notation line.

**Affected files:**

- `ai/game\_ai.py` — populate `thinking` in `choose\_move()` return value; identify top-1 or top-2 heuristic terms from the score breakdown

- `ai/heuristics.py` — `evaluate()` should optionally return a score-breakdown dict so `game\_ai.py` can label the dominant term

- `ai/coordinator.py` — inject GAME FACTS block into session summary prompt

- `web/app.py` — pass `thinking` string in the WS `ai\_move` message

- `web/static/game.js` — "Show AI reasoning" toggle; render thinking string in AI Discussion feed

- `web/templates/index.html` — toggle checkbox in AI Discussion panel


### Bug B-12 — Opening Replay Fails with Illegal Move Error; Bad Moves in Opening DB ⬜

**Symptom (four related issues):**

1. Both "Novel — d7-d6-g4 (18 moves)" openings fail to replay, printing:

   - `Opening replay stopped: move d3 is not legal at this point.`

   - `Opening replay stopped: move d6 is not legal at this point.`

2. After the error, play continues and the message `Game: Opening complete — now playing AI vs AI.` appears, followed by `Error: 'NoneType' object has no attribute 'choose\_move'`.

3. The **Bad Move** button was pressed during a game that then continued, inserting illegal moves into the recorded opening sequence and corrupting the DB entry.

4. The **Watch — AI continues from opening** option on the openings panel does not result in an AI vs AI game being played.

**Root cause:**

- Bad-move records in `learned\_openings.json` contain moves that were flagged during a game but the game continued anyway, so the flagged move appears in the sequence even though it may not be legal for any board state reached from the standard opening position.

- The AI-vs-AI continuation path after an opening replay does not wire up both `Coordinator` instances correctly — one is `None`.

- The "Watch" button handler may not trigger the `start\_ai\_vs\_ai` flow at all.

**Fix:**

**A — Purge corrupt opening entries:**

1. Write a one-off script `tools/fix\_openings.py` that replays each opening from the standard start position move-by-move and removes any entry that produces an illegal-move error.

2. For entries that partially replay, truncate the move list to the last legal move (preserving as much opening data as possible).

3. Run the script before committing and include the cleaned `learned\_openings.json` in the fix commit.

**B — Disable the Bad Move button during opening replay (and remove its effect on DB openings):**

1. In `web/static/game.js`, hide or disable the Bad Move button whenever `state.opening\_active === true`.

2. In `web/app.py`, when a bad-move report is filed, do *not* record it against any move that is part of an active opening replay sequence.

**C — Fix AI-vs-AI NoneType error after opening replay:**

1. In `web/app.py`, after `opening\_replay\_complete` fires, ensure both sides have a fully initialised `Coordinator` (or `GameAI` if LLM is off) before handing off to `ai\_vs\_ai\_loop`.

2. Add a null check and a descriptive error log if either coordinator is still `None`.

**D — Fix "Watch" button:**

1. Trace the WS message path from the Watch button click → server handler; verify it calls `start\_ai\_vs\_ai` with the correct `session\_id`.

2. Add an option to start AI-vs-AI from the end of the **placement phase** (not just from the end of an opening).

**E — Opening rename/delete from GUI:**

1. Add a kebab menu (⋮) or inline icon buttons on each opening row in the Openings panel: **Rename** and **Delete**.

2. Rename: opens an inline text field pre-filled with the current name; saves on Enter/blur.

3. Delete: shows a one-click confirmation ("Delete this opening?") then removes the entry from `learned\_openings.json` and refreshes the list.

**F — LLM opening name is a suggestion, not final:**

1. When a novel opening is saved and the LLM proposes a name, display a modal with the LLM suggestion pre-filled in a text input.

2. The player edits or accepts the name and clicks **Save**; the player's version is written to `learned\_openings.json`, not the raw LLM output.

**Affected files:**

- `tools/fix\_openings.py` — new validation/repair script

- `data/openings/learned\_openings.json` — cleaned by the script

- `web/app.py` — null-check coordinator before AI-vs-AI handoff; Watch button handler; bad-move guard during opening replay

- `web/static/game.js` — disable Bad Move during opening replay; Watch button fix; rename/delete handlers; LLM name modal

- `web/templates/index.html` — rename/delete controls in Openings panel; LLM name suggestion modal


### Bug B-13 — AI-vs-AI Game Not Available on GUI; No Option to Save ⬜

**Symptom:** There is no way to watch two AI personalities play a full game against each other in the browser GUI. The "Watch" button after an opening either does nothing or errors (see B-12). Even if it worked, there is no way to opt-in to saving the AI-vs-AI game to the database.

**Goal:**

1. Add a persistent **"AI vs AI"** button (in the header or Settings panel) that starts a fresh AI-vs-AI game from the initial position, with selectable personalities for each side.

2. The game plays out automatically in the browser, with the board updating move-by-move and the AI Discussion panel showing commentary.

3. **By default, AI-vs-AI games do NOT contribute to the trajectory/endgame DB and are NOT saved to `data/games/`.** A prominent checkbox labelled "Save this game to library" (off by default) can be ticked at any point; if ticked before the game ends, the game record is saved on completion.

4. At game end, display an end-of-game modal with the result and the Save toggle (if not already saved).

**Affected files:**

- `web/app.py` — `start\_ai\_vs\_ai` endpoint / WS handler; `is\_training\_game` flag on `GameSession`; conditional save logic

- `web/static/game.js` — AI-vs-AI start flow; save checkbox; end-of-game modal

- `web/templates/index.html` — AI vs AI button; personality selectors for each side; Save checkbox


### Bug B-14 — AI Herding: Moving Two Groups Apart and Restricting Opponent Mobility ⬜

**Goal:** Improve the AI's ability to herd the opponent's pieces — specifically to recognise when moving two of its own groups in opposite directions forces the opponent into a smaller effective board area, reducing their mobility.

**Observed weakness:** The AI sometimes moves two clusters further apart from each other without any tactical justification, effectively splitting its own force. Conversely, it fails to recognise positions where moving a piece outward from one cluster "pins" an opponent piece by reducing the escape squares adjacent to an opponent's cluster.

**Proposed heuristic enhancement:**

1. Add a `\_herding\_score(board, side)` term in `ai/heuristics.py`:

   - For each pair of opponent pieces that share a mill line with only one or two empty squares between them, count how many of those empty squares are directly adjacent to one of our own pieces (i.e. we "cover" the escape square).

   - A higher coverage count → higher herding score (we restrict their mobility).

2. Weight the herding term against the existing mobility delta so the AI prefers moves that reduce opponent escape squares, especially in the movement and fly phases.

3. Add a `herding` weight to `HeuristicWeights` so it can be tuned via the Settings UI and the evolution driver.

**Affected files:**

- `ai/heuristics.py` — `\_herding\_score()` helper; inject into `evaluate()`

- `ai/heuristics.py` — `HeuristicWeights` — new `herding` field

- `web/app.py` / Settings panel — expose `herding` weight in the tuning UI


### Bug B-15 — AI Does Not Anticipate Opponent Moves That Will Trap a Mill ⬜

**Symptom:** In the placement phase the AI places a piece to form or extend a mill, but does not look ahead to an opponent move that will block the only exit square of that mill, leaving the mill permanently trapped. Observed example:

```
8.a1   c4        ← Black places c4 (possibly chasing diamond reward or impulse score)  
9.g1×c4  c4  
10.a1-a4  d3-e3  ← White immediately plays a1-a4, trapping the Black mill
```

Had Black placed at `e4` instead of `c4`, it would have threatened a mill on the e-line (`e3-e4-e5`), forcing White to close their own mill defensively, and Black could then open the `b`-line mill and continue cycling.

**Root cause:** The heuristic scores the immediate value of the placement without scanning whether the opponent's *best reply* will seal the only exit of the resulting mill. A trapped mill is often worse than no mill at all.

**Fix:**

1. Add a `\_trapped\_mill\_penalty(board, move, side)` helper in `ai/heuristics.py`:

   - After applying `move`, enumerate all closed mills for `side`.

   - For each closed mill, find its "exit squares" — the squares adjacent to a mill piece that are *not* part of the mill and that the mill piece could slide to on a future turn.

   - If the opponent has a piece that is one move away from occupying every exit square of a mill, apply a penalty proportional to the number of mills that would become trapped.

2. Also add a `\_potential\_mill\_threat(board, side)` reward term:

   - Award a smaller bonus for any configuration where `side` has two pieces on a mill line with one empty square, *and* at least one exit square of that future mill is not currently covered by an opponent piece.

   - This nudges the AI toward forming escapable mill threats rather than static mills.

3. Both terms interact with the existing 2-config and mobility scores; tune weights to keep tactical priorities correct.

**Affected files:**

- `ai/heuristics.py` — `\_trapped\_mill\_penalty()`, `\_potential\_mill\_threat()`, injected into `evaluate()`

- `ai/heuristics.py` — `HeuristicWeights` — new `trapped\_mill` and `potential\_mill` fields


---

### Bug B-16 — AI Does Not Recognise Cross-Feeding Dual-Mill Setup (Feeder Mill) ⬜

**Symptom:** The AI passes up moves that create a "feeder mill" — a position with two separate 2-piece groups that mutually sustain each other's mill threats. If the opponent captures from either group, the surviving single piece has the mobility to migrate and complete the other group's mill. The AI cannot see this resilience because it evaluates each 2-config in isolation.

**Example from a recorded game (move 19, Black to move):**

```
19. c4-c5×d3   d6-f6   ← AI played (poor)
    better:    d7-a7   ← AI missed
```

After `d7→a7`, Black has:
- **Group A:** `a7` + `d7`... wait, Black moved *from* `d7`, so `d7` is now empty. The outer-top line (`a7-d7-g7`) has Black at `a7` and `g7` with `d7` empty → 2-config.
- **Group B:** `g1` + `g4` on the outer-right line (`g7-g4-g1`), missing `g7`.

**The cross-feed:** Group A and Group B do *not* share a piece, but:
- If White captures `a7` (Group A loses a piece), Black's remaining `g7` can slide to `g7→...` — or more concretely: `g7` is adjacent to `g4`, and the outer-right mill `g7-g4-g1` already has `g4` and `g1` (Group B). So `g7` moving nowhere is fine — it closes Group B immediately.
- If White captures from Group B (e.g. takes `g4`), Black's `g1` remains. Black plays `g7→d7` closing the outer-top mill (`a7-d7-g7`).

In both cases: **whichever group White attacks, the survivor from that group migrates (or the survivor of the other group completes the undefended mill) within 1–2 moves.** White cannot simultaneously block both lines.

**The general pattern — Cross-Feeding 2-Config Pair:**

Two 2-piece groups G1 and G2 form a cross-feeding pair when:
1. G1 = {P1, P2} are both on a single mill line L1, one square short of completion (closing square X).
2. G2 = {P3, P4} are both on a different mill line L2, one square short of completion (closing square Y).
3. **Cross-feed condition:** at least one piece in G1 is adjacent to Y (and could slide there), OR at least one piece in G2 is adjacent to X. This means if the opponent captures one piece from either group, the surviving piece has the mobility to close the *other* group's mill.

The two groups need **not** share any piece. The threat resilience comes entirely from the cross-mobility between the two closing squares.

**Root cause:** The heuristic scores each 2-config independently and sums them. It does not evaluate the *combined* resilience of pairs of 2-configs where a capture in one enables completion of the other. The AI therefore undervalues positions that set up this cross-feeding structure, choosing tactically inert moves instead.

**Distinction from existing items:**
- **Bug UI-C** is *defensive* — disrupting the opponent's converging mills.
- **SE-10** is about proactive fork planning in fly phase.
- **B-16** is *offensive*, fires in move phase, and specifically captures the cross-mobility property between two otherwise independent 2-config groups.

**Fix:**

1. Add `_cross_feed_bonus(board, side)` in `ai/heuristics.py`:
   - Find all pairs of 2-configs `(G1, G2)` for `side` where G1 and G2 are on different mill lines.
   - For each pair, check the cross-feed condition: can any piece of G1 reach G2's closing square in one move (adjacency in move phase, any square in fly phase)? Can any piece of G2 reach G1's closing square?
   - If the cross-feed condition holds, score the pair with a bonus: `cross_feed_weight × (1 + shared_closing_square_bonus)`, where `shared_closing_square_bonus` fires if G1 and G2 happen to share the same closing square (an even stronger threat — one move closes both mills simultaneously).
   - Sum bonuses across all qualifying pairs.

2. Add `cross_feed_mill` weight to `HeuristicWeights` (suggested initial value ~280 — above a lone 2-config, below an immediate mill closure).

3. Evaluate at every search depth, not only at root, so the gradient pulls the AI toward cross-feeding configurations several moves before they become decisive.

**Affected files:**
- `ai/heuristics.py` — `_cross_feed_bonus()` helper; inject into `evaluate()`
- `ai/heuristics.py` — `HeuristicWeights` — new `cross_feed_mill` field

---

## Search & Evaluation Enhancements

_Source: "Heuristic Search Improvements - Opponent-Anticipation Techniques.md" (2026-05-21). Items ranked by combined win-rate uplift + node-count reduction. Items 1–3 compound each other and should be implemented together as a single search-stack upgrade before benchmarking._

---

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

