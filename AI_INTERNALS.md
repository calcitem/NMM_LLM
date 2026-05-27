# AI Internals — Move Selection and Position Evaluation

## 1. How the AI Decides Where to Play

### Search algorithm

The AI uses **negamax with alpha-beta pruning** (`ai/game_ai.py`). Negamax is a simplification of minimax that works by negating the score at each level, so it always maximises from the current player's perspective without needing separate min and max cases.

Alpha-beta pruning discards branches that cannot affect the final result. When the algorithm finds a move that is already worse than something the opponent could have steered toward, it stops searching that branch. In practice this roughly halves the effective search depth compared to plain minimax.

### Depth and time budget

Difficulty 1–4 map to a fixed search depth (2–5 plies). Difficulties 5–10 use **iterative deepening**: the search starts at depth 2, completes, then starts again at depth 3, and so on until the time budget is exhausted (15 s at difficulty 5, up to 90 s at difficulty 10). The best move found at the last fully completed depth is returned.

A special early-game fast path applies while fewer than 10 pieces are on the board: regardless of difficulty, the search uses a 4-second iterative-deepening budget. The position tree is tiny at that point, so long searches are wasteful.

### Move selection

`choose_move()` calls `_score_all()`, which runs a full negamax search from every legal root move and returns a scored list. The move with the highest score wins. If `blunder_probability > 0`, there is a chance the AI instead picks from the **bottom quartile** of scored moves — a deliberate mistake for teaching purposes.

#### Bad-move bans and LLM override safety

The player can flag any AI move as bad (via the "Bad Move" button in the UI). This bans the move's notation for the exact board FEN it was played from.

Two layers enforce the ban:

1. **In-game positional ban** (`GameAI._pos_bans`): filtered at the top of `choose_move()` before the search runs. The ban is keyed on the exact board FEN so it applies only to the specific position, not to the same move in a different position. Bans are lost when a new `GameAI` instance is created for the next game.

2. **Persistent trajectory ban** (`TrajectoryDB.mark_bad_move()`): called alongside the positional ban in `app.py`. This writes a `−1.0` hard-ban sentinel to the trajectory database and to `data/bad_moves.json`. On subsequent games, when `TrajectoryDB.query()` returns hints, any banned notation receives `−INF+1` in `_apply_trajectory_hints()` so it is always placed last in the scored list — effectively barred while remaining technically legal (safety guard). The persistent ban loads automatically from `data/bad_moves.json` on each server start.

When the `Coordinator` (LLM mode) is active, after `choose_move()` returns its (already filtered) best move, the Coordinator may override it with the LLM's recommendation if the LLM's score exceeds the engine's score plus `LLM_BONUS`. To prevent a banned move from re-entering through this path, `deliberate()` checks the ban set against the LLM's suggestion before adopting it — if the LLM recommends the banned move, the suggestion is discarded and the engine's choice is kept.

### MCTS mode (Stage 12)

`GameAI` optionally delegates move selection to **Monte Carlo Tree Search** (`ai/mcts.py`) when constructed with `use_mcts=True`. MCTS runs within the same time budget as negamax for the chosen difficulty.

The MCTS implementation uses **UCT** (Upper Confidence Trees):

```
UCB(child) = Q(child) ± C × √(ln N(parent) / N(child))
```

where `Q` is the cumulative value divided by visits, `C = √2` is the exploration constant, and the sign flips between `+` (current player maximises) and `−` (opponent minimises) depending on whose turn it is at the parent node. Values are stored from `self.color`'s fixed perspective throughout the tree — no sign-flipping during backpropagation.

Leaf evaluation uses `heuristics.evaluate()` mapped through `tanh` to `[−1, 1]`. If a trained `ValueNet` is loaded (`data/value_net.npz`), it replaces the heuristic at leaves for faster and stronger evaluation. The most-visited child (rather than highest-Q child) is returned as the final move choice, which is more robust under noisy rollouts.

### Value network (Stage 12)

`ai/value_net.py` provides a small MLP (79 → 128 → 64 → 1) trained from self-play game records:

- **Input**: 24 positions × 3 channels (own/opponent/empty) + 7 scalar metadata = 79 features, encoded from the current player's perspective so the same weights handle both colours.
- **Output**: `tanh` scalar in `(−1, 1)` — positive means the current player is likely to win.
- **Training**: `tools/train_value_net.py` reads all `data/games/*.jsonl` files, assigns final-outcome labels to every board position in each game, and trains with mini-batch SGD (MSE loss). Saves to `data/value_net.npz`.
- **Inference**: pure numpy, no deep-learning framework required; predicts in ~0.1 ms per position.

### Opening book and trajectory adjustments

When an opening has been recognised (or synthesised from the `_target_opening`), the scored move list is adjusted before final selection. The adjustment size scales with the **Opening Adherence** slider (0–100 %):

- The book's recommended next move receives an absolute bonus of up to `3000` internal score units at 100 % adherence, scaling linearly down to zero at 0 %.
- Moves listed as common blunders for the current opening receive a penalty of up to `1500` units.

**100% adherence forcing**: when the slider is at exactly 100 and a book move is available and legal (and not banned), `choose_move()` returns it immediately without running any search. This guarantees the AI follows the book exactly for as long as the game remains on-book.

**First-two-placement forcing**: for the first 2 AI placements in a game (regardless of the adherence slider), the Coordinator passes `force_book_early=True` to `choose_move()`. This ensures different games begin with the opening's first move rather than the negamax-preferred cross-node (`d7`) which scores highest unconditionally on an empty board. Together with temperature-based opening selection (see below), this produces visible opening variety across games.

### Opening selection variety (temperature sampling)

`OpeningBook.select_opening()` uses **UCB1 with temperature-weighted random sampling** instead of deterministic `max()`:

```
weight_i = exp((UCB_i − max_UCB) / temperature)
```

with `temperature = 0.18`. A random opening is drawn proportional to these weights. The best-scoring opening is still most likely to be chosen, but under-explored openings with competitive UCB values get genuine play time. This directly fixes the problem where the AI always targeted the same `d6`-family opening on every game.

The **TrajectoryDB** (`ai/trajectory_db.py`) indexes every completed game by move-notation prefix. After the opening phase, winner moves receive positive score deltas and loser moves receive negative ones. Deltas in `[−0.5, +0.5]` are statistical hints; a delta of exactly `−1.0` is a hard ban (set by the Bad Move button) and causes the move to receive `−INF+1` regardless of adherence — it is never chosen.

Bad-move bans are also enforced directly inside `choose_move()` via per-FEN position bans (`_pos_bans`), so a banned move cannot be re-played even if the trajectory hint is somehow bypassed.

### D4 board symmetry in learning databases

Both `TrajectoryDB` and `EndgameDB` use the **D4 dihedral group** (4 rotations + 4 reflections) to pool symmetric game positions. Every prefix or board state is stored in its **canonical (lex-min) form** across all 8 D4 transforms; queries search all 8 equivalents and merge statistics, then inverse-transform move notations back to the actual board orientation.

This multiplies effective sample size by up to 8× with no additional games needed. The helpers are in `ai/board_symmetry.py`:

- `canonical_sequence(notations)` → `(canonical_list, sym_idx)` for trajectory prefixes.
- `canonical_board_str(board_24)` → `(canonical_str, sym_idx)` for endgame positions.
- `prefix_query_canonicals(notations, depth)` → all unique canonical equivalents for a query prefix.
- `board_query_canonicals(board_24)` → all unique canonical equivalents for a query position.
- `SYM_INVERSE[sym_idx]` → the inverse symmetry index for back-transforming move notations.

### Per-game D4 symmetry for White AI opening variety

On top of the learning-database D4 pooling, the **Coordinator** randomises a per-game symmetry index for the White AI player at the start of every game:

```python
# ai/coordinator.py — on_game_start()
if self.game_ai.color == "W":
    self._game_sym_idx = random.randint(0, 7)   # 0 = identity, 1–7 = one of the 7 non-trivial D4 transforms
```

During the placement phase, every book move retrieved from the `OpeningRecognizer` is inverse-transformed through `_game_sym_idx` before being returned to `choose_move()`. The effect is that a `d6` book opening can play as `d6` (sym 0), `b4` (sym 1), `d2` (sym 2), `f4` (sym 3), `a7` (sym 4), etc., depending on the game. A human opponent cannot learn to anticipate a fixed first move. This index is only set when the AI is White; Black's variety comes from the recognizer's own D4 scan (see below).

### Black opening variety via recognizer D4 scan

The `OpeningRecognizer` runs a **D4 symmetry scan** at Step 3 of its pipeline whenever no direct match is found and no symmetry has yet been established (`_active_symmetry == 0`). When the human (White) plays a first move that is a D4 variant of a known opening's first move — for example `f4` instead of `d6` — the scan detects that symmetry index 3 maps `f4` → `d6`, sets `_active_symmetry = 3`, and subsequently inverse-transforms book moves for Black's responses through `SYM_INVERSE[3]`. The Coordinator's `force_book_early=True` flag then forces those transformed book moves for Black's first two placements. The net result is that Black plays a contextually correct reply to whatever rotated opening the human has begun, with no extra code needed in the Coordinator.

### Novel opening storage for "inactive" games

When a game ends with no opening ever matched (the recognizer status stays `"inactive"` throughout — no move in the book matched, including via D4 scan), the game was previously silently dropped and never learned from. `on_game_end()` in `ai/coordinator.py` now treats `"inactive"` the same as `"novel"`:

```python
# Before:  if final.status == "novel":
# After:
if final.status in ("novel", "inactive"):
    self._save_novel_opening(...)
```

The existing guard in `_save_novel_opening()` — `if len(placement_moves) < 6: return` — prevents trivially short or incomplete games from being stored, so this change is safe.

---

## 2. How the Position Strength Meter Works

### Raw evaluation

The static evaluator (`ai/heuristics.py`, `evaluate()`) scores a position as an integer from the perspective of one colour. Higher is better for that colour. The formula is a weighted sum of several features:

```
score = Σ weights × features + mobility_term + threat_term + positional_term + endgame_supplement
```

### Features and weights

The weights change by game phase ("place", "move", "fly"):

| Feature | Place | Move | Fly | Description |
|---------|-------|------|-----|-------------|
| Closed mills (own − opp) | 14 | 14 | 16 | Each completed line of three |
| Blocked opponent pieces | 10 | 43 | 350 | Pieces with no legal move adjacent |
| Piece count difference | 11 | 10 | 1 | Net piece advantage |
| Two-configurations (own − opp) | 8 | 7 | 0 | Lines with 2 own pieces and 1 empty slot |
| Double-mill pivots (own − opp) | 0 | 42 | 0 | Pieces simultaneously in 2+ closed mills |
| Win configuration | 0 | 0 | 1190 | Opponent reduced to 3 pieces (fly phase) |

Additional terms added on top:

| Term | Place | Move | Fly | Description |
|------|-------|------|-----|-------------|
| Mobility (own − opp) | ×3 | ×8 | ×20 | Number of available move destinations |
| Mill threats (own − opp) | ×8 | ×12 | ×18 | Same as two-configurations but treated separately as an immediate-threat signal |
| Position value (own − opp) | ×2 | ×2 | ×2 | Cross/cardinal nodes score 3; corner nodes score 2 |
| Herding / encirclement | ×6 | ×18 | 0 | Own pieces adjacent to each opponent piece; rewards surrounding opponent pieces to shrink their escape space |
| Near-blocked pressure (opp − own) | 0 | ×30 | 0 | Opponent pieces with **exactly 1 legal move** remaining — one step from total blockade |
| Mill-wrapping pressure (own − opp) | 0 | ×40 | ×60 | Own pieces occupying exit squares of opponent closed mills; surrounded mills cannot easily cycle. Returns 0 when the opponent is in fly phase (adjacency confinement irrelevant). |

Cross/cardinal nodes (`d7`, `a4`, `g4`, `d1`, and the equivalent middle and inner ring nodes) connect three lines instead of two, making them more tactically flexible.

### Tactical move bonuses

`tactical_move_bonus()` in `heuristics.py` is added directly to each root-move score *after* negamax returns. Unlike the negamax-internal `evaluate()` score, these bonuses are not negated through the tree — they only reward specific move qualities at the root:

| Bonus | What it rewards |
|-------|-----------------|
| `close_mill` | Mills closed this move (captures enabled) |
| `block_opponent_mill` | Opponent's immediately closeable mills neutralised |
| `stop_opponent_mills` | Opponent 2-configs dismantled this move |
| `setup_mill` | New own 2-configs gained (placement AND move phase) |
| `cycling_mill` | Gaining a mill slide-out opportunity (capped at 1 per move) |
| `feeder_diamond` | Landing on a fork square that simultaneously closes 2+ own 2-configs |
| `mill_opening` | Deliberately opening a cycling-ready mill to enable a future capture |
| `scatter_placement` | Placing non-adjacent to own pieces in the first 6 moves |
| `late_mill_bonus` | Closing an outer/middle mill on placement moves 7–9 |
| `mill_trap_build` | Gaining a 3rd+ open mill while already dominant (zugzwang builder) |
| `mobility_reduction` | Each opponent legal move removed by this move (move phase only; herding bonus) |
| `placement_busy_scan` | Placement-phase busy-opponent forcing chain (see below) |
| `defer_for_chain` | Extra bonus (pieces 7-9 only) for skipping an available mill to execute a level-4 forcing chain |
| `convergence_block` | Bonus for disrupting opponent convergence cluster (placement phase) |
| `ring_crowding_penalty` | Penalty for placing the 6th+ own piece on a single ring (placement phase) |
| `ring_cardinal_bonus` | Bonus for placing on a cardinal connector when opponent has 3+ pieces on one ring (placement phase) |
| `fork_anticipation` | Bonus for blocking a square the opponent could use within 2 placements to create a double mill threat |
| `locked_mill_penalty` | Penalty per own closed mill that has zero exit squares (all neighbours opponent-occupied) — applied in `evaluate()` move phase |
| `locked_mill_escape` | Bonus for moving a piece out of a locked mill toward a new 2-config (move phase) |
| `redirected_pin` | Bonus when a move causes an opponent piece to simultaneously guard two distinct own 2-configs |
| `block_cycling_priority` | Bonus for blocking the fork arm with higher cycling freedom (placement + move phase, not fly) |

### Placement busy-opponent scan-ahead

`_placement_chain_scan()` in `heuristics.py` is called from `tactical_move_bonus` during the placement phase. It performs a forward scan of up to 4 half-moves (AI placements interleaved with opponent responses) to find **forcing chains** — sequences where every AI placement compels an opponent response — and ideally ends with a mill closure on the final piece.

#### Scan quality levels (returned value 0–4)

| Level | Meaning |
|-------|---------|
| 0 | No forcing initiative from this position |
| 1 | One immediate 2-config threat (opponent must block) |
| 2 | Sustained pressure: forcing threat persists after one opponent response |
| 3 | Fork reachable within the chain (2 simultaneous threats opponent cannot both block) |
| 4 | Clean forcing sequence found where the final piece closes a mill |

The bonus added to the root score is `placement_busy_scan × (level − 1)` for level ≥ 2 (level 1 gets 40% of the weight). Default `placement_busy_scan = 120`.

#### Two-for-one placements

The scan explicitly rewards placements that **block an opponent 2-config while simultaneously creating an own 2-config in a different mill line**. These "two-for-one" moves maintain initiative even when defending: the opponent's threat is neutralised but the AI immediately presents a new one.

#### Weak-opponent amplification

When the `Coordinator` observes that the opponent's last move scored below `poor_move_threshold` (the same threshold used for poor-move commentary), it sets `GameAI._opp_last_weak = True`. While this flag is active the placement busy-chain bonus is multiplied by 1.5, expressing the imperative: *exploit passive play by locking in a forcing sequence before the opponent gets back on track*. The flag is cleared automatically after the AI completes its move.

#### Why this matters

A strong human player builds placement sequences where every move creates a threat the opponent cannot ignore. This keeps the opponent "busy" responding while the real mill plan develops in a parallel part of the board. Without this scan the AI evaluates each placement in isolation and may pass up chains like:

```
Black:  e3 (→ future d3-e3-c3 threat)
White responds to something else
Black:  d3 (→ d3-e3 now a 2-config, c3 empty)  
White still busy
Black:  c3 (→ closes c3-d3-e3, captures)
```

With the scan, `c3` placement gets a level-4 bonus because the search found the chain; earlier moves in the chain (`e3`, `d3`) also get level-1/2 bonuses steering the AI toward building the sequence from the start.

### Convergence cluster blocking

`_convergence_cluster_count(board, opp)` counts mills where the opponent has 3 pieces that can each reach a distinct mill square within 2 adjacency moves, along paths not blocked by own pieces. Bipartite matching ensures 3 distinct pieces are assigned to 3 distinct squares.

The tactical bonus fires during placement phase when a move reduces the opponent's convergence cluster count. Default weight `convergence_block = 250` per cluster disrupted.

**Example** (game: 1.d6d7 2.f4a7 3.g7a1 4.b4a4×d6 5.d6b2 6.d1g4 7.f2f6 8.d3d2 9.c4 [Black to place]):

White's pieces c4, d3, f4 form a convergence cluster on the c3-d3-e3 mill:
- c4 → c3 (1 step)
- d3 already at d3 (0 steps)
- f4 → e4 → e3 (2 steps through empty e4)

Black should place at c3 or e3 to disrupt the cluster, not at passive squares like b6.

The detection covers any mill anywhere on the board — outer ring, middle ring, inner ring, and cross-ring mills.

### Ring crowding penalty

The three concentric rings (outer: a7/d7/g7/g4/g1/d1/a1/a4; middle: b6/d6/f6/f4/f2/d2/b2/b4; inner: c5/d5/e5/e4/e3/d3/c3/c4) each contain 8 positions. Concentrating 4 or more own pieces on a single ring during placement creates a "side-column" trap: those pieces share few exit squares and can be completely locked out of the board by an opponent who controls the cardinal connector squares between rings.

`ring_crowding_penalty` (default 150) is subtracted from the tactical bonus whenever a placement would be the 6th+ own piece on any single ring. This fires only in placement phase — in move phase the penalty is not needed since distribution is already fixed.

An opponent over-committing to one ring is *exploitable*, not dangerous — three pieces on a ring can be contained by controlling the two connecting cardinal-point squares adjacent to that ring plus one square within the ring. The AI does not penalise opponent ring crowding; instead, the existing cardinal_block bonus steers the AI to occupy those connector squares naturally.

### Herding and mobility squeeze

The AI can win by reducing the opponent's legal move count to zero — a "blockade" win — rather than by always chasing mill captures. Two dedicated signals support this strategy:

**`_squeeze_count(board, color)`** counts pieces of `color` with **exactly 1 legal adjacent move** remaining. A fully-blocked piece (0 moves) is already counted by the blocked-pieces signal; squeeze captures the intermediate state — a piece with one escape route, close to being trapped. The move-phase weight `_NEAR_BLOCKED_WEIGHTS["move"] = 30` rewards positions where the opponent has more near-blocked pieces than you do.

**`mobility_reduction` bonus** in `tactical_move_bonus`: each opponent legal move removed by the current move earns a direct reward (`default = 15` per move removed). This fires only in move phase. In the game-1 example, White playing `d3→c3` removes one escape from Black's c4 piece; the bonus accumulates alongside the near-blocked positional signal to guide the search toward the herding sequence even before the depth-4 forced-win is visible.

**Move ordering — squeeze targets**: `_squeeze_targets(board)` returns the set of empty squares that are the *last* escape route of a nearly-trapped opponent piece. These squares are promoted to **priority 1** in `_order_moves` (same level as blocking an opponent mill threat), ensuring the search tree explores herding moves early and finds the forced blockade at lower depths.

The herding strategy from book Figure 82 — where Black places one piece adjacent to every White piece and gradually closes in until White has no moves — is reflected in the combination of increased herding weight (`_HERD_WEIGHTS["move"] = 18`) and the new squeeze signal, which collectively reward positions that tighten the opponent's range of motion over several moves.

### Free-piece assembly

In the move phase, two complementary signals create a pull gradient that steers isolated pieces toward productive formations:

**`_free_piece_assembly(board, color)`** counts own pieces that are *not* participating in any closed mill or 2-config but sit **directly adjacent** (1 step) to a piece that *is* in a 2-config. These pieces are one slide away from joining a developing formation. Weight: ×65.

**`_assembly_reach_count(board, color)`** counts the same category of free pieces that are **2 adjacency steps** from any 2-config piece — adjacent to a step-1 neighbour but not adjacent to the 2-config piece itself. This captures pieces that are still converging from further away. Weight: ×22.

Four terms create a distance-graduated incentive:

| Step | Weight | Function |
|------|--------|----------|
| 1 (adjacent) | ×65 | `_free_piece_assembly` |
| 2 | ×22 | `_assembly_reach_count` |
| 3 | ×10 | `_assembly_step3_count` |
| 4 | ×4  | `_assembly_step4_count` |

The gradient (65 / 22 / 10 / 4) creates a smooth pull inward: the closer a free piece gets to a formation, the more valuable its position becomes. A piece 5+ steps from any 2-config earns nothing. Neither term fires in fly phase, where adjacency constraints do not limit assembly. A piece more than 4 hops away provides no positional benefit and the AI should not sacrifice other priorities to drag it across the board.

### Fly-phase pin rule

When the AI enters fly phase (3 pieces remaining), each piece can jump to any empty square. This freedom introduces a trap: if an own piece occupies the **closing square** of an opponent 2-config (opponent has the other two squares in that mill), moving that piece away gives the opponent an immediate mill closure and a capture.

`_pinned_fly_squares(board, color)` detects these situations. For every mill, if it contains 2 opponent pieces and 1 own piece, that own square is "pinned" — the piece is the sole blocker of an opponent closure. The function returns the set of all such pinned squares.

In `choose_move()`, after mandatory-block and bad-move-ban filtering, moves that slide FROM a pinned square are removed from the candidate list. The safety guard — `if unpinned` — ensures this filter never reduces the move list to zero (in the rare case that every remaining fly piece is simultaneously pinned, the AI retains all moves rather than returning an empty result).

**Example**: Black has `g4` and `g1` (2-config in the `g7-g4-g1` mill), White (fly) has a piece on `g7`. `g7` is pinned — moving it gives Black a free mill closure. White's other two pieces remain unpinned and are preferred.

### Endgame supplement

When the EndgameRecognizer marks the game active (≤ 11 pieces total on the board), an extra term is added:
- `(own_mobility − opp_mobility) × 20`
- If the opponent has ≤ 2 moves and we have ≥ 4: `+200`
- If we are the player running a mill cycle (open/close a mill repeatedly to force captures): `+150`

### Normalisation for the graph

The raw integer score is unbounded. To produce the −1…+1 value shown in the position strength graph, `position_eval()` computes:

```
graph_value = tanh(raw_score / scale)
```

where `scale` is phase-dependent: 120 during placement, 180 during movement, 280 during fly. The larger scale in the fly phase prevents the graph from pinning to ±1 on small material swings when so few pieces remain.

A positive value (top half of the graph) means White is ahead; negative (bottom half) means Black is ahead. The dot colour follows the leading side: white circle when White leads, dark circle when Black leads.

---

## 3. Fly-Phase Imperatives and AI Limitations

### What the fly phase is

A player enters fly phase when they are reduced to exactly 3 pieces and all 9 of their pieces have already been placed. In fly phase, the player may move any own piece to any empty square in a single move — adjacency no longer restricts movement. The fly phase is both an opportunity (freedom of motion) and a danger signal (one more capture loses the game).

### Phase transitions that matter

| Own pieces | Opponent pieces | Own phase | Opp phase | Description |
|------------|-----------------|-----------|-----------|-------------|
| ≥ 4 | ≥ 4 | move | move | Normal movement. Both sides restricted to adjacency. |
| 4 | 4 | move | move | **4v4 — imminent fly transition.** See below. |
| ≥ 4 | 3 | move | fly | **3v4 from AI's perspective** (AI has more pieces). AI has adjacency restriction, opponent flies freely. |
| 3 | ≥ 4 | fly | move | **4v3 from AI's perspective** (AI is in fly). AI jumps anywhere, opponent restricted. |
| 3 | 3 | fly | fly | Both players fly. Winner is determined by mill structure and tactical precision. |

### 4v4 — The fly-sacrifice hesitation

When both sides have exactly 4 pieces, neither has entered fly phase yet. An AI capture would reduce the opponent to 3 pieces, giving them fly freedom. This is the source of the **fly-sacrifice hesitation**: unless the AI has a structurally strong position or a dominant mill setup, giving the opponent fly may be disadvantageous — a flying opponent can escape adjacency traps that were nearly complete.

**Heuristic signal:** `_fly_asymmetry()` returns a non-zero penalty when the AI's colour is in move phase and the opponent would enter fly phase as a result of a capture. The penalty weight is `_FLY_ASYM_WEIGHTS["move"] = 80` per asymmetry unit. This is suppressed when `force_aggressive=True`.

**When to override:** The Force Capture button (`btn-force-cap`) sends `force_aggressive: true` to the server, which sets `force_aggressive=True` in `evaluate()`. This removes the fly-asymmetry penalty entirely, making the AI treat 4v4 captures as neutral or positive. The button is meaningful only when the human player has 4 pieces — once the human drops below 4, the hesitation is gone regardless (opponent is already in fly or the position is beyond the 4v4 threshold). Enhancement B-1 (see PLAN.md) gates the button's availability to exactly this condition.

**When the hesitation is correct:**
- The AI has a cycling mill setup (it can force captures without a direct capture).
- The 4v4 position is tense and the AI's mill structure is stronger.
- Giving fly prematurely creates a 3v4 scenario where the flying opponent evades traps.

**When the hesitation is wrong:**
- The AI has a strong 3-piece structure (two pieces in a 2-config, third ready to close) — fly from the better structure wins.
- The opponent's 4 remaining pieces are on a single ring with no connected mills.
- A capture opens an immediate 3v3 flying endgame the AI can win from the mill-structure advantage.

### 4v3 from AI's perspective (AI has 4, opponent flies)

When the AI has 4 pieces and the opponent is flying (3 pieces), the AI retains adjacency restriction while the opponent can jump anywhere.

**Priority order in this phase:**
1. **Close own mills** — mills must be closed via adjacency (AI cannot fly). Any available mill closure is urgent.
2. **Avoid abandoning pinned squares** — the fly-phase pin rule (`_pinned_fly_squares`) does not apply here (AI is not in fly), but effectively the AI should never leave the closing square of an opponent 2-config unguarded if the opponent can fly there immediately.
3. **Block opponent mills** — the opponent can close a mill from any empty square in a single move. Every own 2-config the opponent has is an immediate threat if the closing square is empty. Weight: `block_opponent_mill` (400) fires as normal since the opponent can close their mills regardless of adjacency.
4. **Maintain piece pressure** — with 4 vs 3, the AI has a material advantage. Avoid unnecessary captures that would reduce the AI to 3 (3v3 is less decisive than 4v3 if the AI's structure is weaker).
5. **Encirclement signal** — `_NEAR_BLOCKED_WEIGHTS["move"] = 30` rewards squeezing the opponent's movement. In 4v3, the opponent flies freely so this signal is suppressed for the opponent (fly pieces are never "near-blocked").

**AI limitation in 4v3:** The AI's adjacency restriction means it cannot re-position quickly to respond to the flying opponent's threatening placements. If the AI has pieces scattered on different rings without connecting mills, the flying opponent can exploit this by landing on closing squares the AI cannot reach in one move. The 4v3 phase rewards the side with a **pre-formed 2-config** — one move from a mill — rather than scattered material.

### 3v4 from AI's perspective (AI flies, opponent has 4)

When the AI has 3 pieces and flies, the opponent retains adjacency restriction.

**Priority order in this phase:**
1. **Fly-phase pin rule** — `_pinned_fly_squares()` identifies AI pieces sitting on the closing square of an opponent 2-config (AI piece is the sole blocker of an immediate opponent mill). Moving such a piece gives the opponent a free mill closure + capture. The search filters out moves that slide from a pinned square, unless all squares are pinned (safety guard preserves at least one move).
2. **Build or close own mills** — with fly freedom, the AI can jump to any empty square in one move. The first priority is to close an available mill (capturing an opponent piece) or to complete a 2-config if none exists.
3. **Fly fork setup** — when the AI has two separate 2-configs, both closeable from different empty squares, the opponent cannot block both with one piece placement. This "fly fork" is the fly-phase equivalent of a double mill. Bonus: `fly_fork_bonus` (default 200) in `tactical_move_bonus()` fires when the AI transitions from < 2 own 2-configs to ≥ 2 in the same move.
4. **Fly-free-close bonus** — reward closing a mill using a piece that was previously NOT in a 2-config (jumped in from a non-threatening square). This is the unpredictability advantage of fly: the opponent cannot predict where the closing piece will come from. Bonus weight: `fly_free_close_bonus` (default 150).
5. **Avoid 3v3 if structure is weak** — if the AI has 3 pieces but a poor structure (no 2-configs, pieces scattered), capturing the opponent down to 3 creates 3v3 flying from a disadvantaged position. The `_fly_asymmetry()` signal in evaluate() suppresses aggressive captures in this scenario (`_FLY_ASYM_WEIGHTS["fly"] = 0` means this weight is unused in fly-vs-move; the 3v4 protection is instead encoded in the `_fly_asymmetry()` function's `phase == "fly" and opp_pieces == 4` branch, which rewards separated opponent groups — disconnected pieces are harder to defend when the AI can fly).

**Fly-phase mill-wrapping:** The mill-wrapping signal (`_MILL_WRAP_WEIGHTS["fly"] = 60`) is active when the AI is in fly phase and the opponent is not. The AI can jump to any exit square of an opponent closed mill, physically surrounding it. This score is computed but `_mill_wrapping()` explicitly returns 0 when the opponent is also in fly phase (surrounded mills become irrelevant when the surrounded pieces can jump away).

### 3v3 — Both players fly

When both players have 3 pieces and both are in fly phase:

- Adjacency is irrelevant for both sides.
- Every empty square is 1 move away.
- The winner is determined almost entirely by **mill structure** (who has a 2-config or can form one first) and **tactical precision** (who avoids giving the opponent an unguarded closing square).

**Key signal:** `fly_fork_bonus` becomes decisive. The player who first achieves two simultaneous 2-configs that cannot both be blocked wins by force. All evaluation weights in the `["fly"]` columns apply (mobility ×20, mill-wrapping ×60, blocked ×350).

**AI limitation in 3v3:** The AI's search tree expands rapidly (up to ~54 legal fly moves per side at depth 3). At lower difficulties the search may not reach the decisive fork, allowing the opponent to gain the fork first. At difficulty 7+ with iterative deepening, the search reliably finds the forcing fork within the time budget.

### The `force_aggressive` flag

Setting `force_aggressive=True` (via the Force Capture button) suppresses three distinct hesitation behaviours:
1. **`_fly_asymmetry`**: no penalty for captures that give the opponent fly phase.
2. **3v4 opponent-separation reward**: the term that rewards keeping the flying opponent's pieces separated is disabled, so the AI no longer prefers cautious non-capture moves.
3. **6v4 sacrifice-to-fly quality check**: the AI no longer evaluates whether its own 3-piece structure is strong enough before accepting 6v4 → 3v4 transitions.

This flag is session-local (resets on new game) and is not recorded in training data.

### Current AI limitations in fly phase

| Limitation | Impact | Workaround |
|------------|--------|------------|
| Pin-rule filter can leave obvious winning moves unexplored | In rare positions where all 3 fly pieces are simultaneously pinned, all moves are re-enabled (safety guard). The AI may then choose a suboptimal move. | The safety guard is correct; the position is usually already losing. |
| 3v3 search tree depth at difficulty < 5 | AI may miss a 2-move forced fork at depth 2. | Increase difficulty or use Force Move to get a deeper search result. |
| Fly-fork detection requires 2+ own 2-configs to exist at once | The bonus fires reactively (after moving into the fork), not proactively. The AI does not pre-plan the move sequence that will *create* the fork. | Enhancement B-4 (fork anticipation) addresses the pre-planning gap for the placement phase; a similar anticipation bonus for the fly phase is a future enhancement. |
| `_mill_wrapping` returns 0 in 3v3 | Mill wrapping is disabled in 3v3 since the wrapped pieces can fly away. | Correct by design. Mill wrapping is only useful when opponent pieces are adjacency-constrained. |
| Assembly signals (`_free_piece_assembly`, `_assembly_reach_count`) are off in fly phase | Fly pieces can jump anywhere so step-counting is meaningless. These signals are correctly disabled. | No workaround needed — by fly phase pieces should already be assembled. Enhancement B-5 addresses assembly before reaching fly phase. |

### Terminal positions

If the position is already won or lost, `evaluate()` returns `±INF` immediately without computing any features. The negamax search propagates these wins/losses back through the tree, and a win found at a shallower depth is ranked above one found deeper by subtracting the remaining depth from INF (`INF - depth`). This ensures the AI takes the fastest available win and defends against the most immediate threats first.

---

## 4. Search Stack (SE-Series)

### SE-1 — Transposition Table + Zobrist Hashing

The same board position can be reached via many different move sequences (transpositions). Without a TT, `_negamax` re-evaluates every transposed position from scratch.

**Zobrist keys** (`game/zobrist.py`): 51 random 64-bit integers generated once at import time with a fixed seed:

- `PIECE_KEYS[color_idx][sq_idx]` — XOR in when a piece of that colour occupies a square (48 keys: 2 colours × 24 squares).
- `PLACED_DONE_KEYS[color_idx]` — XOR in when `pieces_placed[color] >= 9` (i.e., that side has finished the placement phase). Two keys, one per side.
- `SIDE_KEY` — XOR in when it is Black's turn to move.

`BoardState.hash_key` is maintained **incrementally** inside `apply_move()`: each move XORs out removed pieces, XORs in placed/moved pieces, toggles the done-placing bit if it crosses 9, and flips `SIDE_KEY`. `from_setup()` and `new_game()` compute the hash from scratch via `hash_board()`. This ensures hashes are consistent across all transposition paths without a full O(24) scan per node.

**TranspositionTable** (`ai/transposition_table.py`): fixed-size 2^18 = 262 144-slot list, indexed by `hash_key & MASK`. Each occupied slot stores `(hash_key, depth, score, flag, from_sq, to_sq)`. Replacement policy: **depth-preferred** — a new entry only evicts the existing one if `new_depth >= stored_depth`, ensuring shallow searches never throw away work from deeper ones.

Flag meanings:

| Flag | Meaning |
| --- | --- |
| `EXACT` | Stored score is the true minimax value |
| `LOWER_BOUND` | Search failed high (beta cutoff); score is a lower bound |
| `UPPER_BOUND` | Search failed low (no move improved alpha); score is an upper bound |

**Integration in `_negamax`**: the TT is probed after the terminal and depth=0 checks, before generating moves. If the entry depth is sufficient: `EXACT` → return immediately; `LOWER_BOUND` + score ≥ beta → return; `UPPER_BOUND` + score ≤ alpha → return. The stored best-move `(from_sq, to_sq)` is promoted to the front of the move list before `_order_moves` runs — this is the primary ordering gain. On exit, the result is stored (computing the flag from `alpha_orig`/`beta`). The TT is cleared at the start of each `choose_move()` call so context-dependent evaluation values (endgame_state, weights, etc.) never leak across turns.

---

## 4b. Advanced Tactical Enhancements (B-Series)

### B-2 — Placement chain deferral

`_placement_chain_scan()` now runs even when an immediate mill closure is available, but **only during the late placement window (pieces 7–9, i.e. `pieces_placed >= 6`) and only when the scan returns level 4** (a clean forcing sequence that closes a mill on the final piece). In that case an extra `defer_for_chain` bonus (default 300) is added on top of the normal chain bonus, giving the AI an incentive to forgo an early mill in favour of a superior 9th-piece closing sequence.

Earlier in the game (pieces 1–6), an immediate mill closure combined with a capture is categorically more forcing than a 2-move-deferred mill, so the defer override is intentionally suppressed. The scan also now detects 4-level chains when the AI has exactly 2 pieces left to place (`our_rem >= 2` instead of the previous `>= 3`).

### B-3 — Ring crowding: cardinal position preference

When the opponent has concentrated 3 or more pieces on a single ring, the AI receives a bonus for placing on **cardinal cross-node squares adjacent to that ring** — the connector positions between rings that control the opponent's exit lines.

- Outer ring concentrated → prefer middle-ring cardinals (`d6`, `f4`, `d2`, `b4`).
- Middle ring concentrated → prefer outer or inner cardinals.
- Inner ring concentrated → prefer middle-ring cardinals.

The bonus is `cardinal_block × 0.5` per ring concentration. This supplements the existing `cardinal_block` bonus (which rewards placing on cross-nodes generally) with a ring-aware context signal.

### B-4 — Fork mill anticipation

`_fork_in_n(board, opp, n=2)` returns the set of empty squares that, if occupied by the opponent within `n` moves, would create a double mill threat (two simultaneous 2-configs). Placing on any of these squares blocks the anticipated fork before it materialises.

A `fork_anticipation` bonus (default 90) fires when the AI's placement or move lands on a fork-in-2 square. This fires in placement phase and move phase; not fly phase. Unlike the existing `block_opponent_mill` (which only reacts to threats closeable this turn), fork anticipation looks one step further ahead, preventing the structural conditions for a fork from arising.

### B-6 — Opponent losing-line exploitation

`TrajectoryDB.query_opponent_loss(notations, opp_color)` provides a second signal complementing the existing `query()`. Where `query()` scores moves by how often *we* win from this position, `query_opponent_loss()` scores moves by how often the *opponent* loses — a different signal when the database contains many drawn games.

The `Coordinator` merges both signals with a blending formula controlled by the `loss_exploit` slider (default 150 → 1.5× weight on the loss-exploit signal):

```
blended = (win_rate_delta + loss_exploit_multiplier × loss_rate_delta) / (1 + loss_exploit_multiplier)
```

If no win-rate data exists for a line, the loss-exploit hint stands alone weighted down by the same formula. This keeps the signal in the same `[-0.5, +0.5]` statistical range as the win-rate signal.

**Personality presets:** Aggressive = 200 (2×); Defensive = 100 (1×); Scholar/Positional = 180 (1.8×); Balanced = 150 (1.5×); Chaos = 50 (0.5×).

### B-7 — Locked mill escape and redirected-pin creation

**Locked mill:** A closed mill is *locked* when every exit square (any neighbour of any mill piece that is not within the mill itself) is occupied by an opponent piece. A locked mill contributes zero cycling value — the pivot piece has nowhere to slide to force repeated captures.

`_is_mill_locked(board, color, mill)` detects this condition. A `locked_mill_penalty` (default 80) is subtracted from `evaluate()` per own locked mill in move phase, reflecting the stranded capital cost.

When the AI moves a piece **out** of a locked mill toward a new 2-config, a `locked_mill_escape` bonus (default 160) is added. The bonus is gated: it does not fire if the destination square does not contribute to any own 2-config in the resulting position (the freed piece must immediately start contributing elsewhere). Neither signal fires in fly phase (adjacency locks dissolve) or placement phase.

**Redirected pin:** `_creates_redirected_pin(board, color, from_sq, to_sq)` detects when a move causes an opponent piece to simultaneously occupy the blocking position for **two distinct** own 2-configs — a "double-pin". A pinned piece in this sense cannot move without surrendering at least one mill threat.

A `redirected_pin` bonus (default 140) fires in move phase when this condition is detected. The bonus is capped at 1 per move regardless of how many pieces are double-pinned. It does not fire in fly phase.

### B-8 — Forked mill blocking: choose the higher-cycling fork arm

When the opponent threatens two mills simultaneously (a fork), the AI must choose which arm to block. The default is to block the arm on a cardinal cross-node square (`cardinal_block` preference). However, the strategically correct choice depends on **cycling freedom** — how many empty exit squares a closed mill has.

A mill with high cycling freedom (several empty exits) can repeatedly open and re-close to force captures. A mill with low cycling freedom (few empty exits, verging on locked) is nearly static. The AI should block the high-cycling arm and surrender the low-cycling arm.

**`_mill_cycling_freedom(board, color, mill)`** counts empty non-mill exit squares. Higher = more dangerous to surrender.

**`_opponent_fork_arms(board, color)`** returns all (mill, closing_square) pairs where the opponent has 2 pieces and 1 empty. When 2+ exist simultaneously it is a fork.

**Cardinal exception:** If an own piece already occupies any square adjacent to the *closing square* of the cardinal arm (`_own_piece_adj_to_closing`), that piece constrains the cardinal mill's cycling in practice. In that case the cardinal priority is removed for that arm and a pure cycling-freedom comparison is used.

**Signal:** A `block_cycling_priority` bonus (default 120) fires when the AI's placement or move occupies the closing square of the highest-effective-cycling-freedom fork arm. The bonus scales with the freedom differential: `block_cycling_priority × (1 + freedom_diff × 0.1)`. Gate: placement and move phase only, not fly phase.

**Example (cardinal exception):** White threatens g7-g4-g1 (1 exit if closed) and g4-f4-e4 (4 exits if closed). Black has a piece at e3, adjacent to e4 (g4-f4-e4's closing square). Black should block g7 (give White the cardinal mill), because e3 already constrains the cardinal mill's most dangerous exit. Without e3, Black would block e4 instead.

### What the LLM sees when it makes a move or commentary

`MillsLLM` receives the following context in each prompt:

- **System prompt** — `_BOARD_RULES` (node list, phase rules, notation format) plus a task-specific system prompt (e.g. `_MOVE_SYSTEM`, `_COMMENT_SYSTEM`).
- **Board** — raw ASCII grid from `board.to_display_grid()`. This shows which squares are occupied by W or B but gives the LLM no explicit summary of closed mills, 2-config threats, piece counts, or the current phase.
- **Move history** — the notation sequence so far (e.g. `1. d7 f4 2. d6 d5 …`), passed via `_move_history_block()`.
- **Legal moves** (deliberation path only) — one per line in notation form.
- **Engine top choice and score** (deliberation path only) — e.g. `ENGINE TOP CHOICE: d2-d3`, `ENGINE SCORE: +0.42`.
- **Opening context** (when a recognised opening is active) — name, confidence, book move for this ply, strategic notes, common blunders.
- **Endgame context** (when `endgame_state.active`) — phase name, pattern notes.
- **Strategic memory** (deliberation path) — nearest-neighbour positions from ChromaDB, retrieved by `_strategy_context()`.

**What the LLM does NOT receive:**

- Which mills are currently closed, and for whom.
- Which 2-configs (one-away threats) exist.
- Explicit piece counts or how many pieces each side has in hand.
- Current game phase stated in plain text (it must infer from move count or board appearance).
- Canonical NMM mill line names (e.g. "Outer bottom: g1-d1-a1") — the LLM guesses line names from the ASCII layout and sometimes gets them wrong.

**Known failure mode (Bug B-9):** Because the LLM only has the ASCII grid to work from, it can misidentify which line a mill was formed on (e.g. commenting "mill on the e-line" when the mill was on the outer bottom), or ask about a mill threat on a line that is already occupied and immobile. See Bug B-9 in PLAN.md for the fix plan.

## 5. Retrograde Endgame Database (B-23)

### What it is

`ai/endgame_solved_db.py` provides a fully solved, compact Win/Draw/Loss (WDL) table for every legal 3v3 fly-phase position. Unlike `EndgameDB` (which learns from self-play games), this table is computed offline by retrograde analysis and is mathematically exact.

The table is built once with `tools/build_endgame_db.py` and saved as `data/endgame/endgame_3_3.wdl`.

### Position encoding

All 3v3 fly positions are encoded as a single integer in `[0, 5_383_840)` using the **combinatorial number system**:

```
pos_id = combo_rank(white_squares) * C(21,3) * 2
       + combo_rank(black_squares_excluding_white) * 2
       + turn_bit           # 0 = White to move, 1 = Black to move
```

- `combo_rank([c₀, c₁, …, c_{k-1}]) = Σ C(cᵢ, i+1)` — a bijective mapping from any sorted k-subset to an integer in `[0, C(n, k))`. This is the standard combinatorial number system and is NOT lexicographic order.
- White occupies 3 of the 24 squares → `C(24,3) = 2024` values.
- Black occupies 3 of the remaining 21 → `C(21,3) = 1330` values.
- `TABLE_SIZE_3_3 = 2024 × 1330 × 2 = 5_383_840`.

WDL values are packed 2 bits per slot (4 slots per byte): `UNKNOWN=0`, `WIN=1`, `LOSS=2`, `DRAW=3`. Total table size: 1_345_960 bytes (~1.3 MB).

### Offline solver (`tools/build_endgame_db.py`)

The solver uses **D4 dihedral symmetry** to reduce computation by ~8×. NMM mills and adjacency are all D4-invariant, so the WDL value of a position is identical to the WDL of any of its 7 symmetric equivalents. The algorithm exploits this as follows:

1. **Canonical precomputation.** All 5_383_840 position IDs are scanned once and those in the **canonical (bitmask-minimum) D4 equivalence class** are collected into `canonical_ids` (~672 K entries, one-eighth of the total). Canonicality is tested via `_canonical_indices(w, b)` in `build_endgame_db.py`, which applies all 7 non-identity D4 bitmask permutations and returns the minimum `(w_mask, b_mask)` form.

2. **Pass 0 — terminal wins (canonical only).** For every canonical position where the side to move can close a mill (fly move to any empty square that completes 3-in-a-row) and then capture any opponent piece, that position is marked `WIN` — the mover can immediately reduce the opponent to 2 pieces.

3. **Propagation passes (canonical only).** Repeated passes over `canonical_ids` apply:
   - A position is `LOSS` if **all** successors are `WIN`.
   - A position is `WIN` if **any** successor is `LOSS`.
   - Successor encoding always canonicalises the resulting `(w, b)` pair before looking up the table, so every lookup lands on a canonical entry.

4. **DRAW assignment (canonical only).** Canonical positions still `UNKNOWN` after convergence are marked `DRAW`.

5. **Fill pass.** A single final pass over all 5_383_840 positions copies the WDL from each position's canonical equivalent to fill the non-canonical slots. The output file is fully populated — `EndgameSolvedDB.query()` requires no canonicalization at runtime.

Mill detection uses bitmasks: `_MILL_MASKS_FOR[i]` stores each mill mask that covers square `i`. `_closes_mill(piece_mask, to_idx)` runs in O(mills_per_square) time with no memory allocation.

D4 permutation pairs are precomputed at module load from `ai/board_symmetry._BOARD_PERM` as `_BPERM_MASKS` — a list of 7 × 24 `(old_bit, new_bit)` pairs — avoiding heap allocations during the inner canonicalization loop.

### Runtime usage

At server startup, `web/app.py` loads the `EndgameSolvedDB` object from `endgame_solved_dir` (default `data/endgame`). This is passed to every `GameAI` constructor. If the file is absent the object silently reports `is_available() == False` and is ignored.

In `choose_move()`, the endgame DB is consulted **before** the fullgame DB when all of these guards are true:

- Both sides have placed all 9 pieces (`pieces_placed >= 9`).
- Each side has exactly 3 pieces on the board (`pieces_on_board == 3`).
- Total pieces on board ≤ 6.

`query(board)` returns `"W"` (AI wins), `"L"` (AI loses), `"D"` (draw), or `None` (out of range / file not loaded).

### Known limitation (B-48)

When the DB returns `"W"`, `choose_move()` currently returns `moves[0]` rather than the specific move that leads to a `LOSS`-labelled successor. In a won position this means the AI always converts eventually, but may take extra moves. The fix (iterate successors, pick the first that decodes to `WDL_LOSS` from the opponent's perspective) is tracked as Bug B-48.

---

### 1-config approach heuristic (`_one_config_approach`)

Added to fill the assembly gap where no 2-config yet exists. The four existing assembly functions (`_free_piece_assembly`, `_assembly_reach_count`, `_assembly_step3_count`, `_assembly_step4_count`) all measure distance from a free piece to the nearest **2-config piece**. If no 2-config exists — common in the early game — all four return zero and the AI gets no signal about building toward future mills.

`_one_config_approach(board, color)` addresses this by measuring free-piece proximity to the **empty squares of 1-config mills** (mills where the color has exactly 1 piece and 2 empty squares):

- Step-1 (+2): free piece is directly adjacent to an empty square of a 1-config mill.
- Step-2 (+1): free piece is one hop away from such an empty square (adjacent to the step-1 halo).

Pieces already in a closed mill, a 2-config, or another 1-config are excluded from counting (they are already contributing to a formation). Weight: `×12` in `evaluate()` move-phase assembly block. Does not apply in fly phase.
