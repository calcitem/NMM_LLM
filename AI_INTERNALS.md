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
| `convergence_block` | Bonus for disrupting opponent convergence cluster (placement phase) |
| `ring_crowding_penalty` | Penalty for placing the 6th+ own piece on a single ring (placement phase) |

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

Together the two terms create a distance-graduated incentive: step-1 pieces (×65) are strongly rewarded for being close; step-2 pieces (×22) receive a softer pull. A piece 3+ steps from any 2-config earns nothing, so only pieces making genuine convergence progress are rewarded. Neither term fires in fly phase, where adjacency constraints do not limit assembly.

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

### Terminal positions

If the position is already won or lost, `evaluate()` returns `±INF` immediately without computing any features. The negamax search propagates these wins/losses back through the tree, and a win found at a shallower depth is ranked above one found deeper by subtracting the remaining depth from INF (`INF - depth`). This ensures the AI takes the fastest available win and defends against the most immediate threats first.
