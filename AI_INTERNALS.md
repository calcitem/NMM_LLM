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

When an opening has been recognised, the scored move list is adjusted before final selection. The adjustment size scales with the **Opening Adherence** slider (0–100 %):

- The book's recommended next move receives an absolute bonus of up to `3000` internal score units at 100 % adherence, scaling linearly down to zero at 0 %.
- Moves listed as common blunders for the current opening receive a penalty of up to `1500` units.

The **TrajectoryDB** (`ai/trajectory_db.py`) indexes every completed game by move-notation prefix. After the opening phase, winner moves receive positive score deltas and loser moves receive negative ones. Deltas in `[−0.5, +0.5]` are statistical hints; a delta of exactly `−1.0` is a hard ban (set by the Bad Move button) and causes the move to receive `−INF+1` regardless of adherence — it is never chosen.

Bad-move bans are also enforced directly inside `choose_move()` via per-FEN position bans (`_pos_bans`), so a banned move cannot be re-played even if the trajectory hint is somehow bypassed.

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

### Free-piece assembly

In the move phase, `_free_piece_assembly(board, color)` counts own pieces that are *not* participating in any closed mill or 2-config but sit adjacent to a piece that *is* in a 2-config. The evaluator rewards the difference between own and opponent assembly counts (weight ×40), steering stranded pieces toward productive formations over several moves without requiring the search to see all the way to the eventual mill.

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
