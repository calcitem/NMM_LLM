# AI Internals — Move Selection and Position Evaluation

## 1. How the AI Decides Where to Play

### Search algorithm

The AI uses **negamax with alpha-beta pruning** (`ai/game_ai.py`). Negamax is a simplification of minimax that works by negating the score at each level, so it always maximises from the current player's perspective without needing separate min and max cases.

Alpha-beta pruning discards branches that cannot affect the final result. When the algorithm finds a move that is already worse than something the opponent could have steered toward, it stops searching that branch. In practice this roughly halves the effective search depth compared to plain minimax.

### Depth and time budget

Difficulty 1–8 map to a fixed search depth (2–9 plies respectively). Difficulties 9 and 10 use **iterative deepening**: the search starts at depth 2, completes, then starts again at depth 3, and so on until the time budget is exhausted (20 seconds for difficulty 9, 45 seconds for difficulty 10). The best move found at the last fully completed depth is returned.

A special early-game fast path applies for the **first two placements** the AI makes: regardless of difficulty, those moves use a 1.5-second iterative-deepening budget. The position tree is tiny when only a handful of pieces are on the board, so deep search is wasteful.

### Move selection

`choose_move()` calls `_score_all()`, which runs a full negamax search from every legal root move and returns a scored list. The move with the highest score wins. If `blunder_probability > 0`, there is a chance the AI instead picks from the **bottom quartile** of scored moves — a deliberate mistake for teaching purposes.

### Opening book adjustments

When an opening has been recognised, the scored move list is adjusted before the final selection:
- If the book's recommended next move matches a candidate, its score is bumped by `0.2 × score_range`.
- Moves landing on a square the book lists as a common blunder have their score reduced by `0.3 × score_range`.

This biases the AI toward book lines without overriding the engine entirely.

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
