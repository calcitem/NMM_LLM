# Learned AI Architecture

This document describes the self-learning AI subsystem added under `learned\_ai/`. It is a PyTorch policy/value network trained by self-play reinforcement learning. It plugs into the existing Nine Men's Morris engine through the same `choose\_move(board)` contract used by the heuristic minimax AI (see `docs/AI\_INTERFACE\_MAPPING.md`).

## Design goals

1. **Never duplicate game rules.** Legal-move enumeration always defers to `game.rules.get\_all\_legal\_moves`. The learned policy only *ranks* moves the engine has already declared legal.

2. **Always mask illegal actions.** Illegal action logits are set to `-1e9` before any softmax, so they receive exactly zero probability.

3. **No oracles in the policy.** No tablebases, perfect-play databases, or handcrafted move heuristics enter the network. It learns purely from self-play reward.

4. **Drop-in replacement.** The agent is selectable via an environment variable and exposes the heuristic AI's interface, so the existing game stays fully playable.

## Phase model

NMM naturally divides into phases. The encoder classifies the side-to-move's position into one of five phases (`learned\_ai/models/state\_encoder.py::detect\_phase`):

| ID | Name | Trigger (for the side to move) |
| - | - | - |
| 0 | opening\_placement | still placing AND `pieces\_placed \< 4` |
| 1 | full\_placement | still placing AND `4 \<= pieces\_placed \<= 9` |
| 2 | midgame | both placed AND stm has `\>= 5` on board AND opp has `\> 3` |
| 3 | endgame | both placed AND stm has 4 on board, or opp is down to 3, not yet flying |
| 4 | flying | placed all 9 AND exactly 3 pieces remain (can fly) |


This 5-way split is layered on top of the engine's coarse `place / move / fly` classification from `game.rules.get\_game\_phase` — it never contradicts it.

## State encoding (84 floats)

`encode\_state(board) -\> torch.Tensor` of shape `(84,)`:

| Slice | Size | Meaning |
| - | - | - |
| `\[0:72)` | 72 | 24 positions × 3-way one-hot (empty / white / black) |
| `\[72\]` | 1 | side to move (0.0 = white, 1.0 = black) |
| `\[73:78)` | 5 | phase one-hot (the 5 phases above) |
| `\[78\]` | 1 | white pieces placed / 9 |
| `\[79\]` | 1 | black pieces placed / 9 |
| `\[80\]` | 1 | white pieces on board / 9 |
| `\[81\]` | 1 | black pieces on board / 9 |
| `\[82\]` | 1 | white mills formed / 3 (capped at 1.0) |
| `\[83\]` | 1 | black mills formed / 3 (capped at 1.0) |


Board positions are indexed in `game.board.POSITIONS` order, so index `i` always maps to the same physical square.

## Action encoding (624 actions)

A single unified action space (`learned\_ai/models/action\_encoder.py`) covers all phases, so every head emits the same 624-wide logit vector:

| Slice | Size | Meaning |
| - | - | - |
| `\[0:24)` | 24 | placement on `POSITIONS\[i\]` |
| `\[24:600)` | 576 | movement / fly: `from = src, to = dst` → `24\*src+dst` |
| `\[600:624)` | 24 | capture `POSITIONS\[i\]` (after a mill is closed) |


Because the existing engine emits *atomic* moves (a mill-forming placement or movement is bundled with its capture), the agent decides the move in a single forward pass: it samples the placement/movement index from the primary slice, and — only when that partial move closes a mill — samples the capture target from the capture slice. Both slices are independently masked to the legal set.

`get\_legal\_mask(board)` builds a `(624,)` boolean mask using only the engine's legal-move list and `board.legal\_captures`; the capture slice is left all-False unless a mill can actually be formed on this turn.

## Network: NMMNet

`learned\_ai/models/backbone.py::NMMNet`

```
state (84) ─► Shared backbone MLP  (default 256 → 256 → 128, ReLU \[+dropout\])  
                       │  
        ┌──────────────┼───────────────────────────┐  
        ▼              ▼                             ▼  
  phase head 0   ...  phase head 4            value head  
  (128→64→624)        (128→64→624)            (128→64→1)
```

- **Shared backbone:** learns board geometry and mill patterns once, shared across all phases.

- **Five phase heads:** a `ModuleDict` keyed by phase name. `forward(state, phase\_id, legal\_mask)` routes through exactly one head per call. Each head outputs the full 624-dim action logits, which are then masked.

- **Value head:** a single shared scalar value estimate `V(s)`, used as the REINFORCE baseline.

Masking happens inside `forward`: `logits.masked\_fill(~legal\_mask, -1e9)`.

### Why a shared backbone + heads instead of five separate models?

- NMM phases share board geometry and mill structure; a shared representation learns these once instead of five times.

- Fewer parameters to train early, when self-play data is scarce.

- One checkpoint to manage instead of five.

- The design degrades gracefully: if one phase's learning diverges, its head can later be split into a standalone model without touching the others, because the head boundary is already explicit.

Color is **not** modeled with separate networks — the `side\_to\_move` bit in the state vector conditions a single network that learns symmetric play.

## Training algorithm

`learned\_ai/training/trainer.py`

Default is **REINFORCE with a value baseline**:

- For each episode, play one self-play game; assign terminal reward `+1` win, `-1` loss, `0` draw, from the perspective of the side that made each move (`learned\_ai/training/self\_play.py::assign\_rewards`, with optional discount-to-end via `gamma`).

- Policy loss: `-(log π(a|s) · advantage)`, where `advantage = return - V(s)`.

- Value loss: MSE between `V(s)` and the realized return.

- An entropy bonus encourages exploration; gradients are clipped at norm 5.

- A **PPO** branch is available via `training.algorithm: ppo` in the config (clipped surrogate objective) for later experiments.

Batches are grouped by phase so each head's forward pass runs once per update.

### Curriculum (`learned\_ai/training/curriculum.py`)

| Stage | Name | Opponent | Exit condition |
| - | - | - | - |
| 1 | sanity | self | episode budget exhausted (10 games) |
| 2 | vs\_random | random agent | rolling win rate ≥ 60 % over 200 games (30 k safety cap) |
| 3 | vs\_heuristic | heuristic difficulty 1 → 10 | ≥ 55 % win rate at each level; graduate when threshold held at difficulty 10 (120 k safety cap) |
| 4 | self\_play | self / pool | episode budget exhausted (70 k games) |
| 5 | human\_finetune | human game data | optional; stub until data exists |


Stage budgets (`stageN\_episodes`) are **safety caps** — stages 2 and 3 exit when the win-rate threshold is held over a rolling 200-game window, not when the budget expires. A model that plateaus is force-advanced by the cap so training never stalls permanently.

**Stage 3 difficulty ramp**: the heuristic opponent starts at `difficulty=1`. Each time the model holds ≥ 55 % win rate over a full 200-game window the difficulty increases by 1 and the window is cleared. A fresh 200-game window must fill at the harder level before the next bump. The model graduates to stage 4 when the threshold is held at `difficulty=10`.

**Temperature resets**: at every stage advance and every difficulty bump within stage 3, temperature is reset to its initial value (default 1.0) so the model enters each new challenge with full exploration headroom. Trained weights are unchanged — only the sampling distribution widens temporarily.

## Replay buffer

`learned\_ai/training/replay\_buffer.py` — a bounded FIFO of `Transition` records `(state, legal\_mask, primary\_index, capture\_index, reward, phase\_id, side\_to\_move, done)`, with `torch.save`/`load` persistence. The trainer mixes fresh self-play experience through it before each update.

## Agents

All three agents in `learned\_ai/agents/` expose `choose\_move(board, \*\*kwargs) -\> move dict`:

- `RandomAgent` — uniform over legal moves.

- `HeuristicAgent` — thin wrapper over the existing `GameAI` minimax engine.

- `LearnedAgent` — NMMNet inference with `argmax` (serving) or temperature `sample` (self-play) modes, full legal-action masking, and a recorded `LearnedDecision` trace for the trainer.

## Checkpoints

Checkpoints embed the model architecture (`model\_config`) alongside the weights, so `LearnedAgent(checkpoint\_path=...)` rebuilds the correctly-sized network without being told the hidden dimensions. `latest.pt` is refreshed on every save and is the default served checkpoint.

To restart training from scratch:

```bash
rm learned_ai/checkpoints/*.pt
rm -f learned_ai/logs/metrics.jsonl
python scripts/train.py --config learned_ai/config/default_config.yaml
```

