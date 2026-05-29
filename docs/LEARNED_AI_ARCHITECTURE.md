# Learned AI Architecture

This document describes the self-learning AI subsystem added under `learned_ai/`.
It is a PyTorch policy/value network trained by self-play reinforcement
learning. It plugs into the existing Nine Men's Morris engine through the same
`choose_move(board)` contract used by the heuristic minimax AI (see
`docs/AI_INTERFACE_MAPPING.md`).

## Design goals

1. **Never duplicate game rules.** Legal-move enumeration always defers to
   `game.rules.get_all_legal_moves`. The learned policy only *ranks* moves the
   engine has already declared legal.
2. **Always mask illegal actions.** Illegal action logits are set to `-1e9`
   before any softmax, so they receive exactly zero probability.
3. **No oracles in the policy.** No tablebases, perfect-play databases, or
   handcrafted move heuristics enter the network. It learns purely from
   self-play reward.
4. **Drop-in replacement.** The agent is selectable via an environment variable
   and exposes the heuristic AI's interface, so the existing game stays fully
   playable.

## Phase model

NMM naturally divides into phases. The encoder classifies the side-to-move's
position into one of five phases (`learned_ai/models/state_encoder.py::detect_phase`):

| ID | Name              | Trigger (for the side to move)                          |
|----|-------------------|---------------------------------------------------------|
| 0  | opening_placement | still placing AND `pieces_placed < 4`                   |
| 1  | full_placement    | still placing AND `4 <= pieces_placed <= 9`             |
| 2  | midgame           | both placed AND stm has `>= 5` on board AND opp has `> 3` |
| 3  | endgame           | both placed AND stm has 4 on board, or opp is down to 3, not yet flying |
| 4  | flying            | placed all 9 AND exactly 3 pieces remain (can fly)      |

This 5-way split is layered on top of the engine's coarse `place / move / fly`
classification from `game.rules.get_game_phase` — it never contradicts it.

## State encoding (84 floats)

`encode_state(board) -> torch.Tensor` of shape `(84,)`:

| Slice    | Size | Meaning                                                    |
|----------|------|------------------------------------------------------------|
| `[0:72)` | 72   | 24 positions × 3-way one-hot (empty / white / black)       |
| `[72]`   | 1    | side to move (0.0 = white, 1.0 = black)                    |
| `[73:78)`| 5    | phase one-hot (the 5 phases above)                         |
| `[78]`   | 1    | white pieces placed / 9                                    |
| `[79]`   | 1    | black pieces placed / 9                                    |
| `[80]`   | 1    | white pieces on board / 9                                  |
| `[81]`   | 1    | black pieces on board / 9                                  |
| `[82]`   | 1    | white mills formed / 3 (capped at 1.0)                     |
| `[83]`   | 1    | black mills formed / 3 (capped at 1.0)                     |

Board positions are indexed in `game.board.POSITIONS` order, so index `i`
always maps to the same physical square.

## Action encoding (624 actions)

A single unified action space (`learned_ai/models/action_encoder.py`) covers
all phases, so every head emits the same 624-wide logit vector:

| Slice       | Size | Meaning                                              |
|-------------|------|------------------------------------------------------|
| `[0:24)`    | 24   | placement on `POSITIONS[i]`                          |
| `[24:600)`  | 576  | movement / fly: `from = src, to = dst` → `24*src+dst`|
| `[600:624)` | 24   | capture `POSITIONS[i]` (after a mill is closed)      |

Because the existing engine emits *atomic* moves (a mill-forming placement or
movement is bundled with its capture), the agent decides the move in a single
forward pass: it samples the placement/movement index from the primary slice,
and — only when that partial move closes a mill — samples the capture target
from the capture slice. Both slices are independently masked to the legal set.

`get_legal_mask(board)` builds a `(624,)` boolean mask using only the engine's
legal-move list and `board.legal_captures`; the capture slice is left all-False
unless a mill can actually be formed on this turn.

## Network: NMMNet

`learned_ai/models/backbone.py::NMMNet`

```
state (84) ─► Shared backbone MLP  (default 256 → 256 → 128, ReLU [+dropout])
                       │
        ┌──────────────┼───────────────────────────┐
        ▼              ▼                             ▼
  phase head 0   ...  phase head 4            value head
  (128→64→624)        (128→64→624)            (128→64→1)
```

- **Shared backbone:** learns board geometry and mill patterns once, shared
  across all phases.
- **Five phase heads:** a `ModuleDict` keyed by phase name. `forward(state,
  phase_id, legal_mask)` routes through exactly one head per call. Each head
  outputs the full 624-dim action logits, which are then masked.
- **Value head:** a single shared scalar value estimate `V(s)`, used as the
  REINFORCE baseline.

Masking happens inside `forward`: `logits.masked_fill(~legal_mask, -1e9)`.

### Why a shared backbone + heads instead of five separate models?

- NMM phases share board geometry and mill structure; a shared representation
  learns these once instead of five times.
- Fewer parameters to train early, when self-play data is scarce.
- One checkpoint to manage instead of five.
- The design degrades gracefully: if one phase's learning diverges, its head
  can later be split into a standalone model without touching the others,
  because the head boundary is already explicit.

Color is **not** modeled with separate networks — the `side_to_move` bit in the
state vector conditions a single network that learns symmetric play.

## Training algorithm

`learned_ai/training/trainer.py`

Default is **REINFORCE with a value baseline**:

- For each episode, play one self-play game; assign terminal reward `+1` win,
  `-1` loss, `0` draw, from the perspective of the side that made each move
  (`learned_ai/training/self_play.py::assign_rewards`, with optional
  discount-to-end via `gamma`).
- Policy loss: `-(log π(a|s) · advantage)`, where `advantage = return - V(s)`.
- Value loss: MSE between `V(s)` and the realized return.
- An entropy bonus encourages exploration; gradients are clipped at norm 5.
- A **PPO** branch is available via `training.algorithm: ppo` in the config
  (clipped surrogate objective) for later experiments.

Batches are grouped by phase so each head's forward pass runs once per update.

### Curriculum (`learned_ai/training/curriculum.py`)

| Stage | Name              | Opponent          | Goal                              |
|-------|-------------------|-------------------|-----------------------------------|
| 1     | sanity            | self              | encoding/runtime sanity, no crash |
| 2     | vs_random         | random agent      | learn basic legal, winning play   |
| 3     | vs_heuristic      | heuristic minimax | learn against a strong baseline   |
| 4     | self_play         | self / pool       | open-ended improvement            |
| 5     | human_finetune    | human game data   | optional; stub until data exists  |

Stage lengths are configured in YAML (`stageN_episodes`). The controller
advances automatically when a stage's episode budget is exhausted.

## Replay buffer

`learned_ai/training/replay_buffer.py` — a bounded FIFO of `Transition`
records `(state, legal_mask, primary_index, capture_index, reward, phase_id,
side_to_move, done)`, with `torch.save`/`load` persistence. The trainer mixes
fresh self-play experience through it before each update.

## Agents

All three agents in `learned_ai/agents/` expose `choose_move(board, **kwargs)
-> move dict`:

- `RandomAgent` — uniform over legal moves.
- `HeuristicAgent` — thin wrapper over the existing `GameAI` minimax engine.
- `LearnedAgent` — NMMNet inference with `argmax` (serving) or temperature
  `sample` (self-play) modes, full legal-action masking, and a recorded
  `LearnedDecision` trace for the trainer.

## Checkpoints

Checkpoints embed the model architecture (`model_config`) alongside the weights,
so `LearnedAgent(checkpoint_path=...)` rebuilds the correctly-sized network
without being told the hidden dimensions. `latest.pt` is refreshed on every
save and is the default served checkpoint.
