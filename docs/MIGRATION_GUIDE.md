# Migration Guide — Switching to the Learned AI

The learned AI is **opt-in**. By default the game uses the existing heuristic
minimax engine and behaves exactly as before. This guide explains how to flip
to the learned engine, A/B test it, and roll back.

## The config flag

Two environment variables control engine selection (read in `main.py`):

```python
import os
AI_ENGINE = os.environ.get("NMM_AI_ENGINE", "heuristic")   # "heuristic" | "learned"
LEARNED_CHECKPOINT = os.environ.get(
    "NMM_LEARNED_CHECKPOINT", "learned_ai/checkpoints/latest.pt"
)
```

- `NMM_AI_ENGINE=heuristic` (default) — unchanged behaviour, no PyTorch needed.
- `NMM_AI_ENGINE=learned` — the AI side is played by `LearnedAgent`, loading the
  checkpoint at `NMM_LEARNED_CHECKPOINT`.

Because `LearnedAgent.choose_move(board)` returns the same move-dict shape as
`GameAI.choose_move(board)`, no calling code changes.

## Switching to learned

```bash
# Make sure a checkpoint exists (train first, or copy one in):
ls learned_ai/checkpoints/latest.pt

NMM_AI_ENGINE=learned python main.py
```

To use a specific checkpoint:

```bash
NMM_AI_ENGINE=learned \
NMM_LEARNED_CHECKPOINT=learned_ai/checkpoints/ckpt-050000.pt \
python main.py
```

If `NMM_AI_ENGINE=learned` but the checkpoint is missing or PyTorch is not
installed, the game prints a warning and falls back to the heuristic engine so
play is never blocked.

## A/B testing

Run head-to-head matches to compare strength before switching the default:

```bash
# Learned vs heuristic, 100 games, alternating colors:
python scripts/evaluate.py \
  --agent1 learned --agent2 heuristic --games 100 \
  --agent1-checkpoint learned_ai/checkpoints/latest.pt

# Focused benchmark with JSON output for tracking over time:
python scripts/benchmark_vs_heuristic.py \
  --checkpoint learned_ai/checkpoints/latest.pt \
  --games 100 --output learned_ai/logs/ab_latest.json
```

Recommended gate before promoting the learned engine to default: it should at
least match the heuristic AI's win rate at the difficulty your players use, and
produce no illegal moves over the full benchmark (guaranteed by masking, but
worth confirming end to end).

For a controlled rollout you can A/B by setting the env var per session/process
(e.g. half your game servers with `NMM_AI_ENGINE=learned`) and comparing
outcome logs.

## Rollback

Rollback is instantaneous and requires no code change — just unset the variable
(or set it back to `heuristic`):

```bash
unset NMM_AI_ENGINE          # back to default heuristic
# or explicitly:
NMM_AI_ENGINE=heuristic python main.py
```

Nothing about the heuristic engine, its data files, or its behaviour is
modified by the learned-AI subsystem, so reverting is fully safe. The learned
code lives entirely under `learned_ai/` plus the small, guarded selection block
in `main.py`.

## What is and isn't touched

- **Added:** everything under `learned_ai/`, the new `scripts/` and `tests/`
  files, and a guarded engine-selection block in `main.py`.
- **Unchanged:** `game/`, the heuristic `ai/` engine, the web app, and all
  existing tests. The default run path is byte-for-byte the same decision logic
  as before.
