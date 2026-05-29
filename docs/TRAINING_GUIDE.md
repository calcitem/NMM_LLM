# Training Guide — Learned AI

How to install, smoke-test, train, monitor, and resume the learned NMM AI.

## 1. Install dependencies

The learned AI needs PyTorch and a few small utilities, in addition to the
base game requirements.

```bash
pip install -r requirements_learned_ai.txt
```

`requirements_learned_ai.txt` pins:

- `torch>=2.0`
- `numpy`
- `pyyaml`
- `jsonlines`
- `tqdm`

CPU-only is fine for smoke tests and modest training. For a CPU wheel:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## 2. Run the smoke suite

Verifies encoders, model routing, self-play, and checkpoint round-trips:

```bash
python scripts/smoke_test.py
```

All tests should pass. This does not train a useful model — it just proves the
pipeline runs without crashing.

## 3. Run a fast end-to-end training smoke run

```bash
python scripts/train.py --config learned_ai/config/smoke_test_config.yaml
```

This uses a tiny network (64-64-32) and only ~20 episodes. Expected outcome:
the model loses essentially every game (it is untrained) but the run completes
with **0 illegal move attempts** and writes:

- checkpoints to `learned_ai/checkpoints/smoke/`
- metrics to `learned_ai/logs/smoke/metrics.jsonl`
- self-play game logs to `learned_ai/self_play_games/smoke/`

## 4. Full training

```bash
python scripts/train.py --config learned_ai/config/default_config.yaml
```

Key hyperparameters live in `learned_ai/config/default_config.yaml`:

- `training.algorithm`: `reinforce` (default) or `ppo`
- `training.lr`, `training.gamma`, `training.episodes_per_batch`
- `training.temperature` / `temperature_decay` / `min_temperature` — exploration
- `curriculum.stageN_episodes` — how long to spend in each stage
- `model.backbone_hidden`, `model.head_hidden`, `model.dropout`

### Curriculum stages and expected win rates

The curriculum advances automatically as episode budgets are spent.

| Stage | Opponent     | What "good" looks like                                    |
|-------|--------------|-----------------------------------------------------------|
| 1     | self (sanity)| run completes, no crashes, 0 illegal attempts             |
| 2     | random       | win rate climbs toward **70%+** vs the random agent       |
| 3     | heuristic    | win rate gradually rises; early on expect heavy losses    |
| 4     | self-play    | strength improves open-endedly; track vs a fixed baseline |
| 5     | human data   | optional fine-tuning; skipped when no human data exists   |

These are rough targets, not guarantees — they depend on episode counts,
network size, and exploration schedule. Stage 2's 70% threshold is the main
sanity milestone: if the model cannot beat a uniform-random opponent most of
the time, something upstream (rewards, masking, encoding) is wrong.

You can force a starting stage:

```bash
python scripts/train.py --config learned_ai/config/default_config.yaml --stage 3
```

## 5. Monitor training

Metrics are JSON-Lines, one object per policy update:

```bash
tail -f learned_ai/logs/metrics.jsonl
```

Each line includes: episode count, stage name, running win/loss/draw totals,
white vs black win split, per-phase move counts, temperature, and the loss
components (`policy_loss`, `value_loss`, `entropy`, `mean_reward`).

Things to watch:

- `illegal` attempts should always be 0 (masking guarantees this).
- `mean_plies` that collapses to a tiny number can indicate degenerate play.
- `entropy` should start high and decline as the policy sharpens.

## 6. Resume from a checkpoint

```bash
python scripts/train.py \
  --config learned_ai/config/default_config.yaml \
  --resume learned_ai/checkpoints/ckpt-010000.pt
```

Checkpoints embed their model architecture, so you do not need to re-specify
hidden sizes.

## 7. Evaluate

Benchmark the learned AI against the heuristic engine:

```bash
python scripts/benchmark_vs_heuristic.py \
  --checkpoint learned_ai/checkpoints/latest.pt --games 50
```

Arbitrary head-to-head:

```bash
python scripts/evaluate.py --agent1 learned --agent2 random \
  --games 100 --agent1-checkpoint learned_ai/checkpoints/latest.pt
```

Generate self-play archives (e.g. to seed analysis):

```bash
python scripts/run_self_play.py --episodes 100 \
  --checkpoint learned_ai/checkpoints/latest.pt
```

## 8. Play against it

```bash
python scripts/human_vs_learned.py \
  --checkpoint learned_ai/checkpoints/latest.pt --side W
```
