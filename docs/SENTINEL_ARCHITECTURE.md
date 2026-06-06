# Strategic Sentinel Overlay — Architecture

> Status: implementation in progress on branch `feat/sentinel-overlay`.
> The sentinel is **advisory only** in its first deployment. It does NOT
> replace the heuristic `GameAI`, and the game runs identically when no
> sentinel checkpoint is loaded.

## 1. Purpose

The sentinel is a learned overlay that:

1. Watches game trajectories from played games (`data/games/*.jsonl`).
2. Uses an external 12 GB solved database as a ground-truth teacher
   **during training only** (the DB lives on the user's PC, not in the repo;
   the adapter is gracefully unavailable when the DB is missing).
3. Learns to detect strategic turning points across all game phases.
4. Provides lightweight advisory signals to the existing heuristic `GameAI`
   at runtime.

It learns **pattern recognition over trajectories**, not stand-alone play.

## 2. Component map

```
learned_ai/sentinel/
  config.py          SentinelConfig dataclass + load_config()
  feature_builder.py 120-float feature vector (84 base via state_encoder + 36 context)
  db_teacher.py      ExternalSolvedDB — read-only teacher adapter (graceful stub)
  labels.py          backward label propagation -> LabelledExample
  dataset.py         SentinelDataset — trajectory-supervised samples + save/load
  model.py           SentinelNet — shared trunk + 4 heads -> SentinelOutput
  infer.py           SentinelAdvisor.advise() -> SentinelAdvice (runtime, fast)

scripts/
  replay_watch_games.py     replay logs + attach supervision -> JSONL
  export_sentinel_dataset.py persist processed training set
  train_sentinel.py          training loop (weighted multi-task loss)
  evaluate_sentinel.py       offline metrics

configs/
  sentinel_default.yaml / sentinel_smoke.yaml
```

## 3. Data flow

### Training time

```
data/games/*.jsonl
   -> parse game record (session_id, winner, human_color, moves[])
   -> replay BoardState by applying each move (parse board_fen_before)
   -> per ply: build move_context dict (candidates, mill flags, trajectory trend)
   -> ExternalSolvedDB.query_* where available (else None)
   -> backward_label_trajectory(): direct / backward-propagated / outcome-proxy labels
   -> SentinelDataset of (feature_tensor[120], label_dict)
   -> SentinelNet trained with weighted multi-task loss
   -> checkpoint (.pt)
```

### Runtime

```
GameAI.choose_move() builds heuristic candidates + scores
   -> _build_sentinel_context(board, candidates, scores)
   -> SentinelAdvisor.advise(board, context) -> SentinelAdvice (single CPU forward pass)
   -> advisory mode: log warning/flags, NO move change
   -> (later) score_adjust / reconsider modes apply bounded effects
All sentinel calls are wrapped in try/except; failures never break the game loop.
```

## 4. Feature vector (120 floats)

| Range      | Size | Source                                                          |
|------------|------|-----------------------------------------------------------------|
| `[0:84)`   | 84   | `learned_ai.models.state_encoder.encode_state()` (REUSED)       |
| `[84:120)` | 36   | context channels (see below)                                    |

Context channels (36):

| Feature                    | Size | Notes |
|----------------------------|------|-------|
| top-5 heuristic scores     | 5    | normalised to [0,1] via sigmoid-like squashing, 0-padded |
| top-5 move-type one-hots   | 20   | 4-way (place/move/fly/capture) per candidate, 0-padded |
| chosen move rank           | 1    | rank / max(n_candidates-1, 1), 0 if single candidate |
| closes_mill                | 1    | bool |
| opens_mill_threat          | 1    | bool |
| reduces_own_mobility       | 1    | bool |
| trajectory score trend     | 4    | last 4 heuristic scores, normalised, 0-padded |
| game_source_is_human       | 1    | 1.0 if human-vs-AI else 0.0 |
| n_candidates_norm          | 1    | n_candidates / 30.0, clipped to 1.0 |
| reserved/padding           | 1    | 0.0 (keeps the context block exactly 36 wide) |

Total context = 5 + 20 + 1 + 1 + 1 + 1 + 4 + 1 + 1 + 1 = **36**.

## 5. Model (`SentinelNet`)

Reuses the small-MLP `_mlp(...)` pattern from `learned_ai/models/backbone.py`.

```
Input: 120-float vector
Shared trunk: Linear(120, h0) -> ReLU [-> Dropout] -> ... -> Linear(h_{k-1}, h_k) -> ReLU
              (hidden_dims default [256,128,64]; smoke [64,32])
Heads (each from trunk output dim T):
  mistake_risk_head:           Linear(T,32) -> ReLU -> Linear(32,1) -> Sigmoid
  opportunity_score_head:      Linear(T,32) -> ReLU -> Linear(32,1) -> Sigmoid
  trajectory_value_delta_head: Linear(T,32) -> ReLU -> Linear(32,1) -> Tanh   ([-1,1])
  turning_point_head:          Linear(T,32) -> ReLU -> Linear(32,1) -> Sigmoid
```

`forward()` returns a `SentinelOutput` dataclass of four tensors.

Loss (per-sample weighted):

```
L = w_mistake * BCE(mistake_risk, target)
  + w_opp     * BCE(opportunity_score, target)
  + w_delta   * MSE(trajectory_value_delta, target)
  + w_tp      * BCE(turning_point_confidence, target)
```

## 6. Label types

| Label                     | Meaning |
|---------------------------|---------|
| `safe_continuation`       | played move preserved a strong trajectory |
| `mistake_start`           | played move began deterioration vs better alternatives |
| `missed_opportunity`      | a stronger move existed but was not chosen |
| `critical_turning_point`  | small choice caused a large downstream outcome change |
| `neutral_state`           | no strong evidence of strategic decisiveness |

Backward decay weights by distance from a confirmed turning point:
`[1.0, 0.8, 0.6, 0.4, 0.2]` (distance 4+ reuses 0.2).

Supervision sources, in priority order:
`direct_solved` > `backward_propagated` > `trajectory_outcome` > `weak_proxy`.

## 7. External DB teacher

`ExternalSolvedDB` is a read-only adapter for the external 12 GB solved DB
(`database.dat` + `preCalculatedVars.dat`). It is **separate** from the
project's internal `ai/endgame_solved_db.py` and must never be merged with it.

Because the external DB binary format is undocumented in the repo, the adapter:

1. Attempts to read `preCalculatedVars.dat` to infer format metadata.
2. If the format cannot be determined, returns `None` for all queries and logs
   a clear one-time warning.
3. **Never crashes** — missing/empty/unreadable paths are non-fatal.

See the TODO block in `db_teacher.py` for the format-decode plan once the real
DB layout is known.

## 8. Safety guarantees

- `GameAI` works identically with no sentinel loaded (`self.sentinel is None`).
- Every sentinel runtime call is wrapped in `try/except`; advisory failures are
  logged at debug level and the heuristic move is used unchanged.
- The external DB adapter is non-fatal when the DB is unavailable.
- `ai/endgame_solved_db.py` is never modified.

## 9. Runtime integration

The overlay attaches to the heuristic engine through three `GameAI` members:

- `set_sentinel(advisor, mode="advisory")` — attach a loaded `SentinelAdvisor`.
- `_build_sentinel_context(board, moves)` — package the finalized candidate set
  into the dict `feature_builder.build_features()` consumes.
- `_consult_sentinel(board, moves)` — run the advisory pass; fully `try/except`
  guarded.

`choose_move()` calls `_consult_sentinel()` once, after all candidate filtering
(pins, bans) and before the search/return paths. In **advisory** mode this only
records `last_sentinel_advice` and logs a warning when a turning point is
flagged — it never changes the move. The `score_adjust` and `reconsider` modes
are reserved for later work; advisory is the only mode wired today.

Loading at runtime (advisory only):

```
python main.py --sentinel-checkpoint learned_ai/sentinel/checkpoints/best.pt
```

If the checkpoint is missing or fails to load, the game prints a notice and
continues with no overlay — play is unchanged.
