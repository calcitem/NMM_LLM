# Strategic Sentinel Overlay Plan for NMM_LLM

## Purpose

This document defines a revised plan for adding a **strategic sentinel** model to the NMM_LLM project. The sentinel is not a replacement for the existing heuristic GameAI, and it is not a new fullgame or endgame database system. Instead, it is a learned overlay that watches games in progress, uses the external 12 GB solved database as ground-truth supervision during training, and learns to recognise **winning and losing turning points across all phases of play**.

The existing project already has:
- a heuristic GameAI engine in `ai/`;
- project-specific endgame and curated fullgame databases used to support that engine;
- self-play tooling and game-record generation scripts;
- a previous learned model stack in `learned_ai/` built around board-state encoders and an MLP backbone.

This plan does **not** replace or rebuild the existing internal databases. It adds a new training and inference path that lets a sentinel model observe board states from human-vs-AI and AI-vs-AI games, consult the external solved database during training, and learn to flag positions where the current heuristic engine is drifting toward a losing trajectory or missing a winning one.

---

## Core Intent

The sentinel overlay should learn the following behaviour:

1. Observe a board state occurring in a real played game, either from self-play or human-vs-AI play.
2. Compare the game trajectory around that state against solved-database truth.
3. Learn whether that board state is a **turning point**, meaning a position where:
   - a move now strongly improves long-term winning chances, or
   - a move now strongly worsens long-term winning chances.
4. At runtime, provide a lightweight advisory signal to the heuristic GameAI such as:
   - “possible mistake,”
   - “strong missed opportunity,” or
   - “high-confidence safe continuation.”

The sentinel therefore learns **pattern recognition over trajectories**, not stand-alone game play.

---

## What This Plan Assumes

### Existing project assets to keep as-is

The following existing systems remain in place and are not to be reinvented:

| Existing asset | Role in current system | What this plan does with it |
|---|---|---|
| Project endgame databases | Assists GameAI in exact/strong late-game decisions | Keep and continue using unchanged |
| Curated fullgame database pruned to user moves | Supports GameAI guidance and historical knowledge | Keep and continue using unchanged |
| Heuristic GameAI | Main playing engine | Remains the primary move selector |
| Self-play and game logging tools | Generate played games and training observations | Reuse as sentinel training data sources |
| `learned_ai/` encoders and model ideas | Existing learned-AI pipeline | Reuse concepts and code patterns where helpful |

### New external asset

| Asset | Role |
|---|---|
| External 12 GB solved database | Provides ground-truth supervision while the sentinel watches game trajectories during training |

The external solved database is used as a **teacher** during training. It is not the same thing as the project’s own internal endgame/fullgame databases, and it should not be merged into them.

---

## Revised System Design

## High-level architecture

The new architecture should have four major pieces:

1. **Game observation layer** — reads board states from human-AI and AI-AI games.
2. **Solved-database supervision layer** — queries the external 12 GB solved database during training to evaluate the strategic quality of observed positions and moves.
3. **Sentinel training pipeline** — turns observed sequences into labelled examples of turning points, near-misses, safe continuations, and mistakes.
4. **Runtime advisory layer** — supplies a lightweight score or warning to GameAI during move selection.

### Training-time flow

```text
human/AI game records or self-play logs
        ↓
extract sequence of board states + chosen moves
        ↓
for each state, query external solved database where possible
        ↓
compute trajectory-quality labels and turning-point labels
        ↓
train sentinel model on board-state + move-context examples
        ↓
save sentinel model weights
```

### Runtime flow

```text
GameAI generates candidate move(s)
        ↓
sentinel evaluates current state and/or candidate move context
        ↓
sentinel emits advisory signal
        ↓
heuristic engine keeps control, but may adjust ranking or trigger caution logic
```

---

## Key Change From the Previous Draft

The earlier draft leaned too heavily on extracting training data directly from the solved database itself. That is **not** the intended plan.

The corrected plan is:
- training examples come primarily from **played games** observed by the system;
- the external 12 GB solved database is used during training to **score and label those observations**;
- the sentinel must learn patterns that apply to **all phases of the game**, not just endgame tablebase states;
- the project’s existing internal databases remain intact and continue serving the heuristic engine separately.

That means the sentinel pipeline is fundamentally **trajectory-supervised**, not database-first.

---

## Training Objective

## What the sentinel should learn

The sentinel should learn to detect strategic board patterns such as:
- transitions from balanced positions into structurally losing lines;
- moves that sacrifice long-term mobility for short-term material or mill gain;
- missed setup moves that create future double threats;
- quiet defensive moves that prevent a forced deterioration;
- board patterns that repeatedly precede wins or losses in real play.

Because the training source is played games, the model should learn from **how positions evolve**, not just from isolated board values.

## Recommended prediction targets

Use a multi-head output rather than a single raw score.

| Output head | Meaning | Runtime use |
|---|---|---|
| `mistake_risk` | Probability current candidate move leads into a worse trajectory | Penalise risky moves |
| `opportunity_score` | Probability a stronger move is being missed | Promote alternatives or trigger re-check |
| `trajectory_value_delta` | Estimated strategic shift caused by current choice | Small ranking adjustment |
| `turning_point_confidence` | Confidence that this position is strategically critical | Gate whether to intervene |

This is better than a single scalar because the sentinel is acting as an advisor, not a replacement evaluator.

---

## How the External 12 GB Solved Database Should Be Used

## Role of the solved database

The external solved database should be used during training as a **ground-truth judge** for observed states and move sequences.

It should answer questions like:
- If the played move from this state is followed forward, does the trajectory move toward a win, draw, or loss?
- Was there a better move available at this point?
- Is this state near a strategically decisive shift?
- How large was the quality drop between the move chosen and a stronger continuation?

## Important design rule

The sentinel is being trained for **all phases**, but the solved database may only be directly queryable or exact for subsets of the game space. Because of that, the training pipeline should support **partial supervision**:

1. **Direct solved supervision** when a state or successor trajectory can be mapped/query-checked in the external database.
2. **Delayed supervision** when later positions in the same observed game do become queryable.
3. **Proxy supervision** when exact lookup is unavailable early in the game, using downstream solved outcomes plus replay analysis.

This lets the sentinel learn opening and midgame turning points indirectly from trajectories that later enter solved territory.

## Practical supervision strategy

For each observed game:

1. Record the full state sequence.
2. For each move index, preserve:
   - board state before move,
   - chosen move,
   - candidate alternatives if available,
   - resulting next state.
3. When the game later reaches positions that the external solved database can classify reliably, propagate that information backward along the trajectory.
4. Mark earlier states with labels such as:
   - “this move preserved a winning trajectory,”
   - “this move began a degrading sequence,”
   - “this move missed a later-proven winning route,”
   - “this state was strategically neutral.”

This backward credit assignment is the core of the training design.

---

## Training Data Sources

## Primary sources

| Source | Why it matters | Use in sentinel training |
|---|---|---|
| Human-vs-AI games | Captures realistic human mistakes and AI responses | Teaches practical turning points users actually create |
| AI-vs-AI self-play games | Generates large volumes of trajectories | Gives broad coverage and repeatable experiments |
| Existing historical game logs | Reuses prior project data | Bootstraps the dataset quickly |

## Secondary sources

| Source | Use |
|---|---|
| Heuristic candidate move lists | Helps compare chosen move vs alternatives |
| Existing project databases | Can provide context features, but not as the main supervisory truth |
| LLM/trajectory annotations already in project | Useful as weak auxiliary labels, not primary truth |

---

## Data Representation

## Board-state representation

The current learned stack already encodes board state into a fixed float vector in `learned_ai/models/state_encoder.py`, with 24 positions encoded as 3-way occupancy plus side-to-move, phase, piece counts, and mill counts. That existing 84-float representation is a strong starting point and should be reused or extended rather than replaced.[cite:42]

### Recommended feature groups

| Feature group | Source | Keep / extend |
|---|---|---|
| 24-position occupancy encoding | Existing state encoder | Keep |
| Side to move | Existing state encoder | Keep |
| Phase indicators | Existing state encoder | Keep |
| Pieces placed / on board | Existing state encoder | Keep |
| Mill counts | Existing state encoder | Keep |
| Candidate move encoding | Add new | Extend |
| Heuristic score before move | Add new | Extend |
| Rank among candidate moves | Add new | Extend |
| Trajectory context window | Add new | Extend |
| Solved-supervision label fields | Training only | Add |

## New context features to add

The sentinel needs more than a static board snapshot. Add context channels such as:
- top-N heuristic move scores at the current ply;
- chosen move rank among candidates;
- whether the move closes or opens a mill;
- whether the move reduces mobility;
- recent trajectory trend, for example deterioration over the last 2–4 plies;
- game source type: human-vs-AI or AI-vs-AI.

This turns the training example into **state + decision context**, which is more appropriate for a sentinel.

---

## Recommended Model Direction

The existing `learned_ai/models/backbone.py` uses a small shared MLP with phase-specific heads and a value head, designed around the existing encoded state vector.[cite:44] That pattern is a better fit for the sentinel than a full end-to-end new architecture because it already matches the project’s learned-AI structure.[cite:44]

## Recommendation

Use a **compact multi-head MLP sentinel** first.

### Why this is the best first version

| Option | Recommendation | Reason |
|---|---|---|
| Reuse/extend existing MLP approach | **Yes** | Fastest integration with current codebase |
| GNN on 24-node graph | Maybe later | Useful research direction, but adds complexity now |
| CNN over board image | No | Board topology is graph-structured, not grid-natural |

### Proposed model shape

- Shared trunk fed by extended state/context features.
- Separate heads for mistake risk, opportunity score, and turning-point confidence.
- Optional temporal variant later that consumes short state windows.

### Why not jump straight to a GNN

A GNN may eventually help because Nine Men’s Morris is naturally a graph. But the fastest path to a useful sentinel is to leverage the existing fixed-vector pipeline and MLP design already present in `learned_ai/models/`.[cite:41][cite:42][cite:44]

---

## Label Construction Strategy

## Core principle

Labels should be generated from **observed game trajectories**, then corrected or enriched using solved-database truth.

## Recommended label types

| Label type | Description |
|---|---|
| `safe_continuation` | Played move preserved a strong trajectory |
| `mistake_start` | Played move began a deterioration relative to stronger alternatives |
| `missed_opportunity` | A stronger move existed but was not chosen |
| `critical_turning_point` | Small choice difference caused large downstream outcome change |
| `neutral_state` | No strong evidence the state is strategically decisive |

## Backward labelling logic

For each game trajectory:

1. Walk forward through the recorded game.
2. Identify later points where the solved database can confidently classify the state or continuation.
3. Propagate the supervision backward to earlier decisions that caused or prevented the later outcome.
4. Assign stronger weights to positions close to the decisive shift, but still retain earlier contributor states.

### Suggested weighting

| Distance from confirmed turning point | Training weight |
|---|---|
| Same ply | 1.0 |
| 1 move earlier | 0.8 |
| 2 moves earlier | 0.6 |
| 3 moves earlier | 0.4 |
| 4+ moves earlier | 0.2 |

This allows the model to learn early warning patterns rather than only late exact-tablebase states.

---

## Files To Read and Interface With

The following files and directories are relevant to this revised plan.

| Path | Current role | Sentinel relevance |
|---|---|---|
| `main.py` | Main application entry point | Load sentinel model and wire training/eval commands |
| `ai/` | Heuristic engine and support systems | Primary runtime integration point |
| `learned_ai/models/state_encoder.py` | Encodes board states into 84-float vectors | Reuse/extend for sentinel features [cite:42] |
| `learned_ai/models/backbone.py` | Existing MLP backbone with heads | Reuse architectural pattern [cite:44] |
| `learned_ai/models/action_encoder.py` | Action encoding support | Candidate move/context encoding reference [cite:41] |
| `scripts/train.py` | Existing learned-model training script | Reference for training loop structure [cite:43] |
| `scripts/run_self_play.py` | Existing self-play generation script | Key source of sentinel observations [cite:43] |
| `scripts/evaluate.py` | Existing evaluation script | Reference for sentinel evaluation plumbing [cite:43] |
| `scripts/human_vs_learned.py` | Human-vs-AI interaction script | Source of real play logs [cite:43] |
| `ai/endgame_solved_db.py` | Project solved DB interface for current engine use | Keep separate from new external DB teacher unless abstraction is reused |
| `trajectory_db_redesign_plan.md` | Parallel redesign plan for trajectory DB | Coordinate carefully, but do not collapse the two projects |

---

## Files To Create or Modify

## New files

| File | Purpose |
|---|---|
| `learned_ai/sentinel_dataset.py` | Build trajectory-supervised training samples from watched games |
| `learned_ai/sentinel_labels.py` | Convert solved-database judgments into training labels |
| `learned_ai/sentinel_model.py` | Sentinel network definition, likely reusing MLP pattern |
| `learned_ai/sentinel_infer.py` | Lightweight runtime inference wrapper |
| `scripts/train_sentinel.py` | Train the sentinel on watched and labelled trajectories |
| `scripts/evaluate_sentinel.py` | Evaluate accuracy, intervention quality, and confidence calibration |
| `scripts/replay_watch_games.py` | Replay game logs and attach solved-database supervision |
| `scripts/export_sentinel_dataset.py` | Persist processed training set for repeatable runs |

## Existing files likely to modify

| File | Why modify it |
|---|---|
| `scripts/run_self_play.py` | Ensure rich logging of states, candidate moves, and outcomes for sentinel training [cite:43] |
| `scripts/human_vs_learned.py` | Capture state/action logs in a sentinel-friendly format [cite:43] |
| `main.py` | Add sentinel loading/configuration hooks |
| `ai/game_ai.py` | Integrate sentinel advisory score into heuristic move selection |
| `ai/coordinator.py` or equivalent runtime orchestration file | Pass sentinel handle into active AI instances |
| `learned_ai/models/state_encoder.py` | Extend feature generation if current 84-float vector is insufficient [cite:42] |

---

## Detailed Pipeline

## Phase 1 — Logging and observation

Goal: ensure every watched game produces rich enough data for sentinel supervision.

### Required log contents per ply

- board state before move;
- side to move;
- move chosen;
- candidate move list if available;
- heuristic scores for top candidates if available;
- resulting board state;
- eventual final game outcome;
- metadata indicating human-vs-AI or AI-vs-AI source.

### Implementation note

`scripts/run_self_play.py` already exists and should be extended rather than duplicated.[cite:43]

## Phase 2 — Solved-database teacher interface

Goal: build a read-only adapter for the external 12 GB solved database.

### Design requirements

- Must be isolated from the project’s own endgame/fullgame databases.
- Must support batch querying during replay/labelling.
- Must expose a simple API such as:

```python
query_state(board_state) -> optional solved assessment
query_move_quality(board_state, move) -> optional score or class
query_trajectory(states) -> supervision packet
```

### Important constraint

Not every early-game state may be directly solvable by the external DB. The adapter therefore needs to return:
- exact supervision when available;
- unknown/unavailable otherwise.

The labelling layer must handle this gracefully.

## Phase 3 — Label generation from watched trajectories

Goal: convert replayed games into supervised examples.

### Process

1. Replay a complete watched game.
2. At each ply, collect the board state and move context.
3. Query the external solved DB for the current state and/or successor states when possible.
4. If exact judgement is unavailable now, continue forward and attach supervision later when the game reaches solvable territory.
5. Propagate the later judgement backward to earlier relevant states.
6. Emit labelled examples for sentinel training.

### Output example

```text
state_t
move_t
candidate_context_t
label = mistake_start
turning_point_confidence = 0.82
value_delta = -0.64
supervision_source = external_solved_db_backward_propagated
```

## Phase 4 — Sentinel training

Goal: train the sentinel model on the generated dataset.

### Recommended first training setup

- Input: extended board-state vector + move context features.
- Model: compact multi-head MLP.
- Losses:
  - binary cross-entropy for mistake risk,
  - regression loss for value delta,
  - binary or multiclass loss for turning-point classification.
- Class balancing: essential, because neutral states will dominate.

### Training curriculum

1. Start with states that have strong solved-database supervision.
2. Add backward-propagated earlier states.
3. Add noisier trajectory-derived examples later with lower sample weight.

## Phase 5 — Runtime integration

Goal: let GameAI consult the sentinel without giving up control.

### Preferred runtime behaviour

The heuristic engine still generates and ranks moves. The sentinel then provides one of these advisory effects:

- small penalty to a risky move;
- small bonus to a promising alternative;
- request for deeper reconsideration if confidence is high;
- no action if confidence is low.

### Integration rule

The sentinel should not hard-override exact database moves already provided by the project’s own proven DB systems. It should operate mainly in heuristic-driven parts of play.

---

## Recommended Runtime Integration Strategy

## Where to call the sentinel

Integrate the sentinel in the move-selection path after heuristic candidate scoring is available but before final move commitment.

### Runtime steps

1. GameAI generates candidate moves and heuristic scores.
2. Build sentinel input from:
   - board state,
   - chosen candidate,
   - top-N alternatives,
   - optional recent trajectory context.
3. Run sentinel inference.
4. Apply a bounded adjustment to heuristic ranking or trigger deeper review.
5. Keep final authority with the heuristic/DB-backed engine.

## Safe intervention modes

| Mode | Description | Use first? |
|---|---|---|
| Advisory only | Log warning/confidence, no move change | Yes, first deployment |
| Score adjustment | Add/subtract small ranking delta | Yes, after validation |
| Reconsideration trigger | Force deeper search on suspicious state | Yes, selectively |
| Hard override | Replace heuristic move directly | No, not initially |

The first production version should probably be **advisory only** or **small score adjustment only**.

---

## Evaluation Criteria

## Offline metrics

| Metric | Meaning |
|---|---|
| Turning-point detection precision | How often flagged turning points are real |
| Turning-point detection recall | How many real turning points are caught |
| Mistake-risk calibration | Whether high-risk outputs correspond to real deteriorations |
| Opportunity detection quality | Whether high opportunity scores align with better missed moves |
| Early-warning usefulness | Whether the model flags states before the exact solved boundary |

## Online metrics

| Metric | Meaning |
|---|---|
| Win-rate improvement vs baseline heuristic AI | Main practical measure |
| Reduction in avoidable strategic mistakes | Counts sentinel-preventable losses |
| Number of false alarms | Too many harms usability and trust |
| Average extra compute per move | Must remain small |
| Human usefulness in play | Whether warnings are understandable and timely |

## Deployment ladder

1. Advisory-only mode.
2. Score-adjustment mode.
3. Limited reconsideration mode.
4. Only after strong evidence, consider stronger intervention.

---

## Risks and Dependencies

## Main risks

| Risk | Why it matters | Mitigation |
|---|---|---|
| Early-game states may not map cleanly to solved DB truth | Could leave sparse supervision | Use delayed and backward supervision |
| Logged games may not contain enough candidate-move context | Weakens missed-opportunity labels | Extend self-play and human-play logging |
| Neutral positions will dominate | Model may learn to predict “nothing happening” | Use class weighting and sampling |
| Sentinel may interfere too aggressively | Can reduce GameAI strength | Start in advisory-only mode |
| Confusion with existing project DB systems | Could create duplicated logic | Keep external teacher DB clearly separate |

## Dependency on existing project work

The sentinel project should coordinate with, but remain distinct from, the trajectory database redesign. If the trajectory redesign improves state-level blame or reward annotations, those signals can later become auxiliary training features. But the sentinel plan should not depend on that redesign being finished first.

---

## Practical Build Order

## Stage 1

- Extend self-play and human-play logging.
- Define standard watched-game record format.
- Create solved-database teacher adapter.

## Stage 2

- Build replay/labelling pipeline.
- Generate a first supervised dataset from existing logs.
- Start with positions that can be labelled confidently.

## Stage 3

- Train compact multi-head MLP sentinel.
- Evaluate offline on held-out watched games.

## Stage 4

- Integrate advisory-only sentinel into GameAI.
- Benchmark AI-vs-AI and human-AI sessions.

## Stage 5

- Add bounded score adjustments.
- Retest and tune thresholds.

---

## Claude Implementation Guidance

When turning this plan into code, follow these rules:

1. Do **not** rebuild or replace the project’s internal endgame/fullgame database systems.
2. Keep the external 12 GB solved database as a separate teacher/supervision source.
3. Treat watched game trajectories as the primary training examples.
4. Train the sentinel for **all phases** by using delayed/backward supervision from later solvable states.
5. Prefer reusing existing learned-AI encoders and MLP structure before introducing new architecture complexity.[cite:41][cite:42][cite:44]
6. Make the first runtime integration conservative: advisory mode or bounded ranking adjustments only.
7. Keep the heuristic engine as the main player.

---

## Recommended Deliverables for the Next Coding Step

The next implementation pass should aim to produce:

| Deliverable | Description |
|---|---|
| Sentinel training data schema | Formal JSONL or parquet structure for watched-game examples |
| External solved-DB adapter | Read-only query interface |
| Replay labelling script | Converts watched games into supervised examples |
| First sentinel model definition | Compact multi-head MLP |
| Training script | Reproducible training entry point |
| Evaluation script | Offline metrics and benchmark harness |
| Conservative GameAI hook | Advisory-only sentinel inference path |

---

## Stage 6

- Audit the old `learned_ai/` end-to-end player code path once the sentinel overlay is working and benchmarked.
- Remove unused training scripts, obsolete model files, dead configuration, and any Python entry points that only supported the failed end-to-end learned-player approach.
- Keep only the learned-AI components that are still reused by the sentinel, such as encoders, feature utilities, or shared model helpers.
- Update imports, requirements, docs, startup paths, and test scripts so the repository reflects the new architecture cleanly.
- Archive or delete superseded artifacts, checkpoints, and notes from the failed learned-player plan after confirming they are no longer referenced.

## Cleanup Stage Details

This final stage should happen **after** the sentinel overlay is implemented, validated, and accepted as the replacement learned-AI direction. The goal is to reduce confusion in the repository, remove dead paths, and make the codebase reflect the actual design going forward.

### Cleanup goals

| Cleanup target | Action | Notes |
|---|---|---|
| Old end-to-end learned player code | Remove or archive | Only after confirming it is not needed for sentinel training or inference |
| Unused training scripts | Remove | Especially scripts that exist only for the failed learned-player objective |
| Obsolete model checkpoints | Archive or delete | Keep only sentinel-relevant models and any deliberately retained historical artifacts |
| Unused requirements/dependencies | Prune | Remove packages only needed by abandoned training paths |
| Docs and plans for failed approach | Mark obsolete or archive | Avoid future confusion about which learned-AI direction is current |
| Runtime hooks pointing to dead code | Remove | Ensure `main.py`, scripts, and tests only reference supported paths |

### Suggested cleanup procedure

1. Make a full inventory of everything under `learned_ai/`, related scripts, saved models, and requirements entries.
2. Mark each item as one of: `keep`, `reuse for sentinel`, `archive`, or `delete`.
3. Remove only items that are confirmed unused by the new sentinel pipeline.
4. Update documentation so the repository clearly describes the heuristic engine plus sentinel overlay architecture.
5. Run smoke tests and benchmark scripts again after cleanup to confirm nothing important was removed.

### Rule for this stage

Do not perform cleanup early. Cleanup is the **last stage**, after the sentinel system is training correctly, integrating correctly, and outperforming or meaningfully assisting the baseline heuristic engine.

## Final Position

The correct design is a **trajectory-watching sentinel** trained from played games and supervised by the external 12 GB solved database. The project’s own endgame and curated fullgame databases remain in their current supporting roles. The sentinel should learn to recognise strategic turning points across the full game by observing how earlier board states lead into later solved outcomes, then provide a lightweight advisory signal back to the heuristic GameAI.[cite:42][cite:44]

---

## Implementation Status (as of 2026-06-06)

### Done

| Stage | Component | Status |
|---|---|---|
| Stage 1 | `learned_ai/sentinel/` package scaffold | **Done** — `feature_builder.py`, `labels.py`, `dataset.py`, `model.py`, `infer.py`, `config.py` |
| Stage 1 | `learned_ai/sentinel/db_teacher.py` — ExternalSolvedDB graceful stub | **Done** — probes path, delegates to MalomDB |
| Stage 1 | `ai/malom_db.py` — MalomDB .sec2 binary reader | **Done** (hash function pending — see below) |
| Stage 1 | `configs/sentinel_default.yaml` — path set to `/mnt/windows/NMM_DB/strong`, enabled=true | **Done** |
| Stage 1 | `scripts/train_sentinel.py`, `evaluate_sentinel.py`, `replay_watch_games.py`, `export_sentinel_dataset.py` | **Done** |
| Stage 1 | `ai/game_ai.py` — advisory integration hook | **Done** — sentinel advisor consulted in choose_move |
| Stage 1 | `main.py` — `--sentinel-checkpoint` flag | **Done** |
| Stage 1 | `tests/test_malom_db.py` — 52 sentinel tests pass | **Done** |

### MalomDB status (`ai/malom_db.py`)

Fully implemented **except the hash function**:

- `.sec2` header parsing (version=2, esize=3, f2off=12) ✓
- `std.secval` parsing (virt_win=299, virt_loss=-299, per-sector values) ✓
- em_set overflow side-table reading ✓
- Entry decoding: Win / Loss / Draw / Count / Sym-redirect all handled ✓
- Sym-redirect resolution (19.3% of entries in 3_3_0_0 are sym-redirects) ✓
- Bitboard coordinate mapping (24-position MALOM_POSITIONS) ✓
- `board_to_wbf(board)` conversion ✓
- Sector file discovery and mmap caching ✓
- `MalomDB.is_available()` returns True for `/mnt/windows/NMM_DB/strong/` ✓
- `MalomDB.query(board)` — returns None while hash is a stub ✓

**Blocker: hash function** (`_board_to_index`).

Sector `std_3_3_0_0.sec2` has 210,140 entries; naive combinatorial ranking gives 2,691,920. D4 board symmetry plus sym-redirect for non-canonical positions is the mechanism, but exact implementation requires the Malom C++ source (`hash.cpp`).

**To unblock:**
1. Download Malom source from http://compalg.inf.elte.hu/~ggevay/mills/index.php (GPL-3)
2. Port `Hash::f_lookup` + `Hash::hash(board)` to Python
3. OR: when the large full DB torrent arrives, check if `preCalculatedVars.dat` contains precomputed lookup tables usable directly

### Next steps (priority order)

| Priority | Task |
|---|---|
| **P0** | Implement `_board_to_index` — port hash from Malom C++ source |
| **P0** | Once hash works: run `scripts/replay_watch_games.py` over `data/games/` for first labelled training set |
| **P1** | `python scripts/train_sentinel.py --config configs/sentinel_default.yaml --game-dir data/games` |
| **P1** | `python scripts/evaluate_sentinel.py` offline eval |
| **P2** | When large DB arrives: update config path to full `.sec2` directory |
| **P2** | Wire MalomDB into `ai/coordinator.py` for runtime advisory (advisory-only first) |
| **P3** | Score-adjustment mode after advisory-only is validated |

### Database path status

| Path | Contents | Status |
|---|---|---|
| `/mnt/windows/NMM_DB/strong/` | Partial `.sec2` DB (~200 sectors) | **Active** — configured in sentinel_default.yaml |
| `/mnt/windows/NMM_DB/Entire DB/` | 11 GB monolithic files — NOT `.sec2` | Undecodable — ignore until source identified |
| Torrent (downloading) | Full 77.8 GB `.sec2` DB | Pending — will replace `strong/` once complete |
