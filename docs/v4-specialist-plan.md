# Specialist AI — V4 Plan (Self-Reflecting Memory Architecture)

## Philosophy

The V4 specialist is not a pure policy network — it is a **learning agent with memory**.
It accumulates experience from every game it plays, reflects on the outcomes, tags winning
patterns, and uses that accumulated knowledge at inference time. Over thousands of games it
builds its own WDL database that gradually substitutes for the Malom oracle (which was used
to bootstrap it during training but is too large to ship to end users).

The three specialists (opening, midgame, endgame) all write to a **single shared SpecialistDB**.
They are trained in parallel from separate scripts and play independently, but at inference they
share both the common coordinator and the shared DB.

---

## What does NOT change from the current setup

- Advancement schedule: 100-game rolling window, P-threshold = 0.70 (advance_stats.py)
- Time budgets: `_specialist_time_budget` (0.5s→20s), `_heuristic_time_budget` (0.1s→14s)
- Base 62-float sentinel features per candidate move (feature_builder.py unchanged)
- Three-specialist routing: open / mid / end phases unchanged
- LR, batch size, reward shaping (sentinel delta + heuristic delta + mill bonus) — unchanged

---

## Feature Layout Change: 5 signals × 12 plies = 60 lookahead floats

**Old v3b (broken):** 3 signals × 20 plies = 60 → `(h_norm, sent_mean, human_norm)`

**V4:** 5 signals × 12 plies = 60 → per ply:

| Index | Signal | Description |
|---|---|---|
| `ply*5 + 0` | `h_norm` | Heuristic eval from learner's perspective, normalised to [0,1] |
| `ply*5 + 1` | `learner_sent` | Mean sentinel quality of learner's legal moves at this ply |
| `ply*5 + 2` | `opp_sent` | Mean sentinel quality of opponent's legal moves at this ply |
| `ply*5 + 3` | `vn_norm` | Value net board evaluation at this ply |
| `ply*5 + 4` | `gap_norm` | Gap net score at this ply (opponent blunder probability) |

`learner_sent` and `opp_sent` are computed separately in `_record_signals`:
- On a learner-turn ply: `learner_sent` = mean over learner's legal moves; `opp_sent` = 0.5
- On an opponent-turn ply: `opp_sent` = mean over opponent's legal moves; `learner_sent` = 0.5

This gives the specialist a **per-ply contrast** between its own options and the
opponent's options. A trap signature reads as: `learner_sent` stays high across plies
while `opp_sent` drops progressively → the specialist learns to prefer moves that produce
this pattern.

Total per candidate: `62 base + 60 lookahead = 122 floats`.

The v3 top-K extras (4 floats: ab_score_norm, ab_rank_norm, human_freq, human_rank) are dropped
because we return to full-legal-moves scoring — there is no alpha-beta pre-ranking step to derive
those values from. All existing checkpoints are **incompatible** and training starts from scratch.

---

## Training Lookahead Depth

- **Normal games (19 of every 20):** `sim_ply_depth = 5`, `ply_depth = 12`
  - Plies 0-4 are simulated; plies 5-11 are filled with neutral (0.5, 0.5, 0.5, 0.5, 0.5)
  - Fast: the specialist sees real data for the first 5 half-moves
- **Deep games (1 in 20):** `sim_ply_depth = 12`, `ply_depth = 12`
  - Full 12-ply simulation; slower but provides ground-truth long-horizon data
  - The mix prevents the specialist from over-fitting to only 5-ply patterns
- **Inference:** always `sim_ply_depth = 12` (as time allows, subject to time budget)

---

## Full Legal Moves

The specialist scores **all legal moves**, not a re-ranked subset from alpha-beta.
Each candidate gets its own 122-float feature vector (62 base + 60 lookahead).
The policy head selects the move with the highest score. This is the v2 design; the
v3 top-5 re-ranking approach is abandoned.

`encode_position_with_lookahead` is used at both training and inference.
`encode_top_k_candidates` is no longer called by the training scripts or the router.

---

## SpecialistDB — Self-Built Experience Database

### Purpose

A **single shared** SQLite database (`data/specialist_db.sqlite`) written to by all three
specialists. Each specialist phase-tags its winning lines and preferred plays, but position
statistics are shared — a position encountered during midgame training and again during
endgame training accumulates from both. No cap on size; even 100 000 games occupy less than 1 GB.

Position keys use **D4 (dihedral-8) symmetry** via `ai/board_symmetry.canonical_board_str()`:
all 8 rotationally/reflectionally equivalent positions share the same key, giving up to 8×
effective data efficiency. This matches how the endgame databases are keyed.

SQLite WAL mode is enabled so all three training processes can write concurrently without locking.

### Schema

```sql
-- Per-position outcome statistics
CREATE TABLE positions (
    pos_hash     TEXT    PRIMARY KEY,
    wins         INTEGER NOT NULL DEFAULT 0,
    draws        INTEGER NOT NULL DEFAULT 0,
    losses       INTEGER NOT NULL DEFAULT 0,
    malom_label  TEXT    DEFAULT NULL,  -- 'W'/'D'/'L' if Malom validated this pos
    last_seen    TEXT    NOT NULL       -- ISO-8601 timestamp
);

-- Tagged winning move sequences
CREATE TABLE winning_lines (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    move_seq     TEXT    NOT NULL,  -- JSON array of UCI-style move strings
    phase        TEXT    NOT NULL,  -- 'open' / 'mid' / 'end'
    result       TEXT    NOT NULL,  -- 'W' or 'D'
    wins         INTEGER NOT NULL DEFAULT 1,
    times_played INTEGER NOT NULL DEFAULT 1,
    win_rate     REAL    NOT NULL DEFAULT 1.0,
    last_seen    TEXT    NOT NULL
);

-- Preferred patterns / plays the specialist has developed
CREATE TABLE preferred_plays (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    tag           TEXT    NOT NULL,  -- human-readable label
    pos_sequence  TEXT    NOT NULL,  -- JSON array of position hashes
    win_rate      REAL    NOT NULL,
    times_played  INTEGER NOT NULL DEFAULT 0,
    promoted      INTEGER NOT NULL DEFAULT 0  -- 1 = specialist actively pursues this
);
```

### Population

After every self-play game the specialist:
1. Iterates over every position it occupied during the game
2. Inserts or updates `positions` with the final result (win/draw/loss)
3. If the game was a win or draw: records the full move sequence in `winning_lines`
4. Updates the `win_rate` on any `winning_lines` entries whose prefix was played
5. Promotes a winning line to `preferred_plays` when `times_played ≥ 5` and `win_rate ≥ 0.65`

### Malom Validation at Training Time

During training self-play (not inference), the specialist has access to Malom. After each
game, for the 10 most consequential positions (measured by heuristic swing), it queries
Malom for the WDL label and stores it in `positions.malom_label`. This means:

- The SpecialistDB accumulates Malom-quality ground truth for common positions
- At inference (no Malom), the specialist reads its own validated labels
- The DB becomes progressively more reliable as training continues

Users who download the Malom DB can optionally enable live Malom queries at inference to
accelerate SpecialistDB growth. Users without Malom still benefit from the training-time
labels baked into the DB.

### Query at Inference

For each candidate move, after computing its 62-base features, an additional lookup is
made in SpecialistDB. If `positions` has an entry for the resulting board hash with
`wins + draws + losses ≥ 10`, the WDL fraction is used to populate the counterfactual
slots `[40:57]` in the base features (the same slots Malom populates during training).
Below 10 samples: neutral 0.5 is used, same as when Malom is absent.

This means the model's learned sensitivity to counterfactual features is immediately
useful the moment SpecialistDB has enough data — no retraining required.

---

## Game Reflection Loop

After every self-play game, before moving to the next:

```
1. Record all positions → SpecialistDB (wins/draws/losses update)
2. If game won or drawn:
   a. Store move sequence as winning_line
   b. Query Malom for top-10 branching positions → malom_label
3. Update win_rate on all matching preferred_play prefixes
4. Promote: any winning_line with times_played ≥ 5, win_rate ≥ 0.65 → preferred_plays
6. Demote: preferred_play whose win_rate drops below 0.45 over last 20 encounters
7. Log: record which preferred_play (if any) was active during this game
```

The reflection loop runs in the same process after each game — it is not a background
task. At the start of each new game, if the current board position matches any promoted
`preferred_plays` prefix, the specialist treats those moves as "planned" and biases its
policy toward following the known winning line (similar to opening book following but
self-generated).

---

## HumanDB Integration

The HumanDB remains wired in as before:
- `human_norm` is computed during `_record_signals` as the max frequency across all
  legal moves from the HumanDB. This signal was previously the third slot in the old
  3-signal layout; it is now **moved into the base sentinel features** (replacing one of
  the always-zero Malom counterfactual slots at inference).
- The specialist sees human game frequency at every position, which biases it away
  from unusual moves that humans consistently avoid — a built-in anti-blunder filter.

---

## Gap Net and Value Net

Both are restored:
- **`gap_norm`** per ply (slot 4 of every ply): how likely is the opponent to blunder
  from this position. Trap planning = sequences where `gap_norm` rises as `opp_sent` falls.
- **`vn_norm`** per ply (slot 3 of every ply): value net board evaluation. This provides
  a signal independent of the heuristic, trained from game outcomes not position features.

Gap net and value net are loaded at training and inference time (same as before removal).
These are small models — they do NOT require the Malom DB to run.

---

## Specialist Coordinator Improvements

### Training — parallel separate scripts

The three specialists train in parallel from separate scripts, each writing to the shared
`data/specialist_db.sqlite`. Difficulty advances independently per specialist; no
coordination script is required.

```bash
# Run simultaneously in three terminals:
.venv/bin/python scripts/train_s_open_v2.py --auto-resume-best
.venv/bin/python scripts/train_s_mid_v2.py  --auto-resume-best
.venv/bin/python scripts/train_s_end_v2.py  --auto-resume-best
```

### Deferred — after first training run

The following coordinator features are planned but NOT yet implemented. They require at least
one full training cycle to validate that the base V4 architecture is stable.

**Smooth handoff zone**: blend outgoing/incoming specialist scores across 3-ply transitions
(60/40 → 50/50 → 40/60).

**Confidence-gated override**: if a specialist's rolling confidence (win rate on moves where
it differed from heuristic) drops below 0.30 for 50 games, fall back to heuristic for that
specialist and log a warning.

**Coordinated difficulty**: assign more games to the lagging specialist when levels diverge
by more than 2 — requires a shared state file read by all three training processes.

---

## Data Sources at Training vs Inference

| Source | Training | Inference |
|---|---|---|
| Malom DB (WDL labels) | Yes — labels 10 key positions per game | Optional (optional download) |
| HumanDB | Yes | Yes |
| Gap net | Yes | Yes |
| Value net | Yes | Yes |
| SpecialistDB | Yes (builds during training) | Yes (continues building) |
| Sentinel features | Yes (full feature_builder.py) | Yes (same) |
| Alpha-beta search | Yes (heuristic opponent) | Yes (base search) |

---

## Files Changed (implemented 2026-07-18)

| File | Status | Change |
|---|---|---|
| `learned_ai/data/specialist_db.py` | **NEW** | SpecialistDB — SQLite, D4-canonical hash, positions/winning_lines/preferred_plays, Malom label support |
| `learned_ai/models/lookahead_advisor.py` | Updated | 5-signal layout; `learner_sent`/`opp_sent` split; value_net + gap_net restored; `feat_dim = ply_depth * 5`; `sim_ply_depth` override on `score_moves_matrix` |
| `learned_ai/models/scaffolded_encoder.py` | Updated | `LOOKAHEAD_PLIES=12`, `LOOKAHEAD_SIGNALS=5` constants added |
| `learned_ai/agents/specialist_router.py` | Updated | Full-legal-moves mode (top-K branch removed); value_net + gap_net wired to LookaheadAdvisors |
| `scripts/train_s_open_v2.py` | Updated | Full-legal-moves rollout; specialist_db record_game; 1-in-20 deep games; VN + gap in LookaheadAdvisor; feat_dim = 122 |
| `scripts/train_s_mid_v2.py` | Updated | Same |
| `scripts/train_s_end_v2.py` | Updated | Same |
| `scripts/bench_scaffolded.py` | Updated | value_net + gap_net passed to router; default ply_depth = 12 |
| `docs/three-specialist-plan.md` | Updated | V4 section added |
| `docs/v4-specialist-plan.md` | **NEW** | This file |

---

## Next Steps

1. Smoke test: `--max-games 20` on all three specialists, confirm no shape errors
2. First real training run: ~500 games each specialist in parallel
3. Check SpecialistDB: `≥300` positions, some Malom-labelled entries
4. After ~5000 games: verify `preferred_plays` table has promoted entries
5. Implement deferred coordinator features (smooth handoff, confidence gate)

---

## Success Criteria

- Shape: `(n_candidates, 122)` — 62 base + 60 lookahead, full-legal-moves
- Bench: specialist overrides heuristic on ≥ 15% of moves at difficulty 1
- After 1 000 training games: SpecialistDB has ≥ 300 unique position entries with ≥ 3 samples
- After 5 000 games: preferred_plays table has ≥ 3 promoted entries
- Advancement: specialist reaches level 3 within 500 games at difficulty 1 (feasibility check)
- At inference without Malom: counterfactual slots populated from SpecialistDB for ≥ 40% of
  move decisions (vs 0% in V3 where they were always zero)
