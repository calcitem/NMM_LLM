# Personality Re-integration Plan for HeuristicsV2

## Problem

With `use_v2_heuristics = True`, the personality preset weights have no effect:

| What's bypassed | Why personality breaks |
|---|---|
| `evaluate()` | `mill_count_scale`, `mobility_scale`, `blocked_scale`, `long_term_position` all applied there |
| `tactical_move_bonus` at root | `close_mill`, `block_opponent_mill`, `cycling_mill`, `feeder_diamond`, `cardinal_block`, etc. all applied there |
| `_populate_thinking` | Early return means no thinking breakdown is shown |

These three still work independently of the evaluator:
- `make_mistakes` — blunder probability applied in `choose_move()` ✓
- `opening_adherence` — book adherence; independent ✓
- `value_net_blend` — Sentinel blend; independent ✓

---

## Three-Layer Plan

### Layer 1 — Re-enable `tactical_move_bonus` at root (immediate, low risk)

`tactical_move_bonus` only scores moves at the root before the tree descends — it affects move ordering, not leaf evaluation. Re-enabling it in v2 mode restores style immediately without touching the v2 leaf evaluator.

**Changes:**

1. `ai/game_ai.py` line ~1567 in `_root_search` — remove the v2 guard:
   ```python
   # Before:
   if abs(score_raw) < INF // 2 and not self.use_v2_heuristics:
       score = score_raw + tactical_move_bonus(...)

   # After:
   if abs(score_raw) < INF // 2:
       score = score_raw + tactical_move_bonus(...)
   ```

2. `ai/game_ai.py` `_populate_thinking` — remove the early return:
   ```python
   # Before:
   if self.use_v2_heuristics:
       return

   # After:
   # (delete those two lines)
   ```

**Personality coverage after Layer 1:**

| Personality | Restored? | Notes |
|---|---|---|
| `aggressive` | ~90% | `close_mill`, `cycling_mill` via tactical; `block_opponent_mill` at floor |
| `defensive` | ~90% | `block_opponent_mill`, `stop_opponent_mills`, `cardinal_block` via tactical |
| `scholar` | ~85% | `opening_adherence` independent; `setup_mill/feeder_diamond` via tactical; `scatter_placement` has no v2 path |
| `chaos` | ~80% | `make_mistakes` independent; `cardinal_block/cycling_mill` via tactical; `scatter_placement` no path |
| `positional` | ~40% | `feeder_diamond/setup_mill` via tactical, but `mill_count_scale/mobility_scale/blocked_scale` still zero-effect |
| `balanced` | 100% | Defaults only; no change |

---

### Layer 2 — Scale v2 leaf constants with weight multipliers (medium effort)

The three "big" multipliers from `HeuristicWeights` each map directly onto v2 constant groups:

| Weight field | Scales these v2 constants |
|---|---|
| `mill_count_scale` | `_V2_*_MILL`, `_V2_*_THREAT` |
| `mobility_scale` | `_V2_*_MOB` |
| `blocked_scale` | `_V2_*_BLOCKED` |

**Implementation approach:**

Add an optional `weights` parameter to `evaluate_v2(board, color, _ply=0, weights=None)`. When provided, apply scale multipliers inline:

```python
def evaluate_v2(board, color, _ply=0, weights=None):
    mill_s   = weights.mill_count_scale / 100 if weights else 1.0
    mob_s    = weights.mobility_scale   / 100 if weights else 1.0
    block_s  = weights.blocked_scale    / 100 if weights else 1.0
    ...
    # In placement section:
    score += int(_V2_PL_MILL   * mill_s)  * own_mills
    score += int(_V2_PL_MOB    * mob_s)   * own_mob
    score += int(_V2_PL_BLOCKED * block_s) * own_blocked
    ...
```

Then in `_negamax` depth-0, pass `self._weights`:
```python
heur = evaluate_v2(board, board.turn, _ply=ply, weights=self._weights)
```

This fully restores `positional` personality (which relies almost entirely on these three scale fields).

**Do this step when:** The personalities are actively being tested or when a user reports that the `positional` style feels the same as `balanced`.

---

### Layer 3 — Full `V2Weights` dataclass (future, after self-play tuning)

Once self-play data reveals which v2 constants have the most influence, define a `V2Weights` dataclass with named per-personality overrides for all v2 constants. Map each v1 personality definition to v2 equivalents.

This is only warranted if the scale-layer (Layer 2) proves insufficient for desired stylistic range.

---

## What has no v2 path (accept or defer)

| Field | Used by | Status |
|---|---|---|
| `scatter_placement` | `scholar`, `chaos` | No v2 placement position logic; accept for now |
| `long_term_position` | `positional` | `_v2_position_value()` exists but no scale hook; Layer 2 addition if needed |
| `feeder_diamond` | many | Works via tactical_move_bonus at root (Layer 1) |

---

## Recommended order

1. **Do Layer 1 now** — two-line change, restores ~80-90% of personality colour for aggressive/defensive/scholar/chaos.
2. **Do Layer 2 next session** — restores positional; minimal risk since multipliers are 0.8–1.33× range.
3. **Defer Layer 3** — until self-play data is available.
