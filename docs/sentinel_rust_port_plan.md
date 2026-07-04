# Sentinel Rust Port Plan — Informed Alpha-Beta

**Goal:** Run SentinelNet inference inside the Rust search to improve move ordering, guide LMR, and add asymmetric extension on the top strategic candidate in open middlegame positions.

**Status:** Planning — not yet started  
**Prerequisite:** Phase 2 (forcing qsearch extension) complete. ✅

---

## Why

Phase 2 adds depth on tactical positions (reachable mill threats, forced blocks). It adds nothing to **open middlegame** positions where the board is not forcing and the legal move set is large (~10–15 moves). Sentinel has a trained quality signal over those positions that alpha-beta lacks at shallow depth.

The blocker for using Sentinel inside the search today is **Python call overhead**: each inference costs ~0.3–1ms. At ~10 moves/node × many nodes/ply, this dwarfs the time budget. Moving inference to Rust via `tract` reduces per-call cost to ~5–15µs.

### What won't work and why

**Beam pruning inside alpha-beta** (the original plan) was tried and found to increase total nodes. The reason is structural: alpha-beta cutoffs come from tight minimax scores propagating up the tree. Beam-pruning changes the scores returned (now `max over top-3 by Sentinel` instead of true minimax) which poisons the TT and breaks cutoff propagation. Sibling subtrees get searched under a weaker alpha, touching more nodes. There is no "smarter guardrail" version of this that fixes it — the empirical result is a known consequence of the design.

**Sentinel prelude (N ply then alpha-beta from leaves)** is budget-fatal. A beam of width 3 for 5 ply = 243 leaf positions. Full alpha-beta on each at ply 6–8 costs ~1s per position × 243 = 243s in a 3s budget. Even a narrow beam of 2 for 4 ply = 16 leaves × ~1s ≈ 16s. Also: if Sentinel picks the wrong direction at ply 1, no amount of tactics from ply 5 recovers.

---

## Current Sentinel Architecture

```
Input:  58 floats (FEATURE_DIM)
Trunk:  Linear(58 → 128) → ReLU → Dropout(0.2)
        Linear(128 → 64) → ReLU → Dropout(0.2)
        Linear(64 → 32)  → ReLU → Dropout(0.2)
Quality head: Linear(32 → 1) → Sigmoid → scalar in [0, 1]
Optional WDL head: Linear(32 → 3) [logits, aux head, not used at inference]
```

Weights: ~28K parameters (tiny — fits in L1 cache). Forward pass cost in Python ~0.3ms, in Rust (tract) ~5–15µs.

---

## Feature Vector Decomposition

Only features 0–39 need to be computed in Rust. Features 40–57 (counterfactual / DB-derived) are zero at inference time, and the model was trained to handle this.

### Board context — 20 floats (features 0–19)

| Index | Feature | Source |
|-------|---------|--------|
| 0:4 | Phase one-hot [place, mid, end, fly] | `get_phase(board, color)` |
| 4 | own piece count / 9 | `board.bits(color).count_ones()` |
| 5 | opp piece count / 9 | `board.bits(opp).count_ones()` |
| 6 | own closed mills / 3 | scan `MILL_MASKS` |
| 7 | opp closed mills / 3 | scan `MILL_MASKS` |
| 8 | own mobility / 24 | `legal_moves(board)` count for own |
| 9 | opp mobility / 24 | `legal_moves(flipped_board)` count |
| 10 | own pieces in double mills / 9 | squares in ≥2 closed mills |
| 11 | opp pieces in double mills / 9 | same for opp |
| 12 | own placed / 9 | `board.white_placed / black_placed` |
| 13 | opp placed / 9 | same |
| 14 | own two-configs (2-of-3 with empty) / 8 | scan `MILL_MASKS` |
| 15 | opp two-configs / 8 | same |
| 16 | side to move is black | `board.side_to_move == Black` |
| 17:20 | padding (0.0) | — |

### Move-specific — 20 floats (features 20–39)

| Index | Feature | Source |
|-------|---------|--------|
| 20 | from-sq / 24 (0 for placements) | `mv.from.unwrap_or(0)` |
| 21 | to-sq / 24 | `mv.to` |
| 22 | is_placement | `mv.from.is_none()` |
| 23 | is_mill_closing | `move_forms_mill(board, color, mv.from, mv.to)` |
| 24 | is_capture | `mv.capture.is_some()` |
| 25 | captured piece index / 24 | `mv.capture.unwrap_or(0)` |
| 26 | would create double mill | check if closing lands in 2 mills |
| 27 | would block opponent mill | `(opp_threats & (1 << mv.to)) != 0` |
| 28 | resulting own piece count / 9 | after move |
| 29 | resulting opp piece count / 9 | after capture |
| 30 | resulting own mobility / 24 | `legal_moves(&nb).len()` for own |
| 31 | resulting opp mobility / 24 | `legal_moves(&nb).len()` for opp |
| 32 | resulting own mills / 3 | scan after move |
| 33 | resulting opp mills / 3 | scan after move |
| 34 | dst is junction (deg ≥ 3) | `ADJACENCY[to].count_ones() >= 3` |
| 35 | dst is corner (deg == 2) | `ADJACENCY[to].count_ones() == 2` |
| 36 | move reduces own mobility | resulting_own_mob < current_own_mob |
| 37 | opens a new mill threat | `(new_opp_threats & !old_opp_threats) != 0` |
| 38:40 | padding (0.0) | — |

### Counterfactual — 18 floats (features 40–57)

All set to `0.0` at inference time.

---

## Integration Strategy: Sentinel Informs, Alpha-Beta Decides

The principle: Sentinel scores are hints that guide alpha-beta's behavior. Alpha-beta remains the authoritative solver. Sentinel never changes what scores are returned — only which branches get searched first, how deeply, and whether to extend.

Three additive mechanisms in priority order:

### S-order — Move Ordering (do first, lowest risk)

Sentinel score is inserted into the move ordering key after TT-best, killers, mill closures, and captures. Quiet moves in open middlegame get ordered by Sentinel quality descending.

**Fires at:** `depth_remaining ≥ 4` (i.e., at least 4 plies left to the horizon). Below that threshold the node count is large but each individual node has few remaining children and ordering has diminishing effect on cutoffs. Above it, each misordering can cause an entire subtree to be searched before the refutation is found. At ply 13 (diff 6), this means Sentinel ordering fires for the top 9 plies and is silent for the bottom 4.

**Why this works:** Alpha-beta's speedup comes from cutoffs, which depend on order. Better-ordered moves produce tighter alpha updates earlier, pruning more siblings. Sentinel makes order better without changing correctness — every move is still searched or proved prunable.

**Expected gain:** More cutoffs → fewer nodes → same time budget reaches ~1 extra ply on open positions.

**Batching note:** Scoring all N moves at a node in one batch call (~20µs for 10 moves) is more efficient than N sequential calls. ~50-80% of those batch inferences are wasted (alpha-beta would have cut off before reaching later moves), but 20µs is cheap enough that the waste doesn't matter. Mark this as a known trade-off, not a bug.

### S-LMR — Sentinel-Guided Late Move Reductions (moderate impact)

LMR already reduces depth on later-ordered moves. Extend this: for moves with low Sentinel score in quiet positions (Move phase, not forcing), reduce more aggressively:
- Sentinel score < 0.3: reduce by 2 (instead of current 1)
- Sentinel score < 0.15: reduce by 3

**Fires at:** `depth_remaining ≥ 2`, which matches the existing LMR gate. Sentinel scores for these moves are already in hand from S-order (same batch call, same node). No additional inference needed — S-LMR consumes the scores S-order already produced. At depth_remaining < 2 (near-leaf nodes), LMR already doesn't fire.

**Safety net preserved:** The LMR re-search rule stays — if reduced search returns `> alpha`, re-search at full depth. This catches tactical refutations that Sentinel scored as quiet.

**Activation condition:** Move phase only. Placement and fly phase have smaller legal sets where LMR is less beneficial. Never reduce captures, mill closures, or Phase 2 forcing moves.

**Expected gain:** Aggressive LMR on Sentinel-low moves frees budget for more depth on Sentinel-high candidates.

### S-extend — Asymmetric Extension on Top Sentinel Move (highest impact, additive to Phase 2)

In open middlegame positions where Phase 2 doesn't fire (no reachable forcing lines), extend depth by +1 on the move Sentinel ranks #1. Never prune — only extend.

**Fires at:** `depth_remaining ≥ 3`. Below 3 the extension adds at most 1 ply from a near-leaf position, where Sentinel scores are noisiest (the model was trained on full middlegame positions, not near-horizon fragments). At depth_remaining = 3, a +1 extension means the top move gets searched to depth 4 instead of 3 — meaningful. At the root of a ply-13 search, depth_remaining = 13, so S-extend fires through most of the tree on the principal variation. Only fires once per node (not recursively extending extensions).

**Activation condition:**
- Move phase (not placement, not fly)
- `can_force` is false (no forcing moves found by Phase 2 qsearch filter)
- `depth_remaining ≥ 3` (see above)
- Top Sentinel score > 0.65 (high-confidence strategic candidate)

**Interaction with Phase 2:** The two mechanisms are complementary. Phase 2 fires on tactical positions. S-extend fires on strategic positions where Phase 2 doesn't. They don't overlap.

**Expected gain:** 1 extra ply of search on the top strategic candidate per open position. At ply 13 this means ply 14 on the Sentinel-best move — directly addressing the open middlegame depth problem.

---

## Export Path

### Step 1 — Export to ONNX

```python
# tools/export_sentinel_onnx.py
import torch
from learned_ai.sentinel.model import SentinelNet
from learned_ai.sentinel.infer import SentinelInfer

inf = SentinelInfer.load("learned_ai/sentinel/checkpoints/best.pt")
model = inf.model.eval()

dummy = torch.zeros(1, 58)
torch.onnx.export(
    model,
    dummy,
    "native/nmm_core/sentinel.onnx",
    input_names=["features"],
    output_names=["quality"],
    opset_version=17,
    dynamic_axes={"features": {0: "batch"}, "quality": {0: "batch"}},
)
```

Dropout layers are no-ops in `model.eval()` mode. Sigmoid is a standard ONNX op.

### Step 2 — Validate ONNX output

```python
import onnxruntime as ort
sess = ort.InferenceSession("native/nmm_core/sentinel.onnx")
out = sess.run(None, {"features": dummy.numpy()})
assert abs(out[0][0] - model(dummy).item()) < 1e-5
```

---

## Rust Inference: `tract` crate

**Choice: `tract`** (pure Rust ONNX inference, no C dependencies)

Why `tract` over `ort`:
- No ONNX Runtime C library (~200MB)
- Sub-10µs forward pass for networks this size (5–8µs for 4-layer MLP on x86)
- Compiles into the Rust binary — zero setup
- All ops used by SentinelNet (Gemm, Relu, Sigmoid, Dropout→Identity) are supported

Add to `native/nmm_core/Cargo.toml`:
```toml
[dependencies]
tract-onnx = "0.21"
```

### Rust inference module — `native/nmm_core/src/sentinel.rs`

```rust
use tract_onnx::prelude::*;

pub struct SentinelEngine {
    model: SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>,
}

impl SentinelEngine {
    pub fn load(path: &str) -> TractResult<Self> {
        let model = tract_onnx::onnx()
            .model_for_path(path)?
            .with_input_fact(0, f32::fact([1, 58]))?
            .into_optimized()?
            .into_runnable()?;
        Ok(Self { model })
    }

    /// Score a single move. Returns quality in [0, 1].
    pub fn score(&self, features: &[f32; 58]) -> f32 {
        let input = tract_ndarray::arr1(features)
            .into_shape([1, 58]).unwrap()
            .into();
        let result = self.model.run(tvec![input]).unwrap();
        result[0].as_slice::<f32>().unwrap()[0]
    }

    /// Score a batch of moves in one forward pass.
    pub fn score_batch(&self, features: &[[f32; 58]]) -> Vec<f32> {
        let n = features.len();
        let flat: Vec<f32> = features.iter().flat_map(|f| f.iter().copied()).collect();
        let input = tract_ndarray::Array2::from_shape_vec([n, 58], flat)
            .unwrap().into();
        let result = self.model.run(tvec![input]).unwrap();
        result[0].as_slice::<f32>().unwrap().to_vec()
    }
}
```

**Performance target:** `score_batch` for 10 moves in <100µs. Expected: ~20µs for a batch of 10.

### Caching Sentinel scores

Sentinel scores for a position can be stored in the TT entry (one extra `f16` per slot, ~4MB for a 2M-entry TT). On TT probe, if the Sentinel score for the same position is available, skip inference. This eliminates inference on transpositions and TT-hit paths — the majority of internal node visits.

---

## Implementation Phases

| Phase | Task | Effort | Status |
|-------|------|--------|--------|
| S0 | Export `sentinel.onnx` via `tools/export_sentinel_onnx.py` | 1h | Not started |
| S0 | Validate ONNX vs PyTorch output (< 1e-4 error) | 0.5h | Not started |
| S1 | Add `tract-onnx` dep + `sentinel.rs` inference module | 2h | Not started |
| S1 | Python test: Rust scores vs Python scores agree on 1000 positions | 1h | Not started |
| S2 | Implement `build_features_rust(board, mv)` → `[f32; 58]` | 4h | Not started |
| S2 | Port board context features (0–19) | 2h | Not started |
| S2 | Port move-specific features (20–39) | 2h | Not started |
| S3 | Add `SentinelEngine` to `Searcher`, pass via FFI | 1h | Not started |
| S3 | Implement S-order: Sentinel as move ordering key for quiet moves | 2h | Not started |
| S3 | Benchmark nodes-to-depth-N before/after on 20 positions | 1h | Not started |
| S4 | Implement S-LMR: feed Sentinel score into LMR reduction table | 2h | Not started |
| S4 | Verify tactical soundness: Phase 2 forcing test still fires; WDL positions correct | 1h | Not started |
| S5 | Implement S-extend: +1 depth on top Sentinel move in open middlegame | 2h | Not started |
| S5 | Measure extra ply on strategic positions where Phase 2 doesn't fire | 1h | Not started |
| S6 | Tune activation thresholds and reduction aggressiveness | 2h | Not started |
| S6 | Run 20-game V2 vs V2+Sentinel to measure ELO delta | 3h | Not started |

Total estimated: ~27h implementation + validation.

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| S-order adds inference overhead that costs more than cutoffs save | Measure nodes reduction in S3 benchmark; if overhead > savings, batch inference only at depth ≥ 3 |
| S-LMR reduces a move that turns out to be a tactical refutation | LMR re-search at full depth if reduced search returns > alpha — this is already in the search |
| S-extend fires on the wrong strategic candidate in a position where the true best move is Sentinel's #2 | S-extend adds +1 ply, not infinity; alpha-beta still searches all moves at full depth and picks the true best; worst case is wasted +1 ply |
| `tract` doesn't support a required ONNX op | `ort` as fallback; confirm all ops before S1 |
| Feature drift: Rust features disagree with Python features | S1 validation step with ≥1000 positions before any search integration |
