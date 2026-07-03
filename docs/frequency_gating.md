# Frequency-Gated Opponent Move Pruning

A forward-pruning technique for the negamax search: at mid-to-deep plies,
skip opponent moves that are both (a) unlikely to be played by a real human
and (b) bad for the opponent by a static/learned measure.

---

## Problem

Standard alpha-beta assumes the opponent plays **optimally** — it searches
responses to every legal opponent move, including moves a real player would
never make. At depth 12 with branching factor 6, ~half the nodes may be
subtrees rooted at opponent moves with near-zero real-world frequency.

---

## Where it fires

- **Only at opponent plies** (opponent is to move at this node).
- **Only at remaining depth ≤ PRUNE_DEPTH** (e.g. 5). At shallower
  remaining depth the subtrees are small anyway; at deeper remaining depth
  the eval gate is too noisy to trust.
- **Not at the root** — the AI must handle any move the human actually plays.
- **Not in quiescence search** — already restricted to tactical moves.
- **Move-phase only (initially)** — in placement, all 24 squares look
  similar statically; the eval gate can't discriminate well. Restrict to
  move/fly phase first and re-evaluate.

---

## The two gates

Both must fire to prune a move.

### Gate 1 — Evaluation gate (always available)

Apply the move to get successor board S. Evaluate S from the opponent's
perspective (positive = good for opponent). Compare to the best opponent
move seen so far at this node:

```
eval(S) < best_opponent_eval - EVAL_MARGIN
```

A move that evaluates more than `EVAL_MARGIN` below the best available move
is a candidate for pruning.

**Three options for the evaluator, from fastest to most informative:**

| Option | Cost per move | Accuracy | Notes |
|--------|--------------|----------|-------|
| A. Static eval (`evaluate_v2`) | ~1–5 µs (Rust) | Moderate | Available everywhere; noisy at mid-depth |
| B. Value net (`value_net.predict`) | ~1–5 ms (PyTorch) | Higher | Neural; per-move cost; batching helps |
| C. Sentinel (`advise` batch) | ~5–15 ms for all moves | Highest | One pass for all N moves; fixed overhead |

See benchmarks section below for actual timings across 100 human-DB positions.

**Initial recommendation:** Use option A (static eval) to keep Rust-native
speed. The eval gate is conservative (large EVAL_MARGIN) so accuracy matters
less than cost.

### Gate 2 — Frequency gate (position-dependent)

The opponent move is unlikely to be played by a real human. Two proxies:

**2a. Root-level frequency carrydown** (cheapest)
At the root (once per AI turn), query `trajectory_db.query_all_frequencies`
for the current position. Moves with total frequency < FREQ_THRESHOLD are
flagged as rare. For tree nodes, carry down the flag from the nearest
ancestor that was a root-level rare move (i.e., "this line started from
an unlikely human choice").

Problem: this only works one ply deep. Doesn't help at depth 4+ opponent plies.

**2b. Structural move properties** (no DB needed)
Some move types are structurally rare in human games regardless of position:
- Placing on a known B-64 dead square (already penalised in eval).
- Voluntarily breaking an almost-complete mill.
- Moving to a square that gives the opponent an immediate capture.
- Placing away from any mill when opponent has an immediate mill threat.

These are rules that can be checked in O(1) from board state. Reliable
because they're structural, not position-lookup.

**2c. Ngram model** (per-node, cheap-ish)
`ngram_model.predict(turn, game_notations)` returns a distribution over
moves given the game sequence. Moves with probability < NGRAM_THRESHOLD
are flagged. But: at tree-internal nodes we don't have the actual game
sequence up to that point — we'd need to extend game_notations with the
path from root to this node, which is bookkeeping overhead.

**Initial recommendation:** Use 2b (structural properties). No DB queries
inside the search. Combine with 2a at the first opponent ply only.

---

## Margin design

```
prune if:
    eval(successor) < best_opp_eval_at_node - EVAL_MARGIN
    AND move_is_structurally_rare(move, board)
```

### EVAL_MARGIN values to try

| Value | Effect |
|-------|--------|
| 50 | Very conservative; prunes only catastrophic blunders |
| 150 | Moderate; prunes losing-a-piece-for-nothing moves |
| 300 | Aggressive; prunes anything more than ~1 piece down |

Start at 150. Benchmark node reduction vs. move quality regression.

### PRUNE_DEPTH values to try

| Value | Effect |
|-------|--------|
| 3 | Only prunes very close to leaves; small gain |
| 5 | Good tradeoff; 5-ply subtrees are ~7,000 nodes each |
| 7 | More pruning but eval noise is higher at 7-from-leaf |

Start at 5.

---

## Keep-N alternative

Instead of dual-gate, keep only the top K opponent moves at each node
(sorted by static eval, descending from opponent's perspective):

```
K = max(MIN_KEEP, ceil(n_moves * KEEP_FRAC))
```

Where `KEEP_FRAC = 0.70` (keep 70%, prune bottom 30%) and `MIN_KEEP = 2`.

Simpler to implement; no frequency gate needed. Risk: bad move ordering
could prune the right move. Mitigate with good move ordering (capture-first,
mill-forming first) so the bottom 30% really are the weak moves.

---

## Value net / Sentinel as the eval gate (Option B/C)

If benchmarks show value net or sentinel can score all N moves in < 1ms
per position, they become viable for the eval gate inside the search.

**Value net (per-move):**
- 1 inference per successor board per pruning check
- At a node with 6 moves and remaining depth 5, that's 6 VN calls per node,
  and there may be thousands of such nodes → likely too slow unless batched.

**Sentinel (batched over all moves at one node):**
- 1 forward pass scoring all N moves at once; fixed overhead ~5–15ms
- If the node has 6 moves, sentinel is ~6× cheaper per move than VN
- BUT: sentinel is a policy network (which move to play), not a value network
  (how good is the position). Its scores reflect move quality for the *current
  player*, not position quality for the *opponent* — different semantics than
  what the eval gate needs.
- Still useful as a frequency proxy: moves sentinel scores very low for the
  opponent are moves the opponent is unlikely to choose.

**Benchmark results (100 move-phase positions, avg 5.9 legal moves):**

| Approach | µs / move | ms / position | Notes |
|----------|-----------|---------------|-------|
| Static eval (`evaluate_v2`) | **8.8** | 0.05 | Python; ~1.3µs in Rust |
| Value net (`predict`) | **13.2** | 0.08 | NumPy; only 1.5× static eval |
| Sentinel (`advise` batch) | **91.1** | 0.54 | Fixed per-position; 10× static |

**Key finding:** Value net is only 1.5× slower than static eval — it's a tiny
NumPy network. However, it lives in Python, so FFI overhead from Rust makes
it impractical inside the search. Sentinel at 0.54ms per node is completely
ruled out (~26s overhead at 50k nodes).

**Recommendation after benchmarking:**
- Use static eval (Rust) as the eval gate. ~1.3µs per move in Rust means
  the gate adds <5% overhead per node while potentially pruning 25-30% of
  subtrees.
- If value net is ever ported to Rust or replaced with a fast linear
  approximation, it becomes essentially free and would give better gate
  accuracy.
- Sentinel: ruled out for in-search use. It can continue to be used at the
  root (once per turn) as it is today.

---

## Empirical result — eval-only implementation

**Result: eval-only gate (without Gate 2b) increased node count by ~40%.**

Baseline (depth 10): 871,188 nodes. With eval-only gate (DEPTH=5, MARGIN=150,
hard prune): ~1.22M nodes (+40%). Depth-reduction variant (gate_red=3) also
increased nodes.

**Root cause:** Move ordering already puts captures/mills first. `best_opp_static`
after MIN_KEEP=2 reflects these tactical moves. Subsequent quiet/positional moves
score ~150–400 below tactically. The eval gate fires on quiet moves that humans
*do* play and that alpha-beta needs for β-cutoffs. Removing them prevents cutoffs
→ more nodes explored downstream. PVS null-window re-searches also inflate when
gated moves look good from AI's perspective.

**Conclusion:** Gate 2b (structural properties) is required. Eval-only is
insufficient and harmful. The dual-gate condition `eval_gate AND structural_gate`
is the correct design — `is_structurally_rare` must be implemented in Rust before
this feature can be turned on.

## Expected gains (with both gates)

| Phase | Branching factor | Bottom 30% pruned | Node reduction | Effective ply gain |
|-------|-----------------|-------------------|---------------|-------------------|
| Move | 5–8 | ~1.5–2.4 moves | ~25–35% | +0.5–1.0 ply |
| Fly | 3–6 | ~1–2 moves | ~20–30% | +0.4–0.7 ply |
| Placement | 15–24 | ~5–7 moves | ~30% | +0.5 ply |

These are rough estimates assuming correct dual-gate implementation.

---

## Implementation location

`native/nmm_core/src/search.rs` — inside `negamax`, in the opponent-move
loop, after move ordering and before the recursive call:

```rust
// Frequency-gated pruning: skip clearly-bad opponent moves at shallow depth.
if depth <= PRUNE_DEPTH && opponent_to_move && move_idx > 0 {
    let succ_eval = evaluate_v2(&after, stm);
    if succ_eval < best_opp_eval - EVAL_MARGIN && is_structurally_rare(&mv, board) {
        continue;  // prune this subtree
    }
    best_opp_eval = best_opp_eval.max(succ_eval);
}
```

`is_structurally_rare` is a cheap Rust predicate checking the structural
properties listed under Gate 2b.

---

## Benchmarks to run before implementing

See `tools/bench_gating.py` — measures per-position cost of:
- Static eval across all legal moves
- Value net across all legal moves  
- Sentinel batch across all legal moves

Across 100 random move-phase positions from `data/human_db.sqlite`.
