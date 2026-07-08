#!/usr/bin/env python3
"""
tools/train_value_net_filtered.py — Train value net on human games.

Three improvements over basic outcome training:

  1. Entire dataset  — default min-elo=0 uses all games in data/human_games.
     Pass --min-elo 1255 to restore the top-25% Elo filter.

  2. Draws included  — neutral games contribute a 0.0 label by default, which
     the old (decisive-only) training missed.  Pass --decisive-only to exclude.

  3. Placement blend — early placement positions have noisy outcome labels
     (the game is far from resolved).  For every unique placement-phase FEN,
     a fraction (--placement-blend, default 0.35) of the label comes from
     evaluate_v2() normalised to [-1,1] via tanh(score / --heuristic-scale).
     Move and fly phase labels are pure outcome.

Output: data/value_net_human_v2.npz   (does NOT touch value_net.npz or
        value_net_human_filtered.npz — both are preserved for A/B bench).

Usage:
    .venv/bin/python tools/train_value_net_filtered.py
    .venv/bin/python tools/train_value_net_filtered.py --min-elo 1255
    .venv/bin/python tools/train_value_net_filtered.py --placement-blend 0.4 --epochs 100
    .venv/bin/python tools/train_value_net_filtered.py --output data/my_net.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np

from game.board import BoardState, POSITIONS
from ai.value_net import ValueNet, board_to_features, _INPUT_DIM

# Normalisation scale for evaluate_v2 → tanh(score / scale) in [-1, 1].
# Placement phase typical max ≈ ±465 (mills/threats dominant), so 200 gives
# a well-spread signal: tanh(200/200)=0.76, tanh(400/200)=0.96.
_DEFAULT_HEURISTIC_SCALE = 200


# ── Board reconstruction ──────────────────────────────────────────────────────

def fen_to_board(fen: str) -> BoardState:
    parts = fen.split("|")
    board_str, turn, w_placed_s, b_placed_s = parts[0], parts[1], parts[2], parts[3]
    w_placed = int(w_placed_s)
    b_placed = int(b_placed_s)
    positions: dict[str, str] = {}
    for i, pos in enumerate(POSITIONS):
        c = board_str[i]
        positions[pos] = "" if c == "." else c
    w_on = sum(1 for v in positions.values() if v == "W")
    b_on = sum(1 for v in positions.values() if v == "B")
    w_cap = max(0, b_placed - b_on)
    b_cap = max(0, w_placed - w_on)
    return BoardState(
        positions=positions,
        turn=turn,
        pieces_on_board={"W": w_on, "B": b_on},
        pieces_placed={"W": w_placed, "B": b_placed},
        pieces_captured={"W": w_cap, "B": b_cap},
    )


# ── Game record iterator ──────────────────────────────────────────────────────

def _iter_records(fpath: Path):
    try:
        content = fpath.read_text().strip()
    except Exception:
        return
    if not content:
        return
    try:
        obj = json.loads(content)
        if isinstance(obj, dict):
            yield obj
            return
        if isinstance(obj, list):
            for o in obj:
                if isinstance(o, dict):
                    yield o
            return
    except json.JSONDecodeError:
        pass
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                yield obj
        except json.JSONDecodeError:
            continue


# ── Dataset extraction with placement blend ───────────────────────────────────

def extract_samples(
    games_dir: Path,
    min_elo: int,
    decisive_only: bool,
    placement_blend: float,
    heuristic_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Scan games_dir/**/*.jsonl.  Returns (X, y) FEN-deduplicated:
      X: (N, _INPUT_DIM) float32
      y: (N,) float32 in [-1, 1]

    Label for each unique FEN:
      - Move/fly phase:  mean game outcome across all games that visited this FEN
      - Placement phase: (1-placement_blend) * mean_outcome
                         + placement_blend  * tanh(evaluate_v2(board, color) / heuristic_scale)

    If min_elo > 0, games where min(white_elo, black_elo) < min_elo are skipped.
    If decisive_only, games without a winner are skipped (draws contribute 0.0 labels).
    """
    try:
        from ai.heuristics import evaluate_v2 as _ev2
    except Exception:
        _ev2 = None
        if placement_blend > 0:
            print("  [warn] evaluate_v2 unavailable — placement blend disabled.")
            placement_blend = 0.0

    fen_labels:    dict[str, list[float]] = {}
    fen_features:  dict[str, np.ndarray]  = {}
    fen_heuristic: dict[str, float]        = {}   # placement FENs only

    files = sorted(games_dir.rglob("*.jsonl"))
    if not files:
        sys.exit(f"No .jsonl files found in {games_dir}")

    print(f"Scanning {len(files):,} game files ...", flush=True)
    kept_games   = 0
    skipped_elo  = 0
    skipped_draw = 0

    for i, fpath in enumerate(files):
        if i % 10_000 == 0 and i > 0:
            print(f"  {i:,} / {len(files):,}  kept={kept_games:,}", flush=True)

        for record in _iter_records(fpath):
            we = record.get("white_elo")
            be = record.get("black_elo")
            if min_elo > 0 and (we is None or be is None or min(we, be) < min_elo):
                skipped_elo += 1
                continue

            winner = record.get("winner")
            moves  = record.get("moves", [])
            if not moves:
                continue
            if decisive_only and winner is None:
                skipped_draw += 1
                continue

            kept_games += 1
            for entry in moves:
                fen = entry.get("board_fen_before")
                if not fen:
                    continue
                try:
                    board = fen_to_board(fen)
                except Exception:
                    continue

                color = board.turn
                if winner == color:
                    label = 1.0
                elif winner is not None and winner != color:
                    label = -1.0
                else:
                    label = 0.0

                if fen not in fen_features:
                    fen_features[fen] = board_to_features(board, color)
                    fen_labels[fen]   = []
                    # Compute placement-phase heuristic signal once per unique FEN.
                    if placement_blend > 0 and _ev2 is not None and board.phase == "place":
                        try:
                            raw = _ev2(board, color)
                            fen_heuristic[fen] = float(np.tanh(raw / heuristic_scale))
                        except Exception:
                            pass
                fen_labels[fen].append(label)

    print(f"  Kept {kept_games:,} games"
          + (f" (min Elo >= {min_elo})" if min_elo > 0 else " (all Elo levels)") + ".")
    if min_elo > 0 and skipped_elo:
        print(f"  Skipped {skipped_elo:,} below Elo threshold.")
    if decisive_only and skipped_draw:
        print(f"  Skipped {skipped_draw:,} draw/unknown games.")

    if not fen_features:
        sys.exit("No training samples extracted — check games directory and filters.")

    fens = list(fen_features.keys())
    X = np.stack([fen_features[f] for f in fens]).astype(np.float32)

    # Build final labels: blend heuristic into placement positions.
    n_blended = 0
    y_vals: list[float] = []
    for f in fens:
        outcome = float(np.mean(fen_labels[f]))
        if f in fen_heuristic:
            label = (1.0 - placement_blend) * outcome + placement_blend * fen_heuristic[f]
            n_blended += 1
        else:
            label = outcome
        y_vals.append(label)

    y = np.array(y_vals, dtype=np.float32)

    n_place = sum(1 for f in fens if f in fen_heuristic)
    n_other = len(fens) - n_place
    if placement_blend > 0:
        print(f"  Placement-phase positions: {n_place:,}  ({100*n_place/len(fens):.1f}%)"
              f"  — blended {placement_blend:.0%} heuristic + {1-placement_blend:.0%} outcome")
        print(f"  Move/fly-phase positions:  {n_other:,}")

    return X, y


# ── Adam training loop (unchanged) ───────────────────────────────────────────

def _mse(net: ValueNet, X: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((net.predict_batch(X) - y) ** 2))


def _snapshot(net: ValueNet) -> dict[str, np.ndarray]:
    return {k: getattr(net, k).copy() for k in ("W1", "b1", "W2", "b2", "W3", "b3")}


def _restore(net: ValueNet, snap: dict[str, np.ndarray]) -> None:
    for k, arr in snap.items():
        setattr(net, k, arr)


def train_adam(
    net: ValueNet,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    patience: int,
) -> None:
    """Train in-place: Adam + L2 weight decay + early stopping on val MSE."""
    β1, β2, ε = 0.9, 0.999, 1e-8
    PARAM_KEYS  = ("W1", "b1", "W2", "b2", "W3", "b3")
    WEIGHT_KEYS = {"W1", "W2", "W3"}

    m = {k: np.zeros_like(getattr(net, k)) for k in PARAM_KEYS}
    v = {k: np.zeros_like(getattr(net, k)) for k in PARAM_KEYS}

    best_val     = float("inf")
    best_weights = _snapshot(net)
    no_improve   = 0
    t            = 0

    rng = np.random.default_rng(0)
    N   = len(X_tr)

    for ep in range(1, epochs + 1):
        order  = rng.permutation(N)
        X_tr_s = X_tr[order]
        y_tr_s = y_tr[order]
        ep_loss = 0.0
        steps   = 0

        for start in range(0, N, batch_size):
            xb = X_tr_s[start:start + batch_size]
            yb = y_tr_s[start:start + batch_size].reshape(-1, 1)
            B  = len(xb)
            t += 1

            h1   = np.maximum(0.0, xb @ net.W1.T + net.b1)
            h2   = np.maximum(0.0, h1 @ net.W2.T + net.b2)
            pred = np.tanh(h2 @ net.W3.T + net.b3)

            err      = pred - yb
            ep_loss += float(np.mean(err ** 2))
            steps   += 1

            d3  = (2.0 / B) * err * (1.0 - pred ** 2)
            dW3 = d3.T @ h2
            db3 = d3.sum(axis=0)

            dh2  = d3 @ net.W3
            dh2 *= (h2 > 0).astype(np.float32)
            dW2  = dh2.T @ h1
            db2  = dh2.sum(axis=0)

            dh1  = dh2 @ net.W2
            dh1 *= (h1 > 0).astype(np.float32)
            dW1  = dh1.T @ xb
            db1  = dh1.sum(axis=0)

            grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}
            for k in WEIGHT_KEYS:
                grads[k] += weight_decay * getattr(net, k)

            bc1 = 1.0 - β1 ** t
            bc2 = 1.0 - β2 ** t
            for k in PARAM_KEYS:
                g    = grads[k]
                m[k] = β1 * m[k] + (1.0 - β1) * g
                v[k] = β2 * v[k] + (1.0 - β2) * g * g
                step = lr * (m[k] / bc1) / (np.sqrt(v[k] / bc2) + ε)
                getattr(net, k).__isub__(step)

        tr_loss  = ep_loss / max(steps, 1)
        val_loss = _mse(net, X_val, y_val)
        improved = val_loss < best_val - 1e-6
        marker   = " *" if improved else ""
        print(f"  epoch {ep:3d}/{epochs}  tr={tr_loss:.5f}  val={val_loss:.5f}{marker}")

        if improved:
            best_val     = val_loss
            best_weights = _snapshot(net)
            no_improve   = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stop — no improvement for {patience} epochs.")
                break

    _restore(net, best_weights)
    print(f"  Restored best weights (val={best_val:.5f}).")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train value net on human games with placement-phase heuristic blend",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--games-dir",       type=Path,  default=_ROOT / "data" / "human_games",
                    help="Directory of JSONL human game files")
    ap.add_argument("--min-elo",         type=int,   default=0,
                    help="Min per-game Elo (weaker player). 0 = include all games.")
    ap.add_argument("--output",          type=Path,
                    default=_ROOT / "data" / "value_net_human_v2.npz",
                    help="Output .npz path")
    ap.add_argument("--placement-blend", type=float, default=0.35,
                    help="Fraction of placement-phase label from evaluate_v2 heuristic. "
                         "0 = pure outcome, 1 = pure heuristic.")
    ap.add_argument("--heuristic-scale", type=float, default=_DEFAULT_HEURISTIC_SCALE,
                    help="Denominator in tanh(score/scale) to normalise evaluate_v2 output.")
    ap.add_argument("--epochs",          type=int,   default=100)
    ap.add_argument("--lr",              type=float, default=3e-4)
    ap.add_argument("--batch-size",      type=int,   default=256)
    ap.add_argument("--val-frac",        type=float, default=0.1)
    ap.add_argument("--patience",        type=int,   default=10,
                    help="Early-stop patience (epochs)")
    ap.add_argument("--weight-decay",    type=float, default=1e-4)
    ap.add_argument("--decisive-only",   dest="decisive_only", action="store_true", default=False,
                    help="Exclude draw/unknown games (default: draws included)")
    ap.add_argument("--no-draws",        dest="decisive_only", action="store_true",
                    help="Alias for --decisive-only")
    args = ap.parse_args()

    if not args.games_dir.is_dir():
        sys.exit(f"Games directory not found: {args.games_dir}")

    # Safety: refuse to overwrite any previously trained net
    _protected = [
        _ROOT / "data" / "value_net.npz",
        _ROOT / "data" / "value_net_human_filtered.npz",
    ]
    for p in _protected:
        if args.output.resolve() == p.resolve():
            sys.exit(f"Refusing to overwrite {p.name} — specify a different --output.")

    # ── Extract ───────────────────────────────────────────────────────────────
    X, y = extract_samples(
        args.games_dir,
        min_elo=args.min_elo,
        decisive_only=args.decisive_only,
        placement_blend=args.placement_blend,
        heuristic_scale=args.heuristic_scale,
    )
    N = len(X)
    print(f"\n{N:,} unique FEN positions extracted.")

    # Label distribution
    n_win  = int((y >  0.5).sum())
    n_draw = int((np.abs(y) <= 0.5).sum())
    n_loss = int((y < -0.5).sum())
    print(f"  +1 (win):  {n_win:,}  ({100*n_win/N:.1f}%)")
    print(f"   0 (draw): {n_draw:,}  ({100*n_draw/N:.1f}%)")
    print(f"  -1 (loss): {n_loss:,}  ({100*n_loss/N:.1f}%)")
    denom = n_win + n_loss
    if denom > 0 and abs(n_win - n_loss) / denom > 0.10:
        print(f"  NOTE: win/loss imbalance {abs(n_win-n_loss)/denom:.1%}"
              f" — may indicate colour bias in the dataset.")

    # ── Train/val split ───────────────────────────────────────────────────────
    rng = np.random.default_rng(42)
    idx = rng.permutation(N)
    X, y  = X[idx], y[idx]
    n_val = max(1, int(N * args.val_frac))
    X_val, y_val = X[:n_val],  y[:n_val]
    X_tr,  y_tr  = X[n_val:],  y[n_val:]
    print(f"\nTrain: {len(X_tr):,}  Val: {len(X_val):,}")

    # ── Train ─────────────────────────────────────────────────────────────────
    net = ValueNet()
    print(f"\nAdam  lr={args.lr}  weight_decay={args.weight_decay}"
          f"  batch={args.batch_size}  patience={args.patience}\n")
    train_adam(
        net, X_tr, y_tr, X_val, y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    args.output.parent.mkdir(parents=True, exist_ok=True)
    net.save(args.output)
    print(f"\nSaved  → {args.output}")
    print(f"Protected nets unchanged: value_net.npz, value_net_human_filtered.npz")


if __name__ == "__main__":
    main()
