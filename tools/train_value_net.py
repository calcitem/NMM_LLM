#!/usr/bin/env python3
"""
tools/train_value_net.py — Train the MLP value network from JSONL game records.

Reads every data/games/*.jsonl file, replays positions, assigns final-outcome
labels, and trains ai.value_net.ValueNet.  Saves the result to data/value_net.npz.

Usage:
    .venv/bin/python tools/train_value_net.py [--epochs N] [--lr F] [--games-dir PATH]

The trained network is automatically picked up by MCTS when data/value_net.npz exists.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on sys.path when run directly.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np

from game.board import BoardState, POSITIONS
from ai.value_net import ValueNet, board_to_features, _INPUT_DIM


# ── Board reconstruction ──────────────────────────────────────────────────────

def fen_to_board(fen: str) -> BoardState:
    """Reconstruct a BoardState from the compact FEN stored in game JSONL files."""
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
    # pieces captured BY color = opponent pieces placed but no longer on board.
    w_cap = max(0, b_placed - b_on)   # W captured B pieces
    b_cap = max(0, w_placed - w_on)   # B captured W pieces
    return BoardState(
        positions=positions,
        turn=turn,
        pieces_on_board={"W": w_on, "B": b_on},
        pieces_placed={"W": w_placed, "B": b_placed},
        pieces_captured={"W": w_cap, "B": b_cap},
    )


# ── Dataset extraction ────────────────────────────────────────────────────────

def _iter_records(fpath: Path):
    """Yield game-record dicts from a JSONL file (single object, array, or line-delimited)."""
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


def extract_samples(games_dirs: list[Path], decisive_only: bool = False) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Read all JSONL files from one or more directories and build feature matrix X and label vector y.

    For each unique board position (identified by FEN):
      color  = board.turn (the player about to move)
      label  = mean of all outcomes seen for that FEN:
               +1.0 if color wins, -1.0 if color loses, 0.0 for draw/unknown.

    Using the mean across all games prevents repeated early-game positions from
    dominating the signal when many games share the same opening positions.
    decisive_only: if True, skip games with no winner (draws/unknowns) entirely.
    """
    # Collect per-FEN labels and features (features are deterministic from the board).
    fen_labels: dict[str, list[float]] = {}
    fen_features: dict[str, np.ndarray] = {}
    files = sorted({f for d in games_dirs for f in d.rglob("*.jsonl")})
    skipped = 0

    for fpath in files:
        for record in _iter_records(fpath):
            winner = record.get("winner")
            moves = record.get("moves", [])
            if not moves:
                continue
            if decisive_only and winner is None:
                skipped += 1
                continue

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
                    fen_labels[fen] = []
                fen_labels[fen].append(label)

    if decisive_only and skipped:
        print(f"  Skipped {skipped} draw/unknown games.")

    if not fen_features:
        dirs_label = ", ".join(str(d) for d in games_dirs)
        print(f"No training samples found.  Ensure these directories contain JSONL files: {dirs_label}")
        sys.exit(1)

    fens = list(fen_features.keys())
    X = np.stack([fen_features[f] for f in fens]).astype(np.float32)
    y = np.array([float(np.mean(fen_labels[f])) for f in fens], dtype=np.float32)
    return X, y, len(files)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train NMM value network")
    parser.add_argument("--epochs",    type=int,   default=30,
                        help="Training epochs (default: 30)")
    parser.add_argument("--lr",        type=float, default=0.001,
                        help="Learning rate (default: 0.001)")
    parser.add_argument("--batch-size",type=int,   default=256,
                        help="Mini-batch size (default: 256)")
    parser.add_argument("--games-dir", type=Path, nargs='+',
                        default=[_ROOT / "data" / "games"],
                        help="One or more directories containing *.jsonl files")
    parser.add_argument("--output",    type=Path,
                        default=_ROOT / "data" / "value_net.npz",
                        help="Output .npz path")
    parser.add_argument("--decisive-only", action="store_true",
                        help="Exclude draw/unknown games; train only on win/loss outcomes")
    args = parser.parse_args()

    dirs_str = ", ".join(str(d) for d in args.games_dir)
    print(f"Loading game records from {dirs_str} ...")
    X, y, n_files = extract_samples(args.games_dir, decisive_only=args.decisive_only)
    N = len(X)
    print(f"  {N:,} unique positions (FEN-deduplicated) from {n_files:,} game files.")

    label_dist = {
        "+1 (win)":  int((y > 0.5).sum()),
        " 0 (draw)": int((y == 0.0).sum()),
        "-1 (loss)": int((y < -0.5).sum()),
    }
    for k, v in label_dist.items():
        print(f"  {k}: {v} ({100*v/N:.1f}%)")

    net = ValueNet()
    print(f"\nTraining  epochs={args.epochs}  lr={args.lr}  batch={args.batch_size} ...")
    losses = net.train(X, y, epochs=args.epochs, batch_size=args.batch_size,
                       lr=args.lr, verbose=True)
    print(f"\nFinal loss: {losses[-1]:.5f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    net.save(args.output)
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
