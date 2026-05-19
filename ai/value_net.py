"""
ai/value_net.py — Tiny MLP value network for Nine Men's Morris.

Predicts a win-probability-style value for a board state from a given color's
perspective.  Uses numpy only — no deep-learning framework required.

Architecture: 79-input → 128 ReLU → 64 ReLU → 1 tanh
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np

from game.board import BoardState, POSITIONS

_INPUT_DIM = 24 * 3 + 7   # 72 board one-hots + 7 metadata scalars = 79
_H1 = 128
_H2 = 64


def board_to_features(board: BoardState, color: str) -> np.ndarray:
    """
    Encode board from color's perspective into a (_INPUT_DIM,) float32 vector.

    Positions are encoded as own / opponent / empty one-hots so the same
    network weights handle both White and Black without sign conventions.
    """
    opp = "B" if color == "W" else "W"
    feats = np.zeros(_INPUT_DIM, dtype=np.float32)
    idx = 0
    for pos in POSITIONS:
        v = board.positions[pos]
        if v == color:
            feats[idx] = 1.0
        elif v == opp:
            feats[idx + 1] = 1.0
        else:
            feats[idx + 2] = 1.0
        idx += 3
    # Metadata (indices 72–78)
    own_placed = board.pieces_placed[color]
    opp_placed = board.pieces_placed[opp]
    feats[72] = 1.0 if board.turn == color else 0.0
    feats[73] = own_placed / 9.0
    feats[74] = opp_placed / 9.0
    feats[75] = board.pieces_on_board[color] / 9.0
    feats[76] = board.pieces_on_board[opp] / 9.0
    # pieces captured BY color = opponent pieces removed
    feats[77] = (opp_placed - board.pieces_on_board[opp]) / 9.0
    feats[78] = (own_placed - board.pieces_on_board[color]) / 9.0
    return feats


class ValueNet:
    """
    Two-hidden-layer MLP.  Output is in (-1, 1): positive means color wins.

    Weights are plain numpy arrays; the model is ~33 KB on disk.
    """

    def __init__(self) -> None:
        rng = np.random.default_rng(42)
        # He initialisation for ReLU layers
        self.W1: np.ndarray = (rng.standard_normal((_H1, _INPUT_DIM)) *
                               math.sqrt(2.0 / _INPUT_DIM)).astype(np.float32)
        self.b1: np.ndarray = np.zeros(_H1, dtype=np.float32)
        self.W2: np.ndarray = (rng.standard_normal((_H2, _H1)) *
                               math.sqrt(2.0 / _H1)).astype(np.float32)
        self.b2: np.ndarray = np.zeros(_H2, dtype=np.float32)
        self.W3: np.ndarray = (rng.standard_normal((1, _H2)) *
                               math.sqrt(2.0 / _H2)).astype(np.float32)
        self.b3: np.ndarray = np.zeros(1, dtype=np.float32)

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, board: BoardState, color: str) -> float:
        """Return value in (-1, 1) from color's perspective."""
        x = board_to_features(board, color).reshape(1, -1)
        return float(self._forward(x).ravel()[0])

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """X: (N, _INPUT_DIM) float32 → (N,) float32 in (-1, 1)."""
        return self._forward(X).ravel()

    def _forward(self, X: np.ndarray) -> np.ndarray:
        h1  = np.maximum(0.0, X  @ self.W1.T + self.b1)
        h2  = np.maximum(0.0, h1 @ self.W2.T + self.b2)
        out = np.tanh(h2 @ self.W3.T + self.b3)
        return out

    # ── Training ─────────────────────────────────────────────────────────────

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 30,
        batch_size: int = 256,
        lr: float = 0.001,
        verbose: bool = False,
    ) -> list[float]:
        """
        Train in-place with mini-batch SGD and MSE loss.

        Parameters
        ----------
        X : (N, _INPUT_DIM) float32
        y : (N,) float32  in [-1, 1]
        Returns per-epoch mean loss list.
        """
        losses: list[float] = []
        rng = np.random.default_rng()
        N = len(X)
        for ep in range(epochs):
            order = rng.permutation(N)
            X, y = X[order], y[order]
            ep_loss = 0.0
            steps = 0
            for start in range(0, N, batch_size):
                xb = X[start:start + batch_size]
                yb = y[start:start + batch_size].reshape(-1, 1)
                B  = len(xb)

                # Forward
                h1   = np.maximum(0.0, xb @ self.W1.T + self.b1)   # (B, H1)
                h2   = np.maximum(0.0, h1 @ self.W2.T + self.b2)   # (B, H2)
                pred = np.tanh(h2 @ self.W3.T + self.b3)             # (B, 1)

                err  = pred - yb                                      # (B, 1)
                ep_loss += float(np.mean(err ** 2))
                steps += 1

                # Backprop through tanh output
                d3  = (2.0 / B) * err * (1.0 - pred ** 2)           # (B, 1)
                dW3 = d3.T @ h2                                       # (1, H2)
                db3 = d3.sum(axis=0)

                # Layer 2 (ReLU)
                dh2  = d3 @ self.W3                                   # (B, H2)
                dh2 *= (h2 > 0).astype(np.float32)
                dW2  = dh2.T @ h1                                     # (H2, H1)
                db2  = dh2.sum(axis=0)

                # Layer 1 (ReLU)
                dh1  = dh2 @ self.W2                                  # (B, H1)
                dh1 *= (h1 > 0).astype(np.float32)
                dW1  = dh1.T @ xb                                     # (H1, input)
                db1  = dh1.sum(axis=0)

                self.W3 -= lr * dW3;  self.b3 -= lr * db3
                self.W2 -= lr * dW2;  self.b2 -= lr * db2
                self.W1 -= lr * dW1;  self.b1 -= lr * db1

            ep_mean = ep_loss / max(steps, 1)
            losses.append(ep_mean)
            if verbose:
                print(f"  epoch {ep+1:3d}/{epochs}  loss={ep_mean:.5f}")
        return losses

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        np.savez(str(path), W1=self.W1, b1=self.b1,
                 W2=self.W2, b2=self.b2, W3=self.W3, b3=self.b3)

    @classmethod
    def load(cls, path: str | Path) -> "ValueNet":
        obj = cls.__new__(cls)
        data = np.load(str(path))
        obj.W1, obj.b1 = data["W1"], data["b1"]
        obj.W2, obj.b2 = data["W2"], data["b2"]
        obj.W3, obj.b3 = data["W3"], data["b3"]
        return obj

    @classmethod
    def load_if_exists(cls, path: str | Path) -> Optional["ValueNet"]:
        """Return a loaded network if the file exists, else None."""
        p = Path(path)
        if p.exists():
            return cls.load(p)
        return None
