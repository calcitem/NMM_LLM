"""Replay buffer storing (state, action, reward, ...) transitions.

Used by the trainer to mix old experience into mini-batches. The buffer is
a bounded deque under the hood (FIFO eviction when full). Save/load uses
torch.save so we get cheap binary persistence without an extra dep.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch


@dataclass
class Transition:
    state: torch.Tensor          # (state_dim,)
    legal_mask: torch.Tensor     # (action_dim,) bool
    primary_index: int
    capture_index: Optional[int]
    reward: float                # return-to-go for this step
    phase_id: int
    side_to_move: str            # "W" or "B"
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int = 50_000, seed: Optional[int] = None) -> None:
        self.capacity = int(capacity)
        self._buf: deque[Transition] = deque(maxlen=self.capacity)
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self._buf)

    def push(self, t: Transition) -> None:
        self._buf.append(t)

    def extend(self, ts: List[Transition]) -> None:
        for t in ts:
            self._buf.append(t)

    def sample(self, batch_size: int) -> List[Transition]:
        n = min(batch_size, len(self._buf))
        if n == 0:
            return []
        return self._rng.sample(list(self._buf), n)

    def clear(self) -> None:
        self._buf.clear()

    # ---- persistence -------------------------------------------------------

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "capacity": self.capacity,
            "transitions": list(self._buf),
        }
        torch.save(payload, path)

    def load(self, path: str) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.capacity = int(payload.get("capacity", self.capacity))
        self._buf = deque(payload["transitions"], maxlen=self.capacity)
