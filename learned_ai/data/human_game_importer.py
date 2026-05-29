"""Stub for ingesting human game logs into the replay buffer.

The existing game records (stored under data/games/) follow the
GameEngine.game_record JSON schema. A future implementation will parse
those records, replay each move through BoardState to recover the
intermediate states, encode them, and emit Transition objects.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from learned_ai.training.replay_buffer import Transition


def import_game_file(path: str) -> List[Transition]:
    """Placeholder: returns an empty list. Implement when human data is available."""
    _ = Path(path)
    return []


def import_directory(directory: str) -> Iterable[Transition]:
    """Yield Transitions parsed from every JSON game record in *directory*.

    This is a stub. Tested only for import-doesn't-crash; concrete parsing is
    deferred to Stage 5 of the curriculum.
    """
    _ = Path(directory)
    return iter([])
