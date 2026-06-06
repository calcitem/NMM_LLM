"""learned_ai/sentinel/dataset.py — trajectory-supervised sentinel dataset.

Reads played-game logs (``data/games/*.jsonl``), replays each game by applying
moves to a BoardState (reconstructed from ``board_fen_before``), builds a
per-ply decision context, attaches supervision via
``labels.backward_label_trajectory``, and exposes the result as a PyTorch
Dataset of ``(feature_tensor[120], label_dict)``.

A processed dataset can be saved to / loaded from a single ``.npz`` file for
reproducible runs without re-replaying every game.
"""

from __future__ import annotations

import glob
import json
import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset as _TorchDataset
except Exception:  # pragma: no cover - torch is a declared dependency
    torch = None  # type: ignore

    class _TorchDataset:  # minimal fallback so the module imports without torch
        pass

from game.board import MILLS, BoardState
from learned_ai.sentinel.feature_builder import FEATURE_DIM, build_features
from learned_ai.sentinel.labels import LabelledExample, backward_label_trajectory

logger = logging.getLogger(__name__)

# Target keys produced per example (must match SentinelNet heads).
TARGET_KEYS = (
    "mistake_risk",
    "opportunity_score",
    "trajectory_value_delta",
    "turning_point_confidence",
    "weight",
)


# ── Game-log parsing / replay ──────────────────────────────────────────────────

def _board_from_fen_before(fen: str) -> Optional[BoardState]:
    """Reconstruct a BoardState from a ``board_fen_before`` string.

    Format: '<24 chars>|<turn>|<W_placed>|<B_placed>'. Returns None on malformed
    input so a single bad ply cannot abort a whole game's processing.
    """
    try:
        return BoardState.from_fen_string(fen)
    except Exception:
        return None


def _move_dict_from_log(move: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the {from,to,capture} apply-move dict from a logged move."""
    return {
        "from": move.get("from"),
        "to": move.get("to"),
        "capture": move.get("capture"),
    }


def _closes_mill(board: BoardState, move: Dict[str, Any]) -> bool:
    """True if applying ``move`` completes a mill for the mover."""
    try:
        after = board.apply_move(move)
        to = move.get("to")
        if to is None:
            return False
        return after.is_mill(to, board.turn)
    except Exception:
        return False


def _opens_mill_threat(board: BoardState, move: Dict[str, Any]) -> bool:
    """True if after the move the mover has a new 2-of-3 mill line with an empty
    third square (an immediate threat to close next turn)."""
    try:
        after = board.apply_move(move)
        color = board.turn
        for ml in MILLS:
            vals = [after.positions[p] for p in ml]
            if vals.count(color) == 2 and vals.count("") == 1:
                return True
        return False
    except Exception:
        return False


def _reduces_own_mobility(board: BoardState, move: Dict[str, Any]) -> bool:
    """True if the move reduces the mover's own legal-move count."""
    try:
        color = board.turn
        before = len(board.legal_moves(color)) + len(board.legal_placements(color))
        after_board = board.apply_move(move)
        after = len(after_board.legal_moves(color)) + len(after_board.legal_placements(color))
        return after < before
    except Exception:
        return False


def _build_move_context(
    board: BoardState,
    move: Dict[str, Any],
    log_move: Dict[str, Any],
    trajectory_scores: List[float],
    game_source: str,
) -> Dict[str, Any]:
    """Assemble the move_context dict used by feature_builder + labels.

    Game logs store only the played move (no full candidate list), so the
    candidate list here is a single-entry list built from the logged move. The
    schema still supports richer candidate lists from enriched self-play logs.
    """
    score = log_move.get("game_ai_score", 0.0)
    candidates = [{"move": move, "score": score, "type": log_move.get("type")}]
    return {
        "candidates": candidates,
        "chosen_rank": 0,
        "closes_mill": _closes_mill(board, move),
        "opens_mill_threat": _opens_mill_threat(board, move),
        "reduces_own_mobility": _reduces_own_mobility(board, move),
        "trajectory_scores": list(trajectory_scores[-4:]),
        "game_source": game_source,
        "color": log_move.get("color", board.turn),
        "was_blunder": bool(log_move.get("was_blunder", False)),
        "game_ai_score": score,
    }


def _game_source(record: Dict[str, Any]) -> str:
    """Classify a game as human-vs-ai or ai-vs-ai from its metadata."""
    src = record.get("game_source")
    if src in ("human_vs_ai", "ai_vs_ai"):
        return src
    return "human_vs_ai" if record.get("human_color") else "ai_vs_ai"


def examples_from_game(record: Dict[str, Any], db=None,
                       backward_decay: Optional[Sequence[float]] = None
                       ) -> List[LabelledExample]:
    """Replay one game record and return its LabelledExamples."""
    moves = record.get("moves") or []
    game_source = _game_source(record)

    states: List[BoardState] = []
    features: List[np.ndarray] = []
    contexts: List[Dict[str, Any]] = []
    traj_scores: List[float] = []

    for log_move in moves:
        fen = log_move.get("board_fen_before")
        if not fen:
            continue
        board = _board_from_fen_before(fen)
        if board is None:
            continue
        mv = _move_dict_from_log(log_move)
        if mv.get("to") is None and mv.get("from") is None:
            continue
        ctx = _build_move_context(board, mv, log_move, traj_scores, game_source)
        try:
            feat = build_features(board, ctx)
        except Exception:
            continue
        states.append(board)
        features.append(feat)
        contexts.append(ctx)
        sc = log_move.get("game_ai_score")
        if isinstance(sc, (int, float)):
            traj_scores.append(float(sc))

    if not states:
        return []
    return backward_label_trajectory(
        record, states, features, contexts, db=db, backward_decay=backward_decay
    )


# ── Dataset ─────────────────────────────────────────────────────────────────────

class SentinelDataset(_TorchDataset):
    """PyTorch Dataset over labelled sentinel examples.

    Each item is ``(feature_tensor[120], label_dict)`` where ``label_dict`` has
    the keys in :data:`TARGET_KEYS`. Also stores the categorical label and
    supervision source per example for diagnostics.
    """

    def __init__(self, examples: Optional[List[LabelledExample]] = None) -> None:
        self.examples: List[LabelledExample] = list(examples or [])

    # ----- construction --------------------------------------------------------

    @classmethod
    def load_from_games(
        cls,
        game_dir: str,
        db=None,
        config=None,
        limit: Optional[int] = None,
    ) -> "SentinelDataset":
        """Build a dataset by replaying every ``*.jsonl`` in ``game_dir``.

        Each file is expected to contain a single JSON game object (one object
        per file, as in data/games). Files with one-object-per-line are also
        supported. Malformed files are skipped with a warning.
        """
        backward_decay = getattr(config, "backward_decay", None) if config else None
        paths = sorted(glob.glob(os.path.join(game_dir, "*.jsonl")))
        if limit is not None:
            paths = paths[:limit]
        all_examples: List[LabelledExample] = []
        for path in paths:
            for record in _iter_game_records(path):
                try:
                    all_examples.extend(
                        examples_from_game(record, db=db, backward_decay=backward_decay)
                    )
                except Exception as exc:
                    logger.warning("[SentinelDataset] failed on %s: %s", path, exc)
        logger.info(
            "[SentinelDataset] loaded %d examples from %d files in %s",
            len(all_examples), len(paths), game_dir,
        )
        return cls(all_examples)

    # ----- Dataset protocol ----------------------------------------------------

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        feat = ex.state_features.astype(np.float32)
        if torch is not None:
            feat = torch.from_numpy(feat)
        label = ex.target_dict()
        return feat, label

    # ----- persistence ---------------------------------------------------------

    def save_to_disk(self, path: str) -> None:
        """Persist all examples to a single ``.npz`` file."""
        if not self.examples:
            np.savez_compressed(path, features=np.zeros((0, FEATURE_DIM), np.float32))
            return
        feats = np.stack([e.state_features for e in self.examples]).astype(np.float32)
        np.savez_compressed(
            path,
            features=feats,
            label=np.array([e.label for e in self.examples], dtype=object),
            turning_point_confidence=np.array(
                [e.turning_point_confidence for e in self.examples], np.float32),
            value_delta=np.array([e.value_delta for e in self.examples], np.float32),
            mistake_risk=np.array([e.mistake_risk for e in self.examples], np.float32),
            opportunity_score=np.array(
                [e.opportunity_score for e in self.examples], np.float32),
            training_weight=np.array(
                [e.training_weight for e in self.examples], np.float32),
            supervision_source=np.array(
                [e.supervision_source for e in self.examples], dtype=object),
            ply=np.array([e.ply for e in self.examples], np.int64),
        )

    @classmethod
    def load_from_disk(cls, path: str) -> "SentinelDataset":
        """Reconstruct a dataset previously written by :meth:`save_to_disk`."""
        data = np.load(path, allow_pickle=True)
        feats = data["features"]
        n = feats.shape[0]
        examples: List[LabelledExample] = []
        for i in range(n):
            examples.append(
                LabelledExample(
                    state_features=feats[i].astype(np.float32),
                    label=str(data["label"][i]),
                    turning_point_confidence=float(data["turning_point_confidence"][i]),
                    value_delta=float(data["value_delta"][i]),
                    mistake_risk=float(data["mistake_risk"][i]),
                    opportunity_score=float(data["opportunity_score"][i]),
                    training_weight=float(data["training_weight"][i]),
                    supervision_source=str(data["supervision_source"][i]),
                    ply=int(data["ply"][i]),
                )
            )
        return cls(examples)

    # ----- diagnostics ---------------------------------------------------------

    def class_distribution(self) -> Dict[str, int]:
        dist: Dict[str, int] = {}
        for e in self.examples:
            dist[e.label] = dist.get(e.label, 0) + 1
        return dist

    def source_distribution(self) -> Dict[str, int]:
        dist: Dict[str, int] = {}
        for e in self.examples:
            dist[e.supervision_source] = dist.get(e.supervision_source, 0) + 1
        return dist


def _iter_game_records(path: str):
    """Yield game-record dicts from a JSONL file.

    Supports both "one big JSON object per file" and "one object per line".
    """
    try:
        with open(path) as f:
            content = f.read().strip()
    except Exception as exc:
        logger.warning("[SentinelDataset] cannot read %s: %s", path, exc)
        return
    if not content:
        return
    # Try whole-file JSON first (data/games stores one object per file).
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
    # Fall back to line-delimited JSON.
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


def collate_examples(batch: List[Tuple[Any, Dict[str, float]]]):
    """Collate fn for DataLoader: stack features and targets into tensors."""
    if torch is None:  # pragma: no cover
        raise RuntimeError("torch required for collate_examples")
    feats = torch.stack([b[0] for b in batch]).float()
    targets = {
        k: torch.tensor([b[1][k] for b in batch], dtype=torch.float32)
        for k in TARGET_KEYS
    }
    return feats, targets
