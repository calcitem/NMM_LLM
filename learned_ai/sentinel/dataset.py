"""learned_ai/sentinel/dataset.py — move-level sentinel dataset.

Reads played-game logs (``data/games/*.jsonl``), replays each game by applying
moves to a BoardState reconstructed from ``board_fen_before``, and for every
position generates ONE training example per legal move. Each example is a
per-move feature vector plus a ``move_quality`` label in [0, 1] from the mover's
perspective (see :mod:`learned_ai.sentinel.labels`).

Generating one example per legal move (rather than one per played position)
multiplies the data by the average branching factor and makes the sentinel an
explicit move scorer.

A processed dataset can be saved to / loaded from a single ``.npz`` file.
"""

from __future__ import annotations

import glob
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset as _TorchDataset
except Exception:  # pragma: no cover - torch is a declared dependency
    torch = None  # type: ignore

    class _TorchDataset:  # minimal fallback so the module imports without torch
        pass

from game.board import BoardState
from learned_ai.sentinel.feature_builder import FEATURE_DIM, build_move_features
from learned_ai.sentinel.labels import MoveExample, label_move

logger = logging.getLogger(__name__)

# Cap candidates per position to keep replay bounded on wide placement positions.
_MAX_CANDIDATES_PER_POS = 30


# ── Game-log parsing / replay ──────────────────────────────────────────────────

def _board_from_fen_before(fen: str) -> Optional[BoardState]:
    try:
        return BoardState.from_fen_string(fen)
    except Exception:
        return None


def _enumerate_legal_moves(board: BoardState, player: str) -> List[Dict[str, Any]]:
    """All legal apply-move dicts {from,to,capture} for ``player`` on ``board``.

    Mill-closing moves expand into one entry per legal capture target. Uses only
    BoardState's own move generation — no game-logic changes here.
    """
    moves: List[Dict[str, Any]] = []
    if board.phase == "place":
        base = [{"from": None, "to": tgt} for tgt in board.legal_placements(player)]
    else:
        base = [{"from": src, "to": tgt} for src, tgt in board.legal_moves(player)]

    for mv in base:
        mv = {"from": mv.get("from"), "to": mv.get("to"), "capture": None}
        try:
            after = board.apply_move(mv)
            closes = mv["to"] is not None and after.is_mill(mv["to"], player)
        except Exception:
            closes = False
        if closes:
            caps = board.legal_captures(player)
            if caps:
                for cap in caps:
                    moves.append({"from": mv["from"], "to": mv["to"], "capture": cap})
                continue
        moves.append(mv)
    return moves


def _heuristic_scores(board: BoardState, moves: List[Dict[str, Any]], player: str) -> List[float]:
    """Heuristic score (mover's perspective) of the board after each move.

    Falls back to a flat 0.0 list when the heuristic is unavailable so the
    dataset still builds (weak labels then default to 0.5).
    """
    try:
        from ai.heuristics import evaluate
    except Exception:
        return [0.0] * len(moves)
    scores: List[float] = []
    for mv in moves:
        try:
            after = board.apply_move(mv)
            scores.append(float(evaluate(after, player, strength_mode=True)))
        except Exception:
            scores.append(0.0)
    return scores


def _normalise_scores(scores: List[float]) -> List[float]:
    """Min-max normalise heuristic scores across candidates into [0, 1]."""
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi <= lo:
        return [0.5] * len(scores)
    span = hi - lo
    return [(s - lo) / span for s in scores]


def _ranks_desc(scores: List[float]) -> List[int]:
    """Return rank of each move (0 = highest score)."""
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    rank = [0] * len(scores)
    for r, idx in enumerate(order):
        rank[idx] = r
    return rank


def examples_from_position(
    board: BoardState,
    player: str,
    ply: int,
    db=None,
) -> List[MoveExample]:
    """Build one MoveExample per legal move at ``board`` for ``player``."""
    moves = _enumerate_legal_moves(board, player)
    if not moves:
        return []
    if len(moves) > _MAX_CANDIDATES_PER_POS:
        moves = moves[:_MAX_CANDIDATES_PER_POS]

    raw_scores = _heuristic_scores(board, moves, player)
    norm_scores = _normalise_scores(raw_scores)
    ranks = _ranks_desc(raw_scores)
    n_legal = len(moves)

    # Solved-DB WDL for every legal move (empty list when DB unavailable).
    all_moves: List[Dict[str, Any]] = []
    if db is not None and getattr(db, "is_available", lambda: False)():
        try:
            all_moves = db.query_all_moves(board, player)
        except Exception:
            all_moves = []
    wdl_by_key: Dict[tuple, str] = {}
    for entry in all_moves:
        mv = entry.get("move", {})
        wdl_by_key[(mv.get("from"), mv.get("to"), mv.get("capture"))] = entry.get("wdl", "unknown")

    examples: List[MoveExample] = []
    for i, mv in enumerate(moves):
        ctx = {
            "all_moves": all_moves,
            "heuristic_rank": ranks[i],
            "n_legal": n_legal,
            "heuristic_score_norm": norm_scores[i],
        }
        try:
            feat = build_move_features(board, mv, player, ctx)
        except Exception:
            continue
        wdl = wdl_by_key.get((mv.get("from"), mv.get("to"), mv.get("capture")))
        quality, weight, source = label_move(wdl, heuristic_score_norm=norm_scores[i])
        examples.append(
            MoveExample(
                features=np.asarray(feat, dtype=np.float32),
                move_quality=float(quality),
                training_weight=float(weight),
                supervision_source=source,
                ply=ply,
                move_notation=f"{mv.get('from') or ''}-{mv.get('to') or ''}",
                meta={"player": player},
            )
        )
    return examples


def examples_from_game(record: Dict[str, Any], db=None) -> List[MoveExample]:
    """Replay one game record and return per-move MoveExamples for every ply."""
    moves = record.get("moves") or []
    out: List[MoveExample] = []
    for ply, log_move in enumerate(moves):
        fen = log_move.get("board_fen_before")
        if not fen:
            continue
        board = _board_from_fen_before(fen)
        if board is None:
            continue
        player = log_move.get("color") or getattr(board, "turn", "W")
        try:
            out.extend(examples_from_position(board, player, ply, db=db))
        except Exception as exc:
            logger.debug("[SentinelDataset] position failed at ply %d: %s", ply, exc)
            continue
    return out


# ── Dataset ─────────────────────────────────────────────────────────────────────

class SentinelDataset(_TorchDataset):
    """PyTorch Dataset over labelled move-level examples.

    Each item is ``(feature_tensor[FEATURE_DIM], (quality, weight))``.
    """

    def __init__(self, examples: Optional[List[MoveExample]] = None) -> None:
        self.examples: List[MoveExample] = list(examples or [])

    # ----- construction --------------------------------------------------------

    @classmethod
    def load_from_games(
        cls,
        game_dir: str,
        db=None,
        config=None,
        limit: Optional[int] = None,
        paths: Optional[List[str]] = None,
        decisive_only: bool = False,
    ) -> "SentinelDataset":
        """Build a dataset by replaying ``*.jsonl`` files in ``game_dir``.

        decisive_only: skip games with no winner (draws/unknowns).
        """
        if paths is None:
            paths = sorted(glob.glob(os.path.join(game_dir, "**", "*.jsonl"), recursive=True))
        if limit is not None:
            paths = paths[:limit]
        all_examples: List[MoveExample] = []
        skipped = 0
        for path in paths:
            for record in _iter_game_records(path):
                if decisive_only and record.get("winner") is None:
                    skipped += 1
                    continue
                try:
                    all_examples.extend(examples_from_game(record, db=db))
                except Exception as exc:
                    logger.warning("[SentinelDataset] failed on %s: %s", path, exc)
        if decisive_only and skipped:
            logger.info("[SentinelDataset] skipped %d draw/unknown games (decisive_only)", skipped)
        logger.info(
            "[SentinelDataset] loaded %d move examples from %d files",
            len(all_examples), len(paths),
        )
        return cls(all_examples)

    @classmethod
    def game_level_split(
        cls,
        game_dir: str,
        val_fraction: float = 0.15,
        db=None,
        config=None,
        seed: int = 42,
        limit: Optional[int] = None,
        decisive_only: bool = False,
    ) -> "Tuple[SentinelDataset, SentinelDataset]":
        """Return (train, val) split at the game-file level (no per-ply leakage)."""
        import random as _random
        rng = _random.Random(seed)
        all_paths = sorted(glob.glob(os.path.join(game_dir, "**", "*.jsonl"), recursive=True))
        if limit is not None:
            all_paths = all_paths[:limit]
        shuffled = list(all_paths)
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * val_fraction))
        val_paths = shuffled[:n_val]
        train_paths = shuffled[n_val:]
        train_ds = cls.load_from_games(game_dir, db=db, config=config, paths=train_paths, decisive_only=decisive_only)
        val_ds = cls.load_from_games(game_dir, db=db, config=config, paths=val_paths, decisive_only=decisive_only)
        logger.info(
            "[SentinelDataset] game-level split: %d train games / %d val games → %d / %d examples",
            len(train_paths), len(val_paths), len(train_ds), len(val_ds),
        )
        return train_ds, val_ds

    # ----- Dataset protocol ----------------------------------------------------

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        feat = ex.features.astype(np.float32)
        if torch is not None:
            feat = torch.from_numpy(feat)
        return feat, (float(ex.move_quality), float(ex.training_weight))

    # ----- persistence ---------------------------------------------------------

    def save_to_disk(self, path: str) -> None:
        """Persist all examples to a single ``.npz`` file."""
        if not self.examples:
            np.savez_compressed(path, features=np.zeros((0, FEATURE_DIM), np.float32))
            return
        feats = np.stack([e.features for e in self.examples]).astype(np.float32)
        np.savez_compressed(
            path,
            features=feats,
            move_quality=np.array([e.move_quality for e in self.examples], np.float32),
            training_weight=np.array([e.training_weight for e in self.examples], np.float32),
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
        examples: List[MoveExample] = []
        for i in range(n):
            examples.append(
                MoveExample(
                    features=feats[i].astype(np.float32),
                    move_quality=float(data["move_quality"][i]),
                    training_weight=float(data["training_weight"][i]),
                    supervision_source=str(data["supervision_source"][i]),
                    ply=int(data["ply"][i]),
                )
            )
        return cls(examples)

    # ----- diagnostics ---------------------------------------------------------

    def quality_distribution(self) -> Dict[str, int]:
        """Bucket examples into win / draw / loss / other by move_quality."""
        dist = {"win": 0, "draw": 0, "loss": 0, "other": 0}
        for e in self.examples:
            q = e.move_quality
            if q >= 0.99:
                dist["win"] += 1
            elif abs(q - 0.5) < 1e-3:
                dist["draw"] += 1
            elif q <= 0.01:
                dist["loss"] += 1
            else:
                dist["other"] += 1
        return dist

    def source_distribution(self) -> Dict[str, int]:
        dist: Dict[str, int] = {}
        for e in self.examples:
            dist[e.supervision_source] = dist.get(e.supervision_source, 0) + 1
        return dist


def _iter_game_records(path: str):
    """Yield game-record dicts from a JSONL file (whole-object or line-delimited)."""
    try:
        with open(path) as f:
            content = f.read().strip()
    except Exception as exc:
        logger.warning("[SentinelDataset] cannot read %s: %s", path, exc)
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


def collate_examples(batch: List[Tuple[Any, Tuple[float, float]]]):
    """Collate fn for DataLoader: stack features, qualities, and weights."""
    if torch is None:  # pragma: no cover
        raise RuntimeError("torch required for collate_examples")
    feats = torch.stack([b[0] for b in batch]).float()
    quality = torch.tensor([b[1][0] for b in batch], dtype=torch.float32)
    weight = torch.tensor([b[1][1] for b in batch], dtype=torch.float32)
    return feats, quality, weight
