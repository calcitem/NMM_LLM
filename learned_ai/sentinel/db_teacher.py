"""learned_ai/sentinel/db_teacher.py — read-only external solved-DB teacher adapter.

This adapter targets the **external 12 GB solved database** that lives on the
user's PC (default path: ``/mnt/windows/NMM_DB/Entire DB/`` with files
``database.dat`` + ``preCalculatedVars.dat``). It is used as a *training-time
teacher only* to label observed game trajectories with ground-truth WDL.

IMPORTANT
---------
This is NOT the project's internal ``ai/endgame_solved_db.py`` (the engine's own
retrograde endgame DB). The two must never be merged. This adapter is read-only
and is designed to mirror that module's query surface so training code can swap
between internal and external teachers.

GRACEFUL UNAVAILABILITY (hard requirement)
------------------------------------------
The external DB is not present in the repo and frequently absent at runtime.
Every public method is therefore non-fatal:
  * construction never raises, even on a bad/missing path;
  * when the DB is unavailable, ``is_available()`` returns False and all
    ``query_*`` methods return ``None`` (or a list of ``None`` for trajectories);
  * a single clear warning is logged the first time an unavailable DB is queried.

UNKNOWN BINARY FORMAT — TODO
----------------------------
The on-disk layout of ``database.dat`` + ``preCalculatedVars.dat`` is not
documented anywhere in this repo, so this adapter cannot decode positions yet.
What it *does* do today:
  1. Resolves the DB directory (or the parent dir of a file path).
  2. Probes for ``database.dat`` and ``preCalculatedVars.dat``.
  3. Attempts to read a small header from ``preCalculatedVars.dat`` and records
     its byte length + first bytes as ``self.format_probe`` for later analysis.
  4. Because the index scheme (how a BoardState maps to a record offset) is
     unknown, ``_lookup()`` returns ``None`` — i.e. the adapter behaves as a
     fully graceful stub that never fabricates supervision.

TODO(when real format is known):
  - Decode ``preCalculatedVars.dat`` to learn record size, count, and the
    combinatorial / hashed index used by ``database.dat``.
  - Implement ``_board_to_record_id(board)`` mirroring the external tool's
    indexing (likely a combinatorial rank similar to ai/endgame_solved_db.py,
    possibly extended to placement/midgame positions).
  - Implement ``_lookup(board)`` to seek into ``database.dat`` and decode the
    packed WDL (and optionally a distance-to-result / move-quality field).
  - Map the decoded value to "W"/"L"/"D" from the side-to-move perspective.
Until then the labelling layer falls back to game-outcome proxy supervision.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DB_FILE = "database.dat"
_VARS_FILE = "preCalculatedVars.dat"
_HEADER_PROBE_BYTES = 64


class ExternalSolvedDB:
    """Read-only adapter for the external solved database (graceful stub)."""

    def __init__(self, db_path: str = "", enabled: bool = True) -> None:
        """Open (probe) the external DB. Never raises.

        Parameters
        ----------
        db_path : path to the DB directory, or to ``database.dat`` itself.
                  Empty string => unavailable.
        enabled : when False the adapter is forced unavailable regardless of path
                  (used to honour ``external_db_enabled: false`` in config).
        """
        self.db_path: str = db_path or ""
        self._enabled = bool(enabled)
        self._available = False
        self._warned = False
        self.db_dir: Optional[Path] = None
        self.database_file: Optional[Path] = None
        self.vars_file: Optional[Path] = None
        self.format_probe: Dict[str, Any] = {}

        try:
            self._probe()
        except Exception as exc:  # absolutely never fatal
            logger.warning("[ExternalSolvedDB] probe failed (non-fatal): %s", exc)
            self._available = False

    # ── Probing ──────────────────────────────────────────────────────────────

    def _probe(self) -> None:
        if not self._enabled or not self.db_path:
            self._available = False
            return

        p = Path(self.db_path)
        if not p.exists():
            self._available = False
            return

        # Accept either a directory or a direct path to database.dat.
        if p.is_file():
            self.db_dir = p.parent
        else:
            self.db_dir = p

        self.database_file = self.db_dir / _DB_FILE
        self.vars_file = self.db_dir / _VARS_FILE

        if not self.database_file.exists():
            self._available = False
            return

        # Attempt to read a small header from preCalculatedVars.dat to record
        # format metadata for future decoding work. Missing vars file is not
        # fatal — we just note it.
        if self.vars_file.exists():
            try:
                size = self.vars_file.stat().st_size
                with open(self.vars_file, "rb") as f:
                    head = f.read(_HEADER_PROBE_BYTES)
                self.format_probe = {
                    "vars_size_bytes": size,
                    "vars_header_hex": head.hex(),
                }
            except Exception as exc:
                self.format_probe = {"vars_read_error": str(exc)}
        else:
            self.format_probe = {"vars_file_missing": True}

        try:
            self.format_probe["database_size_bytes"] = self.database_file.stat().st_size
        except OSError:
            pass

        # The binary index scheme is unknown (see module TODO), so even with the
        # files present we cannot decode records yet. We still mark the DB as
        # "available" only if we could in principle query it — which we cannot
        # today. Keep available False so the labelling layer uses proxies and
        # no fabricated supervision leaks into training.
        self._available = False
        logger.info(
            "[ExternalSolvedDB] Files found at %s but binary format is not yet "
            "decodable; returning None for all queries. Probe: %s",
            self.db_dir, self.format_probe,
        )

    # ── Availability ───────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """True only when the DB can actually answer queries. Currently always
        False (format undecoded) — but the method is the single switch the rest
        of the pipeline checks, so it stays correct once decoding lands."""
        return self._available

    def _warn_unavailable_once(self) -> None:
        if not self._warned:
            self._warned = True
            logger.warning(
                "[ExternalSolvedDB] unavailable (path=%r, enabled=%s) — all "
                "queries return None; training falls back to outcome-proxy "
                "supervision.",
                self.db_path, self._enabled,
            )

    # ── Indexing / lookup (stubbed until format is known) ──────────────────────

    def _board_to_record_id(self, board) -> Optional[int]:  # noqa: ARG002
        """Map a BoardState to a record id in database.dat.

        TODO: implement once the external DB's index scheme is documented.
        Returns None today so no fabricated supervision is produced.
        """
        return None

    def _lookup(self, board) -> Optional[str]:  # noqa: ARG002
        """Decode a WDL result for ``board`` from database.dat.

        TODO: seek + decode packed WDL once the format is known.
        Returns None today (graceful stub).
        """
        return None

    # ── Public query surface (mirrors ai/endgame_solved_db.py style) ────────────

    def query_state(self, board) -> Optional[str]:
        """Return "W" | "L" | "D" for the side to move, or None if unavailable."""
        if not self._available:
            self._warn_unavailable_once()
            return None
        return self._lookup(board)

    def query(self, board) -> Optional[str]:
        """Alias of ``query_state`` matching EndgameSolvedDB.query()."""
        return self.query_state(board)

    def query_move_quality(self, board, move: Dict[str, Any]) -> Optional[float]:
        """Quality delta of ``move`` from ``board``: + good, - bad, None unknown.

        Computed (once decodable) as WDL(before) vs WDL(after move) from the
        mover's perspective. Returns None while the format is undecoded.
        """
        if not self._available:
            self._warn_unavailable_once()
            return None
        try:
            before = self._lookup(board)
            after_board = board.apply_move(move)
            after = self._lookup(after_board)
        except Exception:
            return None
        if before is None or after is None:
            return None
        # Mover's perspective: after applying the move it is the opponent's turn,
        # so an opponent "L" (they lose) is good for the mover.
        rank = {"W": 1.0, "D": 0.0, "L": -1.0}
        before_v = rank.get(before, 0.0)
        after_opp_v = rank.get(after, 0.0)
        after_mover_v = -after_opp_v
        return after_mover_v - before_v

    def query_trajectory(self, states: List[Any]) -> List[Optional[str]]:
        """Return a WDL (or None) for each state in a trajectory.

        Always returns a list of the same length as ``states`` (all None when
        unavailable) so callers can zip without length checks.
        """
        if not self._available:
            self._warn_unavailable_once()
            return [None] * len(states)
        out: List[Optional[str]] = []
        for s in states:
            try:
                out.append(self._lookup(s))
            except Exception:
                out.append(None)
        return out

    def close(self) -> None:
        """No-op (no open file handles held). Present for interface parity."""
        return None

    def __repr__(self) -> str:
        return (
            f"ExternalSolvedDB(path={self.db_path!r}, enabled={self._enabled}, "
            f"available={self._available})"
        )


def open_external_db(config) -> ExternalSolvedDB:
    """Convenience constructor from a SentinelConfig.

    Honours both ``external_db_enabled`` and ``external_db_path``. Falls back to
    the ``NMM_EXTERNAL_DB`` environment variable when the config path is empty.
    """
    path = getattr(config, "external_db_path", "") or os.environ.get("NMM_EXTERNAL_DB", "")
    enabled = bool(getattr(config, "external_db_enabled", False))
    return ExternalSolvedDB(db_path=path, enabled=enabled)
