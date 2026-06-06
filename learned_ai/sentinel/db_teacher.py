"""learned_ai/sentinel/db_teacher.py — read-only external solved-DB teacher adapter.

This adapter targets the **Malom ultra-strong NMM database** (ggevay/malom,
GPL-3 by Gabor E. Gevay and Gabor Danner) stored at the configured
``external_db_path`` (default: ``/mnt/windows/NMM_DB/strong/``).  It is used
as a *training-time teacher only* to label observed game trajectories with
ground-truth WDL.

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

MALOM DATABASE FORMAT
---------------------
The Malom .sec2 files store solved WDL for each board sector (W, B, WF, BF)
where W/B = pieces on board, WF/BF = pieces still to place.  The hash function
is a two-part combinatorial index over canonical symmetry orbits of White pieces
and compressed Black piece positions.  Full details are in ai/malom_db.py.

The previous stub (targeting database.dat + preCalculatedVars.dat) has been
replaced by the working MalomDB adapter in ai/malom_db.py.
  3. Attempts to read a small header from ``preCalculatedVars.dat`` and records
     its byte length + first bytes as ``self.format_probe`` for later analysis.
  4. The MalomDB adapter (ai/malom_db.py) implements the full hash function and
     decodes the .sec2 entries directly.  is_available() returns True when .sec2
     files are found; query() returns WDL results for any position in the DB.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from ai.malom_db import MalomDB as _MalomDB
except ImportError:
    _MalomDB = None  # type: ignore[assignment,misc]


class ExternalSolvedDB:
    """Read-only adapter for the Malom ultra-strong NMM database."""

    def __init__(self, db_path: str = "", enabled: bool = True) -> None:
        """Open the Malom DB at db_path. Never raises.

        Parameters
        ----------
        db_path : path to the directory containing std_*.sec2 files
                  (e.g. /mnt/windows/NMM_DB/strong).
                  Empty string => unavailable.
        enabled : when False the adapter is forced unavailable regardless of path
                  (used to honour ``external_db_enabled: false`` in config).
        """
        self.db_path: str = db_path or ""
        self._enabled = bool(enabled)
        self._warned = False
        self._malom: Optional[_MalomDB] = None  # type: ignore[type-arg]
        self.db_dir: Optional[Path] = None
        self.format_probe: Dict[str, Any] = {}

        try:
            self._probe()
        except Exception as exc:  # absolutely never fatal
            logger.warning("[ExternalSolvedDB] probe failed (non-fatal): %s", exc)

    # ── Probing ──────────────────────────────────────────────────────────────

    def _probe(self) -> None:
        if not self._enabled or not self.db_path:
            return

        p = Path(self.db_path)
        if p.is_file():
            self.db_dir = p.parent
        elif p.is_dir():
            self.db_dir = p
        else:
            return

        if _MalomDB is None:
            logger.warning("[ExternalSolvedDB] ai.malom_db not importable; DB unavailable")
            return

        self._malom = _MalomDB(self.db_dir)
        self.format_probe = {
            "db_dir": str(self.db_dir),
            "available": self._malom.is_available(),
        }
        if self._malom.is_available():
            logger.info("[ExternalSolvedDB] Malom DB ready at %s", self.db_dir)

    # ── Availability ───────────────────────────────────────────────────────────

    @property
    def _available(self) -> bool:
        return self._malom is not None and self._malom.is_available()

    def is_available(self) -> bool:
        """True when the Malom DB files are present and queryable."""
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

    # ── Lookup ──────────────────────────────────────────────────────────────────

    def _lookup(self, board) -> Optional[str]:
        """Return "W"|"L"|"D" for the current mover, or None."""
        if self._malom is None:
            return None
        try:
            result = self._malom.query(board)
            return result["outcome"] if result else None
        except Exception as exc:
            logger.debug("[ExternalSolvedDB] lookup error: %s", exc)
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
        """Release cached sector data."""
        if self._malom is not None:
            self._malom.close()

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
