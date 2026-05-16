"""
ai/endgame_recognizer.py — Endgame phase detection and pattern recognition.

Phase progression:
  opening      — placement phase not yet complete (< 18 pieces placed)
  midgame      — all placed; total on board > active_threshold (default 11)
  endgame      — total on board <= active_threshold
  deep_endgame — total on board <= deep_threshold (default 8)

Patterns detected (heuristic):
  mill_cycle   — a closed mill has a piece that can slide away, enabling
                 repeated open/close to force a capture each cycle
  pincer       — two of the same player's mills both adjoin the same
                 opponent piece, creating a dual capture threat
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from game.board import BoardState

from game.board import ADJACENCY, MILLS
from game.rules import get_game_phase


@dataclass
class EndgameState:
    active: bool            # Total pieces on board <= active_threshold
    deep: bool              # Total pieces on board <= deep_threshold
    phase: str              # "opening" | "midgame" | "endgame" | "deep_endgame"
    total_pieces: int
    pieces_white: int
    pieces_black: int
    mobility_white: int     # Legal moves available to White
    mobility_black: int     # Legal moves available to Black
    zugzwang_risk: bool     # Current player's mobility is far below opponent's
    pattern: Optional[str]  # "mill_cycle" | "pincer" | None
    pattern_notes: str


INACTIVE_ENDGAME = EndgameState(
    active=False,
    deep=False,
    phase="opening",
    total_pieces=18,
    pieces_white=9,
    pieces_black=9,
    mobility_white=0,
    mobility_black=0,
    zugzwang_risk=False,
    pattern=None,
    pattern_notes="",
)


class EndgameRecognizer:
    """
    Tracks board piece counts after every move and classifies the game phase.
    Call update() after each move; use get_current_state() at any time.
    """

    def __init__(
        self,
        active_threshold: int = 11,
        deep_threshold: int = 8,
        zugzwang_threshold: float = 0.4,
    ) -> None:
        self.active_threshold = active_threshold
        self.deep_threshold = deep_threshold
        self.zugzwang_threshold = zugzwang_threshold
        self.current_state: EndgameState = INACTIVE_ENDGAME
        self._announced_endgame: bool = False
        self._announced_deep: bool = False

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self) -> None:
        self.current_state = INACTIVE_ENDGAME
        self._announced_endgame = False
        self._announced_deep = False

    def update(self, board: "BoardState") -> EndgameState:
        """Compute and cache the current endgame state. Call after every move."""
        pieces_w = board.pieces_on_board.get("W", 0)
        pieces_b = board.pieces_on_board.get("B", 0)
        total = pieces_w + pieces_b

        placed_w = board.pieces_placed.get("W", 0)
        placed_b = board.pieces_placed.get("B", 0)
        placement_done = (placed_w >= 9 and placed_b >= 9)

        if not placement_done:
            phase = "opening"
        elif total > self.active_threshold:
            phase = "midgame"
        elif total > self.deep_threshold:
            phase = "endgame"
        else:
            phase = "deep_endgame"

        active = placement_done and total <= self.active_threshold
        deep = placement_done and total <= self.deep_threshold

        mob_w = _mobility(board, "W")
        mob_b = _mobility(board, "B")
        current_mob = mob_w if board.turn == "W" else mob_b
        opp_mob = mob_b if board.turn == "W" else mob_w
        zugzwang_risk = (
            active
            and opp_mob > 0
            and current_mob / max(1, opp_mob) <= self.zugzwang_threshold
        )

        pattern, pattern_notes = _detect_pattern(board) if active else (None, "")

        self.current_state = EndgameState(
            active=active,
            deep=deep,
            phase=phase,
            total_pieces=total,
            pieces_white=pieces_w,
            pieces_black=pieces_b,
            mobility_white=mob_w,
            mobility_black=mob_b,
            zugzwang_risk=zugzwang_risk,
            pattern=pattern,
            pattern_notes=pattern_notes,
        )
        return self.current_state

    def transition_announcements(self) -> list[str]:
        """
        Return phase-transition messages generated since the last call.
        Clears the messages so each transition is announced exactly once.
        """
        msgs: list[str] = []
        s = self.current_state
        if s.deep and not self._announced_deep:
            self._announced_deep = True
            self._announced_endgame = True  # deep implies endgame already passed
            msgs.append(
                f"Deep endgame — {s.total_pieces} pieces remain. "
                "Every move is critical."
            )
        elif s.active and not self._announced_endgame:
            self._announced_endgame = True
            msgs.append(
                f"Endgame reached — {s.total_pieces} pieces on the board."
            )
        return msgs

    def get_current_state(self) -> EndgameState:
        return self.current_state


# ── Module-level helpers ──────────────────────────────────────────────────────

def _mobility(board: "BoardState", color: str) -> int:
    """Count legal moves for `color` without requiring it to be board.turn."""
    phase = get_game_phase(board, color)
    if phase == "place":
        return len(board.legal_placements(color))
    return len(board.legal_moves(color))


def _detect_pattern(board: "BoardState") -> tuple[Optional[str], str]:
    """
    Scan for high-value endgame patterns.
    Returns (pattern_name, description) or (None, "").
    Checks mill_cycle first (more common), then pincer.
    """
    positions = board.positions

    # Mill cycle: a closed mill has at least one piece with a free neighbour
    # → the player can open/close at will to force repeated captures.
    for color in ("W", "B"):
        for mill in MILLS:
            if all(positions[p] == color for p in mill):
                for pos in mill:
                    if any(positions[nb] == "" for nb in ADJACENCY[pos]):
                        return (
                            "mill_cycle",
                            f"{color} can cycle the {'-'.join(mill)} mill",
                        )

    # Pincer: two closed mills of the same colour both adjoin the same
    # opponent piece → that piece faces a dual capture threat.
    for color in ("W", "B"):
        opp = "B" if color == "W" else "W"
        own_mills = [m for m in MILLS if all(positions[p] == color for p in m)]
        if len(own_mills) >= 2:
            for opp_pos in [p for p in positions if positions[p] == opp]:
                neighbours = set(ADJACENCY[opp_pos])
                touching = sum(
                    1 for m in own_mills if any(p in neighbours for p in m)
                )
                if touching >= 2:
                    return (
                        "pincer",
                        f"{color} has two mills pressuring {opp_pos}",
                    )

    return None, ""
