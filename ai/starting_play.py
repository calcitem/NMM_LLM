"""ai/starting_play.py — Early starting-play family detection (Stage 5.16).

Fires during the placement phase at three checkpoints:
  ply 6  — early family:  broad shape intent (≥3 pieces per side placed)
  ply 12 — mid variant:   structural commitment (≥6 pieces per side placed)
  ply 18 — final:         handed off to OpeningRecognizer (existing system)

Shape families detected
-----------------------
  Outer Square   — pieces concentrated on the outer ring corners/edges
  Cardinal Cross — pieces targeting the four mid-edge cross positions (d1/a4/g4/d7)
  Diamond        — pieces on middle-ring edge positions (d2/b4/f4/d6)
  Inner Web      — pieces on inner-ring positions (c3/d3/e3/c4/e4/c5/d5/e5)
  Side Column    — two or more pieces in the same outer-ring column (a1/a4/a7 or g1/g4/g7)
  Parallel Mill  — two pieces already on the same mill line with the third empty
  Wrap Setup     — pieces positioned to threaten mill-wrapping (parallel adjacent mills)
  Flexible       — no dominant structural theme yet

StartingPlayVariant
-------------------
A lightweight dataclass capturing a named starting-play sequence with tags and
outcome stats.  Intended to evolve into a searchable database; initially
populated from the existing openings.json and self-play discoveries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from game.board import BoardState

# ── Position sets ─────────────────────────────────────────────────────────────
#
# Outer ring:  a7 d7 g7 g4 g1 d1 a1 a4
# Middle ring: b6 d6 f6 f4 f2 d2 b2 b4
# Inner ring:  c5 d5 e5 e4 e3 d3 c3 c4

_OUTER_CORNERS  = {"a7", "g7", "g1", "a1"}
_OUTER_EDGES    = {"d7", "g4", "d1", "a4"}   # mid-edge cross positions
_MIDDLE_CORNERS = {"b6", "f6", "f2", "b2"}
_MIDDLE_EDGES   = {"d6", "f4", "d2", "b4"}   # middle-ring diamond positions
_INNER_ALL      = {"c5", "d5", "e5", "e4", "e3", "d3", "c3", "c4"}

# Column groups for side-column detection
_COL_A = {"a1", "a4", "a7"}
_COL_G = {"g1", "g4", "g7"}
_ROW_1 = {"a1", "d1", "g1"}
_ROW_7 = {"a7", "d7", "g7"}


@dataclass
class StartingPlayVariant:
    """A named starting-play sequence with metadata and outcome statistics."""
    variant_id: str
    name: str
    family: str                   # broad family name (matches FAMILY_ constants below)
    stage: str                    # "early" | "mid_placement" | "final_placement"
    side: str                     # "W", "B", or "both"
    move_sequence: list[str]      # canonical move list (placement notations)
    tags: list[str] = field(default_factory=list)
    strategic_notes: str = ""
    parent_variant_id: str = ""
    outcome_stats: dict = field(default_factory=lambda: {"W": 0, "B": 0, "D": 0})

    def win_rate(self, color: str) -> float:
        w = self.outcome_stats.get("W", 0)
        b = self.outcome_stats.get("B", 0)
        d = self.outcome_stats.get("D", 0)
        total = w + b + d
        if total == 0:
            return 0.5
        wins = w if color == "W" else b
        return (wins + 0.4 * d) / total


# ── Family constants ──────────────────────────────────────────────────────────

FAMILY_OUTER_SQUARE  = "Outer Square"
FAMILY_CARDINAL      = "Cardinal Cross"
FAMILY_DIAMOND       = "Diamond"
FAMILY_INNER_WEB     = "Inner Web"
FAMILY_SIDE_COLUMN   = "Side Column"
FAMILY_PARALLEL_MILL = "Parallel Mill"
FAMILY_WRAP_SETUP    = "Wrap Setup"
FAMILY_FLEXIBLE      = "Flexible"

# Human-readable descriptions for each family
FAMILY_NOTES: dict[str, str] = {
    FAMILY_OUTER_SQUARE: (
        "Pieces occupying outer-ring corner positions. Solid defensive base; "
        "long routes between mills means opponent has time to develop."
    ),
    FAMILY_CARDINAL: (
        "Control of the four mid-edge positions (d1, a4, g4, d7) that connect rings. "
        "Maximises mobility and cross-ring mill threats."
    ),
    FAMILY_DIAMOND: (
        "Middle-ring edge occupation (d2, b4, f4, d6) forming a rotated square. "
        "Classic high-pressure opening; many 2-move mill threats."
    ),
    FAMILY_INNER_WEB: (
        "Inner-ring focus. Tight control near the centre; dangerous at low piece "
        "counts but can be surrounded by outer threats."
    ),
    FAMILY_SIDE_COLUMN: (
        "Two or more pieces on the same outer column (a-column or g-column). "
        "Builds a fast mill on one side; watch for wrap threats."
    ),
    FAMILY_PARALLEL_MILL: (
        "Already forming a mill line — two pieces on the same mill with the third empty. "
        "Immediate threat pressure; forces opponent to react or lose a piece."
    ),
    FAMILY_WRAP_SETUP: (
        "Pieces positioned to place parallel mills side-by-side. "
        "Mill-wrapping pins the opponent's pieces and denies mobility."
    ),
    FAMILY_FLEXIBLE: (
        "No dominant structural theme yet — flexible approach. "
        "Will likely commit to a family after the next 1–2 placements."
    ),
}


def detect_early_family(board: "BoardState", color: str) -> tuple[str, str]:
    """
    Detect the starting-play family for `color` from the current board state.

    Returns (family_name, strategic_note).
    Should be called after ply ≥ 6 (at least 3 pieces per side placed).
    """
    from game.board import MILLS

    placed = {pos for pos, c in board.positions.items() if c == color}
    n = len(placed)
    if n < 2:
        return FAMILY_FLEXIBLE, FAMILY_NOTES[FAMILY_FLEXIBLE]

    # ── Check for parallel mill (immediate threat — highest priority) ─────────
    for mill in MILLS:
        mill_set = set(mill)
        own_in_mill = placed & mill_set
        if len(own_in_mill) == 2:
            empty_in_mill = mill_set - placed - {
                pos for pos, c in board.positions.items() if c != "" and c != color
            }
            if empty_in_mill:
                return FAMILY_PARALLEL_MILL, FAMILY_NOTES[FAMILY_PARALLEL_MILL]

    # ── Check for wrap setup (two adjacent parallel mills forming) ────────────
    # Simplified: ≥2 pieces on middle-ring edges alongside outer-ring edges
    if (placed & _MIDDLE_EDGES) and (placed & _OUTER_EDGES):
        outer_mid_pairs = [
            ("d7", "d6"), ("g4", "f4"), ("d1", "d2"), ("a4", "b4"),
        ]
        for outer_pos, mid_pos in outer_mid_pairs:
            if outer_pos in placed and mid_pos in placed:
                return FAMILY_WRAP_SETUP, FAMILY_NOTES[FAMILY_WRAP_SETUP]

    # ── Count pieces by ring / theme ──────────────────────────────────────────
    outer_corner_n = len(placed & _OUTER_CORNERS)
    outer_edge_n   = len(placed & _OUTER_EDGES)
    middle_edge_n  = len(placed & _MIDDLE_EDGES)
    inner_n        = len(placed & _INNER_ALL)
    col_a_n        = len(placed & _COL_A)
    col_g_n        = len(placed & _COL_G)

    # Side column: two or more own pieces in the same outer column
    if col_a_n >= 2 or col_g_n >= 2:
        return FAMILY_SIDE_COLUMN, FAMILY_NOTES[FAMILY_SIDE_COLUMN]

    # Diamond: dominant middle-ring edge occupation
    if middle_edge_n >= 2 and middle_edge_n >= outer_corner_n:
        return FAMILY_DIAMOND, FAMILY_NOTES[FAMILY_DIAMOND]

    # Cardinal cross: dominant outer-edge (cross-position) occupation
    if outer_edge_n >= 2 and outer_edge_n >= outer_corner_n:
        return FAMILY_CARDINAL, FAMILY_NOTES[FAMILY_CARDINAL]

    # Outer square: dominant corner occupation
    if outer_corner_n >= 2:
        return FAMILY_OUTER_SQUARE, FAMILY_NOTES[FAMILY_OUTER_SQUARE]

    # Inner web: dominant inner-ring occupation
    if inner_n >= 2:
        return FAMILY_INNER_WEB, FAMILY_NOTES[FAMILY_INNER_WEB]

    return FAMILY_FLEXIBLE, FAMILY_NOTES[FAMILY_FLEXIBLE]


def combined_family_summary(board: "BoardState") -> dict[str, str]:
    """
    Return the early-family detection for both sides.
    Suitable for injecting into the coordinator's strategic context.
    """
    w_fam, w_note = detect_early_family(board, "W")
    b_fam, b_note = detect_early_family(board, "B")
    return {
        "white_family": w_fam,
        "white_note":   w_note,
        "black_family": b_fam,
        "black_note":   b_note,
    }
