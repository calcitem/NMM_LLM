"""
game/board.py — Board state, adjacency graph, mill definitions, and display.

Coordinate system (algebraic notation):
  Outer ring:  a7 d7 g7 g4 g1 d1 a1 a4
  Middle ring: b6 d6 f6 f4 f2 d2 b2 b4
  Inner ring:  c5 d5 e5 e4 e3 d3 c3 c4

All 24 positions are named. 'e1' in early drafts was a typo; the correct
inner-right midpoint is 'e4'.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from game.zobrist import PIECE_KEYS, PLACED_DONE_KEYS, SIDE_KEY, SQ_INDEX, hash_board

# ── Positions ─────────────────────────────────────────────────────────────────

POSITIONS: List[str] = [
    # Outer ring (clockwise from top-left)
    "a7", "d7", "g7", "g4", "g1", "d1", "a1", "a4",
    # Middle ring (clockwise from top-left)
    "b6", "d6", "f6", "f4", "f2", "d2", "b2", "b4",
    # Inner ring (clockwise from top-left)
    "c5", "d5", "e5", "e4", "e3", "d3", "c3", "c4",
]

# ── Adjacency graph ───────────────────────────────────────────────────────────
# Every undirected edge is listed in both directions.
# Cross-ring connections link the midpoint of each ring's side to the next ring.

ADJACENCY: Dict[str, List[str]] = {
    # Outer ring
    "a7": ["d7", "a4"],
    "d7": ["a7", "g7", "d6"],          # cross: d7-d6-d5
    "g7": ["d7", "g4"],
    "g4": ["g7", "g1", "f4"],          # cross: g4-f4-e4
    "g1": ["g4", "d1"],
    "d1": ["g1", "a1", "d2"],          # cross: d1-d2-d3
    "a1": ["d1", "a4"],
    "a4": ["a1", "a7", "b4"],          # cross: a4-b4-c4
    # Middle ring
    "b6": ["d6", "b4"],
    "d6": ["b6", "f6", "d7", "d5"],   # hub: outer + inner cross
    "f6": ["d6", "f4"],
    "f4": ["f6", "f2", "g4", "e4"],   # hub: outer + inner cross
    "f2": ["f4", "d2"],
    "d2": ["f2", "b2", "d1", "d3"],   # hub: outer + inner cross
    "b2": ["d2", "b4"],
    "b4": ["b2", "b6", "a4", "c4"],   # hub: outer + inner cross
    # Inner ring
    "c5": ["d5", "c4"],
    "d5": ["c5", "e5", "d6"],          # cross: d5-d6-d7
    "e5": ["d5", "e4"],
    "e4": ["e5", "e3", "f4"],          # cross: e4-f4-g4
    "e3": ["e4", "d3"],
    "d3": ["e3", "c3", "d2"],          # cross: d3-d2-d1
    "c3": ["d3", "c4"],
    "c4": ["c3", "c5", "b4"],          # cross: c4-b4-a4
}

# ── Mills ─────────────────────────────────────────────────────────────────────
# A mill is any straight line of three pieces owned by the same player.
# 4 sides × 3 rings = 12 ring mills, plus 4 cross-ring connecting lines = 16 total.

MILLS: List[Tuple[str, str, str]] = [
    # Outer ring sides
    ("a7", "d7", "g7"),
    ("g7", "g4", "g1"),
    ("g1", "d1", "a1"),
    ("a1", "a4", "a7"),
    # Middle ring sides
    ("b6", "d6", "f6"),
    ("f6", "f4", "f2"),
    ("f2", "d2", "b2"),
    ("b2", "b4", "b6"),
    # Inner ring sides
    ("c5", "d5", "e5"),
    ("e5", "e4", "e3"),
    ("e3", "d3", "c3"),
    ("c3", "c4", "c5"),
    # Cross-ring connecting lines
    ("d7", "d6", "d5"),
    ("g4", "f4", "e4"),
    ("d1", "d2", "d3"),
    ("a4", "b4", "c4"),
]

# ── Display template ──────────────────────────────────────────────────────────
# Fixed 13-row × 26-char template. Column alignment verified:
#   outer corners   @ cols  1, 25
#   outer midpoints @ col  13 (d-column) and rows 6 (a4/g4)
#   middle corners  @ cols  5, 21
#   middle midpts   @ col  13 and rows 2/10
#   inner corners   @ cols  9, 17
#   inner midpts    @ col  13 and rows 4/8

_DISPLAY = (
    " {a7}───────────{d7}───────────{g7}\n"
    " │           │           │\n"
    " │   {b6}───────{d6}───────{f6}   │\n"
    " │   │         │         │   │\n"
    " │   │   {c5}───{d5}───{e5}   │   │\n"
    " │   │   │           │   │   │\n"
    " {a4}───{b4}───{c4}       {e4}───{f4}───{g4}\n"
    " │   │   │           │   │   │\n"
    " │   │   {c3}───{d3}───{e3}   │   │\n"
    " │   │         │         │   │\n"
    " │   {b2}───────{d2}───────{f2}   │\n"
    " │           │           │\n"
    " {a1}───────────{d1}───────────{g1}"
)

# Same template but with position names for the reference display shown at game start.
BOARD_REFERENCE = (
    " a7───────────d7───────────g7\n"
    " │            │            │\n"
    " │   b6───────d6───────f6  │\n"
    " │   │         │         │  │\n"
    " │   │   c5───d5───e5   │  │\n"
    " │   │   │           │  │  │\n"
    " a4───b4───c4       e4───f4───g4\n"
    " │   │   │           │  │  │\n"
    " │   │   c3───d3───e3   │  │\n"
    " │   │         │         │  │\n"
    " │   b2───────d2───────f2  │\n"
    " │            │            │\n"
    " a1───────────d1───────────g1"
)


# ── BoardState ────────────────────────────────────────────────────────────────

@dataclass
class BoardState:
    """
    Immutable snapshot of the board.  apply_move() returns a new BoardState.
    """
    positions: Dict[str, str]       # pos -> "W" | "B" | ""
    turn: str                       # "W" | "B"
    pieces_on_board: Dict[str, int] # {"W": n, "B": n}
    pieces_placed: Dict[str, int]   # {"W": n, "B": n}  (cumulative placements)
    pieces_captured: Dict[str, int] # {"W": n, "B": n}  (pieces captured *by* color)
    hash_key: int = 0               # Zobrist hash; maintained incrementally by apply_move()

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def new_game(cls) -> "BoardState":
        # New game: all squares empty, W to move, neither side done placing.
        # hash_board() on this state returns 0 (no keys XOR'd), so we skip the call.
        return cls(
            positions={pos: "" for pos in POSITIONS},
            turn="W",
            pieces_on_board={"W": 0, "B": 0},
            pieces_placed={"W": 0, "B": 0},
            pieces_captured={"W": 0, "B": 0},
            hash_key=0,
        )

    @classmethod
    def from_setup(
        cls,
        positions: Dict[str, str],
        turn: str,
        phase: str,
    ) -> "BoardState":
        """Create a BoardState from an arbitrary editor setup.

        phase must be 'place' or 'move'.  In 'move' phase both players are
        treated as having placed all 9 pieces (pieces_placed = 9 each), so the
        board immediately enters movement/fly rules.  In 'place' phase
        pieces_placed is set to the count currently on the board (allowing
        continued placement).
        """
        pos = {p: positions.get(p, "") for p in POSITIONS}
        w_on = sum(1 for v in pos.values() if v == "W")
        b_on = sum(1 for v in pos.values() if v == "B")
        if phase == "move":
            placed = {"W": 9, "B": 9}
        else:
            placed = {"W": w_on, "B": b_on}
        b = cls(
            positions=pos,
            turn=turn,
            pieces_on_board={"W": w_on, "B": b_on},
            pieces_placed=placed,
            pieces_captured={"W": 0, "B": 0},
            hash_key=0,
        )
        b.hash_key = hash_board(b)
        return b

    @classmethod
    def from_fen_string(cls, fen: str) -> "BoardState":
        """Parse a FEN string produced by to_fen_string().
        Format: '<24 chars>|<turn>|<W_placed>|<B_placed>'
        """
        pos_str, turn, w_placed_s, b_placed_s = fen.split("|")
        w_placed, b_placed = int(w_placed_s), int(b_placed_s)
        positions = {POSITIONS[i]: (pos_str[i] if pos_str[i] != "." else "") for i in range(24)}
        w_on = sum(1 for v in positions.values() if v == "W")
        b_on = sum(1 for v in positions.values() if v == "B")
        b = cls(
            positions=positions,
            turn=turn,
            pieces_on_board={"W": w_on, "B": b_on},
            pieces_placed={"W": w_placed, "B": b_placed},
            pieces_captured={"W": b_placed - b_on, "B": w_placed - w_on},
            hash_key=0,
        )
        b.hash_key = hash_board(b)
        return b

    # ── Phase ─────────────────────────────────────────────────────────────────

    @property
    def phase(self) -> str:
        """
        'place' while either player still has pieces to place.
        'move'  once both players have placed all 9.
        Individual per-colour phase (including 'fly') is resolved by rules.get_game_phase.
        """
        if self.pieces_placed["W"] < 9 or self.pieces_placed["B"] < 9:
            return "place"
        return "move"

    # ── Mill detection ────────────────────────────────────────────────────────

    def is_mill(self, pos: str, color: str) -> bool:
        """Return True if pos (occupied by color) is part of any mill for color."""
        for mill in MILLS:
            if pos in mill and all(self.positions[p] == color for p in mill):
                return True
        return False

    # ── Legal move generation ─────────────────────────────────────────────────

    def legal_placements(self, color: str) -> List[str]:
        """All empty positions (valid targets during the placement phase)."""
        return [pos for pos in POSITIONS if self.positions[pos] == ""]

    def legal_moves(self, color: str) -> List[Tuple[str, str]]:
        """
        (from, to) pairs for the movement phase.
        In the fly phase (≤3 pieces remaining after all placed) the player
        may move to any empty square; otherwise only to adjacent empties.
        """
        flying = (
            self.pieces_placed[color] == 9
            and self.pieces_on_board[color] <= 3
        )
        own_pieces = [pos for pos in POSITIONS if self.positions[pos] == color]
        empty = [pos for pos in POSITIONS if self.positions[pos] == ""]
        moves: List[Tuple[str, str]] = []
        for src in own_pieces:
            targets = empty if flying else [t for t in ADJACENCY[src] if self.positions[t] == ""]
            for tgt in targets:
                moves.append((src, tgt))
        return moves

    def legal_captures(self, color: str) -> List[str]:
        """
        Opponent positions that the player of 'color' may capture after forming a mill.
        Prefers non-mill opponent pieces; falls back to all opponent pieces if every
        opponent piece is currently in a mill.
        """
        opponent = "B" if color == "W" else "W"
        opp_pieces = [pos for pos in POSITIONS if self.positions[pos] == opponent]
        non_mill = [pos for pos in opp_pieces if not self.is_mill(pos, opponent)]
        return non_mill if non_mill else opp_pieces

    # ── Move application ──────────────────────────────────────────────────────

    def apply_move(self, move: dict) -> "BoardState":
        """
        Return a new BoardState after applying a complete move dict:
          {"from": str | None, "to": str, "capture": str | None}
        'from' is None for placement moves.
        Does not validate legality; caller is responsible.
        """
        new_pos = dict(self.positions)
        new_on  = dict(self.pieces_on_board)
        new_placed = dict(self.pieces_placed)
        new_captured = dict(self.pieces_captured)

        color = self.turn
        opponent = "B" if color == "W" else "W"
        color_idx = 0 if color == "W" else 1
        opp_idx = 1 - color_idx

        # Incremental Zobrist hash update
        new_hash = self.hash_key

        if move["from"] is None and move.get("to") is not None:
            # Placement: add color piece at to-square
            new_pos[move["to"]] = color
            new_on[color] += 1
            new_placed[color] += 1
            new_hash ^= PIECE_KEYS[color_idx][SQ_INDEX[move["to"]]]
            # If this placement completes the 9th piece, toggle the done-placing bit
            if new_placed[color] >= 9 and self.pieces_placed[color] < 9:
                new_hash ^= PLACED_DONE_KEYS[color_idx]
        elif move["from"] is not None:
            # Movement (or fly): relocate color piece from→to
            new_pos[move["from"]] = ""
            new_pos[move["to"]] = color
            new_hash ^= PIECE_KEYS[color_idx][SQ_INDEX[move["from"]]]
            new_hash ^= PIECE_KEYS[color_idx][SQ_INDEX[move["to"]]]

        if move.get("capture"):
            # Remove captured opponent piece
            new_pos[move["capture"]] = ""
            new_on[opponent] -= 1
            new_captured[color] += 1
            new_hash ^= PIECE_KEYS[opp_idx][SQ_INDEX[move["capture"]]]

        # Flip side-to-move
        new_hash ^= SIDE_KEY

        return BoardState(
            positions=new_pos,
            turn=opponent,
            pieces_on_board=new_on,
            pieces_placed=new_placed,
            pieces_captured=new_captured,
            hash_key=new_hash,
        )

    def swap_turn(self) -> "BoardState":
        """Return a copy with the side-to-move flipped (for null-move pruning).

        Only valid during movement and fly phases — caller must guard against placement.
        All piece counts, positions, and the Zobrist hash are preserved; only `turn`
        changes (hash update mirrors the SIDE_KEY toggle in apply_move).
        """
        import copy
        nb = copy.copy(self)
        nb.turn = "B" if self.turn == "W" else "W"
        nb.hash_key = self.hash_key ^ SIDE_KEY
        return nb

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_fen_string(self) -> str:
        """
        Compact board string used as a ChromaDB vector key and for logging.
        Format: <24 board chars>|<turn>|<W_placed>|<B_placed>
        Board chars are in POSITIONS order: W, B, or . for empty.
        """
        board = "".join(
            self.positions[pos] if self.positions[pos] else "." for pos in POSITIONS
        )
        return f"{board}|{self.turn}|{self.pieces_placed['W']}|{self.pieces_placed['B']}"

    def to_display_grid(self) -> str:
        """ASCII board with W / B / · for each of the 24 positions."""
        def piece(pos: str) -> str:
            v = self.positions[pos]
            if v == "W":
                return "W"
            if v == "B":
                return "B"
            return "·"

        return _DISPLAY.format(
            a7=piece("a7"), d7=piece("d7"), g7=piece("g7"),
            g4=piece("g4"), g1=piece("g1"), d1=piece("d1"),
            a1=piece("a1"), a4=piece("a4"),
            b6=piece("b6"), d6=piece("d6"), f6=piece("f6"),
            f4=piece("f4"), f2=piece("f2"), d2=piece("d2"),
            b2=piece("b2"), b4=piece("b4"),
            c5=piece("c5"), d5=piece("d5"), e5=piece("e5"),
            e4=piece("e4"), e3=piece("e3"), d3=piece("d3"),
            c3=piece("c3"), c4=piece("c4"),
        )

    def __repr__(self) -> str:
        return (
            f"BoardState(turn={self.turn!r}, phase={self.phase!r}, "
            f"on_board={self.pieces_on_board}, placed={self.pieces_placed})"
        )
