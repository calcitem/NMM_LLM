"""
ai/debriefer.py — Post-game analysis: critical moment detection and LLM commentary.

GameDebriefer replays the stored move sequence, scores each played move against
the minimax best, flags positions where the score drop exceeds the configured
threshold, and calls MillsLLM for targeted commentary on those turning points.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ai.mills_llm import MillsLLM

from game.board import BoardState
from game.rules import get_all_legal_moves
from ai.game_ai import GameAI


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class CriticalMoment:
    ply: int
    color: str
    phase: str
    move_played: dict
    best_move: dict
    score_played: float   # 0.0 (worst legal) – 1.0 (best legal)
    score_drop: float     # 1.0 – score_played
    board_fen: str
    comment: str
    was_blunder: bool
    opening_name: Optional[str]
    deviation: bool       # move deviated from recognised opening


@dataclass
class DebriefReport:
    winner: Optional[str]
    loser: Optional[str]
    human_color: str
    opening_name: Optional[str]
    total_moves: int
    game_record: dict
    critical_moments: list[CriticalMoment]
    summary: str


# ── Debriefer ─────────────────────────────────────────────────────────────────

class GameDebriefer:
    """
    Analyse a completed game record and produce a DebriefReport.

    Parameters
    ----------
    mills_llm:
        LLM interface used for position and game-level commentary.
        If its internal client is None (offline) all comments are empty strings.
    analysis_depth:
        Minimax search depth used when evaluating each position.
        Higher = more accurate but slower. Default 4 ≈ difficulty 3.
    critical_threshold:
        Minimum score drop (0.0–1.0) for a move to be flagged as a
        critical moment. Default 0.4.
    max_comments:
        Maximum number of critical moments that receive LLM commentary
        (to limit API calls). Default 5.
    """

    def __init__(
        self,
        mills_llm: "MillsLLM",
        analysis_depth: int = 4,
        critical_threshold: float = 0.4,
        max_comments: int = 5,
    ) -> None:
        self.mills_llm = mills_llm
        self.critical_threshold = critical_threshold
        self.max_comments = max_comments
        # Clamp depth so we map to a valid difficulty (2–5 ≈ depth 3–6).
        difficulty = max(1, min(5, analysis_depth))
        self._ai = GameAI(color="W", difficulty=difficulty)

    # ── Analysis ──────────────────────────────────────────────────────────────

    def analyse(self, game_record: dict) -> DebriefReport:
        """
        Replay the game, score every move, and build a DebriefReport.
        LLM commentary is generated only for moves that exceed the
        critical_threshold and within the max_comments budget.
        """
        winner = game_record.get("winner")
        loser = ("B" if winner == "W" else "W") if winner else None
        human_color = game_record.get("human_color", "W")
        moves = game_record.get("moves", [])

        # Extract the recognised opening name (first exact/transposition match)
        opening_name: Optional[str] = None
        opening_id: Optional[str] = None
        for m in moves:
            rec = (m.get("opening_recognition") or {})
            if rec.get("status") in ("exact", "transposition") and rec.get("name"):
                opening_name = rec["name"]
                opening_id = rec.get("opening_id")
                break

        critical_moments: list[CriticalMoment] = []
        board = BoardState.new_game()
        comment_count = 0

        for move_record in moves:
            played_move = {
                "from": move_record.get("from"),
                "to": move_record["to"],
                "capture": move_record.get("capture"),
            }

            legal = get_all_legal_moves(board)
            if not legal:
                board = board.apply_move(played_move)
                continue

            score_played = self._ai.score_move(board, played_move)
            score_drop = 1.0 - score_played

            rec = (move_record.get("opening_recognition") or {})
            deviation = bool(rec.get("deviation"))

            if score_drop > self.critical_threshold and comment_count < self.max_comments:
                best_move = self._ai.choose_move(board)
                comment = ""
                if self.mills_llm._client is not None:
                    comment = self.mills_llm.debrief_position(
                        board=board,
                        ply=move_record.get("turn", 0),
                        move_played=played_move,
                        best_move=best_move,
                        score_played=score_played,
                        score_best=1.0,
                        is_critical=True,
                        opening_name=opening_name,
                        context=f"Phase: {move_record.get('type', '?')}",
                    )
                comment_count += 1
                critical_moments.append(CriticalMoment(
                    ply=move_record.get("turn", 0),
                    color=move_record.get("color", "?"),
                    phase=move_record.get("type", "?"),
                    move_played=played_move,
                    best_move=best_move,
                    score_played=score_played,
                    score_drop=score_drop,
                    board_fen=move_record.get("board_fen_before", ""),
                    comment=comment,
                    was_blunder=move_record.get("was_blunder", False),
                    opening_name=opening_name,
                    deviation=deviation,
                ))

            board = board.apply_move(played_move)

        # Overall LLM game summary
        summary = ""
        if self.mills_llm._client is not None:
            summary = self.mills_llm.debrief_game(
                _SimpleReport(
                    winner=winner,
                    loser=loser,
                    opening_name=opening_name,
                    game_record=game_record,
                )
            )

        return DebriefReport(
            winner=winner,
            loser=loser,
            human_color=human_color,
            opening_name=opening_name,
            total_moves=len(moves),
            game_record=game_record,
            critical_moments=critical_moments,
            summary=summary,
        )

    # ── Console output ────────────────────────────────────────────────────────

    def print_report(self, report: DebriefReport, file=None) -> None:
        """Print a formatted debrief report to `file` (default: stdout)."""
        f = file or sys.stdout

        def pr(line: str = "") -> None:
            print(line, file=f)

        W = 56
        pr()
        pr("═" * W)
        pr("  POST-GAME DEBRIEF")
        pr("═" * W)

        winner_label = (
            f"{'White' if report.winner == 'W' else 'Black'} wins"
            if report.winner else "Draw"
        )
        pr(f"  Result:   {winner_label}")
        pr(f"  Opening:  {report.opening_name or 'Unknown / Novel'}")
        pr(f"  Moves:    {report.total_moves}")
        pr(f"  You played: {'White (W)' if report.human_color == 'W' else 'Black (B)'}")

        # ── Annotated move list with deviation markers ─────────────────────
        pr()
        pr("─" * W)
        pr("  MOVE RECORD")
        pr("─" * W)
        self._print_annotated_moves(report, f)

        # ── Critical moments ───────────────────────────────────────────────
        if report.critical_moments:
            pr()
            pr("─" * W)
            pr("  CRITICAL MOMENTS")
            pr("─" * W)
            for cm in report.critical_moments:
                color_name = "White" if cm.color == "W" else "Black"
                played_str = _move_str(cm.move_played)
                best_str = _move_str(cm.best_move)

                labels = []
                if cm.was_blunder:
                    labels.append("deliberate mistake")
                if cm.deviation:
                    labels.append("opening deviation")
                label = f"  [{', '.join(labels)}]" if labels else ""

                pr()
                pr(f"  Ply {cm.ply:>2}  {color_name} played {played_str}{label}")
                pr(f"  Score {cm.score_played:.2f}  (drop {cm.score_drop:.2f})  "
                   f"Best was: {best_str}")
                if cm.comment:
                    for line in cm.comment.splitlines():
                        stripped = line.strip()
                        if stripped:
                            pr(f"    {stripped}")
        else:
            pr()
            pr("  No critical moments — solid game throughout.")

        # ── LLM game summary ───────────────────────────────────────────────
        if report.summary:
            pr()
            pr("─" * W)
            pr("  AI SUMMARY")
            pr("─" * W)
            for line in report.summary.splitlines():
                pr(f"  {line.rstrip()}")

        pr()
        pr("═" * W)
        pr()

    @staticmethod
    def _print_annotated_moves(report: DebriefReport, f) -> None:
        """Print the move list two columns wide with deviation/blunder annotations."""
        moves = report.game_record.get("moves", [])
        pairs: list[list] = []
        pair: list = []
        for m in moves:
            pair.append(m)
            if len(pair) == 2:
                pairs.append(pair)
                pair = []
        if pair:
            pairs.append(pair)

        for i, p in enumerate(pairs):
            line = f"  {i+1:>2}."
            for m in p:
                notation = m.get("notation", _move_str({
                    "from": m.get("from"), "to": m["to"], "capture": m.get("capture")
                }))
                flags = ""
                rec = (m.get("opening_recognition") or {})
                if rec.get("deviation"):
                    flags += "?"
                if m.get("was_blunder"):
                    flags += "!"
                line += f"  {notation}{flags}"
            print(line, file=f)


# ── Helpers ───────────────────────────────────────────────────────────────────

class _SimpleReport:
    """Minimal duck-typed object for mills_llm.debrief_game()."""
    def __init__(self, winner, loser, opening_name, game_record):
        self.winner = winner
        self.loser = loser
        self.opening_name = opening_name
        self.game_record = game_record


def _move_str(move: dict) -> str:
    s = f"{move['from']}-{move['to']}" if move.get("from") else move.get("to", "?")
    if move.get("capture"):
        s += f"x{move['capture']}"
    return s
