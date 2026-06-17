"""
scripts/sentinel_assessment.py — Sentinel performance assessment.

Runs AI vs AI games: one side with sentinel (score_adjust), the other plain.
Colors are swapped every game for fairness.

Usage:
    .venv/bin/python scripts/sentinel_assessment.py [--games N] [--diff D] [--gap G]
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from ai.game_ai import GameAI
from game.board import BoardState
from game.rules import get_all_legal_moves, is_terminal
from learned_ai.sentinel.infer import load_advisor

CHECKPOINT = _ROOT / "learned_ai/sentinel/checkpoints/best.pt"


@dataclass
class GameStats:
    winner: Optional[str]           # "W" | "B" | None (draw)
    sentinel_color: str             # which color had sentinel
    plies: int
    sentinel_interventions: int     # times sentinel changed the move
    sentinel_gaps: list[float]      # opportunity_gap on each intervention
    total_sentinel_calls: int       # total moves sentinel evaluated
    avg_sentinel_quality: float     # average played_move_quality across all sentinel calls


@dataclass
class AssessmentResult:
    games: list[GameStats] = field(default_factory=list)

    def summarise(self) -> str:
        n = len(self.games)
        if n == 0:
            return "No games played."

        sent_wins = sum(1 for g in self.games if g.winner == g.sentinel_color)
        plain_wins = sum(1 for g in self.games if g.winner is not None and g.winner != g.sentinel_color)
        draws = sum(1 for g in self.games if g.winner is None)

        total_intervene = sum(g.sentinel_interventions for g in self.games)
        total_calls = sum(g.total_sentinel_calls for g in self.games)
        all_gaps = [gap for g in self.games for gap in g.sentinel_gaps]
        avg_gap = sum(all_gaps) / len(all_gaps) if all_gaps else 0.0
        avg_quality = sum(g.avg_sentinel_quality for g in self.games) / n
        avg_plies = sum(g.plies for g in self.games) / n

        lines = [
            "",
            "═" * 52,
            "  SENTINEL ASSESSMENT RESULTS",
            "═" * 52,
            f"  Games played      : {n}",
            f"  Avg game length   : {avg_plies:.1f} plies",
            "",
            "  ── Win/Draw/Loss (from sentinel's perspective) ──",
            f"  Sentinel wins     : {sent_wins:3d}  ({sent_wins/n*100:.1f}%)",
            f"  Plain AI wins     : {plain_wins:3d}  ({plain_wins/n*100:.1f}%)",
            f"  Draws             : {draws:3d}  ({draws/n*100:.1f}%)",
            "",
            "  ── Sentinel behaviour ───────────────────────────",
            f"  Total moves eval'd: {total_calls}",
            f"  Interventions     : {total_intervene}  ({total_intervene/max(1,total_calls)*100:.1f}% of moves)",
            f"  Avg gap on intervene: {avg_gap*100:.1f}%",
            f"  Avg move quality  : {avg_quality*100:.1f}%",
            "═" * 52,
        ]

        # per-game table
        lines.append("\n  Per-game breakdown:")
        lines.append(f"  {'#':>3}  {'Sent':>4}  {'Winner':>6}  {'Plies':>5}  {'Interv':>6}  {'Avg gap':>7}")
        lines.append("  " + "-" * 44)
        for i, g in enumerate(self.games, 1):
            winner_str = g.winner if g.winner else "draw"
            sent_str = "W" if g.sentinel_color == "W" else "B"
            outcome = "✓" if g.winner == g.sentinel_color else ("=" if g.winner is None else "✗")
            avg_g = sum(g.sentinel_gaps) / len(g.sentinel_gaps) if g.sentinel_gaps else 0.0
            lines.append(
                f"  {i:>3}  {sent_str:>4}  {winner_str:>6}  {g.plies:>5}  "
                f"{g.sentinel_interventions:>5}{outcome}  {avg_g*100:>6.1f}%"
            )

        return "\n".join(lines)


def run_game(
    sentinel_advisor,
    sentinel_color: str,
    difficulty: int,
    sentinel_gap: float,
    move_budget: float = 1.5,
    max_plies: int = 300,
) -> GameStats:
    plain_color = "B" if sentinel_color == "W" else "W"

    ai_sent = GameAI(color=sentinel_color, difficulty=difficulty,
                     override_time_budget=move_budget)
    ai_sent.set_sentinel(sentinel_advisor, mode="score_adjust")
    ai_sent._sentinel_min_gap = sentinel_gap
    ai_sent._sentinel_activation_prob = 1.0

    ai_plain = GameAI(color=plain_color, difficulty=difficulty,
                      override_time_budget=move_budget)

    board = BoardState.new_game()
    plies = 0
    interventions = 0
    gaps = []
    quality_sum = 0.0
    sentinel_calls = 0

    for _ in range(max_plies):
        done, winner = is_terminal(board)
        if done:
            break
        moves = get_all_legal_moves(board)
        if not moves:
            break

        mover = ai_sent if board.turn == sentinel_color else ai_plain
        move = mover.choose_move(board)
        if move is None:
            break

        # Collect sentinel stats after each sentinel-side move
        if board.turn == sentinel_color and ai_sent.last_sentinel_advice is not None:
            adv = ai_sent.last_sentinel_advice
            sentinel_calls += 1
            quality_sum += float(getattr(adv, "played_move_quality", 0.5))
            if getattr(adv, "intervention_applied", None) is not None:
                interventions += 1
                gaps.append(float(getattr(adv, "opportunity_gap", 0.0)))

        board = board.apply_move(move)
        plies += 1

    done, winner = is_terminal(board)
    if not done:
        winner = None  # game hit max_plies — treat as draw

    avg_quality = quality_sum / sentinel_calls if sentinel_calls else 0.5

    return GameStats(
        winner=winner,
        sentinel_color=sentinel_color,
        plies=plies,
        sentinel_interventions=interventions,
        sentinel_gaps=gaps,
        total_sentinel_calls=sentinel_calls,
        avg_sentinel_quality=avg_quality,
    )


def main():
    parser = argparse.ArgumentParser(description="Sentinel AI assessment")
    parser.add_argument("--games",      type=int,   default=20,   help="Number of games (default 20)")
    parser.add_argument("--diff",       type=int,   default=5,    help="AI difficulty level (default 5)")
    parser.add_argument("--gap",        type=float, default=0.15, help="Sentinel min gap 0-1 (default 0.15)")
    parser.add_argument("--budget",     type=float, default=1.5,  help="Move time budget in seconds (default 1.5)")
    parser.add_argument("--checkpoint", type=str,   default=str(CHECKPOINT),
                        help="Path to sentinel checkpoint (default: best.pt)")
    args = parser.parse_args()

    ckpt_path = args.checkpoint
    print(f"Loading sentinel from {ckpt_path}...")
    advisor = load_advisor(ckpt_path)
    if advisor is None or not advisor.is_loaded():
        print("ERROR: sentinel checkpoint not loaded.")
        sys.exit(1)
    print(f"Sentinel loaded. Running {args.games} games at difficulty {args.diff}, "
          f"budget={args.budget}s/move, gap≥{args.gap*100:.0f}%\n", flush=True)


    result = AssessmentResult()
    t0 = time.time()

    for i in range(args.games):
        # Alternate which side gets sentinel
        sentinel_color = "W" if i % 2 == 0 else "B"
        t_game = time.time()
        stats = run_game(
            sentinel_advisor=advisor,
            sentinel_color=sentinel_color,
            difficulty=args.diff,
            sentinel_gap=args.gap,
            move_budget=args.budget,
        )
        result.games.append(stats)
        elapsed = time.time() - t_game
        winner_str = stats.winner if stats.winner else "draw"
        outcome = "✓" if stats.winner == sentinel_color else ("=" if stats.winner is None else "✗")
        print(
            f"  Game {i+1:>2}/{args.games}  sent={sentinel_color}  "
            f"winner={winner_str}  {outcome}  "
            f"plies={stats.plies}  interv={stats.sentinel_interventions}  "
            f"({elapsed:.1f}s)",
            flush=True,
        )

    total_elapsed = time.time() - t0
    print(result.summarise())
    print(f"\n  Total time: {total_elapsed:.1f}s  ({total_elapsed/args.games:.1f}s/game)")


if __name__ == "__main__":
    main()
