"""scripts/sentinel_review.py — replay games and annotate sentinel move quality.

Loads a trained move-level sentinel checkpoint and replays game files, printing
a move-by-move table with the played move's quality and the opportunity gap
(how much better the sentinel's best alternative was). Positions where the
played move was materially worse than the best available are flagged as
"gaps" — the move-level analogue of a turning point.

Usage:
    # Review all games in a directory (summary only)
    python scripts/sentinel_review.py --checkpoint learned_ai/sentinel/checkpoints/best.pt \
        --game-dir learned_ai/self_play_games

    # Review a single game with full move table
    python scripts/sentinel_review.py --checkpoint learned_ai/sentinel/checkpoints/best.pt \
        --game-file learned_ai/self_play_games/game_2026-06-06_015a34db.jsonl

    # Limit to top-N most flagged games
    python scripts/sentinel_review.py --checkpoint learned_ai/sentinel/checkpoints/best.pt \
        --game-dir learned_ai/self_play_games --top 5

    # Show all moves (not just flagged gaps) for a single game
    python scripts/sentinel_review.py --checkpoint learned_ai/sentinel/checkpoints/best.pt \
        --game-file my_game.jsonl --all-moves
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.board import BoardState
from learned_ai.sentinel.config import load_config
from learned_ai.sentinel.dataset import _enumerate_legal_moves
from learned_ai.sentinel.infer import SentinelAdvisor, load_advisor

# Default opportunity-gap threshold for flagging a ply.
_DEFAULT_GAP = 0.3

# ANSI colours (disabled when not a tty)
_TTY = sys.stdout.isatty()
_RED    = "\033[91m" if _TTY else ""
_YELLOW = "\033[93m" if _TTY else ""
_GREEN  = "\033[92m" if _TTY else ""
_CYAN   = "\033[96m" if _TTY else ""
_BOLD   = "\033[1m"  if _TTY else ""
_RESET  = "\033[0m"  if _TTY else ""


def _bar(value: float, width: int = 10) -> str:
    filled = int(round(max(0.0, min(1.0, value)) * width))
    return "█" * filled + "░" * (width - filled)


def _colour_gap(v: float) -> str:
    if v >= 0.5:
        return f"{_RED}{_BOLD}{v:.2f}{_RESET}"
    if v >= 0.3:
        return f"{_YELLOW}{v:.2f}{_RESET}"
    return f"{v:.2f}"


def _colour_quality(v: float) -> str:
    if v <= 0.3:
        return f"{_RED}{v:.2f}{_RESET}"
    if v <= 0.5:
        return f"{_YELLOW}{v:.2f}{_RESET}"
    return f"{v:.2f}"


def _board_from_fen(fen: str):
    try:
        return BoardState.from_fen_string(fen)
    except Exception:
        return None


def _played_index(candidates: list[dict], log_move: dict) -> int:
    """Index of the played move within the enumerated candidate list (else 0)."""
    key = (log_move.get("from"), log_move.get("to"), log_move.get("capture"))
    for i, mv in enumerate(candidates):
        if (mv.get("from"), mv.get("to"), mv.get("capture")) == key:
            return i
    # Match on from/to only when the log omitted the capture target.
    for i, mv in enumerate(candidates):
        if (mv.get("from"), mv.get("to")) == (key[0], key[1]):
            return i
    return 0


def _replay_game(record: dict, advisor: SentinelAdvisor, gap_threshold: float) -> list[dict]:
    """Replay one game record; return list of annotated move dicts."""
    moves = record.get("moves") or []
    results = []

    for ply, log_move in enumerate(moves):
        fen = log_move.get("board_fen_before")
        if not fen:
            continue
        board = _board_from_fen(fen)
        if board is None:
            continue
        player = log_move.get("color") or getattr(board, "turn", "W")

        candidates = _enumerate_legal_moves(board, player)
        if not candidates:
            continue
        played_idx = _played_index(candidates, log_move)

        try:
            advice = advisor.advise(board, candidates, player, played_move_idx=played_idx)
        except Exception:
            continue
        if advice is None:
            continue

        results.append({
            "ply": ply,
            "color": player,
            "notation": log_move.get("notation",
                                     f"{log_move.get('from','?')}→{log_move.get('to','?')}"),
            "type": log_move.get("type", ""),
            "board": board,
            "advice": advice,
            "is_gap": advice.opportunity_gap >= gap_threshold,
        })

    return results


def _print_game_header(record: dict, game_path: str) -> None:
    winner = record.get("winner") or "Draw"
    wp = record.get("white_personality", "?")
    bp = record.get("black_personality", "?")
    wd = record.get("white_difficulty", "?")
    bd = record.get("black_difficulty", "?")
    mc = record.get("move_count", "?")
    print(f"\n{_BOLD}{'─'*64}{_RESET}")
    print(f"{_BOLD}{os.path.basename(game_path)}{_RESET}")
    print(f"  Winner: {_BOLD}{winner}{_RESET}  |  Moves: {mc}")
    print(f"  White: {wp} (diff {wd})    Black: {bp} (diff {bd})")
    print(f"{'─'*64}")


def _print_move_row(entry: dict, show_board: bool = False) -> None:
    adv = entry["advice"]
    gap_str = _colour_gap(adv.opportunity_gap)
    q_str = _colour_quality(adv.played_move_quality)
    flag = f" {_RED}{_BOLD}◀ GAP{_RESET}" if entry["is_gap"] else ""
    print(
        f"  ply {entry['ply']:>3}  {entry['color']}  {entry['notation']:<14}"
        f"  q={q_str} {_bar(adv.played_move_quality, 8)}"
        f"  best={adv.best_available_quality:.2f}"
        f"  gap={gap_str}"
        f"  [{adv.advisory_message}]{flag}"
    )
    if show_board and entry["board"] is not None:
        for line in entry["board"].to_display_grid().splitlines():
            print(f"      {line}")
        print()


def _review_single(record: dict, game_path: str, advisor: SentinelAdvisor,
                   gap_threshold: float, show_all: bool) -> list[dict]:
    entries = _replay_game(record, advisor, gap_threshold)
    gaps = [e for e in entries if e["is_gap"]]

    _print_game_header(record, game_path)
    print(f"  {_CYAN}Opportunity gaps: {len(gaps)} / {len(entries)} plies{_RESET}\n")

    if show_all:
        print(f"  {'ply':>3}  col  {'notation':<14}  {'quality':^14}  best    gap")
        for e in entries:
            _print_move_row(e, show_board=e["is_gap"])
    else:
        if not gaps:
            print(f"  {_GREEN}No opportunity gaps above threshold {gap_threshold:.2f}{_RESET}")
        else:
            print(f"  {'ply':>3}  col  {'notation':<14}  {'quality':^14}  best    gap")
            for e in gaps:
                _print_move_row(e, show_board=True)

    return gaps


def main() -> int:
    p = argparse.ArgumentParser(description="Replay games with sentinel move-quality annotations")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--game-dir", default=None)
    p.add_argument("--game-file", default=None)
    p.add_argument("--config", default=None)
    p.add_argument("--threshold", type=float, default=None,
                   help="Opportunity-gap threshold for flagging (default: 0.3)")
    p.add_argument("--top", type=int, default=None,
                   help="Show only top-N games by flagged-gap count")
    p.add_argument("--all-moves", action="store_true",
                   help="Print every move (not just flagged gaps)")
    p.add_argument("--limit", type=int, default=None, help="Max game files to scan")
    args = p.parse_args()

    if not args.game_dir and not args.game_file:
        p.error("Provide --game-dir or --game-file")

    config = load_config(args.config)
    gap_threshold = args.threshold if args.threshold is not None else _DEFAULT_GAP

    advisor = load_advisor(args.checkpoint, config)
    if advisor is None or not advisor.is_loaded():
        print(f"Failed to load checkpoint: {args.checkpoint}")
        return 1
    print(f"Sentinel loaded  |  opportunity-gap threshold: {gap_threshold:.2f}\n")

    # Collect game files
    if args.game_file:
        paths = [args.game_file]
    else:
        paths = sorted(glob.glob(os.path.join(args.game_dir, "**", "*.jsonl"), recursive=True))
    if args.limit:
        paths = paths[:args.limit]

    # Process games
    game_summaries: list[tuple[str, dict, int]] = []  # (path, record, gap_count)
    for path in paths:
        try:
            with open(path) as f:
                content = f.read().strip()
            record = json.loads(content)
            if not isinstance(record, dict):
                continue
        except Exception:
            continue

        entries = _replay_game(record, advisor, gap_threshold)
        gap_count = sum(1 for e in entries if e["is_gap"])
        game_summaries.append((path, record, gap_count))

    if not game_summaries:
        print("No games found.")
        return 1

    # Single file: full review
    if args.game_file:
        path, record, _ = game_summaries[0]
        _review_single(record, path, advisor, gap_threshold, args.all_moves)
        return 0

    # Directory: sort by gap_count, optionally limit to top-N
    game_summaries.sort(key=lambda x: x[2], reverse=True)
    if args.top:
        to_show = game_summaries[:args.top]
    else:
        to_show = game_summaries

    # Print summary table
    total_games = len(game_summaries)
    total_gaps = sum(s[2] for s in game_summaries)
    games_with_gap = sum(1 for s in game_summaries if s[2] > 0)
    print(f"{'─'*64}")
    print(f"Games scanned: {total_games}  |  "
          f"Games with gaps: {games_with_gap}  |  "
          f"Total gaps flagged: {total_gaps}")
    print(f"{'─'*64}")
    print(f"  {'Game file':<42}  {'winner':<6}  gaps")
    print(f"  {'─'*42}  {'─'*6}  ────")
    for path, record, gap_count in game_summaries[:40]:
        winner = (record.get("winner") or "draw")[:6]
        gap_col = f"{_RED}{gap_count:>4}{_RESET}" if gap_count >= 3 else f"{gap_count:>4}"
        print(f"  {os.path.basename(path):<42}  {winner:<6}  {gap_col}")
    if len(game_summaries) > 40:
        print(f"  ... ({len(game_summaries) - 40} more games)")

    # Detailed review for selected games
    if args.top or args.all_moves:
        for path, record, gap_count in to_show:
            _review_single(record, path, advisor, gap_threshold, args.all_moves)

    print(f"\n{_CYAN}Tip: use --game-file <path> to review a specific game in detail,")
    print(f"or --top 5 to see the 5 games with the most opportunity gaps.{_RESET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
