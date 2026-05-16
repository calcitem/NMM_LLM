#!/usr/bin/env python3
"""
tools/self_play.py — AI self-play training loop.

Runs GameAI vs GameAI for N games, recording every game through the normal
MemoryManager pipeline (data/games/) so the LLM reads them before future
games and the opening book accumulates real win-rate statistics.

How the AI improves over runs
------------------------------
  Opening book  — UCB1 win-rate scores update after every game; future game
                  starts select statistically stronger openings.
  Novel lines   — Sequences with no book match are saved as "learned" openings
                  and named by the LLM (--llm mode) or auto-named (fast mode).
  LLM context   — All games land in data/games/; MillsLLM reads the last 10
                  before each web-game, giving it richer positional context.
  Pattern cache — MemoryManager.analyse_patterns() distils placement and
                  weakness patterns from recent games into the coordinator's
                  narrative_memory prompt.

What does NOT change from self-play
-------------------------------------
  Minimax weights are fixed; heuristic parameters are hard-coded in
  heuristics.py.  Stage 7 (see PLAN.md) will add genetic weight evolution
  driven by self-play fitness scores.

Usage
-----
  # 20 games at equal strength, fast mode (no LLM calls):
  python tools/self_play.py --games 20 --no-llm

  # 50 games with LLM commentary, White plays harder:
  python tools/self_play.py --games 50 --white 7 --black 4

  # 100 games alternating colours, White blunders 10% of moves:
  python tools/self_play.py --games 100 --no-llm --swap --blunder 0.1

  # Continuous training run:
  python tools/self_play.py --games 500 --no-llm --white 6 --black 6 -v
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from game.game_engine import GameEngine
from game.rules import get_all_legal_moves, get_game_phase
from ai.game_ai import GameAI
from ai.opening_book import OpeningBook
from ai.opening_recognizer import OpeningRecognizer
from ai.endgame_recognizer import EndgameRecognizer
from ai.memory_manager import MemoryManager


_MAX_MOVES       = 300  # hard safety cap
_REPEAT_DRAW     = 3    # declare draw when the same FEN appears this many times


def _load_settings() -> dict:
    path = ROOT / "data" / "settings.json"
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _move_str(move: dict) -> str:
    s = f"{move['from']}-{move['to']}" if move.get("from") else move["to"]
    if move.get("capture"):
        s += f"x{move['capture']}"
    return s


# ── Fast mode: pure AI vs AI, no LLM ─────────────────────────────────────────

def _run_fast_game(
    white_ai: GameAI,
    black_ai: GameAI,
    book: OpeningBook,
    verbose: bool = False,
) -> dict:
    """Play one game with no LLM calls. Returns a minimal game-record dict."""
    from collections import Counter
    engine     = GameEngine(human_color="B")
    white_rec  = OpeningRecognizer(book)
    black_rec  = OpeningRecognizer(book)
    egr        = EndgameRecognizer()
    session_id = str(uuid.uuid4())
    moves_log: list[dict] = []
    move_count = 0
    fen_counts: Counter = Counter()
    draw_by_repetition = False

    while not engine.finished and move_count < _MAX_MOVES:
        board = engine.board
        fen   = board.to_fen_string()

        # Draw by threefold repetition
        fen_counts[fen] += 1
        if fen_counts[fen] >= _REPEAT_DRAW:
            draw_by_repetition = True
            break

        color = board.turn
        ai    = white_ai if color == "W" else black_ai
        rec   = white_rec if color == "W" else black_rec

        recognition   = rec.get_current_result()
        endgame_state = egr.update(board)

        move = ai.choose_move(board, recognition=recognition, endgame_state=endgame_state)

        moves_log.append({
            "turn":             move_count + 1,
            "color":            color,
            "type":             get_game_phase(board, color),
            "from":             move.get("from"),
            "to":               move.get("to"),
            "capture":          move.get("capture"),
            "notation":         _move_str(move),
            "board_fen_before": fen,
        })

        engine.apply_move(move)
        move_count += 1

        # Update both recognizers so each sees the full game sequence
        white_rec.update(move.get("to", ""), engine.board)
        black_rec.update(move.get("to", ""), engine.board)

        if verbose:
            print(f"    Move {move_count:3d}: {color} {_move_str(move)}", end="\r")

    winner = engine.winner  # None → draw (repetition or move cap)

    # Record opening book outcomes
    for rec_inst in (white_rec, black_rec):
        result = rec_inst.get_current_result()
        if result and result.opening_id and result.status in ("exact", "probable", "transposition"):
            book.update_outcome_stats(result.opening_id, winner=winner or "D")

    # Auto-name novel openings (no LLM; generate a placeholder)
    for rec_inst in (white_rec, black_rec):
        result = rec_inst.get_current_result()
        if result and result.status == "novel":
            placement_moves = [m["to"] for m in moves_log if m["type"] == "place"]
            if len(placement_moves) >= 6:
                sigs = _compute_fen_signatures(placement_moves)
                novel = book.save_novel_opening(placement_moves, sigs, outcome=winner)
                if not novel.name or novel.name.startswith("Novel"):
                    novel.name = f"Self-Play Line {novel.opening_id[:6]}"
                book.save_opening(novel)
            break  # one novel registration per game

    return {
        "session_id":      session_id,
        "date":            datetime.now().isoformat(),
        "human_color":     "self_play",
        "winner":          winner,
        "move_count":      move_count,
        "white_difficulty": white_ai.difficulty,
        "black_difficulty": black_ai.difficulty,
        "self_play":       True,
        "moves":           moves_log,
    }


def _compute_fen_signatures(placement_moves: list[str]) -> list[dict]:
    from game.board import BoardState
    board = BoardState.new_game()
    sigs: list[dict] = []
    for i, pos in enumerate(placement_moves):
        board = board.apply_move({"from": None, "to": pos, "capture": None})
        ply = i + 1
        if ply in (4, 6, 8, 10):
            sigs.append({"ply": ply, "fen": board.to_fen_string()})
    return sigs


# ── LLM mode: coordinator-driven, one side has full deliberation ──────────────

def _run_llm_game(
    white_ai: GameAI,
    black_ai: GameAI,
    settings: dict,
    verbose: bool = False,
) -> dict:
    """
    Play one game using the Coordinator for the primary (White) AI.
    Black uses raw GameAI but its moves are passed through the coordinator's
    react_to_human_move so the opening recognizer and commentary still fire.
    The game record is saved directly (skipping the per-game LLM summary to
    keep self-play fast; run with --summary for a batch summary at the end).
    """
    from ai.coordinator import Coordinator
    from ai.mills_llm import MillsLLM

    url   = settings.get("ollama_url",   "http://localhost:11434")
    model = settings.get("ollama_model", "llama3.1:8b")

    mem  = MemoryManager(
        ollama_url=url, ollama_model=model,
        chroma_path=str(ROOT / "data" / "chroma"),
        games_path=str(ROOT / "data" / "games"),
        session_path=str(ROOT / "data" / "session_memory"),
    )
    llm  = MillsLLM(memory=mem, ollama_url=url, model=model)
    book = OpeningBook()
    rec  = OpeningRecognizer(book)
    egr  = EndgameRecognizer(
        active_threshold=settings.get("endgame_active_threshold", 11),
        deep_threshold=settings.get("endgame_deep_threshold", 8),
        zugzwang_threshold=settings.get("endgame_zugzwang_threshold", 0.4),
    )
    coord = Coordinator(
        game_ai=white_ai, mills_llm=llm, memory=mem,
        poor_move_threshold=settings.get("poor_move_threshold", 0.3),
        max_poor_move_comments=settings.get("max_poor_move_comments_per_game", 5),
        opening_recognizer=rec, endgame_recognizer=egr,
    )

    # Black-side recognizer (coordinator only tracks from White's perspective)
    black_rec = OpeningRecognizer(book)
    black_egr = EndgameRecognizer()

    coord.on_game_start()
    engine     = GameEngine(human_color="B")
    move_count = 0
    from collections import Counter
    fen_counts: Counter = Counter()

    while not engine.finished and move_count < _MAX_MOVES:
        board        = engine.board
        fen          = board.to_fen_string()
        fen_counts[fen] += 1
        if fen_counts[fen] >= _REPEAT_DRAW:
            break  # draw by repetition
        color        = board.turn
        board_before = board

        if color == "W":
            move = coord.deliberate(board)
            engine.apply_move(move)
            # White's move from Black's perspective:
            black_rec.update(move.get("to", ""), engine.board)
        else:
            # Black chooses its own move; coordinator reacts as if to a human move
            black_recognition   = black_rec.get_current_result()
            black_endgame_state = black_egr.update(board)
            move = black_ai.choose_move(
                board, recognition=black_recognition, endgame_state=black_endgame_state
            )
            engine.apply_move(move)
            coord.react_to_human_move(board_before, engine.board, move)
            black_rec.update(move.get("to", ""), engine.board)

        if verbose:
            for line in coord.flush_dialogue():
                print(f"    {line}")
        else:
            coord.flush_dialogue()

        move_count += 1

    winner = engine.winner
    record = coord.build_game_record(winner=winner, human_color="self_play")
    record["white_difficulty"] = white_ai.difficulty
    record["black_difficulty"] = black_ai.difficulty
    record["self_play"]        = True
    record["move_count"]       = move_count

    # Save game record and update opening book — skip slow LLM summary
    mem.save_game_record(record)

    if rec.get_current_result().opening_id:
        result = rec.get_current_result()
        if result.status in ("exact", "probable", "transposition"):
            book.update_outcome_stats(result.opening_id, winner=winner or "D")
    elif rec.get_current_result().status == "novel":
        placement_moves = [m["to"] for m in record.get("moves", []) if m.get("type") == "place"]
        if len(placement_moves) >= 6:
            sigs   = _compute_fen_signatures(placement_moves)
            name   = llm.name_novel_opening(placement_moves)
            novel  = book.save_novel_opening(placement_moves, sigs, outcome=winner)
            novel.name = name or f"Self-Play Line {novel.opening_id[:6]}"
            book.save_opening(novel)
            if verbose:
                print(f"    Novel opening saved: {novel.name}")

    return record


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Nine Men's Morris AI self-play training loop",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--games",   "-n", type=int,   default=10,  metavar="N",
                        help="Number of games to play (default: 10)")
    parser.add_argument("--white",         type=int,   default=5,   metavar="D",
                        help="White AI difficulty 1-10 (default: 5)")
    parser.add_argument("--black",         type=int,   default=5,   metavar="D",
                        help="Black AI difficulty 1-10 (default: 5)")
    parser.add_argument("--blunder",       type=float, default=0.0, metavar="P",
                        help="Blunder probability for White 0-1 (default: 0.0)")
    parser.add_argument("--no-llm",        action="store_true",
                        help="Skip all LLM calls — fast mode (recommended for bulk runs)")
    parser.add_argument("--swap",          action="store_true",
                        help="Alternate which AI plays White each game (reduces first-mover bias)")
    parser.add_argument("--summary",       action="store_true",
                        help="Ask LLM for a batch summary after all games finish")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print each move and any commentary lines")
    args = parser.parse_args()

    settings  = _load_settings()
    n_games   = max(1, args.games)
    w_diff    = max(1, min(10, args.white))
    b_diff    = max(1, min(10, args.black))
    use_llm   = not args.no_llm

    print(f"\nNine Men's Morris — Self-Play Training")
    print(f"  Games:       {n_games}")
    print(f"  White diff:  {w_diff}  |  Black diff: {b_diff}")
    print(f"  Mode:        {'LLM commentary' if use_llm else 'fast (no LLM)'}")
    print(f"  Blunder:     {args.blunder:.0%}  |  Swap colours: {args.swap}")
    print()

    # Shared objects for fast mode
    book = OpeningBook()
    mem  = None
    if not use_llm:
        url   = settings.get("ollama_url",   "http://localhost:11434")
        model = settings.get("ollama_model", "llama3.1:8b")
        mem   = MemoryManager(
            ollama_url=url, ollama_model=model,
            chroma_path=str(ROOT / "data" / "chroma"),
            games_path=str(ROOT / "data" / "games"),
            session_path=str(ROOT / "data" / "session_memory"),
            use_ollama_embeddings=False,  # no LLM embedding calls in fast mode
        )

    results     = {"W": 0, "B": 0, "D": 0}
    total_moves = 0
    total_time  = 0.0
    all_records: list[dict] = []

    for game_num in range(1, n_games + 1):
        # Optionally swap which colour the harder engine plays
        if args.swap and game_num % 2 == 0:
            wd, bd = b_diff, w_diff
        else:
            wd, bd = w_diff, b_diff

        white_ai = GameAI(
            color="W", difficulty=wd,
            blunder_probability=args.blunder,
        )
        black_ai = GameAI(color="B", difficulty=bd)

        t0 = time.perf_counter()
        print(f"  Game {game_num:3d}/{n_games} ... ", end="", flush=True)
        if args.verbose:
            print()

        try:
            if use_llm:
                record = _run_llm_game(white_ai, black_ai, settings, verbose=args.verbose)
            else:
                record = _run_fast_game(white_ai, black_ai, book, verbose=args.verbose)
                mem.save_game_record(record)  # type: ignore[union-attr]
        except KeyboardInterrupt:
            print("\n  Interrupted.")
            break
        except Exception as exc:
            print(f"ERROR: {exc}")
            import traceback
            if args.verbose:
                traceback.print_exc()
            continue

        elapsed = time.perf_counter() - t0
        winner  = record.get("winner")
        moves   = record.get("move_count", len(record.get("moves", [])))

        total_moves += moves
        total_time  += elapsed
        results[winner or "D"] += 1
        all_records.append(record)

        if args.verbose:
            print(f"  Game {game_num:3d}: ", end="")
        label = "White" if winner == "W" else "Black" if winner == "B" else "Draw "
        print(f"{label}  ({moves:3d} moves, {elapsed:.1f}s)")

    played = sum(results.values())
    if played == 0:
        print("No games completed.")
        return

    # ── Stats summary ─────────────────────────────────────────────────────────
    print()
    print("─" * 44)
    print(f"Results after {played} games:")
    print(f"  White wins : {results['W']:4d}  ({results['W']/played*100:5.1f}%)")
    print(f"  Black wins : {results['B']:4d}  ({results['B']/played*100:5.1f}%)")
    print(f"  Draws      : {results['D']:4d}  ({results['D']/played*100:5.1f}%)")
    print(f"  Avg moves  : {total_moves/played:.1f}")
    print(f"  Avg time   : {total_time/played:.1f}s / game")
    print(f"  Total time : {total_time:.0f}s")
    print()
    print(f"  Game records  → data/games/")
    print(f"  Opening book  → data/openings/openings.json")

    # ── Optional LLM batch summary ────────────────────────────────────────────
    if args.summary and all_records:
        print()
        print("  Generating batch summary via LLM ...", end="", flush=True)
        try:
            url   = settings.get("ollama_url",   "http://localhost:11434")
            model = settings.get("ollama_model", "llama3.1:8b")
            _mem  = mem or MemoryManager(ollama_url=url, ollama_model=model,
                                         chroma_path=str(ROOT / "data" / "chroma"),
                                         games_path=str(ROOT / "data" / "games"),
                                         session_path=str(ROOT / "data" / "session_memory"))
            from ai.mills_llm import MillsLLM
            _llm = MillsLLM(memory=_mem, ollama_url=url, model=model)
            summary = _llm.summarise_session(all_records[-20:])  # last 20 records
            if summary:
                _mem.save_session_narrative(summary)
                print(" done.")
                print()
                print(summary)
            else:
                print(" (no summary returned)")
        except Exception as exc:
            print(f" failed: {exc}")

    print()


if __name__ == "__main__":
    main()
