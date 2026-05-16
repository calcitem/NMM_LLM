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

  # Run 4 games simultaneously across CPU cores (fast mode only):
  python tools/self_play.py --games 40 --no-llm --parallel 4

  # Continuous training run with live board view:
  python tools/self_play.py --games 10 --no-llm --white 4 --black 4 -v
  
  # Parallel self play (only available with no llm)
  python tools/self_play.py --games 20 --no-llm --white 8 --black 8 --parallel 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from collections import Counter
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


_MAX_MOVES   = 300  # hard safety cap
_REPEAT_DRAW = 3    # declare draw when the same FEN appears this many times


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
    game_label: str = "",
) -> dict:
    """Play one game with no LLM calls. Returns a minimal game-record dict."""
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

        fen_counts[fen] += 1
        if fen_counts[fen] >= _REPEAT_DRAW:
            draw_by_repetition = True
            break

        color = board.turn
        ai    = white_ai if color == "W" else black_ai
        rec   = white_rec if color == "W" else black_rec

        recognition   = rec.get_current_result()
        endgame_state = egr.update(board)

        t_move = time.perf_counter()
        move   = ai.choose_move(board, recognition=recognition, endgame_state=endgame_state)
        elapsed_move = time.perf_counter() - t_move

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

        white_rec.update(move.get("to", ""), engine.board)
        black_rec.update(move.get("to", ""), engine.board)

        if verbose:
            color_name = "White" if color == "W" else "Black"
            print(f"\n{'─'*44}")
            print(f"{game_label}Move {move_count:3d}: {color_name} plays {_move_str(move)}  ({elapsed_move:.2f}s)")
            print(engine.board.to_display_grid())

    winner = engine.winner

    # Record opening book outcomes
    for rec_inst in (white_rec, black_rec):
        result = rec_inst.get_current_result()
        if result and result.opening_id and result.status in ("exact", "probable", "transposition"):
            book.update_outcome_stats(result.opening_id, winner=winner or "D")

    # Auto-name novel openings
    for rec_inst in (white_rec, black_rec):
        result = rec_inst.get_current_result()
        if result and result.status == "novel":
            placement_moves = [m["to"] for m in moves_log if m["type"] == "place"]
            if len(placement_moves) >= 6:
                sigs  = _compute_fen_signatures(placement_moves)
                novel = book.save_novel_opening(placement_moves, sigs, outcome=winner)
                if not novel.name or novel.name.startswith("Novel"):
                    novel.name = f"Self-Play Line {novel.opening_id[:6]}"
                book.save_opening(novel)
            break

    return {
        "session_id":       session_id,
        "date":             datetime.now().isoformat(),
        "human_color":      "self_play",
        "winner":           winner,
        "move_count":       move_count,
        "white_difficulty": white_ai.difficulty,
        "black_difficulty": black_ai.difficulty,
        "self_play":        True,
        "draw_repetition":  draw_by_repetition,
        "moves":            moves_log,
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


# ── Parallel worker (module-level — required for multiprocessing pickling) ────

def _parallel_worker(params: dict) -> dict:
    """
    Run one fast-mode game; return the record.
    All objects are created locally so the worker is self-contained across
    process boundaries.  Opening book updates are done in the main process
    after all workers finish, using the winner/opening_id data in the record.
    """
    sys.path.insert(0, str(ROOT))  # ensure import path in child process

    game_num    = params["game_num"]
    white_diff  = params["white_diff"]
    black_diff  = params["black_diff"]
    blunder     = params["blunder"]
    verbose     = params["verbose"]

    book     = OpeningBook()
    white_ai = GameAI(color="W", difficulty=white_diff, blunder_probability=blunder)
    black_ai = GameAI(color="B", difficulty=black_diff)
    label    = f"[Game {game_num}] " if verbose else ""

    record = _run_fast_game(white_ai, black_ai, book, verbose=verbose, game_label=label)
    return record


# ── LLM mode ──────────────────────────────────────────────────────────────────

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

    black_rec = OpeningRecognizer(book)
    black_egr = EndgameRecognizer()

    coord.on_game_start()
    engine     = GameEngine(human_color="B")
    move_count = 0
    fen_counts: Counter = Counter()

    while not engine.finished and move_count < _MAX_MOVES:
        board        = engine.board
        fen          = board.to_fen_string()
        fen_counts[fen] += 1
        if fen_counts[fen] >= _REPEAT_DRAW:
            break
        color        = board.turn
        board_before = board

        t_move = time.perf_counter()
        if color == "W":
            move = coord.deliberate(board)
            engine.apply_move(move)
            black_rec.update(move.get("to", ""), engine.board)
        else:
            black_recognition   = black_rec.get_current_result()
            black_endgame_state = black_egr.update(board)
            move = black_ai.choose_move(
                board, recognition=black_recognition, endgame_state=black_endgame_state
            )
            engine.apply_move(move)
            coord.react_to_human_move(board_before, engine.board, move)
            black_rec.update(move.get("to", ""), engine.board)

        elapsed_move = time.perf_counter() - t_move

        if verbose:
            color_name = "White" if color == "W" else "Black"
            print(f"\n{'─'*44}")
            print(f"Move {move_count+1:3d}: {color_name} plays {_move_str(move)}  ({elapsed_move:.2f}s)")
            print(engine.board.to_display_grid())
            for line in coord.flush_dialogue():
                print(f"  MillsAI: {line}")
        else:
            coord.flush_dialogue()

        move_count += 1

    winner = engine.winner
    record = coord.build_game_record(winner=winner, human_color="self_play")
    record["white_difficulty"] = white_ai.difficulty
    record["black_difficulty"] = black_ai.difficulty
    record["self_play"]        = True
    record["move_count"]       = move_count

    mem.save_game_record(record)

    result = rec.get_current_result()
    if result.opening_id and result.status in ("exact", "probable", "transposition"):
        book.update_outcome_stats(result.opening_id, winner=winner or "D")
    elif result.status == "novel":
        placement_moves = [m["to"] for m in record.get("moves", []) if m.get("type") == "place"]
        if len(placement_moves) >= 6:
            sigs  = _compute_fen_signatures(placement_moves)
            name  = llm.name_novel_opening(placement_moves)
            novel = book.save_novel_opening(placement_moves, sigs, outcome=winner)
            novel.name = name or f"Self-Play Line {novel.opening_id[:6]}"
            book.save_opening(novel)
            if verbose:
                print(f"  Novel opening saved: {novel.name}")

    return record


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Nine Men's Morris AI self-play training loop",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--games",    "-n",  type=int,   default=10,  metavar="N",
                        help="Number of games to play (default: 10)")
    parser.add_argument("--white",           type=int,   default=5,   metavar="D",
                        help="White AI difficulty 1-10 (default: 5)")
    parser.add_argument("--black",           type=int,   default=5,   metavar="D",
                        help="Black AI difficulty 1-10 (default: 5)")
    parser.add_argument("--blunder",         type=float, default=0.0, metavar="P",
                        help="Blunder probability for White 0-1 (default: 0.0)")
    parser.add_argument("--no-llm",          action="store_true",
                        help="Skip all LLM calls — fast mode (recommended for bulk runs)")
    parser.add_argument("--swap",            action="store_true",
                        help="Alternate which AI plays White each game (reduces first-mover bias)")
    parser.add_argument("--summary",         action="store_true",
                        help="Ask LLM for a batch summary after all games finish")
    parser.add_argument("--parallel", "-p",  type=int,   default=1,   metavar="N",
                        help="Run N games simultaneously across CPU cores (fast mode only, default: 1)")
    parser.add_argument("--verbose",  "-v",  action="store_true",
                        help="Print each move with board display and per-move timing")
    args = parser.parse_args()

    settings  = _load_settings()
    n_games   = max(1, args.games)
    w_diff    = max(1, min(10, args.white))
    b_diff    = max(1, min(10, args.black))
    use_llm   = not args.no_llm
    n_workers = max(1, args.parallel)

    if n_workers > 1 and use_llm:
        print("Note: --parallel is only available in fast mode (--no-llm). Ignoring --parallel.")
        n_workers = 1

    print(f"\nNine Men's Morris — Self-Play Training")
    print(f"  Games:       {n_games}")
    print(f"  White diff:  {w_diff}  |  Black diff: {b_diff}")
    print(f"  Mode:        {'LLM commentary' if use_llm else 'fast (no LLM)'}")
    print(f"  Blunder:     {args.blunder:.0%}  |  Swap colours: {args.swap}")
    if n_workers > 1:
        print(f"  Parallel:    {n_workers} workers")
    print()

    results     = {"W": 0, "B": 0, "D": 0}
    total_moves = 0
    total_time  = 0.0
    all_records: list[dict] = []

    # ── Shared objects for sequential fast mode ───────────────────────────────
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
            use_ollama_embeddings=False,
        )

    # ── Parallel fast mode ────────────────────────────────────────────────────
    if n_workers > 1:
        import concurrent.futures

        # Build worker params for every game upfront
        worker_params: list[dict] = []
        for game_num in range(1, n_games + 1):
            if args.swap and game_num % 2 == 0:
                wd, bd = b_diff, w_diff
            else:
                wd, bd = w_diff, b_diff
            worker_params.append({
                "game_num":   game_num,
                "white_diff": wd,
                "black_diff": bd,
                "blunder":    args.blunder,
                "verbose":    args.verbose,
            })

        print(f"  Dispatching {n_games} games across {n_workers} workers …\n")
        t0_all = time.perf_counter()

        completed = 0
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_parallel_worker, p): p for p in worker_params}
            for future in concurrent.futures.as_completed(futures):
                params = futures[future]
                gn     = params["game_num"]
                try:
                    record  = future.result()
                    winner  = record.get("winner")
                    moves   = record.get("move_count", 0)
                    results[winner or "D"] += 1
                    total_moves += moves
                    all_records.append(record)
                    completed += 1
                    label = ("White" if winner == "W" else
                             "Black" if winner == "B" else "Draw ")
                    print(f"  Game {gn:3d}/{n_games}: {label}  ({moves:3d} moves)  "
                          f"[{completed}/{n_games} done]")
                except Exception as exc:
                    print(f"  Game {gn:3d} ERROR: {exc}")

        elapsed_all = time.perf_counter() - t0_all
        total_time  = elapsed_all

        # Consolidate opening book and save records in main process
        print(f"\n  Saving {len(all_records)} game records …")
        for record in all_records:
            try:
                mem.save_game_record(record)  # type: ignore[union-attr]
            except Exception as exc:
                print(f"  Warning: failed to save record: {exc}")

    # ── Sequential mode (fast or LLM) ────────────────────────────────────────
    else:
        for game_num in range(1, n_games + 1):
            if args.swap and game_num % 2 == 0:
                wd, bd = b_diff, w_diff
            else:
                wd, bd = w_diff, b_diff

            white_ai = GameAI(color="W", difficulty=wd, blunder_probability=args.blunder)
            black_ai = GameAI(color="B", difficulty=bd)

            t0 = time.perf_counter()
            print(f"  Game {game_num:3d}/{n_games} …")
            if args.verbose:
                print()

            try:
                if use_llm:
                    record = _run_llm_game(white_ai, black_ai, settings, verbose=args.verbose)
                else:
                    record = _run_fast_game(
                        white_ai, black_ai, book,
                        verbose=args.verbose,
                        game_label=f"[Game {game_num}] ",
                    )
                    mem.save_game_record(record)  # type: ignore[union-attr]
            except KeyboardInterrupt:
                print("\n  Interrupted.")
                break
            except Exception as exc:
                print(f"  ERROR: {exc}")
                if args.verbose:
                    import traceback
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
                print()
            label = "White" if winner == "W" else "Black" if winner == "B" else "Draw "
            print(f"  → {label}  ({moves:3d} moves, {elapsed:.1f}s)\n")

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
    avg_t = total_time / played if n_workers == 1 else total_time / max(n_games, 1)
    print(f"  Avg time   : {avg_t:.1f}s / game  (wall clock)")
    print(f"  Total time : {total_time:.0f}s")
    print()
    print(f"  Game records  → data/games/")
    print(f"  Opening book  → data/openings/openings.json")

    # ── Optional LLM batch summary ────────────────────────────────────────────
    if args.summary and all_records:
        print()
        print("  Generating batch summary via LLM …", end="", flush=True)
        try:
            url   = settings.get("ollama_url",   "http://localhost:11434")
            model = settings.get("ollama_model", "llama3.1:8b")
            _mem  = mem or MemoryManager(
                ollama_url=url, ollama_model=model,
                chroma_path=str(ROOT / "data" / "chroma"),
                games_path=str(ROOT / "data" / "games"),
                session_path=str(ROOT / "data" / "session_memory"),
            )
            from ai.mills_llm import MillsLLM
            _llm    = MillsLLM(memory=_mem, ollama_url=url, model=model)
            summary = _llm.summarise_session(all_records[-20:])
            if summary:
                _mem.save_session_narrative(summary)
                print(" done.\n")
                print(summary)
            else:
                print(" (no summary returned)")
        except Exception as exc:
            print(f" failed: {exc}")

    print()


if __name__ == "__main__":
    main()
