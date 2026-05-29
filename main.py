"""
main.py — Console entry point for Nine Men's Morris.

Usage:
    python main.py                        # Human (W) vs AI difficulty 3
    python main.py --difficulty 5         # harder AI
    python main.py --human B              # Human plays Black, AI plays White
    python main.py --blunder 0.3          # AI blunders ~30% of moves (training mode)
    python main.py --hvh                  # Human vs Human
    python main.py --no-llm               # Disable LLM commentary entirely
"""

from __future__ import annotations

import argparse
import json
import os

from game.board import BOARD_REFERENCE
from game.game_engine import GameEngine
from game.rules import get_all_legal_moves, get_game_phase
from ai.game_ai import GameAI
from ai.memory_manager import MemoryManager
from ai.mills_llm import MillsLLM
from ai.coordinator import Coordinator
from ai.opening_book import OpeningBook
from ai.opening_recognizer import OpeningRecognizer
from ai.endgame_recognizer import EndgameRecognizer

# ── AI engine selection ────────────────────────────────────────────────────────
# Default is the heuristic minimax engine (unchanged behaviour). Set
# NMM_AI_ENGINE=learned to use the trained neural agent instead.
# See docs/MIGRATION_GUIDE.md.
AI_ENGINE = os.environ.get("NMM_AI_ENGINE", "heuristic")
LEARNED_CHECKPOINT = os.environ.get(
    "NMM_LEARNED_CHECKPOINT", "learned_ai/checkpoints/latest.pt"
)


def _make_learned_ai(ai_color: str):
    """Build a LearnedAgent, or return None (with a warning) on any failure.

    Falls back to the heuristic engine if PyTorch is missing or the checkpoint
    cannot be loaded, so play is never blocked.
    """
    try:
        from learned_ai.agents.learned_agent import LearnedAgent
    except Exception as exc:  # torch not installed, etc.
        print(f"[NMM] Learned AI unavailable ({exc}); using heuristic engine.")
        return None
    if not os.path.exists(LEARNED_CHECKPOINT):
        print(
            f"[NMM] Checkpoint {LEARNED_CHECKPOINT} not found; "
            "using heuristic engine."
        )
        return None
    try:
        return LearnedAgent(
            color=ai_color,
            checkpoint_path=LEARNED_CHECKPOINT,
            mode="argmax",
        )
    except Exception as exc:
        print(f"[NMM] Failed to load learned AI ({exc}); using heuristic engine.")
        return None


def _load_settings(path: str = "data/settings.json") -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


# ── Prompt helpers ────────────────────────────────────────────────────────────

def _prompt_placement(engine: GameEngine) -> dict:
    board = engine.board
    color = board.turn
    legal = board.legal_placements(color)
    while True:
        raw = input("  Place piece at (e.g. d2): ").strip().lower()
        if raw not in legal:
            print(f"  ! '{raw}' is not a legal placement. Legal: {sorted(legal)}")
            continue
        return {"from": None, "to": raw, "capture": None}


def _prompt_movement(engine: GameEngine) -> dict:
    board = engine.board
    color = board.turn
    phase = get_game_phase(board, color)
    legal_pairs = set(board.legal_moves(color))
    legal_srcs = sorted({s for s, _ in legal_pairs})
    while True:
        raw = input(
            f"  {'Fly' if phase == 'fly' else 'Move'} piece (e.g. c5-c4): "
        ).strip().lower()
        if "-" not in raw:
            print("  ! Format must be src-dst, e.g. c5-c4")
            continue
        src, dst = raw.split("-", 1)
        if (src, dst) not in legal_pairs:
            print(f"  ! '{raw}' is not a legal move. Legal sources: {legal_srcs}")
            continue
        return {"from": src, "to": dst, "capture": None}


def _prompt_capture(engine: GameEngine) -> str:
    board = engine.board
    color = board.turn
    legal = board.legal_captures(color)
    print(f"  Mill formed! Legal captures: {sorted(legal)}")
    while True:
        raw = input("  Capture: ").strip().lower()
        if raw not in legal:
            print(f"  ! '{raw}' is not a legal capture.")
            continue
        return raw


# ── Game loop ─────────────────────────────────────────────────────────────────

def _print_dialogue(lines: list[str]) -> None:
    for line in lines:
        print(f"  {line}")


def run_game(
    human_color: str = "W",
    difficulty: int = 3,
    blunder_probability: float = 0.0,
    vs_human: bool = False,
    use_llm: bool = True,
) -> None:
    settings = _load_settings()
    ollama_url = settings.get("ollama_url", "http://localhost:11434")
    ollama_model = settings.get("ollama_model", "llama3.1:8b")
    poor_move_threshold = settings.get("poor_move_threshold", 0.3)
    max_comments = settings.get("max_poor_move_comments_per_game", 5)

    ai_color = "B" if human_color == "W" else "W"
    engine = GameEngine(human_color=human_color)

    game_ai: GameAI | None = None
    coordinator: Coordinator | None = None

    # Engine selection: heuristic (default) or learned. See docs/MIGRATION_GUIDE.md.
    if not vs_human:
        learned_ai = _make_learned_ai(ai_color) if AI_ENGINE == "learned" else None
        if learned_ai is not None:
            # The learned agent exposes choose_move(board) / last_was_blunder,
            # so it is a drop-in for GameAI. LLM commentary (which depends on
            # GameAI internals) is disabled when the learned engine is active.
            game_ai = learned_ai
            print(f"[NMM] Using learned AI engine ({LEARNED_CHECKPOINT})")
        else:
            game_ai = GameAI(
                color=ai_color,
                difficulty=difficulty,
                blunder_probability=blunder_probability,
            )
        if use_llm and learned_ai is None:
            memory = MemoryManager(
                ollama_url=ollama_url,
                ollama_model=ollama_model,
            )
            mills_llm = MillsLLM(
                memory=memory,
                ollama_url=ollama_url,
                model=ollama_model,
            )
            opening_book = OpeningBook()
            opening_recognizer = OpeningRecognizer(opening_book)
            endgame_recognizer = EndgameRecognizer(
                active_threshold=settings.get("endgame_active_threshold", 11),
                deep_threshold=settings.get("endgame_deep_threshold", 8),
                zugzwang_threshold=settings.get("endgame_zugzwang_threshold", 0.4),
            )
            coordinator = Coordinator(
                game_ai=game_ai,
                mills_llm=mills_llm,
                memory=memory,
                poor_move_threshold=poor_move_threshold,
                max_poor_move_comments=max_comments,
                opening_recognizer=opening_recognizer,
                endgame_recognizer=endgame_recognizer,
            )
            coordinator.on_game_start()

    print("\n═══ Nine Men's Morris ═══\n")
    print(BOARD_REFERENCE)
    if vs_human:
        print("\nHuman vs Human")
    else:
        mode = f"difficulty {difficulty}"
        if blunder_probability > 0:
            mode += f", blunder rate {blunder_probability:.0%}"
        llm_status = "LLM on" if (use_llm and coordinator) else "LLM off"
        print(f"\nYou are {'White (W)' if human_color == 'W' else 'Black (B)'}  |  AI: {mode}  |  {llm_status}")
    print()

    while not engine.finished:
        board = engine.board
        color = board.turn
        phase = get_game_phase(board, color)
        name = "White" if color == "W" else "Black"
        is_human_turn = vs_human or (color == human_color)

        print(engine.status_line())
        print(board.to_display_grid())

        if is_human_turn:
            print(f"\n{name}'s turn [{phase}]")
            board_before = board
            if phase == "place":
                move = _prompt_placement(engine)
            else:
                move = _prompt_movement(engine)
            if engine.move_forms_mill(move):
                cap = _prompt_capture(engine)
                move["capture"] = cap

            engine.apply_move(move)

            if coordinator and not vs_human:
                board_after = engine.board
                coordinator.react_to_human_move(board_before, board_after, move)
                _print_dialogue(coordinator.flush_dialogue())
        else:
            assert game_ai is not None
            print(f"\nAI ({name}) thinking... [{phase}]")

            if coordinator:
                move = coordinator.deliberate(board)
            else:
                move = game_ai.choose_move(board)

            move_str = (
                f"{move.get('from')}-{move['to']}"
                if move.get("from")
                else move["to"]
            )
            if move.get("capture"):
                move_str += f"x{move['capture']}"
            if game_ai.last_was_blunder:
                print(f"  AI plays: {move_str}  ← deliberate mistake!")
            else:
                print(f"  AI plays: {move_str}")

            if coordinator:
                _print_dialogue(coordinator.flush_dialogue())

            engine.apply_move(move)

        print()

    print(engine.board.to_display_grid())
    winner_name = "White" if engine.winner == "W" else "Black"
    print(f"\n{'═' * 40}")
    print(f"  Game over — {winner_name} wins!")
    print(f"{'═' * 40}\n")
    print(engine.export())

    if coordinator:
        record = coordinator.build_game_record(
            winner=engine.winner,
            human_color=human_color,
        )
        coordinator.on_game_end(record)
        _print_dialogue(coordinator.flush_dialogue())

        if settings.get("auto_open_debrief", False):
            from ai.debriefer import GameDebriefer
            debriefer = GameDebriefer(
                mills_llm=coordinator.mills_llm,
                analysis_depth=settings.get("debrief_analysis_depth", 4),
                critical_threshold=settings.get("debrief_critical_threshold", 0.4),
            )
            print("\nAnalysing game...", end="", flush=True)
            report = debriefer.analyse(record)
            print(" done.")
            debriefer.print_report(report)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Nine Men's Morris console")
    p.add_argument("--difficulty", "-d", type=int, default=3, choices=range(1, 6),
                   help="AI difficulty 1-5 (default 3)")
    p.add_argument("--human", "-p", default="W", choices=["W", "B"],
                   help="Human plays W or B (default W)")
    p.add_argument("--blunder", "-b", type=float, default=0.0, metavar="PROB",
                   help="AI blunder probability 0.0-1.0 (default 0, training mode)")
    p.add_argument("--hvh", action="store_true",
                   help="Human vs Human (no AI)")
    p.add_argument("--no-llm", action="store_true",
                   help="Disable LLM commentary (faster startup)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_game(
        human_color=args.human,
        difficulty=args.difficulty,
        blunder_probability=args.blunder,
        vs_human=args.hvh,
        use_llm=not args.no_llm,
    )
