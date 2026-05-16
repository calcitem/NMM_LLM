"""
tools/teach_opening.py — Interactive CLI for teaching the AI a custom opening.

Usage:
  python tools/teach_opening.py
  python tools/teach_opening.py --output data/openings/openings.json

The tool walks through:
  1. Opening metadata (name, family, side, notes)
  2. The main move line, validated against live board rules
  3. Optional branch variations at any ply
  4. Saves with seed_source="human", confidence=0.8 to openings.json
"""
from __future__ import annotations

import argparse
import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from game.board import BoardState
from ai.opening_book import BranchMove, Opening, OpeningBook


# ── Helpers ───────────────────────────────────────────────────────────────────

def _hr() -> None:
    print("\n" + "─" * 50)


def _ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    val = input(f"{prompt}{suffix}: ").strip()
    return val if val else default


def _ask_choice(prompt: str, choices: list[str], default: str) -> str:
    opts = "/".join(choices)
    while True:
        val = _ask(f"{prompt} ({opts})", default=default)
        if val in choices:
            return val
        print(f"  Please enter one of: {opts}")


def _display_board(board: BoardState, ply: int, turn: str) -> None:
    print(f"\n  Board after ply {ply} (next: {'White (W)' if turn == 'W' else 'Black (B)'}):")
    for line in board.to_display_grid().splitlines():
        print(f"    {line}")


def _enter_line(existing_board: BoardState, existing_moves: list[str]) -> list[str]:
    """
    Interactively enter placement moves on top of an existing board state.
    Returns the list of NEW moves entered.
    """
    board = existing_board
    new_moves: list[str] = []
    ply_offset = len(existing_moves)
    turn = board.turn

    print("\n  Enter placement positions one at a time (e.g. d2).")
    print("  Type 'done' to finish, 'undo' to remove last move, 'board' to show grid.\n")

    while True:
        ply = ply_offset + len(new_moves) + 1
        legal = board.legal_placements(turn)
        raw = input(f"  Ply {ply} ({'W' if turn == 'W' else 'B'}): ").strip().lower()

        if raw == "done":
            break
        if raw == "board":
            _display_board(board, ply - 1, turn)
            continue
        if raw == "undo":
            if not new_moves:
                print("  Nothing to undo.")
                continue
            removed = new_moves.pop()
            # Replay from scratch to undo
            board = existing_board
            for m in new_moves:
                t = board.turn
                board = board.apply_move({"from": None, "to": m, "capture": None})
            turn = board.turn
            print(f"  Removed: {removed}")
            continue
        if raw not in legal:
            print(f"  '{raw}' is not legal. Legal: {sorted(legal)}")
            continue

        board = board.apply_move({"from": None, "to": raw, "capture": None})
        # If the placement formed a mill, skip capture (teaching mode — simplified)
        turn = board.turn
        new_moves.append(raw)
        print(f"  ✓ {raw}")

    return new_moves


def _enter_branch(
    base_moves: list[str], branch_ply: int
) -> tuple[list[str], str] | None:
    """
    Let the user enter a variation starting from ply `branch_ply`.
    Returns (new_moves_from_branch_ply, variation_name) or None if aborted.
    """
    # Replay up to (branch_ply - 1) moves to get the pre-branch board.
    board = BoardState.new_game()
    for m in base_moves[: branch_ply - 1]:
        board = board.apply_move({"from": None, "to": m, "capture": None})

    print(f"\n  Enter an alternative move at ply {branch_ply} "
          f"(book plays '{base_moves[branch_ply - 1]}'):")
    legal = board.legal_placements(board.turn)
    alt = input(f"  Ply {branch_ply} ({'W' if board.turn == 'W' else 'B'}): ").strip().lower()
    if not alt or alt == "done":
        return None
    if alt not in legal:
        print(f"  '{alt}' is not legal. Legal: {sorted(legal)}")
        return None
    if alt == base_moves[branch_ply - 1]:
        print("  That's the same as the book move — no variation needed.")
        return None

    board = board.apply_move({"from": None, "to": alt, "capture": None})
    var_moves = [alt]

    print("  Continue the variation (or 'done' to stop here):")
    more = _enter_line(board, base_moves[: branch_ply - 1] + [alt])
    var_moves.extend(more)

    var_name = _ask("  Variation name", default=f"Variation at ply {branch_ply}")
    return var_moves, var_name


def _make_fen_signatures(moves: list[str]) -> list[dict]:
    board = BoardState.new_game()
    sigs = []
    for i, pos in enumerate(moves):
        board = board.apply_move({"from": None, "to": pos, "capture": None})
        ply = i + 1
        if ply in (4, 6, 8, 10):
            sigs.append({"ply": ply, "fen": board.to_fen_string()})
    return sigs


# ── Main ──────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Interactively teach the AI a custom Nine Men's Morris opening."
    )
    parser.add_argument(
        "--output", "-o",
        default="data/openings/openings.json",
        help="Destination openings.json (default: data/openings/openings.json)",
    )
    args = parser.parse_args(argv)

    print("\n╔══════════════════════════════════════╗")
    print("║  Nine Men's Morris — Teach Opening   ║")
    print("╚══════════════════════════════════════╝\n")
    print("You'll enter the opening's metadata, then play through the move line.")
    print("Type 'done' at any move prompt to finish the line.\n")

    # ── Metadata ──────────────────────────────────────────────────────────────
    _hr()
    print("  METADATA\n")
    name = _ask("  Opening name (e.g. 'Corner Rush Defence')")
    if not name:
        print("Name is required.")
        return 1

    family = _ask("  Family (e.g. 'Mill Rush', 'Corner Gambit')", default="Custom")
    side = _ask_choice("  Designed for which side", ["W", "B", "both"], default="both")
    strategic_notes = _ask("  Strategic notes (optional)", default="")
    blunders_raw = _ask("  Common blunders to avoid (comma-separated positions, optional)", default="")
    common_blunders = [b.strip() for b in blunders_raw.split(",") if b.strip()]
    tags_raw = _ask("  Tags (comma-separated, optional)", default="")
    tags = [t.strip() for t in tags_raw.split(",") if t.strip()]
    tags = list(set(tags + ["human"]))

    # ── Main line ─────────────────────────────────────────────────────────────
    _hr()
    print("  MAIN LINE\n")
    board = BoardState.new_game()
    line_moves = _enter_line(board, [])

    if len(line_moves) < 2:
        print("\nNeed at least 2 moves to define an opening. Aborting.")
        return 1

    print(f"\n  Main line ({len(line_moves)} moves): {', '.join(line_moves)}")

    # ── Variations ────────────────────────────────────────────────────────────
    branch_moves: list[BranchMove] = []

    while True:
        _hr()
        add_var = _ask_choice(
            "  Add a variation?", ["y", "n"], default="n"
        )
        if add_var != "y":
            break

        if len(line_moves) < 2:
            print("  Need at least 2 main-line moves to branch from.")
            break

        ply_str = _ask(
            f"  Branch at which ply? (1–{len(line_moves)})", default="2"
        )
        try:
            branch_ply = int(ply_str)
            if not (1 <= branch_ply <= len(line_moves)):
                raise ValueError
        except ValueError:
            print(f"  Please enter a number between 1 and {len(line_moves)}.")
            continue

        result = _enter_branch(line_moves, branch_ply)
        if result is None:
            print("  Variation skipped.")
            continue

        var_moves, var_name = result
        var_notes = _ask("  Strategic notes for this variation", default="")
        branch_id = f"branch-{uuid.uuid4().hex[:8]}"
        branch = BranchMove(
            branch_id=branch_id,
            deviation_ply=branch_ply,
            deviation_move=var_moves[0],
            name=var_name,
            line_continuation=var_moves[1:],
            strategic_notes=var_notes,
            seed_source="human",
            outcome_stats={"W": 0, "B": 0, "D": 0},
        )
        branch_moves.append(branch)
        print(f"  ✓ Variation '{var_name}' added ({len(var_moves)} move(s) from branch point).")

    # ── Build and save ────────────────────────────────────────────────────────
    _hr()
    print("  SUMMARY\n")
    print(f"  Name:       {name}")
    print(f"  Family:     {family}")
    print(f"  Side:       {side}")
    print(f"  Moves:      {', '.join(line_moves)}")
    print(f"  Variations: {len(branch_moves)}")
    if strategic_notes:
        print(f"  Notes:      {strategic_notes}")

    confirm = _ask_choice("\n  Save this opening?", ["y", "n"], default="y")
    if confirm != "y":
        print("Aborted — nothing saved.")
        return 0

    opening_id = f"human-{name.lower().replace(' ', '-')}-{uuid.uuid4().hex[:6]}"
    fen_sigs = _make_fen_signatures(line_moves)

    opening = Opening(
        opening_id=opening_id,
        name=name,
        aliases=[],
        family=family,
        side=side,
        seed_source="human",
        line_moves=line_moves,
        branch_moves=branch_moves,
        opening_fen_signatures=fen_sigs,
        strategic_notes=strategic_notes,
        common_blunders=common_blunders,
        recommended_responses={"W": [], "B": []},
        outcome_stats={"W": 0, "B": 0, "D": 0},
        confidence=0.8,
        tags=tags,
    )

    book = OpeningBook(
        book_path="data/openings/book_openings.json",
        openings_path=args.output,
    )
    book.save_opening(opening)

    print(f"\n  ✓ Saved '{name}' ({opening_id}) to {args.output}")
    print(f"    {len(line_moves)} moves · {len(branch_moves)} variation(s) · "
          f"{len(fen_sigs)} FEN signatures\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
