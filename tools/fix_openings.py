"""tools/fix_openings.py — Purge corrupt entries from learned_openings.json.

For each opening, replays its move list from the standard start position using
the actual game rules.  If a move is illegal the list is truncated to the last
legal move.  Entries with fewer than 3 surviving moves are removed entirely.

Usage:
    python -m tools.fix_openings [--dry-run]
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent

sys.path.insert(0, str(_ROOT))

from game.board import BoardState
from game.rules import get_all_legal_moves


_OPENINGS_PATH = _ROOT / "data" / "openings" / "learned_openings.json"
MIN_MOVES = 3


def _replay_and_truncate(moves: list[str]) -> list[str]:
    """Replay placement moves from the start; return the longest legal prefix."""
    board = BoardState.new_game()
    legal_prefix: list[str] = []

    for pos in moves:
        legal = get_all_legal_moves(board)
        # All openings consist of placement moves only
        legal_placements = {m["to"] for m in legal if m.get("from") is None}
        if pos not in legal_placements:
            # This move is illegal — stop here
            break

        # Build move; auto-pick first legal capture if a mill is formed
        move: dict = {"from": None, "to": pos, "capture": None}

        # Check if placing here forms a mill (same logic as replay_opening handler)
        from game.board import MILLS
        new_pos = dict(board.positions)
        new_pos[pos] = board.turn
        forms_mill = any(
            pos in mill and all(new_pos[p] == board.turn for p in mill)
            for mill in MILLS
        )
        if forms_mill:
            caps = sorted(board.legal_captures(board.turn))
            if caps:
                move["capture"] = caps[0]

        board = board.apply_move(move)
        legal_prefix.append(pos)

    return legal_prefix


def main(dry_run: bool = False) -> None:
    if not _OPENINGS_PATH.exists():
        print(f"ERROR: {_OPENINGS_PATH} not found.")
        sys.exit(1)

    raw = json.loads(_OPENINGS_PATH.read_text(encoding="utf-8"))
    total = len(raw)
    truncated = 0
    removed = 0
    cleaned: list[dict] = []

    for entry in raw:
        original_moves: list[str] = entry.get("line_moves", [])
        valid_moves = _replay_and_truncate(original_moves)

        if len(valid_moves) < len(original_moves):
            # Truncation needed
            if len(valid_moves) < MIN_MOVES:
                removed += 1
                print(
                    f"  REMOVE  {entry.get('opening_id', '?')} "
                    f"\"{entry.get('name', '?')}\" "
                    f"({len(original_moves)} moves → {len(valid_moves)} after fix)"
                )
                continue
            else:
                truncated += 1
                print(
                    f"  TRUNCATE {entry.get('opening_id', '?')} "
                    f"\"{entry.get('name', '?')}\" "
                    f"({len(original_moves)} moves → {len(valid_moves)})"
                )
                entry = dict(entry)
                entry["line_moves"] = valid_moves
                # Trim FEN signatures beyond the new move count
                sigs = entry.get("opening_fen_signatures", [])
                entry["opening_fen_signatures"] = [
                    s for s in sigs if s.get("ply", 0) <= len(valid_moves)
                ]

        cleaned.append(entry)

    print(
        f"\nSummary: {total} total, {truncated} truncated, {removed} removed, "
        f"{len(cleaned)} remaining."
    )

    if dry_run:
        print("Dry-run mode — no files written.")
        return

    # Back up before overwriting
    backup_path = _OPENINGS_PATH.with_suffix(".json.pre-fix-backup")
    shutil.copy2(_OPENINGS_PATH, backup_path)
    print(f"Backup written to {backup_path.name}")

    _OPENINGS_PATH.write_text(
        json.dumps(cleaned, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Cleaned data written to {_OPENINGS_PATH.name}")


if __name__ == "__main__":
    dry = "--dry-run" in sys.argv
    main(dry_run=dry)
