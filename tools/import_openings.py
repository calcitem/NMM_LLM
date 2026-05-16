"""
tools/import_openings.py — CLI tool to validate and import curated book openings.

Usage:
  python tools/import_openings.py --input raw_openings.json --validate \
      --output data/openings/book_openings.json
  python tools/import_openings.py --input raw_openings.json --dry-run
  python tools/import_openings.py --input raw_openings.json --merge \
      --output data/openings/book_openings.json
"""
from __future__ import annotations

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from game.board import BoardState
from game.rules import get_all_legal_moves


# ── Validation ────────────────────────────────────────────────────────────────

def _validate_opening(opening: dict) -> list[str]:
    """Return list of error strings; empty list means valid."""
    errors: list[str] = []
    oid = opening.get("opening_id", "<unknown>")

    required = [
        "opening_id", "name", "family", "side", "seed_source",
        "line_moves", "confidence",
    ]
    for field in required:
        if field not in opening:
            errors.append(f"{oid}: missing field '{field}'")

    if errors:
        return errors  # can't validate further without required fields

    if opening["side"] not in ("W", "B", "both"):
        errors.append(f"{oid}: 'side' must be 'W', 'B', or 'both'")

    if opening["seed_source"] != "book":
        errors.append(f"{oid}: 'seed_source' must be 'book' for imported openings")

    if opening["confidence"] != 1.0:
        errors.append(f"{oid}: 'confidence' must be 1.0 for book openings")

    if not opening.get("opening_fen_signatures"):
        errors.append(f"{oid}: must have at least one 'opening_fen_signatures' entry")

    # Validate move legality by replaying the sequence
    moves = opening.get("line_moves", [])
    board = BoardState.new_game()
    for i, pos in enumerate(moves):
        legal_placements = board.legal_placements(board.turn)
        if pos not in legal_placements:
            errors.append(
                f"{oid}: move {i+1} '{pos}' is not a legal placement "
                f"(turn={board.turn}, legal={sorted(legal_placements)[:6]}...)"
            )
            break
        board = board.apply_move({"from": None, "to": pos, "capture": None})

    # Validate branch moves
    for branch in opening.get("branch_moves", []):
        if not branch.get("branch_id"):
            errors.append(f"{oid}: branch missing 'branch_id'")
        if branch.get("seed_source") not in ("book", "human", "learned", None):
            errors.append(f"{oid}: branch '{branch.get('branch_id')}' has invalid seed_source")

    return errors


def _generate_fen_signatures(opening: dict) -> list[dict]:
    """Play through line_moves and capture FENs at plies 4, 6, 8, 10."""
    moves = opening.get("line_moves", [])
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
        description="Import and validate curated Nine Men's Morris book openings."
    )
    parser.add_argument("--input", "-i", required=True,
                        help="Path to raw JSON file with opening definitions")
    parser.add_argument("--output", "-o", default="data/openings/book_openings.json",
                        help="Destination path (default: data/openings/book_openings.json)")
    parser.add_argument("--validate", "-v", action="store_true",
                        help="Run legality checks before writing")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print validation results without writing")
    parser.add_argument("--merge", action="store_true",
                        help="Merge into existing output file (keep non-conflicting entries)")
    parser.add_argument("--gen-fens", action="store_true",
                        help="Auto-generate opening_fen_signatures from line_moves")
    args = parser.parse_args(argv)

    # Load input
    try:
        with open(args.input, encoding="utf-8") as f:
            raw: list[dict] = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"ERROR: cannot read {args.input}: {e}")
        return 1

    if not isinstance(raw, list):
        print("ERROR: input JSON must be a list of opening objects")
        return 1

    # Check for duplicate IDs in input
    seen_ids: set[str] = set()
    dup_errors: list[str] = []
    for o in raw:
        oid = o.get("opening_id", "")
        if oid in seen_ids:
            dup_errors.append(f"Duplicate opening_id: '{oid}'")
        seen_ids.add(oid)

    # Generate FEN signatures if requested
    if args.gen_fens:
        for o in raw:
            if not o.get("opening_fen_signatures"):
                o["opening_fen_signatures"] = _generate_fen_signatures(o)

    # Validate
    all_errors = dup_errors[:]
    if args.validate or args.dry_run:
        for opening in raw:
            all_errors.extend(_validate_opening(opening))

    if all_errors:
        print(f"\nValidation FAILED — {len(all_errors)} error(s):\n")
        for err in all_errors:
            print(f"  ✗ {err}")
        print()
        if args.dry_run or not args.validate:
            pass  # dry-run shows errors but doesn't block
        else:
            return 1
    else:
        print(f"Validation passed — {len(raw)} opening(s) OK")

    if args.dry_run:
        print("(dry-run: no files written)")
        return 0

    # Merge with existing if requested
    output: list[dict] = []
    if args.merge and os.path.exists(args.output):
        try:
            with open(args.output, encoding="utf-8") as f:
                existing: list[dict] = json.load(f)
            existing_ids = {o["opening_id"] for o in existing}
            output = existing[:]
            added = 0
            for o in raw:
                if o.get("opening_id") not in existing_ids:
                    output.append(o)
                    added += 1
                else:
                    # Update existing entry
                    output = [o if x["opening_id"] == o["opening_id"] else x
                               for x in output]
                    added += 1
            print(f"Merged: {added} opening(s) added/updated")
        except (OSError, json.JSONDecodeError) as e:
            print(f"WARNING: could not read existing {args.output}: {e}. Overwriting.")
            output = raw
    else:
        output = raw

    # Write output
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(output)} opening(s) to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
