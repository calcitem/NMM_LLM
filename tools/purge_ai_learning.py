#!/usr/bin/env python3
"""
tools/purge_ai_learning.py — Remove AI self-play generated openings and game records.

KEEPS:
  • Openings with seed_source='book'                        (strategy book structure)
  • Openings with seed_source='learned' + source_reference  (book game imports)
  • Human vs AI game records (human_color != 'self_play')
  • bad_moves.json, ChromaDB, player profiles, settings     (user-taught data)

REMOVES:
  • Openings with seed_source='learned' + no source_reference  (AI self-generated)
  • All self-play JSONL files (human_color == 'self_play')

The TrajectoryDB and EndgameDB both rebuild from game files at startup, so
removing self-play game files automatically clears those indexes.

A full backup is written to data/backups/<timestamp>/ before any changes.
Run with --dry-run to preview without changing anything.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT      = Path(__file__).resolve().parent.parent
DATA      = ROOT / "data"
GAMES_DIR = DATA / "games"
OPENINGS  = DATA / "openings" / "openings.json"
BACKUP_ROOT = DATA / "backups"


# ── Categorisation helpers ────────────────────────────────────────────────────

def _categorise_openings(path: Path) -> tuple[list[dict], list[dict]]:
    """Return (keep, purge) lists of opening dicts."""
    raw  = json.loads(path.read_text())
    items = list(raw.values()) if isinstance(raw, dict) else raw
    keep, purge = [], []
    for o in items:
        seed = o.get("seed_source", "")
        ref  = o.get("source_reference", "")
        if seed == "book" or (seed == "learned" and ref):
            keep.append(o)
        else:
            purge.append(o)
    return keep, purge


def _categorise_games(games_dir: Path) -> tuple[list[Path], list[Path]]:
    """Return (keep_paths, purge_paths) for JSONL game files."""
    keep, purge = [], []
    for path in sorted(games_dir.glob("game_*.jsonl")):
        try:
            record = json.loads(path.read_text())
            if record.get("self_play") or record.get("human_color") == "self_play":
                purge.append(path)
            else:
                keep.append(path)
        except Exception:
            keep.append(path)   # unreadable → keep it, don't risk deleting
    return keep, purge


# ── Report ────────────────────────────────────────────────────────────────────

def _print_report(
    keep_openings: list[dict],
    purge_openings: list[dict],
    keep_games: list[Path],
    purge_games: list[Path],
) -> None:
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║         NMM — AI Learning Purge Preview              ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()
    print("─── Opening book ───────────────────────────────────────")
    print(f"  Keep  : {len(keep_openings):4d}  (book-structure + book-game imports)")
    print(f"  Purge : {len(purge_openings):4d}  (AI self-play generated)")
    if purge_openings:
        sample = purge_openings[:5]
        print("  Sample entries to be removed:")
        for o in sample:
            name = (o.get("name") or "unnamed")[:50]
            stats = o.get("outcome_stats", {})
            print(f"    • {name}  stats={stats}")
        if len(purge_openings) > 5:
            print(f"    … and {len(purge_openings) - 5} more")
    print()
    print("─── Game records ───────────────────────────────────────")
    purge_mb = sum(p.stat().st_size for p in purge_games) / 1024 / 1024
    keep_mb  = sum(p.stat().st_size for p in keep_games)  / 1024 / 1024
    print(f"  Keep  : {len(keep_games):4d} files  ({keep_mb:.1f} MB)  [human vs AI]")
    print(f"  Purge : {len(purge_games):4d} files  ({purge_mb:.1f} MB)  [self-play]")
    print()
    print("─── Preserved (not touched) ────────────────────────────")
    print("  data/bad_moves.json       (your bad-move marks)")
    print("  data/chroma/              (vector memory)")
    print("  data/players/             (player profiles)")
    print("  data/settings.json        (settings)")
    print()


# ── Backup ────────────────────────────────────────────────────────────────────

def _make_backup(
    purge_openings: list[dict],
    keep_openings: list[dict],
    purge_games: list[Path],
    timestamp: str,
) -> Path:
    backup_dir = BACKUP_ROOT / timestamp
    backup_dir.mkdir(parents=True, exist_ok=True)

    # 1. Full original openings.json
    shutil.copy2(OPENINGS, backup_dir / "openings_original.json")

    # 2. Purged openings only (for easy restore of just these)
    (backup_dir / "openings_purged_entries.json").write_text(
        json.dumps(purge_openings, indent=2)
    )

    # 3. Self-play game files → backup/games/
    games_backup = backup_dir / "games"
    games_backup.mkdir(exist_ok=True)
    for path in purge_games:
        shutil.move(str(path), games_backup / path.name)

    return backup_dir


# ── Write cleaned openings ────────────────────────────────────────────────────

def _write_cleaned_openings(keep_openings: list[dict]) -> None:
    # Preserve dict-keyed format (opening_id → opening)
    keyed = {o["opening_id"]: o for o in keep_openings}
    OPENINGS.write_text(json.dumps(keyed, indent=2))


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Purge AI self-play learning data")
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Print what would be removed without making any changes",
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip the confirmation prompt",
    )
    args = parser.parse_args()

    if not OPENINGS.exists():
        print(f"ERROR: {OPENINGS} not found.")
        sys.exit(1)

    keep_openings, purge_openings = _categorise_openings(OPENINGS)
    keep_games,    purge_games    = _categorise_games(GAMES_DIR)

    _print_report(keep_openings, purge_openings, keep_games, purge_games)

    if args.dry_run:
        print("Dry run — no changes made.")
        return

    if not purge_openings and not purge_games:
        print("Nothing to purge. Exiting.")
        return

    if not args.yes:
        try:
            answer = input("Proceed? This will move files to a backup and rewrite openings.json. [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            sys.exit(0)
        if answer not in ("y", "yes"):
            print("Aborted.")
            sys.exit(0)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\nCreating backup at data/backups/{timestamp}/ …")
    backup_dir = _make_backup(purge_openings, keep_openings, purge_games, timestamp)
    print(f"  ✓ Backup written to {backup_dir}")

    print("Rewriting openings.json …")
    _write_cleaned_openings(keep_openings)
    print(f"  ✓ {len(keep_openings)} openings kept, {len(purge_openings)} removed")

    print(f"  ✓ {len(purge_games)} self-play game files moved to backup")

    print()
    print("Done. On next server start the TrajectoryDB and EndgameDB will")
    print("rebuild automatically from the remaining human game records.")
    print()
    print(f"To restore: copy files from {backup_dir} back into place.")


if __name__ == "__main__":
    main()
