"""
tools/debrief.py — CLI to run a post-game debrief on saved game records.

Usage:
  python tools/debrief.py                     # Debrief the most recent game
  python tools/debrief.py --list              # List all saved game records
  python tools/debrief.py --game <path>       # Debrief a specific .jsonl file
  python tools/debrief.py --no-llm            # Analysis only, no LLM commentary
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ai.debriefer import GameDebriefer
from ai.memory_manager import MemoryManager
from ai.mills_llm import MillsLLM


_GAMES_DIR = Path("data/games")
_SETTINGS  = Path("data/settings.json")


def _load_settings() -> dict:
    try:
        return json.loads(_SETTINGS.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _list_games() -> list[Path]:
    if not _GAMES_DIR.exists():
        return []
    return sorted(_GAMES_DIR.glob("*.jsonl"), reverse=True)


def _read_game(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def _print_list(games: list[Path]) -> None:
    if not games:
        print("No saved game records found in data/games/")
        return
    print(f"\n  {'#':>3}  {'File':<40}  {'Date':<12}  Winner")
    print("  " + "─" * 70)
    for i, path in enumerate(games, 1):
        records = _read_game(path)
        if not records:
            continue
        rec = records[-1]
        date = rec.get("date", "")[:10]
        winner = rec.get("winner") or "?"
        winner_label = {"W": "White", "B": "Black"}.get(winner, winner)
        print(f"  {i:>3}  {path.name:<40}  {date:<12}  {winner_label}")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Post-game debrief for Nine Men's Morris."
    )
    parser.add_argument("--list", action="store_true",
                        help="List all saved game records and exit")
    parser.add_argument("--game", "-g", metavar="PATH",
                        help="Path to a specific game .jsonl file")
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip LLM commentary (analysis only)")
    args = parser.parse_args(argv)

    if args.list:
        _print_list(_list_games())
        return 0

    # Resolve game path
    if args.game:
        game_path = Path(args.game)
    else:
        games = _list_games()
        if not games:
            print("No saved game records found. Play a game first.")
            return 1
        game_path = games[0]
        print(f"  Using most recent game: {game_path.name}")

    records = _read_game(game_path)
    if not records:
        print(f"ERROR: No game records found in {game_path}")
        return 1

    # Use the last record in the file (most recent game in that session)
    game_record = records[-1]

    settings = _load_settings()

    # Build LLM (or stub if --no-llm)
    if args.no_llm:
        mem = MemoryManager(use_ollama_embeddings=False)
        llm = MillsLLM(memory=mem, model="")
        llm._client = None
    else:
        ollama_url   = settings.get("ollama_url", "http://localhost:11434")
        ollama_model = settings.get("ollama_model", "llama3.1:8b")
        mem = MemoryManager(ollama_url=ollama_url, ollama_model=ollama_model)
        llm = MillsLLM(memory=mem, ollama_url=ollama_url, model=ollama_model)

    debriefer = GameDebriefer(
        mills_llm=llm,
        analysis_depth=settings.get("debrief_analysis_depth", 4),
        critical_threshold=settings.get("debrief_critical_threshold", 0.4),
    )

    total = game_record.get("moves", [])
    print(f"  Analysing {len(total)} moves", end="", flush=True)
    report = debriefer.analyse(game_record)
    print(" — done.")

    debriefer.print_report(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
