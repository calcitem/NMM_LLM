#!/usr/bin/env python3
"""tools/import_playok.py — Import PlayOK Nine Men's Morris game archives.

Converts PlayOK .txt game files (PGN-like format, positions 1–24) into the
project's JSONL format (named positions a7/d7/…, FEN strings, per-move dicts).
Output goes to data/human_games/; already-imported games are skipped via
a manifest at data/human_games/imported.json.

Position mapping (PlayOK 1-24 → project notation, verified against live games):
    Row 1 (top):    1=a7  2=d7  3=g7
    Row 2:          4=b6  5=d6  6=f6
    Row 3:          7=c5  8=d5  9=e5
    Row 4 (middle): 10=a4 11=b4 12=c4  13=e4 14=f4 15=g4
    Row 5:          16=c3 17=d3 18=e3
    Row 6:          19=b2 20=d2 21=f2
    Row 7 (bottom): 22=a1 23=d1 24=g1

Usage:
    python tools/import_playok.py \\
        --archive ~/playok_archive/games \\
        --output  data/human_games       \\
        [--dry-run]        # count only, no files written
        [--validate-only]  # check all moves are legal, report errors
        [--limit N]        # stop after N new games (testing)
        [--verbose]        # print per-game status
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterator

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from game.board import BoardState, POSITIONS

# ── Position mapping ──────────────────────────────────────────────────────────

PLAYOK_TO_PROJECT: dict[int, str] = {
    1: "a7",  2: "d7",  3: "g7",
    4: "b6",  5: "d6",  6: "f6",
    7: "c5",  8: "d5",  9: "e5",
    10: "a4", 11: "b4", 12: "c4", 13: "e4", 14: "f4", 15: "g4",
    16: "c3", 17: "d3", 18: "e3",
    19: "b2", 20: "d2", 21: "f2",
    22: "a1", 23: "d1", 24: "g1",
}

_PROJECT_TO_PLAYOK = {v: k for k, v in PLAYOK_TO_PROJECT.items()}

# Verify all 24 project positions are covered
assert set(PLAYOK_TO_PROJECT.values()) == set(POSITIONS), (
    "PLAYOK_TO_PROJECT does not cover all project positions"
)


def _pos(n: int) -> str:
    """Convert PlayOK integer to project position name; raise on unknown."""
    try:
        return PLAYOK_TO_PROJECT[n]
    except KeyError:
        raise ValueError(f"Unknown PlayOK position number: {n}")


# ── PlayOK file parser ────────────────────────────────────────────────────────

_HEADER_RE = re.compile(r'^\[(\w+)\s+"([^"]*)"\]')
_RESULT_TOKENS = {"1-0", "0-1", "1/2-1/2", "*"}

# placement token: digits, optionally followed by xDigits (mill capture)
_PLACE_RE = re.compile(r'^(\d+)(?:x(\d+))?$')
# movement token: from-to, optionally followed by xDigits
_MOVE_RE   = re.compile(r'^(\d+)-(\d+)(?:x(\d+))?$')


def _parse_result(result_str: str) -> str | None:
    if result_str == "1-0":
        return "W"
    if result_str == "0-1":
        return "B"
    return None   # draw or unknown


def _tokenize_moves(body: str) -> list[str]:
    """Strip turn numbers and return a flat list of move tokens."""
    # Remove turn numbers like "10." or "123."
    cleaned = re.sub(r'\d+\.', ' ', body)
    tokens = cleaned.split()
    return tokens


_DOUBLE_CAP_RE = re.compile(r'^\d+x\d+x\d+$')   # e.g. 3x9x23 — double-mill placement


def _classify_token(tok: str) -> str:
    """Return 'result', 'place', 'move', 'double_cap', or raise ValueError."""
    if tok in _RESULT_TOKENS:
        return "result"
    if _DOUBLE_CAP_RE.match(tok):
        return "double_cap"   # rare variant — caller decides how to handle
    if _MOVE_RE.match(tok):
        return "move"
    if _PLACE_RE.match(tok):
        return "place"
    raise ValueError(f"Unrecognised token: {tok!r}")


class ParseError(Exception):
    pass


def parse_playok_file(path: Path) -> dict:
    """Parse one PlayOK .txt file into a raw dict (pre-FEN-reconstruction)."""
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    headers: dict[str, str] = {}
    body_lines: list[str] = []
    in_body = False

    for line in lines:
        line = line.strip()
        if not line:
            if headers:
                in_body = True
            continue
        if line.startswith("[") and not in_body:
            m = _HEADER_RE.match(line)
            if m:
                headers[m.group(1)] = m.group(2)
        else:
            in_body = True
            body_lines.append(line)

    game_type = headers.get("GameType", "")
    if not game_type.startswith("70"):
        raise ParseError(f"Not a standard NMM game: GameType={game_type!r}")

    result_str = headers.get("Result", "*")
    winner = _parse_result(result_str)

    date_str = headers.get("Date", "").replace(".", "-")

    tokens = _tokenize_moves(" ".join(body_lines))

    # Build a flat list of (color, token) pairs.
    # PlayOK alternates W / B within each turn number.
    half_moves: list[str] = []
    for tok in tokens:
        kind = _classify_token(tok)
        if kind == "result":
            break
        if kind == "double_cap":
            raise ParseError(f"double-mill capture not supported: {tok!r}")
        half_moves.append(tok)

    return {
        "game_id":     path.stem,   # e.g. "ml11756018"
        "date":        date_str,
        "white":       headers.get("White", ""),
        "black":       headers.get("Black", ""),
        "white_elo":   _safe_int(headers.get("WhiteElo")),
        "black_elo":   _safe_int(headers.get("BlackElo")),
        "winner":      winner,
        "result_raw":  result_str,
        "half_moves":  half_moves,   # flat list of tokens, W then B alternating
    }


def _safe_int(s: str | None) -> int | None:
    try:
        return int(s) if s else None
    except ValueError:
        return None


# ── FEN reconstruction ────────────────────────────────────────────────────────

def _move_type(board: BoardState, from_pos: str | None) -> str:
    """Determine move type label given the current board and from_pos."""
    if from_pos is None:
        return "place"
    color = board.turn
    if board.pieces_placed[color] >= 9 and board.pieces_on_board[color] == 3:
        return "fly"
    return "move"


def _build_notation(from_pos: str | None, to_pos: str, capture: str | None) -> str:
    if from_pos is None:
        base = to_pos
    else:
        base = f"{from_pos}-{to_pos}"
    return f"{base}x{capture}" if capture else base


def reconstruct_game(raw: dict) -> list[dict]:
    """
    Replay half_moves against a BoardState, returning a list of move dicts
    each containing board_fen_before and all fields needed by the sentinel.
    """
    board = BoardState.new_game()  # empty board, White to move
    half_moves = raw["half_moves"]
    moves_out: list[dict] = []

    colors = ["W", "B"]
    color_idx = 0   # index into colors; White goes first

    turn_number = 1   # 1-indexed; increments every two half-moves

    for i, tok in enumerate(half_moves):
        color = colors[color_idx]

        fen_before = board.to_fen_string()

        # Parse the token
        mm = _MOVE_RE.match(tok)
        pm = _PLACE_RE.match(tok)

        if mm:
            from_pos  = _pos(int(mm.group(1)))
            to_pos    = _pos(int(mm.group(2)))
            cap_raw   = mm.group(3)
            capture   = _pos(int(cap_raw)) if cap_raw else None
        elif pm:
            from_pos  = None
            to_pos    = _pos(int(pm.group(1)))
            cap_raw   = pm.group(2)
            capture   = _pos(int(cap_raw)) if cap_raw else None
        else:
            raise ParseError(f"Cannot parse token {tok!r} at half-move {i}")

        move_dict = {"from": from_pos, "to": to_pos, "capture": capture}
        mtype     = _move_type(board, from_pos)
        notation  = _build_notation(from_pos, to_pos, capture)

        # Advance board
        try:
            board = board.apply_move(move_dict)
        except Exception as exc:
            raise ParseError(
                f"Illegal move {tok!r} ({notation}) at half-move {i}: {exc}"
            )

        moves_out.append({
            "turn":             turn_number,
            "color":            color,
            "type":             mtype,
            "from":             from_pos,
            "to":               to_pos,
            "capture":          capture,
            "notation":         notation,
            "board_fen_before": fen_before,
        })

        color_idx = 1 - color_idx
        if color_idx == 0:
            turn_number += 1

    return moves_out


# ── JSONL record assembly ─────────────────────────────────────────────────────

def build_record(raw: dict, moves: list[dict]) -> dict:
    return {
        "session_id":   raw["game_id"],
        "source":       "playok",
        "source_type":  "human_vs_human",
        "date":         raw["date"],
        "white_player": raw["white"],
        "black_player": raw["black"],
        "white_elo":    raw["white_elo"],
        "black_elo":    raw["black_elo"],
        "human_color":  None,          # both players are human
        "winner":       raw["winner"],
        "draw_reason":  None,
        "result_raw":   raw["result_raw"],
        "moves":        moves,
    }


# ── Discovery + dedup ─────────────────────────────────────────────────────────

def iter_txt_files(archive_root: Path) -> Iterator[Path]:
    """Yield every .txt game file under the archive directory tree."""
    for p in sorted(archive_root.rglob("*.txt")):
        yield p


def load_manifest(manifest_path: Path) -> dict[str, str]:
    if manifest_path.exists():
        try:
            return json.loads(manifest_path.read_text())
        except Exception:
            return {}
    return {}


def save_manifest(manifest_path: Path, manifest: dict[str, str]) -> None:
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))


# ── Main import logic ─────────────────────────────────────────────────────────

def run_import(
    archive_root: Path,
    output_dir: Path,
    *,
    dry_run: bool = False,
    validate_only: bool = False,
    limit: int | None = None,
    verbose: bool = False,
) -> dict:
    manifest_path = output_dir / "imported.json"
    manifest = load_manifest(manifest_path)

    stats = {
        "found":    0,
        "skipped":  0,
        "imported": 0,
        "errors":   0,
        "draws":    0,
    }
    errors: list[str] = []

    for txt_path in iter_txt_files(archive_root):
        game_id = txt_path.stem
        stats["found"] += 1

        if game_id in manifest:
            stats["skipped"] += 1
            continue

        # Parse headers + move tokens
        try:
            raw = parse_playok_file(txt_path)
        except Exception as exc:
            stats["errors"] += 1
            errors.append(f"{txt_path.name}: parse error — {exc}")
            if verbose:
                print(f"  ERROR  {txt_path.name}: {exc}", file=sys.stderr)
            continue

        # Reconstruct FEN sequence
        try:
            moves = reconstruct_game(raw)
        except Exception as exc:
            stats["errors"] += 1
            errors.append(f"{txt_path.name}: replay error — {exc}")
            if verbose:
                print(f"  ERROR  {txt_path.name}: {exc}", file=sys.stderr)
            continue

        if raw["winner"] is None:
            stats["draws"] += 1

        if not validate_only and not dry_run:
            record = build_record(raw, moves)
            out_path = output_dir / f"human_{game_id}.jsonl"
            out_path.write_text(json.dumps(record))
            manifest[game_id] = datetime.utcnow().isoformat()

        stats["imported"] += 1
        if verbose:
            print(f"  OK     {txt_path.name}  ({len(moves)} half-moves, winner={raw['winner']})")

        if limit and stats["imported"] >= limit:
            break

    if not dry_run and not validate_only and stats["imported"] > 0:
        save_manifest(manifest_path, manifest)

    stats["error_details"] = errors
    return stats


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--archive", default=str(Path.home() / "playok_archive" / "games"),
                   help="Root of the PlayOK archive tree (default: ~/playok_archive/games)")
    p.add_argument("--output",  default=str(_ROOT / "data" / "human_games"),
                   help="Output directory for JSONL files (default: data/human_games)")
    p.add_argument("--dry-run",       action="store_true", help="Count games; don't write files")
    p.add_argument("--validate-only", action="store_true", help="Check legality; don't write files")
    p.add_argument("--limit", type=int, default=None, help="Stop after N new games")
    p.add_argument("--verbose", "-v", action="store_true", help="Per-game status lines")
    args = p.parse_args()

    archive_root = Path(args.archive).expanduser().resolve()
    output_dir   = Path(args.output).expanduser().resolve()

    if not archive_root.exists():
        print(f"ERROR: archive directory not found: {archive_root}", file=sys.stderr)
        sys.exit(1)

    if not args.dry_run and not args.validate_only:
        output_dir.mkdir(parents=True, exist_ok=True)

    mode = "DRY RUN" if args.dry_run else ("VALIDATE" if args.validate_only else "IMPORT")
    print(f"[{mode}] archive={archive_root}  output={output_dir}")

    stats = run_import(
        archive_root, output_dir,
        dry_run=args.dry_run,
        validate_only=args.validate_only,
        limit=args.limit,
        verbose=args.verbose,
    )

    print(f"\nResults:")
    print(f"  Files found:  {stats['found']}")
    print(f"  Skipped:      {stats['skipped']}  (already imported)")
    print(f"  Imported:     {stats['imported']}")
    print(f"  Draws:        {stats['draws']}")
    print(f"  Errors:       {stats['errors']}")

    if stats["error_details"] and (args.verbose or args.validate_only):
        print("\nErrors:")
        for e in stats["error_details"][:20]:
            print(f"  {e}")
        if len(stats["error_details"]) > 20:
            print(f"  ... and {len(stats['error_details']) - 20} more")


if __name__ == "__main__":
    main()
