#!/usr/bin/env python3
"""
tools/import_book_games.py — Import game sequences from 'book games.docx'
into the game's JSONL record format AND the opening book.

Each table in the docx is one game.  Moves are validated against the engine;
any table with an illegal move is skipped.  Games that begin in the movement
phase (no starting position known) are also skipped.

Opening recognition runs automatically.  Any game with >= 10 placement
half-moves is also added to openings.json so the recogniser can match it
in future play.  Names are taken from the figure → opening map below.

Usage:
    python tools/import_book_games.py [--dry-run] [--docx PATH]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import uuid
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from docx import Document          # python-docx
from game.board import BoardState
from game.rules import get_all_legal_moves, is_terminal
from ai.opening_book import Opening, OpeningBook
from ai.opening_recognizer import OpeningRecognizer


# ── Figure → (name, family) mapping ──────────────────────────────────────────
# Derived from the strategy book's chapter headings and figure captions.
# Table index is the key (matches the order of tables in book games.docx).

_FIGURE_MAP: dict[int, tuple[str, str]] = {
    0:  ("Game One",                                 "Early Game"),
    1:  ("Try Try Again",                            "Early Game"),
    2:  ("Alternate Game Two",                       "Early Game"),
    3:  ("Game Three",                               "Early Game"),
    4:  ("Gambit Play",                              "Early Game"),
    6:  ("Game Four",                                "Early Game"),
    7:  ("Game Five A",                              "Early Game"),
    8:  ("Game Five B",                              "Early Game"),
    10: ("Game Seven — Patience",                    "Early Game"),
    11: ("Creating Opportunities",                   "Early Game"),
    12: ("Which Piece to Take",                      "Early Game"),
    13: ("Right Angle Potential Mills",              "Early Game"),
    14: ("End Play Placements",                      "Early Game"),
    15: ("Early Game to Victory",                    "Early Game"),
    16: ("Cannon Fodder",                            "Early Game"),
    19: ("Sacrificial Mills",                        "Early Game"),
    20: ("Cardinal Point Abandonment",               "Early Game"),
    22: ("Man-to-Man Marking",                       "Man-to-Man Marking"),
    23: ("Black Diamond",                            "Black Diamond"),
    24: ("Mill Rush — Misplacement Loss",            "Mill Rush"),
    25: ("Mill Rush — Parallel Lines",               "Mill Rush"),
    26: ("Mill Rush — Parallel Lines (Alt)",         "Mill Rush"),
    27: ("Mill Rush — Perpendicular Lines",          "Mill Rush"),
    28: ("Mill Rush — Perpendicular Lines (Alt)",    "Mill Rush"),
    29: ("Mill Rush — Extended Parallel",            "Mill Rush"),
    30: ("Mill Rush — Alt Extended Parallel",        "Mill Rush"),
    31: ("Mill Rush — Black's Alternate Response",   "Mill Rush"),
    32: ("Mill Rush — Alternate Black Response",     "Mill Rush"),
    33: ("Inverted Mill Rush — Parallel Lines",      "Inverted Mill Rush"),
    34: ("Inverted Mill Rush — Perpendicular",       "Inverted Mill Rush"),
    35: ("Inverted Mill Rush — White Mistake",       "Inverted Mill Rush"),
    36: ("Mill Rush — Endgame Stalemate",            "Mill Rush"),
    37: ("Battle Lines — Black Loss",                "Battle Lines"),
    38: ("Battle Lines — Better Response",           "Battle Lines"),
    39: ("Closed Z Mill",                            "Z Mill"),
    40: ("Open Z Mill",                              "Z Mill"),
    41: ("Z Mill — Variation A",                     "Z Mill"),
    42: ("Z Mill — Variation B",                     "Z Mill"),
    43: ("Z Mill — Variation C",                     "Z Mill"),
    44: ("Z Mill — Variation D",                     "Z Mill"),
    45: ("Z Mill — Variation E",                     "Z Mill"),
    47: ("Z Mill c4/e3 — Parallel Mills",            "Z Mill"),
    48: ("Z Mill c4/e3 — Ring Round the Rosie",      "Z Mill"),
    49: ("Z Mill c4/e3 — All Along the Watchtower",  "Z Mill"),
    50: ("Z Mill c4/e3 — Perpendicular Mills",       "Z Mill"),
    51: ("Z Mill c4/e3 — A Shoot-Out",               "Z Mill"),
    52: ("Z Mill c4/e3 — Misdirection",              "Z Mill"),
    53: ("Z Mill c4/e3 — Cut the Bottom Out",        "Z Mill"),
    54: ("Z Mill c4/e3 — Two L's",                   "Z Mill"),
    55: ("Z Mill c4/e3 — Hangman",                   "Z Mill"),
    56: ("Z Mill c4/e3 — Perpendicular Mills 2",     "Z Mill"),
    57: ("Z Mill c4/e3 — Roof Falls In",             "Z Mill"),
    58: ("Black L, White Wrap",                      "Z Mill"),
    59: ("Into White — White to a1",                 "Z Mill"),
}

# Minimum placement half-moves for a game to be added to the opening book.
_MIN_PLACEMENT_MOVES = 10


# ── Notation parser ───────────────────────────────────────────────────────────

_POS_RE = re.compile(r'^[a-g][1-7]$')


def _parse_token(token: str):
    token = token.strip().replace('×', 'x').replace('X', 'x')
    if not token or token == '*':
        return None
    capture: str | None = None
    if 'x' in token:
        token, capture = token.split('x', 1)
        capture = capture.strip()
        if not _POS_RE.match(capture):
            capture = None
    if '-' in token:
        parts = token.split('-', 1)
        frm, to = parts[0].strip(), parts[1].strip()
        if _POS_RE.match(frm) and _POS_RE.match(to):
            return ('move', frm, to, capture)
    else:
        to = token.strip()
        if _POS_RE.match(to):
            return ('place', None, to, capture)
    return None


def _parse_cell(text: str) -> list[tuple]:
    text = text.strip()
    if not text:
        return []
    m = re.match(r'^\d+\.\s+(\S+)(?:\s+(\S+))?$', text)
    if not m:
        return []
    results = []
    for color, raw in (('W', m.group(1)), ('B', m.group(2) or '')):
        parsed = _parse_token(raw)
        if parsed:
            results.append((color, *parsed))
    return results


def parse_table(table) -> list[tuple]:
    half_moves: list[tuple] = []
    for row in table.rows:
        for cell in row.cells:
            half_moves.extend(_parse_cell(cell.text))
    return half_moves


# ── Game replay ───────────────────────────────────────────────────────────────

def _find_legal_move(board: BoardState, mtype: str, frm, to: str, capture) -> dict | None:
    legal = get_all_legal_moves(board)
    candidates = [m for m in legal if m.get('to') == to and m.get('from') == frm]
    if not candidates:
        return None
    if capture is not None:
        exact = [m for m in candidates if m.get('capture') == capture]
        if exact:
            return exact[0]
        return candidates[0]
    no_cap = [m for m in candidates if not m.get('capture')]
    return no_cap[0] if no_cap else candidates[0]


def replay_game(half_moves: list[tuple], recognizer: OpeningRecognizer) -> dict | None:
    recognizer.reset()
    board = BoardState.new_game()
    move_records: list[dict] = []
    turn_num = 0

    for color, mtype, frm, to, book_capture in half_moves:
        if board.turn != color:
            return None

        legal_move = _find_legal_move(board, mtype, frm, to, book_capture)
        if legal_move is None:
            return None

        fen_before = board.to_fen_string()
        turn_num += 1
        capture = legal_move.get('capture')

        if mtype == 'place':
            notation = to + (f'×{capture}' if capture else '')
        else:
            notation = f'{frm}-{to}' + (f'×{capture}' if capture else '')

        move_records.append({
            'turn': turn_num, 'color': color, 'type': mtype,
            'from': frm, 'to': to, 'capture': capture,
            'notation': notation, 'board_fen_before': fen_before,
        })

        board = board.apply_move(legal_move)

        if mtype == 'place':
            recognizer.update(to, board)

        terminal, _ = is_terminal(board)
        if terminal:
            break

    if not move_records:
        return None

    _, winner = is_terminal(board)
    result = recognizer.get_current_result()

    record: dict = {
        'session_id': str(uuid.uuid4()),
        'date': datetime.now().isoformat(),
        'human_color': 'book_import',
        'winner': winner,
        'move_count': len(move_records),
        'white_difficulty': 0, 'black_difficulty': 0,
        'self_play': True, 'book_import': True, 'draw_repetition': False,
        'moves': move_records,
        'recognised_opening_name': None,
        'recognised_opening_id': None,
        'recognised_opening_status': result.status,
    }
    if result.status not in ('inactive', 'novel'):
        record['recognised_opening_name'] = result.name
        record['recognised_opening_id'] = result.opening_id

    return record


# ── Opening book import ───────────────────────────────────────────────────────

def _fen_sigs(placement_moves: list[str]) -> list[dict]:
    board = BoardState.new_game()
    sigs = []
    for i, pos in enumerate(placement_moves):
        board = board.apply_move({'from': None, 'to': pos, 'capture': None})
        ply = i + 1
        if ply in (4, 6, 8, 10):
            sigs.append({'ply': ply, 'fen': board.to_fen_string()})
    return sigs


def add_to_opening_book(
    opening_book: OpeningBook,
    table_idx: int,
    record: dict,
    dry_run: bool,
) -> str | None:
    """
    Add this game's placement sequence to the opening book.
    Returns the opening name if added, else None.
    """
    placement_dests = [
        m['to'] for m in record['moves'] if m['type'] == 'place'
    ]
    if len(placement_dests) < _MIN_PLACEMENT_MOVES:
        return None

    name, family = _FIGURE_MAP.get(table_idx, (f"Book Game {table_idx}", "Book"))
    winner = record.get('winner')
    outcome_stats: dict = {'W': 0, 'B': 0, 'D': 0}
    if winner in outcome_stats:
        outcome_stats[winner] = 1

    # side encodes whose moves we recommend: the winner's, or 'both' for draws/unknown.
    side = winner if winner in ('W', 'B') else 'both'

    opening_id = f"book-{table_idx:02d}-{uuid.uuid4().hex[:6]}"
    sigs = _fen_sigs(placement_dests) if not dry_run else []

    opening = Opening(
        opening_id=opening_id,
        name=name,
        aliases=[],
        family=family,
        side=side,
        seed_source='learned',
        line_moves=placement_dests,
        branch_moves=[],
        opening_fen_signatures=sigs,
        strategic_notes='',
        common_blunders=[],
        recommended_responses={'W': [], 'B': []},
        outcome_stats=outcome_stats,
        confidence=0.8,
        tags=['book', family.lower().replace(' ', '-')],
        source_reference="Nine Men's Morris Strategy — Brandwood",
        needs_llm_name=False,
    )

    if not dry_run:
        opening_book.save_opening(opening)
    return name


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dry-run', action='store_true',
                    help='Parse and replay but do not write any files')
    ap.add_argument('--docx', default=str(ROOT / 'book games.docx'))
    args = ap.parse_args()

    book_path = ROOT / 'data' / 'openings' / 'openings.json'
    opening_book = OpeningBook(book_path)
    recognizer   = OpeningRecognizer(opening_book)

    doc = Document(args.docx)
    out_dir = ROOT / 'data' / 'games'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Remove old book-import entries to start clean
    if not args.dry_run:
        # Old book-import game files
        removed_files = list(out_dir.glob('game_book_*.jsonl'))
        for old in removed_files:
            old.unlink()
        # Old book-* openings in the opening book
        old_book_ids = [oid for oid in list(opening_book._index) if oid.startswith('book-')]
        for oid in old_book_ids:
            del opening_book._index[oid]
        if old_book_ids:
            opening_book._write_openings_json()
            print(f'Cleared {len(old_book_ids)} stale book-import openings and '
                  f'{len(removed_files)} game files.')

    ok = skipped = failed = book_added = 0

    for i, table in enumerate(doc.tables):
        half_moves = parse_table(table)

        if not half_moves:
            print(f'Table {i:2d}: SKIP  (no parseable moves)')
            skipped += 1
            continue

        if half_moves[0][1] == 'move':
            print(f'Table {i:2d}: SKIP  (movement phase start)')
            skipped += 1
            continue

        record = replay_game(half_moves, recognizer)
        if record is None:
            print(f'Table {i:2d}: FAIL  ({len(half_moves)} half-moves — illegal move)')
            failed += 1
            continue

        winner_str = record['winner'] or '?'
        n = record['move_count']
        name, _ = _FIGURE_MAP.get(i, (f"Book Game {i}", "Book"))
        print(f'Table {i:2d}: OK    {n} moves  winner={winner_str}  "{name}"')
        ok += 1

        # Add to opening book
        added_name = add_to_opening_book(opening_book, i, record, args.dry_run)
        if added_name:
            book_added += 1
            print(f'           ↳ added to opening book: {added_name}')

        # Update outcome stats if we recognised an existing opening AND have a winner
        if (
            not args.dry_run
            and record['winner'] in ('W', 'B', 'D')
            and record.get('recognised_opening_id')
        ):
            opening_book.update_outcome_stats(
                record['recognised_opening_id'],
                winner=record['winner'],
                human_color=None,
            )

        if not args.dry_run:
            sid = record['session_id'][:8]
            fname = out_dir / f'game_book_{i:02d}_{sid}.jsonl'
            fname.write_text(json.dumps(record) + '\n')

    print(
        f'\nResult: {ok} games imported, {book_added} added to opening book, '
        f'{skipped} skipped, {failed} failed'
    )


if __name__ == '__main__':
    main()
