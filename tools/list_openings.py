"""
tools/list_openings.py — Console CLI browser for the Nine Men's Morris opening book.

Usage:
  python tools/list_openings.py                   # interactive menu
  python tools/list_openings.py --filter book      # only book openings
  python tools/list_openings.py --filter learned   # AI-discovered
  python tools/list_openings.py --filter human     # human-taught
  python tools/list_openings.py --family "Mill Rush"
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ai.opening_book import Opening, OpeningBook

# ── Constants ─────────────────────────────────────────────────────────────────

PAGE_SIZE = 20
SOURCE_ORDER = {"book": 0, "human": 1, "learned": 2}

# ── Display helpers ───────────────────────────────────────────────────────────


def _divider(char: str = "─", width: int = 72) -> str:
    return char * width


def _fmt_wbd(stats: dict) -> str:
    w = stats.get("W", 0)
    b = stats.get("B", 0)
    d = stats.get("D", 0)
    return f"{w}/{b}/{d}"


def _sort_key(o: Opening) -> tuple:
    return (SOURCE_ORDER.get(o.seed_source, 99), o.family.lower(), o.name.lower())


def _count_by_source(openings: list[Opening]) -> dict[str, int]:
    counts: dict[str, int] = {"book": 0, "human": 0, "learned": 0}
    for o in openings:
        key = o.seed_source if o.seed_source in counts else "learned"
        counts[key] += 1
    return counts


# ── List view ─────────────────────────────────────────────────────────────────


def _print_list(openings: list[Opening], page: int, total_pages: int) -> None:
    counts = _count_by_source(openings)
    total = len(openings)

    print()
    print("=" * 72)
    print("   Nine Men's Morris  —  Opening Book Browser")
    print("=" * 72)
    print(
        f"\n  {total} opening{'s' if total != 1 else ''}  "
        f"(book: {counts['book']}  human: {counts['human']}  learned: {counts['learned']})"
    )
    if total_pages > 1:
        print(f"  Page {page + 1} of {total_pages}")
    print()

    # Column widths
    # #  │ Name (30) │ Family (16) │ Src (7) │ Moves (5) │ W/B/D
    hdr = f"{'#':>3}  {'Name':<30}  {'Family':<16}  {'Src':<7}  {'Moves':>5}  {'W/B/D'}"
    print(hdr)
    print(_divider())

    start = page * PAGE_SIZE
    end = min(start + PAGE_SIZE, total)
    for display_idx, i in enumerate(range(start, end), start=1):
        o = openings[i]
        num = start + display_idx
        name_trunc = o.name[:29] if len(o.name) > 29 else o.name
        family_trunc = o.family[:15] if len(o.family) > 15 else o.family
        wbd = _fmt_wbd(o.outcome_stats)
        print(
            f"  {num:>2}  {name_trunc:<30}  {family_trunc:<16}  {o.seed_source:<7}  "
            f"{len(o.line_moves):>5}  {wbd}"
        )

    print()
    cmds = ["[number] view", "[f] filter", "[q] quit"]
    if total_pages > 1:
        if page > 0:
            cmds.insert(0, "[p] prev")
        if page < total_pages - 1:
            cmds.insert(0, "[n] next")
    print("  Commands: " + "  ".join(cmds))


# ── Detail view ───────────────────────────────────────────────────────────────


def _print_detail(o: Opening) -> None:
    print()
    print("=" * 72)
    print(f"  {o.name}  ({o.opening_id})")
    print("=" * 72)

    conf_pct = f"{int(o.confidence * 100)}%"
    print(
        f"  Family: {o.family}   Side: {o.side}   "
        f"Confidence: {conf_pct}   Source: {o.seed_source}"
    )
    if o.aliases:
        print(f"  Aliases: {', '.join(o.aliases)}")
    if o.tags:
        print(f"  Tags: {', '.join(o.tags)}")

    # Move line
    print()
    print(f"  Move line ({len(o.line_moves)} moves):")
    if o.line_moves:
        last_ply = len(o.line_moves)
        for i in range(0, last_ply, 2):
            ply_w = i + 1
            move_w = o.line_moves[i]
            line = f"  Ply {ply_w:>2}  W: {move_w:<6}"
            if i + 1 < last_ply:
                ply_b = i + 2
                move_b = o.line_moves[i + 1]
                last_marker = "   (<- last book move)" if ply_b == last_ply else ""
                line += f"  Ply {ply_b:>2}  B: {move_b:<6}{last_marker}"
            else:
                line += "   (<- last book move)"
            print(f"  {line}")
    else:
        print("    (no moves recorded)")

    # Strategic notes
    if o.strategic_notes:
        print()
        print("  Strategic notes:")
        # Wrap at ~66 chars
        words = o.strategic_notes.split()
        line_buf: list[str] = []
        line_len = 0
        for w in words:
            if line_len + len(w) + 1 > 66 and line_buf:
                print(f"    {' '.join(line_buf)}")
                line_buf = [w]
                line_len = len(w)
            else:
                line_buf.append(w)
                line_len += len(w) + 1
        if line_buf:
            print(f"    {' '.join(line_buf)}")

    # Common blunders
    if o.common_blunders:
        print()
        print(f"  Common blunders: {', '.join(o.common_blunders)}")

    # Recommended responses
    resp_w = o.recommended_responses.get("W", [])
    resp_b = o.recommended_responses.get("B", [])
    if resp_w or resp_b:
        print()
        print("  Recommended responses:")
        if resp_w:
            print(f"    W: {', '.join(resp_w)}")
        if resp_b:
            print(f"    B: {', '.join(resp_b)}")

    # Variations (branch moves)
    if o.branch_moves:
        print()
        print(f"  Variations ({len(o.branch_moves)}):")
        for branch in o.branch_moves:
            cont = ", ".join(branch.line_continuation) if branch.line_continuation else "—"
            print(
                f"    • {branch.name} — deviation at ply {branch.deviation_ply}, "
                f"{('W' if branch.deviation_ply % 2 == 1 else 'B')} plays {branch.deviation_move}"
            )
            if branch.line_continuation:
                print(f"      continuation: {cont}")
            if branch.strategic_notes:
                print(f"      notes: {branch.strategic_notes}")
            print(f"      (seed: {branch.seed_source})")
    else:
        print()
        print("  Variations: none")

    # Outcome stats
    print()
    wbd = o.outcome_stats
    print(
        f"  Outcome stats: W: {wbd.get('W', 0)}  B: {wbd.get('B', 0)}  D: {wbd.get('D', 0)}"
    )

    # Source reference
    if getattr(o, "source_reference", ""):
        print(f"  Source: {o.source_reference}")

    print()
    print("  [b] back  [q] quit")


# ── Filter prompt ─────────────────────────────────────────────────────────────


def _prompt_filter(all_openings: list[Opening]) -> list[Opening]:
    families = sorted({o.family for o in all_openings})
    print()
    print("  Filter options:")
    print("    [1] All openings")
    print("    [2] Book only")
    print("    [3] Human-taught only")
    print("    [4] AI-learned only")
    print("    [5] By family")
    print("    [c] Cancel")
    print()
    choice = input("  > ").strip().lower()

    if choice == "1":
        return all_openings
    elif choice == "2":
        return [o for o in all_openings if o.seed_source == "book"]
    elif choice == "3":
        return [o for o in all_openings if o.seed_source == "human"]
    elif choice == "4":
        return [o for o in all_openings if o.seed_source == "learned"]
    elif choice == "5":
        print()
        for i, fam in enumerate(families, start=1):
            print(f"    [{i}] {fam}")
        raw = input("\n  Enter family number (or name): ").strip()
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(families):
                selected = families[idx]
                return [o for o in all_openings if o.family == selected]
        except ValueError:
            pass
        # Try name match
        needle = raw.lower()
        matched = [o for o in all_openings if needle in o.family.lower()]
        if matched:
            return matched
        print("  No matching family found.")
        return all_openings
    else:
        return all_openings


# ── One-shot print (non-interactive) ─────────────────────────────────────────


def _print_oneshot(openings: list[Opening], label: str) -> None:
    counts = _count_by_source(openings)
    total = len(openings)
    print()
    print("=" * 72)
    print("   Nine Men's Morris  —  Opening Book Browser")
    print("=" * 72)
    print(
        f"\n  {label}: {total} opening{'s' if total != 1 else ''}  "
        f"(book: {counts['book']}  human: {counts['human']}  learned: {counts['learned']})"
    )
    print()
    hdr = f"  {'#':>2}  {'Name':<30}  {'Family':<16}  {'Src':<7}  {'Moves':>5}  {'W/B/D'}"
    print(hdr)
    print("  " + _divider())
    for i, o in enumerate(openings, start=1):
        name_trunc = o.name[:29] if len(o.name) > 29 else o.name
        family_trunc = o.family[:15] if len(o.family) > 15 else o.family
        wbd = _fmt_wbd(o.outcome_stats)
        print(
            f"  {i:>2}  {name_trunc:<30}  {family_trunc:<16}  {o.seed_source:<7}  "
            f"{len(o.line_moves):>5}  {wbd}"
        )
    print()


# ── Interactive loop ──────────────────────────────────────────────────────────


def _run_interactive(all_openings: list[Opening]) -> None:
    current = list(all_openings)
    page = 0

    while True:
        total_pages = max(1, (len(current) + PAGE_SIZE - 1) // PAGE_SIZE)
        page = min(page, total_pages - 1)
        _print_list(current, page, total_pages)

        try:
            raw = input("  > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if raw == "q":
            break
        elif raw == "f":
            current = _prompt_filter(all_openings)
            current.sort(key=_sort_key)
            page = 0
        elif raw == "n" and page < total_pages - 1:
            page += 1
        elif raw == "p" and page > 0:
            page -= 1
        elif raw.isdigit():
            num = int(raw)
            if 1 <= num <= len(current):
                _show_detail_loop(current[num - 1])
            else:
                print(f"  Please enter a number between 1 and {len(current)}.")
        else:
            print("  Unrecognised command.")


def _show_detail_loop(opening: Opening) -> None:
    while True:
        _print_detail(opening)
        try:
            raw = input("  > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if raw in ("b", "back"):
            return
        elif raw == "q":
            sys.exit(0)
        else:
            print("  Press [b] to go back or [q] to quit.")


# ── Entry point ───────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Browse the Nine Men's Morris opening book."
    )
    parser.add_argument(
        "--filter",
        choices=["book", "learned", "human"],
        help="Show only openings from this source (non-interactive)",
    )
    parser.add_argument(
        "--family",
        help="Show only openings from this family (non-interactive)",
    )
    args = parser.parse_args(argv)

    # Change to project root so relative paths work
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)

    try:
        book = OpeningBook(
            book_path="data/openings/book_openings.json",
            openings_path="data/openings/openings.json",
        )
    except Exception as exc:
        print(f"ERROR: could not load opening book: {exc}", file=sys.stderr)
        return 1

    all_openings: list[Opening] = sorted(book.values(), key=_sort_key)

    # Non-interactive one-shot modes
    if args.filter or args.family:
        filtered = all_openings

        if args.filter:
            filtered = [o for o in filtered if o.seed_source == args.filter]
            label = f"Filter: source={args.filter!r}"

        if args.family:
            needle = args.family.lower()
            filtered = [o for o in filtered if needle in o.family.lower()]
            label = f"Filter: family={args.family!r}"

        if args.filter and args.family:
            label = f"Filter: source={args.filter!r}, family={args.family!r}"

        _print_oneshot(filtered, label)
        return 0

    # Interactive mode
    try:
        _run_interactive(all_openings)
    except KeyboardInterrupt:
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
