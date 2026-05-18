"""tools/name_openings.py — Merge duplicate openings and batch-name unnamed ones.

Usage:
  python tools/name_openings.py               # merge duplicates + name all unnamed
  python tools/name_openings.py --dry-run     # report only, no saves
  python tools/name_openings.py --merge-only  # deduplicate but skip LLM naming
  python tools/name_openings.py --min-common 6  # stricter duplicate threshold

Ollama must be running for LLM naming. If it is not reachable, the tool
still performs deduplication but skips naming and marks entries with
needs_llm_name=True so they can be named in a future run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ai.opening_book import OpeningBook, is_auto_named
from ai.memory_manager import MemoryManager
from ai.mills_llm import MillsLLM


def _total_games(opening) -> int:
    return sum(opening.outcome_stats.get(k, 0) for k in ("W", "B", "D"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge duplicate openings and name unnamed ones via LLM."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would change without saving anything.",
    )
    parser.add_argument(
        "--merge-only", action="store_true",
        help="Only deduplicate; skip LLM naming.",
    )
    parser.add_argument(
        "--min-common", type=int, default=4,
        help="Minimum shared leading moves to count as a duplicate (default 4).",
    )
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--model", default="llama3.1:8b")
    args = parser.parse_args()

    book = OpeningBook()
    total_before = len(list(book.values()))

    # ── Step 1: merge duplicates ──────────────────────────────────────────────
    print(f"Opening book: {total_before} entries")
    print(f"\n=== Merging duplicates (min_common={args.min_common}) ===")

    if args.dry_run:
        from collections import defaultdict
        groups: dict = defaultdict(list)
        for o in book.values():
            if len(o.line_moves) >= args.min_common:
                key = tuple(o.line_moves[:args.min_common])
                groups[key].append(o)
        # Only count groups that actually have auto-named entries to remove
        actionable = {}
        would_remove = 0
        for k, g in groups.items():
            named_in_g   = [o for o in g if not is_auto_named(o.name)]
            unnamed_in_g = [o for o in g if is_auto_named(o.name)]
            if not unnamed_in_g:
                continue  # all named → nothing to do
            n_remove = len(unnamed_in_g) if named_in_g else len(unnamed_in_g) - 1
            if n_remove > 0:
                actionable[k] = (g, named_in_g, unnamed_in_g, n_remove)
                would_remove += n_remove
        print(f"  Would remove {would_remove} auto-named duplicates across {len(actionable)} groups:")
        for key, (group, named, unnamed, n_rm) in sorted(actionable.items(), key=lambda x: -x[1][3])[:10]:
            print(f"    prefix={list(key)}: keep {'named' if named else 'most-played unnamed'}, remove {n_rm}")
            for o in named:
                print(f"      ✓ (keep) {o.name!r:50s} games={_total_games(o)}")
            for o in unnamed:
                action = "merge" if named or o != max(unnamed, key=_total_games) else "keep+rename"
                print(f"        ({action}) {o.name!r:50s} games={_total_games(o)}")
    else:
        removed = book.merge_duplicates(min_common=args.min_common)
        total_after = len(list(book.values()))
        print(f"  Removed {removed} duplicates. Book now has {total_after} entries.")

    if args.merge_only:
        print("\nDone (merge only).")
        return

    # ── Step 2: name unnamed openings ─────────────────────────────────────────
    unnamed = [
        o for o in book.values()
        if o.needs_llm_name or is_auto_named(o.name)
    ]
    print(f"\n=== Naming {len(unnamed)} unnamed opening(s) ===")

    if not unnamed:
        print("  Nothing to name.")
        return

    if args.dry_run:
        for o in unnamed:
            print(f"  Would name: {o.name!r}  moves={o.line_moves[:4]}")
        print("\nDone (dry run).")
        return

    mem = MemoryManager()
    llm = MillsLLM(mem, ollama_url=args.ollama_url, model=args.model)

    if llm._client is None:
        print(
            "  Ollama not reachable — skipping LLM naming.\n"
            "  Entries marked needs_llm_name=True will be named next time Ollama is running."
        )
        for o in unnamed:
            if not o.needs_llm_name:
                o.needs_llm_name = True
                book.save_opening(o)
        return

    named = 0
    for o in unnamed:
        moves_preview = " ".join(o.line_moves[:6])
        print(f"  [{named + 1}/{len(unnamed)}] {o.name!r}  ({moves_preview}) … ", end="", flush=True)
        name = llm.name_novel_opening(o.line_moves)
        if name and not is_auto_named(name):
            o.name = name
            o.needs_llm_name = False
            book.save_opening(o)
            print(f"→ {name!r}")
            named += 1
        else:
            print("(LLM returned unusable name — keeping flag)")
            o.needs_llm_name = True
            book.save_opening(o)

    print(f"\nDone. Named {named} / {len(unnamed)} opening(s).")


if __name__ == "__main__":
    main()
