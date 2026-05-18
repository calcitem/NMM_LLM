#!/usr/bin/env python3
"""tools/self_test.py — Project self-test: unit suite + Ollama connectivity + smoke test."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
import urllib.request
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
# also insert tools dir so _run_fast_game can be imported
sys.path.insert(0, str(ROOT / "tools"))


# ── Section helpers ───────────────────────────────────────────────────────────

def _section(title: str) -> None:
    print(f"\n{'─' * 52}")
    print(f"  {title}")
    print(f"{'─' * 52}")


# ── 1. Unit test suite ────────────────────────────────────────────────────────

def run_unit_tests() -> unittest.TestResult:
    _section("Unit tests")

    test_files = [
        "test_board.py",
        "test_ai.py",
        "test_stage3.py",
        "test_stage4.py",
        "test_stage5.py",
        "test_stage6.py",
    ]

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    tests_dir = ROOT / "tests"
    for fname in test_files:
        fpath = tests_dir / fname
        if not fpath.exists():
            print(f"  [SKIP] {fname} not found")
            continue
        try:
            file_suite = loader.discover(start_dir=str(tests_dir), pattern=fname)
            suite.addTests(file_suite)
        except Exception as exc:
            print(f"  [ERROR] Loading {fname}: {exc}")

    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    return result


# ── 2. Ollama connectivity probe ──────────────────────────────────────────────

def probe_ollama() -> tuple[bool, str]:
    """
    Return (reachable: bool, url: str).
    Reads OLLAMA_HOST and OLLAMA_PORT from env, defaults to localhost:11434.
    """
    host = os.environ.get("OLLAMA_HOST", "localhost")
    port = os.environ.get("OLLAMA_PORT", "11434")
    url = f"http://{host}:{port}"
    tags_url = f"{url}/api/tags"

    _section("Ollama connectivity")
    print(f"  Probing {tags_url} …")

    try:
        req = urllib.request.Request(tags_url)
        with urllib.request.urlopen(req, timeout=3) as resp:
            if resp.status == 200:
                print(f"  [OK] Ollama reachable at {url}")
                return True, url
            else:
                print(f"  [FAIL] Unexpected HTTP status {resp.status} from {tags_url}")
                return False, url
    except urllib.error.URLError as exc:
        print(f"  [FAIL] Cannot reach Ollama at {tags_url}: {exc.reason}")
        print("  Hint: start Ollama with 'ollama serve', or set OLLAMA_HOST / OLLAMA_PORT.")
        return False, url
    except Exception as exc:
        print(f"  [FAIL] Unexpected error probing {tags_url}: {exc}")
        return False, url


# ── 3. LLM smoke test ─────────────────────────────────────────────────────────

def run_llm_smoke_test(ollama_url: str) -> bool:
    """
    Instantiate MillsLLM, call player_chat, verify a non-empty string is returned.
    Returns True on pass, False on failure.
    """
    _section("LLM smoke test")
    try:
        from game.board import BoardState
        from ai.memory_manager import MemoryManager
        from ai.mills_llm import MillsLLM

        board = BoardState.new_game()

        with tempfile.TemporaryDirectory() as tmp:
            mem = MemoryManager(
                chroma_path=f"{tmp}/chroma",
                games_path=f"{tmp}/games",
                session_path=f"{tmp}/session",
                ollama_url=ollama_url,
                use_ollama_embeddings=False,
            )
            llm = MillsLLM(memory=mem, ollama_url=ollama_url)
            reply = llm.player_chat("What game are we playing?", board)

        if reply and isinstance(reply, str) and len(reply.strip()) > 0:
            print(f"  [PASS] Got non-empty reply ({len(reply)} chars)")
            print(f"  Reply preview: {reply[:120]!r}")
            return True
        else:
            print(f"  [FAIL] Empty or non-string reply: {reply!r}")
            return False

    except Exception as exc:
        print(f"  [FAIL] Exception during LLM smoke test: {exc}")
        return False


# ── 4. Self-play smoke test ───────────────────────────────────────────────────

def run_self_play_smoke_test() -> bool:
    """
    Run one fast-mode game (difficulty 3 vs 3, no LLM) and verify the result dict
    has expected keys.
    Returns True on pass, False on failure.
    """
    _section("Self-play smoke test (1 game, difficulty 3, no LLM)")
    try:
        from self_play import _run_fast_game, OpeningBook
        from ai.game_ai import GameAI
        from ai.heuristics import HeuristicWeights

        book = OpeningBook()
        white_ai = GameAI(color="W", difficulty=3)
        black_ai = GameAI(color="B", difficulty=3)

        record = _run_fast_game(
            white_ai, black_ai, book,
            verbose=False,
            game_label="[smoke] ",
            white_personality="balanced",
            black_personality="balanced",
        )

        # Basic sanity checks on the returned record
        required_keys = {"session_id", "winner", "move_count", "moves", "self_play"}
        missing = required_keys - set(record.keys())
        if missing:
            print(f"  [FAIL] Record missing keys: {missing}")
            return False

        move_count = record.get("move_count", 0)
        winner = record.get("winner")
        print(f"  [PASS] Game completed: {move_count} moves, winner={winner!r}")
        return True

    except Exception as exc:
        import traceback
        print(f"  [FAIL] Exception during self-play smoke test: {exc}")
        traceback.print_exc()
        return False


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Nine Men's Morris — Project Self-Test")
    print(f"ROOT: {ROOT}")

    # 1. Unit tests
    unit_result = run_unit_tests()

    # 2. Ollama connectivity
    ollama_ok, ollama_url = probe_ollama()

    # 3. LLM smoke test (only if Ollama is reachable)
    if ollama_ok:
        llm_ok = run_llm_smoke_test(ollama_url)
    else:
        _section("LLM smoke test")
        print("  [SKIP] Ollama unreachable — skipping LLM smoke test.")
        llm_ok = None  # None means skipped

    # 4. Self-play smoke test
    self_play_ok = run_self_play_smoke_test()

    # ── Final summary ─────────────────────────────────────────────────────────
    _section("Summary")

    tests_run    = unit_result.testsRun
    tests_failed = len(unit_result.failures)
    tests_errored = len(unit_result.errors)
    tests_passed = tests_run - tests_failed - tests_errored

    print(f"  Unit tests   : {tests_passed} passed, {tests_failed} failed, "
          f"{tests_errored} errors  (total {tests_run})")
    print(f"  Ollama       : {'REACHABLE at ' + ollama_url if ollama_ok else 'UNREACHABLE'}")

    if llm_ok is None:
        llm_status = "SKIPPED (Ollama unreachable)"
    elif llm_ok:
        llm_status = "PASS"
    else:
        llm_status = "FAIL"
    print(f"  LLM smoke    : {llm_status}")
    print(f"  Self-play    : {'PASS' if self_play_ok else 'FAIL'}")

    # Determine overall exit code
    any_failure = (
        tests_failed > 0
        or tests_errored > 0
        or llm_ok is False
        or not self_play_ok
    )

    print()
    if any_failure:
        print("  RESULT: FAIL")
        sys.exit(1)
    else:
        print("  RESULT: PASS")
        sys.exit(0)


if __name__ == "__main__":
    main()
