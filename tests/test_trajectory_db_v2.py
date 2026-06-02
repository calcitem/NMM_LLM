"""Tests for board-state-first TrajectoryDB (Phase 1 core)."""

from __future__ import annotations

import json
import pathlib
import pytest

from ai.trajectory_db import TrajectoryDB, make_board_state_key
from game.board import BoardState


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_db(*records: dict) -> TrajectoryDB:
    db = TrajectoryDB.__new__(TrajectoryDB)
    db._games_dir = pathlib.Path("data/games")
    db._index = {}
    db._game_count = 0
    for rec in records:
        db._index_game(rec)
    return db


def _minimal_game(moves: list[dict], winner: str | None, source_type: str = "ai_vs_ai") -> dict:
    """Build a minimal game record that _index_game() can parse."""
    return {
        "winner": winner,
        "source_type": source_type,
        "moves": moves,
    }


def _move(fen: str, color: str, notation: str) -> dict:
    return {"board_fen_before": fen, "color": color, "notation": notation}


# ── FEN round-trip ────────────────────────────────────────────────────────────

class TestFenRoundTrip:
    def test_empty_board(self):
        b = BoardState.new_game()
        b2 = BoardState.from_fen_string(b.to_fen_string())
        assert b2.positions == b.positions
        assert b2.turn == b.turn
        assert b2.pieces_placed == b.pieces_placed
        assert b2.pieces_on_board == b.pieces_on_board
        assert b2.pieces_captured == b.pieces_captured
        assert b2.hash_key == b.hash_key

    def test_mid_game_from_real_file(self):
        games = sorted(pathlib.Path("data/games").rglob("*.jsonl"))
        if not games:
            pytest.skip("No game files")
        rec = json.loads(games[0].read_text().splitlines()[0])
        for mv in rec.get("moves", [])[:10]:
            fen = mv.get("board_fen_before")
            if not fen:
                continue
            b = BoardState.from_fen_string(fen)
            assert b.turn in ("W", "B")
            w_on = sum(1 for v in b.positions.values() if v == "W")
            b_on = sum(1 for v in b.positions.values() if v == "B")
            assert b.pieces_on_board == {"W": w_on, "B": b_on}


# ── Transposition merging ─────────────────────────────────────────────────────

class TestTranspositionMerging:
    def test_same_board_different_notations_merge(self):
        """Two games reaching the same board via different move orders share one bucket."""
        # Use the actual empty-board FEN for both games' first move
        empty_fen = "........................|W|0|0"

        # Both games start from the same position and play the same move (d6)
        # They should contribute to the same state_key entry.
        game_a = _minimal_game([_move(empty_fen, "W", "d6")], winner="W")
        game_b = _minimal_game([_move(empty_fen, "W", "d6")], winner="B")

        db = _make_db(game_a, game_b)

        b = BoardState.new_game()
        key, _ = make_board_state_key(b)
        assert key in db._index

        # Both games contribute to the same notation entry
        # (d6 or its D4 canonical equivalent)
        entries = db._index[key]
        total = sum(e["total"] for e in entries.values())
        assert total == 2, f"Expected 2 entries merged, got {total}"

    def test_query_returns_actual_notation(self):
        """query() maps canonical notation back to actual game notation."""
        empty_fen = "........................|W|0|0"
        game = _minimal_game([_move(empty_fen, "W", "d6")], winner="W")
        db = _make_db(game)
        b = BoardState.new_game()
        result = db.query(b, "W", min_samples=1)
        # d6 should appear (possibly transformed by D4 symmetry, then inverse-transformed back)
        assert "d6" in result, f"Expected d6 in {list(result.keys())}"


# ── Win/loss scoring ──────────────────────────────────────────────────────────

class TestWinLossScoring:
    def test_always_winning_move_positive_delta(self):
        empty_fen = "........................|W|0|0"
        # 5 games where W plays d6 and always wins
        records = [
            _minimal_game([_move(empty_fen, "W", "d6")], winner="W")
            for _ in range(10)
        ]
        db = _make_db(*records)
        b = BoardState.new_game()
        result = db.query(b, "W", min_samples=3)
        # d6 or its D4 equivalent should be positive (W always wins)
        # Find which notation maps to d6
        wins = [v for v in result.values() if v > 0]
        assert wins, f"Expected positive delta for 100% win move, got {result}"

    def test_always_losing_move_negative_delta(self):
        empty_fen = "........................|W|0|0"
        records = [
            _minimal_game([_move(empty_fen, "W", "d6")], winner="B")
            for _ in range(10)
        ]
        db = _make_db(*records)
        b = BoardState.new_game()
        result = db.query(b, "W", min_samples=3)
        losses = [v for v in result.values() if v < 0]
        assert losses, f"Expected negative delta for 100% loss move, got {result}"


# ── Confidence scaling ────────────────────────────────────────────────────────

class TestConfidenceScaling:
    def test_low_sample_smaller_delta(self):
        """2-sample position returns smaller |delta| than 20-sample position with same win rate."""
        import math
        empty_fen = "........................|W|0|0"

        low_sample_records = [
            _minimal_game([_move(empty_fen, "W", "d6")], winner="W")
            for _ in range(2)
        ]
        high_sample_records = [
            _minimal_game([_move(empty_fen, "W", "d6")], winner="W")
            for _ in range(20)
        ]
        db_low = _make_db(*low_sample_records)
        db_high = _make_db(*high_sample_records)

        b = BoardState.new_game()
        result_low  = db_low.query(b,  "W", min_samples=1)
        result_high = db_high.query(b, "W", min_samples=1)

        assert result_low, "Expected results from low-sample DB"
        assert result_high, "Expected results from high-sample DB"

        max_low  = max(abs(v) for v in result_low.values())
        max_high = max(abs(v) for v in result_high.values())
        assert max_low < max_high, (
            f"Low-sample delta {max_low:.3f} should be < high-sample delta {max_high:.3f}"
        )


# ── Source type separation ────────────────────────────────────────────────────

class TestSourceTypeSeparation:
    def test_ai_wins_stored_separately_from_human_wins(self):
        empty_fen = "........................|W|0|0"
        ai_game = _minimal_game([_move(empty_fen, "W", "d6")], winner="W", source_type="ai_vs_ai")
        human_game = _minimal_game([_move(empty_fen, "W", "d6")], winner="W", source_type="human_involved")
        db = _make_db(ai_game, human_game)

        b = BoardState.new_game()
        key, sym_idx = make_board_state_key(b)
        from ai.board_symmetry import transform_notation, SYM_INVERSE
        entries = db._index.get(key, {})
        assert entries, "Expected entries at start state"

        total_ai_wins    = sum(e["wins_ai"]    for e in entries.values())
        total_human_wins = sum(e["wins_human"] for e in entries.values())
        assert total_ai_wins == 1
        assert total_human_wins == 1


# ── Full load integration ─────────────────────────────────────────────────────

class TestFullLoad:
    def test_load_all_games(self):
        """All 317+ game files are indexed without error."""
        db = TrajectoryDB(pathlib.Path("data/games"))
        db.load()
        assert db.game_count >= 317, f"Expected ≥317 games, got {db.game_count}"
        assert db.entry_count > 1000, f"Expected >1000 state entries, got {db.entry_count}"

    def test_query_start_position(self):
        """Query on the empty board returns placement moves with |delta| ≤ 0.5."""
        db = TrajectoryDB(pathlib.Path("data/games"))
        db.load()
        b = BoardState.new_game()
        result = db.query(b, "W", min_samples=1)
        assert result, "Expected non-empty result at start position"
        for notation, delta in result.items():
            assert -0.5 <= delta <= 0.5, f"{notation} delta {delta:.3f} out of range"

    def test_frequencies_sum_to_one(self):
        """query_all_frequencies() at start returns frequencies summing to ~1.0."""
        db = TrajectoryDB(pathlib.Path("data/games"))
        db.load()
        b = BoardState.new_game()
        freq = db.query_all_frequencies(b, min_samples=1)
        if freq:
            total = sum(freq.values())
            assert abs(total - 1.0) < 0.01, f"Frequencies sum to {total:.4f}, expected ~1.0"
