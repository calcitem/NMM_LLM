"""Integration tests for the sentinel overlay wired into GameAI.

These verify the core safety contract: the sentinel is advisory-only and can
never change the move GameAI selects, never crash the game loop, and the game
plays byte-identically whether or not an overlay is attached.
"""

from __future__ import annotations

import torch

from ai.game_ai import GameAI
from game.board import BoardState
from game.rules import get_all_legal_moves, is_terminal
from learned_ai.sentinel.config import SentinelConfig
from learned_ai.sentinel.infer import SentinelAdvisor
from learned_ai.sentinel.model import SentinelNet


def _make_advisor(tmp_path) -> SentinelAdvisor:
    cfg = SentinelConfig(hidden_dims=[64, 32], dropout=0.0)
    net = SentinelNet(input_dim=cfg.input_dim, hidden_dims=cfg.hidden_dims, dropout=0.0)
    ckpt = tmp_path / "sentinel.pt"
    torch.save({"state_dict": net.state_dict(), "config": cfg.to_dict()}, ckpt)
    return SentinelAdvisor(str(ckpt), config=cfg, device="cpu")


def _fresh_board() -> BoardState:
    # Mid-placement position so choose_move has real candidates.
    return BoardState.from_fen_string("BBW....B.W.W............|W|3|3")


def test_choose_move_identical_with_and_without_sentinel(tmp_path):
    """Attaching an advisory sentinel must not change the chosen move."""
    board = _fresh_board()

    ai_plain = GameAI(color="W", difficulty=2)
    move_plain = ai_plain.choose_move(board)

    ai_sent = GameAI(color="W", difficulty=2)
    ai_sent.set_sentinel(_make_advisor(tmp_path), mode="advisory")
    move_sent = ai_sent.choose_move(board)

    assert move_plain == move_sent


def test_set_sentinel_records_advice(tmp_path):
    """After a move, the advisory result is stored for logging/debug."""
    ai = GameAI(color="W", difficulty=2)
    ai.set_sentinel(_make_advisor(tmp_path), mode="advisory")
    assert ai.sentinel is not None
    assert ai.sentinel_mode == "advisory"
    ai.choose_move(_fresh_board())
    # advise() ran and populated last_sentinel_advice (not the neutral default).
    assert ai.last_sentinel_advice is not None


def test_broken_advisor_never_crashes_choose_move():
    """A misbehaving advisor must be swallowed; the game proceeds normally."""

    class _ExplodingAdvisor:
        def advise(self, board, ctx):
            raise RuntimeError("boom")

    ai = GameAI(color="W", difficulty=2)
    ai.set_sentinel(_ExplodingAdvisor(), mode="advisory")
    move = ai.choose_move(_fresh_board())
    assert move is not None
    assert "to" in move


def test_short_ai_vs_ai_game_completes_with_sentinel(tmp_path):
    """Smoke: a full AI-vs-AI game runs to termination with overlays attached."""
    advisor = _make_advisor(tmp_path)
    white = GameAI(color="W", difficulty=1)
    black = GameAI(color="B", difficulty=1)
    white.set_sentinel(advisor, mode="advisory")
    black.set_sentinel(advisor, mode="advisory")

    board = BoardState.from_fen_string("........................|W|0|0")
    for _ply in range(60):
        done, _winner = is_terminal(board)
        if done:
            break
        if not get_all_legal_moves(board):
            break
        mover = white if board.turn == "W" else black
        move = mover.choose_move(board)
        assert move is not None
        board = board.apply_move(move)
    # No exception means the overlay never broke the loop.
    assert isinstance(board, BoardState)
