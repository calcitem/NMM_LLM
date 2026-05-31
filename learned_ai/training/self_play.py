"""Self-play game runner.

Plays a single game between two agents (both with `choose_move(board)`),
returns a structured trajectory plus the final winner. The trajectory carries
the per-step state, legal mask, primary/capture indices, phase id, and the
side to move; the trainer turns those into REINFORCE / PPO updates.

Move-cap safety: if a game exceeds ``max_plies`` half-moves it is declared a
draw to keep training loops bounded.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import torch

from game.board import BoardState
from game.rules import get_all_legal_moves, get_game_phase, is_terminal
from learned_ai.agents.learned_agent import LearnedAgent, LearnedDecision
from learned_ai.models.action_encoder import (
    encode_action,
    get_legal_mask,
)
from learned_ai.models.state_encoder import detect_phase, encode_state
from learned_ai.training.replay_buffer import Transition

DEFAULT_MAX_PLIES = 400


@dataclass
class TrajectoryStep:
    state: torch.Tensor
    legal_mask: torch.Tensor
    primary_index: int
    capture_index: Optional[int]
    phase_id: int
    side_to_move: str            # "W" or "B"
    primary_log_prob: Optional[torch.Tensor]
    capture_log_prob: Optional[torch.Tensor]
    value: Optional[torch.Tensor]
    heuristic_eval: Optional[float] = None  # evaluate(board_before_move, stm); for reward shaping


@dataclass
class GameResult:
    winner: Optional[str]            # "W" / "B" / None for draw
    draw_reason: Optional[str]
    plies: int
    trajectory: List[TrajectoryStep] = field(default_factory=list)
    move_log: List[dict] = field(default_factory=list)


def _record_step(agent_obj, board: BoardState) -> TrajectoryStep:
    """Pull a TrajectoryStep from an agent's last decision when available.

    When the acting agent is a LearnedAgent we copy its `last_decision`
    (it already encoded state/mask). For non-learned agents (random,
    heuristic) we re-encode here so the trainer can still learn from those
    transitions — useful when training against the heuristic AI as opponent.
    """
    if isinstance(agent_obj, LearnedAgent) and agent_obj.last_decision is not None:
        d: LearnedDecision = agent_obj.last_decision
        return TrajectoryStep(
            state=d.state,
            legal_mask=d.legal_mask,
            primary_index=d.primary_index,
            capture_index=d.capture_index,
            phase_id=d.phase_id,
            side_to_move=board.turn,
            primary_log_prob=d.primary_log_prob,
            capture_log_prob=d.capture_log_prob,
            value=d.value,
        )
    state = encode_state(board)
    mask = get_legal_mask(board)
    phase = detect_phase(board)
    return TrajectoryStep(
        state=state,
        legal_mask=mask,
        primary_index=-1,
        capture_index=None,
        phase_id=phase,
        side_to_move=board.turn,
        primary_log_prob=None,
        capture_log_prob=None,
        value=None,
    )


def play_game(
    white_agent,
    black_agent,
    max_plies: int = DEFAULT_MAX_PLIES,
    record_trajectory: bool = True,
    shaping_eval_fn: Optional[Callable] = None,  # evaluate(board, color) -> int | None
) -> GameResult:
    board = BoardState.new_game()
    plies = 0
    trajectory: List[TrajectoryStep] = []
    move_log: List[dict] = []
    draw_reason: Optional[str] = None

    while plies < max_plies:
        terminal, winner = is_terminal(board)
        if terminal:
            return GameResult(
                winner=winner,
                draw_reason=None,
                plies=plies,
                trajectory=trajectory,
                move_log=move_log,
            )

        legal = get_all_legal_moves(board)
        if not legal:
            # Side to move is stalemated -> loses.
            opp = "B" if board.turn == "W" else "W"
            return GameResult(
                winner=opp,
                draw_reason=None,
                plies=plies,
                trajectory=trajectory,
                move_log=move_log,
            )

        agent_obj = white_agent if board.turn == "W" else black_agent
        move = agent_obj.choose_move(board)
        if not move:
            opp = "B" if board.turn == "W" else "W"
            return GameResult(
                winner=opp,
                draw_reason="no-move",
                plies=plies,
                trajectory=trajectory,
                move_log=move_log,
            )

        if record_trajectory:
            step = _record_step(agent_obj, board)
            # For non-learned agents we still record the actual primary/cap
            # so trainers using behaviour cloning could leverage it.
            if step.primary_index == -1:
                primary, cap = encode_action(move)
                step.primary_index = primary
                step.capture_index = cap
            # C: store heuristic eval before the move for reward shaping.
            if shaping_eval_fn is not None:
                try:
                    step.heuristic_eval = float(shaping_eval_fn(board, board.turn))
                except Exception:
                    step.heuristic_eval = None
            trajectory.append(step)

        move_log.append({"color": board.turn, **move})
        board = board.apply_move(move)
        plies += 1

    return GameResult(
        winner=None,
        draw_reason="ply-cap",
        plies=plies,
        trajectory=trajectory,
        move_log=move_log,
    )


def assign_rewards(
    result: GameResult,
    win_reward: float = 1.0,
    loss_reward: float = -1.0,
    draw_reward: float = 0.0,
    gamma: float = 1.0,
    shaping_scale: float = 0.0,
    shaping_norm: float = 500.0,
) -> List[Transition]:
    """Convert a GameResult into a list of Transition objects with returns.

    Each step gets the discounted return-to-go from the perspective of the
    side that moved. Wins propagate +win_reward (discounted) backward to that
    side's moves; losses propagate loss_reward to the loser's moves. Draws
    assign draw_reward uniformly.

    When shaping_scale > 0 and trajectory steps carry heuristic_eval, an
    intermediate shaped reward is added: scale * tanh(Δeval / shaping_norm).
    Δeval = eval_after - eval_before, both from the acting side's perspective.
    Since the heuristic is symmetric (eval(board, W) == -eval(board, B)),
    eval_after = -trajectory[t+1].heuristic_eval (opponent's eval negated).
    """
    transitions: List[Transition] = []
    if not result.trajectory:
        return transitions

    winner = result.winner
    # First compute the per-side terminal reward, then walk backwards.
    n = len(result.trajectory)

    # Pre-compute per-step terminal reward (only at the final step the side
    # acting on that step is settled; intermediate rewards are 0).
    per_step_reward = [0.0] * n
    if winner is None:
        for i in range(n):
            per_step_reward[i] = draw_reward
    else:
        for i, step in enumerate(result.trajectory):
            if step.side_to_move == winner:
                per_step_reward[i] = win_reward
            else:
                per_step_reward[i] = loss_reward
    # Discount toward the *end* of the game (later moves matter more).
    # Build returns as: G_t = r_t + gamma * G_{t+1} for moves of the same side
    # but here we keep it simple: each move gets the side-specific outcome,
    # optionally damped by how far from the end it is.
    if gamma < 1.0:
        n_minus = n - 1
        for i in range(n):
            per_step_reward[i] *= gamma ** max(0, n_minus - i)

    # C: reward shaping — add heuristic eval delta as intermediate signal.
    if shaping_scale > 0.0:
        norm = max(shaping_norm, 1.0)
        for i, step in enumerate(result.trajectory):
            if step.heuristic_eval is None:
                continue
            eval_before = step.heuristic_eval
            if i + 1 < n:
                next_h = result.trajectory[i + 1].heuristic_eval
                # next step is opponent's move; their positive eval = our negative eval
                eval_after = -next_h if next_h is not None else 0.0
            else:
                # terminal: proxy by game outcome scaled to heuristic range
                stm = step.side_to_move
                if winner is None:
                    eval_after = 0.0
                elif winner == stm:
                    eval_after = norm
                else:
                    eval_after = -norm
            delta = eval_after - eval_before
            shaped = shaping_scale * math.tanh(delta / norm)
            per_step_reward[i] += shaped

    for step, reward in zip(result.trajectory, per_step_reward):
        transitions.append(
            Transition(
                state=step.state,
                legal_mask=step.legal_mask,
                primary_index=step.primary_index,
                capture_index=step.capture_index,
                reward=float(reward),
                phase_id=step.phase_id,
                side_to_move=step.side_to_move,
                done=False,
            )
        )
    if transitions:
        transitions[-1].done = True
    return transitions
