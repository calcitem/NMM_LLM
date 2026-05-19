"""
ai/mcts.py — UCT-based Monte Carlo Tree Search for Nine Men's Morris.

Values are stored from self.color's fixed perspective throughout the tree.
UCB selection adapts sign based on whose turn it is at each node.
"""

from __future__ import annotations

import math
import random
import time
from typing import Optional

from game.board import BoardState
from game.rules import get_all_legal_moves, is_terminal
from .heuristics import evaluate, HeuristicWeights, DEFAULT_WEIGHTS, TANH_SCALE

_UCT_C = 1.414  # exploration constant (√2)


class MCTSNode:
    __slots__ = ("board", "move", "parent", "children", "visits",
                 "value_sum", "untried_moves")

    def __init__(
        self,
        board: BoardState,
        move: Optional[dict] = None,
        parent: Optional["MCTSNode"] = None,
    ) -> None:
        self.board = board
        self.move = move
        self.parent = parent
        self.children: list[MCTSNode] = []
        self.visits: int = 0
        self.value_sum: float = 0.0
        self.untried_moves: Optional[list] = None  # None = not yet initialised


class MCTS:
    """
    UCT Monte Carlo Tree Search for Nine Men's Morris.

    Parameters
    ----------
    color : "W" or "B"
        The side MCTS plays for.
    time_limit : float
        Default thinking budget in seconds.
    weights : HeuristicWeights | None
        Heuristic weights for leaf evaluation.
    value_net : ValueNet | None
        If provided, its predict(board, color) replaces the heuristic rollout.
    rollout_depth : int
        Random moves to simulate before the leaf heuristic (0 = pure heuristic).
    """

    def __init__(
        self,
        color: str = "B",
        time_limit: float = 5.0,
        weights: Optional[HeuristicWeights] = None,
        value_net=None,
        rollout_depth: int = 0,
    ) -> None:
        self.color = color
        self.time_limit = time_limit
        self._weights = weights or DEFAULT_WEIGHTS
        self._value_net = value_net
        self.rollout_depth = rollout_depth
        self.nodes_searched: int = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def choose_move(
        self,
        board: BoardState,
        deadline: Optional[float] = None,
    ) -> dict:
        """Run MCTS and return the chosen move for self.color."""
        moves = get_all_legal_moves(board)
        if not moves:
            return {}
        if len(moves) == 1:
            return moves[0]

        if deadline is None:
            deadline = time.time() + self.time_limit

        root = MCTSNode(board)
        root.untried_moves = list(moves)
        random.shuffle(root.untried_moves)

        self.nodes_searched = 0
        while time.time() < deadline:
            node = self._select(root)
            node = self._expand(node)
            value = self._simulate(node)
            self._backpropagate(node, value)
            self.nodes_searched += 1

        if not root.children:
            return moves[0]
        # Pick the most-visited child (robust child selection).
        return max(root.children, key=lambda c: c.visits).move

    # ── Tree operations ───────────────────────────────────────────────────────

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Descend the tree using UCT until a node with untried moves is found."""
        while not self._is_terminal(node):
            if node.untried_moves is None:
                node.untried_moves = list(get_all_legal_moves(node.board))
                random.shuffle(node.untried_moves)
            if node.untried_moves:
                return node
            node = self._best_child(node)
        return node

    def _is_terminal(self, node: MCTSNode) -> bool:
        terminal, _ = is_terminal(node.board)
        return terminal

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Add one unexplored child and return it."""
        if self._is_terminal(node):
            return node
        if node.untried_moves is None:
            node.untried_moves = list(get_all_legal_moves(node.board))
            random.shuffle(node.untried_moves)
        if not node.untried_moves:
            return node
        move = node.untried_moves.pop()
        child = MCTSNode(node.board.apply_move(move), move=move, parent=node)
        node.children.append(child)
        return child

    def _best_child(self, node: MCTSNode) -> MCTSNode:
        """UCT selection.  Sign adapts to whose turn it is at node."""
        log_n = math.log(node.visits)
        my_turn = (node.board.turn == self.color)

        def ucb(c: MCTSNode) -> float:
            q = c.value_sum / c.visits if c.visits else 0.0
            explore = _UCT_C * math.sqrt(log_n / max(c.visits, 1))
            # self.color maximises; opponent minimises self.color's value.
            return q + explore if my_turn else q - explore

        return max(node.children, key=ucb) if my_turn else min(node.children, key=ucb)

    # ── Simulation ────────────────────────────────────────────────────────────

    def _simulate(self, node: MCTSNode) -> float:
        """Return a value in [-1, 1] from self.color's perspective."""
        terminal, winner = is_terminal(node.board)
        if terminal:
            if winner == self.color:
                return 1.0
            if winner is None:
                return 0.0
            return -1.0

        if self._value_net is not None:
            return float(self._value_net.predict(node.board, self.color))

        board = node.board
        for _ in range(self.rollout_depth):
            terminal, winner = is_terminal(board)
            if terminal:
                if winner == self.color:
                    return 1.0
                if winner is None:
                    return 0.0
                return -1.0
            moves = get_all_legal_moves(board)
            if not moves:
                break
            board = board.apply_move(random.choice(moves))

        return self._heuristic_value(board)

    def _heuristic_value(self, board: BoardState) -> float:
        """Map heuristic score to [-1, 1] from self.color's perspective."""
        from game.rules import get_game_phase
        opp   = "B" if self.color == "W" else "W"
        s_own = evaluate(board, self.color, weights=self._weights)
        s_opp = evaluate(board, opp, weights=self._weights)
        raw   = s_own - s_opp
        phase = get_game_phase(board, board.turn)
        scale = TANH_SCALE.get(phase, 180)
        return math.tanh(raw / scale)

    # ── Backpropagation ───────────────────────────────────────────────────────

    def _backpropagate(self, node: MCTSNode, value: float) -> None:
        """Walk up to root updating visit counts and cumulative value."""
        current: Optional[MCTSNode] = node
        while current is not None:
            current.visits += 1
            current.value_sum += value
            current = current.parent
