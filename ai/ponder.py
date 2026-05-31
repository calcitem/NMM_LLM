"""
ai/ponder.py — B-75: Background search during the opponent's turn.

After the AI makes a move, predict the most likely opponent reply and start
a full-depth search from that position.  If the human plays the predicted
move, the result is used immediately (ponder hit); otherwise it is discarded
and a fresh search runs as normal.
"""
from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from game.board import BoardState
    from ai.game_ai import GameAI

log = logging.getLogger(__name__)


class PonderManager:
    """Manages a single background search thread during the opponent's turn."""

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._ponder_ai: GameAI | None = None
        self._predicted_hash: int | None = None
        self._cached_move: dict | None = None
        self._lock = threading.Lock()

    def start(
        self,
        board: BoardState,         # board after AI's move — opponent to move here
        game_ai: GameAI,           # main AI (source of config: difficulty, weights, VN)
        game_notations: list[str], # full move-notation list up to and including AI's move
        trajectory_db=None,
        fullgame_db=None,
        endgame_state=None,
    ) -> None:
        """Predict the opponent's best reply and begin searching the response.

        Uses priority-based move ordering to predict the opponent move, with an
        optional value-network re-score of the top 3 candidates.
        """
        self.stop()  # cancel any previously running ponder

        from game.rules import get_all_legal_moves
        from ai.game_ai import GameAI, _order_moves

        opp_moves = get_all_legal_moves(board)
        if not opp_moves:
            return

        # Predict opponent move: take the highest-priority move, optionally
        # refined by value network across the top-3 priority candidates.
        ordered = _order_moves(board, opp_moves, None, None)
        predicted_move = ordered[0]

        if game_ai._value_net is not None and len(ordered) >= 2:
            candidates = ordered[:min(3, len(ordered))]
            best_vn: float | None = None
            for m in candidates:
                nb = board.apply_move(m)
                vn = game_ai._value_net.predict(nb, board.turn)  # from opponent's POV
                if best_vn is None or vn > best_vn:
                    best_vn = vn
                    predicted_move = m

        ponder_board = board.apply_move(predicted_move)
        predicted_hash = ponder_board.hash_key
        pred_notation = _move_notation(predicted_move)

        # Shadow AI: same config, fresh transposition table (avoids contaminating main).
        ponder_ai = GameAI(
            color=game_ai.color,
            difficulty=game_ai.difficulty,
            weights=game_ai._weights,
            value_net=game_ai._value_net,
            fullgame_db=fullgame_db,
            endgame_solved_db=game_ai._endgame_solved_db,
            neural_evaluator=game_ai._neural_evaluator,
        )

        with self._lock:
            self._ponder_ai = ponder_ai
            self._predicted_hash = predicted_hash
            self._cached_move = None

        ponder_notations = list(game_notations) + [pred_notation]

        def _run() -> None:
            try:
                move = ponder_ai.choose_move(
                    ponder_board,
                    endgame_state=endgame_state,
                    trajectory_db=trajectory_db,
                    game_notations=ponder_notations,
                    fullgame_db=fullgame_db,
                )
                with self._lock:
                    if self._predicted_hash == predicted_hash:
                        self._cached_move = move
                        log.info(
                            "Ponder complete: opp %s → cached AI reply %s",
                            pred_notation,
                            _move_notation(move) if move else "None",
                        )
            except Exception as exc:
                log.debug("Ponder aborted: %s", exc)

        self._thread = threading.Thread(target=_run, daemon=True, name="ponder")
        self._thread.start()
        log.info("Ponder started: expecting opponent to play %s", pred_notation)

    def stop(self) -> None:
        """Interrupt the ponder search and wait up to 0.5 s for thread exit."""
        with self._lock:
            ai = self._ponder_ai
        if ai is not None:
            ai.force_stop()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=0.5)
        self._thread = None
        with self._lock:
            self._ponder_ai = None

    def get_result(self, board: BoardState) -> dict | None:
        """Return the cached move if board matches the predicted position.

        Must be called AFTER stop() so there is no concurrent write race.
        """
        with self._lock:
            if self._cached_move is None:
                return None
            if board.hash_key != self._predicted_hash:
                log.debug(
                    "Ponder miss: predicted hash %s, actual hash %s",
                    self._predicted_hash, board.hash_key,
                )
                return None
            log.info("Ponder hit — skipping fresh search")
            return self._cached_move

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()


def _move_notation(move: dict) -> str:
    s = f"{move['from']}-{move['to']}" if move.get("from") else move.get("to", "")
    if move.get("capture"):
        s += f"x{move['capture']}"
    return s
