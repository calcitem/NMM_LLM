"""
ai/ponder.py — B-75: Background search during the opponent's turn.

After the AI makes a move, predict the most likely opponent replies and start
a full-depth search from each predicted position in parallel (T-E4).  If the
human plays any of the predicted moves, the corresponding result is used
immediately (ponder hit); otherwise all results are discarded and a fresh
search runs as normal.

N_PONDER_BRANCHES (default 2) controls how many opponent replies are pondered
in parallel.  Each branch runs its own GameAI instance with the shared
config but a fresh TT, in a daemon thread.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from game.board import BoardState
    from ai.game_ai import GameAI

log = logging.getLogger(__name__)

N_PONDER_BRANCHES = 2   # how many opponent replies to ponder in parallel


@dataclass
class _Branch:
    """State for one parallel ponder branch."""
    predicted_hash: int
    pred_notation: str
    ponder_ai: "GameAI"
    thread: threading.Thread
    cached_move: "dict | None" = field(default=None)
    completed_ponder_ai: "GameAI | None" = field(default=None)


class PonderManager:
    """Manages N parallel background searches during the opponent's turn (T-E4)."""

    def __init__(self) -> None:
        self._branches: list[_Branch] = []
        self._lock = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────────────

    def start(
        self,
        board: "BoardState",
        game_ai: "GameAI",
        game_notations: list[str],
        trajectory_db=None,
        fullgame_db=None,
        endgame_state=None,
        ngram_model=None,
    ) -> None:
        """Predict the top-N opponent replies and ponder each in a daemon thread."""
        self.stop()  # cancel any previously running branches

        from game.rules import get_all_legal_moves
        from ai.game_ai import GameAI, _order_moves

        opp_moves = get_all_legal_moves(board)
        if not opp_moves:
            return

        ordered = _order_moves(board, opp_moves, None, None)

        # Value-net reorder of top-3 candidates for the primary prediction.
        if game_ai._value_net is not None and len(ordered) >= 2:
            candidates = ordered[:min(3, len(ordered))]
            best_vn: float | None = None
            best_vn_move = candidates[0]
            for m in candidates:
                nb = board.apply_move(m)
                vn = game_ai._value_net.predict(nb, board.turn)
                if best_vn is None or vn > best_vn:
                    best_vn = vn
                    best_vn_move = m
            # Promote VN winner to front.
            ordered = [best_vn_move] + [m for m in ordered if m != best_vn_move]

        # Score each candidate: priority rank + trajectory/fullgame/ngram boosts.
        scored: list[tuple[float, dict]] = []
        if trajectory_db is not None or fullgame_db is not None or ngram_model is not None:
            freq_scores: dict[str, float] = {}
            if trajectory_db is not None:
                try:
                    freq_scores = trajectory_db.query_all_frequencies(board)
                except Exception:
                    pass

            fgdb_best: str | None = None
            if fullgame_db is not None:
                try:
                    fgdb_best = fullgame_db.best_move_validated(board)
                except Exception:
                    pass

            ngram_scores: dict[str, float] = {}
            if ngram_model is not None:
                try:
                    ngram_scores = ngram_model.predict(board.turn, game_notations)
                except Exception:
                    pass

            for i, m in enumerate(ordered):
                notation = _move_notation(m)
                score = float(-i)
                score += freq_scores.get(notation, 0.0) * 5.0
                if fgdb_best is not None and notation == fgdb_best:
                    score += 3.0
                score += ngram_scores.get(notation, 0.0) * 4.0
                scored.append((score, m))
            scored.sort(key=lambda x: x[0], reverse=True)
            top_moves = [m for _, m in scored[:N_PONDER_BRANCHES]]
        else:
            top_moves = ordered[:N_PONDER_BRANCHES]

        branches: list[_Branch] = []
        for predicted_move in top_moves:
            ponder_board = board.apply_move(predicted_move)
            predicted_hash = ponder_board.hash_key
            pred_notation = _move_notation(predicted_move)

            ponder_ai = GameAI(
                color=game_ai.color,
                difficulty=game_ai.difficulty,
                weights=game_ai._weights,
                value_net=game_ai._value_net,
                fullgame_db=fullgame_db,
                endgame_solved_db=game_ai._endgame_solved_db,
            )

            branch = _Branch(
                predicted_hash=predicted_hash,
                pred_notation=pred_notation,
                ponder_ai=ponder_ai,
                thread=threading.Thread(target=lambda b=ponder_board, pa=ponder_ai,
                                        ph=predicted_hash, pn=pred_notation,
                                        br=None: None,  # placeholder; set below
                                        daemon=True, name=f"ponder-{pred_notation}"),
            )

            ponder_notations = list(game_notations) + [pred_notation]

            def _run(pb=ponder_board, pa=ponder_ai, ph=predicted_hash, pn=pred_notation,
                     b=branch) -> None:
                try:
                    move = pa.choose_move(
                        pb,
                        endgame_state=endgame_state,
                        trajectory_db=trajectory_db,
                        game_notations=ponder_notations,
                        fullgame_db=fullgame_db,
                    )
                    with self._lock:
                        b.cached_move = move
                        b.completed_ponder_ai = pa
                    log.info("Ponder complete [%s]: cached AI reply %s", pn,
                             _move_notation(move) if move else "None")
                except Exception as exc:
                    log.debug("Ponder aborted [%s]: %s", pn, exc)

            t = threading.Thread(target=_run, daemon=True, name=f"ponder-{pred_notation}")
            branch.thread = t
            branches.append(branch)

        with self._lock:
            self._branches = branches

        for br in branches:
            br.thread.start()
            log.info("Ponder started: expecting opponent to play %s", br.pred_notation)

    def stop(self) -> None:
        """Interrupt all running ponder branches and wait up to 0.5 s each."""
        with self._lock:
            branches = list(self._branches)
            self._branches = []

        for br in branches:
            try:
                br.ponder_ai.force_stop()
            except Exception:
                pass
        for br in branches:
            if br.thread.is_alive():
                br.thread.join(timeout=0.5)

    def get_result(self, board: "BoardState") -> "tuple[dict, GameAI | None] | None":
        """Return (cached_move, completed_ponder_ai) if any branch predicted this board.

        Checks all branches; returns the first hit.  Must be called after stop()
        so there is no concurrent write race on cached_move.
        """
        with self._lock:
            branches = list(self._branches)

        for br in branches:
            if board.hash_key != br.predicted_hash:
                continue
            if br.cached_move is None:
                return None  # predicted correctly but search not complete
            log.info("Ponder hit [%s] — TT pre-warmed for deepening (B-94)", br.pred_notation)
            return br.cached_move, br.completed_ponder_ai

        return None

    def is_running(self) -> bool:
        with self._lock:
            return any(br.thread.is_alive() for br in self._branches)


def _move_notation(move: dict) -> str:
    s = f"{move['from']}-{move['to']}" if move.get("from") else move.get("to", "")
    if move.get("capture"):
        s += f"x{move['capture']}"
    return s
