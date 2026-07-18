"""learned_ai/agents/specialist_router.py — three-specialist inference router.

Loads the three v2 phase specialists (opening, midgame, endgame) and routes
each move choice to the appropriate specialist based on game phase.

Drop-in replacement for `OverseerAdvisor`: same interface (`is_loaded`,
`score_moves`, `set_db`, etc.), so the web/app.py wiring for the "Overseer
player" toggle drives it unchanged.

Routing rule (matches ScaffoldedAgent.choose_move_for_phase):
  * placement phase             → opening specialist
  * move/fly + ≤5 own or opp    → endgame specialist
  * else                        → midgame specialist

At inference each specialist sees:
  * feat_matrix (k, 122) — base 62 + 15-ply lookahead (h/vn/sent/gap)
  * The specialist's forward pass is instant; wall time is dominated by the
    LookaheadAdvisor's 15-ply sentinel calls.

Checkpoint search:
  learned_ai/checkpoints/scaffolded/{s_open_v2,s_mid_v2,s_end_v2}/best.pt

Returns None if all three specialists fail to load — caller falls back to
the classical coordinator.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from game.board import BoardState
from game.rules import get_game_phase

log = logging.getLogger("nmm.specialist_router")


def _move_key(mv: dict) -> tuple:
    return (mv.get("from"), mv.get("to"), mv.get("capture"))


def _load_spec_model(path: Path):
    """Load a ScaffoldedPolicyNet checkpoint. Returns (model, cfg) or (None, {})."""
    if not path.exists():
        return None, {}
    try:
        from learned_ai.models.scaffolded_net import ScaffoldedPolicyNet
        ckpt  = torch.load(str(path), map_location="cpu", weights_only=False)
        cfg   = ckpt.get("model_config", {})
        model = ScaffoldedPolicyNet.from_config(cfg)
        state = ckpt.get("model") or ckpt
        model.load_state_dict(state)
        model.eval()
        return model, cfg
    except Exception as e:
        log.warning("Specialist load failed at %s: %s", path, e)
        return None, {}


class SpecialistRouter:
    """Phase-router over three v2 specialists.  API-compatible with OverseerAdvisor."""

    def __init__(
        self,
        spec_open,
        spec_mid,
        spec_end,
        ckpt_paths: dict[str, str],
        sentinel_advisor=None,
        db=None,
        value_net=None,
        gap_net=None,
        endgame_db=None,
        human_db=None,
        gameai=None,
        lookahead_advisor_open=None,
        lookahead_advisor_mid=None,
        lookahead_advisor_end=None,
        specialist_db=None,
    ) -> None:
        self._spec_open = spec_open
        self._spec_mid  = spec_mid
        self._spec_end  = spec_end
        self._ckpt_paths = ckpt_paths
        self._sentinel  = sentinel_advisor
        self._db        = db          # Malom perfect DB (compat with Overseer set_db)
        self._value_net = value_net
        self._gap_net   = gap_net
        self._endgame_db = endgame_db
        self._human_db  = human_db
        self._gameai    = gameai
        self._la_open   = lookahead_advisor_open
        self._la_mid    = lookahead_advisor_mid
        self._la_end    = lookahead_advisor_end
        self._specialist_db = specialist_db

    # ── OverseerAdvisor-compatible surface ────────────────────────────────────

    def is_loaded(self) -> bool:
        """True when at least one specialist is loaded (router still usable)."""
        return any(m is not None for m in (self._spec_open, self._spec_mid, self._spec_end))

    def set_sentinel(self, sentinel_advisor) -> None:
        self._sentinel = sentinel_advisor
        for la in (self._la_open, self._la_mid, self._la_end):
            if la is not None:
                la._sentinel = sentinel_advisor  # type: ignore[attr-defined]

    def set_db(self, db) -> None:
        """Compat with OverseerAdvisor — Malom perfect DB.  Also wires the
        endgame LookaheadAdvisor's early-terminate DB probe if it isn't set."""
        self._db = db
        if self._endgame_db is None and self._la_end is not None:
            self._la_end._endgame_db = db  # type: ignore[attr-defined]

    def set_value_net(self, value_net) -> None:
        self._value_net = value_net
        for la in (self._la_open, self._la_mid, self._la_end):
            if la is not None:
                la._value_net = value_net  # type: ignore[attr-defined]

    def set_human_db(self, human_db) -> None:
        self._human_db = human_db

    def set_gameai(self, gameai) -> None:
        self._gameai = gameai

    # ── continuous learning ───────────────────────────────────────────────────

    def record_game_result(self, game_record: dict) -> None:
        """Record AI-played boards from a completed game into the SpecialistDB."""
        if self._specialist_db is None:
            return
        try:
            human_color = game_record.get("human_color")
            ai_color = "B" if human_color == "W" else ("W" if human_color == "B" else None)
            if ai_color is None:
                return

            winner = game_record.get("winner")
            result = "W" if winner == ai_color else ("D" if winner is None else "L")

            moves = game_record.get("moves", [])
            ai_boards: list = []
            ai_move_seq: list = []
            phase_counts = {"open": 0, "mid": 0, "end": 0}

            for entry in moves:
                fen_before = entry.get("board_fen_before", "")
                if not fen_before:
                    continue
                try:
                    b = BoardState.from_fen_string(fen_before)
                except Exception:
                    continue
                mv = {
                    "from":    entry.get("from"),
                    "to":      entry.get("to"),
                    "capture": entry.get("capture"),
                }
                try:
                    b_after = b.apply_move(mv)
                except Exception:
                    continue

                if entry.get("color") == ai_color:
                    ai_boards.append(b)
                    ai_boards.append(b_after)
                    notation = entry.get("notation", "")
                    if notation:
                        ai_move_seq.append(notation)
                    move_type = entry.get("type", "move")
                    if move_type == "place":
                        phase_counts["open"] += 1
                    elif move_type == "fly":
                        phase_counts["end"] += 1
                    else:
                        own = int(b.pieces_on_board.get(ai_color, 0))
                        opp_color = "B" if ai_color == "W" else "W"
                        opp = int(b.pieces_on_board.get(opp_color, 0))
                        if own <= 5 or opp <= 5:
                            phase_counts["end"] += 1
                        else:
                            phase_counts["mid"] += 1

            if not ai_boards:
                return

            phase = max(phase_counts, key=lambda k: phase_counts[k])
            self._specialist_db.record_game(
                ai_boards, result, ai_move_seq, phase,
                learner_color=ai_color,
            )
            log.debug("record_game_result: %s boards, phase=%s, result=%s",
                      len(ai_boards) // 2, phase, result)
        except Exception as e:
            log.warning("SpecialistRouter.record_game_result failed: %s", e)

    # ── routing ───────────────────────────────────────────────────────────────

    def _pick_specialist(self, board: BoardState, color: str):
        """Return (specialist_model, lookahead_advisor, phase_label)."""
        phase = get_game_phase(board, color)
        if phase == "place":
            return self._spec_open, self._la_open, "opening"
        own = int(board.pieces_on_board.get(color, 0))
        opp_color = "B" if color == "W" else "W"
        opp = int(board.pieces_on_board.get(opp_color, 0))
        if own <= 5 or opp <= 5:
            return self._spec_end, self._la_end, "endgame"
        return self._spec_mid, self._la_mid, "midgame"

    # ── inference ─────────────────────────────────────────────────────────────

    def score_moves(self, board: BoardState, candidates: list[dict], color: str) -> Optional[list[float]]:
        """Return per-candidate pick probabilities (sum to 1.0).

        Routes to the phase-appropriate specialist; falls back to whichever
        specialist is loaded if the preferred one is missing.

        v4: scores ALL legal moves via encode_position_with_lookahead (full-legal-moves
        mode).  The specialist is not limited to re-ranking a top-K subset.
        """
        if not candidates:
            return None
        try:
            from learned_ai.models.scaffolded_encoder import encode_position_with_lookahead

            spec, la, phase_label = self._pick_specialist(board, color)
            # Fallback ladder: preferred → any other loaded specialist
            if spec is None:
                for alt, alt_la in ((self._spec_mid, self._la_mid),
                                    (self._spec_end, self._la_end),
                                    (self._spec_open, self._la_open)):
                    if alt is not None:
                        spec, la = alt, alt_la
                        phase_label += "→fallback"
                        break
            if spec is None:
                return None

            enc = encode_position_with_lookahead(
                board, color,
                sentinel_advisor=self._sentinel,
                db=None,
                value_net=self._value_net,
                lookahead_advisor=la,
                specialist_db=self._specialist_db,
            )
            if enc is None or not enc.legal_moves:
                return None

            feat = torch.from_numpy(enc.feat_matrix).to(torch.float32)
            with torch.no_grad():
                probs = spec.policy_probs(feat)   # (k,)
            probs_np = probs.cpu().numpy()

            enc_key_to_idx = {_move_key(m): i for i, m in enumerate(enc.legal_moves)}
            result: list[float] = []
            for cand in candidates:
                idx = enc_key_to_idx.get(_move_key(cand))
                result.append(float(probs_np[idx]) if idx is not None and idx < len(probs_np) else 0.0)

            total = sum(result)
            if total > 1e-9:
                result = [v / total for v in result]
            return result

        except Exception as e:
            log.warning("SpecialistRouter.score_moves failed: %s", e, exc_info=True)
            return None


# ── generalist (single model, no phase routing) ───────────────────────────────

class GeneralistAgent:
    """Wraps a single s_gen_v2 ScaffoldedPolicyNet.  Same public interface as SpecialistRouter."""

    def __init__(self, model, la, sentinel_advisor=None, value_net=None, specialist_db=None):
        self._model    = model
        self._la       = la
        self._sentinel = sentinel_advisor
        self._value_net = value_net
        self._specialist_db = specialist_db
        self._gameai   = None
        self._db       = None

    def is_loaded(self) -> bool:
        return self._model is not None

    def set_db(self, db) -> None:
        self._db = db

    def set_sentinel(self, sentinel_advisor) -> None:
        self._sentinel = sentinel_advisor
        if self._la is not None:
            self._la._sentinel = sentinel_advisor  # type: ignore[attr-defined]

    def set_value_net(self, value_net) -> None:
        self._value_net = value_net
        if self._la is not None:
            self._la._value_net = value_net  # type: ignore[attr-defined]

    def set_gameai(self, gameai) -> None:
        self._gameai = gameai

    def record_game_result(self, game_record: dict) -> None:
        pass  # generalist doesn't use specialist_db routing

    def score_moves(self, board: BoardState, candidates: list[dict], color: str) -> Optional[list[float]]:
        if not candidates or self._model is None:
            return None
        try:
            from learned_ai.models.scaffolded_encoder import encode_position_with_lookahead

            enc = encode_position_with_lookahead(
                board, color,
                sentinel_advisor=self._sentinel,
                db=None,
                value_net=self._value_net,
                lookahead_advisor=self._la,
                specialist_db=self._specialist_db,
            )
            if enc is None or not enc.legal_moves:
                return None

            feat = torch.from_numpy(enc.feat_matrix).to(torch.float32)
            with torch.no_grad():
                probs = self._model.policy_probs(feat)
            probs_np = probs.cpu().numpy()

            enc_key_to_idx = {_move_key(m): i for i, m in enumerate(enc.legal_moves)}
            result: list[float] = []
            for cand in candidates:
                idx = enc_key_to_idx.get(_move_key(cand))
                result.append(float(probs_np[idx]) if idx is not None and idx < len(probs_np) else 0.0)

            total = sum(result)
            if total > 1e-9:
                result = [v / total for v in result]
            return result
        except Exception as e:
            log.warning("GeneralistAgent.score_moves failed: %s", e, exc_info=True)
            return None


def load_generalist(
    ckpt_dir: Optional[Path] = None,
    sentinel_advisor=None,
    value_net=None,
    gap_net=None,
    human_db=None,
    specialist_db=None,
    ply_depth: int = 12,
) -> Optional[GeneralistAgent]:
    """Load the s_gen_v2 generalist checkpoint. Returns None if not found."""
    from learned_ai.models.lookahead_advisor import LookaheadAdvisor
    from learned_ai.agents.heuristic_agent import get_heuristic_evaluate

    root = Path(__file__).parent.parent.parent
    if ckpt_dir is None:
        ckpt_dir = root / "learned_ai" / "checkpoints" / "scaffolded"

    gen_path = ckpt_dir / "s_gen_v2" / "best.pt"
    m_gen, _ = _load_spec_model(gen_path)
    if m_gen is None:
        log.info("GeneralistAgent: no checkpoint at %s", gen_path)
        return None

    evaluate_fn = get_heuristic_evaluate()
    try:
        la = LookaheadAdvisor(
            sentinel=sentinel_advisor,
            evaluate_fn=evaluate_fn,
            value_net=value_net,
            gap_net=gap_net,
            human_db=human_db,
            use_sentinel=True,
            ply_depth=ply_depth,
        )
    except Exception as e:
        log.warning("GeneralistAgent LookaheadAdvisor init failed: %s", e)
        la = None

    log.info("GeneralistAgent loaded from %s ply_depth=%d", gen_path, ply_depth)
    return GeneralistAgent(
        model=m_gen,
        la=la,
        sentinel_advisor=sentinel_advisor,
        value_net=value_net,
        specialist_db=specialist_db,
    )


# ── loader ────────────────────────────────────────────────────────────────────

def load_specialist_router(
    ckpt_dir: Optional[Path] = None,
    sentinel_advisor=None,
    db=None,
    human_db=None,
    value_net=None,
    gap_net=None,
    specialist_db=None,
    ply_depth: int = 12,
) -> Optional[SpecialistRouter]:
    """Load the three v2 specialists and their LookaheadAdvisors.

    Returns None only if ALL three specialist checkpoints fail to load.
    """
    from learned_ai.models.lookahead_advisor import LookaheadAdvisor
    from learned_ai.agents.heuristic_agent import get_heuristic_evaluate

    root = Path(__file__).parent.parent.parent
    if ckpt_dir is None:
        ckpt_dir = root / "learned_ai" / "checkpoints" / "scaffolded"

    open_path = ckpt_dir / "s_open_v2" / "best.pt"
    mid_path  = ckpt_dir / "s_mid_v2"  / "best.pt"
    end_path  = ckpt_dir / "s_end_v2"  / "best.pt"

    m_open, cfg_open = _load_spec_model(open_path)
    m_mid,  cfg_mid  = _load_spec_model(mid_path)
    m_end,  cfg_end  = _load_spec_model(end_path)

    if not any((m_open, m_mid, m_end)):
        log.info("SpecialistRouter: no v2 checkpoints found (searched %s, %s, %s)",
                 open_path, mid_path, end_path)
        return None

    evaluate_fn = get_heuristic_evaluate()

    def _mk_la(endgame_db_arg=None):
        try:
            return LookaheadAdvisor(
                sentinel=sentinel_advisor,
                evaluate_fn=evaluate_fn,
                value_net=value_net,
                gap_net=gap_net,
                human_db=human_db,
                use_sentinel=True,
                endgame_db=endgame_db_arg,
                ply_depth=ply_depth,
            )
        except Exception as e:
            log.warning("LookaheadAdvisor init failed: %s", e)
            return None

    la_open = _mk_la() if m_open is not None else None
    la_mid  = _mk_la() if m_mid  is not None else None
    la_end  = _mk_la(endgame_db_arg=db) if m_end is not None else None

    log.info("SpecialistRouter loaded — open=%s mid=%s end=%s ply_depth=%d",
             "OK" if m_open else "missing",
             "OK" if m_mid  else "missing",
             "OK" if m_end  else "missing",
             ply_depth)

    return SpecialistRouter(
        spec_open=m_open, spec_mid=m_mid, spec_end=m_end,
        ckpt_paths={
            "open": str(open_path) if m_open else "",
            "mid":  str(mid_path)  if m_mid  else "",
            "end":  str(end_path)  if m_end  else "",
        },
        sentinel_advisor=sentinel_advisor,
        db=db,
        value_net=value_net,
        gap_net=gap_net,
        endgame_db=db,
        human_db=human_db,
        specialist_db=specialist_db,
        lookahead_advisor_open=la_open,
        lookahead_advisor_mid=la_mid,
        lookahead_advisor_end=la_end,
    )
