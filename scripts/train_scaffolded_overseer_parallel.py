"""scripts/train_scaffolded_overseer_parallel.py — parallel rollout version.

Identical to train_scaffolded_overseer.py but runs game rollouts across N CPU
worker processes (--workers N, default 4).  Each worker initialises its own
sentinel / DB / value-net / specialist copies once via ProcessPoolExecutor
initializer; the main process sends a fresh model state-dict with every
submitted task so workers always use up-to-date weights.

Usage
-----
.venv/bin/python scripts/train_scaffolded_overseer_parallel.py --workers 4 \\
    --opening-ckpt learned_ai/checkpoints/scaffolded/s_open/best.pt \\
    --midgame-ckpt learned_ai/checkpoints/scaffolded/s_mid/best.pt \\
    --endgame-ckpt learned_ai/checkpoints/scaffolded/s_end/best.pt

Single-worker fallback (same as original):
.venv/bin/python scripts/train_scaffolded_overseer_parallel.py --workers 1
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from collections import deque, Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from game.board import BoardState
from game.rules import is_terminal
from learned_ai.agents.heuristic_agent import HeuristicAgent
from ai.game_ai import GameAI as _GameAI
from ai.human_db import HumanDB as _HumanDB
from learned_ai.models.lookahead_advisor import LookaheadAdvisor
from learned_ai.models.overseer_extras import build_overseer_extras, OVERSEER_EXTRA_DIM
from learned_ai.models.scaffolded_encoder import encode_position_with_lookahead, MOVE_FEAT_DIM, MOVE_FEAT_DIM_WITH_LOOKAHEAD
from learned_ai.models.scaffolded_net import ScaffoldedPolicyNet
from learned_ai.sentinel.infer import load_advisor
from learned_ai.sentinel.labels import dtm_quality
from learned_ai.training.scaffolded_a2c import (
    ScaffoldedStep,
    scaffolded_a2c_update,
    scaffolded_ppo_update,
)

# ── OverseerNet ────────────────────────────────────────────────────────────────

OVERSEER_LOOKAHEAD_PLY = 8
OVERSEER_LOOKAHEAD_DIM = OVERSEER_LOOKAHEAD_PLY * 3          # 36
OVERSEER_BASE_DIM      = MOVE_FEAT_DIM + OVERSEER_LOOKAHEAD_DIM  # 62 + 36 = 98
OVERSEER_FEAT_DIM      = OVERSEER_BASE_DIM + OVERSEER_EXTRA_DIM  # 98 + 8 = 106

STAGE_TAG = "s_over"


class OverseerNet(ScaffoldedPolicyNet):
    """ScaffoldedPolicyNet with move_feat_dim=106 for Overseer training."""

    def __init__(self, **kwargs):
        kwargs.setdefault("move_feat_dim", OVERSEER_FEAT_DIM)
        super().__init__(**kwargs)

    @classmethod
    def from_config(cls, cfg: dict) -> "OverseerNet":
        cfg = dict(cfg)
        cfg["move_feat_dim"] = OVERSEER_FEAT_DIM
        return cls(
            move_feat_dim=cfg["move_feat_dim"],
            value_input_dim=cfg.get("value_input_dim", 23),
            policy_hidden=tuple(cfg.get("policy_hidden", (128, 64))),
            value_hidden=tuple(cfg.get("value_hidden", (64, 32))),
            dropout=float(cfg.get("dropout", 0.0)),
        )


# ── Opening book ──────────────────────────────────────────────────────────────

def _load_opening_book() -> list[list[str]]:
    lines: list[list[str]] = []
    for fname in ("book_openings.json", "learned_openings.json"):
        fpath = _ROOT / "data" / "openings" / fname
        if not fpath.exists():
            continue
        try:
            with open(fpath, encoding="utf-8") as f:
                entries = json.load(f)
            for entry in entries:
                moves = entry.get("line_moves", [])
                if isinstance(moves, list) and len(moves) >= 2:
                    lines.append(moves)
        except Exception:
            pass
    return lines


_OPENING_LINES: list[list[str]] = _load_opening_book()

BOOK_GAME_PROB = 0.50


def _sample_forced_placements(line_moves: list[str], learner_color: str) -> list[str]:
    start = 0 if learner_color == "W" else 1
    return [line_moves[i] for i in range(start, len(line_moves), 2)][:4]


# ── Reward weights ────────────────────────────────────────────────────────────

MALOM_WIN_REWARD    = 0.40
MALOM_LOSS_PENALTY  = 0.30
LAMBDA              = 0.60
DECAY               = 0.98

WIN_REWARD  =  1.0
LOSS_REWARD = -1.0
DRAW_SHORT  =  0.0
DRAW_LONG   =  0.0

_SPEC_FEAT_OFFSET = OVERSEER_BASE_DIM

# ── Optimiser / schedule ──────────────────────────────────────────────────────

LR            = 1e-4
GAMMA_TD      = 0.99
TEMP_START    = 0.50
TEMP_MIN      = 0.45
TEMP_MAX      = 0.90
ENTROPY_COEF  = 0.01
UPDATE_EVERY  = 16
ROLLING_WIN   = 50
DIFF_START    = 1
DIFF_MAX      = 7
ADVANCE_THRESHOLDS = {1: 0.60, 2: 0.60, 3: 0.60, 4: 0.60, 5: 0.60, 6: 0.60}
EXIT_THRESHOLD = 0.60

S1B_REFRESHER_EPOCHS        = 3
S1B_REFRESHER_LR            = 3e-4
S1B_REFRESHER_BATCH         = 32
S1B_WINNER_REFRESHER_EPOCHS = 10
S1B_WINNER_REFRESHER_LR_MUL = 2.0
MAX_PLY       = 140
MAX_PLY_BRANCH = 100
TIME_BUDGET   = 0.05

LOG_EVERY     = 50
LR_SCALE_WIN  = 0.35
LR_SCALE_MIN  = 0.50
LR_SCALE_MAX  = 2.00
RECOVERY_THRESHOLD  = 0.12
RECOVERY_MIN_GAMES  = 30

UPDATE_TARGET_EVERY    = 50
SELF_PLAY_RATIO        = 0.5
BRANCH_EVERY           = 10
MAX_BRANCHES_PER_GAME  = 2
BUCKET_WINDOW          = 300
MAX_PER_BUCKET         = 80

OPENING_EXTENSION_PLY  = 6

PHASE_BUCKETS = ("opening", "midgame", "endgame")

# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class RewardBreakdown:
    total:      float = 0.0
    sentinel:   float = 0.0
    heuristic:  float = 0.0
    value_net:  float = 0.0
    malom_win:  float = 0.0
    malom_trap: float = 0.0
    mill_formed: float = 0.0
    retro:      float = 0.0


@dataclass
class StepDiag:
    reward:           RewardBreakdown
    legal_moves:      int
    chosen_idx:       int
    chosen_prob:      float
    entropy:          float
    top1_prob:        float
    sentinel_mean:    float
    sentinel_chosen:  float
    h_before:         float
    h_after:          float
    h_delta:          float
    vn_before:        float
    vn_after:         float
    vn_delta:         float
    malom_chosen_wdl: str
    malom_chosen_dtm: Optional[float]
    was_top1_policy:  int
    was_top1_heuristic: int


@dataclass
class GameDiag:
    game:                   int
    difficulty:             int
    learner_color:          str
    temperature:            float
    outcome:                float
    win_rate_200:           float
    ply:                    int
    steps:                  int
    update_policy_loss:     Optional[float]
    update_value_loss:      Optional[float]
    update_entropy:         Optional[float]
    reward_total_mean:      float
    reward_sentinel_mean:   float
    reward_heuristic_mean:  float
    reward_value_mean:      float
    reward_malom_win_mean:  float
    reward_malom_trap_mean: float
    reward_retro_mean:      float
    sentinel_mean:          float
    sentinel_chosen_mean:   float
    h_delta_mean:           float
    vn_delta_mean:          float
    chosen_prob_mean:       float
    entropy_mean:           float
    top1_prob_mean:         float
    legal_moves_mean:       float
    policy_top1_rate:       float
    heuristic_top1_rate:    float
    malom_win_move_rate:    float
    malom_unknown_rate:     float
    best_win_rate:          float
    temp_frozen:            int
    lr:                     float
    source_checkpoint:      str
    game_type:              str
    phase_bucket:           str
    is_branch:              int
    branch_ply_start:       int
    target_age:             int
    bucket_opening:         int
    bucket_midgame:         int
    bucket_endgame:         int


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_settings() -> dict:
    p = _ROOT / "data" / "settings.json"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _move_key(mv: dict):
    return (mv.get("from"), mv.get("to"), mv.get("capture"))


def _safe_mean(xs: list[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def _phase_bucket(board: BoardState, moves_into_movement: Optional[int] = None) -> str:
    total_on_board = board.pieces_on_board["W"] + board.pieces_on_board["B"]
    if board.phase == "place":
        return "opening"
    if total_on_board < 12:
        return "endgame"
    if moves_into_movement is not None and moves_into_movement < OPENING_EXTENSION_PLY:
        return "opening"
    return "midgame"


# ── Specialist loading ─────────────────────────────────────────────────────────

def _load_specialist(ckpt_path: Optional[str], label: str) -> Optional[ScaffoldedPolicyNet]:
    if not ckpt_path or not Path(ckpt_path).exists():
        print(f"[s_over] {label} specialist not found: {ckpt_path}")
        return None
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ckpt.get("model_config", {})
        sd_key = "model" if "model" in ckpt else "state_dict"
        sd = ckpt[sd_key]
        if "policy_mlp.0.weight" in sd:
            cfg["move_feat_dim"] = sd["policy_mlp.0.weight"].shape[1]
        else:
            cfg["move_feat_dim"] = MOVE_FEAT_DIM_WITH_LOOKAHEAD
        model = ScaffoldedPolicyNet.from_config(cfg)
        model.load_state_dict(sd)
        model.eval()
        print(f"[s_over] {label} specialist loaded: {ckpt_path} (feat_dim={cfg['move_feat_dim']})")
        return model
    except Exception as e:
        print(f"[s_over] {label} specialist load failed: {e}")
        return None


# ── s1b refresher ─────────────────────────────────────────────────────────────

def _run_s1b_refresher(
    model: OverseerNet,
    device: torch.device,
    data_path: str,
    epochs: int = S1B_REFRESHER_EPOCHS,
    lr: float = S1B_REFRESHER_LR,
    batch: int = S1B_REFRESHER_BATCH,
    deviate_bonus: float = 1.5,
    winner_epochs: int = S1B_WINNER_REFRESHER_EPOCHS,
    winner_lr_mul: float = S1B_WINNER_REFRESHER_LR_MUL,
) -> None:
    p = Path(data_path)
    if not p.exists():
        print(f"[s_over] s1b refresher: data not found ({data_path}) — skipping")
        return

    data          = np.load(str(p), allow_pickle=True)
    feat_matrices = data["feat_matrices"]
    label_dists   = data["label_dists"]
    h_top1_idxs   = data["h_top1_idxs"]
    weights       = data["weights"]
    deviates      = data["deviates"]
    is_winner     = data["is_winner"] if "is_winner" in data else np.ones(len(weights), dtype=bool)
    N             = len(weights)

    effective_weights = weights.copy()
    bonus_mask        = (is_winner) & deviates
    effective_weights[bonus_mask] *= deviate_bonus

    loser_idxs  = [i for i in range(N) if not is_winner[i]]
    winner_idxs = [i for i in range(N) if is_winner[i]]

    for param in model.value_mlp.parameters():
        param.requires_grad = False

    opt_s1b = torch.optim.Adam(
        filter(lambda param: param.requires_grad, model.parameters()), lr=lr
    )

    model.train()
    print(f"[s_over] s1b refresher: loser={len(loser_idxs)} winner={len(winner_idxs)} positions  lr={lr:.2e}")

    def _pad_feat(fm: np.ndarray) -> np.ndarray:
        k, d = fm.shape
        if d >= OVERSEER_FEAT_DIM:
            return fm[:, :OVERSEER_FEAT_DIM]
        pad = np.zeros((k, OVERSEER_FEAT_DIM - d), dtype=np.float32)
        return np.concatenate([fm, pad], axis=1)

    def _run_phase(phase_idxs: list[int], phase_label: str, use_heuristic_target: bool,
                   override_epochs: int = 0) -> None:
        if not phase_idxs:
            return
        n_epochs = override_epochs if override_epochs > 0 else epochs
        for epoch in range(1, n_epochs + 1):
            random.shuffle(phase_idxs)
            ep_loss  = 0.0
            ep_w_sum = 0.0
            for b_start in range(0, len(phase_idxs), batch):
                b = phase_idxs[b_start : b_start + batch]
                if not b:
                    continue
                terms    = []
                bweights = []
                for i in b:
                    feat = torch.tensor(_pad_feat(feat_matrices[i]), dtype=torch.float32).to(device)
                    if use_heuristic_target:
                        k     = feat.shape[0]
                        h_idx = int(h_top1_idxs[i])
                        tgt   = np.full(k, 0.05 / max(k - 1, 1), dtype=np.float32)
                        if 0 <= h_idx < k:
                            tgt[h_idx] = 0.95
                        else:
                            tgt[:] = 1.0 / k
                        target = torch.tensor(tgt, dtype=torch.float32).to(device)
                    else:
                        target = torch.tensor(label_dists[i], dtype=torch.float32).to(device)
                    logits = model.policy_logits(feat)
                    log_p  = F.log_softmax(logits, dim=-1)
                    terms.append(-(target * log_p).sum())
                    bweights.append(float(effective_weights[i]))
                w_t  = torch.tensor(bweights, dtype=torch.float32).to(device)
                loss = (w_t * torch.stack(terms)).sum() / w_t.sum().clamp(min=1e-9)
                opt_s1b.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt_s1b.step()
                ep_loss  += float(loss.item()) * float(w_t.sum())
                ep_w_sum += float(w_t.sum())
            print(f"[s_over]   refresher [{phase_label}] epoch {epoch}/{n_epochs}  loss={ep_loss / max(ep_w_sum, 1e-9):.4f}")

    _run_phase(loser_idxs, "loser→heuristic", use_heuristic_target=True)

    winner_lr = lr * winner_lr_mul
    for param_group in opt_s1b.param_groups:
        param_group["lr"] = winner_lr
    print(f"[s_over]   refresher [winner] using {winner_epochs} epochs at lr={winner_lr:.2e}")
    _run_phase(winner_idxs, "winner", use_heuristic_target=False, override_epochs=winner_epochs)

    for param in model.value_mlp.parameters():
        param.requires_grad = True

    model.eval()
    print("[s_over] s1b refresher done")


# ── Resume / model loading ─────────────────────────────────────────────────────

def _choose_resume_path(args: argparse.Namespace) -> tuple[Optional[Path], str]:
    if getattr(args, "scratch", False):
        return None, "scratch"
    if args.resume:
        p = Path(args.resume)
        if p.exists():
            return p, "explicit_resume"
    s_over_best   = _ROOT / "learned_ai" / "checkpoints" / "scaffolded" / "s_over" / "best.pt"
    s_over_latest = _ROOT / "learned_ai" / "checkpoints" / "scaffolded" / "s_over" / "latest.pt"
    s1c_best      = _ROOT / "learned_ai" / "checkpoints" / "scaffolded" / "s1c" / "best.pt"
    s1b_best      = _ROOT / "learned_ai" / "checkpoints" / "scaffolded" / "s1b" / "best.pt"
    s1_best       = _ROOT / "learned_ai" / "checkpoints" / "scaffolded" / "s1"  / "best.pt"
    candidates = []
    if args.auto_resume_best:
        candidates.append((s_over_best,   "s_over_best"))
        candidates.append((s_over_latest, "s_over_latest"))
    candidates += [
        (s1c_best, "s1c_best"),
        (s1b_best, "s1b_best"),
        (s1_best,  "s1_best"),
    ]
    for p, tag in candidates:
        if p.exists():
            return p, tag
    return None, "scratch"


def _load_model(device: torch.device, resume_path: Optional[Path], force_start_diff: bool = False) -> tuple[OverseerNet, int, float, int, str]:
    if resume_path is None:
        return OverseerNet().to(device), 0, 0.0, DIFF_START, "scratch"
    ckpt = torch.load(resume_path, map_location=device, weights_only=False)
    stage = ckpt.get("stage", "unknown")
    if stage == STAGE_TAG:
        cfg = ckpt.get("model_config", {})
        model = OverseerNet.from_config(cfg).to(device)
        sd_key = "model" if "model" in ckpt else "state_dict"
        model.load_state_dict(ckpt[sd_key])
        start_game  = int(ckpt.get("game_count", 0))
        best_wr     = float(ckpt.get("best_win_rate", 0.0))
        difficulty  = int(ckpt.get("difficulty", DIFF_START))
        if force_start_diff:
            difficulty = max(difficulty, DIFF_START)
        return model, start_game, best_wr, difficulty, str(resume_path)
    else:
        print(f"[s_over] Non-overseer checkpoint ({stage}) — starting fresh OverseerNet")
        model = OverseerNet().to(device)
        return model, 0, 0.0, DIFF_START, str(resume_path)


def _apply_diff_start_override(difficulty: int, args: argparse.Namespace) -> int:
    if args.diff_start is not None:
        return max(1, min(args.diff_start, DIFF_MAX))
    return difficulty


# ── Frozen opponent ────────────────────────────────────────────────────────────

class FrozenOverseerOpponent:
    def __init__(
        self,
        model: OverseerNet,
        device: torch.device,
        sentinel=None,
        value_net=None,
        spec_open: Optional[ScaffoldedPolicyNet] = None,
        spec_mid: Optional[ScaffoldedPolicyNet] = None,
        spec_end: Optional[ScaffoldedPolicyNet] = None,
        gameai=None,
        human_db=None,
        gameai_depth: int = 3,
    ):
        self._model       = copy.deepcopy(model).to(device)
        self._model.eval()
        self._device      = device
        self._sentinel    = sentinel
        self._value_net   = value_net
        self._spec_open   = spec_open
        self._spec_mid    = spec_mid
        self._spec_end    = spec_end
        self._gameai      = gameai
        self._human_db    = human_db
        self._gameai_depth = gameai_depth
        self.last_was_blunder = False
        self.last_thinking    = "frozen_overseer"

    def refresh(self, model: OverseerNet) -> None:
        self._model.load_state_dict(copy.deepcopy(model).state_dict())
        self._model.eval()

    def choose_move(self, board: BoardState) -> dict:
        player = board.turn
        enc = encode_position_with_lookahead(board, player,
                                             sentinel_advisor=self._sentinel,
                                             db=None, value_net=self._value_net,
                                             lookahead_advisor=None,
                                             lookahead_dim=OVERSEER_LOOKAHEAD_DIM)
        if enc is None or not enc.legal_moves:
            return {}
        feat_85 = build_overseer_extras(
            enc.feat_matrix, board, enc, player,
            self._spec_open, self._spec_mid, self._spec_end,
            self._gameai, self._human_db, self._gameai_depth,
            self._device,
        )
        feat_t = torch.tensor(feat_85, dtype=torch.float32).to(self._device)
        with torch.no_grad():
            logits = self._model.policy_logits(feat_t)
            idx    = int(torch.argmax(logits).item())
        return enc.legal_moves[idx]


# ── Temperature / LR helpers ───────────────────────────────────────────────────

def _compute_temperature(game_count: int, max_games: int) -> float:
    progress = min(1.0, game_count / max(max_games * 0.8, 1))
    return float(TEMP_START + (TEMP_MAX - TEMP_START) * progress)


def _adapt_lr(opt: torch.optim.Optimizer, win_rate: float, lr_base: float) -> None:
    scale  = max(LR_SCALE_MIN, min(LR_SCALE_MAX, win_rate / LR_SCALE_WIN))
    new_lr = lr_base * scale
    for g in opt.param_groups:
        g["lr"] = new_lr


def _check_advance(win_history_heuristic: deque, rolling_win: int) -> bool:
    if len(win_history_heuristic) < rolling_win:
        return False
    recent = list(win_history_heuristic)[-rolling_win:]
    wr = sum(1 for x in recent if x == 1.0) / len(recent)
    return wr >= 0.60


# ── Reward computation ─────────────────────────────────────────────────────────

def _get_active_specialist_col(board) -> int:
    if board.phase == "place":
        return 0
    total = board.pieces_on_board["W"] + board.pieces_on_board["B"]
    return 2 if total < 12 else 1


def _compute_per_move_reward(
    enc,
    chosen_idx: int,
    enc_after,
    db_moves=None,
    feat_85: "np.ndarray | None" = None,
    board=None,
) -> tuple[float, RewardBreakdown, dict[str, Any]]:
    rb = RewardBreakdown()
    extra: dict[str, Any] = {"malom_chosen_wdl": "unknown", "malom_chosen_dtm": None}

    if db_moves and feat_85 is not None and board is not None:
        spec_col = _SPEC_FEAT_OFFSET + _get_active_specialist_col(board)
        spec_probs = feat_85[:, spec_col]
        spec_top1_idx = int(np.argmax(spec_probs))

        spec_mv_key  = _move_key(enc.legal_moves[spec_top1_idx])
        spec_db_entry = next((m for m in db_moves if _move_key(m.get("move", {})) == spec_mv_key), None)
        spec_endorses_malom = (
            spec_db_entry is not None
            and str(spec_db_entry.get("wdl", "unknown")) == "win"
        )

        mv_key   = _move_key(enc.legal_moves[chosen_idx])
        db_entry = next((m for m in db_moves if _move_key(m.get("move", {})) == mv_key), None)
        if db_entry:
            wdl = str(db_entry.get("wdl", "unknown"))
            dtm = db_entry.get("dtm")
            extra["malom_chosen_wdl"] = wdl
            extra["malom_chosen_dtm"] = dtm
            if wdl == "win" and spec_endorses_malom:
                rb.malom_win = MALOM_WIN_REWARD * float(dtm_quality("win", dtm))

    rb.total = rb.malom_win
    return float(rb.total), rb, extra


def _retroactive_rescore(trajectory: list[ScaffoldedStep], step_diags: list[StepDiag], outcome: float) -> None:
    n = len(trajectory)
    for t_idx, step in enumerate(trajectory):
        plies_remaining  = n - t_idx - 1
        delta            = LAMBDA * outcome * (DECAY ** plies_remaining)
        step.reward     += delta
        step_diags[t_idx].reward.retro += float(delta)
        step_diags[t_idx].reward.total += float(delta)


# ── Single-game rollout ────────────────────────────────────────────────────────

RETRY_PLY_MIN  =  5
RETRY_PLY_MAX  = 15


@dataclass
class RolloutResult:
    trajectory: list[ScaffoldedStep]
    step_diags: list[StepDiag]
    outcome:    float
    ply:        int
    branch_candidates: list[tuple[int, BoardState, str]]
    retry_board: Optional[BoardState] = None


def _rollout(
    model:         OverseerNet,
    device:        torch.device,
    start_board:   BoardState,
    learner_color: str,
    opponent,
    opp_color:     str,
    sentinel,
    db,
    value_net,
    temperature:   float,
    max_ply:       int,
    record_branches: bool,
    branch_every:  int,
    retry_ply:     int,
    spec_open:     Optional[ScaffoldedPolicyNet],
    spec_mid:      Optional[ScaffoldedPolicyNet],
    spec_end:      Optional[ScaffoldedPolicyNet],
    lookahead_advisor: Optional[LookaheadAdvisor],
    gameai=None,
    human_db=None,
    gameai_depth:  int = 3,
    forced_placements: Optional[list[str]] = None,
) -> RolloutResult:
    board                  = start_board
    ply                    = 0
    move_phase_start_ply:  Optional[int] = None
    game_trajectory:       list[ScaffoldedStep] = []
    step_diags:            list[StepDiag]       = []
    branch_candidates:     list[tuple[int, BoardState, str]] = []
    done                   = False
    outcome                = 0.0
    learner_move_count     = 0
    learner_placement_count = 0
    retry_board: Optional[BoardState] = None

    while ply < max_ply:
        if ply == retry_ply:
            retry_board = board
        if board.phase != "place" and move_phase_start_ply is None:
            move_phase_start_ply = ply
        terminal, winner = is_terminal(board)
        if terminal:
            if winner == learner_color:
                outcome = WIN_REWARD
            elif winner is not None:
                outcome = LOSS_REWARD
            else:
                outcome = DRAW_SHORT if ply < MAX_PLY else DRAW_LONG
            done = True
            break

        player = board.turn

        if player == learner_color:
            if lookahead_advisor is not None:
                lookahead_advisor._use_sentinel = (board.phase != "place")
            enc = encode_position_with_lookahead(board, player,
                                                 sentinel_advisor=sentinel, db=db,
                                                 value_net=value_net,
                                                 lookahead_advisor=lookahead_advisor)
            if enc is None or not enc.legal_moves:
                outcome = LOSS_REWARD
                done = True
                break

            feat_85 = build_overseer_extras(
                enc.feat_matrix, board, enc, learner_color,
                spec_open, spec_mid, spec_end,
                gameai, human_db, gameai_depth,
                device,
            )

            feat_t = torch.tensor(feat_85, dtype=torch.float32).to(device)
            with torch.no_grad():
                logits     = model.policy_logits(feat_t)
                scaled     = logits / max(temperature, 1e-6)
                log_probs  = F.log_softmax(scaled, dim=-1)
                probs      = log_probs.exp()
                if not torch.isfinite(probs).all():
                    probs  = torch.where(torch.isfinite(probs), probs, torch.zeros_like(probs))
                probs      = probs / probs.sum().clamp(min=1e-9)
                entropy    = float((-(probs * log_probs).sum()).item())

                forced_idx = None
                if (forced_placements
                        and board.phase == "place"
                        and learner_placement_count < len(forced_placements)):
                    book_pos = forced_placements[learner_placement_count]
                    for _fi, _m in enumerate(enc.legal_moves):
                        if _m.get("to") == book_pos:
                            forced_idx = _fi
                            break

                if forced_idx is not None:
                    chosen_idx = forced_idx
                else:
                    chosen_idx = int(torch.multinomial(probs.cpu(), 1).item())
                chosen_prob = float(probs[chosen_idx].item())
                top1_prob   = float(probs.max().item())
                was_top1_policy = int(chosen_idx == int(torch.argmax(probs).item()))
                log_prob_old    = float(log_probs[chosen_idx].item())

            move = enc.legal_moves[chosen_idx]
            if board.phase == "place":
                learner_placement_count += 1
            board_after = board.apply_move(move)
            enc_after   = encode_position_with_lookahead(board_after, opp_color,
                                                         sentinel_advisor=sentinel,
                                                         db=None, value_net=value_net,
                                                         lookahead_advisor=None,
                                                         lookahead_dim=OVERSEER_LOOKAHEAD_DIM)

            db_moves = []
            if db is not None:
                try:
                    db_moves = db.query_all_moves(board, player) or []
                except Exception:
                    pass

            reward, rb, extra = _compute_per_move_reward(
                enc, chosen_idx, enc_after, db_moves=db_moves,
                feat_85=feat_85, board=board,
            )

            if enc_after is not None and enc_after.legal_moves:
                next_mf = build_overseer_extras(
                    enc_after.feat_matrix, board_after, enc_after, opp_color,
                    spec_open, spec_mid, spec_end,
                    gameai, human_db, gameai_depth,
                    device,
                )
                next_vi = enc_after.value_input
            else:
                next_mf = np.zeros((1, OVERSEER_FEAT_DIM), dtype=np.float32)
                next_vi = np.zeros(enc.value_input.shape, dtype=np.float32)

            terminal_next, _ = is_terminal(board_after)
            step = ScaffoldedStep(
                move_features=feat_85,
                value_input=enc.value_input,
                chosen_idx=chosen_idx,
                log_prob_old=log_prob_old,
                reward=reward,
                next_move_features=next_mf,
                next_value_input=next_vi,
                done=terminal_next,
            )
            game_trajectory.append(step)

            sentinel_scores   = list(getattr(enc, "sentinel_scores", []) or [])
            sentinel_mean     = float(sum(sentinel_scores) / len(sentinel_scores)) if sentinel_scores else 0.0
            sentinel_chosen   = float(sentinel_scores[chosen_idx]) if sentinel_scores else 0.0
            h_before  = float(getattr(enc, "h_before", 0.0))
            h_after   = float(enc.h_scores_abs[chosen_idx]) if getattr(enc, "h_scores_abs", None) else h_before
            vn_before = float(getattr(enc, "vn_before", 0.0))
            vn_after  = float(enc.vn_scores_abs[chosen_idx]) if getattr(enc, "vn_scores_abs", None) else vn_before
            heuristic_top1 = 0
            if getattr(enc, "h_scores_abs", None):
                heuristic_top1 = int(chosen_idx == int(np.argmax(np.asarray(enc.h_scores_abs))))

            step_diags.append(StepDiag(
                reward=rb,
                legal_moves=len(enc.legal_moves),
                chosen_idx=chosen_idx,
                chosen_prob=chosen_prob,
                entropy=entropy,
                top1_prob=top1_prob,
                sentinel_mean=sentinel_mean,
                sentinel_chosen=sentinel_chosen,
                h_before=h_before,
                h_after=h_after,
                h_delta=h_after - h_before,
                vn_before=vn_before,
                vn_after=vn_after,
                vn_delta=vn_after - vn_before,
                malom_chosen_wdl=extra["malom_chosen_wdl"],
                malom_chosen_dtm=extra["malom_chosen_dtm"],
                was_top1_policy=was_top1_policy,
                was_top1_heuristic=heuristic_top1,
            ))

            learner_move_count += 1
            if record_branches and branch_every > 0 and (learner_move_count % branch_every == 0):
                moves_into_movement = (ply - move_phase_start_ply) if move_phase_start_ply is not None else None
                branch_candidates.append((ply, board, _phase_bucket(board, moves_into_movement)))

            board = board_after

        else:
            try:
                opp_move = opponent.choose_move(board)
            except Exception:
                opp_move = None
            if not opp_move:
                outcome = WIN_REWARD
                done    = True
                break
            board = board.apply_move(opp_move)

        ply += 1

    if not done:
        outcome = DRAW_LONG

    return RolloutResult(
        trajectory=game_trajectory,
        step_diags=step_diags,
        outcome=outcome,
        ply=ply,
        branch_candidates=branch_candidates,
        retry_board=retry_board,
    )


# ── Diagnostic logging ────────────────────────────────────────────────────────

def _build_game_diag(
    game_count:      int,
    difficulty:      int,
    learner_color:   str,
    temperature:     float,
    result:          RolloutResult,
    best_win_rate:   float,
    win_history:     deque,
    last_update_pl:  Optional[float],
    last_update_vl:  Optional[float],
    last_update_ent: Optional[float],
    opt:             torch.optim.Optimizer,
    temp_frozen:     bool,
    source_ckpt:     str,
    game_type:       str,
    phase_bucket:    str,
    is_branch:       bool,
    branch_ply_start: int,
    target_age:      int,
    bucket_counts:   Counter,
) -> GameDiag:
    sd = result.step_diags
    win_rate = sum(win_history) / max(len(win_history), 1)
    return GameDiag(
        game=game_count,
        difficulty=difficulty,
        learner_color=learner_color,
        temperature=round(temperature, 4),
        outcome=float(result.outcome),
        win_rate_200=round(win_rate, 4),
        ply=int(result.ply),
        steps=len(sd),
        update_policy_loss=None if last_update_pl is None else float(last_update_pl),
        update_value_loss=None if last_update_vl is None else float(last_update_vl),
        update_entropy=None if last_update_ent is None else float(last_update_ent),
        reward_total_mean=_safe_mean([d.reward.total for d in sd]),
        reward_sentinel_mean=_safe_mean([d.reward.sentinel for d in sd]),
        reward_heuristic_mean=_safe_mean([d.reward.heuristic for d in sd]),
        reward_value_mean=_safe_mean([d.reward.value_net for d in sd]),
        reward_malom_win_mean=_safe_mean([d.reward.malom_win for d in sd]),
        reward_malom_trap_mean=_safe_mean([d.reward.malom_trap for d in sd]),
        reward_retro_mean=_safe_mean([d.reward.retro for d in sd]),
        sentinel_mean=_safe_mean([d.sentinel_mean for d in sd]),
        sentinel_chosen_mean=_safe_mean([d.sentinel_chosen for d in sd]),
        h_delta_mean=_safe_mean([d.h_delta for d in sd]),
        vn_delta_mean=_safe_mean([d.vn_delta for d in sd]),
        chosen_prob_mean=_safe_mean([d.chosen_prob for d in sd]),
        entropy_mean=_safe_mean([d.entropy for d in sd]),
        top1_prob_mean=_safe_mean([d.top1_prob for d in sd]),
        legal_moves_mean=_safe_mean([float(d.legal_moves) for d in sd]),
        policy_top1_rate=_safe_mean([float(d.was_top1_policy) for d in sd]),
        heuristic_top1_rate=_safe_mean([float(d.was_top1_heuristic) for d in sd]),
        malom_win_move_rate=_safe_mean([1.0 if d.malom_chosen_wdl == "win" else 0.0 for d in sd]),
        malom_unknown_rate=_safe_mean([1.0 if d.malom_chosen_wdl == "unknown" else 0.0 for d in sd]),
        best_win_rate=float(best_win_rate),
        temp_frozen=int(temp_frozen),
        lr=float(opt.param_groups[0]["lr"]),
        source_checkpoint=source_ckpt,
        game_type=game_type,
        phase_bucket=phase_bucket,
        is_branch=int(is_branch),
        branch_ply_start=branch_ply_start,
        target_age=target_age,
        bucket_opening=bucket_counts.get("opening", 0),
        bucket_midgame=bucket_counts.get("midgame", 0),
        bucket_endgame=bucket_counts.get("endgame", 0),
    )


# ── Parallel worker infrastructure ────────────────────────────────────────────
# Workers are long-lived processes (via ProcessPoolExecutor initializer).
# Each worker initialises its own copies of sentinel / DB / value-net /
# specialists / lookahead / GameAI / HumanDB once, then accepts rollout tasks.
# The main process sends a fresh model state-dict with every task so workers
# always use up-to-date weights.

_W: dict = {}   # worker-local resource cache (populated by _worker_init)


def _worker_init(args: argparse.Namespace) -> None:
    """Called once per worker process by ProcessPoolExecutor."""
    global _W
    device = torch.device("cpu")

    sentinel = None
    sent_path = args.sentinel or str(_ROOT / "learned_ai" / "sentinel" / "checkpoints" / "best.pt")
    if Path(sent_path).exists():
        try:
            sentinel = load_advisor(sent_path)
            if sentinel and not sentinel.is_loaded():
                sentinel = None
        except Exception:
            sentinel = None

    db = None
    malom_path = getattr(args, "malom", "") or _load_settings().get("malom_db_path", "")
    if malom_path and Path(malom_path).exists():
        try:
            from learned_ai.sentinel.db_teacher import ExternalSolvedDB
            db = ExternalSolvedDB(malom_path)
            if not db.is_available():
                db = None
        except Exception:
            db = None

    value_net = None
    vn_path = getattr(args, "value_net", "") or str(_ROOT / "data" / "value_net.npz")
    if vn_path and Path(vn_path).exists():
        try:
            from ai.value_net import ValueNet as _VN
            value_net = _VN.load(vn_path)
        except Exception:
            value_net = None

    spec_open = _load_specialist(getattr(args, "opening_ckpt", None) or None, "Opening")
    spec_mid  = _load_specialist(getattr(args, "midgame_ckpt", None) or None, "Midgame")
    spec_end  = _load_specialist(getattr(args, "endgame_ckpt", None) or None, "Endgame")

    lookahead_advisor = None
    if not getattr(args, "no_lookahead", False):
        try:
            from learned_ai.agents.heuristic_agent import get_heuristic_evaluate
            evaluate_fn = get_heuristic_evaluate()
            lookahead_advisor = LookaheadAdvisor(
                sentinel=sentinel,
                value_net=value_net,
                evaluate_fn=evaluate_fn,
                use_sentinel=True,
                ply_depth=OVERSEER_LOOKAHEAD_PLY,
            )
        except Exception:
            lookahead_advisor = None

    gameai_depth = getattr(args, "gameai_depth", 7)
    gameai = None
    try:
        gameai = _GameAI(color="W", difficulty=gameai_depth)
    except Exception:
        gameai = None

    human_db = None
    human_db_path = getattr(args, "human_db", "") or str(_ROOT / "data" / "human_db.sqlite")
    if Path(human_db_path).exists():
        try:
            human_db = _HumanDB(human_db_path)
            if not human_db.is_available():
                human_db = None
        except Exception:
            human_db = None

    model = OverseerNet().to(device)
    model.eval()

    frozen_opp = FrozenOverseerOpponent(
        model, device,
        sentinel=sentinel, value_net=value_net,
        spec_open=spec_open, spec_mid=spec_mid, spec_end=spec_end,
        gameai=gameai, human_db=human_db,
        gameai_depth=gameai_depth,
    )

    _W.update({
        "device":           device,
        "model":            model,
        "frozen_opp":       frozen_opp,
        "sentinel":         sentinel,
        "db":               db,
        "value_net":        value_net,
        "spec_open":        spec_open,
        "spec_mid":         spec_mid,
        "spec_end":         spec_end,
        "lookahead_advisor": lookahead_advisor,
        "gameai":           gameai,
        "human_db":         human_db,
        "args":             args,
        "last_sd_version":  -1,
    })


def _worker_rollout(state_dict: dict, sd_version: int, task: dict) -> dict:
    """Execute one rollout in the worker process.  Returns a picklable dict."""
    w = _W
    device = w["device"]
    args   = w["args"]

    if sd_version != w["last_sd_version"]:
        sd_torch = {k: torch.from_numpy(v) for k, v in state_dict.items()}
        w["model"].load_state_dict(sd_torch)
        w["model"].eval()
        w["frozen_opp"].refresh(w["model"])
        w["last_sd_version"] = sd_version

    learner_color = task["learner_color"]
    opp_color     = task["opp_color"]

    if task["opponent_type"] == "frozen":
        opponent = w["frozen_opp"]
    else:
        from learned_ai.agents.heuristic_agent import GameAI as _GA
        _h = HeuristicAgent(color=opp_color, difficulty=task["difficulty"], game_ai=None)
        _h._inner = _GA(color=opp_color, difficulty=task["difficulty"],
                        override_time_budget=args.time_budget)
        opponent = _h

    start_board = task.get("start_board") or BoardState.new_game()

    result = _rollout(
        model=w["model"],
        device=device,
        start_board=start_board,
        learner_color=learner_color,
        opponent=opponent,
        opp_color=opp_color,
        sentinel=w["sentinel"],
        db=w["db"],
        value_net=w["value_net"],
        temperature=task["temperature"],
        max_ply=task["max_ply"],
        record_branches=task.get("record_branches", False),
        branch_every=task.get("branch_every", 0),
        retry_ply=task.get("game_retry_ply", 0),
        spec_open=w["spec_open"],
        spec_mid=w["spec_mid"],
        spec_end=w["spec_end"],
        lookahead_advisor=w["lookahead_advisor"],
        gameai=w["gameai"],
        human_db=w["human_db"],
        gameai_depth=task.get("gameai_depth", 7),
        forced_placements=task.get("forced_placements"),
    )

    return {"task": task, "result": result}


# ── Parallel training loop ─────────────────────────────────────────────────────

def _run_parallel(args: argparse.Namespace) -> None:
    """Training loop using ProcessPoolExecutor for parallel rollouts."""
    from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
    import multiprocessing as mp

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[s_over] Device: {device}  Workers: {args.workers}")
    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    resume_path, source_tag = _choose_resume_path(args)
    model, start_game, best_win_rate, difficulty, source_checkpoint = _load_model(
        device, resume_path, force_start_diff=args.force_start_diff
    )
    difficulty = _apply_diff_start_override(difficulty, args)
    if resume_path is None:
        print("[s_over] No checkpoint — starting from scratch")
    else:
        print(f"[s_over] Resuming ({source_tag}): {resume_path}")
    print(f"[s_over] Starting at game {start_game}, difficulty {difficulty}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    update_fn = scaffolded_ppo_update if args.ppo else scaffolded_a2c_update

    game_count              = start_game
    win_history:              deque[float] = deque(maxlen=args.rolling_win)
    win_history_heuristic:    deque[float] = deque(maxlen=args.rolling_win)
    malom_win_rate_history:   deque[float] = deque(maxlen=10)
    ep_steps: list[ScaffoldedStep] = []
    best_win_rate_at_diff   = 0.0
    last_update_pl          = None
    last_update_vl          = None
    last_update_ent         = None
    branch_bucket_history:    deque[str]   = deque(maxlen=args.bucket_window)
    diag_buffer: list[GameDiag] = []

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path        = out_dir / "train_log.jsonl"
    update_log_path = out_dir / "update_log.jsonl"

    if getattr(args, "scratch", False):
        for stale in (log_path, update_log_path):
            if stale.exists():
                stale.unlink()
                print(f"[s_over] --scratch: cleared {stale.name}")

    if not args.no_s1b_refresher:
        print("[s_over] Running s1b refresher in main process before starting workers")
        _run_s1b_refresher(model, device, args.s1b_data,
                           epochs=args.s1b_refresher_epochs,
                           lr=args.s1b_refresher_lr)

    sd_version = 0

    def _get_state_dict() -> dict:
        # Return numpy arrays rather than tensors so pickle sends plain bytes
        # instead of using PyTorch's fd-sharing mechanism, which races with
        # spawn workers connecting to the /tmp/pymp-*/listener-* socket.
        return {k: v.cpu().detach().numpy() for k, v in model.state_dict().items()}

    def _make_task(
        task_type: str,
        learner_color: str,
        opponent_type: str,
        start_board=None,
        branch_ply: int = 0,
        bucket: str = "main",
        max_ply: int = None,
        record_branches: bool = False,
        game_retry_ply: int = 0,
        forced_placements=None,
    ) -> dict:
        opp_color = "B" if learner_color == "W" else "W"
        return {
            "type":             task_type,
            "learner_color":    learner_color,
            "opp_color":        opp_color,
            "opponent_type":    opponent_type,
            "difficulty":       difficulty,
            "temperature":      _compute_temperature(game_count, args.max_games),
            "max_ply":          max_ply if max_ply is not None else args.max_ply,
            "branch_every":     args.branch_every,
            "record_branches":  record_branches,
            "game_retry_ply":   game_retry_ply,
            "forced_placements": forced_placements,
            "start_board":      start_board,
            "gameai_depth":     args.gameai_depth,
            "branch_ply":       branch_ply,
            "bucket":           bucket,
        }

    def _new_main_task() -> dict:
        lc       = "W" if rng.random() < 0.5 else "B"
        opp_type = "frozen" if rng.random() < args.self_play_ratio else "heuristic"
        retry_ply = rng.randint(RETRY_PLY_MIN, RETRY_PLY_MAX)
        fp = None
        if _OPENING_LINES and rng.random() < BOOK_GAME_PROB:
            line = _OPENING_LINES[rng.randint(0, len(_OPENING_LINES) - 1)]
            fp = _sample_forced_placements(line, lc)
        return _make_task(
            "main", lc, opp_type,
            record_branches=(args.max_branches_per_game > 0),
            game_retry_ply=retry_ply,
            forced_placements=fp,
        )

    ctx = mp.get_context("spawn")

    print(f"[s_over] Spawning {args.workers} worker processes…")
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_worker_init,
        initargs=(args,),
        mp_context=ctx,
    ) as executor:

        pending: set = set()

        def _submit(task: dict):
            sd = _get_state_dict()
            f = executor.submit(_worker_rollout, sd, sd_version, task)
            pending.add(f)

        # Pre-fill pipeline
        n_initial = min(args.workers * 2, max(args.max_games - game_count, 1))
        for _ in range(n_initial):
            _submit(_new_main_task())

        training_done = False

        while game_count < args.max_games and pending and not training_done:
            done_set, _ = wait(pending, return_when=FIRST_COMPLETED)
            for f in done_set:
                pending.discard(f)

            for future in done_set:
                try:
                    work = future.result()
                except Exception as exc:
                    print(f"[s_over] Worker exception: {exc}")
                    if game_count < args.max_games:
                        _submit(_new_main_task())
                    continue

                task          = work["task"]
                result        = work["result"]
                task_type     = task["type"]
                learner_color = task["learner_color"]
                opp_color     = task["opp_color"]
                is_heuristic  = task["opponent_type"] == "heuristic"
                game_type     = "vs_heuristic" if is_heuristic else "vs_frozen"
                bucket        = task.get("bucket", "main")
                branch_ply_v  = task.get("branch_ply", 0)

                if result.trajectory:
                    _retroactive_rescore(result.trajectory, result.step_diags, result.outcome)

                def _wh_append(outcome: float) -> None:
                    win_history.append(
                        1.0 if outcome == WIN_REWARD else
                        (0.5 if outcome == DRAW_SHORT else 0.0)
                    )
                    if is_heuristic:
                        win_history_heuristic.append(
                            1.0 if outcome == WIN_REWARD else
                            (0.5 if outcome == DRAW_SHORT else 0.0)
                        )

                if task_type == "main":
                    if result.outcome == WIN_REWARD:
                        ep_steps.extend(result.trajectory)
                    _wh_append(result.outcome)
                    game_count += 1

                    # Dispatch retry for non-wins
                    if result.outcome != WIN_REWARD and result.retry_board is not None and game_count < args.max_games:
                        _submit(_make_task(
                            "retry", learner_color, task["opponent_type"],
                            start_board=result.retry_board,
                            game_retry_ply=0,
                            record_branches=False,
                            max_ply=args.max_ply,
                        ))

                    # Dispatch branch games
                    branches_spawned = 0
                    for bp_ply, bp_board, bp_bucket in result.branch_candidates:
                        if branches_spawned >= args.max_branches_per_game:
                            break
                        bc = Counter(branch_bucket_history)
                        if bc.get(bp_bucket, 0) >= args.max_per_bucket:
                            continue
                        if game_count < args.max_games:
                            _submit(_make_task(
                                "branch", learner_color, "frozen",
                                start_board=bp_board,
                                branch_ply=bp_ply,
                                bucket=bp_bucket,
                                max_ply=args.max_ply_branch,
                                record_branches=False,
                            ))
                            branches_spawned += 1

                    bucket_counts = Counter(branch_bucket_history)
                    _diag = _build_game_diag(
                        game_count, difficulty, learner_color,
                        task["temperature"], result,
                        best_win_rate, win_history,
                        last_update_pl, last_update_vl, last_update_ent,
                        opt, False, source_checkpoint,
                        game_type=game_type, phase_bucket="main", is_branch=False,
                        branch_ply_start=0, target_age=0,
                        bucket_counts=bucket_counts,
                    )
                    diag_buffer.append(_diag)
                    malom_win_rate_history.append(_diag.malom_win_move_rate)

                    if game_count % 10 == 0:
                        _hwr = sum(win_history_heuristic) / max(len(win_history_heuristic), 1)
                        _awr = sum(win_history) / max(len(win_history), 1)
                        _mwr = sum(malom_win_rate_history) / max(len(malom_win_rate_history), 1)
                        _oc  = "W" if result.outcome == WIN_REWARD else ("L" if result.outcome == LOSS_REWARD else "D")
                        _gt  = "heur" if is_heuristic else "self"
                        print(f"[s_over] {game_count:6d} {_gt:4s} {learner_color} | diff {difficulty} | {_oc} ply={result.ply:3d} | hwr={_hwr:.3f} awr={_awr:.3f} malom={_mwr:.1%} | temp={task['temperature']:.2f} lr={opt.param_groups[0]['lr']:.5f}")

                    # Submit replacement main task
                    if game_count < args.max_games:
                        _submit(_new_main_task())

                elif task_type == "retry":
                    if result.outcome in (WIN_REWARD, DRAW_SHORT):
                        ep_steps.extend(result.trajectory)
                    _wh_append(result.outcome)
                    game_count += 1
                    if game_count % 10 == 0:
                        _roc = "W" if result.outcome == WIN_REWARD else ("L" if result.outcome == LOSS_REWARD else "D")
                        print(f"[s_over] {game_count:6d} retry {learner_color} |          | {_roc} ply={result.ply:3d}")

                elif task_type == "branch":
                    if result.outcome in (WIN_REWARD, DRAW_SHORT):
                        ep_steps.extend(result.trajectory)
                    if result.trajectory:
                        branch_bucket_history.append(bucket)
                    _wh_append(result.outcome)
                    game_count += 1
                    bucket_counts = Counter(branch_bucket_history)
                    diag_buffer.append(_build_game_diag(
                        game_count, difficulty, learner_color,
                        task["temperature"], result,
                        best_win_rate, win_history,
                        last_update_pl, last_update_vl, last_update_ent,
                        opt, False, source_checkpoint,
                        game_type="branch", phase_bucket=bucket, is_branch=True,
                        branch_ply_start=branch_ply_v, target_age=0,
                        bucket_counts=bucket_counts,
                    ))
                    if game_count % 10 == 0:
                        _boc = "W" if result.outcome == WIN_REWARD else ("L" if result.outcome == LOSS_REWARD else "D")
                        print(f"[s_over] {game_count:6d}  +b  {learner_color} | {bucket:7s} | {_boc} ply={result.ply:3d}")

                # ── Model update ──────────────────────────────────────────────
                if len(ep_steps) >= args.update_every:
                    last_update_pl, last_update_vl, last_update_ent = update_fn(
                        model, opt, ep_steps, device,
                        gamma=args.gamma_td, entropy_coef=args.entropy_coef,
                    )
                    upd_entry = {
                        "game":        game_count,
                        "policy_loss": None if last_update_pl  is None else float(last_update_pl),
                        "value_loss":  None if last_update_vl  is None else float(last_update_vl),
                        "entropy":     None if last_update_ent is None else float(last_update_ent),
                        "lr":          float(opt.param_groups[0]["lr"]),
                        "batch_steps": len(ep_steps),
                    }
                    with open(update_log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(upd_entry) + "\n")
                    ep_steps.clear()
                    sd_version += 1   # workers will reload weights on next task

                # ── Periodic log + checkpoint ─────────────────────────────────
                if game_count % args.log_every == 0 and diag_buffer:
                    win_rate     = sum(win_history_heuristic) / max(len(win_history_heuristic), 1)
                    win_rate_all = sum(win_history) / max(len(win_history), 1)
                    _adapt_lr(opt, win_rate, args.lr)

                    # Recovery
                    if (len(win_history_heuristic) >= RECOVERY_MIN_GAMES
                            and win_rate < RECOVERY_THRESHOLD):
                        best_ckpt = out_dir / f"best{difficulty}.pt"
                        if best_ckpt.exists():
                            ckpt_r = torch.load(str(best_ckpt), map_location=device, weights_only=False)
                            model.load_state_dict(ckpt_r["model"])
                            model.to(device)
                            opt = torch.optim.Adam(model.parameters(), lr=args.lr)
                            win_history.clear()
                            win_history_heuristic.clear()
                            sd_version += 1
                            print(f"[s_over] Recovery: reloaded best{difficulty}.pt (win rate was {win_rate:.2f})")

                    with open(log_path, "a", encoding="utf-8") as f:
                        for d in diag_buffer:
                            f.write(json.dumps(asdict(d)) + "\n")

                    main_diags = [d for d in diag_buffer if not d.is_branch]
                    branch_diags = [d for d in diag_buffer if d.is_branch]
                    diag_buffer.clear()
                    bc = Counter(branch_bucket_history)
                    win_rate = sum(win_history_heuristic) / max(len(win_history_heuristic), 1)
                    win_rate_all = sum(win_history) / max(len(win_history), 1)

                    last_main = next((d for d in reversed(main_diags) if main_diags), None)
                    if last_main:
                        d = last_main
                        _sign = lambda v: f"{'+' if v >= 0 else ''}{v:.3f}"
                        print(
                            f"[s_over] game {game_count:6d} | diff {difficulty} | "
                            f"win-{args.rolling_win}={win_rate:.3f} | all={win_rate_all:.3f} | "
                            f"temp={_compute_temperature(game_count, args.max_games):.2f} | "
                            f"outcome={d.outcome:+.2f} | lr={opt.param_groups[0]['lr']:.5f} | "
                            f"rew={_sign(d.reward_total_mean)} | "
                            f"mw={_sign(d.reward_malom_win_mean)} | "
                            f"p_top1={d.policy_top1_rate:.2f} h_top1={d.heuristic_top1_rate:.2f} | "
                            f"branches={len(branch_diags)} "
                            f"[op={bc.get('opening',0)} mid={bc.get('midgame',0)} end={bc.get('endgame',0)}]"
                        )

                    ckpt = {
                        "model":             model.state_dict(),
                        "model_config":      model.get_config(),
                        "stage":             STAGE_TAG,
                        "game_count":        game_count,
                        "best_win_rate":     best_win_rate,
                        "difficulty":        difficulty,
                        "source_checkpoint": source_checkpoint,
                        "lr":                float(opt.param_groups[0]["lr"]),
                        "temperature":       float(_compute_temperature(game_count, args.max_games)),
                    }
                    torch.save(ckpt, out_dir / "latest.pt")

                    if (win_rate > best_win_rate_at_diff
                            and len(win_history_heuristic) >= min(100, args.rolling_win)):
                        best_win_rate_at_diff = win_rate
                        ckpt["best_win_rate"] = best_win_rate_at_diff
                        torch.save(ckpt, out_dir / f"best{difficulty}.pt")
                        torch.save(ckpt, out_dir / "best.pt")
                        if win_rate > best_win_rate:
                            best_win_rate = win_rate
                        print(f"[s_over]  → best diff-{difficulty} win rate: {best_win_rate_at_diff:.3f}  (saved best{difficulty}.pt)")

                # ── Difficulty advancement ────────────────────────────────────
                if len(win_history_heuristic) >= args.rolling_win:
                    win_rate = sum(win_history_heuristic) / len(win_history_heuristic)
                    advance_thr = ADVANCE_THRESHOLDS.get(difficulty, args.advance_threshold)
                    if difficulty >= args.diff_max:
                        if win_rate >= args.exit_threshold:
                            print(f"[s_over] *** {win_rate:.3f} win rate vs difficulty {difficulty} — done! ***")
                            training_done = True
                            for pf in pending:
                                pf.cancel()
                            pending.clear()
                            break
                    elif win_rate >= advance_thr or _check_advance(win_history_heuristic, args.rolling_win):
                        prev_diff = difficulty
                        difficulty += 1
                        win_history.clear()
                        win_history_heuristic.clear()
                        best_win_rate_at_diff = 0.0
                        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
                        sd_version += 1
                        print(f"[s_over] *** Advanced to difficulty {difficulty} (was {win_rate:.3f} vs diff {prev_diff}) ***")
                        prev_best = out_dir / f"best{prev_diff}.pt"
                        if prev_best.exists():
                            ckpt_prev = torch.load(str(prev_best), map_location=device, weights_only=False)
                            model.load_state_dict(ckpt_prev["model"])
                            model.to(device)
                            sd_version += 1
                            print(f"[s_over] Loaded best{prev_diff}.pt as starting point for diff {difficulty}")

    # ── Final flush ───────────────────────────────────────────────────────────
    if ep_steps:
        update_fn(model, opt, ep_steps, device, gamma=args.gamma_td, entropy_coef=args.entropy_coef)
    if diag_buffer:
        with open(log_path, "a", encoding="utf-8") as f:
            for d in diag_buffer:
                f.write(json.dumps(asdict(d)) + "\n")

    ckpt = {
        "model":             model.state_dict(),
        "model_config":      model.get_config(),
        "stage":             STAGE_TAG,
        "game_count":        game_count,
        "best_win_rate":     best_win_rate,
        "difficulty":        difficulty,
        "source_checkpoint": source_checkpoint,
        "lr":                float(opt.param_groups[0]["lr"]),
        "temperature":       float(_compute_temperature(game_count, args.max_games)),
    }
    torch.save(ckpt, out_dir / "latest.pt")
    print(f"\n[s_over] Done. Games: {game_count}  Best win rate: {best_win_rate:.3f}")
    print(f"[s_over] Checkpoint: {out_dir / 'best.pt'}")
    print(f"[s_over] Logs: {log_path} and {update_log_path}")


# ── Sequential training loop (--workers 1) ────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    if getattr(args, "workers", 1) > 1:
        _run_parallel(args)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[s_over] Device: {device}")
    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    sentinel = None
    sent_path = args.sentinel or str(_ROOT / "learned_ai" / "sentinel" / "checkpoints" / "best.pt")
    if Path(sent_path).exists():
        sentinel = load_advisor(sent_path)
        if sentinel and sentinel.is_loaded():
            print(f"[s_over] Sentinel loaded: {sent_path}")
        else:
            sentinel = None
    if sentinel is None:
        print("[s_over] Sentinel unavailable — sentinel reward = 0")

    db = None
    malom_path = args.malom or _load_settings().get("malom_db_path", "")
    if malom_path and Path(malom_path).exists():
        try:
            from learned_ai.sentinel.db_teacher import ExternalSolvedDB
            db = ExternalSolvedDB(malom_path)
            if db.is_available():
                print(f"[s_over] Malom DB loaded: {malom_path}")
            else:
                db = None
        except Exception as e:
            print(f"[s_over] Malom DB failed ({e})")
    if db is None:
        print("[s_over] Malom DB unavailable — Malom rewards = 0")

    value_net = None
    vn_path = args.value_net or str(_ROOT / "data" / "value_net.npz")
    if vn_path and Path(vn_path).exists():
        try:
            from ai.value_net import ValueNet as _ValueNet
            value_net = _ValueNet.load(vn_path)
            print(f"[s_over] Value net loaded: {vn_path}")
        except Exception as e:
            print(f"[s_over] Value net load failed ({e}) — VN features will be 0")
    else:
        print("[s_over] No value net — VN features will be 0")

    spec_open = _load_specialist(args.opening_ckpt or None, "Opening")
    spec_mid  = _load_specialist(args.midgame_ckpt or None, "Midgame")
    spec_end  = _load_specialist(args.endgame_ckpt or None, "Endgame")

    resume_path, source_tag = _choose_resume_path(args)
    model, start_game, best_win_rate, difficulty, source_checkpoint = _load_model(
        device, resume_path, force_start_diff=args.force_start_diff
    )
    difficulty = _apply_diff_start_override(difficulty, args)
    if resume_path is None:
        print("[s_over] No checkpoint found — starting from scratch")
    else:
        print(f"[s_over] Resuming from ({source_tag}): {resume_path}")
    print(f"[s_over] Starting at game {start_game}, difficulty {difficulty}")

    from learned_ai.agents.heuristic_agent import get_heuristic_evaluate
    _evaluate_fn = get_heuristic_evaluate()
    lookahead_advisor: Optional[LookaheadAdvisor] = None
    if not args.no_lookahead:
        lookahead_advisor = LookaheadAdvisor(
            sentinel=sentinel,
            value_net=value_net,
            evaluate_fn=_evaluate_fn,
            use_sentinel=True,
            ply_depth=OVERSEER_LOOKAHEAD_PLY,
        )
        print(f"[s_over] LookaheadAdvisor enabled ({OVERSEER_LOOKAHEAD_PLY}-ply, feat_dim={OVERSEER_LOOKAHEAD_DIM})")
    else:
        print("[s_over] LookaheadAdvisor disabled (--no-lookahead)")

    gameai_depth = args.gameai_depth
    overseer_gameai: Optional[_GameAI] = None
    try:
        overseer_gameai = _GameAI(color="W", difficulty=gameai_depth)
        print(f"[s_over] GameAI loaded (depth={gameai_depth})")
    except Exception as _e:
        print(f"[s_over] GameAI load failed ({_e}) — gameai features will be neutral")

    overseer_human_db = None
    human_db_path = args.human_db or str(_ROOT / "data" / "human_db.sqlite")
    if Path(human_db_path).exists():
        try:
            overseer_human_db = _HumanDB(human_db_path)
            if overseer_human_db.is_available():
                print(f"[s_over] HumanDB loaded: {human_db_path}")
            else:
                overseer_human_db = None
        except Exception as _e:
            print(f"[s_over] HumanDB load failed ({_e})")
    else:
        print(f"[s_over] HumanDB not found at {human_db_path}")

    frozen_opp = FrozenOverseerOpponent(
        model, device, sentinel=sentinel, value_net=value_net,
        spec_open=spec_open, spec_mid=spec_mid, spec_end=spec_end,
        gameai=overseer_gameai, human_db=overseer_human_db,
        gameai_depth=gameai_depth,
    )
    games_since_target_update = 0

    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    opt       = torch.optim.Adam(model.parameters(), lr=args.lr)
    update_fn = scaffolded_ppo_update if args.ppo else scaffolded_a2c_update

    game_count             = start_game
    temperature            = args.temp_start
    win_history:              deque[float] = deque(maxlen=args.rolling_win)
    win_history_heuristic:    deque[float] = deque(maxlen=args.rolling_win)
    malom_win_rate_history:   deque[float] = deque(maxlen=10)
    ep_steps: list[ScaffoldedStep] = []
    last_update_pl   = None
    last_update_vl   = None
    last_update_ent  = None
    best_win_rate_at_diff = 0.0

    branch_bucket_history: deque[str] = deque(maxlen=args.bucket_window)

    log_path        = out_dir / "train_log.jsonl"
    update_log_path = out_dir / "update_log.jsonl"

    if getattr(args, "scratch", False):
        for stale in (log_path, update_log_path):
            if stale.exists():
                stale.unlink()
                print(f"[s_over] --scratch: cleared {stale.name}")

    print(f"[s_over] Starting at game {game_count}, difficulty {difficulty}")
    print(f"[s_over] Self-play ratio {args.self_play_ratio:.0%}, "
          f"branch every {args.branch_every} turns, "
          f"max {args.max_branches_per_game} branches/game")

    if not args.no_s1b_refresher:
        print(f"[s_over] Running s1b refresher before diff {difficulty} training")
        _run_s1b_refresher(model, device, args.s1b_data,
                           epochs=args.s1b_refresher_epochs,
                           lr=args.s1b_refresher_lr)

    diag_buffer: list[GameDiag] = []

    def _do_rollout(start_board, learner_color, opp_color, opponent,
                    max_ply, record_branches, branch_every, retry_ply,
                    forced_placements=None):
        return _rollout(
            model=model,
            device=device,
            start_board=start_board,
            learner_color=learner_color,
            opponent=opponent,
            opp_color=opp_color,
            sentinel=sentinel,
            db=db,
            value_net=value_net,
            temperature=temperature,
            max_ply=max_ply,
            record_branches=record_branches,
            branch_every=branch_every,
            retry_ply=retry_ply,
            spec_open=spec_open,
            spec_mid=spec_mid,
            spec_end=spec_end,
            lookahead_advisor=lookahead_advisor,
            gameai=overseer_gameai,
            human_db=overseer_human_db,
            gameai_depth=gameai_depth,
            forced_placements=forced_placements,
        )

    while game_count < args.max_games:
        temperature = _compute_temperature(game_count, args.max_games)

        if games_since_target_update >= args.update_target_every:
            frozen_opp.refresh(model)
            games_since_target_update = 0
            print(f"[s_over] Frozen model updated at game {game_count}")

        learner_color = "W" if rng.random() < 0.5 else "B"
        opp_color     = "B" if learner_color == "W" else "W"

        use_self_play = rng.random() < args.self_play_ratio
        if use_self_play:
            opponent  = frozen_opp
            game_type = "vs_frozen"
        else:
            from learned_ai.agents.heuristic_agent import GameAI as _GA
            _h = HeuristicAgent(color=opp_color, difficulty=difficulty, game_ai=None)
            _h._inner = _GA(color=opp_color, difficulty=difficulty, override_time_budget=args.time_budget)
            opponent  = _h
            game_type = "vs_heuristic"

        game_forced_placements: Optional[list[str]] = None
        if _OPENING_LINES and rng.random() < BOOK_GAME_PROB:
            line = _OPENING_LINES[rng.randint(0, len(_OPENING_LINES) - 1)]
            game_forced_placements = _sample_forced_placements(line, learner_color)

        game_retry_ply = rng.randint(RETRY_PLY_MIN, RETRY_PLY_MAX)
        result = _do_rollout(
            start_board=BoardState.new_game(),
            learner_color=learner_color,
            opp_color=opp_color,
            opponent=opponent,
            max_ply=args.max_ply,
            record_branches=(args.max_branches_per_game > 0),
            branch_every=args.branch_every,
            retry_ply=game_retry_ply,
            forced_placements=game_forced_placements,
        )

        if result.trajectory:
            _retroactive_rescore(result.trajectory, result.step_diags, result.outcome)

        if result.outcome == WIN_REWARD:
            ep_steps.extend(result.trajectory)
        elif result.outcome in (LOSS_REWARD, DRAW_SHORT) and result.retry_board is not None:
            confirm_result = _do_rollout(
                start_board=result.retry_board,
                learner_color=learner_color,
                opp_color=opp_color,
                opponent=opponent,
                max_ply=args.max_ply,
                record_branches=False,
                branch_every=0,
                retry_ply=0,
            )
            if confirm_result.trajectory:
                _retroactive_rescore(confirm_result.trajectory, confirm_result.step_diags,
                                     confirm_result.outcome)
            confirmed = (
                (result.outcome == LOSS_REWARD and confirm_result.outcome == LOSS_REWARD) or
                (result.outcome == DRAW_SHORT  and confirm_result.outcome == DRAW_SHORT)
            )
            if confirmed and result.trajectory:
                ep_steps.extend(result.trajectory)
            if confirm_result.outcome in (WIN_REWARD, DRAW_SHORT):
                ep_steps.extend(confirm_result.trajectory)
            game_count += 1
            games_since_target_update += 1
            win_history.append(1.0 if confirm_result.outcome == WIN_REWARD else
                               (0.5 if confirm_result.outcome == DRAW_SHORT else 0.0))
            if game_type == "vs_heuristic":
                win_history_heuristic.append(
                    1.0 if confirm_result.outcome == WIN_REWARD else
                    (0.5 if confirm_result.outcome == DRAW_SHORT else 0.0)
                )
            _coc = "W" if confirm_result.outcome == WIN_REWARD else ("L" if confirm_result.outcome == LOSS_REWARD else "D")
            if game_count % 10 == 0:
                print(f"[s_over] {game_count:6d}  r{game_retry_ply:2d} {learner_color} |          | {_coc} ply={confirm_result.ply:3d} | (from ply {game_retry_ply}) {'[learn]' if confirmed else '[skip]'}")

        win_history.append(1.0 if result.outcome == WIN_REWARD else
                           (0.5 if result.outcome == DRAW_SHORT else 0.0))
        if game_type == "vs_heuristic":
            win_history_heuristic.append(
                1.0 if result.outcome == WIN_REWARD else
                (0.5 if result.outcome == DRAW_SHORT else 0.0)
            )
        game_count += 1
        games_since_target_update += 1

        bucket_counts = Counter(branch_bucket_history)
        _diag = _build_game_diag(
            game_count, difficulty, learner_color, temperature, result,
            best_win_rate, win_history, last_update_pl, last_update_vl, last_update_ent,
            opt, False, source_checkpoint,
            game_type=game_type, phase_bucket="main", is_branch=False,
            branch_ply_start=0, target_age=games_since_target_update,
            bucket_counts=bucket_counts,
        )
        diag_buffer.append(_diag)
        malom_win_rate_history.append(_diag.malom_win_move_rate)

        if game_count % 10 == 0:
            _hwr = sum(win_history_heuristic) / max(len(win_history_heuristic), 1)
            _awr = sum(win_history) / max(len(win_history), 1)
            _mwr = sum(malom_win_rate_history) / max(len(malom_win_rate_history), 1)
            _oc  = "W" if result.outcome == WIN_REWARD else ("L" if result.outcome == LOSS_REWARD else "D")
            _gt  = "heur" if game_type == "vs_heuristic" else "self"
            print(f"[s_over] {game_count:6d} {_gt:4s} {learner_color} | diff {difficulty} | {_oc} ply={result.ply:3d} | hwr={_hwr:.3f} awr={_awr:.3f} malom={_mwr:.1%} | temp={temperature:.2f} lr={opt.param_groups[0]['lr']:.5f}")

        if result.outcome != WIN_REWARD and result.retry_board is not None:
            retry_result = _do_rollout(
                start_board=result.retry_board,
                learner_color=learner_color,
                opp_color=opp_color,
                opponent=opponent,
                max_ply=args.max_ply,
                record_branches=False,
                branch_every=0,
                retry_ply=0,
            )
            if retry_result.trajectory:
                _retroactive_rescore(retry_result.trajectory, retry_result.step_diags, retry_result.outcome)
                if retry_result.outcome in (WIN_REWARD, DRAW_SHORT):
                    ep_steps.extend(retry_result.trajectory)
            win_history.append(1.0 if retry_result.outcome == WIN_REWARD else
                               (0.5 if retry_result.outcome == DRAW_SHORT else 0.0))
            if game_type == "vs_heuristic":
                win_history_heuristic.append(
                    1.0 if retry_result.outcome == WIN_REWARD else
                    (0.5 if retry_result.outcome == DRAW_SHORT else 0.0)
                )
            game_count += 1
            games_since_target_update += 1
            _roc = "W" if retry_result.outcome == WIN_REWARD else ("L" if retry_result.outcome == LOSS_REWARD else "D")
            if game_count % 10 == 0:
                print(f"[s_over] {game_count:6d} retry {learner_color} |          | {_roc} ply={retry_result.ply:3d} | (from ply {game_retry_ply})")

        branches_spawned = 0
        candidates = list(result.branch_candidates)
        rng.shuffle(candidates)
        seen_buckets: set[str] = set()
        ordered_candidates: list[tuple[int, BoardState, str]] = []
        for cand in candidates:
            if cand[2] not in seen_buckets:
                ordered_candidates.insert(0, cand)
                seen_buckets.add(cand[2])
            else:
                ordered_candidates.append(cand)

        for branch_ply, branch_board, bucket in ordered_candidates:
            if branches_spawned >= args.max_branches_per_game:
                break
            bucket_counts = Counter(branch_bucket_history)
            if bucket_counts.get(bucket, 0) >= args.max_per_bucket:
                continue

            branch_result = _do_rollout(
                start_board=branch_board,
                learner_color=learner_color,
                opp_color=opp_color,
                opponent=frozen_opp,
                max_ply=args.max_ply_branch,
                record_branches=False,
                branch_every=0,
                retry_ply=0,
            )

            if branch_result.trajectory:
                _retroactive_rescore(branch_result.trajectory, branch_result.step_diags, branch_result.outcome)
                if branch_result.outcome in (WIN_REWARD, DRAW_SHORT):
                    ep_steps.extend(branch_result.trajectory)
                branch_bucket_history.append(bucket)
                branches_spawned += 1
                game_count += 1
                games_since_target_update += 1
                win_history.append(1.0 if branch_result.outcome == WIN_REWARD else
                                   (0.5 if branch_result.outcome == DRAW_SHORT else 0.0))

                bucket_counts = Counter(branch_bucket_history)
                diag_buffer.append(_build_game_diag(
                    game_count, difficulty, learner_color, temperature, branch_result,
                    best_win_rate, win_history, last_update_pl, last_update_vl, last_update_ent,
                    opt, False, source_checkpoint,
                    game_type="branch", phase_bucket=bucket, is_branch=True,
                    branch_ply_start=branch_ply, target_age=games_since_target_update,
                    bucket_counts=bucket_counts,
                ))

                if game_count % 10 == 0:
                    _boc = "W" if branch_result.outcome == WIN_REWARD else ("L" if branch_result.outcome == LOSS_REWARD else "D")
                    print(f"[s_over] {game_count:6d}  +b  {learner_color} | {bucket:7s} | {_boc} ply={branch_result.ply:3d} | (from ply {branch_ply})")

        if len(ep_steps) >= args.update_every:
            last_update_pl, last_update_vl, last_update_ent = update_fn(
                model, opt, ep_steps, device, gamma=args.gamma_td, entropy_coef=args.entropy_coef
            )
            upd_entry = {
                "game":         game_count,
                "policy_loss":  None if last_update_pl  is None else float(last_update_pl),
                "value_loss":   None if last_update_vl  is None else float(last_update_vl),
                "entropy":      None if last_update_ent is None else float(last_update_ent),
                "lr":           float(opt.param_groups[0]["lr"]),
                "batch_steps":  len(ep_steps),
            }
            with open(update_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(upd_entry) + "\n")
            ep_steps.clear()

        if game_count % args.log_every == 0 and diag_buffer:
            win_rate     = sum(win_history_heuristic) / max(len(win_history_heuristic), 1)
            win_rate_all = sum(win_history) / max(len(win_history), 1)

            _adapt_lr(opt, win_rate, args.lr)

            if (len(win_history_heuristic) >= RECOVERY_MIN_GAMES
                    and win_rate < RECOVERY_THRESHOLD):
                best_ckpt = out_dir / f"best{difficulty}.pt"
                if best_ckpt.exists():
                    ckpt_r = torch.load(str(best_ckpt), map_location=device, weights_only=False)
                    model.load_state_dict(ckpt_r["model"])
                    model.to(device)
                    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
                    frozen_opp.refresh(model)
                    win_history.clear()
                    win_history_heuristic.clear()
                    temperature = TEMP_START
                    print(f"[s_over] Recovery: reloaded best{difficulty}.pt (win rate was {win_rate:.2f}, temp reset to {TEMP_START})")

            main_diags   = [d for d in diag_buffer if not d.is_branch]
            branch_diags = [d for d in diag_buffer if d.is_branch]
            bc = Counter(branch_bucket_history)

            with open(log_path, "a", encoding="utf-8") as f:
                for d in diag_buffer:
                    f.write(json.dumps(asdict(d)) + "\n")
            diag_buffer.clear()

            last_main = next((d for d in reversed(main_diags) if main_diags), None)
            if last_main:
                d = last_main
                _sign = lambda v: f"{'+' if v >= 0 else ''}{v:.3f}"
                print(
                    f"[s_over] game {game_count:6d} | diff {difficulty} | "
                    f"win-{args.rolling_win}={win_rate:.3f} | all={win_rate_all:.3f} | "
                    f"temp={temperature:.2f} | "
                    f"outcome={d.outcome:+.2f} | lr={opt.param_groups[0]['lr']:.5f} | "
                    f"rew={_sign(d.reward_total_mean)} | "
                    f"mw={_sign(d.reward_malom_win_mean)} | "
                    f"p_top1={d.policy_top1_rate:.2f} h_top1={d.heuristic_top1_rate:.2f} | "
                    f"branches={len(branch_diags)} "
                    f"[op={bc.get('opening',0)} mid={bc.get('midgame',0)} end={bc.get('endgame',0)}]"
                )

            ckpt = {
                "model":             model.state_dict(),
                "model_config":      model.get_config(),
                "stage":             STAGE_TAG,
                "game_count":        game_count,
                "best_win_rate":     best_win_rate,
                "difficulty":        difficulty,
                "source_checkpoint": source_checkpoint,
                "lr":                float(opt.param_groups[0]["lr"]),
                "temperature":       float(temperature),
            }
            torch.save(ckpt, out_dir / "latest.pt")

            if win_rate > best_win_rate_at_diff and len(win_history_heuristic) >= min(100, args.rolling_win):
                best_win_rate_at_diff = win_rate
                ckpt["best_win_rate"] = best_win_rate_at_diff
                torch.save(ckpt, out_dir / f"best{difficulty}.pt")
                torch.save(ckpt, out_dir / "best.pt")
                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                print(f"[s_over]  → best diff-{difficulty} win rate: {best_win_rate_at_diff:.3f}  (saved best{difficulty}.pt)")

        if len(win_history_heuristic) >= args.rolling_win:
            win_rate = sum(win_history_heuristic) / len(win_history_heuristic)
            advance_thr = ADVANCE_THRESHOLDS.get(difficulty, args.advance_threshold)
            if difficulty >= args.diff_max:
                if win_rate >= args.exit_threshold:
                    print(f"[s_over] *** {win_rate:.3f} win rate vs difficulty {difficulty} — done! ***")
                    break
            elif win_rate >= advance_thr or _check_advance(win_history_heuristic, args.rolling_win):
                prev_diff = difficulty
                difficulty += 1
                win_history.clear()
                win_history_heuristic.clear()
                print(f"[s_over] *** Advanced to difficulty {difficulty} (was {win_rate:.3f} vs diff {prev_diff}) ***")

                prev_best = out_dir / f"best{prev_diff}.pt"
                if prev_best.exists():
                    ckpt_prev = torch.load(str(prev_best), map_location=device, weights_only=False)
                    model.load_state_dict(ckpt_prev["model"])
                    model.to(device)
                    print(f"[s_over] Loaded best{prev_diff}.pt as starting point for diff {difficulty}")
                else:
                    print(f"[s_over] best{prev_diff}.pt not found — continuing from current weights")

                best_win_rate_at_diff = 0.0
                opt = torch.optim.Adam(model.parameters(), lr=args.lr)
                frozen_opp.refresh(model)

    if ep_steps:
        update_fn(model, opt, ep_steps, device, gamma=args.gamma_td, entropy_coef=args.entropy_coef)
    if diag_buffer:
        with open(log_path, "a", encoding="utf-8") as f:
            for d in diag_buffer:
                f.write(json.dumps(asdict(d)) + "\n")

    ckpt = {
        "model":             model.state_dict(),
        "model_config":      model.get_config(),
        "stage":             STAGE_TAG,
        "game_count":        game_count,
        "best_win_rate":     best_win_rate,
        "difficulty":        difficulty,
        "source_checkpoint": source_checkpoint,
        "lr":                float(opt.param_groups[0]["lr"]),
        "temperature":       float(temperature),
    }
    torch.save(ckpt, out_dir / "latest.pt")
    print(f"\n[s_over] Done. Games: {game_count}  Best win rate: {best_win_rate:.3f}")
    print(f"[s_over] Checkpoint: {out_dir / 'best.pt'}")
    print(f"[s_over] Logs: {log_path} and {update_log_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Overseer meta-policy training — parallel rollouts")
    p.add_argument("--workers",             type=int,   default=4,
                   help="Number of parallel rollout worker processes (1 = sequential, default 4)")
    p.add_argument("--resume",             default="",  type=str)
    p.add_argument("--auto-resume-best",   action="store_true")
    p.add_argument("--scratch",            action="store_true")
    p.add_argument("--out-dir",  default=str(_ROOT / "learned_ai" / "checkpoints" / "scaffolded" / "s_over"))
    p.add_argument("--sentinel", default=str(_ROOT / "learned_ai" / "sentinel" / "checkpoints" / "best.pt"))
    p.add_argument("--malom",    default="", type=str)
    p.add_argument("--value-net",default=str(_ROOT / "data" / "value_net.npz"), type=str)
    p.add_argument("--ppo",      action="store_true")
    p.add_argument("--max-games",           type=int,   default=5000)
    p.add_argument("--seed",                type=int,   default=42)
    p.add_argument("--lr",                  type=float, default=LR)
    p.add_argument("--gamma-td",            type=float, default=GAMMA_TD)
    p.add_argument("--entropy-coef",        type=float, default=ENTROPY_COEF)
    p.add_argument("--update-every",        type=int,   default=UPDATE_EVERY)
    p.add_argument("--rolling-win",         type=int,   default=ROLLING_WIN)
    p.add_argument("--force-start-diff",    action="store_true")
    p.add_argument("--diff-start",          type=int,   default=None)
    p.add_argument("--diff-max",            type=int,   default=DIFF_MAX)
    p.add_argument("--advance-threshold",   type=float, default=0.50)
    p.add_argument("--exit-threshold",      type=float, default=EXIT_THRESHOLD)
    p.add_argument("--temp-start",          type=float, default=TEMP_START)
    p.add_argument("--log-every",           type=int,   default=LOG_EVERY)
    p.add_argument("--max-ply",             type=int,   default=MAX_PLY)
    p.add_argument("--max-ply-branch",      type=int,   default=MAX_PLY_BRANCH)
    p.add_argument("--time-budget",         type=float, default=TIME_BUDGET)
    p.add_argument("--self-play-ratio",     type=float, default=SELF_PLAY_RATIO)
    p.add_argument("--update-target-every", type=int,   default=UPDATE_TARGET_EVERY)
    p.add_argument("--branch-every",        type=int,   default=BRANCH_EVERY)
    p.add_argument("--max-branches-per-game", type=int, default=0)
    p.add_argument("--bucket-window",       type=int,   default=BUCKET_WINDOW)
    p.add_argument("--max-per-bucket",      type=int,   default=MAX_PER_BUCKET)
    p.add_argument("--opening-ckpt",        type=str,   default="")
    p.add_argument("--midgame-ckpt",        type=str,   default="")
    p.add_argument("--endgame-ckpt",        type=str,   default="")
    p.add_argument("--no-lookahead",        action="store_true")
    p.add_argument("--human-db",            type=str,   default="")
    p.add_argument("--gameai-depth",        type=int,   default=7)
    p.add_argument("--s1b-data",             type=str,   default=str(_ROOT / "learned_ai" / "data" / "human_imitation.npz"))
    p.add_argument("--s1b-refresher-epochs", type=int,   default=S1B_REFRESHER_EPOCHS)
    p.add_argument("--s1b-refresher-lr",     type=float, default=S1B_REFRESHER_LR)
    p.add_argument("--no-s1b-refresher",     action="store_true")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
