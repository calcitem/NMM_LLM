"""scripts/train_s_mid_v2.py — Midgame specialist v2: 15-ply lookahead + gap net.

Trains on the movement phase when >= 12 total pieces are on the board.
Reward: sentinel delta + heuristic delta + mill bonus; Malom reward = 0.
Gap net included in lookahead (15-ply × 4 signals).  No mill-dance, no
mobility reward, no Malom trap — simpler, cleaner signal.

Resume chain: explicit --resume → s_mid_v2/best.pt → scratch

Usage
-----
.venv/bin/python scripts/train_s_mid_v2.py --max-games 20
.venv/bin/python scripts/train_s_mid_v2.py --auto-resume-best
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from game.board import BoardState, MILLS
from game.rules import is_terminal
from learned_ai.agents.heuristic_agent import HeuristicAgent
from learned_ai.agents.heuristic_agent import GameAI as _GA
from learned_ai.models.lookahead_advisor import LookaheadAdvisor
from learned_ai.models.scaffolded_encoder import (
    encode_position_with_lookahead,
    MOVE_FEAT_DIM_WITH_LOOKAHEAD,
    MOVE_FEAT_DIM_WITH_TOPK,
    VALUE_INPUT_DIM,
)
from learned_ai.data.specialist_db import SpecialistDB
from learned_ai.models.scaffolded_net import ScaffoldedPolicyNet
from learned_ai.sentinel.infer import load_advisor
from learned_ai.training.scaffolded_a2c import (
    ScaffoldedStep,
    scaffolded_a2c_update,
    scaffolded_ppo_update,
)
from learned_ai.training.advance_stats import (
    check_advance as _sanmill_check_advance,
    advance_target,
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


# Board position index lookup for history feature encoding
_POS_NAMES = [
    "a1","a4","a7","b2","b4","b6","c3","c4","c5",
    "d1","d2","d3","d5","d6","d7","e3","e4","e5",
    "f2","f4","f6","g1","g4","g7",
]
_POS_IDX: dict[str, int] = {p: i for i, p in enumerate(_POS_NAMES)}
RAW_BOARD_FEATURES = len(_POS_NAMES) * 2   # 24 positions × 2 colors = 48


def _build_raw_board_features(board) -> np.ndarray:
    """One-hot occupancy per position, learner-agnostic. Layout: [w0,b0, w1,b1, ...]."""
    feats = np.zeros(RAW_BOARD_FEATURES, dtype=np.float32)
    for i, pos in enumerate(_POS_NAMES):
        p = board.positions.get(pos)
        if p == "W":
            feats[2 * i]     = 1.0
        elif p == "B":
            feats[2 * i + 1] = 1.0
    return feats


# ── Simplified rollout heuristic (no extended tactical search) ────────────────

def _simple_evaluate(board: BoardState, color: str) -> float:
    """Fast heuristic for rollout move selection: mills + mobility + blocked."""
    terminal, winner = is_terminal(board)
    if terminal:
        return 1e9 if winner == color else -1e9
    opp = "B" if color == "W" else "W"
    from game.board import ADJACENCY
    our_mills = sum(1 for m in MILLS if all(board.positions.get(p) == color for p in m))
    opp_mills = sum(1 for m in MILLS if all(board.positions.get(p) == opp for p in m))
    our_mob = sum(
        1 for pos, piece in board.positions.items()
        if piece == color
        for adj in ADJACENCY.get(pos, [])
        if board.positions.get(adj) is None
    )
    opp_mob = sum(
        1 for pos, piece in board.positions.items()
        if piece == opp
        for adj in ADJACENCY.get(pos, [])
        if board.positions.get(adj) is None
    )
    blocked = sum(
        1 for pos, piece in board.positions.items()
        if piece == opp
        and all(board.positions.get(adj) is not None for adj in ADJACENCY.get(pos, []))
    )
    return float(500 * (our_mills - opp_mills) + 10 * (our_mob - opp_mob) + 50 * blocked)


# ── Difficulty / history helpers ─────────────────────────────────────────────

def _heuristic_time_budget(level: int) -> float:
    """Heuristic opponent time budget: 0.1 s at L1 → 14.0 s at L20 (exponential ramp)."""
    return 0.1 * (140.0 ** ((level - 1) / 19.0))


def _specialist_time_budget(level: int) -> float:
    """Specialist (learner) alpha-beta time budget: 0.5 s at L1 → 20.0 s at L20."""
    return 0.5 * (40.0 ** ((level - 1) / 19.0))


def _build_history_features(history: deque, n: int = 3) -> np.ndarray:
    """Encode the last n moves as normalised position indices (-1 if absent)."""
    feats = np.full(n * HIST_FLOATS_PER_MOVE, -1.0, dtype=np.float32)
    for slot, mv in enumerate(list(history)[-n:]):
        base    = slot * HIST_FLOATS_PER_MOVE
        from_p  = mv.get("from")
        to_p    = mv.get("to")
        cap_p   = mv.get("capture")
        feats[base]   = _POS_IDX.get(from_p, -1) / 23.0 if from_p else -1.0
        feats[base+1] = _POS_IDX.get(to_p,   -1) / 23.0 if to_p   else -1.0
        feats[base+2] = _POS_IDX.get(cap_p,  -1) / 23.0 if cap_p  else -1.0
    return feats


# ── Stage tag ─────────────────────────────────────────────────────────────────

STAGE_TAG = "s_mid_v2"
OUT_DIR   = "learned_ai/checkpoints/scaffolded/s_mid_v2"

# ── Reward weights ────────────────────────────────────────────────────────────

ALPHA      = 0.20   # sentinel quality delta
BETA       = 0.15   # heuristic delta
MILL_BONUS = 0.25   # larger mill bonus — midgame mills more decisive
LAMBDA     = 0.70   # Batch 1: 0.5 → 0.7 (outcome matters more)
DECAY      = 0.99   # Batch 1: 0.98 → 0.99 (outcome reaches further back)
EXPLORE_COEF = 0.08 # bonus for winning with non-heuristic-top1 moves (Option A)

WIN_REWARD  =  1.0
LOSS_REWARD = -1.0
DRAW_SHORT  = -0.10   # 2026-07-14: 0.00 → -0.10
DRAW_LONG   = -0.25   # 2026-07-14: -0.15 → -0.25

# ── Optimiser / schedule ──────────────────────────────────────────────────────

LR            = 1e-4
GAMMA_TD      = 0.99
TEMP_START    = 0.50
TEMP_MAX      = 0.90
ENTROPY_COEF  = 0.01
UPDATE_EVERY  = 16
ROLLING_WIN   = 100
DIFF_START    = 1
DIFF_MAX      = 20

S1B_REFRESHER_EPOCHS = 3
S1B_REFRESHER_LR     = 3e-4
S1B_REFRESHER_BATCH  = 32
MAX_PLY        = 60
MAX_PLY_BRANCH = 60

# ── History features + raw board ─────────────────────────────────────────────
N_HISTORY             = 3    # last N moves appended to value input as context
HIST_FLOATS_PER_MOVE  = 3    # from_idx_norm, to_idx_norm, capture_idx_norm
# Value input layout: [23 encoder base | 9 history | 48 raw-board one-hot] = 80 floats
VALUE_INPUT_DIM_WITH_HISTORY = VALUE_INPUT_DIM + N_HISTORY * HIST_FLOATS_PER_MOVE + 48  # 80

LOG_EVERY    = 50
LR_SCALE_WIN = 0.35
LR_SCALE_MIN = 0.50
LR_SCALE_MAX = 2.00
RECOVERY_THRESHOLD  = 0.12
RECOVERY_MIN_GAMES  = 30

UPDATE_TARGET_EVERY   = 50
SELF_PLAY_RATIO       = 0.5
BRANCH_EVERY          = 10
MAX_BRANCHES_PER_GAME = 2
BUCKET_WINDOW         = 300
MAX_PER_BUCKET        = 80

OPENING_EXTENSION_PLY = 6

PHASE_BUCKETS = ("opening", "midgame", "endgame")


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class RewardBreakdown:
    total:       float = 0.0
    sentinel:    float = 0.0
    heuristic:   float = 0.0
    mill_formed: float = 0.0
    retro:       float = 0.0


@dataclass
class StepDiag:
    reward:            RewardBreakdown
    legal_moves:       int
    chosen_idx:        int
    chosen_prob:       float
    entropy:           float
    top1_prob:         float
    sentinel_mean:     float
    sentinel_chosen:   float
    h_before:          float
    h_after:           float
    h_delta:           float
    vn_before:         float
    vn_after:          float
    vn_delta:          float
    malom_chosen_wdl:  str
    malom_chosen_dtm:  Optional[float]
    was_top1_policy:   int
    was_top1_heuristic: int


@dataclass
class GameDiag:
    game:                    int
    difficulty:              int
    learner_color:           str
    temperature:             float
    outcome:                 float
    win_rate_200:            float
    ply:                     int
    steps:                   int
    update_policy_loss:      Optional[float]
    update_value_loss:       Optional[float]
    update_entropy:          Optional[float]
    reward_total_mean:       float
    reward_sentinel_mean:    float
    reward_heuristic_mean:   float
    reward_retro_mean:       float
    sentinel_mean:           float
    sentinel_chosen_mean:    float
    h_delta_mean:            float
    vn_delta_mean:           float
    chosen_prob_mean:        float
    entropy_mean:            float
    top1_prob_mean:          float
    legal_moves_mean:        float
    policy_top1_rate:        float
    heuristic_top1_rate:     float
    malom_win_move_rate:     float
    malom_unknown_rate:      float
    best_win_rate:           float
    temp_frozen:             int
    lr:                      float
    source_checkpoint:       str
    game_type:               str
    phase_bucket:            str
    is_branch:               int
    branch_ply_start:        int
    target_age:              int
    bucket_opening:          int
    bucket_midgame:          int
    bucket_endgame:          int


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_settings() -> dict:
    p = _ROOT / "data" / "settings.json"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


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


def _run_s1b_refresher(
    model: ScaffoldedPolicyNet,
    device: torch.device,
    data_path: str,
    epochs: int = S1B_REFRESHER_EPOCHS,
    lr: float = S1B_REFRESHER_LR,
    batch: int = S1B_REFRESHER_BATCH,
    deviate_bonus: float = 1.5,
) -> None:
    p = Path(data_path)
    if not p.exists():
        print(f"[s_mid_v2] s1b refresher: data not found ({data_path}) — skipping")
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
    print(f"[s_mid_v2] s1b refresher: loser={len(loser_idxs)} winner={len(winner_idxs)} positions  lr={lr:.2e}")

    def _pad_feat(fm: np.ndarray) -> np.ndarray:
        k, d = fm.shape
        if d >= MOVE_FEAT_DIM_WITH_LOOKAHEAD:
            return fm[:, :MOVE_FEAT_DIM_WITH_LOOKAHEAD]
        pad = np.zeros((k, MOVE_FEAT_DIM_WITH_LOOKAHEAD - d), dtype=np.float32)
        return np.concatenate([fm, pad], axis=1)

    def _run_phase(phase_idxs: list[int], phase_label: str, use_heuristic_target: bool) -> None:
        if not phase_idxs:
            return
        for epoch in range(1, epochs + 1):
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
            print(f"[s_mid_v2]   refresher [{phase_label}] epoch {epoch}/{epochs}  loss={ep_loss / max(ep_w_sum, 1e-9):.4f}")

    _run_phase(loser_idxs, "loser→heuristic", use_heuristic_target=True)
    _run_phase(winner_idxs, "winner", use_heuristic_target=False)

    for param in model.value_mlp.parameters():
        param.requires_grad = True

    model.eval()
    print("[s_mid_v2] s1b refresher done")


def _choose_resume_path(args: argparse.Namespace) -> tuple[Optional[Path], str]:
    if args.resume:
        p = Path(args.resume)
        if p.exists():
            return p, "explicit_resume"
    s_mid_v2_best = _ROOT / "learned_ai" / "checkpoints" / "scaffolded" / "s_mid_v2" / "best.pt"
    if args.auto_resume_best and s_mid_v2_best.exists():
        return s_mid_v2_best, "s_mid_v2_best"
    return None, "scratch"


def _load_model(
    device: torch.device,
    resume_path: Optional[Path],
    policy_hidden: tuple[int, ...] = (512, 256, 128),
) -> tuple[ScaffoldedPolicyNet, int, float, int, str]:
    feat_dim = MOVE_FEAT_DIM_WITH_LOOKAHEAD

    def _fresh():
        return ScaffoldedPolicyNet(
            move_feat_dim=feat_dim,
            value_input_dim=VALUE_INPUT_DIM_WITH_HISTORY,
            policy_hidden=policy_hidden,
        ).to(device), 0, 0.0, DIFF_START, "scratch"

    if resume_path is None or not Path(resume_path).exists():
        return _fresh()

    ckpt   = torch.load(resume_path, map_location=device, weights_only=False)
    cfg    = ckpt.get("model_config", {})

    # If the requested architecture differs from the checkpoint, start fresh.
    ckpt_hidden = tuple(cfg.get("policy_hidden", (512, 256, 128)))
    if ckpt_hidden != policy_hidden:
        print(f"[s_mid_v2] policy_hidden mismatch: ckpt={ckpt_hidden} vs requested={policy_hidden} — starting fresh")
        return _fresh()

    cfg["move_feat_dim"]   = feat_dim
    cfg["value_input_dim"] = VALUE_INPUT_DIM_WITH_HISTORY
    model  = ScaffoldedPolicyNet.from_config(cfg).to(device)
    sd_key = "model" if "model" in ckpt else "state_dict"
    try:
        model.load_state_dict(ckpt[sd_key])
    except RuntimeError:
        pol_state = {k: v for k, v in ckpt[sd_key].items() if k.startswith("policy_mlp")}
        try:
            model.load_state_dict(pol_state, strict=False)
            print("[s_mid_v2] Warning: value_mlp shape mismatch — policy weights loaded, value head reinitialized")
        except RuntimeError:
            print(f"[s_mid_v2] State dict incompatible — starting fresh with policy_hidden={policy_hidden}")
            return _fresh()
    stage      = ckpt.get("stage", "unknown")
    is_mine    = (stage == STAGE_TAG)
    start_game = int(ckpt.get("game_count",    0))         if is_mine else 0
    best_wr    = float(ckpt.get("best_win_rate", 0.0))     if is_mine else 0.0
    difficulty = int(ckpt.get("difficulty",   DIFF_START)) if is_mine else DIFF_START
    return model, start_game, best_wr, difficulty, str(resume_path)


def _apply_diff_start_override(difficulty: int, args: argparse.Namespace) -> int:
    if args.diff_start is not None:
        return max(1, min(args.diff_start, DIFF_MAX))
    return difficulty


def _compute_temperature(game_count: int, max_games: int) -> float:
    progress = min(1.0, game_count / max(max_games * 0.8, 1))
    return float(TEMP_START + (TEMP_MAX - TEMP_START) * progress)


def _adapt_lr(opt: torch.optim.Optimizer, win_rate: float, lr_base: float) -> None:
    scale  = max(LR_SCALE_MIN, min(LR_SCALE_MAX, win_rate / LR_SCALE_WIN))
    new_lr = lr_base * scale
    for g in opt.param_groups:
        g["lr"] = new_lr


def _compute_per_move_reward(
    enc,
    chosen_idx: int,
    enc_after,
    board_phase: str = "move",
    total_pieces: int = 18,
    move_phase_start_ply: Optional[int] = None,
    current_ply: int = 0,
) -> tuple[float, RewardBreakdown]:
    rb = RewardBreakdown()

    in_midgame = total_pieces >= 12 and board_phase != "place"

    if in_midgame:
        if getattr(enc, "sentinel_scores", None):
            mean_s   = float(sum(enc.sentinel_scores) / len(enc.sentinel_scores))
            played_s = float(enc.sentinel_scores[chosen_idx])
            rb.sentinel = ALPHA * (played_s - mean_s)

        if enc_after is not None:
            h_before = float(getattr(enc, "h_before", 0.0))
            h_after  = float(enc.h_scores_abs[chosen_idx]) if getattr(enc, "h_scores_abs", None) else h_before
            rb.heuristic = BETA * math.tanh(h_after - h_before)

    rb.total = rb.sentinel + rb.heuristic
    return float(rb.total), rb


def _retroactive_rescore(trajectory: list[ScaffoldedStep], step_diags: list[StepDiag], outcome: float) -> None:
    n = len(trajectory)
    outcome_positive = 1.0 if outcome == WIN_REWARD else 0.0
    for t_idx, step in enumerate(trajectory):
        plies_remaining  = n - t_idx - 1
        delta            = LAMBDA * outcome * (DECAY ** plies_remaining)
        if outcome_positive > 0.0:
            not_top1 = 1.0 - float(step_diags[t_idx].was_top1_heuristic)
            delta   += EXPLORE_COEF * not_top1
        step.reward     += delta
        step_diags[t_idx].reward.retro += float(delta)
        step_diags[t_idx].reward.total += float(delta)


def _outcome_to_history_float(outcome: float) -> float:
    if outcome == WIN_REWARD:
        return 1.0
    if outcome in (DRAW_SHORT, DRAW_LONG):
        return 0.5
    return 0.0


def _check_advance(win_history_heuristic: deque, rolling_win: int, difficulty: int) -> bool:
    """Advance when (wins + 0.5×draws)/total >= threshold AND draw_rate < 33%.

    Threshold ramps linearly: 51% at level 1 → 60% at level 20.
    Draw-rate cap prevents advancing on stale defensive play.
    Requires ≥20 games in window."""
    recent = list(win_history_heuristic)[-rolling_win:]
    if len(recent) < 20:
        return False
    wins   = sum(1 for x in recent if x == 1.0)
    draws  = sum(1 for x in recent if x == 0.5)
    n      = len(recent)
    draw_rate = draws / n
    if draw_rate >= 0.33:
        return False
    score  = (wins + 0.5 * draws) / n
    threshold = 0.51 + (difficulty - 1) * (0.09 / 19.0)
    return score >= threshold


# ── Frozen-model opponent ─────────────────────────────────────────────────────

class FrozenModelOpponent:
    def __init__(self, model: ScaffoldedPolicyNet, device: torch.device, sentinel=None, value_net=None):
        self._model     = copy.deepcopy(model).to(device)
        self._model.eval()
        self._device    = device
        self._sentinel  = sentinel
        self._value_net = value_net
        self.last_was_blunder = False
        self.last_thinking    = "frozen"

    def refresh(self, model: ScaffoldedPolicyNet) -> None:
        self._model.load_state_dict(copy.deepcopy(model).state_dict())
        self._model.eval()

    def choose_move(self, board: BoardState) -> dict:
        player = board.turn
        enc = encode_position_with_lookahead(board, player,
                                             sentinel_advisor=self._sentinel,
                                             db=None,
                                             value_net=self._value_net,
                                             lookahead_advisor=None)
        if enc is None or not enc.legal_moves:
            return {}
        feat_t = torch.tensor(enc.feat_matrix, dtype=torch.float32).to(self._device)
        with torch.no_grad():
            logits = self._model.policy_logits(feat_t)
            idx    = int(torch.argmax(logits).item())
        return enc.legal_moves[idx]


# ── Single-game rollout ────────────────────────────────────────────────────────

RETRY_PLY_MIN =  5
RETRY_PLY_MAX = 15

@dataclass
class _GameConfig:
    learner_color:          str
    opp_color:              str
    game_type:              str
    game_difficulty:        int
    is_full_diff:           bool
    game_forced_placements: Optional[list[str]]
    retry_ply:              int
    temperature:            float
    start_board:            BoardState


@dataclass
class RolloutResult:
    trajectory:        list[ScaffoldedStep]
    step_diags:        list[StepDiag]
    outcome:           float
    ply:               int
    branch_candidates: list[tuple[int, BoardState, str]]
    retry_board:       Optional[BoardState] = None


def _move_notation(mv: dict) -> str:
    frm = mv.get("from")
    to  = mv.get("to") or ""
    cap = mv.get("capture")
    s = f"{frm}-{to}" if frm else to
    if cap:
        s += f"x{cap}"
    return s


def _rollout(
    model:          ScaffoldedPolicyNet,
    device:         torch.device,
    start_board:    BoardState,
    learner_color:  str,
    opponent,
    opp_color:      str,
    sentinel,
    value_net,
    temperature:    float,
    max_ply:        int,
    record_branches: bool,
    branch_every:   int,
    retry_ply:      int,
    forced_placements: Optional[list[str]] = None,
    lookahead_advisor=None,
    game_difficulty: int = 1,
    human_db=None,
    trajectory_db=None,
    specialist_db=None,
    deep_game: bool = False,
) -> RolloutResult:
    # For deep games (1-in-20): temporarily run full ply_depth simulation
    _saved_sim_ply = None
    if deep_game and lookahead_advisor is not None:
        _saved_sim_ply = lookahead_advisor._sim_ply_depth
        lookahead_advisor._sim_ply_depth = lookahead_advisor._ply_depth

    board                   = start_board
    ply                     = 0
    move_phase_start_ply:   Optional[int] = None
    game_trajectory:        list[ScaffoldedStep] = []
    step_diags:             list[StepDiag]       = []
    branch_candidates:      list[tuple[int, BoardState, str]] = []
    done                    = False
    outcome                 = 0.0
    learner_move_count      = 0
    learner_placement_count = 0
    retry_board: Optional[BoardState] = None
    move_history: deque[dict] = deque(maxlen=N_HISTORY)
    learner_boards: list[BoardState] = []
    learner_moves_notation: list[str] = []

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
            # v4: full-legal-moves scoring via encode_position_with_lookahead.
            learner_boards.append(board)
            enc = encode_position_with_lookahead(
                board, player,
                sentinel_advisor=sentinel,
                db=None,
                value_net=value_net,
                lookahead_advisor=lookahead_advisor,
            )
            if enc is None or not enc.legal_moves:
                outcome = LOSS_REWARD
                done    = True
                break

            feat_t = torch.tensor(enc.feat_matrix, dtype=torch.float32).to(device)
            with torch.no_grad():
                logits    = model.policy_logits(feat_t)
                scaled    = logits / max(temperature, 1e-6)
                log_probs = F.log_softmax(scaled, dim=-1)
                probs     = log_probs.exp()
                if not torch.isfinite(probs).all():
                    probs = torch.where(torch.isfinite(probs), probs, torch.zeros_like(probs))
                probs     = probs / probs.sum().clamp(min=1e-9)
                entropy   = float((-(probs * log_probs).sum()).item())

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
                chosen_prob     = float(probs[chosen_idx].item())
                top1_prob       = float(probs.max().item())
                was_top1_policy = int(chosen_idx == int(torch.argmax(probs).item()))
                log_prob_old    = float(log_probs[chosen_idx].item())

            # History features: snapshot BEFORE appending current move
            hist_feats_now = _build_history_features(move_history)

            move = enc.legal_moves[chosen_idx]
            learner_moves_notation.append(_move_notation(move))
            if board.phase == "place":
                learner_placement_count += 1
            move_history.append(move)   # advance history for next-state context
            hist_feats_next = _build_history_features(move_history)

            board_after = board.apply_move(move)
            enc_after   = encode_position_with_lookahead(board_after, opp_color,
                                                          sentinel_advisor=sentinel, db=None,
                                                          value_net=value_net,
                                                          lookahead_advisor=None)

            total_pieces = board.pieces_on_board.get("W", 0) + board.pieces_on_board.get("B", 0)
            reward, rb = _compute_per_move_reward(
                enc, chosen_idx, enc_after,
                board_phase=board.phase,
                total_pieces=total_pieces,
                move_phase_start_ply=move_phase_start_ply,
                current_ply=ply,
            )

            # Mill formation bonus (un-gated)
            mills_before = sum(1 for m in MILLS if all(board.positions.get(p) == learner_color for p in m))
            mills_after  = sum(1 for m in MILLS if all(board_after.positions.get(p) == learner_color for p in m))
            if mills_after > mills_before:
                mill_bonus = MILL_BONUS * (mills_after - mills_before)
                reward    += mill_bonus
                rb.mill_formed += mill_bonus
                rb.total  += mill_bonus

            raw_now  = _build_raw_board_features(board)
            raw_next = _build_raw_board_features(board_after)
            vi_now = np.concatenate([enc.value_input, hist_feats_now, raw_now])
            if enc_after is not None and enc_after.legal_moves:
                _row_after = enc_after.feat_matrix
                _pad_w = MOVE_FEAT_DIM_WITH_LOOKAHEAD - _row_after.shape[1]
                if _pad_w > 0:
                    _row_after = np.concatenate(
                        [_row_after, np.zeros((_row_after.shape[0], _pad_w), dtype=np.float32)],
                        axis=1,
                    ).astype(np.float32)
                elif _pad_w < 0:
                    _row_after = _row_after[:, :MOVE_FEAT_DIM_WITH_LOOKAHEAD]
                next_mf = _row_after
                next_vi = np.concatenate([enc_after.value_input, hist_feats_next, raw_next])
            else:
                next_mf = np.zeros((1, MOVE_FEAT_DIM_WITH_LOOKAHEAD), dtype=np.float32)
                next_vi = np.zeros(VALUE_INPUT_DIM_WITH_HISTORY, dtype=np.float32)

            terminal_next, _ = is_terminal(board_after)
            step = ScaffoldedStep(
                move_features=enc.feat_matrix,
                value_input=vi_now,
                chosen_idx=chosen_idx,
                log_prob_old=log_prob_old,
                reward=reward,
                next_move_features=next_mf,
                next_value_input=next_vi,
                done=terminal_next,
            )
            game_trajectory.append(step)

            sentinel_scores = list(getattr(enc, "sentinel_scores", []) or [])
            sentinel_mean   = float(sum(sentinel_scores) / len(sentinel_scores)) if sentinel_scores else 0.0
            sentinel_chosen = float(sentinel_scores[chosen_idx]) if sentinel_scores else 0.0
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
                malom_chosen_wdl="n/a",
                malom_chosen_dtm=None,
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
            move_history.append(opp_move)
            board = board.apply_move(opp_move)

        ply += 1

    if not done:
        outcome = DRAW_LONG

    if _saved_sim_ply is not None and lookahead_advisor is not None:
        lookahead_advisor._sim_ply_depth = _saved_sim_ply

    if specialist_db is not None and learner_boards:
        try:
            _res = "W" if outcome == WIN_REWARD else ("D" if outcome in (DRAW_SHORT, DRAW_LONG) else "L")
            specialist_db.record_game(learner_boards, _res, learner_moves_notation, "mid")
        except Exception:
            pass

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
    sd       = result.step_diags
    win_rate = sum(1 for x in win_history if x == 1.0) / max(len(win_history), 1)
    return GameDiag(
        game=game_count,
        difficulty=difficulty,
        learner_color=learner_color,
        temperature=round(temperature, 4),
        outcome=float(result.outcome),
        win_rate_200=round(win_rate, 4),
        ply=int(result.ply),
        steps=len(sd),
        update_policy_loss=None if last_update_pl  is None else float(last_update_pl),
        update_value_loss =None if last_update_vl  is None else float(last_update_vl),
        update_entropy    =None if last_update_ent is None else float(last_update_ent),
        reward_total_mean    =_safe_mean([d.reward.total      for d in sd]),
        reward_sentinel_mean =_safe_mean([d.reward.sentinel   for d in sd]),
        reward_heuristic_mean=_safe_mean([d.reward.heuristic  for d in sd]),
        reward_retro_mean    =_safe_mean([d.reward.retro      for d in sd]),
        sentinel_mean        =_safe_mean([d.sentinel_mean     for d in sd]),
        sentinel_chosen_mean =_safe_mean([d.sentinel_chosen   for d in sd]),
        h_delta_mean         =_safe_mean([d.h_delta           for d in sd]),
        vn_delta_mean        =_safe_mean([d.vn_delta          for d in sd]),
        chosen_prob_mean     =_safe_mean([d.chosen_prob       for d in sd]),
        entropy_mean         =_safe_mean([d.entropy           for d in sd]),
        top1_prob_mean       =_safe_mean([d.top1_prob         for d in sd]),
        legal_moves_mean     =_safe_mean([float(d.legal_moves) for d in sd]),
        policy_top1_rate     =_safe_mean([float(d.was_top1_policy)    for d in sd]),
        heuristic_top1_rate  =_safe_mean([float(d.was_top1_heuristic) for d in sd]),
        malom_win_move_rate  =0.0,
        malom_unknown_rate   =0.0,
        best_win_rate  =float(best_win_rate),
        temp_frozen    =int(temp_frozen),
        lr             =float(opt.param_groups[0]["lr"]),
        source_checkpoint=source_ckpt,
        game_type      =game_type,
        phase_bucket   =phase_bucket,
        is_branch      =int(is_branch),
        branch_ply_start=branch_ply_start,
        target_age     =target_age,
        bucket_opening =bucket_counts.get("opening",  0),
        bucket_midgame =bucket_counts.get("midgame",  0),
        bucket_endgame =bucket_counts.get("endgame",  0),
    )


# ── Main training loop ────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[s_mid_v2] Device: {device}")
    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ── Load components ────────────────────────────────────────────────────────
    sentinel = None
    sent_path = args.sentinel or str(_ROOT / "learned_ai" / "sentinel" / "checkpoints" / "best.pt")
    if Path(sent_path).exists():
        sentinel = load_advisor(sent_path)
        if sentinel and sentinel.is_loaded():
            print(f"[s_mid_v2] Sentinel loaded: {sent_path}")
        else:
            sentinel = None
    if sentinel is None:
        print("[s_mid_v2] Sentinel unavailable — sentinel reward = 0")

    db = None
    malom_path = args.malom or _load_settings().get("malom_db_path", "")
    if malom_path and Path(malom_path).exists():
        try:
            from learned_ai.sentinel.db_teacher import ExternalSolvedDB
            db = ExternalSolvedDB(malom_path)
            if db.is_available():
                print(f"[s_mid_v2] Malom DB loaded (lookahead termination only): {malom_path}")
            else:
                db = None
        except Exception as e:
            print(f"[s_mid_v2] Malom DB failed ({e})")
    if db is None:
        print("[s_mid_v2] Malom DB unavailable — lookahead uses no endgame early-exit")

    value_net = None
    vn_path = args.value_net or str(_ROOT / "data" / "value_net.npz")
    if vn_path and Path(vn_path).exists():
        try:
            from ai.value_net import ValueNet as _ValueNet
            value_net = _ValueNet.load(vn_path)
            print(f"[s_mid_v2] Value net loaded: {vn_path}")
        except Exception as e:
            print(f"[s_mid_v2] Value net load failed ({e}) — VN features will be 0")
    else:
        print("[s_mid_v2] No value net — VN features will be 0")

    gap_net = None
    gap_path = args.gap_net or str(_ROOT / "data" / "gap_net.npz")
    if gap_path and Path(gap_path).exists():
        try:
            from ai.gap_net import GapNet as _GapNet
            gap_net = _GapNet.load(gap_path)
            print(f"[s_mid_v2] Gap net loaded: {gap_path}")
        except Exception as e:
            print(f"[s_mid_v2] Gap net load failed ({e}) — gap features will be 0.5")
    else:
        print("[s_mid_v2] No gap net — gap features will be 0.5")

    # v3: HumanDB — for per-candidate human-play-frequency feature
    human_db = None
    hdb_path = _ROOT / "data" / "human_db.sqlite"
    if hdb_path.exists():
        try:
            from ai.human_db import HumanDB
            human_db = HumanDB(hdb_path)
            print(f"[s_mid_v2] HumanDB loaded: {human_db.game_count} games "
                  f"({human_db.entry_count} positions)")
        except Exception as e:
            print(f"[s_mid_v2] HumanDB load failed ({e}) — human_freq features will be 0")
    else:
        print("[s_mid_v2] No HumanDB — human_freq features will be 0")

    # ── LookaheadAdvisor ─────────────────────────────────────────────────────
    lookahead_advisor = LookaheadAdvisor(
        sentinel=sentinel,
        evaluate_fn=_simple_evaluate,
        value_net=value_net,
        gap_net=gap_net,
        human_db=human_db,
        use_sentinel=True,
        ply_depth=12,
        sim_ply_depth=args.sim_ply_depth,
        endgame_db=db,
    )
    print(f"[s_mid_v2] LookaheadAdvisor: 12-ply width, {args.sim_ply_depth}-ply sim, 5 signals (h+learner_sent+opp_sent+vn+gap)")

    # ── SpecialistDB ─────────────────────────────────────────────────────────
    specialist_db = SpecialistDB(_ROOT / "data" / "specialist_db.sqlite")
    print(f"[s_mid_v2] SpecialistDB: {specialist_db.stats()}")

    # ── Load model ─────────────────────────────────────────────────────────────
    resume_path, source_tag = _choose_resume_path(args)
    model, start_game, best_win_rate, difficulty, source_checkpoint = _load_model(device, resume_path, args.policy_hidden)
    difficulty = _apply_diff_start_override(difficulty, args)
    if resume_path is None:
        print("[s_mid_v2] No checkpoint found — starting from scratch")
    else:
        print(f"[s_mid_v2] Resuming from ({source_tag}): {resume_path}")
    print(f"[s_mid_v2] feat_dim={MOVE_FEAT_DIM_WITH_LOOKAHEAD}, starting game={start_game}, diff={difficulty}")

    frozen_opp = FrozenModelOpponent(model, device, sentinel=sentinel, value_net=value_net)
    # Option C: lookahead uses same frozen snapshot for learner-side simulated moves.
    lookahead_advisor.set_frozen_model(frozen_opp._model, device=device)
    print("[s_mid_v2] LookaheadAdvisor: frozen-model driven learner-side (Option C)")
    games_since_target_update = 0
    games_at_level            = 0   # for Sanmill time-of-flight target relaxation

    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    opt       = torch.optim.Adam(model.parameters(), lr=args.lr)
    update_fn = scaffolded_ppo_update if args.ppo else scaffolded_a2c_update

    game_count             = start_game
    temperature            = args.temp_start
    win_history:             deque[float] = deque(maxlen=args.rolling_win)
    win_history_heuristic:   deque[float] = deque(maxlen=args.rolling_win)
    ep_steps: list[ScaffoldedStep] = []
    last_update_pl  = None
    last_update_vl  = None
    last_update_ent = None
    best_win_rate_at_diff = 0.0

    branch_bucket_history: deque[str] = deque(maxlen=args.bucket_window)

    log_path        = out_dir / "train_log.jsonl"
    update_log_path = out_dir / "update_log.jsonl"

    # ── Position pool ─────────────────────────────────────────────────────────
    midgame_pool: list = []
    if not args.no_position_pool:
        try:
            from learned_ai.training.position_pool import load_position_pool
            midgame_pool = load_position_pool(
                _ROOT, phase="midgame", movement_turn=10, window=2,
            )
            print(f"[s_mid_v2] Position pool: {len(midgame_pool)} midgame positions")
        except Exception as e:
            print(f"[s_mid_v2] Position pool load failed ({e}) — starting from new_game()")

    print(f"[s_mid_v2] Starting at game {game_count}, difficulty {difficulty}")
    print(f"[s_mid_v2] Self-play ratio {args.self_play_ratio:.0%}, "
          f"branch every {args.branch_every} turns, "
          f"max {args.max_branches_per_game} branches/game")

    # s1a warm-start: run once before RL if starting from game 0
    if not args.no_s1a_warmstart and start_game == 0:
        print(f"[s_mid_v2] Running s1a warm-start (pre-RL imitation) from {args.s1a_data}")
        _run_s1b_refresher(model, device, args.s1a_data,
                           epochs=args.s1b_refresher_epochs,
                           lr=args.s1b_refresher_lr)

    # Warm the lazy-init heuristic eval global before spawning threads
    if args.batch_games > 1:
        encode_position_with_lookahead(BoardState.new_game(), "W",
                                       sentinel_advisor=None, db=None,
                                       value_net=None, lookahead_advisor=None)
        print(f"[s_mid_v2] Encoder warmed for {args.batch_games}-game parallel batches")

    diag_buffer: list[GameDiag] = []
    _executor = ThreadPoolExecutor(max_workers=args.batch_games) if args.batch_games > 1 else None

    while game_count < args.max_games:
        temperature = _compute_temperature(game_count, args.max_games)

        if games_since_target_update >= args.update_target_every:
            frozen_opp.refresh(model)
            games_since_target_update = 0
            print(f"[s_mid_v2] Frozen model updated at game {game_count}")

        # ── Build N game configs ──────────────────────────────────────────────
        batch_slots: list[tuple[_GameConfig, Any]] = []
        for _ in range(max(1, min(args.batch_games, args.max_games - game_count))):
            if not midgame_pool:
                raise RuntimeError("[s_mid_v2] midgame position pool is empty — refusing to train from placement. "
                                   "Remove --no-position-pool or ensure the pool loader finds positions.")
            _sb = midgame_pool[rng.randint(0, len(midgame_pool) - 1)]
            _lc = _sb.turn
            _oc = "B" if _lc == "W" else "W"
            if rng.random() < args.self_play_ratio:
                _opp, _gt, _gd = frozen_opp, "vs_frozen", difficulty
            else:
                _gd = difficulty
                if difficulty > 1 and rng.random() < 0.15:
                    _gd = rng.randint(1, difficulty - 1)
                _tb = _heuristic_time_budget(_gd) if args.time_budget <= 0 else args.time_budget
                _h  = HeuristicAgent(color=_oc, difficulty=_gd, game_ai=None)
                _h._inner = _GA(color=_oc, difficulty=_gd, override_time_budget=_tb)
                _opp, _gt = _h, "vs_heuristic"
            _fp: Optional[list[str]] = None
            if _OPENING_LINES and rng.random() < BOOK_GAME_PROB:
                _ln = _OPENING_LINES[rng.randint(0, len(_OPENING_LINES) - 1)]
                _fp = _sample_forced_placements(_ln, _lc)
            batch_slots.append((
                _GameConfig(
                    learner_color=_lc, opp_color=_oc, game_type=_gt,
                    game_difficulty=_gd,
                    is_full_diff=(_gt == "vs_heuristic" and _gd == difficulty),
                    game_forced_placements=_fp,
                    retry_ply=rng.randint(RETRY_PLY_MIN, RETRY_PLY_MAX),
                    temperature=temperature,
                    start_board=_sb,
                ),
                _opp,
            ))

        # ── Run primary rollouts (parallel when batch_games > 1) ─────────────
        _is_deep_game = (game_count % 20 == 0)   # 1-in-20 games use full 12-ply sim

        def _primary(cfg: _GameConfig, opp: Any) -> RolloutResult:
            return _rollout(
                model=model, device=device, start_board=cfg.start_board,
                learner_color=cfg.learner_color, opponent=opp, opp_color=cfg.opp_color,
                sentinel=sentinel, value_net=value_net, temperature=cfg.temperature,
                max_ply=args.max_ply, record_branches=(args.max_branches_per_game > 0),
                branch_every=args.branch_every, retry_ply=cfg.retry_ply,
                forced_placements=cfg.game_forced_placements,
                lookahead_advisor=lookahead_advisor,
                game_difficulty=cfg.game_difficulty,
                human_db=human_db,
                specialist_db=specialist_db,
                deep_game=_is_deep_game,
            )

        if _executor is not None and len(batch_slots) > 1:
            _futs = {_executor.submit(_primary, cfg, opp): (cfg, opp) for cfg, opp in batch_slots}
            batch_results = [(cfg_opp[0], cfg_opp[1], f.result()) for f, cfg_opp in _futs.items()]
        else:
            batch_results = [(cfg, opp, _primary(cfg, opp)) for cfg, opp in batch_slots]

        # ── Process each result sequentially ──────────────────────────────────
        _advance_done = False
        for cfg, opponent, result in batch_results:
            learner_color          = cfg.learner_color
            opp_color              = cfg.opp_color
            game_type              = cfg.game_type
            game_difficulty        = cfg.game_difficulty
            is_full_diff           = cfg.is_full_diff
            game_forced_placements = cfg.game_forced_placements
            game_retry_ply         = cfg.retry_ply

            if result.trajectory:
                _retroactive_rescore(result.trajectory, result.step_diags, result.outcome)

            if result.outcome == WIN_REWARD:
                ep_steps.extend(result.trajectory)
            elif (not args.minimal_rollouts
                  and result.outcome in (LOSS_REWARD, DRAW_SHORT)
                  and result.retry_board is not None):
                confirm_result = _rollout(
                    model=model,
                    device=device,
                    start_board=result.retry_board,
                    learner_color=learner_color,
                    opponent=opponent,
                    opp_color=opp_color,
                    sentinel=sentinel,
                    value_net=value_net,
                    temperature=temperature,
                    max_ply=args.max_ply,
                    record_branches=False,
                    branch_every=0,
                    retry_ply=0,
                    lookahead_advisor=lookahead_advisor,
                    game_difficulty=game_difficulty,
                    human_db=human_db,
                    specialist_db=specialist_db,
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
                games_at_level += 1
                games_since_target_update += 1
                _hv = _outcome_to_history_float(confirm_result.outcome)
                win_history.append(_hv)
                if is_full_diff:
                    win_history_heuristic.append(_hv)
                _coc = "W" if confirm_result.outcome == WIN_REWARD else ("L" if confirm_result.outcome == LOSS_REWARD else "D")
                if game_count % 10 == 0:
                    print(f"[s_mid_v2] {game_count:6d}  r{game_retry_ply:2d} {learner_color} |          | {_coc} ply={confirm_result.ply:3d} | (from ply {game_retry_ply}) {'[learn]' if confirmed else '[skip]'}")

            _hv = _outcome_to_history_float(result.outcome)
            win_history.append(_hv)
            if is_full_diff:
                win_history_heuristic.append(_hv)
            game_count += 1
            games_at_level += 1
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

            if game_count % 10 == 0:
                recent_h = list(win_history_heuristic)
                hwr = sum(1 for x in recent_h if x == 1.0) / max(len(recent_h), 1)
                hdr = sum(1 for x in recent_h if x == 0.5) / max(len(recent_h), 1)
                _awr = sum(1 for x in win_history if x == 1.0) / max(len(win_history), 1)
                _oc  = "W" if result.outcome == WIN_REWARD else ("L" if result.outcome == LOSS_REWARD else "D")
                _gt  = "heur" if game_type == "vs_heuristic" else "self"
                _dif = f"d{game_difficulty}" if game_difficulty != difficulty else f"diff {difficulty}"
                print(f"[s_mid_v2] {game_count:6d} {_gt:4s} {learner_color} | {_dif} | {_oc} ply={result.ply:3d} | hwr={hwr:.3f} hdr={hdr:.3f} awr={_awr:.3f} | temp={temperature:.2f} lr={opt.param_groups[0]['lr']:.5f}")

            if (not args.minimal_rollouts
                and result.outcome != WIN_REWARD
                and result.retry_board is not None):
                retry_result = _rollout(
                    model=model,
                    device=device,
                    start_board=result.retry_board,
                    learner_color=learner_color,
                    opponent=opponent,
                    opp_color=opp_color,
                    sentinel=sentinel,
                    value_net=value_net,
                    temperature=temperature,
                    max_ply=args.max_ply,
                    record_branches=False,
                    branch_every=0,
                    retry_ply=0,
                    lookahead_advisor=lookahead_advisor,
                    game_difficulty=game_difficulty,
                    human_db=human_db,
                    specialist_db=specialist_db,
                )
                if retry_result.trajectory:
                    _retroactive_rescore(retry_result.trajectory, retry_result.step_diags, retry_result.outcome)
                    if retry_result.outcome in (WIN_REWARD, DRAW_SHORT):
                        ep_steps.extend(retry_result.trajectory)
                _rv = _outcome_to_history_float(retry_result.outcome)
                win_history.append(_rv)
                if is_full_diff:
                    win_history_heuristic.append(_rv)
                game_count += 1
                games_at_level += 1
                games_since_target_update += 1
                _roc = "W" if retry_result.outcome == WIN_REWARD else ("L" if retry_result.outcome == LOSS_REWARD else "D")
                if game_count % 10 == 0:
                    print(f"[s_mid_v2] {game_count:6d} retry {learner_color} |          | {_roc} ply={retry_result.ply:3d} | (from ply {game_retry_ply})")

            # ── Branch games ───────────────────────────────────────────────────
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

                branch_result = _rollout(
                    model=model,
                    device=device,
                    start_board=branch_board,
                    learner_color=learner_color,
                    opponent=frozen_opp,
                    opp_color=opp_color,
                    sentinel=sentinel,
                    value_net=value_net,
                    temperature=temperature,
                    max_ply=args.max_ply_branch,
                    record_branches=False,
                    branch_every=0,
                    retry_ply=0,
                    lookahead_advisor=lookahead_advisor,
                    game_difficulty=game_difficulty,
                    human_db=human_db,
                    specialist_db=specialist_db,
                )

                if branch_result.trajectory:
                    _retroactive_rescore(branch_result.trajectory, branch_result.step_diags, branch_result.outcome)
                    if branch_result.outcome in (WIN_REWARD, DRAW_SHORT):
                        ep_steps.extend(branch_result.trajectory)
                    branch_bucket_history.append(bucket)
                    branches_spawned += 1
                    game_count += 1
                    games_at_level += 1
                    games_since_target_update += 1
                    win_history.append(_outcome_to_history_float(branch_result.outcome))

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
                        print(f"[s_mid_v2] {game_count:6d}  +b  {learner_color} | {bucket:7s} | {_boc} ply={branch_result.ply:3d} | (from ply {branch_ply})")

            # ── Update ─────────────────────────────────────────────────────────
            if len(ep_steps) >= args.update_every:
                last_update_pl, last_update_vl, last_update_ent = update_fn(
                    model, opt, ep_steps, device, gamma=args.gamma_td, entropy_coef=args.entropy_coef
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

            # ── Periodic log + checkpoint ──────────────────────────────────────
            if game_count % args.log_every == 0 and diag_buffer:
                recent_h     = list(win_history_heuristic)
                win_rate     = sum(1 for x in recent_h if x == 1.0) / max(len(recent_h), 1)
                draw_rate    = sum(1 for x in recent_h if x == 0.5) / max(len(recent_h), 1)
                win_rate_all = sum(1 for x in win_history  if x == 1.0) / max(len(win_history), 1)

                _adapt_lr(opt, win_rate, args.lr)

                if (len(win_history_heuristic) >= RECOVERY_MIN_GAMES
                        and win_rate < RECOVERY_THRESHOLD):
                    best_ckpt = out_dir / f"best{difficulty}.pt"
                    if best_ckpt.exists():
                        ckpt_r = torch.load(str(best_ckpt), map_location=device, weights_only=False)
                        try:
                            model.load_state_dict(ckpt_r["model"])
                        except RuntimeError:
                            pol_state = {k: v for k, v in ckpt_r["model"].items() if k.startswith("policy_mlp")}
                            model.load_state_dict(pol_state, strict=False)
                            print(f"[s_mid_v2] Recovery: value_mlp shape mismatch — policy weights loaded, value head kept")
                        model.to(device)
                        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
                        frozen_opp.refresh(model)
                        win_history.clear()
                        win_history_heuristic.clear()
                        temperature = TEMP_START
                        print(f"[s_mid_v2] Recovery: reloaded best{difficulty}.pt (win rate was {win_rate:.2f})")

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
                        f"[s_mid_v2] game {game_count:6d} | diff {difficulty} | "
                        f"win={win_rate:.3f} draw={draw_rate:.3f} all={win_rate_all:.3f} | "
                        f"temp={temperature:.2f} | "
                        f"outcome={d.outcome:+.2f} | lr={opt.param_groups[0]['lr']:.5f} | "
                        f"rew={_sign(d.reward_total_mean)} | "
                        f"sent={_sign(d.reward_sentinel_mean)} "
                        f"h={_sign(d.reward_heuristic_mean)} | "
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

                if win_rate > best_win_rate_at_diff and len(win_history_heuristic) >= 10:
                    best_win_rate_at_diff = win_rate
                    ckpt["best_win_rate"] = best_win_rate_at_diff
                    torch.save(ckpt, out_dir / f"best{difficulty}.pt")
                    torch.save(ckpt, out_dir / "best.pt")
                    if win_rate > best_win_rate:
                        best_win_rate = win_rate
                    print(f"[s_mid_v2]  → best diff-{difficulty} win rate: {best_win_rate_at_diff:.3f}")

            # ── Difficulty advancement (Sanmill superiority-probability) ──────
            # Throttle: evaluate the P-value only every 10 games at the current
            # level to limit false-positive advances from variance blips.
            _adv = None
            if games_at_level >= 20 and games_at_level % 10 == 0:
                _adv = _sanmill_check_advance(win_history_heuristic,
                                              difficulty=difficulty,
                                              games_at_level=games_at_level)
                if game_count % 50 == 0:
                    print(f"[s_mid_v2] advance-check @ diff {difficulty}: {_adv.reason}")
            if _adv is not None and _adv.should_advance:
                if difficulty >= args.diff_max:
                    print(f"[s_mid_v2] *** DONE at diff {difficulty}: {_adv.reason} ***")
                    _advance_done = True
                    break
                else:
                    prev_diff = difficulty
                    difficulty += 1
                    win_history.clear()
                    win_history_heuristic.clear()
                    games_at_level = 0
                    print(f"[s_mid_v2] *** Advanced to diff {difficulty} (was diff {prev_diff}: "
                          f"score={_adv.score_pct:.3f} P={_adv.p_super:.3f} target={_adv.target:.3f}) ***")
                    wr = _adv.score_pct

                prev_best = out_dir / f"best{prev_diff}.pt"
                if not prev_best.exists():
                    _adv_ckpt = {
                        "model":             model.state_dict(),
                        "model_config":      model.get_config(),
                        "stage":             STAGE_TAG,
                        "game_count":        game_count,
                        "best_win_rate":     wr,
                        "difficulty":        prev_diff,
                        "source_checkpoint": source_checkpoint,
                        "lr":                float(opt.param_groups[0]["lr"]),
                        "temperature":       float(temperature),
                    }
                    torch.save(_adv_ckpt, prev_best)
                    print(f"[s_mid_v2] Saved best{prev_diff}.pt at advancement (wr={wr:.3f})")
                if prev_best.exists():
                    ckpt_prev = torch.load(str(prev_best), map_location=device, weights_only=False)
                    try:
                        model.load_state_dict(ckpt_prev["model"])
                    except RuntimeError:
                        pol_state = {k: v for k, v in ckpt_prev["model"].items() if k.startswith("policy_mlp")}
                        model.load_state_dict(pol_state, strict=False)
                        print(f"[s_mid_v2] Advance-load: value_mlp shape mismatch — policy weights loaded, value head kept")
                    model.to(device)
                    print(f"[s_mid_v2] Loaded best{prev_diff}.pt as starting point for diff {difficulty}")

                best_win_rate_at_diff = 0.0
                opt = torch.optim.Adam(model.parameters(), lr=args.lr)
                frozen_opp.refresh(model)

        if _advance_done:
            break

    # ── Final flush ────────────────────────────────────────────────────────────
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
    print(f"\n[s_mid_v2] Done. Games: {game_count}  Best win rate: {best_win_rate:.3f}")
    print(f"[s_mid_v2] Checkpoint: {out_dir / 'best.pt'}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Midgame specialist v2: 15-ply lookahead + gap net")
    p.add_argument("--resume",             default="",   type=str)
    p.add_argument("--auto-resume-best",   action="store_true")
    p.add_argument("--out-dir",  default=str(_ROOT / "learned_ai" / "checkpoints" / "scaffolded" / "s_mid_v2"))
    p.add_argument("--sentinel", default=str(_ROOT / "learned_ai" / "sentinel" / "checkpoints" / "best.pt"))
    p.add_argument("--malom",    default="", type=str)
    p.add_argument("--value-net",default=str(_ROOT / "data" / "value_net.npz"), type=str)
    p.add_argument("--gap-net",  default=str(_ROOT / "data" / "gap_net.npz"),   type=str)
    p.add_argument("--ppo",      action="store_true")
    p.add_argument("--max-games",           type=int,   default=5000)
    p.add_argument("--seed",                type=int,   default=42)
    p.add_argument("--lr",                  type=float, default=LR)
    p.add_argument("--gamma-td",            type=float, default=GAMMA_TD)
    p.add_argument("--entropy-coef",        type=float, default=ENTROPY_COEF)
    p.add_argument("--update-every",        type=int,   default=UPDATE_EVERY)
    p.add_argument("--rolling-win",         type=int,   default=ROLLING_WIN)
    p.add_argument("--diff-start",          type=int,   default=None)
    p.add_argument("--diff-max",            type=int,   default=DIFF_MAX)
    p.add_argument("--temp-start",          type=float, default=TEMP_START)
    p.add_argument("--log-every",           type=int,   default=LOG_EVERY)
    p.add_argument("--max-ply",             type=int,   default=MAX_PLY)
    p.add_argument("--max-ply-branch",      type=int,   default=MAX_PLY_BRANCH)
    p.add_argument("--time-budget",         type=float, default=-1.0)
    p.add_argument("--self-play-ratio",     type=float, default=SELF_PLAY_RATIO)
    p.add_argument("--update-target-every", type=int,   default=UPDATE_TARGET_EVERY)
    p.add_argument("--branch-every",        type=int,   default=BRANCH_EVERY)
    p.add_argument("--max-branches-per-game", type=int, default=0)
    p.add_argument("--bucket-window",       type=int,   default=BUCKET_WINDOW)
    p.add_argument("--max-per-bucket",      type=int,   default=MAX_PER_BUCKET)
    p.add_argument("--s1b-data",             type=str,  default=str(_ROOT / "learned_ai" / "data" / "human_imitation.npz"))
    p.add_argument("--s1b-refresher-epochs", type=int,  default=S1B_REFRESHER_EPOCHS)
    p.add_argument("--s1b-refresher-lr",     type=float,default=S1B_REFRESHER_LR)
    p.add_argument("--no-s1b-refresher",     action="store_true")
    p.add_argument("--no-position-pool",     action="store_true")
    p.add_argument("--s1a-data",             type=str,  default=str(_ROOT / "learned_ai" / "data" / "human_imitation2.npz"))
    p.add_argument("--no-s1a-warmstart",     action="store_true")
    p.add_argument("--minimal-rollouts",    action="store_true",
                   help="Skip retry + confirm rollouts (branches are already off by default). "
                        "Trades sample efficiency for wall-clock speed — one primary rollout per game.")
    p.add_argument("--sim-ply-depth",       type=int,   default=5,
                   help="LookaheadAdvisor simulation depth during training (default 5). "
                        "Feature width stays at 15-ply * 4 = 60 floats via padding, so inference "
                        "at full 15 plies matches. Big training speed-up.")
    p.add_argument("--policy-hidden",       type=str,   default="1024,512,256",
                   help="Comma-separated hidden layer widths for the policy MLP "
                        "(default '1024,512,256'). Checkpoint is reset if this differs from the "
                        "saved architecture.")
    p.add_argument("--batch-games",          type=int,  default=1,
                   help="Number of games to run in parallel per batch (default 1 = sequential)")
    args = p.parse_args()
    args.policy_hidden = tuple(int(x) for x in args.policy_hidden.split(","))
    run(args)


if __name__ == "__main__":
    main()
