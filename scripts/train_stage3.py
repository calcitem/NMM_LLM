"""scripts/train_stage3.py — Stage 3: Curriculum vs Heuristic + Value Net.

New features over Stage 2:
  - Opponent move replay: in lost games, the opponent's Malom-confirmed good
    moves (quality >= 0, i.e. W or D for the opponent) are added to a
    supervised imitation batch.  A small CE loss (IMITATION_WEIGHT=0.1) on
    these transitions teaches the model what winning play looks like from the
    exact positions where it failed.  Only the Malom-exact signal is used —
    heuristic moves that Malom disagrees with are discarded.
  - Malom reward shaping active throughout — no game-count cutoff.  Both
    signals (move quality + trap reward) run for every game.
  - Stronger opponent: difficulty ramps 3→8; vn_blend=80% at diff 6+;
    full time budget (0.3s/move at diff<6, 0.8s/move at diff>=6); fullgame DB
    and endgame solved DB wired in when available.
  - Sentinel blunder filter active from game 1 (no warmup at this stage).

Curriculum:
  diff 3 → diff 8 in steps; each bump requires rolling-200 win rate >= 55%.
  Exit when 55% win rate at diff 8 + vn_blend=80%.

Usage:
    .venv/bin/python scripts/train_stage3.py [--resume CKPT] [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from game.board import BoardState
from game.rules import get_all_legal_moves, is_terminal
import learned_ai.agents.heuristic_agent as _ha_mod
from learned_ai.agents.heuristic_agent import HeuristicAgent
from learned_ai.agents.learned_agent import LearnedAgent
from learned_ai.models.action_encoder import _primary_index, get_legal_mask
from learned_ai.models.backbone import NMMNet
from learned_ai.models.state_encoder import PHASE_NAMES, encode_state_with_phase
from learned_ai.sentinel.infer import SentinelAdvisor
from learned_ai.sentinel.db_teacher import ExternalSolvedDB
from learned_ai.training.replay_buffer import Transition

# ── Defaults ──────────────────────────────────────────────────────────────────

LR              = 5e-5          # lower than Stage 2 — opponent is much stronger
GAMMA           = 0.99
TEMPERATURE     = 0.4           # tighter than Stage 2
ENTROPY_COEF    = 0.01
UPDATE_EVERY    = 16
MIN_BATCH       = 32

WIN_REWARD          = 2.0
IMITATION_WEIGHT    = 0.1       # CE loss weight for opponent imitation examples
SENTINEL_THRESHOLD  = 0.1
ROLLING_WINDOW      = 200
WIN_RATE_TARGET     = 0.55      # lower than Stage 2 — opponent is much stronger
DIFF_START          = 3
DIFF_TARGET         = 8
MAX_PLIES           = 400

# Time budget scales with difficulty
TIME_BUDGET_WEAK    = 0.3       # diff 3–5
TIME_BUDGET_STRONG  = 0.8       # diff 6–8 (vn_blend active)
VN_BLEND_DIFF       = 6         # difficulty at which vn_blend=80% kicks in

DEFAULT_MALOM_DB = (
    "/mnt/windows/NMM_DB/Malom_Standard_Ultra-strong_1.1.0/Std_DD_89adjusted"
)
DEFAULT_VALUE_NET   = str(_ROOT / "data/value_net.npz")
DEFAULT_FULLGAME_DB = "/mnt/windows/NMM_DB/fullgame.bin"
DEFAULT_ENDGAME_DB  = str(_ROOT / "data/endgame")


# ── Database loaders ──────────────────────────────────────────────────────────

def _load_value_net(path: str):
    try:
        from ai.value_net import ValueNet
        vn = ValueNet.load_if_exists(path)
        if vn is not None:
            print(f"Value net loaded: {path}")
        return vn
    except Exception as e:
        print(f"  WARNING: value net not loaded ({e})")
        return None


def _load_fullgame_db(path: str):
    try:
        from ai.fullgame_db import FullGameDB
        p = Path(path)
        if not p.exists():
            return None
        db = FullGameDB(str(p))
        print(f"FullGameDB loaded: {path}")
        return db
    except Exception as e:
        print(f"  WARNING: fullgame DB not loaded ({e})")
        return None


def _load_endgame_solved_db(directory: str):
    try:
        from ai.endgame_solved_db import EndgameSolvedDB
        p = Path(directory)
        if not p.exists():
            return None
        db = EndgameSolvedDB(str(p))
        print(f"EndgameSolvedDB loaded: {directory}")
        return db
    except Exception as e:
        print(f"  WARNING: endgame solved DB not loaded ({e})")
        return None


# ── Opponent factory ──────────────────────────────────────────────────────────

def make_opponent(
    difficulty: int,
    value_net,
    fullgame_db,
    endgame_solved_db,
) -> HeuristicAgent:
    time_budget = TIME_BUDGET_STRONG if difficulty >= VN_BLEND_DIFF else TIME_BUDGET_WEAK

    if difficulty >= VN_BLEND_DIFF and value_net is not None:
        from ai.heuristics import HeuristicWeights
        weights = HeuristicWeights(value_net_blend=80)
    else:
        weights = None

    inner = _ha_mod.GameAI(
        color="B",
        difficulty=difficulty,
        value_net=value_net if difficulty >= VN_BLEND_DIFF else None,
        fullgame_db=fullgame_db,
        endgame_solved_db=endgame_solved_db,
        weights=weights,
        override_time_budget=time_budget,
    )
    return HeuristicAgent(color="B", difficulty=difficulty, game_ai=inner)


# ── Episode runner ─────────────────────────────────────────────────────────────

def run_episode(
    model: NMMNet,
    learner_color: str,
    opponent: HeuristicAgent,
    sentinel: Optional[SentinelAdvisor],
    malom_db: Optional[ExternalSolvedDB],
    device: torch.device,
    temperature: float = TEMPERATURE,
    sentinel_threshold: float = SENTINEL_THRESHOLD,
    win_reward: float = WIN_REWARD,
    malom_weight: float = 0.3,
    gamma: float = GAMMA,
    max_plies: int = MAX_PLIES,
) -> tuple[Optional[str], list[Transition], list[tuple], int, int]:
    """Play one game.

    Returns (winner, learner_transitions, opp_imitation_steps, n_kept, n_filtered).
    opp_imitation_steps: list of (state, phase_id, primary_idx, legal_mask) for
    Malom-confirmed good opponent moves — used for CE imitation loss on losses.
    """
    use_sentinel = sentinel is not None and sentinel.is_loaded()
    use_malom    = malom_db is not None and malom_db.is_available()
    opp_color    = "B" if learner_color == "W" else "W"

    learner = LearnedAgent(
        color=learner_color, model=model, device=str(device),
        mode="sample", temperature=temperature,
    )

    board = BoardState.new_game()
    # Learner steps: (state, phase_id, primary_idx, legal_mask, keep, malom_bonus)
    steps: list[tuple] = []
    # Opponent imitation steps: (state, phase_id, primary_idx, legal_mask)
    opp_steps: list[tuple] = []
    n_filtered = 0
    opp_moves = 0
    learner_just_moved = False
    plies = 0

    while plies < max_plies:
        terminal, winner = is_terminal(board)
        if terminal:
            break
        legal = get_all_legal_moves(board)
        if not legal:
            winner = opp_color if board.turn == learner_color else learner_color
            break

        if board.turn == learner_color:
            move = learner.choose_move(board)
            if not move:
                winner = opp_color
                break
            d = learner.last_decision

            # Sentinel blunder filter
            keep = True
            if use_sentinel:
                try:
                    adv = sentinel.advise(board, [move], board.turn, played_move_idx=0)
                    if adv is not None and adv.played_move_quality < sentinel_threshold:
                        keep = False
                        n_filtered += 1
                except Exception:
                    pass

            # Malom signal 1: move quality
            malom_bonus = 0.0
            if use_malom:
                try:
                    q = malom_db.query_move_quality(board, move)
                    if q is not None:
                        malom_bonus = malom_weight * float(q)
                except Exception:
                    pass

            steps.append((d.state, d.phase_id, d.primary_index, d.legal_mask,
                          keep, malom_bonus))
            learner_just_moved = True

        else:
            # Capture opponent state+legal mask BEFORE they play (for imitation)
            if use_malom:
                try:
                    opp_state, opp_phase = encode_state_with_phase(board)
                    opp_mask = get_legal_mask(board)
                except Exception:
                    opp_state = None
            else:
                opp_state = None

            if opp_moves == 0:
                move = random.choice(legal)   # random first move for variety
            else:
                move = opponent.choose_move(board)
            opp_moves += 1
            if not move:
                winner = learner_color
                break

            # Opponent imitation: check if Malom confirms this was a good move
            if use_malom and opp_state is not None:
                try:
                    q = malom_db.query_move_quality(board, move)
                    if q is not None and float(q) >= 0.0:   # W or D for opponent
                        pidx = _primary_index(move)
                        opp_steps.append((opp_state, opp_phase, pidx, opp_mask))
                except Exception:
                    pass

            learner_just_moved = False

        board = board.apply_move(move)
        plies += 1

        # Malom signal 2: trap reward (after learner's move, opponent now to move)
        if learner_just_moved and use_malom and steps:
            try:
                q_trap = malom_db.query(board)
                if q_trap == "L":
                    s = steps[-1]
                    steps[-1] = (*s[:-1], s[-1] + malom_weight)
            except Exception:
                pass

    else:
        winner = None  # ply cap → draw

    # Only use opponent imitation steps from LOST games
    if winner != learner_color:
        imitation = opp_steps
    else:
        imitation = []

    # Assign discounted terminal reward + Malom shaped reward
    n = len(steps)
    transitions: list[Transition] = []
    for i, (state, phase_id, primary_idx, legal_mask, keep, malom_bonus) in enumerate(steps):
        if not keep:
            continue
        dist = n - 1 - i
        if winner is None:
            r_term = 0.0
        elif winner == learner_color:
            r_term = win_reward * (gamma ** dist)
        else:
            r_term = -win_reward * (gamma ** dist)
        transitions.append(Transition(
            state=state,
            legal_mask=legal_mask,
            primary_index=primary_idx,
            capture_index=None,
            reward=r_term + malom_bonus,
            phase_id=phase_id,
            side_to_move=learner_color,
            done=(i == n - 1),
        ))

    return winner, transitions, imitation, len(transitions), n_filtered


# ── Update ────────────────────────────────────────────────────────────────────

def update(
    model: NMMNet,
    optimizer: torch.optim.Optimizer,
    transitions: list[Transition],
    imitation_steps: list[tuple],
    device: torch.device,
) -> tuple[float, float, float]:
    """REINFORCE + imitation CE. Returns (policy_loss, value_loss, imitation_loss)."""
    if len(transitions) < MIN_BATCH:
        return 0.0, 0.0, 0.0

    states        = torch.stack([t.state for t in transitions]).to(device)
    legal_masks   = torch.stack([t.legal_mask for t in transitions]).to(device)
    primary_indices = torch.tensor(
        [t.primary_index for t in transitions], device=device, dtype=torch.long)
    rewards       = torch.tensor(
        [t.reward for t in transitions], device=device, dtype=torch.float32)
    phase_ids     = [t.phase_id for t in transitions]

    model.train()
    feats = model.backbone(states)

    # Value-head baseline
    values = model.value_head(feats).squeeze(-1)
    advantages = rewards - values.detach()
    if advantages.std() > 1e-3:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Policy loss (REINFORCE)
    pl_sum  = torch.zeros([], device=device)
    ent_sum = torch.zeros([], device=device)
    n_total = 0

    for ph in range(model.num_phases):
        idx = [i for i, p in enumerate(phase_ids) if p == ph]
        if not idx:
            continue
        idx_t    = torch.tensor(idx, device=device)
        logits_p = model.phase_heads[PHASE_NAMES[ph]](feats[idx_t])
        logits_p = logits_p.masked_fill(~legal_masks[idx_t], -1e9)
        log_probs_p = F.log_softmax(logits_p, dim=-1)
        sel_lp   = log_probs_p.gather(1, primary_indices[idx_t].unsqueeze(1)).squeeze(1)
        pl_sum   = pl_sum - (sel_lp * advantages[idx_t]).sum()
        probs_p  = log_probs_p.exp()
        ent_sum  = ent_sum + (-(probs_p * log_probs_p).sum(dim=-1)).sum()
        n_total += len(idx)

    policy_loss  = pl_sum  / max(n_total, 1)
    entropy_loss = ent_sum / max(n_total, 1)
    value_loss   = F.mse_loss(values, rewards)

    # Imitation CE loss (opponent good moves from lost games)
    imitation_loss = torch.zeros([], device=device)
    n_imitation = 0
    if imitation_steps:
        # Group by phase
        by_phase: dict[int, list] = {}
        for state, phase_id, pidx, mask in imitation_steps:
            by_phase.setdefault(phase_id, []).append((state, pidx, mask))

        for ph, samples in by_phase.items():
            if ph >= model.num_phases:
                continue
            im_states  = torch.stack([s for s, _, _ in samples]).to(device)
            im_targets = torch.tensor([p for _, p, _ in samples], device=device, dtype=torch.long)
            im_masks   = torch.stack([m for _, _, m in samples]).to(device)
            im_feats   = model.backbone(im_states)
            im_logits  = model.phase_heads[PHASE_NAMES[ph]](im_feats)
            im_logits  = im_logits.masked_fill(~im_masks, -1e9)
            imitation_loss = imitation_loss + F.cross_entropy(im_logits, im_targets) * len(samples)
            n_imitation   += len(samples)

        if n_imitation > 0:
            imitation_loss = imitation_loss / n_imitation

    loss = (policy_loss
            - ENTROPY_COEF * entropy_loss
            + 0.5 * value_loss
            + IMITATION_WEIGHT * imitation_loss)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return float(policy_loss.item()), float(value_loss.item()), float(imitation_loss.item())


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    pa = argparse.ArgumentParser(description="Stage 3: Curriculum vs Heuristic + Value Net")
    pa.add_argument("--resume",        default=str(_ROOT / "learned_ai/checkpoints/stage2/best.pt"))
    pa.add_argument("--out-dir",       default=str(_ROOT / "learned_ai/checkpoints/stage3"))
    pa.add_argument("--sentinel",      default=str(_ROOT / "learned_ai/sentinel/checkpoints/best.pt"))
    pa.add_argument("--malom-db",      default=DEFAULT_MALOM_DB)
    pa.add_argument("--value-net",     default=DEFAULT_VALUE_NET)
    pa.add_argument("--fullgame-db",   default=DEFAULT_FULLGAME_DB)
    pa.add_argument("--endgame-db",    default=DEFAULT_ENDGAME_DB)
    pa.add_argument("--max-games",     type=int,   default=10_000)
    pa.add_argument("--temperature",   type=float, default=TEMPERATURE)
    pa.add_argument("--win-reward",    type=float, default=WIN_REWARD)
    pa.add_argument("--malom-weight",  type=float, default=0.3)
    pa.add_argument("--imitation-weight", type=float, default=IMITATION_WEIGHT)
    pa.add_argument("--diff-start",    type=int,   default=DIFF_START)
    pa.add_argument("--no-sentinel",   action="store_true")
    pa.add_argument("--no-malom",      action="store_true")
    pa.add_argument("--no-imitation",  action="store_true")
    args = pa.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = NMMNet()
    resume = Path(args.resume)
    if resume.exists():
        ckpt = torch.load(str(resume), map_location="cpu", weights_only=False)
        sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(sd)
        print(f"Resumed from {resume}")
    else:
        print(f"WARNING: no checkpoint at {resume} — using random weights")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # ── Databases ─────────────────────────────────────────────────────────────
    sentinel: Optional[SentinelAdvisor] = None
    if not args.no_sentinel:
        sp = Path(args.sentinel)
        if sp.exists():
            sentinel = SentinelAdvisor(checkpoint_path=str(sp), device="cpu")
            print(f"Sentinel loaded: {sp}  (threshold={SENTINEL_THRESHOLD})")

    malom_db: Optional[ExternalSolvedDB] = None
    if not args.no_malom:
        malom_db = ExternalSolvedDB(db_path=args.malom_db)
        if malom_db.is_available():
            print(f"Malom DB loaded: {args.malom_db}")
        else:
            print(f"Malom DB unavailable — no reward shaping or imitation filtering")

    value_net   = _load_value_net(args.value_net)
    fullgame_db = _load_fullgame_db(args.fullgame_db)
    endgame_db  = _load_endgame_solved_db(args.endgame_db)

    # ── Curriculum ────────────────────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    current_diff = args.diff_start
    opponent = make_opponent(current_diff, value_net, fullgame_db, endgame_db)

    results: deque[str] = deque(maxlen=ROLLING_WINDOW)
    accumulated: list[Transition] = []
    accumulated_imitation: list[tuple] = []
    best_win_rate = 0.0
    total_filtered = 0
    total_kept = 0

    print(f"\nStage 3  lr={LR}  γ={GAMMA}  T={args.temperature}"
          f"  win_reward={args.win_reward}  imitation_weight={args.imitation_weight}")
    print(f"  sentinel_thresh={SENTINEL_THRESHOLD}"
          f"  malom_weight={args.malom_weight}  (active throughout)")
    print(f"  update_every={UPDATE_EVERY}  min_batch={MIN_BATCH}")
    print(f"  diff {current_diff}→{DIFF_TARGET}  bump@{WIN_RATE_TARGET:.0%}"
          f"  exit@{WIN_RATE_TARGET:.0%}@diff{DIFF_TARGET}\n")

    t0 = time.time()
    for game in range(args.max_games):
        learner_color = "W" if game % 2 == 0 else "B"
        opponent.color = "B" if learner_color == "W" else "W"
        opponent._inner.color = opponent.color

        winner, transitions, imitation, n_kept, n_filt = run_episode(
            model, learner_color, opponent, sentinel, malom_db, device,
            temperature=args.temperature,
            sentinel_threshold=SENTINEL_THRESHOLD,
            win_reward=args.win_reward,
            malom_weight=args.malom_weight,
        )

        total_kept     += n_kept
        total_filtered += n_filt
        accumulated.extend(transitions)
        if not args.no_imitation:
            accumulated_imitation.extend(imitation)

        r_str = "D" if winner is None else ("W" if winner == learner_color else "L")
        results.append(r_str)

        n_res    = len(results)
        win_rate = results.count("W") / n_res
        filt_pct = total_filtered / max(total_kept + total_filtered, 1)
        elapsed  = time.time() - t0
        vn_tag   = "+vn80" if current_diff >= VN_BLEND_DIFF else ""

        print(f"  game {game+1:5d}  {r_str}  diff={current_diff}{vn_tag}"
              f"  wr={win_rate:.1%} ({n_res:3d})"
              f"  filt={filt_pct:.0%}"
              f"  im={len(imitation):2d}"
              f"  trans={n_kept:3d}"
              f"  t={elapsed:.0f}s")

        # ── Update ─────────────────────────────────────────────────────────
        if (game + 1) % UPDATE_EVERY == 0 and accumulated:
            batch_size = len(accumulated)
            im_size    = len(accumulated_imitation)
            pl, vl, il = update(model, optimizer, accumulated,
                                accumulated_imitation, device)
            accumulated.clear()
            accumulated_imitation.clear()
            if pl != 0.0:
                print(f"    → update  policy={pl:.4f}  value={vl:.4f}"
                      f"  imitation={il:.4f}  batch={batch_size}  im={im_size}")

        # ── Best checkpoint ────────────────────────────────────────────────
        if n_res >= 50 and win_rate > best_win_rate:
            best_win_rate = win_rate
            torch.save({"model": model.state_dict()}, out_dir / "best.pt")

        # ── Curriculum bump ────────────────────────────────────────────────
        if (current_diff < DIFF_TARGET
                and n_res >= ROLLING_WINDOW
                and win_rate >= WIN_RATE_TARGET):
            prev_diff = current_diff
            current_diff += 1
            opponent = make_opponent(current_diff, value_net, fullgame_db, endgame_db)
            results.clear()
            total_filtered = 0
            total_kept = 0
            vn_note = " + vn_blend=80%" if current_diff >= VN_BLEND_DIFF else ""
            print(f"\n  ★ difficulty {prev_diff}→{current_diff}{vn_note}"
                  f"  (win_rate was {win_rate:.1%})\n")

        # ── Exit criterion ─────────────────────────────────────────────────
        if (current_diff >= DIFF_TARGET
                and n_res >= ROLLING_WINDOW
                and win_rate >= WIN_RATE_TARGET):
            print(f"\n  ★ EXIT: {win_rate:.1%} win rate at diff {current_diff} (game {game+1})")
            break

    torch.save({"model": model.state_dict()}, out_dir / "latest.pt")
    n_res    = len(results)
    win_rate = results.count("W") / n_res if n_res else 0.0
    print(f"\nStage 3 done.  win_rate={win_rate:.1%}  diff={current_diff}"
          f"  best={best_win_rate:.1%}")
    print(f"Checkpoints → {out_dir}")


if __name__ == "__main__":
    main()
