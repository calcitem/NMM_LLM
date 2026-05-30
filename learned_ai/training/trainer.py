"""REINFORCE-with-baseline trainer (PPO option toggleable via config).

Each episode:
    1. Pick the opponent for the current curriculum stage.
    2. Play one game with the current LearnedAgent on a random side.
    3. Convert the trajectory into Transitions with returns.
    4. Push to the replay buffer.
    5. When ``episodes_per_batch`` games are accumulated, run a policy
       gradient update over the batch.

Logging is plain JSON-Lines so we don't need tensorboard at runtime.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from learned_ai.agents.heuristic_agent import HeuristicAgent
from learned_ai.agents.learned_agent import LearnedAgent
from learned_ai.agents.random_agent import RandomAgent
from learned_ai.data.game_logger import GameLogger
from learned_ai.models.action_encoder import (
    ACTION_DIM,
    CAPTURE_OFFSET,
    PLACE_OFFSET,
    move_requires_capture,
)
from learned_ai.models.backbone import NMMNet, NEG_INF
from learned_ai.models.state_encoder import PHASE_NAMES
from learned_ai.training.curriculum import Curriculum
from learned_ai.training.replay_buffer import ReplayBuffer, Transition
from learned_ai.training.self_play import assign_rewards, play_game


@dataclass
class TrainerStats:
    episodes: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    illegal_attempts: int = 0
    total_plies: int = 0
    phase_move_counts: Dict[str, int] = field(default_factory=dict)
    white_wins: int = 0
    black_wins: int = 0


class Trainer:
    """Drives self-play episodes and runs policy-gradient updates."""

    def __init__(self, config: dict, resume_path: Optional[str] = None) -> None:
        self.config = config
        train_cfg = config.get("training", {})
        self.algorithm = str(train_cfg.get("algorithm", "reinforce")).lower()
        self.lr = float(train_cfg.get("lr", 3e-4))
        self.gamma = float(train_cfg.get("gamma", 0.99))
        self.episodes_per_batch = int(train_cfg.get("episodes_per_batch", 32))
        self.max_episodes = int(train_cfg.get("max_episodes", 50_000))
        self.checkpoint_every = int(train_cfg.get("checkpoint_every", 1000))
        self.eval_every = int(train_cfg.get("eval_every", 500))
        self.eval_games = int(train_cfg.get("eval_games", 50))
        self.temperature = float(train_cfg.get("temperature", 1.0))
        self._initial_temperature = self.temperature  # restored at each stage/difficulty boundary
        self.temperature_decay = float(train_cfg.get("temperature_decay", 0.9995))
        self.min_temperature = float(train_cfg.get("min_temperature", 0.1))
        self.value_coef = float(train_cfg.get("value_coef", 0.5))
        self.entropy_coef = float(train_cfg.get("entropy_coef", 0.01))
        self.ppo_clip = float(train_cfg.get("ppo_clip", 0.2))
        self.ppo_epochs = int(train_cfg.get("ppo_epochs", 4))
        self.seed = int(train_cfg.get("seed", 42))
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Auto-select CUDA if available; allow override via config "device" key.
        requested = str(train_cfg.get("device", "auto")).lower()
        if requested == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(requested)

        model_cfg = config.get("model", {})
        self.model = NMMNet(
            backbone_hidden=tuple(model_cfg.get("backbone_hidden", (256, 256, 128))),
            head_hidden=tuple(model_cfg.get("head_hidden", (64,))),
            dropout=float(model_cfg.get("dropout", 0.0)),
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        paths = config.get("paths", {})
        self.checkpoint_dir = Path(paths.get("checkpoint_dir", "learned_ai/checkpoints"))
        self.log_dir = Path(paths.get("log_dir", "learned_ai/logs"))
        self.game_log_dir = Path(paths.get("game_log_dir", "learned_ai/self_play_games"))
        for d in (self.checkpoint_dir, self.log_dir, self.game_log_dir):
            d.mkdir(parents=True, exist_ok=True)

        buf_cfg = config.get("replay_buffer", {})
        self.replay = ReplayBuffer(
            capacity=int(buf_cfg.get("capacity", 50_000)),
            seed=self.seed,
        )
        self.min_fill_before_train = int(buf_cfg.get("min_fill_before_train", 0))

        self.curriculum = Curriculum.from_config(config.get("curriculum", {}))
        self.stats = TrainerStats()
        self.metrics_path = self.log_dir / "metrics.jsonl"
        self.game_logger = GameLogger(str(self.game_log_dir))

        if resume_path:
            self.load_checkpoint(resume_path)

    # ------------------------------------------------------------------

    def _make_learned_agent(self, color: str, sample: bool = True) -> LearnedAgent:
        agent = LearnedAgent(
            color=color,
            model=self.model,
            mode="sample" if sample else "argmax",
            temperature=self.temperature,
            device=str(self.device),
        )
        return agent

    def _make_opponent(self, color: str, kind: str):
        if kind == "random":
            return RandomAgent(color=color, seed=random.randint(0, 1 << 31))
        if kind == "heuristic":
            diff, blunder = self.curriculum.heuristic_params()
            return HeuristicAgent(color=color, difficulty=diff, blunder_probability=blunder)
        # default: a fresh self-play opponent that shares the same model.
        return self._make_learned_agent(color=color, sample=True)

    # ------------------------------------------------------------------

    def play_episode(self) -> Tuple[List[Transition], dict]:
        opp_kind = self.curriculum.opponent_kind()
        # Random side assignment so the model sees both colours.
        learned_color = "W" if random.random() < 0.5 else "B"
        opp_color = "B" if learned_color == "W" else "W"
        learned = self._make_learned_agent(color=learned_color, sample=True)
        opponent = self._make_opponent(color=opp_color, kind=opp_kind)

        if learned_color == "W":
            result = play_game(learned, opponent)
        else:
            result = play_game(opponent, learned)

        all_transitions = assign_rewards(result, gamma=self.gamma)
        learned_transitions = [
            t for t in all_transitions if t.side_to_move == learned_color and t.primary_index >= 0
        ]

        meta = {
            "winner": result.winner,
            "draw_reason": result.draw_reason,
            "plies": result.plies,
            "learned_color": learned_color,
            "opponent": opp_kind,
        }
        self.stats.episodes += 1
        self.stats.total_plies += result.plies
        won = result.winner == learned_color
        if result.winner is None:
            self.stats.draws += 1
        elif won:
            self.stats.wins += 1
            if learned_color == "W":
                self.stats.white_wins += 1
            else:
                self.stats.black_wins += 1
        else:
            self.stats.losses += 1
        self.curriculum.record_outcome(won)

        for tr in learned_transitions:
            name = PHASE_NAMES[tr.phase_id]
            self.stats.phase_move_counts[name] = self.stats.phase_move_counts.get(name, 0) + 1

        self.game_logger.log_game(
            winner=result.winner, moves=result.move_log, meta=meta
        )

        return learned_transitions, meta

    # ------------------------------------------------------------------

    def update(self, batch: List[Transition]) -> Dict[str, float]:
        if not batch:
            return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        dev = self.device
        states = torch.stack([t.state for t in batch]).to(dev)
        masks = torch.stack([t.legal_mask for t in batch]).to(dev)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32).to(dev)
        primary_idx = torch.tensor([t.primary_index for t in batch], dtype=torch.long).to(dev)
        cap_idx = torch.tensor(
            [
                t.capture_index if t.capture_index is not None else -1
                for t in batch
            ],
            dtype=torch.long,
        ).to(dev)
        phases = [t.phase_id for t in batch]

        # Forward pass — group by phase so we hit each head exactly once.
        primary_log_probs = torch.zeros(len(batch), device=dev)
        capture_log_probs = torch.zeros(len(batch), device=dev)
        capture_present = torch.zeros(len(batch), dtype=torch.bool, device=dev)
        values = torch.zeros(len(batch), device=dev)
        entropies = torch.zeros(len(batch), device=dev)

        unique_phases = sorted(set(phases))
        for phase in unique_phases:
            sel = [i for i, p in enumerate(phases) if p == phase]
            if not sel:
                continue
            sub_states = states[sel]
            sub_masks = masks[sel]
            out = self.model.forward(sub_states, phase_id=phase, legal_mask=sub_masks)
            logits = out["logits"]      # (N, action_dim)
            value_pred = out["value"]   # (N,)
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)
                value_pred = value_pred.unsqueeze(0)

            primary_logits = logits[:, PLACE_OFFSET:CAPTURE_OFFSET]
            primary_mask = sub_masks[:, PLACE_OFFSET:CAPTURE_OFFSET]
            primary_logits = primary_logits.masked_fill(~primary_mask, NEG_INF)
            primary_log_softmax = F.log_softmax(primary_logits, dim=-1)
            primary_softmax = primary_log_softmax.exp()

            cap_logits = logits[:, CAPTURE_OFFSET:ACTION_DIM]
            cap_mask = sub_masks[:, CAPTURE_OFFSET:ACTION_DIM]
            cap_logits = cap_logits.masked_fill(~cap_mask, NEG_INF)

            for k, batch_pos in enumerate(sel):
                p_idx = primary_idx[batch_pos].item() - PLACE_OFFSET
                if p_idx < 0 or p_idx >= primary_logits.shape[-1]:
                    continue
                primary_log_probs[batch_pos] = primary_log_softmax[k, p_idx]

                # Entropy over legal primary distribution (ignore -inf).
                lp = primary_log_softmax[k]
                p = primary_softmax[k]
                ent = -(p * torch.where(torch.isfinite(lp), lp, torch.zeros_like(lp))).sum()
                entropies[batch_pos] = ent
                values[batch_pos] = value_pred[k]

                c_full = cap_idx[batch_pos].item()
                if c_full >= 0 and cap_mask[k].any():
                    cap_log_softmax = F.log_softmax(cap_logits[k], dim=-1)
                    c_local = c_full - CAPTURE_OFFSET
                    capture_log_probs[batch_pos] = cap_log_softmax[c_local]
                    capture_present[batch_pos] = True

        baseline = values.detach()
        advantages = rewards - baseline

        log_probs = primary_log_probs + torch.where(
            capture_present, capture_log_probs, torch.zeros_like(capture_log_probs)
        )

        if self.algorithm == "ppo":
            old_log_probs = log_probs.detach()
            ratio = torch.exp(log_probs - old_log_probs)
            unclipped = ratio * advantages
            clipped = torch.clamp(ratio, 1 - self.ppo_clip, 1 + self.ppo_clip) * advantages
            policy_loss = -torch.min(unclipped, clipped).mean()
        else:
            policy_loss = -(log_probs * advantages).mean()

        value_loss = F.mse_loss(values, rewards)
        entropy_term = entropies.mean()
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_term

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
        self.optimizer.step()

        return {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy_term.item()),
            "mean_reward": float(rewards.mean().item()),
        }

    # ------------------------------------------------------------------

    def _log_metrics(self, payload: dict) -> None:
        with self.metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

    def save_checkpoint(self, path: Optional[str] = None) -> str:
        out_path = Path(path) if path else self.checkpoint_dir / f"ckpt-{self.stats.episodes:06d}.pt"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        model_config = dict(self.config.get("model", {}))
        payload = {
            "model": self.model.state_dict(),
            "model_config": model_config,
            "optimizer": self.optimizer.state_dict(),
            "stats": self.stats.__dict__,
            "temperature": self.temperature,
            "config": self.config,
        }
        torch.save(payload, str(out_path))
        latest = self.checkpoint_dir / "latest.pt"
        torch.save(
            {"model": self.model.state_dict(), "model_config": model_config},
            str(latest),
        )
        return str(out_path)

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if isinstance(ckpt, dict) and "model" in ckpt:
            self.model.load_state_dict(ckpt["model"])
            if "optimizer" in ckpt:
                try:
                    self.optimizer.load_state_dict(ckpt["optimizer"])
                except Exception:
                    pass
            if "temperature" in ckpt:
                self.temperature = float(ckpt["temperature"])
        else:
            self.model.load_state_dict(ckpt)
        self.model.to(self.device)

    # ------------------------------------------------------------------

    def train(self, max_episodes: Optional[int] = None, verbose: bool = True) -> None:
        if max_episodes is None:
            max_episodes = self.max_episodes

        batch: List[Transition] = []
        start_time = time.time()
        episode = 0
        last_stage = self.curriculum.state.current_stage
        last_print_ep = 0

        if verbose:
            n_params = sum(p.numel() for p in self.model.parameters())
            dev_str = str(self.device)
            if self.device.type == "cuda":
                dev_str += f" ({torch.cuda.get_device_name(self.device)})"
            print(f"\n{'─'*60}")
            print(f"  NMM Learned AI — Training")
            print(f"{'─'*60}")
            print(f"  Device      : {dev_str}")
            print(f"  Algorithm   : {self.algorithm.upper()}")
            print(f"  Max episodes: {max_episodes:,}")
            print(f"  Batch size  : {self.episodes_per_batch}")
            print(f"  LR          : {self.lr}  γ={self.gamma}")
            print(f"  Temperature : {self.temperature:.3f} → {self.min_temperature} (decay {self.temperature_decay})")
            print(f"  Model params: {n_params:,}")
            print(f"  Checkpoints : every {self.checkpoint_every} episodes → {self.checkpoint_dir}")
            print(f"  Metrics log : {self.metrics_path}")
            print(f"{'─'*60}")
            print(f"  Stage {self.curriculum.state.current_stage}: {self.curriculum.state.stage_name()}")
            print(f"{'─'*60}\n")
            print(f"  {'ep':>7}  {'level':<12}  {'W/L/D':>13}  {'win%(roll)':>11}  {'plies':>5}  "
                  f"{'temp':>5}  {'loss':>8}  {'entropy':>7}  {'eps/s':>5}")
            print(f"  {'─'*7}  {'─'*12}  {'─'*13}  {'─'*11}  {'─'*5}  "
                  f"{'─'*5}  {'─'*8}  {'─'*7}  {'─'*5}")

        while episode < max_episodes and not self.curriculum.finished():
            t0 = time.time()
            transitions, meta = self.play_episode()
            self.replay.extend(transitions)
            batch.extend(transitions)
            episode += 1
            self.curriculum.step()

            if verbose:
                elapsed = time.time() - t0
                print(
                    f"  [game {self.stats.episodes}] plies={meta['plies']} "
                    f"winner={meta['winner'] or 'draw'} "
                    f"opp={meta['opponent']} "
                    f"replay={len(self.replay)} "
                    f"batch={len(batch)}/{self.episodes_per_batch} "
                    f"({elapsed:.1f}s)",
                    flush=True,
                )

            new_stage = self.curriculum.state.current_stage
            event = self.curriculum.state.last_event or ""
            if new_stage != last_stage or event.startswith("difficulty_bump:"):
                self.temperature = self._initial_temperature
            if verbose:
                if new_stage != last_stage:
                    reason_str = ""
                    if "budget" in event:
                        reason_str = " [safety cap]"
                    elif "threshold" in event:
                        wr = event.split(":")[-1]
                        reason_str = f" [win rate {float(wr):.1%}]"
                    print(f"\n  ── Stage {new_stage}: {self.curriculum.state.stage_name()} "
                          f"(episode {self.stats.episodes:,}){reason_str}  [temp reset → {self._initial_temperature}]\n")
                    last_stage = new_stage
                elif event.startswith("difficulty_bump:"):
                    parts = event.split(":")
                    new_label = parts[1]
                    wr = float(parts[2])
                    print(f"\n  ── Level → {new_label}  (win rate {wr:.1%} over last "
                          f"{self.curriculum._eval_window} games, ep {self.stats.episodes:,})"
                          f"  [temp reset → {self._initial_temperature}]\n")

            update_metrics: dict = {}
            if len(batch) >= self.episodes_per_batch and len(self.replay) >= self.min_fill_before_train:
                update_metrics = self.update(batch)
                wall = time.time() - start_time
                self._log_metrics(
                    {
                        "episode": self.stats.episodes,
                        "stage": self.curriculum.state.current_stage,
                        "stage_name": self.curriculum.state.stage_name(),
                        "heuristic_level": self.curriculum.level_label(),
                        "wall_seconds": round(wall, 2),
                        "mean_plies": round(
                            self.stats.total_plies / max(1, self.stats.episodes), 2
                        ),
                        "wins": self.stats.wins,
                        "losses": self.stats.losses,
                        "draws": self.stats.draws,
                        "white_wins": self.stats.white_wins,
                        "black_wins": self.stats.black_wins,
                        "phase_move_counts": dict(self.stats.phase_move_counts),
                        "rolling_win_rate": round(self.curriculum.rolling_win_rate(), 4),
                        "temperature": self.temperature,
                        **update_metrics,
                        "last_meta": meta,
                    }
                )
                batch.clear()

                if verbose:
                    ep = self.stats.episodes
                    total = self.stats.wins + self.stats.losses + self.stats.draws
                    win_pct = 100.0 * self.stats.wins / max(1, total)
                    rolling_wr = self.curriculum.rolling_win_rate()
                    mean_plies = self.stats.total_plies / max(1, ep)
                    wall = time.time() - start_time
                    eps_s = ep / max(0.001, wall)
                    stage = self.curriculum.state.current_stage
                    if stage == 3:
                        stage_label = f"s3/{self.curriculum.level_label()}"
                    else:
                        stage_label = self.curriculum.state.stage_name()[:12]
                    wld = f"{self.stats.wins}/{self.stats.losses}/{self.stats.draws}"
                    loss_val = update_metrics.get("loss", 0.0)
                    entropy  = update_metrics.get("entropy", 0.0)
                    print(f"  {ep:>7,}  {stage_label:<12}  {wld:>13}  {win_pct:>4.1f}%"
                          f"({rolling_wr:>4.1%})  "
                          f"{mean_plies:>5.1f}  {self.temperature:>5.3f}  "
                          f"{loss_val:>8.4f}  {entropy:>7.4f}  {eps_s:>5.1f}")
                    last_print_ep = ep

            self.temperature = max(self.min_temperature, self.temperature * self.temperature_decay)

            if self.checkpoint_every and self.stats.episodes % self.checkpoint_every == 0:
                ckpt = self.save_checkpoint()
                if verbose:
                    print(f"  → checkpoint saved: {ckpt}")

        # Final flush.
        if batch and len(self.replay) >= self.min_fill_before_train:
            self.update(batch)
        ckpt = self.save_checkpoint(str(self.checkpoint_dir / "final.pt"))
        if verbose:
            wall = time.time() - start_time
            total = self.stats.wins + self.stats.losses + self.stats.draws
            win_pct = 100.0 * self.stats.wins / max(1, total)
            print(f"\n{'─'*60}")
            print(f"  Training complete — {self.stats.episodes:,} episodes in {wall:.0f}s")
            print(f"  Win/Loss/Draw: {self.stats.wins}/{self.stats.losses}/{self.stats.draws}  "
                  f"({win_pct:.1f}% win rate)")
            print(f"  Final checkpoint: {ckpt}")
            print(f"{'─'*60}\n")
