"""Staged curriculum controller.

Stages (configurable lengths per config YAML):
    1: encoding sanity (1-2 self-play games, no learning)
    2: train vs random opponent — graduates when rolling win rate >= stage2_win_threshold
    3: train vs heuristic through a series of levels:
         - blunder sub-levels: difficulty_start at 80 / 60 / 40 / 20 % blunder rate
         - then full-strength difficulty ramp: difficulty_start → difficulty_max
       each level is gated by stage3_difficulty_threshold; temperature resets on every bump
    4: self-play with checkpoint opponent pool
    5: human-data fine-tuning (no-op stub when no human data present)

Stage 2 and 3 thresholds use a rolling evaluation window (deque).  The budget
keys (stage2_episodes etc.) act as *hard safety caps* so a plateauing model
cannot stay stuck forever.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


STAGE_NAMES = [
    "stage1_sanity",
    "stage2_vs_random",
    "stage3_vs_heuristic",
    "stage4_self_play",
    "stage5_human_finetune",
]


@dataclass
class CurriculumState:
    current_stage: int = 1
    episodes_in_stage: int = 0
    stage_budgets: Dict[int, int] = None  # type: ignore[assignment]
    heuristic_level_idx: int = 0          # index into Curriculum._levels
    last_event: Optional[str] = None      # set by step(); read by trainer for display

    def episodes_left(self) -> int:
        budget = self.stage_budgets.get(self.current_stage, 0)
        return max(0, budget - self.episodes_in_stage)

    def stage_name(self) -> str:
        idx = max(1, min(len(STAGE_NAMES), self.current_stage))
        return STAGE_NAMES[idx - 1]


class Curriculum:
    """Track which stage we are in and advance when the conditions are met.

    Stage 3 level list (built from config):
        (difficulty_start, 0.80)  ← blunder sub-levels
        (difficulty_start, 0.60)
        (difficulty_start, 0.40)
        (difficulty_start, 0.20)
        (difficulty_start, 0.00)  ← full-strength difficulty ramp begins
        (difficulty_start+1, 0.00)
        ...
        (difficulty_max, 0.00)    ← must hold threshold here to graduate
    """

    def __init__(
        self,
        stage_budgets: Dict[int, int],
        start_stage: int = 1,
        stage2_win_threshold: float = 0.60,
        stage3_blunder_rates: Optional[List[float]] = None,
        stage3_difficulty_start: int = 1,
        stage3_difficulty_max: int = 10,
        stage3_difficulty_threshold: float = 0.55,
        eval_window: int = 200,
    ) -> None:
        if start_stage not in stage_budgets:
            raise ValueError(
                f"start_stage {start_stage} not in budgets {list(stage_budgets)}"
            )

        # Build the ordered level list for stage 3.
        blunder_rates = stage3_blunder_rates if stage3_blunder_rates is not None else [0.80, 0.60, 0.40, 0.20]
        self._levels: List[Tuple[int, float]] = []
        for rate in blunder_rates:
            self._levels.append((int(stage3_difficulty_start), float(rate)))
        for d in range(int(stage3_difficulty_start), int(stage3_difficulty_max) + 1):
            self._levels.append((d, 0.0))

        self.state = CurriculumState(
            current_stage=start_stage,
            episodes_in_stage=0,
            stage_budgets=dict(stage_budgets),
            heuristic_level_idx=0,
        )
        self._stage2_win_threshold = float(stage2_win_threshold)
        self._stage3_difficulty_threshold = float(stage3_difficulty_threshold)
        self._eval_window = int(eval_window)
        self._recent_results: deque = deque(maxlen=self._eval_window)

    # ------------------------------------------------------------------
    # Outcome tracking

    def record_outcome(self, won: bool) -> None:
        """Call once per episode with the learned agent's result."""
        self._recent_results.append(1.0 if won else 0.0)

    def rolling_win_rate(self) -> float:
        if not self._recent_results:
            return 0.0
        return sum(self._recent_results) / len(self._recent_results)

    def window_full(self) -> bool:
        return len(self._recent_results) >= self._eval_window

    # ------------------------------------------------------------------
    # Current heuristic opponent parameters

    def heuristic_params(self) -> Tuple[int, float]:
        """Return (difficulty, blunder_probability) for the current stage 3 level."""
        if not self._levels:
            return (1, 0.0)
        idx = min(self.state.heuristic_level_idx, len(self._levels) - 1)
        return self._levels[idx]

    def heuristic_difficulty(self) -> int:
        return self.heuristic_params()[0]

    def level_label(self) -> str:
        """Short label for display: 'b80', 'b60', 'd1', 'd5', etc."""
        diff, blunder = self.heuristic_params()
        if blunder > 0:
            return f"b{int(round(blunder * 100))}"
        return f"d{diff}"

    # ------------------------------------------------------------------
    # Core advance logic

    def step(self) -> None:
        self.state.last_event = None
        self.state.episodes_in_stage += 1
        stage = self.state.current_stage
        max_stage = max(self.state.stage_budgets)

        if stage >= max_stage:
            return  # final stage; just count

        budget_exhausted = self.state.episodes_left() <= 0

        if stage <= 1:
            if budget_exhausted:
                self._advance_stage()
            return

        if stage == 2:
            win_rate = self.rolling_win_rate()
            threshold_met = self.window_full() and win_rate >= self._stage2_win_threshold
            if threshold_met or budget_exhausted:
                self._advance_stage()
                reason = "threshold" if threshold_met else "budget"
                self.state.last_event = f"stage_advance:{reason}:{win_rate:.3f}"
            return

        if stage == 3:
            win_rate = self.rolling_win_rate()
            threshold_met = self.window_full() and win_rate >= self._stage3_difficulty_threshold
            if threshold_met:
                at_last_level = self.state.heuristic_level_idx >= len(self._levels) - 1
                if not at_last_level:
                    self.state.heuristic_level_idx += 1
                    self._recent_results.clear()
                    self.state.last_event = (
                        f"difficulty_bump:{self.level_label()}:{win_rate:.3f}"
                    )
                else:
                    self._advance_stage()
                    self.state.last_event = f"stage_advance:threshold:{win_rate:.3f}"
            elif budget_exhausted:
                self._advance_stage()
                self.state.last_event = f"stage_advance:budget:{win_rate:.3f}"
            return

        # Stage 4+: pure budget
        if budget_exhausted:
            self._advance_stage()

    def _advance_stage(self) -> None:
        self.state.current_stage += 1
        self.state.episodes_in_stage = 0
        self._recent_results.clear()

    # ------------------------------------------------------------------

    def opponent_kind(self) -> str:
        """Map stage -> opponent type label used by the trainer."""
        stage = self.state.current_stage
        if stage <= 2:
            return "random"
        if stage == 3:
            return "heuristic"
        return "self"

    def finished(self) -> bool:
        return (
            self.state.current_stage == max(self.state.stage_budgets)
            and self.state.episodes_left() == 0
        )

    @classmethod
    def from_config(cls, cfg: dict, start_stage: int = 1) -> "Curriculum":
        budgets = {
            1: int(cfg.get("stage1_episodes", 10)),
            2: int(cfg.get("stage2_episodes", 30000)),
            3: int(cfg.get("stage3_episodes", 120000)),
            4: int(cfg.get("stage4_episodes", 70000)),
            5: int(cfg.get("stage5_episodes", 0)),
        }
        blunder_rates = cfg.get("stage3_blunder_rates", [0.80, 0.60, 0.40, 0.20])
        return cls(
            stage_budgets=budgets,
            start_stage=start_stage,
            stage2_win_threshold=float(cfg.get("stage2_win_threshold", 0.60)),
            stage3_blunder_rates=[float(r) for r in blunder_rates],
            stage3_difficulty_start=int(cfg.get("stage3_difficulty_start", 1)),
            stage3_difficulty_max=int(cfg.get("stage3_difficulty_max", 10)),
            stage3_difficulty_threshold=float(cfg.get("stage3_difficulty_threshold", 0.55)),
            eval_window=int(cfg.get("eval_window", 200)),
        )
