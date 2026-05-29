"""Staged curriculum controller.

Stages (configurable lengths per config YAML):
    1: encoding sanity (1-2 self-play games, no learning)
    2: train vs random opponent until win-rate threshold
    3: train vs heuristic opponent
    4: self-play with checkpoint opponent pool
    5: human-data fine-tuning (no-op stub when no human data present)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


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

    def episodes_left(self) -> int:
        budget = self.stage_budgets.get(self.current_stage, 0)
        return max(0, budget - self.episodes_in_stage)

    def stage_name(self) -> str:
        idx = max(1, min(len(STAGE_NAMES), self.current_stage))
        return STAGE_NAMES[idx - 1]


class Curriculum:
    """Track which stage we are in and advance when the budget expires."""

    def __init__(self, stage_budgets: Dict[int, int], start_stage: int = 1) -> None:
        if start_stage not in stage_budgets:
            raise ValueError(
                f"start_stage {start_stage} not in budgets {list(stage_budgets)}"
            )
        self.state = CurriculumState(
            current_stage=start_stage,
            episodes_in_stage=0,
            stage_budgets=dict(stage_budgets),
        )

    def step(self) -> None:
        self.state.episodes_in_stage += 1
        if (
            self.state.episodes_left() <= 0
            and self.state.current_stage < max(self.state.stage_budgets)
        ):
            self.state.current_stage += 1
            self.state.episodes_in_stage = 0

    def opponent_kind(self) -> str:
        """Map stage -> opponent type label used by the trainer."""
        stage = self.state.current_stage
        if stage <= 2:
            return "random"
        if stage == 3:
            return "heuristic"
        if stage == 4:
            return "self"
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
            2: int(cfg.get("stage2_episodes", 5000)),
            3: int(cfg.get("stage3_episodes", 15000)),
            4: int(cfg.get("stage4_episodes", 30000)),
            5: int(cfg.get("stage5_episodes", 0)),
        }
        return cls(stage_budgets=budgets, start_stage=start_stage)
