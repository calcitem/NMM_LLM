"""Run N-game matches between two agents and emit a metrics report."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from learned_ai.models.state_encoder import PHASE_NAMES
from learned_ai.training.self_play import play_game


@dataclass
class EvalResult:
    games: int
    agent1_name: str
    agent2_name: str
    agent1_wins: int = 0
    agent2_wins: int = 0
    draws: int = 0
    avg_plies: float = 0.0
    phase_move_counts: Dict[str, int] = field(default_factory=dict)
    per_game: List[dict] = field(default_factory=list)

    @property
    def agent1_winrate(self) -> float:
        return self.agent1_wins / max(1, self.games)

    @property
    def agent2_winrate(self) -> float:
        return self.agent2_wins / max(1, self.games)

    def summary(self) -> str:
        return (
            f"{self.agent1_name} vs {self.agent2_name} over {self.games} games:\n"
            f"  {self.agent1_name}: {self.agent1_wins} wins ({self.agent1_winrate:.1%})\n"
            f"  {self.agent2_name}: {self.agent2_wins} wins ({self.agent2_winrate:.1%})\n"
            f"  draws: {self.draws}\n"
            f"  avg plies: {self.avg_plies:.1f}\n"
            f"  phase move distribution: {self.phase_move_counts}"
        )

    def to_dict(self) -> dict:
        out = asdict(self)
        out["agent1_winrate"] = self.agent1_winrate
        out["agent2_winrate"] = self.agent2_winrate
        return out


def evaluate_match(
    agent1_factory: Callable[[str], object],
    agent2_factory: Callable[[str], object],
    games: int = 50,
    agent1_name: str = "agent1",
    agent2_name: str = "agent2",
    max_plies: int = 400,
    alternate_colors: bool = True,
    output_json_path: Optional[str] = None,
) -> EvalResult:
    """Play *games* between two agents.

    Each ``*_factory`` takes a color string ("W"/"B") and returns a fresh
    agent for that side — this lets us instantiate a clean state per game
    (important for any agent that caches across moves) and to swap colors
    each round when ``alternate_colors`` is True.
    """
    result = EvalResult(
        games=games,
        agent1_name=agent1_name,
        agent2_name=agent2_name,
    )
    total_plies = 0
    for g in range(games):
        agent1_color = "W" if (not alternate_colors or g % 2 == 0) else "B"
        agent2_color = "B" if agent1_color == "W" else "W"
        a1 = agent1_factory(agent1_color)
        a2 = agent2_factory(agent2_color)
        if agent1_color == "W":
            outcome = play_game(a1, a2, max_plies=max_plies)
        else:
            outcome = play_game(a2, a1, max_plies=max_plies)

        if outcome.winner is None:
            result.draws += 1
        elif outcome.winner == agent1_color:
            result.agent1_wins += 1
        else:
            result.agent2_wins += 1

        total_plies += outcome.plies
        for step in outcome.trajectory:
            name = PHASE_NAMES[step.phase_id]
            result.phase_move_counts[name] = result.phase_move_counts.get(name, 0) + 1
        result.per_game.append(
            {
                "game": g,
                "winner": outcome.winner,
                "plies": outcome.plies,
                "agent1_color": agent1_color,
                "draw_reason": outcome.draw_reason,
            }
        )

    result.avg_plies = total_plies / max(1, games)

    if output_json_path:
        Path(output_json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)

    return result
