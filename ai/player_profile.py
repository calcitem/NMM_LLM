"""ai/player_profile.py — Persistent player profile with Elo rating."""
from __future__ import annotations
import json
import re
from pathlib import Path
from datetime import date

_PROFILES_DIR = Path(__file__).parent.parent / "data" / "players"
_SAFE_NAME_RE = re.compile(r'^[a-zA-Z0-9 _\-\.]{1,50}$')
_K = 32  # Elo K-factor

# AI Elo by difficulty (approximate)
_DIFFICULTY_ELO = {
    1: 600, 2: 700, 3: 800, 4: 900, 5: 1000,
    6: 1100, 7: 1250, 8: 1400, 9: 1550, 10: 1700,
}


def is_valid_name(name: str) -> bool:
    return bool(_SAFE_NAME_RE.match(name.strip()))


class PlayerProfile:
    """Player profile persisted as JSON in data/players/."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.elo: int = 1000
        self.wins: int = 0
        self.losses: int = 0
        self.draws: int = 0
        self.current_difficulty: int = 3
        self.win_streak: int = 0
        self.loss_streak: int = 0
        self.extra_blunder: float = 0.0
        self.created_at: str = str(date.today())
        self.last_played: str = str(date.today())

    @property
    def games_played(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def win_rate(self) -> float | None:
        return self.wins / self.games_played if self.games_played else None

    def to_dict(self) -> dict:
        return {
            "name":               self.name,
            "elo":                self.elo,
            "wins":               self.wins,
            "losses":             self.losses,
            "draws":              self.draws,
            "games_played":       self.games_played,
            "win_rate":           self.win_rate,
            "current_difficulty": self.current_difficulty,
            "win_streak":         self.win_streak,
            "loss_streak":        self.loss_streak,
            "extra_blunder":      self.extra_blunder,
            "created_at":         self.created_at,
            "last_played":        self.last_played,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PlayerProfile":
        p = cls(d["name"])
        p.elo                = int(d.get("elo", 1000))
        p.wins               = int(d.get("wins", 0))
        p.losses             = int(d.get("losses", 0))
        p.draws              = int(d.get("draws", 0))
        p.current_difficulty = int(d.get("current_difficulty", 3))
        p.win_streak         = int(d.get("win_streak", 0))
        p.loss_streak        = int(d.get("loss_streak", 0))
        p.extra_blunder      = float(d.get("extra_blunder", 0.0))
        p.created_at         = d.get("created_at", str(date.today()))
        p.last_played        = d.get("last_played", str(date.today()))
        return p

    def record_result(self, human_won: bool | None, difficulty: int) -> None:
        """Record one game outcome and update Elo."""
        self.last_played = str(date.today())
        opponent_elo = _DIFFICULTY_ELO.get(difficulty, 800)

        if human_won is True:
            self.wins += 1
            actual = 1.0
            self.win_streak += 1
            self.loss_streak = 0
        elif human_won is False:
            self.losses += 1
            actual = 0.0
            self.loss_streak += 1
            self.win_streak = 0
        else:
            self.draws += 1
            actual = 0.5

        expected = 1.0 / (1.0 + 10.0 ** ((opponent_elo - self.elo) / 400.0))
        self.elo = max(100, int(self.elo + _K * (actual - expected)))

    def sync_adaptive(self, adaptive) -> None:
        """Copy adaptive tracker state into the profile for persistence."""
        self.current_difficulty = adaptive.current_difficulty
        self.win_streak         = adaptive.win_streak
        self.loss_streak        = adaptive.loss_streak
        self.extra_blunder      = adaptive.extra_blunder


def _profile_path(name: str) -> Path:
    safe = name.strip().replace(" ", "_")[:40]
    return _PROFILES_DIR / f"{safe}.json"


def load_profile(name: str) -> "PlayerProfile":
    _PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    path = _profile_path(name)
    if path.exists():
        try:
            return PlayerProfile.from_dict(json.loads(path.read_text()))
        except Exception:
            pass
    return PlayerProfile(name)


def save_profile(profile: "PlayerProfile") -> None:
    _PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    _profile_path(profile.name).write_text(json.dumps(profile.to_dict(), indent=2))
