"""ai/memory_manager.py — ChromaDB vector store + session JSONL log."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction, DefaultEmbeddingFunction

_STRATEGY_SEEDS = [
    {
        "id": "midgame_01",
        "text": (
            "Mill hierarchy (priorities): Double mill with feeders > double mill > "
            "mill with feeder > two independent mills > single mill. Use this to decide "
            "which piece to capture and when to make a mill."
        ),
        "topic": "mill_hierarchy",
    },
    {
        "id": "midgame_02",
        "text": (
            "Cannon fodder: Form mills that are purely targets to force the opponent to "
            "remove a specific piece rather than the one they'd prefer. Useful when "
            "blocking a double-mill setup."
        ),
        "topic": "cannon_fodder",
    },
    {
        "id": "midgame_03",
        "text": (
            "When to allow a mill: Sometimes letting the opponent form a mill is better "
            "than blocking it — especially when blocking would over-commit pieces and "
            "reduce your mobility."
        ),
        "topic": "allow_mill",
    },
    {
        "id": "midgame_04",
        "text": (
            "Mill redundancy: An opponent's mill can be made redundant when you have an "
            "open mill threatening to close. They are forced to stay put while you "
            "manoeuvre. Patience is required."
        ),
        "topic": "mill_redundancy",
    },
    {
        "id": "midgame_05",
        "text": (
            "Mill abandonment: Abandoning your own mill frees your pieces while the "
            "opponent must stay in position to hold the mill closed. Gains positional "
            "advantage at the cost of the mill."
        ),
        "topic": "mill_abandonment",
    },
    {
        "id": "midgame_06",
        "text": (
            "Sacrificial mills: Give up a potential mill to disable the opponent's "
            "double-mill setup. Often forces them into a series of captures that "
            "ultimately leaves you with the upper hand."
        ),
        "topic": "sacrificial_mills",
    },
    {
        "id": "midgame_07",
        "text": (
            "Cardinal point abandonment: Vacating a cardinal point (midpoint of a ring "
            "side) can force opponent pieces into worse positions, link three of your "
            "own pieces, or bait a forced response. Never abandon without a reason."
        ),
        "topic": "cardinal_point",
    },
    {
        "id": "midgame_08",
        "text": (
            "Feeder pieces: Pieces adjacent to an open mill that can close it on the "
            "next move. A mill with a feeder is worth significantly more than a "
            "standalone mill — plan for feeders before closing."
        ),
        "topic": "feeder_pieces",
    },
    {
        "id": "blunder_01",
        "text": (
            "Capitalising on unprotected pieces: When the opponent's piece is "
            "unprotected (no adjacent own pieces), prioritise moves that threaten or "
            "capture it before they can retreat."
        ),
        "topic": "unprotected_pieces",
    },
    {
        "id": "blunder_02",
        "text": (
            "Identifying deliberate AI mistakes: When the AI flags last_was_blunder, "
            "look for: an open mill that could have been blocked, a piece left "
            "isolated, a cardinal point vacated without reason, or a missed capture "
            "opportunity."
        ),
        "topic": "identify_blunder",
    },
]


class MemoryManager:
    def __init__(
        self,
        chroma_path: str = "data/chroma",
        games_path: str = "data/games",
        session_path: str = "data/session_memory",
        ollama_url: str = "http://localhost:11434",
        ollama_model: str = "llama3.2",
        use_ollama_embeddings: bool = True,
    ) -> None:
        self._games_path = Path(games_path)
        self._session_path = Path(session_path)
        self._games_path.mkdir(parents=True, exist_ok=True)
        self._session_path.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(path=chroma_path)

        if use_ollama_embeddings:
            self._embed_fn = self._make_embed_fn(ollama_url, ollama_model)
        else:
            self._embed_fn = DefaultEmbeddingFunction()

        self._bad_moves = self._client.get_or_create_collection(
            name="bad_moves",
            embedding_function=self._embed_fn,
        )
        self._narratives = self._client.get_or_create_collection(
            name="game_narratives",
            embedding_function=self._embed_fn,
        )
        self._strategy = self._client.get_or_create_collection(
            name="strategy_knowledge",
            embedding_function=self._embed_fn,
        )
        self._seed_strategy()

    @staticmethod
    def _make_embed_fn(ollama_url: str, ollama_model: str):
        try:
            fn = OllamaEmbeddingFunction(
                url=f"{ollama_url}/api/embeddings",
                model_name=ollama_model,
            )
            # Probe with a tiny embed to verify the model is available.
            fn(["test"])
            return fn
        except Exception:
            print(
                f"  [MemoryManager] Ollama embeddings unavailable "
                f"(model '{ollama_model}' not found or Ollama offline). "
                f"Run: ollama pull {ollama_model}"
            )
            return DefaultEmbeddingFunction()

    # ── Strategy seeding ──────────────────────────────────────────────────────

    def _seed_strategy(self) -> None:
        existing = self._strategy.get(ids=[s["id"] for s in _STRATEGY_SEEDS])
        existing_ids = set(existing["ids"])
        new_entries = [s for s in _STRATEGY_SEEDS if s["id"] not in existing_ids]
        if not new_entries:
            return
        self._strategy.add(
            ids=[s["id"] for s in new_entries],
            documents=[s["text"] for s in new_entries],
            metadatas=[{"topic": s["topic"]} for s in new_entries],
        )

    # ── Bad move memory ───────────────────────────────────────────────────────

    def store_bad_move(
        self,
        board_fen: str,
        move: dict,
        reason: str,
        full_board_ascii: str = "",
    ) -> None:
        doc = f"FEN: {board_fen}\nMove: {move}\nReason: {reason}\n{full_board_ascii}"
        self._bad_moves.add(
            ids=[str(uuid.uuid4())],
            documents=[doc],
            metadatas=[{"board_fen": board_fen, "move": json.dumps(move), "reason": reason}],
        )

    def retrieve_similar_positions(self, board_fen: str, n_results: int = 5) -> list[dict]:
        count = self._bad_moves.count()
        if count == 0:
            return []
        try:
            results = self._bad_moves.query(
                query_texts=[board_fen],
                n_results=min(n_results, count),
            )
        except Exception:
            return []
        out = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            out.append({"document": doc, "metadata": meta})
        return out

    # ── Game records ──────────────────────────────────────────────────────────

    def save_game_record(self, record: dict) -> None:
        session_id = record.get("session_id", str(uuid.uuid4()))
        date_str = record.get("date", datetime.now().isoformat())[:10]
        fname = self._games_path / f"game_{date_str}_{session_id[:8]}.jsonl"
        with open(fname, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def load_recent_games(self, n: int = 10) -> list[dict]:
        files = sorted(self._games_path.glob("*.jsonl"), reverse=True)
        records: list[dict] = []
        for fpath in files:
            with open(fpath, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            if len(records) >= n:
                break
        return records[:n]

    # ── Session narratives ────────────────────────────────────────────────────

    def save_session_narrative(self, text: str) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = self._session_path / f"narrative_{ts}.md"
        fname.write_text(text, encoding="utf-8")
        entry_id = f"narrative_{ts}"
        self._narratives.add(
            ids=[entry_id],
            documents=[text],
            metadatas=[{"timestamp": ts}],
        )

    def retrieve_relevant_narratives(self, query: str, n: int = 3) -> list[str]:
        count = self._narratives.count()
        if count == 0:
            return []
        try:
            results = self._narratives.query(
                query_texts=[query],
                n_results=min(n, count),
            )
        except Exception:
            return []
        return results["documents"][0]

    # ── Strategy retrieval ────────────────────────────────────────────────────

    def retrieve_strategy(self, query: str, n: int = 3) -> list[str]:
        count = self._strategy.count()
        if count == 0:
            return []
        try:
            results = self._strategy.query(
                query_texts=[query],
                n_results=min(n, count),
            )
        except Exception:
            return []
        return results["documents"][0]

    # ── Pattern analysis ──────────────────────────────────────────────────────

    def analyse_patterns(self, recent_games: list[dict]) -> dict:
        from collections import Counter

        placement_counts: Counter = Counter()
        human_weakness: Counter = Counter()

        for game in recent_games:
            for move in game.get("moves", []):
                if move.get("type") == "place":
                    placement_counts[move.get("to", "")] += 1
                comment = move.get("llm_poor_move_comment")
                if comment:
                    human_weakness[move.get("to", "")] += 1

        return {
            "human_preferred_placements": placement_counts.most_common(5),
            "human_weakness_positions": human_weakness.most_common(5),
        }
