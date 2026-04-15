"""
Episodic memory store for the Cognitive Agent (Reflexion-style).

Records anomaly→OTM→outcome cycles, generates verbal self-reflections
via the LLM, and retrieves similar past episodes with their reflections
to augment future LLM prompts with operational experience.

Key insight from Reflexion (Shinn et al., NeurIPS 2023): store verbal
analysis of WHY something worked/failed, not just raw data. The reflection
is generated once at recording time and reused at retrieval time, giving
the LLM actionable guidance rather than raw numbers to re-interpret.

References:
  - Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (2020)
  - Shinn et al., "Reflexion: Language Agents with Verbal Reinforcement Learning" (2023)
"""

import json
import logging
import os
import tempfile
import threading
import uuid
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Reflexion bounds memory to 1-3 reflections per task.
# We keep a larger window since episodes span different anomaly types,
# but still prune to prevent unbounded growth.
DEFAULT_MAX_EPISODES = 50
DEFAULT_MIN_SIMILARITY = 0.3


class EpisodeStore:

    def __init__(self, store_path: str = "models/episodes.jsonl",
                 model_name: str = "all-MiniLM-L6-v2",
                 max_episodes: int = DEFAULT_MAX_EPISODES):
        self.store_path = Path(store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_episodes = max_episodes
        self._lock = threading.Lock()  # Protects episodes list + embeddings + disk writes

        self.episodes: List[dict] = []
        self.embeddings: Optional[np.ndarray] = None  # (N, dim) matrix

        # Load embedding model (CPU-only, ~80MB)
        try:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(model_name, device="cpu")
            logger.info(f"[EPISODE_STORE] Loaded embedding model: {model_name}")
        except Exception as e:
            logger.warning(f"[EPISODE_STORE] Failed to load embedding model: {e}. "
                           "Retrieval will be disabled.")
            self.encoder = None

        self._load_from_disk()

    # ------------------------------------------------------------------ #
    #  Persistence                                                        #
    # ------------------------------------------------------------------ #

    def _load_from_disk(self):
        """Rebuild episode list and embedding matrix from JSONL on startup."""
        if not self.store_path.exists():
            logger.info("[EPISODE_STORE] No existing episodes found. Starting fresh.")
            return

        loaded = []
        with open(self.store_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        loaded.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        if not loaded:
            return

        # Apply memory bound, keep most recent episodes
        if len(loaded) > self.max_episodes:
            loaded = loaded[-self.max_episodes:]

        self.episodes = loaded
        logger.info(f"[EPISODE_STORE] Loaded {len(self.episodes)} episodes from disk.")

        # Rebuild embeddings
        if self.encoder:
            texts = [self._context_to_text(ep.get("anomaly", {})) for ep in self.episodes]
            self.embeddings = self.encoder.encode(texts, normalize_embeddings=True)
            logger.info(f"[EPISODE_STORE] Rebuilt embedding index: {self.embeddings.shape}")

    def _append_to_disk(self, episode: dict):
        """Append a single episode to the JSONL file."""
        with open(self.store_path, "a") as f:
            f.write(json.dumps(episode, default=str) + "\n")

    def _rewrite_disk(self):
        """Atomically rewrite the JSONL file (used after pruning / reflection updates).

        Writes to a temp file in the same directory, then renames. This prevents
        data loss if the process crashes mid-write — either the old file or the
        new file exists, never a truncated one.
        """
        fd, tmp_path = tempfile.mkstemp(
            dir=str(self.store_path.parent), suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w") as f:
                for ep in self.episodes:
                    f.write(json.dumps(ep, default=str) + "\n")
            os.replace(tmp_path, str(self.store_path))
        except BaseException:
            # Clean up temp file on failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def _prune_if_needed(self):
        """Enforce memory bound — drop oldest episodes beyond max_episodes."""
        if len(self.episodes) <= self.max_episodes:
            return
        overflow = len(self.episodes) - self.max_episodes
        self.episodes = self.episodes[overflow:]
        if self.embeddings is not None:
            self.embeddings = self.embeddings[overflow:]
        self._rewrite_disk()
        logger.info(f"[EPISODE_STORE] Pruned {overflow} old episodes. "
                     f"Remaining: {len(self.episodes)}")

    # ------------------------------------------------------------------ #
    #  Record                                                             #
    # ------------------------------------------------------------------ #

    def record(self, anomaly: dict, otm: dict, outcome: dict) -> str:
        """Record a completed anomaly→OTM→outcome episode. Returns episode ID."""
        episode = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "anomaly": {
                "metric": anomaly.get("metric"),
                "value": anomaly.get("value"),
                "direction": anomaly.get("direction", "lower_better"),
                "scope": anomaly.get("scope", {}),
            },
            "otm_prescribed": {
                "objective": otm.get("objective", {}) if otm else {},
                # Constraints may include applied_value (actual value sent to ns-3)
                # and threshold_original (LLM's boundary before objective biasing).
                "constraints": [dict(c) for c in otm.get("constraints", [])] if otm else [],
                "procedure_id": otm.get("metadata", {}).get("procedure_id") if otm else None,
            },
            "outcome": outcome,
            "reflection": None,  # Populated async by CognitiveAgent
        }

        with self._lock:
            self.episodes.append(episode)
            self._append_to_disk(episode)

            # Update embedding index
            if self.encoder:
                new_emb = self.encoder.encode(
                    [self._context_to_text(episode["anomaly"])],
                    normalize_embeddings=True,
                )
                if self.embeddings is None:
                    self.embeddings = new_emb
                else:
                    self.embeddings = np.vstack([self.embeddings, new_emb])

            self._prune_if_needed()

        resolved = outcome.get("resolved", False)
        logger.info(f"[EPISODE_STORE] Recorded episode {episode['id'][:8]} "
                     f"(metric={anomaly.get('metric')}, resolved={resolved}, "
                     f"total_episodes={len(self.episodes)})")
        return episode["id"]

    # ------------------------------------------------------------------ #
    #  Self-Reflection (Reflexion pattern)                                #
    # ------------------------------------------------------------------ #

    def add_reflection(self, episode_id: str, reflection: str):
        """Attach a verbal self-reflection to an existing episode.

        Called by CognitiveAgent after generating a reflection via the LLM.
        This is the core Reflexion mechanism: analyze WHY something worked/failed
        once at recording time, so future retrievals get actionable guidance.
        """
        with self._lock:
            for ep in self.episodes:
                if ep["id"] == episode_id:
                    ep["reflection"] = reflection
                    self._rewrite_disk()
                    logger.info(f"[EPISODE_STORE] Reflection added to episode {episode_id[:8]}")
                    return
        # Episode was pruned before reflection arrived — log at WARNING level
        logger.warning(f"[EPISODE_STORE] Episode {episode_id[:8]} not found for reflection "
                       f"(likely pruned). Reflection discarded.")

    def get_latest_episode(self) -> Optional[dict]:
        """Return the most recent episode (for feeding into adaptation loops)."""
        return self.episodes[-1] if self.episodes else None

    # ------------------------------------------------------------------ #
    #  Retrieve                                                           #
    # ------------------------------------------------------------------ #

    def retrieve(self, anomaly_context: dict, k: int = 3,
                 min_similarity: float = DEFAULT_MIN_SIMILARITY) -> List[dict]:
        """Retrieve the K most similar past episodes for an anomaly context.

        Only returns episodes above the minimum similarity threshold to avoid
        surfacing irrelevant old episodes.
        """
        with self._lock:
            if not self.encoder or self.embeddings is None or len(self.episodes) == 0:
                return []

            query_text = self._context_to_text(anomaly_context)
            query_emb = self.encoder.encode([query_text], normalize_embeddings=True)

            # Cosine similarity (embeddings are already L2-normalized)
            similarities = (self.embeddings @ query_emb.T).flatten()

            # Top-K indices (descending similarity), filtered by threshold
            top_k = min(k, len(self.episodes))
            top_indices = np.argsort(similarities)[-top_k:][::-1]

            results = []
            for idx in top_indices:
                sim = float(similarities[idx])
                if sim < min_similarity:
                    continue
                ep = self.episodes[idx]
                results.append({**ep, "_similarity": sim})

        if results:
            logger.info(f"[EPISODE_STORE] Retrieved {len(results)} episodes for "
                         f"metric={anomaly_context.get('metric')} "
                         f"(best_sim={results[0]['_similarity']:.3f})")
        return results

    # ------------------------------------------------------------------ #
    #  Prompt formatting                                                  #
    # ------------------------------------------------------------------ #

    def format_for_prompt(self, episodes: List[dict]) -> str:
        """Format retrieved episodes for LLM prompt injection.

        If an episode has a reflection (Reflexion-style verbal analysis),
        show that it's more actionable than raw data. Fall back to raw
        data for episodes that haven't been reflected on yet.
        """
        if not episodes:
            return ""

        lines = []
        for i, ep in enumerate(episodes, 1):
            anomaly = ep.get("anomaly", {})
            outcome = ep.get("outcome", {})
            reflection = ep.get("reflection")

            resolved_str = "RESOLVED" if outcome.get("resolved") else "FAILED"
            metric = anomaly.get("metric", "unknown")
            val_before = anomaly.get("value", "?")
            val_after = outcome.get("metric_after", "?")

            header = f"Episode {i} [{resolved_str}]: {metric} = {val_before} → {val_after}"

            if reflection:
                # Reflexion-style: show the verbal analysis
                lines.append(f"{header}\n  Reflection: {reflection}")
            else:
                # Fallback: show raw constraint data with applied values
                otm = ep.get("otm_prescribed", {})
                proc_id = otm.get("procedure_id", "none")
                constraints_str = ""
                for c in otm.get("constraints", []):
                    kpi = c.get('kpi')
                    op = c.get('operator')
                    thresh = c.get('threshold')
                    applied = c.get('applied_value')
                    unit = c.get('unit', '')
                    if applied is not None and applied != thresh:
                        constraints_str += f"    - {kpi} {op} {thresh} {unit} (applied: {applied})\n"
                    else:
                        constraints_str += f"    - {kpi} {op} {thresh} {unit}\n"
                lines.append(
                    f"{header}\n"
                    f"  Procedure: {proc_id}\n"
                    f"  Constraints applied:\n{constraints_str}"
                )

        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  Helpers                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _context_to_text(anomaly: dict) -> str:
        """Convert an anomaly context dict to a string for embedding."""
        metric = anomaly.get("metric", "unknown")
        value = anomaly.get("value", 0)
        direction = anomaly.get("direction", "unknown")
        cell = anomaly.get("scope", {}).get("cell_id", "unknown")
        return f"anomaly: metric={metric} value={value} direction={direction} cell={cell}"
