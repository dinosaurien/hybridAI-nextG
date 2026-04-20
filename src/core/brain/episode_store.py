"""
Episodic memory for the Cognitive Agent.

ReflexionMemory — faithful to Shinn et al. (NeurIPS 2023), Algorithm 1.
Per-task verbal memory mem_i (task = (metric, cell_id)), reflections
generated ONLY on failure, retrieval is a tail-take of the last Omega
reflections (no embedding similarity). This is what the paper calls
"verbal reinforcement learning."

Future work: a retrieval-augmented grounding layer for the manual-intent
path (mock of the planned VSA-based symbolic-semantic lookup) would live
alongside this module.

Reference:
  Shinn, Cassano, Berman, Gopinath, Narasimhan, Yao.
    "Reflexion: Language Agents with Verbal Reinforcement Learning."
    NeurIPS 2023.
"""

import json
import logging
import os
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Shinn et al. §3: "Omega is usually set to 1-3 to adhere to max context LLM
# limitations." AlfWorld experiments use 3. We follow the same default.
DEFAULT_OMEGA = 3

# Hard cap per task bucket to prevent unbounded growth on disk. Reflections
# beyond Omega are never read by the agent (tail-take), but we keep a few
# extra in case future analysis needs the older trials.
PER_TASK_HARD_CAP = 20


# ====================================================================== #
#  Reflexion memory (Shinn et al., 2023)                                  #
# ====================================================================== #

class ReflexionMemory:
    """Per-task verbal memory, aligned with Shinn et al. Algorithm 1.

    A "task" is the tuple (metric, cell_id). Episodes are grouped so that a
    reflection written after a latency failure in CELL_001 only surfaces on
    subsequent latency anomalies in CELL_001 — never on a throughput anomaly
    in CELL_002. This deterministic, symbolic scoping matches how AlfWorld
    trials in the paper's reference implementation share memory per task
    (see alfworld_runs/alfworld_trial.py in the authors' repo).
    """

    def __init__(self,
                 store_path: str = "models/episodes.jsonl",
                 omega: int = DEFAULT_OMEGA,
                 per_task_cap: int = PER_TASK_HARD_CAP):
        self.store_path = Path(store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self.omega = omega
        self.per_task_cap = per_task_cap
        self._lock = threading.Lock()

        # task_key -> list[episode], ordered oldest -> newest.
        self.memory: Dict[Tuple[str, str], List[dict]] = {}
        # Flat id -> (task_key, index) lookup for O(1) reflection attachment.
        self._id_index: Dict[str, Tuple[Tuple[str, str], int]] = {}

        self._load_from_disk()

    # ------------------------------------------------------------------ #
    #  Task scoping                                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def task_key(anomaly: dict) -> Tuple[str, str]:
        metric = anomaly.get("metric", "unknown") or "unknown"
        cell = (anomaly.get("scope") or {}).get("cell_id", "unknown") or "unknown"
        return (metric, cell)

    # ------------------------------------------------------------------ #
    #  Persistence                                                        #
    # ------------------------------------------------------------------ #

    def _load_from_disk(self):
        if not self.store_path.exists():
            logger.info("[REFLEXION_MEM] No prior episodes on disk. Starting fresh.")
            return

        with open(self.store_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ep = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # Accept both new-format (task field present) and legacy
                # episodes (anomaly-only) so existing data survives.
                task = tuple(ep.get("task") or self.task_key(ep.get("anomaly", {})))
                bucket = self.memory.setdefault(task, [])
                bucket.append(ep)
                self._id_index[ep["id"]] = (task, len(bucket) - 1)

        # Enforce per-task cap on load (tail-take).
        for task, bucket in list(self.memory.items()):
            if len(bucket) > self.per_task_cap:
                self.memory[task] = bucket[-self.per_task_cap:]
                # Rebuild id index for this bucket.
                for i, ep in enumerate(self.memory[task]):
                    self._id_index[ep["id"]] = (task, i)

        total = sum(len(b) for b in self.memory.values())
        logger.info(f"[REFLEXION_MEM] Loaded {total} episodes across "
                    f"{len(self.memory)} task buckets.")

    def _append_to_disk(self, episode: dict):
        with open(self.store_path, "a") as f:
            f.write(json.dumps(episode, default=str) + "\n")

    def _rewrite_disk(self):
        """Atomically rewrite the JSONL file (used when a reflection is
        attached or the per-task cap is enforced). Same-directory temp file
        plus os.replace guarantees the file is never seen in a truncated
        state — either the old contents or the new contents are present.
        """
        fd, tmp_path = tempfile.mkstemp(
            dir=str(self.store_path.parent), suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w") as f:
                for bucket in self.memory.values():
                    for ep in bucket:
                        f.write(json.dumps(ep, default=str) + "\n")
            os.replace(tmp_path, str(self.store_path))
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    # ------------------------------------------------------------------ #
    #  Record a trial                                                     #
    # ------------------------------------------------------------------ #

    def record_trial(self,
                     anomaly: dict,
                     otm: Optional[dict],
                     outcome: dict) -> Optional[str]:
        """Persist a completed trial.

        Returns the episode id ONLY if the trial failed — the caller uses
        that id to request a reflection. Successful trials are still stored
        (for audit / metrics) but no reflection is generated, matching the
        canonical Reflexion loop where Msr is invoked when the evaluator
        reports failure.
        """
        otm = otm or {}
        task = self.task_key(anomaly)
        resolved = bool(outcome.get("resolved"))

        # Build a short trajectory the reflection prompt can quote.
        trajectory = [
            {"step": "observe_anomaly",
             "metric": anomaly.get("metric"),
             "value_before": anomaly.get("value"),
             "scope": anomaly.get("scope", {})},
            {"step": "apply_otm",
             "procedure_id": (otm.get("metadata") or {}).get("procedure_id"),
             "objective": otm.get("objective", {}),
             # applied_value (what actually went to ns-3) is already folded
             # into threshold by the orchestrator adapter before we get here.
             "constraints": [dict(c) for c in otm.get("constraints", [])]},
            {"step": "observe_outcome",
             "resolved": resolved,
             "value_after": outcome.get("metric_after"),
             "e2_acks": outcome.get("e2_acks")},
        ]

        episode = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "task": list(task),
            "anomaly": {
                "metric": anomaly.get("metric"),
                "value": anomaly.get("value"),
                "direction": anomaly.get("direction", "lower_better"),
                "scope": anomaly.get("scope", {}),
            },
            "trajectory": trajectory,
            "outcome": outcome,
            "resolved": resolved,
            "reflection": None,
        }

        with self._lock:
            bucket = self.memory.setdefault(task, [])
            bucket.append(episode)
            self._id_index[episode["id"]] = (task, len(bucket) - 1)
            self._append_to_disk(episode)

            if len(bucket) > self.per_task_cap:
                overflow = len(bucket) - self.per_task_cap
                dropped = bucket[:overflow]
                self.memory[task] = bucket[overflow:]
                for ep in dropped:
                    self._id_index.pop(ep["id"], None)
                for i, ep in enumerate(self.memory[task]):
                    self._id_index[ep["id"]] = (task, i)
                self._rewrite_disk()
                logger.info(f"[REFLEXION_MEM] Pruned {overflow} old episodes "
                            f"from task {task}.")

        logger.info(f"[REFLEXION_MEM] Recorded trial {episode['id'][:8]} "
                    f"(task={task}, resolved={resolved}).")

        # Canonical Reflexion: reflect only on failure.
        return episode["id"] if not resolved else None

    # ------------------------------------------------------------------ #
    #  Attach reflection                                                  #
    # ------------------------------------------------------------------ #

    def add_reflection(self, episode_id: str, reflection: str):
        with self._lock:
            entry = self._id_index.get(episode_id)
            if entry is None:
                logger.warning(f"[REFLEXION_MEM] Episode {episode_id[:8]} not found "
                               f"(likely pruned). Reflection discarded.")
                return
            task, idx = entry
            bucket = self.memory.get(task, [])
            if idx >= len(bucket) or bucket[idx]["id"] != episode_id:
                logger.warning(f"[REFLEXION_MEM] Episode {episode_id[:8]} index stale.")
                return
            bucket[idx]["reflection"] = reflection
            self._rewrite_disk()
        logger.info(f"[REFLEXION_MEM] Reflection attached to episode {episode_id[:8]}.")

    # ------------------------------------------------------------------ #
    #  Retrieval — tail-take of last Omega reflections                    #
    # ------------------------------------------------------------------ #

    def get_recent_reflections(self, anomaly: dict) -> List[str]:
        """Return up to Omega most recent reflections for this task.

        This is the paper's exact retrieval: no similarity, no ranking,
        just mem[-Omega:] for the current task. Reflections are returned
        oldest -> newest so the most recent guidance appears last.
        """
        task = self.task_key(anomaly)
        with self._lock:
            bucket = self.memory.get(task, [])
            reflections = [ep["reflection"] for ep in bucket
                           if ep.get("reflection")]
        return reflections[-self.omega:]

    def get_latest_episode(self) -> Optional[dict]:
        """Return the most recently recorded episode across all tasks."""
        with self._lock:
            latest = None
            latest_ts = ""
            for bucket in self.memory.values():
                if not bucket:
                    continue
                ep = bucket[-1]
                ts = ep.get("timestamp", "")
                if ts > latest_ts:
                    latest = ep
                    latest_ts = ts
            return latest

    # ------------------------------------------------------------------ #
    #  Prompt formatting                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def format_reflections_for_prompt(reflections: List[str]) -> str:
        """Render tail-taken reflections as a compact block for the LLM."""
        if not reflections:
            return ""
        lines = ["PRIOR REFLECTIONS ON THIS TASK (most recent last):"]
        for i, r in enumerate(reflections, 1):
            lines.append(f"  {i}. {r}")
        return "\n".join(lines)


# ====================================================================== #
#  Backward-compat adapter                                                #
# ====================================================================== #

class EpisodeStore(ReflexionMemory):
    """Thin alias preserving the import path used by main.py / orchestrator /
    cognitive_agent. Exposes a small shim over the Reflexion API so the
    existing call sites (record / add_reflection) keep working.
    """

    def record(self, anomaly: dict, otm: dict, outcome: dict) -> str:
        """Legacy signature. Always returns an episode id (success or fail).
        The caller decides whether to request a reflection — for the
        canonical loop, only publish ai.reflect when outcome['resolved']
        is False.
        """
        ep_id_if_failed = self.record_trial(anomaly, otm, outcome)
        if ep_id_if_failed is not None:
            return ep_id_if_failed
        # Successful trial: fish out the id we just stored.
        task = self.task_key(anomaly)
        with self._lock:
            bucket = self.memory.get(task, [])
            return bucket[-1]["id"] if bucket else ""
