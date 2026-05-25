"""
Append-only CSV logger for per-call LLM token usage and inference latency.
"""
import csv
import logging
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_HEADER = ["ts", "call_type", "prompt_tokens", "completion_tokens", "total_tokens", "latency_ms"]


class TokenLogger:
    def __init__(self, path: str = "tokens.csv"):
        self.path = Path(path)
        self._lock = threading.Lock()
        if not self.path.exists():
            with self.path.open("w", newline="") as f:
                csv.writer(f).writerow(_HEADER)
        logger.info(f"[TOKEN] Logging LLM token usage to {self.path.resolve()}")

    def log(self, call_type: str, usage: Optional[dict], latency_ms: float, ts: str):
        if usage is None:
            usage = {}
        row = [
            ts,
            call_type,
            usage.get("prompt_tokens", ""),
            usage.get("completion_tokens", ""),
            usage.get("total_tokens", ""),
            f"{latency_ms:.1f}",
        ]
        try:
            with self._lock, self.path.open("a", newline="") as f:
                csv.writer(f).writerow(row)
        except Exception as e:
            logger.error(f"[TOKEN] Failed to write token log row: {e}")
