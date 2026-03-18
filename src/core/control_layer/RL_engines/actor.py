from __future__ import annotations
import json
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from core.common.types import ControlAction, Playbook

# ---- Actor ----

class Actor:
    """Converts Playbook objects to structured JSON and saves them as playbook files."""

    def __init__(self, out_dir: Union[str, Path] = "playbooks"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Public API ----
    def make_payload(
        self,
        playbook: Playbook,
        intent: Optional[Dict[str, Any]] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Return a dict payload ready to serialize to JSON."""
        playbook_id = playbook.playbook_id or datetime.now(timezone.utc).strftime("pb_%Y%m%dT%H%M%S%fZ")
        created_at = datetime.now(timezone.utc).isoformat()

        payload: Dict[str, Any] = {
            "playbook_id": playbook_id,
            "created_at": created_at,
            "intent": intent or {},
            "metadata": {**(playbook.metadata or {}), **(extra_meta or {})},
            "actions": playbook.to_list(),
        }

        self._validate_payload(payload)
        return payload

    def save_payload(
        self,
        payload: Dict[str, Any],
        filename: Optional[str] = None,
    ) -> Path:
        """Write payload to disk and return the path."""
        if not filename:
            filename = f"{payload.get('playbook_id','playbook')}.json"
        path = self.out_dir / filename
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return path

    def make_and_save(
        self,
        playbook: Playbook,
        intent: Optional[Dict[str, Any]] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
        filename: Optional[str] = None,
    ) -> Path:
        payload = self.make_payload(playbook, intent=intent, extra_meta=extra_meta)
        return self.save_payload(payload, filename=filename)

    # ---- Validation ----
    def _validate_payload(self, payload: Dict[str, Any]) -> None:
        if "actions" not in payload or not isinstance(payload["actions"], list) or len(payload["actions"]) == 0:
            raise ValueError("Payload must include a non-empty 'actions' list.")
        for i, a in enumerate(payload["actions"]):
            if "type" not in a or "scope" not in a:
                raise ValueError(f"Action #{i} must include 'type' and 'scope'.")
            # Basic whitelist checks; extend as needed.
            if a["type"] not in {"SCHEDULER_POLICY", "MCS_CAP", "PRB_WEIGHT", "SLICE_QOS", "TX_POWER", "POWER_CONTROL", "REPORTING"}:
                raise ValueError(f"Unsupported action type: {a['type']}")
            if a["scope"] not in {"CELL", "UE", "SLICE"}:
                raise ValueError(f"Unsupported action scope: {a['scope']}")
            # Scope-target consistency (soft check)
            # REPORTING actions are no-ops and don't need cell_id/slice_id
            if a["type"] != "REPORTING":
                if a["scope"] == "CELL" and not a.get("cell_id"):
                    raise ValueError(f"Action #{i} has scope=CELL but no 'cell_id'.")
                if a["scope"] == "SLICE" and not a.get("slice_id"):
                    raise ValueError(f"Action #{i} has scope=SLICE but no 'slice_id'.")
            # Params presence
            if "params" not in a or not isinstance(a["params"], dict):
                raise ValueError(f"Action #{i} must include 'params' dict (can be empty {{}})." )


# If you already have your own ControlAction/Playbook classes, you can still use 
# Actor.make_payload() — it only requires that playbook.actions contain items with 
# a to_dict() method returning the same keys. (Good GPT prompt i think)