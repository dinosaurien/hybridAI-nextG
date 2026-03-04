from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

ActionType = str   # {"SCHEDULER_POLICY","MCS_CAP","PRB_WEIGHT","SLICE_QOS","TX_POWER","POWER_CONTROL","REPORTING"}
ScopeType = str    # {"CELL","UE","SLICE"}

@dataclass(frozen=True)
class ControlAction:
    type: ActionType
    scope: ScopeType
    cell_id: Optional[str] = None
    ue_id: Optional[str] = None
    slice_id: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        # JSON-safe serialization
        return {
            "type": self.type,
            "scope": self.scope,
            "cell_id": self.cell_id,
            "ue_id": self.ue_id,
            "slice_id": self.slice_id,
            "params": self.params or {},
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ControlAction":
        return ControlAction(
            type=d["type"],
            scope=d["scope"],
            cell_id=d.get("cell_id"),
            ue_id=d.get("ue_id"),
            slice_id=d.get("slice_id"),
            params=d.get("params", {}),
        )

@dataclass
class Playbook:
    actions: List[ControlAction]
    playbook_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_list(self) -> List[Dict[str, Any]]:
        return [a.to_dict() for a in self.actions]

    @staticmethod
    def from_list(arr: List[Dict[str, Any]]) -> "Playbook":
        return Playbook(actions=[ControlAction.from_dict(x) for x in arr])
