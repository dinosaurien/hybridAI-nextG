from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Optional, Dict, Any, List, Tuple
import logging
import uuid as _uuid

_logger = logging.getLogger(__name__)

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


# Temporal scheduler

class ScheduleState(Enum):
    PENDING = auto()    # Waiting for activate_at
    ACTIVE = auto()     # Currently steering the RL engine
    EXPIRED = auto()    # Window ended, revert triggered
    CANCELLED = auto()  # User or system cancelled

@dataclass
class ScheduledIntent:
    id: str
    raw_text: str
    otm: Optional[Dict[str, Any]]
    activate_at: datetime
    deactivate_at: Optional[datetime] = None
    state: ScheduleState = ScheduleState.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    pre_intent_otm: Optional[Dict[str, Any]] = None  # snapshot for revert
    corr_id: str = field(default_factory=lambda: str(_uuid.uuid4()))


# Procedure queue

@dataclass
class ProcedureStep:
    order: int
    name: str                                           # e.g., "reduce_mcs"
    description: str                                    # human-readable
    otm_fragment: Optional[Dict[str, Any]]              # None = not actionable in ns-3
    ns3_capable: bool
    ns3_skip_reason: Optional[str] = None               # why this step is skipped
    health_check: Optional[Dict[str, Any]] = None       # {"checks": [{"metric": ..., "operator": ..., "threshold": ...}]}
    on_fail_goto: Optional[str] = None                  # step name to jump to on health check failure
    branch_only: bool = False                           # True = skip in normal linear flow, only reachable via on_fail_goto

class ProcedureState(Enum):
    ACTIVE = auto()
    COMPLETED = auto()
    FAILED = auto()

@dataclass
class ActiveProcedure:
    scenario_id: str
    scenario_name: str
    steps: List[ProcedureStep]
    current_step_idx: int = 0
    state: ProcedureState = ProcedureState.ACTIVE
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    pre_procedure_otm: Optional[Dict[str, Any]] = None   # snapshot for rollback
    step_failures: int = 0                               # consecutive failures on current step
    branched_to: bool = False                            # set True after on_fail_goto jump
