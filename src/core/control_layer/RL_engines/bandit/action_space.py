from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, Set
from enum import Enum
import itertools
import numpy as np

try:
    from ain.common.types import ControlAction, Playbook
except ImportError:
    # Fallback
    from dataclasses import dataclass as dc
    
    @dc
    class ControlAction:
        type: str
        scope: str
        cell_id: Optional[str] = None
        slice_id: Optional[str] = None
        ue_id: Optional[str] = None
        params: Dict[str, Any] = field(default_factory=dict)
    
    @dc 
    class Playbook:
        actions: List[ControlAction] = field(default_factory=list)


# -----------------------------
# Enhanced Action Categories
# -----------------------------

class ActionCategory(Enum):
    """Categories of RAN actions for contextual bandit."""
    SCHEDULER = "scheduler_policy"
    MODULATION = "modulation_coding"
    RESOURCE = "resource_allocation"  
    QOS = "quality_of_service"
    POWER = "power_control"
    REPORTING = "reporting_noop"

class ActionScope(Enum):
    """Scope levels for RAN actions."""
    CELL = "CELL"
    SLICE = "SLICE" 
    UE = "UE"
    BEARER = "BEARER"
    GLOBAL = "GLOBAL"

class ActionImpact(Enum):
    """Expected impact level of actions."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

# -----------------------------
# Enhanced Action Definitions
# -----------------------------

@dataclass
class ActionTemplate:
    """Template for generating contextual actions."""
    action_type: str
    scope: ActionScope
    parameter_grids: Dict[str, List[Any]]
    impact_level: ActionImpact
    target_metrics: List[str]  # Which metrics this action primarily affects
    conflict_groups: List[str]  # Which other actions conflict with this
    cooldown_weight: float = 1.0  # Multiplier for cooldown duration
    context_sensitivity: float = 1.0  # How much context affects this action
    
    def generate_actions(self, entities: List[str]) -> List[ControlAction]:
        """Generate all possible actions from this template."""
        actions = []
        
        # Generate parameter combinations
        param_names = list(self.parameter_grids.keys())
        if not param_names:
            param_combinations = [{}]
        else:
            param_values = [self.parameter_grids[name] for name in param_names]
            param_combinations = [
                dict(zip(param_names, combo))
                for combo in itertools.product(*param_values)
            ]
        
        # Generate actions for each entity and parameter combination
        for entity in entities:
            for params in param_combinations:
                action = ControlAction(
                    type=self.action_type,
                    scope=self.scope.value,
                    params=params
                )
                
                # Set entity ID based on scope
                if self.scope == ActionScope.CELL:
                    action.cell_id = entity
                elif self.scope == ActionScope.SLICE:
                    action.slice_id = entity
                elif self.scope == ActionScope.UE:
                    action.ue_id = entity
                
                actions.append(action)
        
        return actions

# -----------------------------
# Contextual Action Space
# -----------------------------

class ContextualActionSpace:
    """Enhanced action space with contextual bandit intelligence."""
    
    def __init__(self, cells: List[str] = None, slices: List[str] = None, ues: List[str] = None):
        self.cells = cells or ["CELL_001", "CELL_002"]
        self.slices = slices or ["SLICE_A", "SLICE_B"] 
        self.ues = ues or []
        
        # Initialize action templates
        self.action_templates = self._create_action_templates()
        
        # Cache for generated actions
        self._action_cache = {}
        self._context_action_cache = {}
        
        # Conflict and constraint definitions
        self.conflict_matrix = self._build_conflict_matrix()
        self.constraint_rules = self._build_constraint_rules()

    def _create_action_templates(self) -> Dict[str, ActionTemplate]:
        """Create comprehensive action templates."""
        templates = {}
        
        # Scheduler Policy Actions
        templates["SCHEDULER_POLICY"] = ActionTemplate(
            action_type="SCHEDULER_POLICY",
            scope=ActionScope.CELL,
            parameter_grids={
                "policy": ["PF", "RR", "MAX_THROUGHPUT", "WEIGHTED_FAIR", "QOS_AWARE"]
            },
            impact_level=ActionImpact.HIGH,
            target_metrics=["latency_ms", "throughput_mbps", "fairness_index"],
            conflict_groups=["scheduler"],
            context_sensitivity=0.9
        )
        
        # Modulation and Coding Scheme (MCS) Actions
        templates["MCS_CAP"] = ActionTemplate(
            action_type="MCS_CAP", 
            scope=ActionScope.CELL,
            parameter_grids={
                "dl_mcs_max": [12, 14, 16, 18, 20, 22, 24, 26, 28],
                "ul_mcs_max": [12, 14, 16, 18, 20, 22, 24, 26, 28]
            },
            impact_level=ActionImpact.MEDIUM,
            target_metrics=["throughput_mbps", "bler", "spectral_efficiency"],
            conflict_groups=["mcs"],
            context_sensitivity=0.8
        )
        
        # Physical Resource Block (PRB) Allocation
        templates["PRB_WEIGHT"] = ActionTemplate(
            action_type="PRB_WEIGHT",
            scope=ActionScope.SLICE,
            parameter_grids={
                "weight": [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8]
            },
            impact_level=ActionImpact.HIGH,
            target_metrics=["throughput_mbps", "latency_ms", "prb_utilization"],
            conflict_groups=["resource_allocation"],
            context_sensitivity=0.9
        )
        
        # Quality of Service (QoS) Parameters
        templates["SLICE_QOS"] = ActionTemplate(
            action_type="SLICE_QOS",
            scope=ActionScope.SLICE,
            parameter_grids={
                "priority": [1, 2, 3, 4, 5],
                "weight": [0.8, 1.0, 1.2, 1.5],
                "guaranteed_bitrate": [10, 20, 50, 100]  # Mbps
            },
            impact_level=ActionImpact.MEDIUM,
            target_metrics=["latency_ms", "throughput_mbps", "packet_loss"],
            conflict_groups=["qos"],
            context_sensitivity=0.7
        )
        
        # Power Control Actions
        templates["POWER_CONTROL"] = ActionTemplate(
            action_type="POWER_CONTROL",
            scope=ActionScope.CELL,
            parameter_grids={
                "tx_power_dbm": [40, 43, 46, 49, 52],
                "power_control_mode": ["OPEN_LOOP", "CLOSED_LOOP", "HYBRID"]
            },
            impact_level=ActionImpact.MEDIUM,
            target_metrics=["sinr_db", "interference", "energy_efficiency"],
            conflict_groups=["power"],
            context_sensitivity=0.6
        )
        
        # Admission Control
        templates["ADMISSION_CONTROL"] = ActionTemplate(
            action_type="ADMISSION_CONTROL",
            scope=ActionScope.CELL,
            parameter_grids={
                "max_ue_count": [50, 75, 100, 125, 150],
                "load_threshold": [0.7, 0.8, 0.85, 0.9, 0.95]
            },
            impact_level=ActionImpact.MEDIUM,
            target_metrics=["latency_ms", "blocking_rate", "cell_load"],
            conflict_groups=["admission"],
            context_sensitivity=0.5
        )
        
        # Handover Parameters
        templates["HANDOVER_PARAMS"] = ActionTemplate(
            action_type="HANDOVER_PARAMS",
            scope=ActionScope.CELL,
            parameter_grids={
                "ho_threshold_db": [1, 2, 3, 4, 5],
                "time_to_trigger_ms": [40, 80, 160, 320, 640]
            },
            impact_level=ActionImpact.LOW,
            target_metrics=["handover_rate", "call_drop_rate", "mobility_performance"],
            conflict_groups=["handover"],
            context_sensitivity=0.4
        )
        
        # No-op / Reporting Action
        templates["REPORTING"] = ActionTemplate(
            action_type="REPORTING",
            scope=ActionScope.CELL,
            parameter_grids={
                "noop": [True]
            },
            impact_level=ActionImpact.LOW,
            target_metrics=[],
            conflict_groups=[],
            context_sensitivity=0.0
        )
        
        return templates

    def _build_conflict_matrix(self) -> Dict[Tuple[str, str], bool]:
        """Build matrix of conflicting action types."""
        conflicts = {}
        
        # Actions that conflict within same scope/entity
        same_type_conflicts = [
            ("SCHEDULER_POLICY", "SCHEDULER_POLICY"),
            ("MCS_CAP", "MCS_CAP"),
            ("PRB_WEIGHT", "PRB_WEIGHT"), 
            ("SLICE_QOS", "SLICE_QOS"),
            ("POWER_CONTROL", "POWER_CONTROL"),
            ("ADMISSION_CONTROL", "ADMISSION_CONTROL"),
            ("HANDOVER_PARAMS", "HANDOVER_PARAMS")
        ]
        
        for type1, type2 in same_type_conflicts:
            conflicts[(type1, type2)] = True
            
        # Cross-type conflicts
        cross_conflicts = [
            ("SCHEDULER_POLICY", "PRB_WEIGHT"),  # Both affect resource allocation
            ("MCS_CAP", "POWER_CONTROL"),        # Both affect radio parameters
            ("ADMISSION_CONTROL", "PRB_WEIGHT")  # Both affect capacity
        ]
        
        for type1, type2 in cross_conflicts:
            conflicts[(type1, type2)] = True
            conflicts[(type2, type1)] = True  # Symmetric
            
        return conflicts

    def _build_constraint_rules(self) -> List[Dict[str, Any]]:
        """Build constraint rules for action validity."""
        rules = []
        
        # MCS constraint: UL should not exceed DL significantly
        rules.append({
            "name": "mcs_ul_dl_consistency",
            "action_type": "MCS_CAP",
            "validator": lambda params: (
                abs(params.get("ul_mcs_max", 18) - params.get("dl_mcs_max", 18)) <= 4
            )
        })
        
        # PRB weight constraint: Total weights shouldn't exceed reasonable bounds
        rules.append({
            "name": "prb_weight_bounds",
            "action_type": "PRB_WEIGHT", 
            "validator": lambda params: 0.5 <= params.get("weight", 1.0) <= 2.0
        })
        
        # Power control constraint: Reasonable power levels
        rules.append({
            "name": "power_bounds",
            "action_type": "POWER_CONTROL",
            "validator": lambda params: 30 <= params.get("tx_power_dbm", 46) <= 60
        })
        
        return rules

    def get_all_actions(self, use_cache: bool = True) -> List[ControlAction]:
        """Get all possible actions in this action space."""
        if use_cache and "all_actions" in self._action_cache:
            return self._action_cache["all_actions"]
        
        all_actions = []
        
        for template_name, template in self.action_templates.items():
            if template.scope == ActionScope.CELL:
                actions = template.generate_actions(self.cells)
            elif template.scope == ActionScope.SLICE:
                actions = template.generate_actions(self.slices)
            elif template.scope == ActionScope.UE:
                actions = template.generate_actions(self.ues)
            else:  # GLOBAL or other
                actions = template.generate_actions(["GLOBAL"])
            
            # Apply constraint rules
            valid_actions = []
            for action in actions:
                if self._is_action_valid(action, template):
                    valid_actions.append(action)
            
            all_actions.extend(valid_actions)
        
        if use_cache:
            self._action_cache["all_actions"] = all_actions
        
        return all_actions

    def get_contextual_actions(self, situation: str, top_k: int = None) -> List[ControlAction]:
        """Get actions most relevant for specific network situation."""
        cache_key = f"contextual_{situation}_{top_k}"
        if cache_key in self._context_action_cache:
            return self._context_action_cache[cache_key]
        
        all_actions = self.get_all_actions()
        
        # Score actions by contextual relevance
        scored_actions = []
        for action in all_actions:
            relevance_score = self._calculate_action_relevance(action, situation)
            scored_actions.append((action, relevance_score))
        
        # Sort by relevance
        scored_actions.sort(key=lambda x: x[1], reverse=True)
        
        # Take top-k if specified
        if top_k:
            scored_actions = scored_actions[:top_k]
        
        contextual_actions = [action for action, score in scored_actions]
        
        self._context_action_cache[cache_key] = contextual_actions
        return contextual_actions

    def _is_action_valid(self, action: ControlAction, template: ActionTemplate) -> bool:
        """Check if action satisfies constraint rules."""
        for rule in self.constraint_rules:
            if rule["action_type"] == action.type:
                if not rule["validator"](action.params):
                    return False
        return True

    def _calculate_action_relevance(self, action: ControlAction, situation: str) -> float:
        """Calculate how relevant an action is for a given network situation."""
        template = self.action_templates.get(action.type)
        if not template:
            return 0.5  # Neutral score for unknown actions
        
        base_score = 0.5
        
        # Situation-specific scoring
        if situation == "high_latency":
            if action.type in ["SCHEDULER_POLICY", "PRB_WEIGHT"]:
                base_score += 0.4
            elif action.type == "MCS_CAP":
                mcs_value = action.params.get("dl_mcs_max", 18)
                if mcs_value >= 20:  # Higher MCS for latency
                    base_score += 0.3
            elif action.type == "SLICE_QOS":
                priority = action.params.get("priority", 3)
                if priority >= 4:  # Higher priority for latency
                    base_score += 0.2
                    
        elif situation == "low_throughput":
            if action.type in ["PRB_WEIGHT", "MCS_CAP"]:
                base_score += 0.4
            elif action.type == "SCHEDULER_POLICY":
                policy = action.params.get("policy", "PF")
                if policy in ["PF", "MAX_THROUGHPUT"]:
                    base_score += 0.3
                    
        elif situation == "high_error_rate":
            if action.type == "MCS_CAP":
                mcs_value = action.params.get("dl_mcs_max", 18)
                if mcs_value <= 16:  # Lower MCS for reliability
                    base_score += 0.4
            elif action.type == "POWER_CONTROL":
                base_score += 0.3
                
        elif situation == "high_load":
            if action.type in ["ADMISSION_CONTROL", "PRB_WEIGHT"]:
                base_score += 0.4
            elif action.type == "SCHEDULER_POLICY":
                policy = action.params.get("policy", "PF")
                if policy == "PF":  # Fair scheduling for high load
                    base_score += 0.2
        
        # Apply context sensitivity multiplier
        if template:
            base_score *= template.context_sensitivity
        
        return min(1.0, max(0.0, base_score))

    def check_action_conflicts(self, action1: ControlAction, action2: ControlAction) -> bool:
        """Check if two actions conflict with each other."""
        # Same type and scope conflicts
        if (action1.type == action2.type and 
            action1.scope == action2.scope):
            
            if action1.scope == "CELL" and action1.cell_id == action2.cell_id:
                return True
            elif action1.scope == "SLICE" and action1.slice_id == action2.slice_id:
                return True
            elif action1.scope == "UE" and action1.ue_id == action2.ue_id:
                return True
        
        # Cross-type conflicts from conflict matrix
        conflict_key = (action1.type, action2.type)
        if self.conflict_matrix.get(conflict_key, False):
            # Check if they affect same entity
            if (action1.scope == action2.scope and 
                getattr(action1, f"{action1.scope.lower()}_id", None) == 
                getattr(action2, f"{action2.scope.lower()}_id", None)):
                return True
        
        return False

    def validate_playbook(self, playbook: Playbook) -> Tuple[bool, List[str]]:
        """Validate a playbook for conflicts and constraint violations."""
        errors = []
        
        # Check for action conflicts
        for i, action1 in enumerate(playbook.actions):
            for j, action2 in enumerate(playbook.actions[i+1:], i+1):
                if self.check_action_conflicts(action1, action2):
                    errors.append(f"Conflict between action {i} ({action1.type}) and action {j} ({action2.type})")
        
        # Check constraint violations
        for i, action in enumerate(playbook.actions):
            template = self.action_templates.get(action.type)
            if template and not self._is_action_valid(action, template):
                errors.append(f"Action {i} ({action.type}) violates constraints")
        
        return len(errors) == 0, errors

    def get_action_impact_metrics(self, action: ControlAction) -> Dict[str, float]:
        """Get expected impact metrics for an action."""
        template = self.action_templates.get(action.type)
        if not template:
            return {}
        
        impact_map = {
            ActionImpact.LOW: 0.2,
            ActionImpact.MEDIUM: 0.5,
            ActionImpact.HIGH: 0.8,
            ActionImpact.CRITICAL: 1.0
        }
        
        base_impact = impact_map[template.impact_level]
        
        return {
            "expected_impact": base_impact,
            "target_metrics": template.target_metrics,
            "impact_level": template.impact_level.value,
            "context_sensitivity": template.context_sensitivity
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the action space."""
        all_actions = self.get_all_actions()
        
        action_counts = {}
        for action in all_actions:
            action_counts[action.type] = action_counts.get(action.type, 0) + 1
        
        return {
            "total_actions": len(all_actions),
            "action_types": len(self.action_templates),
            "cells": len(self.cells),
            "slices": len(self.slices),
            "ues": len(self.ues),
            "actions_per_type": action_counts,
            "conflict_rules": len(self.conflict_matrix),
            "constraint_rules": len(self.constraint_rules)
        }


# -----------------------------
# Enhanced Playbook Class
# -----------------------------

@dataclass
class EnhancedPlaybook:
    """Enhanced playbook with contextual metadata."""
    actions: List[ControlAction] = field(default_factory=list)
    
    # Contextual metadata
    context_situation: str = "unknown"
    generation_method: str = "random"
    expected_impact: Dict[str, float] = field(default_factory=dict)
    confidence_score: float = 0.5
    q_score: float = 0.0
    
    # Execution metadata
    playbook_id: str = ""
    creation_timestamp: str = ""
    intent_reference: str = ""
    
    # Validation results
    is_valid: bool = True
    validation_errors: List[str] = field(default_factory=list)
    
    def to_basic_playbook(self) -> Playbook:
        """Convert to basic playbook for compatibility."""
        return Playbook(actions=self.actions)
    
    def add_metadata(self, **kwargs):
        """Add arbitrary metadata to the playbook."""
        for key, value in kwargs.items():
            setattr(self, key, value)


# -----------------------------
# Utility Functions
# -----------------------------

def create_default_action_space() -> ContextualActionSpace:
    """Create a default contextual action space for demos."""
    return ContextualActionSpace(
        cells=["CELL_001", "CELL_002"],
        slices=["SLICE_A", "SLICE_B", "SLICE_C"],
        ues=[]  # Can be populated dynamically
    )

def analyze_action_space_coverage(action_space: ContextualActionSpace) -> Dict[str, Any]:
    """Analyze coverage of action space across different situations."""
    situations = ["high_latency", "low_throughput", "high_error_rate", "high_load", "normal"]
    coverage = {}
    
    for situation in situations:
        contextual_actions = action_space.get_contextual_actions(situation, top_k=50)
        coverage[situation] = {
            "action_count": len(contextual_actions),
            "action_types": len(set(a.type for a in contextual_actions)),
            "avg_relevance": np.mean([
                action_space._calculate_action_relevance(a, situation) 
                for a in contextual_actions[:10]  # Sample first 10
            ])
        }
    
    return coverage


if __name__ == "__main__":
    # Demo usage
    action_space = create_default_action_space()
    
    print("=== Contextual Action Space Demo ===")
    stats = action_space.get_statistics()
    print(f"Total actions: {stats['total_actions']}")
    print(f"Action types: {list(stats['actions_per_type'].keys())}")
    
    print("\n=== Contextual Actions for High Latency ===")
    high_lat_actions = action_space.get_contextual_actions("high_latency", top_k=5)
    for i, action in enumerate(high_lat_actions):
        impact = action_space.get_action_impact_metrics(action)
        print(f"{i+1}. {action.type} - {action.params} (impact: {impact['expected_impact']:.2f})")
    
    print("\n=== Action Space Coverage Analysis ===")
    coverage = analyze_action_space_coverage(action_space)
    for situation, data in coverage.items():
        print(f"{situation}: {data['action_count']} actions, avg relevance: {data['avg_relevance']:.3f}")