import uuid
from typing import Dict, List, Any

class OTMGenerator:
    def create_otm_from_outcome(self, outcome: Dict, episode_id: str) -> Dict[str, Any]:
        """
        Creates a single OTM containing multiple directives based on the LLM's procedure.
        """
        procedure = outcome.get("procedure", [])
        scope = outcome.get("scope", "CELL")
        intent_name = outcome.get("intent_name", "Network_Optimization")
        
        directives = []
        
        # We iterate through what the LLM suggested
        for step in procedure:
            # Handle both formats: simple string or object {"action": "..."}
            action_str = step.get("action", "") if isinstance(step, dict) else str(step)
            
            action_type = None
            params = {}

            if "MODIFY_TX_POWER" in action_str:
                action_type = "TX_POWER"
                # If LLM says "Reduce", we set a lower value, otherwise standard
                params = {"tx_power_dbm": 12.0 if "reduce" in str(step).lower() else 16.0}
            
            elif "ADJUST_MCS_CAP" in action_str:
                action_type = "MCS_CAP"
                # If LLM says "Decrease", set a lower cap
                params = {"dl_mcs_max": 16 if "decrease" in str(step).lower() else 28}
            
            elif "CHANGE_PRB_WEIGHT" in action_str:
                action_type = "PRB_WEIGHT"
                params = {"weight": 0.8 if "reduce" in str(step).lower() else 1.0}

            if action_type:
                directives.append({
                    "type": action_type,
                    "scope": scope,
                    "params": params,
                    "id": f"act_{uuid.uuid4().hex[:4]}"
                })

        # If LLM didn't suggest anything valid, default to a safe PRB weight
        if not directives:
            directives.append({
                "type": "PRB_WEIGHT",
                "scope": scope,
                "params": {"weight": 1.0},
                "id": "default_act"
            })

        return {
            "version": "1.0",
            "metadata": {
                "episode": episode_id,
                "intent": intent_name,
                "timestamp": str(uuid.uuid4())
            },
            "directives": directives
        }

    def create_cleanup_otm(self, original_scope: str) -> Dict[str, Any]:
        return {
            "version": "1.0",
            "metadata": {"episode": "cleanup", "intent": "Reset", "timestamp": str(uuid.uuid4())},
            "directives": [
                {"type": "PRB_WEIGHT", "scope": original_scope, "params": {"weight": 1.0}, "id": "reset_prb"},
                {"type": "MCS_CAP", "scope": original_scope, "params": {"dl_mcs_max": 28}, "id": "reset_mcs"},
                {"type": "TX_POWER", "scope": original_scope, "params": {"tx_power_dbm": 16.0}, "id": "reset_pow"}
            ]
        }

class OTMToCommandConverter:
    @staticmethod
    def convert(otm: Dict, meid: str, default_node_id: int = 2) -> List[Dict[str, Any]]:
        commands = []
        for action in otm.get("directives", []):
            a_type = action["type"]
            params = action["params"]
            node = action.get("node", default_node_id)
            
            cmd = None
            if a_type == "MCS_CAP":
                cmd = {"type": "control", "meid": meid, "cmd": {"cmd": "set-mcs", "node": node, "mcs": int(params["dl_mcs_max"])}}
            elif a_type == "TX_POWER":
                cmd = {"type": "control", "meid": meid, "cmd": {"cmd": "set-enb-txpower", "node": node, "txPowerDbm": float(params["tx_power_dbm"])}}
            elif a_type == "PRB_WEIGHT":
                cmd = {"type": "control", "meid": meid, "cmd": {"cmd": "set-bandwidth", "node": node, "bandwidth": int(100 * params["weight"])}}
            
            if cmd:
                # Attach UE ID if scope is UE
                if action["scope"] == "UE" and action.get("ue_id"):
                    cmd["cmd"]["ue_id"] = action["ue_id"]
                commands.append(cmd)
        return commands