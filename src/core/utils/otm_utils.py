import uuid
from datetime import datetime
from typing import Dict, List, Any

class OTMGenerator:
    #Used by PolicyAgent to format JSON into 
    # 2026-02-17, not used anymore... but i'll keep all the code in case something similar needs to be done in the future.
    def create_otm_from_outcome(self, outcome: Dict, episode_id: str, current_state: Dict = None) -> Dict[str, Any]:
        seen_kpis = set()
        procedure = outcome.get("procedure", [])
        intent_name = outcome.get("intent_name", "throughput")
        
        state = current_state or {"tx_power_dbm": 16.0, "dl_mcs_max": 28, "prb_weight": 1.0}
        constraints = []

        for step in procedure:
            step_text = str(step).upper() # Uppercase for easier matching
            kpi_name = None
            
            # Which knob did the LLM select?
            if "MODIFY_TX_POWER" in step_text or "POWER" in step_text:
                kpi_name = "tx_power_dbm"
            elif "ADJUST_MCS_CAP" in step_text or "MCS" in step_text:
                kpi_name = "dl_mcs_max"
            elif "CHANGE_PRB_WEIGHT" in step_text or "WEIGHT" in step_text:
                kpi_name = "prb_weight"

            # Process the knob if we haven't seen it yet in this OTM
            if kpi_name and kpi_name not in seen_kpis:
                seen_kpis.add(kpi_name)
                
                should_decrease = any(x in step_text for x in ["DECREASE", "REDUCE", "LOWER"])
                
                threshold_val = 0.0
                unit = ""
                
                if kpi_name == "tx_power_dbm":
                    curr = state.get("tx_power_dbm", 16.0)
                    threshold_val = (curr * 0.8) if should_decrease else (curr * 1.2)
                    unit = "dBm"
                
                elif kpi_name == "dl_mcs_max":
                    curr = state.get("dl_mcs_max", 28)
                    threshold_val = int(curr * 0.8) if should_decrease else int(curr * 1.1)
                    unit = "index"

                elif kpi_name == "prb_weight":
                    curr = state.get("prb_weight", 1.0)
                    threshold_val = (curr * 0.8) if should_decrease else (curr * 1.0)
                    unit = "ratio"

                # Build the OTM constraint block -- TODO: Placeholder for now, needs more logic from how network optimizer works in future
                constraints.append({
                    "service": "mbb",
                    "kpi": kpi_name,
                    "operator": "le" if should_decrease else "ge",
                    "threshold": round(float(threshold_val), 2),
                    "aggregation": "min",
                    "unit": unit,
                    "scope": "per_cell_window",
                    "origin": "Qwen2.5-7B-Instruct",
                    "adapted_by": "ICL_LLM",
                    "id": f"C{len(constraints) + 1}"
                })

        return {
            "version": "1.0",
            "objective": {
                "service": "mbb",
                "kpi": "throughput",
                "aggregation": "mean",
                "unit": "Mbps",
                "maximize": True
            },
            "constraints": constraints,
            "metadata": {
                "timescale": "10s_window",
                "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "episode": episode_id,
                "adaptation_log": []
            }
        }


class OTMToCommandConverter:
    # Used by ActuatorAgent to format OTM into xApp Commands.
    # In the future might want some sort of Network Optimizer that figures out the ideal values of the parameters
    # based on the declarative OTM. 
    @staticmethod
    def convert(otm: Dict, meid: str, node_id: int) -> List[Dict[str, Any]]: # TODO: node_id needs to be fixed...
        commands = []
        for constraint in otm.get("constraints", []):
            kpi_id = constraint["kpi"] 
            val = constraint["threshold"] 

        # TODO: Actually implement an optimizer agent somehow. Need to discuss how.
        # Currently these fields are just hardcoded as a proof of concept.

        cmd_body = None
        if kpi_id == "dl_mcs_max":
            cmd_body = {"cmd": "set-mcs", "node": node_id, "mcs": int(val)}
        elif kpi_id == "tx_power_dbm":
            cmd_body = {"cmd": "set-enb-txpower", "node": node_id, "txPowerDbm": float(val)}
        elif kpi_id == "prb_weight":
            cmd_body = {"cmd": "set-bandwidth", "node": node_id, "bandwidth": int(100 * val)}
        
        if cmd_body:
            commands.append({
                "type": "control",
                "meid": meid,
                "cmd": cmd_body
            })
        return commands