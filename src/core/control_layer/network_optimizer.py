import copy
import json
import logging
from core.bus.messages import make_msg

logger = logging.getLogger(__name__)

class NetworkOptimizer:
    """
    Acts as the bridge between the Cognitive Core (LLM) and the RL Engine.
    Takes declarative OTMs, formats them for the UI, and pushes them to the RL loop.
    """
    def __init__(self, bus):
        self.bus = bus

    async def run(self):
        # Listen for OTMs from the Cognitive Agent or Orchestrator
        q_target = await self.bus.sub("intent.execute")
        logger.info("[OPTIMIZER] Online and listening for Orchestrator execution commands.")
        
        while True:
            msg = await q_target.get()
            otm = msg.payload
            
            # scrub previous UI metadata to prevent nesting effect where raw_otm is saved insidew the metadata of the next OTM
            logical_otm = copy.deepcopy(otm)
            ui_keys = (
                "raw_otm", "procedure_steps", "type", "intent_id", 
                "scheduled", "is_procedure_step", "activate_at", "deactivate_at"
            )
            for key in ui_keys:
                logical_otm.pop(key, None)

            # prepare UI display data
            ui_steps = []
            metadata = logical_otm.get("metadata", {})
            obj = logical_otm.get("objective", {})
            
            # Check if this is a failure autopsy from the Orchestrator
            is_fallback = metadata.get("is_critical_fallback", False)

            if is_fallback:
                display_type = "CRITICAL: SYSTEM FALLBACK (LLM FAILED)"
                display_id = "CRITICAL_FALLBACK"
                
                # Push the autopsy log into the UI view
                for log_entry in metadata.get("adaptation_log", []):
                    ui_steps.append(log_entry)
                    
                # Highlight the safe mode constraints currently in effect
                for c in logical_otm.get("constraints", []):
                    ui_steps.append(f"Safe Mode Enforced: {c.get('kpi')} {c.get('operator')} {c.get('threshold')} {c.get('unit', '')}")
            else:
                # Normal OTM display path
                direction = "Maximize" if obj.get("maximize") else "Minimize"
                ui_steps.append(f"Objective: {direction} {obj.get('kpi')}")
                
                for c in logical_otm.get("constraints", []):
                    ui_steps.append(f"Constraint: {c.get('kpi')} {c.get('operator')} {c.get('threshold')} {c.get('unit', '')}")
                
                display_type = f"LLM Optimization: {obj.get('kpi', 'Network')}"
                display_id = f"optimization_task_{obj.get('kpi', 'general')}"

                # Forward adaptation trail for the UI
                adaptation_log = metadata.get("adaptation_log", [])
                if adaptation_log:
                    latest = adaptation_log[-1]
                    if "merge" in latest.lower():
                        ui_steps.append(f"Merged Intent: {latest}")
                    elif "adapt" in latest.lower():
                        ui_steps.append(f"Adapted: {latest}")

            # Construct final payload
            # merge the clean logical OTM with the UI-only fields
            final_payload = copy.deepcopy(logical_otm)
            final_payload["type"] = display_type
            final_payload["intent_id"] = display_id
            final_payload["procedure_steps"] = ui_steps
            
            # Handle Scheduling metadata for UI
            temporal = metadata.get("temporal_resolved", {})
            if temporal.get("has_schedule"):
                final_payload["scheduled"] = True
                final_payload["activate_at"] = temporal.get("activate_at_utc")
                final_payload["deactivate_at"] = temporal.get("deactivate_at_utc")

            # Handle Procedure metadata for UI
            procedure_id = metadata.get("procedure_id")
            if procedure_id:
                final_payload["procedure_id"] = procedure_id
                final_payload["is_procedure_step"] = True

            # Stringified JSON for display (The UI bridge will use this)
            # We stringify the 'logical_otm' so the UI shows the clean version 
            # without these display-only fields.
            final_payload["raw_otm"] = json.dumps(logical_otm, indent=2)

            logger.info(f"[OPTIMIZER] Forwarding cleaned OTM to RL Engine (Trace: {msg.corr_id})")
            
            # Final hop to the RL engine and UI bridge
            await self.bus.pub("intent.current", make_msg("optimizer", "INTENT", "v1", final_payload, corr_id=msg.corr_id))