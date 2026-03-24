import asyncio
import json
import logging
from core.bus.messages import make_msg

logger = logging.getLogger(__name__)

class NetworkOptimizer:
    """
    Acts as the bridge between the Cognitive Core (LLM) and the RL Engine.
    Takes declarative OTMs, formats them for the UI, and pushes them to the RL loop.
    """
    def __init__(self, bus, action_space, predictor, cache):
        self.bus = bus
        self.action_space = action_space
        self.predictor = predictor
        self.cache = cache

    async def run(self):
        # Listen for OTMs from the Cognitive Agent
        q_target = await self.bus.sub("optimizer.target")
        logger.info("[OPTIMIZER] Online and listening for Cognitive OTMs.")
        
        while True:
            msg = await q_target.get()
            otm = msg.payload
            
            logger.info("==================================================")
            logger.info("[OPTIMIZER] NEW OTM RECEIVED FROM COGNITIVE CORE")
            logger.info(json.dumps(otm, indent=2))
            logger.info("==================================================")
            
            # Format the OTM for the UI Dashboard
            # The UI looks for 'type' and 'procedure_steps'
            ui_steps = []
            obj = otm.get("objective", {})
            if obj:
                direction = "Maximize" if obj.get("maximize") else "Minimize"
                ui_steps.append(f"Objective: {direction} {obj.get('kpi')}")
                
            for c in otm.get("constraints", []):
                ui_steps.append(f"Constraint: {c.get('kpi')} {c.get('operator')} {c.get('threshold')} {c.get('unit', '')}")
                
            otm["procedure_steps"] = ui_steps
            otm["type"] = f"LLM Optimization: {obj.get('kpi', 'Network')}"
            otm["intent_id"] = f"optimization_task_{obj.get('kpi', 'general')}"

            logger.info(f"[OPTIMIZER] Forwarding OTM to RL Engine -> intent.current (Trace: {msg.corr_id})")
            # Final hop to ensure the RL engine receives the same Trace ID
            await self.bus.pub("intent.current", make_msg("optimizer", "INTENT", "v1", otm, corr_id=msg.corr_id))