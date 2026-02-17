import asyncio
import json
import logging
from enum import Enum, auto
from typing import Dict
from ain.bus.mem import MemBus
from demo.ain.bus.messages import make_msg

logger = logging.getLogger(__name__)

class IntentState(Enum):
    MONITORING = auto()
    ACTIVATING = auto() 
    THINKING = auto()   
    ASSURANCE = auto()  
    WITHDRAWAL = auto()

class OrchestratorAgent:
    def __init__(self, bus: MemBus):
        self.bus = bus
        self.state = IntentState.MONITORING
        self.current_metrics = {} # Keep track of network state
        
        # Assurance Logic
        self.current_otm = None
        self.assurance_timer = 0
        self.ASSURANCE_DURATION = 20 # TODO: Do we even need assurance? It will need complete overhaul
        self.active_config = {"tx_power_dbm": 16.0, "dl_mcs_max": 28, "prb_weight": 1.0} # Keep track of state somehow until we have a database...

    async def run(self):
        q_kpi = await self.bus.sub("kpi.raw")
        q_dev = await self.bus.sub("deviation.detected")
        q_otm = await self.bus.sub("otm.created")
        q_user = await self.bus.sub("ui.input")

        logger.info("[CONTROLLER] HybridControllerAgent Online.")

        while True:
            while not q_kpi.empty():
                msg = q_kpi.get_nowait()
                payload = msg.payload.get("kpi", {})
                self.current_metrics.update(payload.get("CellMetrics", {}))

            while not q_user.empty():
                msg = q_user.get_nowait()
                text = msg.payload.get("text")
                logger.info(f"[CONTROLLER] Manual Request: {text}")
                self.state = IntentState.THINKING
                await self.bus.pub("ai.request", make_msg("ai.request", "REQ", "v1", {
                    "type": "manual",
                    "text": text,
                    "state": self.active_config 
                }))

            # Handle OTM Creation
            while not q_otm.empty():
                msg = q_otm.get_nowait()
                self.current_otm = msg.payload
                
                i_name = self.current_otm.get("metadata", {}).get("intent", "Optimization")
                # This is where the list is now:
                proc_queue = self.current_otm.get("metadata", {}).get("human_readable_procedure", [])

                # Log the procedure queue clearly
                logger.info(f"[CONTROLLER] Plan Received: '{i_name}'")
                logger.info(f"[CONTROLLER] Procedure Queue: {proc_queue}")

                self.state = IntentState.ASSURANCE
                self.assurance_timer = 0
                
                # Update UI
                await self.bus.pub("intent.current", make_msg("intent", "INFO", "v1", {
                    "intent_id": "active_intent",
                    "type": i_name,
                    "procedure_steps": proc_queue, # Send list to UI
                    "status": "Starting..."
                }))

            while not q_dev.empty():
                msg = q_dev.get_nowait()
                dev = msg.payload
                
                # GUARD: Only process if we are currently in Monitoring mode
                if self.state == IntentState.MONITORING:
                    if self.current_metrics:
                        logger.info(f"[CONTROLLER] Anomaly Accepted: {dev['metric']}")
                        self.state = IntentState.THINKING
                        
                        await self.bus.pub("ai.request", make_msg("ai.request", "REQ", "v1", {
                            "type": "anomaly",
                            "metric": dev['metric'],
                            "value": dev['value'],
                            "state": self.active_config
                        }))
                        await self.bus.pub("deviation.broadcast", make_msg("dev", "INFO", "v1", dev))
                        
                        # IMPORTANT: Exit the loop. We found our anomaly. 
                        # This prevents the "Acceptance Flood" you saw in your logs.
                        break 
                else:
                    # Busy processing something else; silently discard incoming queue items
                    pass

            if self.state == IntentState.ASSURANCE:                
                self.assurance_timer += 1
                
                if self.assurance_timer >= self.ASSURANCE_DURATION:
                    logger.info("[CONTROLLER] Assurance timer expired. Withdrawing.")
                    self.state = IntentState.WITHDRAWAL

            elif self.state == IntentState.WITHDRAWAL:
                # Cleanup Logic TODO: What to do in withdrawal..? Update database? For now just clear UI and reset state
                await self.bus.pub("ui.clear", make_msg("ui", "CMD", "v1", {}))
                self.current_otm = None
                self.state = IntentState.MONITORING

            await asyncio.sleep(0.5)