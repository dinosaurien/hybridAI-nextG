import asyncio
import time
import logging
from enum import Enum, auto
from core.bus.mem import MemBus
from core.bus.messages import make_msg
from core.utils.knowledge_base import KnowledgeBase 

logger = logging.getLogger(__name__)

class IntentState(Enum):
    MONITORING = auto()
    THINKING = auto()   
    ASSURANCE = auto()  # 30s grace period to let RL settle

class OrchestratorAgent:
    def __init__(self, bus: MemBus, kb: KnowledgeBase):
        self.bus = bus
        self.kb = kb
        self.state = IntentState.MONITORING
        self.assurance_end_time = 0.0
        self.active_otm = None
        self.active_anomaly_metric = None

    async def run(self):
        # Subscribe to all necessary channels
        q_dev = await self.bus.sub("deviation.detected")
        q_otm = await self.bus.sub("optimizer.target") 
        q_ui  = await self.bus.sub("ui.input")

        logger.info("[ORCHESTRATOR] Orchestrator Online. Monitoring for Anomalies & Manual Intents.")

        while True:
            current_time = time.time()

            if self.state == IntentState.ASSURANCE and current_time >= self.assurance_end_time:
                logger.info("[ORCHESTRATOR] Assurance window expired. Returning to MONITORING.")
                self.state = IntentState.MONITORING

            while not q_ui.empty():
                msg = q_ui.get_nowait()
                text = msg.payload.get("text", "").upper()
                
                # buttons that break sim
                if "BREAK_SIM" in text:
                    action = "trigger-traffic-spike" if "LATENCY" in text else "trigger-blockage"
                    logger.info(f"[ORCHESTRATOR] Routing manual chaos command: {text} -> {action}")
                    await self.bus.pub("sim.control", make_msg("orch", "CONTROL", "v1", {"action": action}))
                
                else:
                    # Normal natural language intent
                    logger.info(f"[ORCHESTRATOR] Manual User Intent: '{text}' -> Sending to AI")
                    await self.bus.pub("ai.request", make_msg("ai.request", "REQ", "v1", {
                        "type": "manual", "text": text
                    }))

            while not q_dev.empty():
                msg = q_dev.get_nowait()
                dev = msg.payload
                metric = dev['metric']
                
                if self.state == IntentState.MONITORING:
                    logger.info(f"[ORCHESTRATOR] Anomaly Detected: {metric}.")
                    self.state = IntentState.THINKING
                    await self.bus.pub("deviation.broadcast", make_msg("orch", "DEV", "v1", dev))
                    
                    if self.active_otm and self.active_anomaly_metric == metric:
                        logger.warning(f"[ORCHESTRATOR] Persistent failure. Requesting ADAPTATION.")
                        await self.bus.pub("ai.request", make_msg("ai.request", "REQ", "v1", {
                            "type": "adapt_otm", "metric": metric, "value": dev['value'], "previous_otm": self.active_otm
                        }))
                    else:
                        logger.info(f"[ORCHESTRATOR] Requesting NEW OTM from Cognitive Core.")
                        self.active_anomaly_metric = metric
                        await self.bus.pub("ai.request", make_msg("ai.request", "REQ", "v1", {
                            "type": "generate_otm", "metric": metric, "value": dev['value']
                        }))
                    break # Stop processing other anomalies while the AI is thinking
                else:
                    logger.debug(f"[ORCHESTRATOR] Busy. Ignoring anomaly on {metric}")

            # 4. Listen for OTMs to start the Assurance Window
            while not q_otm.empty():
                msg = q_otm.get_nowait()
                self.active_otm = msg.payload
                self.state = IntentState.ASSURANCE
                self.assurance_end_time = time.time() + 30.0
                logger.info(f"[ORCHESTRATOR] OTM Received. Entering 30s ASSURANCE window.")

            await asyncio.sleep(0.2)