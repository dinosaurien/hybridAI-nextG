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
    THINKING_TIMEOUT = 60.0  # seconds before assuming LLM failed

    def __init__(self, bus: MemBus, kb: KnowledgeBase):
        self.bus = bus
        self.kb = kb
        self.state = IntentState.MONITORING
        self.thinking_start_time = 0.0
        self.assurance_end_time = 0.0
        self.active_otm = None
        self.active_anomaly_metric = None
        self.pending_manual_intents = []

    async def run(self):
        q_dev = await self.bus.sub("deviation.detected")
        q_otm = await self.bus.sub("optimizer.target")
        q_ui  = await self.bus.sub("ui.input")

        logger.info("[ORCHESTRATOR] Orchestrator Online. Monitoring for Anomalies & Manual Intents.")

        while True:
            current_time = time.time()

            # Timeout: if LLM hasn't responded, unlock the loop
            if self.state == IntentState.THINKING and current_time >= self.thinking_start_time + self.THINKING_TIMEOUT:
                logger.warning("[ORCHESTRATOR] THINKING timeout reached — Cognitive Core did not respond. Returning to MONITORING.")
                self.state = IntentState.MONITORING

            if self.state == IntentState.ASSURANCE and current_time >= self.assurance_end_time:
                logger.info("[ORCHESTRATOR] Assurance window expired. Returning to MONITORING.")
                self.state = IntentState.MONITORING

            while not q_ui.empty():
                msg = q_ui.get_nowait()
                text = msg.payload.get("text", "").upper()

                if "BREAK_SIM" in text:
                    action = "trigger-traffic-spike" if "LATENCY" in text else "trigger-blockage"
                    logger.info(f"[ORCHESTRATOR] Routing manual chaos command: {text} -> {action}")
                    await self.bus.pub("sim.control", make_msg("orch", "CONTROL", "v1", {"action": action}))
                else:
                    # Only process manual intents if we are idle (MONITORING).
                    # This prevents race conditions where the LLM tries to process a manual 
                    # intent and an anomaly intent at the exact same time.
                    if self.state == IntentState.MONITORING:
                        logger.info(f"[ORCHESTRATOR] Manual User Intent: '{text}' -> Sending to AI")
                        self.state = IntentState.THINKING
                        self.thinking_start_time = current_time
                        await self.bus.pub("ai.request", make_msg("ai.request", "REQ", "v1", {
                            "type": "manual", "text": text
                        }))
                    else:
                        # Queue the intent to be processed once we return to MONITORING.
                        logger.info(f"[ORCHESTRATOR] Busy ({self.state.name}). Queuing manual intent: '{text}'")
                        self.pending_manual_intents.append(text)
                
            # Process queued manual intents the moment we become idle.
            # Placed BEFORE the anomaly queue (q_dev) so human requests take priority.
            if self.state == IntentState.MONITORING and self.pending_manual_intents:
                queued_text = self.pending_manual_intents.pop(0)
                logger.info(f"[ORCHESTRATOR] Processing queued User Intent: '{queued_text}'")
                self.state = IntentState.THINKING
                self.thinking_start_time = current_time
                await self.bus.pub("ai.request", make_msg("ai.request", "REQ", "v1", {
                    "type": "manual", "text": queued_text
                }))


            while not q_dev.empty():
                msg = q_dev.get_nowait()
                dev = msg.payload
                metric = dev['metric']

                if self.state == IntentState.MONITORING:
                    logger.info(f"[ORCHESTRATOR] Anomaly Detected: {metric}.")
                    self.state = IntentState.THINKING
                    self.thinking_start_time = current_time

                    await self.bus.pub("deviation.broadcast", make_msg("orch", "DEV", "v1", dev, corr_id=msg.corr_id))

                    if self.active_otm and self.active_anomaly_metric == metric:
                        logger.warning(f"[ORCHESTRATOR] Persistent failure. Requesting ADAPTATION. (Trace: {msg.corr_id})")
                        await self.bus.pub("ai.request", make_msg("ai.request", "REQ", "v1", {
                            "type": "adapt_otm", "metric": metric, "value": dev['value'], "previous_otm": self.active_otm
                        }, corr_id=msg.corr_id))
                    else:
                        logger.info(f"[ORCHESTRATOR] Requesting NEW OTM from Cognitive Core. (Trace: {msg.corr_id})")
                        self.active_anomaly_metric = metric
                        await self.bus.pub("ai.request", make_msg("ai.request", "REQ", "v1", {
                            "type": "generate_otm", "metric": metric, "value": dev['value']
                        }, corr_id=msg.corr_id))
                    break
                else:
                    logger.debug(f"[ORCHESTRATOR] Busy ({self.state.name}). Ignoring anomaly on {metric}")

            while not q_otm.empty():
                msg = q_otm.get_nowait()
                if msg.type == "LLM_FAILURE":
                    logger.warning(f"[ORCHESTRATOR] Cognitive Core reported failure: {msg.payload.get('error')}. Returning to MONITORING.")
                    self.state = IntentState.MONITORING
                else:
                    self.active_otm = msg.payload
                    self.state = IntentState.ASSURANCE
                    self.assurance_end_time = time.time() + 30.0
                    logger.info(f"[ORCHESTRATOR] OTM Received. Entering 30s ASSURANCE window.")

            await asyncio.sleep(0.2)