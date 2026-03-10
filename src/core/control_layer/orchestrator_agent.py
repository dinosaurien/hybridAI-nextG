import asyncio
import datetime
import json
import logging
import aiohttp
import uuid
from enum import Enum, auto
from typing import Dict
from core.bus.mem import MemBus
from core.bus.messages import make_msg

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
        self.current_metrics = {} 
        self.current_otm = None
        self.assurance_timer = 0
        self.ASSURANCE_DURATION = 20 
        self.active_config = {"tx_power_dbm": 16.0, "dl_mcs_max": 28, "prb_weight": 1.0}

    async def run(self):
        q_kpi = await self.bus.sub("kpi.raw")
        q_dev = await self.bus.sub("deviation.detected")
        q_otm = await self.bus.sub("otm.created")
        q_user = await self.bus.sub("ui.input")

        logger.info("[CONTROLLER] HybridControllerAgent Online.")

        while True:
            # 1. POLITE KPI DRAIN (Max 100) - Prevents Event Loop Starvation
            for _ in range(100):
                if q_kpi.empty(): break
                msg = q_kpi.get_nowait()
                payload = msg.payload.get("kpi", {})
                self.current_metrics.update(payload.get("CellMetrics", {}))

            while not q_user.empty():
                msg = q_user.get_nowait()
                text = msg.payload.get("text", "").upper()
                
                # Map Buttons to Bus Events
                if "BREAK_SIM: LATENCY_SPIKE" in text:
                    logger.info("[ORCHESTRATOR] Routing Spike to SimControl")
                    await self.bus.pub("sim.control", make_msg("sim", "CMD", "v1", {"action": "trigger-traffic-spike"}))
                
                elif "BREAK_SIM: PACKET_LOSS" in text:
                    logger.info("[ORCHESTRATOR] Routing Blockage to SimControl")
                    await self.bus.pub("sim.control", make_msg("sim", "CMD", "v1", {"action": "trigger-blockage"}))
                
                # Standard AI Intents
                else:
                    logger.info(f"[ORCHESTRATOR] Manual Request: {text}")
                    self.state = IntentState.THINKING
                    await self.bus.pub("ai.request", make_msg("ai.request", "REQ", "v1", {
                        "type": "manual", "text": text, "state": self.active_config 
                }))

            # 3. DRAIN PLAN/OTM CREATIONS
            while not q_otm.empty():
                msg = q_otm.get_nowait()
                self.current_otm = msg.payload
                i_name = self.current_otm.get("metadata", {}).get("intent", "Optimization")
                proc_queue = self.current_otm.get("metadata", {}).get("human_readable_procedure", [])
                
                logger.info(f"[CONTROLLER] Plan Received: '{i_name}'")
                self.state = IntentState.ASSURANCE
                self.assurance_timer = 0
                
                await self.bus.pub("intent.current", make_msg("intent", "INFO", "v1", {
                    "intent_id": "active_intent", "type": i_name,
                    "procedure_steps": proc_queue, "status": "Starting..."
                }))

            # 4. DRAIN ANOMALIES (Moved out of the OTM loop!)
            while not q_dev.empty():
                msg = q_dev.get_nowait()
                dev = msg.payload
                if self.state == IntentState.MONITORING and self.current_metrics:
                    logger.info(f"[CONTROLLER] Anomaly Accepted: {dev['metric']}. Sending to Cognitive Core...")
                    
                    # Create Turtle Fact
                    fact_id = f"urn:uuid:kpi-alert-{uuid.uuid4()}"
                    turtle_payload = f"""
                    @prefix r: <http://ontology.cf.ericsson.net/reasoner/> .
                    @prefix : <http://hybridai.org/ontology#> .

                    <{fact_id}> a r:Message ;
                    r:to r:KnowledgeBase ;
                    r:subject "assert" ;
                    r:payload [
                        a :KPIDeviation ;
                        :source :MiniRocket ;
                        :kpiStream "UE_DRB_PdcpSduDelayDl_UEID" ;
                        :deviationValue "136.75" ;
                        :timestamp "2026-03-09T11:41:13Z"
                    ] .
                    """
                    # Fire-and-forget task to AWS/CC
                    asyncio.create_task(self._send_to_cognitive_core(turtle_payload))
                    self.state = IntentState.THINKING
                    break 

            # 5. STATE MACHINE LOGIC (Fixed Indentation!)
            if self.state == IntentState.ASSURANCE:                
                self.assurance_timer += 1
                if self.assurance_timer >= self.ASSURANCE_DURATION:
                    logger.info("[CONTROLLER] Assurance timer expired. Withdrawing.")
                    self.state = IntentState.WITHDRAWAL

            elif self.state == IntentState.WITHDRAWAL:
                await self.bus.pub("ui.clear", make_msg("ui", "CMD", "v1", {}))
                self.current_otm = None
                self.state = IntentState.MONITORING

            # 6. CRITICAL: THE TICK (Yields control to the WebServer)
            await asyncio.sleep(0.5)
    
    async def auto_recovery_timer(self, delay_seconds):
        """Waits for X seconds, then restores the simulation to normal."""
        await asyncio.sleep(delay_seconds)
        logger.info(f"🕒 [SYSTEM] {delay_seconds}s elapsed. Auto-recovering simulation.")
        # -1 tells ns-3 to restore adaptive MCS
        await self.bus.pub("sim.control", make_msg("sim", "CMD", "v1", {
            "action": "set-mcs", "value": -1 
        }))

    async def _send_to_cognitive_core(self, payload: str):
        cc_url = "http://localhost:3020/cc" # Tunnel address
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(cc_url, data=payload, headers={"Content-Type": "text/turtle"}) as resp:
                    logger.info(f"[COGNITIVE CORE] Injected fact. Status: {resp.status}")
        except Exception as e:
            logger.error(f"[COGNITIVE CORE] Failed to connect to reasoner: {e}")