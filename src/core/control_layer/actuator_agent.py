import asyncio
import logging
import concurrent.futures

logger = logging.getLogger(__name__)

class ActuatorAgent:
    def __init__(self, bus):
        self.bus = bus
        self.trigger_file = "/tmp/chaos_trigger.txt"
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        
        # This maps the Orchestrator's internal bus command 
        # to the physical protocol keywords expected by the C++ simulator
        self.protocol_map = {
            "trigger-blockage": "BLOCKAGE",
            "trigger-traffic-spike": "SPIKE"
        }

    async def run(self):
        q_sim = await self.bus.sub("sim.control")
        loop = asyncio.get_event_loop()
        logger.info("[ACTUATOR] Bridge established: Bus → Simulation File Interface")

        while True:
            msg = await q_sim.get()
            action = msg.payload.get("action")
            
            # Logic: Translate and Execute
            if action in self.protocol_map:
                keyword = self.protocol_map[action]
                
                # Non-blocking write to simulation file
                await loop.run_in_executor(self.executor, self._write_to_sim, keyword)
                
                logger.info(f"📡 [ACTUATOR] Orchestrator Intent '{action}' → Translated to Protocol '{keyword}'")
            
            elif action in ["set-mcs", "set-bandwidth"]:
                await self.send_to_relay(msg.payload)
                logger.info(f"📡 [ACTUATOR] Forwarded '{action}' to E2 Relay")

    def _write_to_sim(self, keyword):
        """Translates logic to the physical file trigger."""
        with open(self.trigger_file, "w") as f:
            f.write(keyword)

    async def send_to_relay(self, payload):
        # Implementation for relay server (port 5002)
        pass