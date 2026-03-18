import asyncio
import logging
import concurrent.futures
from core.bus.messages import make_msg

logger = logging.getLogger(__name__)

class ActuatorAgent:
    def __init__(self, bus):
        self.bus = bus
        self.trigger_file = "/tmp/chaos_trigger.txt"
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        
        self.protocol_map = {
            "trigger-blockage": "BLOCKAGE",
            "trigger-traffic-spike": "SPIKE"
        }

    async def run(self):
        q_sim = await self.bus.sub("sim.control")
        q_actor = await self.bus.sub("actor.apply")
        
        loop = asyncio.get_event_loop()
        logger.info("[ACTUATOR] Bridge established: Bus → Simulation File Interface & E2 Relay")

        while True:
            while not q_sim.empty():
                msg = q_sim.get_nowait()
                action = msg.payload.get("action")
                
                if action in self.protocol_map:
                    keyword = self.protocol_map[action]
                    await loop.run_in_executor(self.executor, self._write_to_sim, keyword)
                    logger.info(f"[ACTUATOR] Orchestrator Intent '{action}' → Translated to Chaos Protocol '{keyword}'")

            while not q_actor.empty():
                msg = q_actor.get_nowait()
                playbook = msg.payload.get("playbook")
                
                if hasattr(playbook, "actions"):
                    for action in playbook.actions:

                        if action.type == "REPORTING":
                            continue
    
                        cmd_body = self._translate_rl_action(action)
                        if cmd_body:
                            logger.info(f"[ACTUATOR] Translating RL Action {action.type} -> E2 Command: {cmd_body}")
                            
                            await self.bus.pub("xapp.control", make_msg("actuator", "CMD", "v1", cmd_body))
            
            await asyncio.sleep(0.1)

    def _translate_rl_action(self, action):
        """Translates abstract RL ActionSpace types into strict xApp/ns-3 E2 commands."""
        cmd_body = None
        
        #TODO: are we going to do multi-node?
        node_id = 1 
        
        if action.type == "MCS_CAP":
            val = action.params.get("dl_mcs_max", 28)
            cmd_body = {"cmd": "set-mcs", "node": node_id, "mcs": int(val)}
            
        elif action.type in ["TX_POWER", "POWER_CONTROL"]:
            val = action.params.get("txPowerDbm", action.params.get("tx_power_dbm", 10.0))
            cmd_body = {"cmd": "set-enb-txpower", "node": node_id, "txPowerDbm": float(val)}
            
        elif action.type == "PRB_WEIGHT":
            val = action.params.get("weight", 1.0)
            cmd_body = {"cmd": "set-bandwidth", "node": node_id, "bandwidth": int(100 * val)}
            
        return cmd_body

    def _write_to_sim(self, keyword):
        with open(self.trigger_file, "w") as f:
            f.write(keyword)