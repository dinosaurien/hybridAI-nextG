import asyncio
import logging
from core.bus.messages import make_msg

logger = logging.getLogger(__name__)

class ActuatorAgent:
    def __init__(self, bus, cell_to_node_map=None):
        self.bus = bus
        self.cell_to_node_map = cell_to_node_map or {}

        self.protocol_map = {
            "trigger-blockage": "BLOCKAGE",
            "trigger-traffic-spike": "SPIKE"
        }

    async def run(self):
        q_sim = await self.bus.sub("sim.control")
        q_actor = await self.bus.sub("actor.apply")

        logger.info("[ACTUATOR] Bridge established: Bus → E2 Relay (chaos + control commands)")

        while True:
            while not q_sim.empty():
                msg = q_sim.get_nowait()
                action = msg.payload.get("action")

                if action in self.protocol_map:
                    keyword = self.protocol_map[action]
                    # Send chaos event over the E2 TCP connection instead of writing to a file
                    await self.bus.pub("xapp.control", make_msg(
                        "actuator", "CHAOS", "v1",
                        {"cmd": "chaos", "event": keyword}
                    ))
                    logger.info(f"[ACTUATOR] Orchestrator Intent '{action}' → Chaos command '{keyword}' via E2")

            while not q_actor.empty():
                msg = q_actor.get_nowait()
                playbook = msg.payload.get("playbook")

                if hasattr(playbook, "actions"):
                    for action in playbook.actions:

                        if action.type == "REPORTING":
                            continue

                        cmd_body = self._translate_action(action)
                        if cmd_body:
                            logger.info(f"[ACTUATOR] Translating {action.type} -> E2 Command: {cmd_body}")
                            await self.bus.pub("xapp.control", make_msg("actuator", "CMD", "v1", cmd_body))
                            # Publish E2 ack so the Orchestrator can verify commands were dispatched
                            await self.bus.pub("command.notify", make_msg(
                                "actuator", "E2_ACK", "v1",
                                {"command": action.type, "params": action.params, "e2_cmd": cmd_body}
                            ))

            await asyncio.sleep(0.1)

    def _resolve_node_id(self, action) -> int:
        """Derive node_id from the action's cell_id, falling back to heuristic."""
        cell_id = action.cell_id
        if cell_id and cell_id in self.cell_to_node_map:
            return self.cell_to_node_map[cell_id]
            
        # Hard fallback for the specific ns-3 topology
        return 2

    def _translate_action(self, action):
        """Translate abstract action types into strict xApp/ns-3 E2 commands."""
        cmd_body = None
        node_id = self._resolve_node_id(action)

        if action.type == "MCS_CAP":
            val = action.params.get("dl_mcs_max", 28)
            cmd_body = {"cmd": "set-mcs", "node": node_id, "mcs": int(val)}

        elif action.type in ["TX_POWER", "POWER_CONTROL"]:
            val = action.params.get("txPowerDbm", action.params.get("tx_power_dbm", 10.0))
            cmd_body = {"cmd": "set-enb-txpower", "node": node_id, "txPowerDbm": float(val)}

        return cmd_body
