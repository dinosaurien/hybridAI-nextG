import asyncio
from core.bus.messages import make_msg
from core.control_layer.RL_engines.observer_rl import RLObserver as RewardEngine

class ObserverAgent:
    def __init__(self, bus, engine: RewardEngine):
        self.bus = bus
        self.engine = engine

    async def run(self):
        q_kpi = await self.bus.sub("kpi.raw")
        q_intent = await self.bus.sub("intent.current")
        q_actor = await self.bus.sub("actor.apply")
        last_pb = None

        while True:
            while not q_actor.empty():
                last_pb = q_actor.get_nowait().payload.get("playbook")

            while not q_intent.empty():
                self.engine.active_otm = q_intent.get_nowait().payload

            msg = await q_kpi.get()
            
            actual_kpi_data = msg.payload.get("kpi", msg.payload)
            
            state_tensor = self.engine.step(last_pb, kpi_dict=actual_kpi_data)
            
            if state_tensor is not None:
                await self.bus.pub(
                    "kpi.window",
                    make_msg("kpi.window", "STATE_WINDOW", "kpi.window.v1", {
                        "state": state_tensor.tolist(), 
                        "reward": getattr(self.engine, 'last_reward', 0.0),
                        "situation": self.engine.get_context_situation()
                    })
                )