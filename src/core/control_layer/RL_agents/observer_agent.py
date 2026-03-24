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
                msg = q_intent.get_nowait()
                otm = msg.payload
                self.engine.active_otm = otm
                
                # Sync the Observer's underlying Intent dataclass with the OTM
                # This ensures feature completeness checks and fallback rewards use the right metric
                obj = otm.get("objective", {})
                if obj:
                    raw_metric = obj.get("kpi", "latency").lower()
                    
                    # Map semantic KPI to actual CSV metric
                    metric = "UE_DRB_UEThpDl_UEID" if "thp" in raw_metric else "DRB_PdcpSduDelayDl"
                    direction = "higher_better" if obj.get("maximize", True) else "lower_better"
                    target = 50000000.0 if "thp" in raw_metric else 40.0
                    
                    self.engine.intent.metric = metric
                    self.engine.intent.direction = direction
                    self.engine.intent.target = target
                    self.engine.intent.type = f"OPTIMIZE_{raw_metric.upper()}"

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