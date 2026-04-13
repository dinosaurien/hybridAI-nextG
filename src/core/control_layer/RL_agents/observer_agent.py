import asyncio
import functools
import time
from core.bus.messages import make_msg
from core.common.types import map_otm_metric
from core.control_layer.RL_engines.observer_rl import RLObserver as RewardEngine

class ObserverAgent:
    def __init__(self, bus, engine: RewardEngine, min_action_interval: float = 5.0):
        self.bus = bus
        self.engine = engine
        self.min_action_interval = min_action_interval
        self._last_action_time = 0.0

    async def run(self):
        q_kpi = await self.bus.sub("kpi.raw")
        q_intent = await self.bus.sub("intent.current")
        q_actor = await self.bus.sub("actor.apply")
        last_pb = None
        loop = asyncio.get_running_loop()

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
                    raw_kpi = obj.get("kpi", "latency")
                    metric, direction, default_target = map_otm_metric(raw_kpi)

                    # OTM maximize flag overrides the table default when explicitly set
                    if "maximize" in obj:
                        direction = "higher_better" if obj["maximize"] else "lower_better"

                    self.engine.intent.metric = metric
                    self.engine.intent.direction = direction
                    self.engine.intent.target = default_target
                    self.engine.intent.type = f"OPTIMIZE_{raw_kpi.upper()}"

            msg = await q_kpi.get()

            actual_kpi_data = msg.payload.get("kpi", msg.payload)

            # Run the heavy RL step (feature extraction + reward + gradient update)
            # off the event loop so it doesn't block other agents.
            state_tensor = await loop.run_in_executor(
                None, functools.partial(self.engine.step, last_pb, kpi_dict=actual_kpi_data)
            )

            # Only trigger the propose→score→act cycle at a controlled rate.
            # Radio parameter changes need time to propagate through the network.
            if state_tensor is not None:
                now = time.monotonic()
                if now - self._last_action_time >= self.min_action_interval:
                    self._last_action_time = now
                    await self.bus.pub(
                        "kpi.window",
                        make_msg("kpi.window", "STATE_WINDOW", "kpi.window.v1", {
                            "state": state_tensor.tolist(),
                            "reward": getattr(self.engine, 'last_reward', 0.0),
                            "situation": self.engine.get_context_situation()
                        })
                    )