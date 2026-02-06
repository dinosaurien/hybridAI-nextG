import asyncio
from .actor_agent import ActorAgent
from .utils import make_msg
from ain.bus.mem import MemBus

async def run_all(intent, action_space, predictor, actor, source):
    bus = MemBus()

    # --- Pick KPI source ---
    if source == "fake":
        from ain.RL_demo.fake_kpi import fake_kpi_producer
        asyncio.create_task(fake_kpi_producer(bus))
    #else:
        #from ..loop.real_adapter import ran_adapter
        #asyncio.create_task(ran_adapter(bus))

    # --- Agents ---
    act = ActorAgent(bus, actor)

    await asyncio.gather(act.run())
