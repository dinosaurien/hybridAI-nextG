import asyncio
import uuid

from demo.ain.bus.messages import make_msg
from demo.ain.utils.procedures import SymbolicProcedures


class ActuatorAgent:
    def __init__(self, bus, tcp_server, kb):
        self.bus = bus
        self.procedures = SymbolicProcedures(bus, tcp_server, kb)
        self.latest_metrics = {}

    async def run(self):
        q_exec = await self.bus.sub("execution.request")
        q_kpi = await self.bus.sub("kpi.raw")

        while True:
            while not q_kpi.empty():
                self.latest_metrics.update(q_kpi.get_nowait().payload.get("kpi", {}).get("CellMetrics", {}))

            if not q_exec.empty():
                req = q_exec.get_nowait().payload
                steps = req["selected_procedures"]
                cell_id = req["cell_id"]

                for step_id in steps:
                    if hasattr(self.procedures, step_id):
                        tool = getattr(self.procedures, step_id)
                        # We pass the metrics and the ID
                        result = await tool(cell_id=cell_id, current_metrics=self.latest_metrics)
                        if step_id == "GENERATE_OTM_LATENCY":
                            # TODO: We can add metadata here if the toolbox didn't
                            result["metadata"]["episode"] = f"auto_{uuid.uuid4().hex[:4]}"
                            await self.bus.pub("otm.ui_display", make_msg("otm", "DATA", "v1", result))
                    
                    await asyncio.sleep(1.0)
            await asyncio.sleep(0.1)