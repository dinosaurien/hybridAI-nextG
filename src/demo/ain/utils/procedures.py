import asyncio
from asyncio.log import logger
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class SymbolicProcedures:
    def __init__(self, bus, tcp_server, kb):
        self.bus = bus
        self.tcp_server = tcp_server
        self.kb = kb

    async def DRAIN_USERS(self, cellid):
        logger.info(f"[PROCEDURE] Executing procedure: Draining users from {cellid}")
        command = {
            "type": "control",
            "cmd": {"cmd": "set-cio", "cell_id": cellid, "value": -10}
        }

        for cid in self.tcp_server.client():
            await self.tcp_server.send_command(cid, cmd)

        await asyncio.sleep(5)
        return True

    async def GENERATE_OTM_LATENCY(self, cell_id, **kwargs):
        """Passive Step: Returns the OTM JSON for the UI."""
        logger.info(f"[TOOL] Formulating OTM for {cell_id}")
        
        # Return the full structure so the Actuator doesn't crash
        return {
            "version": "1.0",
            "objective": {
                "service": "mbb", 
                "kpi": "throughput", 
                "aggregation": "mean",
                "unit": "Mbps",
                "maximize": True
            },
            "constraints": [
                {
                    "service": "mbb",
                    "kpi": "latency", 
                    "operator": "ge", 
                    "threshold": 7.00, 
                    "unit": "ms", 
                    "scope": "per_cell",
                    "origin": "LLM",
                    "adapted_by": "LLM",
                    "id": "None"
                }
            ],
            "metadata": {
                "target_cell": cell_id,
                "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "adaptation_log": []
                # 'episode' will be added by the Actuator
            }
        }
    
    async def HEALTH_CHECK(self, current_metrics, cell_id):
        logger.info(f"[HEALTH] Starting check for {cell_id}...")

        policy = self.kb.get_health_thresholds(cell_id)
        actual_latency = current_metrics.get('DRB_PdcpSduDelayDl', 0)
        actual_throughput = current_metrics.get('thr_dl_bps', 0)
        
        violations = []

        if actual_latency > policy['latency_max_ms']:
            violations.append(f"Latency spike: {actual_latency:.2f}ms (Limit: {policy['latency_max_ms']}ms)")

        if actual_throughput < (policy['throughput_min_mbps'] * 1_000_000):
            actual_mbps = actual_throughput / 1_000_000
            violations.append(f"Throughput drop: {actual_mbps:.2f}Mbps (SLA: {policy['throughput_min_mbps']}Mbps)")

        if not violations:
            logger.info(f"[HEALTH] Result for {cell_id}: PASSED ✅")
            return True
        else:
            for v in violations:
                logger.warning(f"[HEALTH] VIOLATION: {v}")
            logger.error(f"[HEALTH] Result for {cell_id}: FAILED ❌")
            return False