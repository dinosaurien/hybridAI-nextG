import argparse
import asyncio
import logging
from pathlib import Path

from core.bus.mem import MemBus
from core.utils.knowledge_base import KnowledgeBase
from core.ui_layer.dashboard.dashboard_server import UnifiedWebServer
from core.control_layer.orchestrator_agent import OrchestratorAgent
from core.control_layer.xapp_adapter import XAppTCPServer
from core.telemetry_layer.telemetry_agent import TelemetryAgent


logger = logging.getLogger(__name__)


async def run_ai_loop_with_membus(tcp_server, target_metric, args):
    bus = MemBus()
    tcp_server.bus = bus
    kb = KnowledgeBase()
    
    web_server = UnifiedWebServer(bus, port=args.web_port)
    orchestrator = OrchestratorAgent(bus)
    telemetry_agents = []
    
    # gNB Level Monitoring
    if args.minirocket_gnb_model or args.minirocket_model:
        gnb_m = "DRB_PdcpSduDelayDl" if target_metric == "delay_p95_ms" else target_metric
        path = args.minirocket_gnb_model or args.minirocket_model
        telemetry_agents.append(TelemetryAgent(
            bus=bus, 
            model_path=path, 
            metric=gnb_m, 
            window_size=128
        ))
    
    # UE Level Monitoring
    if args.minirocket_ue_model or args.minirocket_model:
        ue_m = "UE_DRB_PdcpSduDelayDl_UEID" if target_metric in ("delay_p95_ms", "DRB_PdcpSduDelayDl") else target_metric
        path = args.minirocket_ue_model or args.minirocket_model
        telemetry_agents.append(TelemetryAgent(
            bus=bus, 
            model_path=path, 
            metric=ue_m, 
            window_size=128
        ))

    tasks = [
        asyncio.create_task(tcp_server.start()), 
        asyncio.create_task(web_server.start()),
        asyncio.create_task(orchestrator.run()),
        # asyncio.create_task(actuator.run()),  # Add back when Actuator logic is ready
    ]
    
    for agent in telemetry_agents: 
        tasks.append(asyncio.create_task(agent.run()))

    logger.info(f"System fully orchestrated. Dashboard: http://localhost:{args.web_port}")
    
    try: 
        await asyncio.gather(*tasks)
    except KeyboardInterrupt: 
        logger.info("Keyboard interrupt received. Shutting down...")
        await tcp_server.stop()


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6000)
    parser.add_argument("--web-port", type=int, default=8080)
    parser.add_argument("--target-metric", default="DRB_PdcpSduDelayDl")
    parser.add_argument("--minirocket-model", default=None)
    parser.add_argument("--minirocket-gnb-model", default=None)
    parser.add_argument("--minirocket-ue-model", default=None)
    parser.add_argument("--log-level", type=str, default="INFO") 
    
    args = parser.parse_args()
    
    # Configure logging based on args #TODO fix logging level
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    tcp_server = XAppTCPServer(host=args.host, port=args.port)
    
    await run_ai_loop_with_membus(tcp_server, args.target_metric, args)

if __name__ == "__main__":
    asyncio.run(main())