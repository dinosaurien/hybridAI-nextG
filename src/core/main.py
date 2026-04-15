import argparse
import asyncio
import logging
from pathlib import Path
import sys

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent))

from core.bus.mem import MemBus
from core.bus.messages import make_msg
from core.utils.knowledge_base import KnowledgeBase
from core.ui_layer.dashboard.dashboard_server import UnifiedWebServer
from core.control_layer.orchestrator_agent import OrchestratorAgent
from core.control_layer.network_optimizer import NetworkOptimizer
from core.control_layer.actuator_agent import ActuatorAgent
from core.control_layer.constraint_executor import ConstraintExecutor
from core.control_layer.xapp_adapter import XAppTCPServer
from core.telemetry_layer.telemetry_agent import TelemetryAgent
from core.brain.cognitive_agent import CognitiveAgent
from core.brain.episode_store import EpisodeStore

logger = logging.getLogger(__name__)

# Static expert-tuned OTM for fixed-otm evaluation mode.
# Represents the simplest non-trivial control strategy:
# cap MCS to reduce retransmissions, boost TX power for coverage.
FIXED_OTM = {
    "version": "1.0",
    "objective": {
        "service": "mbb", "kpi": "DRB_PdcpSduDelayDl",
        "aggregation": "mean", "unit": "ms", "maximize": False,
    },
    "constraints": [
        {"service": "mbb", "kpi": "dl_mcs_max", "operator": "le",
         "threshold": 20.0, "unit": "", "id": "STATIC_MCS"},
        {"service": "mbb", "kpi": "tx_power_dbm", "operator": "ge",
         "threshold": 46.0, "unit": "dBm", "id": "STATIC_PWR"},
    ],
    "metadata": {
        "timescale": "10s_window",
        "procedure_id": "static_baseline",
        "adaptation_log": ["Static expert rules — no LLM, no adaptation."],
    },
}


async def wait_forever():
    try:
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        logger.info("Main loop cancelled.")


async def run_baseline_logging(tcp_server, args):
    """Baseline mode: log KPIs from the simulator with no actions taken."""
    bus = MemBus()
    tcp_server.bus = bus

    baseline_csv = Path("kpms_baseline.csv")
    tcp_server.kpi_csv_file = baseline_csv
    tcp_server.kpi_csv_initialized = False
    tcp_server._init_kpi_csv()

    web_server = UnifiedWebServer(bus, port=args.web_port)
    tcp_server.commands_enabled = False

    tasks = [
        asyncio.create_task(tcp_server.start()),
        asyncio.create_task(web_server.start()),
    ]

    logger.info("=" * 50)
    logger.info("BASELINE MODE — logging KPIs only, NO actions taken")
    logger.info(f"KPI data will be saved to: {baseline_csv.resolve()}")
    logger.info(f"Dashboard (read-only): http://localhost:{args.web_port}")
    logger.info(f"Waiting for ns-3 KPIs on {args.host}:{args.port}...")
    logger.info("=" * 50)
    await wait_forever()


async def run_fixed_otm(tcp_server, args):
    """Fixed-OTM mode: static expert rules with deterministic constraint execution.

    No LLM, no anomaly detection, no orchestrator FSM.
    A predefined OTM is published at startup and periodically re-published.
    The ConstraintExecutor applies it. KPIs are logged for comparison.
    This is the "rule-based expert" evaluation baseline.
    """
    bus = MemBus()
    tcp_server.bus = bus

    web_server = UnifiedWebServer(bus, port=args.web_port)
    actuator = ActuatorAgent(bus, cell_to_node_map=tcp_server.cell_to_node_map)
    # objective_aware=False: static expert rules use constraint thresholds as-is,
    # no objective-driven biasing. This is the "rule-based" evaluation baseline.
    constraint_executor = ConstraintExecutor(bus, objective_aware=False)

    async def publish_fixed_otm():
        """Publish the fixed OTM after ns-3 connects, then keep it alive."""
        await asyncio.sleep(3)
        logger.info(f"[FIXED-OTM] Publishing static OTM: MCS <= 20, TX Power >= 46 dBm")
        await bus.pub("intent.current", make_msg("intent", "INTENT", "v1", FIXED_OTM))
        while True:
            await asyncio.sleep(60)
            await bus.pub("intent.current", make_msg("intent", "INTENT", "v1", FIXED_OTM))

    tasks = [
        asyncio.create_task(tcp_server.start()),
        asyncio.create_task(web_server.start()),
        asyncio.create_task(actuator.run()),
        asyncio.create_task(constraint_executor.run()),
        asyncio.create_task(publish_fixed_otm()),
    ]

    logger.info("=" * 50)
    logger.info("FIXED-OTM MODE — static expert rules, NO LLM, NO anomaly detection")
    logger.info(f"OTM: minimize latency | MCS <= 20, TX Power >= 46 dBm")
    logger.info(f"Dashboard: http://localhost:{args.web_port}")
    logger.info(f"Waiting for ns-3 KPIs on {args.host}:{args.port}...")
    logger.info("=" * 50)
    await asyncio.gather(*tasks)


async def run_deploy(tcp_server, args):
    """Deploy mode: full LLM + deterministic constraint execution pipeline.

    With --no-episodes: disables episodic memory (Reflexion ablation).
    The LLM still uses KB procedures but gets no past experience in its prompt.
    """
    bus = MemBus()
    tcp_server.bus = bus
    kb = KnowledgeBase()

    # Episode store for RAG — records anomaly→OTM→outcome cycles
    # Disabled with --no-episodes for Reflexion ablation experiments
    episode_store = None
    if not args.no_episodes:
        episode_store = EpisodeStore(store_path="models/episodes.jsonl")
        logger.info("[DEPLOY] Episodic memory (Reflexion) ENABLED.")
    else:
        logger.info("[DEPLOY] Episodic memory DISABLED (--no-episodes ablation mode).")

    # Start network-facing services first so KPIs aren't dropped during LLM load
    web_server = UnifiedWebServer(bus, port=args.web_port)
    orchestrator = OrchestratorAgent(bus, kb, episode_store=episode_store)
    optimizer = NetworkOptimizer(bus)
    actuator = ActuatorAgent(bus, cell_to_node_map=tcp_server.cell_to_node_map)

    # Deterministic constraint executor: OTM constraints → E2 commands directly
    constraint_executor = ConstraintExecutor(bus)

    telemetry_agents = []
    if args.minirocket_gnb_model:
        logger.info("Initializing gNB Telemetry Agent for metric: DRB_PdcpSduDelayDl")
        telemetry_agents.append(
            TelemetryAgent(bus, model_path=args.minirocket_gnb_model, metric="DRB_PdcpSduDelayDl")
        )
    if args.minirocket_ue_model:
        logger.info("Initializing UE Telemetry Agent for metric: UE_DRB_PdcpSduDelayDl_UEID")
        telemetry_agents.append(
            TelemetryAgent(bus, model_path=args.minirocket_ue_model, metric="UE_DRB_PdcpSduDelayDl_UEID")
        )
        logger.info("Initializing UE Throughput Agent for Blockage Detection")
        telemetry_agents.append(
            TelemetryAgent(bus, model_path=args.minirocket_ue_model, metric="UE_DRB_UEThpDl_UEID")
        )
        
    if not telemetry_agents:
        logger.warning("No MiniRocket models provided. Anomaly detection will be disabled.")

    # Launch all lightweight agents immediately
    tasks = [
        asyncio.create_task(tcp_server.start()),
        asyncio.create_task(web_server.start()),
        asyncio.create_task(orchestrator.run()),
        asyncio.create_task(optimizer.run()),
        asyncio.create_task(actuator.run()),
        asyncio.create_task(constraint_executor.run()),
    ]
    for agent in telemetry_agents:
        tasks.append(asyncio.create_task(agent.run()))

    logger.info(f"Dashboard online: http://localhost:{args.web_port}")

    # Load LLM in background thread so the event loop keeps processing KPIs
    loop = asyncio.get_running_loop()
    cognitive_agent = await loop.run_in_executor(
        None, lambda: CognitiveAgent(bus, kb, episode_store=episode_store)
    )
    tasks.append(asyncio.create_task(cognitive_agent.run()))
    logger.info("Cognitive agent loaded — full pipeline active.")

    await asyncio.gather(*tasks)


async def main():
    parser = argparse.ArgumentParser(
        description="HybridAI-NextG: LLM-driven 5G/6G network optimizer"
    )
    parser.add_argument("--mode", type=str, default="deploy",
                        choices=["deploy", "baseline", "fixed-otm"],
                        help="deploy: full LLM system | baseline: no actions | fixed-otm: static rules only")
    parser.add_argument("--no-episodes", action="store_true",
                        help="Disable episodic memory (Reflexion ablation). Only affects deploy mode.")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6000)
    parser.add_argument("--web-port", type=int, default=8080)
    parser.add_argument("--target-metric", default="DRB_PdcpSduDelayDl")
    parser.add_argument("--minirocket-gnb-model", default="models/minirocket_xapp_gnb.joblib")
    parser.add_argument("--minirocket-ue-model", default="models/minirocket_xapp_ue.joblib")
    parser.add_argument("--log-level", type=str, default="INFO")

    args = parser.parse_args()

    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(level=log_level, format='[%(asctime)s] [%(name)s] [%(levelname)s] - %(message)s')

    tcp_server = XAppTCPServer(host=args.host, port=args.port)

    if args.mode == "baseline":
        await run_baseline_logging(tcp_server, args)
    elif args.mode == "fixed-otm":
        await run_fixed_otm(tcp_server, args)
    else:
        await run_deploy(tcp_server, args)

if __name__ == "__main__":
    asyncio.run(main())
