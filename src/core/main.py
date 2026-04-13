import argparse
import asyncio
import logging
import os
from pathlib import Path
import signal
import sys

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent))

from core.bus.mem import MemBus
from core.utils.knowledge_base import KnowledgeBase
from core.ui_layer.dashboard.dashboard_server import UnifiedWebServer
from core.control_layer.orchestrator_agent import OrchestratorAgent
from core.control_layer.network_optimizer import NetworkOptimizer
from core.control_layer.actuator_agent import ActuatorAgent
from core.control_layer.xapp_adapter import XAppTCPServer
from core.telemetry_layer.telemetry_agent import TelemetryAgent
from core.brain.cognitive_agent import CognitiveAgent 
from core.control_layer.RL_agents.proposer_agent import ProposerAgent
from core.control_layer.RL_agents.predictor_agent import PredictorAgent
from core.control_layer.RL_agents.actor_agent import ActorAgent
from core.control_layer.RL_agents.observer_agent import ObserverAgent
from core.control_layer.RL_engines.observer_rl import RLObserver, Intent, DEFAULT_FEATURES
from core.control_layer.RL_engines.proposer import ActionSpace, CacheLibrary
from core.control_layer.RL_engines.predictor import SlateDQNPredictor, DEVICE as RL_DEVICE
from core.control_layer.RL_engines.actor import Actor

logger = logging.getLogger(__name__)

# Synthetic OTMs for training, built from the Knowledge Base procedures.
TRAINING_OTMS = [
    {
        "name": "High Latency / Congestion",
        "otm": {
            "version": "1.0",
            "objective": {"service": "mbb", "kpi": "DRB_PdcpSduDelayDl", "aggregation": "mean", "unit": "ms", "maximize": False},
            "constraints": [
                {"service": "mbb", "kpi": "DRB_PdcpSduDelayDl", "operator": "le", "threshold": 50.0, "unit": "ms", "id": "C1"},
                {"service": "mbb", "kpi": "dl_mcs_max", "operator": "le", "threshold": 20.0, "unit": "int", "id": "C2"},
                {"service": "mbb", "kpi": "prb_weight", "operator": "ge", "threshold": 1.2, "unit": "float", "id": "C3"},
            ],
            "metadata": {"timescale": "10s_window", "procedure_id": "graceful_degradation"},
        },
    },
    {
        "name": "Low Throughput",
        "otm": {
            "version": "1.0",
            "objective": {"service": "mbb", "kpi": "UE_DRB_UEThpDl_UEID", "aggregation": "sum", "unit": "Mbps", "maximize": True},
            "constraints": [
                {"service": "mbb", "kpi": "UE_DRB_UEThpDl_UEID", "operator": "ge", "threshold": 50.0, "unit": "Mbps", "id": "C1"},
                {"service": "mbb", "kpi": "dl_mcs_max", "operator": "ge", "threshold": 22.0, "unit": "int", "id": "C2"},
                {"service": "mbb", "kpi": "prb_weight", "operator": "ge", "threshold": 1.5, "unit": "float", "id": "C3"},
            ],
            "metadata": {"timescale": "10s_window", "procedure_id": "demand_surge"},
        },
    },
    {
        "name": "High BLER",
        "otm": {
            "version": "1.0",
            "objective": {"service": "mbb", "kpi": "UE_DRB_BlerDl_UEID", "aggregation": "mean", "unit": "", "maximize": False},
            "constraints": [
                {"service": "mbb", "kpi": "UE_DRB_BlerDl_UEID", "operator": "le", "threshold": 0.01, "unit": "", "id": "C1"},
                {"service": "mbb", "kpi": "dl_mcs_max", "operator": "le", "threshold": 16.0, "unit": "int", "id": "C2"},
                {"service": "mbb", "kpi": "tx_power_dbm", "operator": "ge", "threshold": 45.0, "unit": "dBm", "id": "C3"},
            ],
            "metadata": {"timescale": "10s_window", "procedure_id": "interference_response"},
        },
    },
    {
        "name": "Energy Saving",
        "otm": {
            "version": "1.0",
            "objective": {"service": "mbb", "kpi": "tx_power_dbm", "aggregation": "mean", "unit": "dBm", "maximize": False},
            "constraints": [
                {"service": "mbb", "kpi": "tx_power_dbm", "operator": "le", "threshold": 35.0, "unit": "dBm", "id": "C1"},
                {"service": "mbb", "kpi": "dl_mcs_max", "operator": "le", "threshold": 18.0, "unit": "int", "id": "C2"},
                {"service": "mbb", "kpi": "DRB_PdcpSduDelayDl", "operator": "le", "threshold": 80.0, "unit": "ms", "id": "C3"},
            ],
            "metadata": {"timescale": "10s_window", "procedure_id": "energy_optimization"},
        },
    },
]

async def wait_forever():
    try:
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        logger.info("Main loop cancelled.")

async def run_rl_training_loop(tcp_server, args):
    bus = MemBus()
    tcp_server.bus = bus
    from core.bus.messages import make_msg

    # Initialize RL components
    action_space = ActionSpace(cells=["CELL_001"], slices=["SLICE_A"])
    kb = CacheLibrary()
    feature_dim = len(DEFAULT_FEATURES)
    predictor = SlateDQNPredictor(action_space, feat_dim=feature_dim)

    # Load a pre-trained model to continue training if specified
    if args.dqn_model and os.path.exists(args.dqn_model):
        try:
            if predictor.load_checkpoint(args.dqn_model, load_replay_buffer=True):
                logger.info(f"Resuming training from checkpoint: {args.dqn_model}")
                predictor.load_loss_history("models/loss_history.json")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")

    tmp_intent = Intent(type="REDUCE_LATENCY", metric=args.target_metric, target=40.0)
    observer_engine = RLObserver(predictor, intent=tmp_intent)

    proposer_agent = ProposerAgent(bus, action_space, kb)
    predictor_agent = PredictorAgent(bus, predictor)
    actor_agent = ActorAgent(bus, Actor("playbooks"))
    observer_agent = ObserverAgent(bus, observer_engine, min_action_interval=5.0)
    actuator = ActuatorAgent(bus, cell_to_node_map=tcp_server.cell_to_node_map)

    save_path = "models/qnet_online.pt"

    def save_model_on_exit(signum, frame):
        logger.info("Shutdown signal received. Saving trained model...")
        predictor.save_checkpoint(save_path, save_replay_buffer=True)
        logger.info(f"Model saved to {save_path} (step={predictor.steps}, replay={len(predictor.replay)})")
        if predictor.loss_history:
            predictor.save_loss_history("models/loss_history.json")
        sys.exit(0)

    signal.signal(signal.SIGINT, save_model_on_exit)
    signal.signal(signal.SIGTERM, save_model_on_exit)

    async def otm_curriculum():
        # Cycle through synthetic OTMs so the RL can learn to handle different intents
        otm_rotate_secs = 90
        idx = 0
        # Wait for ns-3 to connect and send first KPIs
        await asyncio.sleep(5)
        while True:
            scenario = TRAINING_OTMS[idx % len(TRAINING_OTMS)]
            logger.info(f"Switching OTM -> '{scenario['name']}' (rotating every {otm_rotate_secs}s)")
            await bus.pub("intent.current", make_msg(
                "intent", "INTENT", "v1", scenario["otm"]
            ))
            idx += 1
            await asyncio.sleep(otm_rotate_secs)

    async def training_monitor():
        # Periodically log training progress
        interval = 15
        last_steps = 0
        while True:
            await asyncio.sleep(interval)
            steps = predictor.steps
            buf_size = len(predictor.replay)
            eps = predictor.epsilon()
            new_steps = steps - last_steps
            last_steps = steps

            if buf_size < 128:
                logger.info(f"Collecting experience... replay={buf_size}/128, no learning yet")
            else:
                recent_loss = ""
                if predictor.loss_history:
                    avg = sum(e['loss'] for e in predictor.loss_history[-10:]) / min(10, len(predictor.loss_history))
                    recent_loss = f", avg_loss={avg:.6f}"
                logger.info(
                    f"step={steps} (+{new_steps}), replay={buf_size}, "
                    f"eps={eps:.3f}{recent_loss}"
                )

    tasks = [
        asyncio.create_task(tcp_server.start()),
        asyncio.create_task(proposer_agent.run()),
        asyncio.create_task(predictor_agent.run()),
        asyncio.create_task(actor_agent.run()),
        asyncio.create_task(observer_agent.run()),
        asyncio.create_task(actuator.run()),
        asyncio.create_task(otm_curriculum()),
        asyncio.create_task(training_monitor()),
    ]

    logger.info("="*50)
    logger.info("RL Training Loop is LIVE.")
    logger.info(f"Device: {RL_DEVICE} | Model dim: feat={feature_dim}, action_onehot={predictor.action_onehot_dim}")
    logger.info(f"Training curriculum: {len(TRAINING_OTMS)} OTMs, rotating every 90s")
    logger.info(f"Action interval: 5.0s | Batch size: 128 | Learning starts after ~11 min")
    logger.info(f"Waiting for ns-3 KPIs on {args.host}:{args.port}...")
    logger.info("="*50)
    await asyncio.gather(*tasks)

async def run_baseline_logging(tcp_server, args):
    # Baseline mode logs KPIs from the simulator when no actions are taken

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


async def run_eval_loop(tcp_server, args):
    # Eval mode: trained DQN + fixed OTM, no LLM, no miniRocket

    bus = MemBus()
    tcp_server.bus = bus
    from core.bus.messages import make_msg

    action_space = ActionSpace(cells=["CELL_001"], slices=["SLICE_A"])
    cache = CacheLibrary()
    feature_dim = len(DEFAULT_FEATURES)
    predictor = SlateDQNPredictor(action_space, feat_dim=feature_dim)

    if args.dqn_model:
        try:
            predictor.load_offline(args.dqn_model)
            logger.info(f"Loaded trained DQN for eval: {args.dqn_model}")
        except Exception as e:
            logger.error(f"Failed to load DQN model: {e}. Running with untrained model.")
    else:
        logger.warning("No --dqn-model provided. Eval will use untrained (random) DQN.")

    # Use the first training OTM as a fixed intent (latency optimization)
    fixed_otm = TRAINING_OTMS[0]["otm"]

    web_server = UnifiedWebServer(bus, port=args.web_port)
    actuator = ActuatorAgent(bus, cell_to_node_map=tcp_server.cell_to_node_map)

    tmp_intent = Intent(type="REDUCE_LATENCY", metric=args.target_metric, target=40.0)
    observer_engine = RLObserver(predictor, intent=tmp_intent)
    proposer_agent = ProposerAgent(bus, action_space, cache)
    predictor_agent = PredictorAgent(bus, predictor)
    actor_agent = ActorAgent(bus, Actor("playbooks"))
    observer_agent = ObserverAgent(bus, observer_engine, min_action_interval=5.0)

    async def publish_fixed_otm():
        """Publish the fixed OTM once after startup, then keep it alive."""
        await asyncio.sleep(3)  # Wait for ns-3 to connect
        logger.info(f"[EVAL] Publishing fixed OTM: '{TRAINING_OTMS[0]['name']}'")
        await bus.pub("intent.current", make_msg("intent", "INTENT", "v1", fixed_otm))
        # Republish periodically so late subscribers pick it up
        while True:
            await asyncio.sleep(60)
            await bus.pub("intent.current", make_msg("intent", "INTENT", "v1", fixed_otm))

    tasks = [
        asyncio.create_task(tcp_server.start()),
        asyncio.create_task(web_server.start()),
        asyncio.create_task(actuator.run()),
        asyncio.create_task(proposer_agent.run()),
        asyncio.create_task(predictor_agent.run()),
        asyncio.create_task(actor_agent.run()),
        asyncio.create_task(observer_agent.run()),
        asyncio.create_task(publish_fixed_otm()),
    ]

    logger.info("=" * 50)
    logger.info("EVAL MODE — RL with fixed OTM, NO LLM, NO anomaly detection")
    logger.info(f"Fixed OTM: '{TRAINING_OTMS[0]['name']}'")
    logger.info(f"DQN model: {args.dqn_model or 'UNTRAINED'}")
    logger.info(f"Dashboard: http://localhost:{args.web_port}")
    logger.info(f"Waiting for ns-3 KPIs on {args.host}:{args.port}...")
    logger.info("=" * 50)
    await asyncio.gather(*tasks)


async def run_ai_loop(tcp_server, args):
    bus = MemBus()
    tcp_server.bus = bus
    kb = KnowledgeBase()

    action_space = ActionSpace(cells=["CELL_001"], slices=["SLICE_A"])
    cache = CacheLibrary()
    feature_dim = len(DEFAULT_FEATURES)
    predictor = SlateDQNPredictor(action_space, feat_dim=feature_dim)

    if args.dqn_model:
        try:
            predictor.load_offline(args.dqn_model)
            logger.info(f"Loaded pre-trained DQN model for inference: {args.dqn_model}")
        except Exception as e:
            logger.error(f"Failed to load DQN model: {e}. Optimizer will be un-trained.")
    else:
        logger.warning("No DQN model provided. NetworkOptimizer will use random exploration.")

    # Start network-facing services first so KPIs aren't dropped during LLM load
    web_server = UnifiedWebServer(bus, port=args.web_port)
    orchestrator = OrchestratorAgent(bus, kb)
    optimizer = NetworkOptimizer(bus)
    actuator = ActuatorAgent(bus, cell_to_node_map=tcp_server.cell_to_node_map)

    tmp_intent = Intent(type="REDUCE_LATENCY", metric=args.target_metric, target=40.0)
    observer_engine = RLObserver(predictor, intent=tmp_intent)
    proposer_agent = ProposerAgent(bus, action_space, cache)
    predictor_agent = PredictorAgent(bus, predictor)
    actor_agent = ActorAgent(bus, Actor("playbooks"))
    observer_agent = ObserverAgent(bus, observer_engine)

    telemetry_agents = []
    if args.minirocket_gnb_model:
        logger.info(f"Initializing gNB Telemetry Agent for metric: DRB_PdcpSduDelayDl")
        telemetry_agents.append(
            TelemetryAgent(bus, model_path=args.minirocket_gnb_model, metric="DRB_PdcpSduDelayDl")
        )
    if args.minirocket_ue_model:
        logger.info(f"Initializing UE Telemetry Agent for metric: UE_DRB_PdcpSduDelayDl_UEID")
        telemetry_agents.append(
            TelemetryAgent(bus, model_path=args.minirocket_ue_model, metric="UE_DRB_PdcpSduDelayDl_UEID")
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
        asyncio.create_task(proposer_agent.run()),
        asyncio.create_task(predictor_agent.run()),
        asyncio.create_task(actor_agent.run()),
        asyncio.create_task(observer_agent.run()),
    ]
    for agent in telemetry_agents:
        tasks.append(asyncio.create_task(agent.run()))

    logger.info(f"Dashboard online: http://localhost:{args.web_port}")

    # Load LLM in background thread so the event loop keeps processing KPIs
    loop = asyncio.get_running_loop()
    cognitive_agent = await loop.run_in_executor(
        None, lambda: CognitiveAgent(bus, kb)
    )
    tasks.append(asyncio.create_task(cognitive_agent.run()))
    logger.info("Cognitive agent loaded — full pipeline active.")

    await asyncio.gather(*tasks)

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="deploy", choices=["train", "deploy", "baseline", "eval"], help="train: RL curriculum | deploy: full system | baseline: no actions | eval: RL only, fixed OTM, no LLM")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6000)
    parser.add_argument("--web-port", type=int, default=8080)
    parser.add_argument("--target-metric", default="DRB_PdcpSduDelayDl")
    parser.add_argument("--minirocket-gnb-model", default="models/minirocket_xapp_gnb.joblib")
    parser.add_argument("--minirocket-ue-model", default="models/minirocket_xapp_ue.joblib")
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--dqn-model", type=str, default="models/qnet_online.pt", help="Path to the SlateDQNPredictor model file.")
    
    args = parser.parse_args()
    
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(level=log_level, format='[%(asctime)s] [%(name)s] [%(levelname)s] - %(message)s')
    
    tcp_server = XAppTCPServer(host=args.host, port=args.port)
    
    if args.mode == "train":
        await run_rl_training_loop(tcp_server, args)
    elif args.mode == "baseline":
        await run_baseline_logging(tcp_server, args)
    elif args.mode == "eval":
        await run_eval_loop(tcp_server, args)
    else:  # deploy
        await run_ai_loop(tcp_server, args)

if __name__ == "__main__":
    asyncio.run(main())