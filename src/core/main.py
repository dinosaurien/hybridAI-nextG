import argparse
import asyncio
import logging
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
from core.control_layer.RL_engines.observer_rl import RLObserver, Intent
from core.control_layer.RL_engines.proposer import ActionSpace, CacheLibrary
from core.control_layer.RL_engines.predictor import SlateDQNPredictor
from core.control_layer.RL_engines.actor import Actor

logger = logging.getLogger(__name__)

async def run_rl_training_loop(tcp_server, args):
    bus = MemBus()
    tcp_server.bus = bus
    
    # Initialize RL components
    action_space = ActionSpace(cells=["CELL_001"], slices=["SLICE_A"])
    kb = CacheLibrary() # Use CacheLibrary as KB for RL
    feature_dim = len(RLObserver(None, Intent("DUMMY", "DUMMY", 0)).features)
    predictor = SlateDQNPredictor(action_space, feat_dim=feature_dim)

    # Load a pre-trained model to continue training if specified
    if args.dqn_model:
        try:
            predictor.load_checkpoint(args.dqn_model)
            logger.info(f"Resuming training from model: {args.dqn_model}")
        except Exception as e:
            logger.error(f"Failed to load model for training: {e}. Starting fresh.")

    tmp_intent = Intent(type="REDUCE_LATENCY", metric=args.target_metric, target=40.0)
    observer_engine = RLObserver(predictor, intent=tmp_intent)
    
    proposer_agent = ProposerAgent(bus, action_space, kb)
    predictor_agent = PredictorAgent(bus, predictor)
    actor_agent = ActorAgent(bus, Actor("playbooks"))
    observer_agent = ObserverAgent(bus, observer_engine)

    def save_model_on_exit(signum, frame):
        logger.info("Shutdown signal received. Saving trained model...")
        predictor.save_checkpoint("models/qnet_online.pt")
        logger.info("Model saved to models/qnet_online.pt")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, save_model_on_exit)
    signal.signal(signal.SIGTERM, save_model_on_exit)
    
    tasks = [
        asyncio.create_task(tcp_server.start()),
        asyncio.create_task(proposer_agent.run()),
        asyncio.create_task(predictor_agent.run()),
        asyncio.create_task(actor_agent.run()),
        asyncio.create_task(observer_agent.run()),
    ]
    
    await asyncio.sleep(1)
    from core.bus.messages import make_msg
    await bus.pub("intent.current", make_msg("intent", "INFO", "v1", { "type": "REDUCE_LATENCY", "metric": args.target_metric, "target": 40.0 }))
    
    logger.info("RL Training Loop is LIVE. Connect ns-3 and train the model.")
    await asyncio.gather(*tasks)

async def run_ai_loop(tcp_server, args):
    bus = MemBus()
    tcp_server.bus = bus
    kb = KnowledgeBase()
    
    action_space = ActionSpace(cells=["CELL_001"], slices=["SLICE_A"])
    cache = CacheLibrary()
    feature_dim = len(RLObserver(None, Intent("DUMMY", "DUMMY", 0)).features)
    predictor = SlateDQNPredictor(action_space, feat_dim=feature_dim)

    if args.dqn_model:
        try:
            predictor.load_offline(args.dqn_model)
            logger.info(f"Loaded pre-trained DQN model for inference: {args.dqn_model}")
        except Exception as e:
            logger.error(f"Failed to load DQN model: {e}. Optimizer will be un-trained.")
    else:
        logger.warning("No DQN model provided. NetworkOptimizer will use random exploration.")

    web_server = UnifiedWebServer(bus, port=args.web_port)
    orchestrator = OrchestratorAgent(bus, kb)
    optimizer = NetworkOptimizer(bus, action_space, predictor, cache)
    actuator = ActuatorAgent(bus)
    cognitive_agent = CognitiveAgent(bus, kb)
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
    
    tasks = [
        asyncio.create_task(tcp_server.start()),
        asyncio.create_task(web_server.start()),
        asyncio.create_task(orchestrator.run()),
        asyncio.create_task(optimizer.run()),
        asyncio.create_task(actuator.run()),
        asyncio.create_task(cognitive_agent.run()),
        asyncio.create_task(proposer_agent.run()),
        asyncio.create_task(predictor_agent.run()),
        asyncio.create_task(actor_agent.run()),
        asyncio.create_task(observer_agent.run())
    ]

    for agent in telemetry_agents:
        tasks.append(asyncio.create_task(agent.run()))
    
    logger.info(f"Dashboard online: http://localhost:{args.web_port}")
    await asyncio.gather(*tasks)

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="deploy", choices=["train", "deploy"], help="Run in training or deployment mode.")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6000)
    parser.add_argument("--web-port", type=int, default=8080)
    parser.add_argument("--target-metric", default="DRB_PdcpSduDelayDl")
    parser.add_argument("--minirocket-gnb-model", default="models/minirocket_xapp_gnb.joblib")
    parser.add_argument("--minirocket-ue-model", default="models/minirocket_xapp_ue.joblib")
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--dqn-model", type=str, default=None, help="Path to the SlateDQNPredictor model file.")
    
    args = parser.parse_args()
    
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(level=log_level, format='[%(asctime)s] [%(name)s] [%(levelname)s] - %(message)s')
    
    tcp_server = XAppTCPServer(host=args.host, port=args.port)
    
    if args.mode == "train":
        await run_rl_training_loop(tcp_server, args)
    else: # deploy
        await run_ai_loop(tcp_server, args)

if __name__ == "__main__":
    asyncio.run(main())