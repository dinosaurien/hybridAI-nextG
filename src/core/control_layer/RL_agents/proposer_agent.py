import asyncio
import logging
from core.bus.messages import make_msg
from core.common.log_config import should_log, LOG_INTENT
from core.control_layer.RL_engines.proposer import ProposerSampler, CANDIDATE_N, PLAYBOOK_K

logger = logging.getLogger(__name__)

class ProposerAgent:
    def __init__(self, bus, action_space, knowledge_base=None):
        self.bus = bus
        self.action_space = action_space
        self.knowledge_base = knowledge_base
        self.active_otm = None
        self.last_state = None
        self.current_otm_id = None

    async def run(self):
        q_state = await self.bus.sub("kpi.window")
        q_intent = await self.bus.sub("intent.current")

        while True:
            while not q_intent.empty():
                msg = q_intent.get_nowait()
                self.active_otm = msg.payload
                new_id = self.active_otm.get("metadata", {}).get("episode")
                if new_id != self.current_otm_id:
                    self.current_otm_id = new_id
                    logger.info(f"[PROPOSER] New OTM received (episode={new_id}). RL loop will run continuously under these constraints.")
                else:
                    if should_log(LOG_INTENT):
                        logger.debug(f"[PROPOSER] OTM re-confirmed (episode={new_id}).")

            while not q_state.empty():
                msg = q_state.get_nowait()
                self.last_state = msg.payload

                if self.active_otm and self.last_state:
                    try:
                        situation = self.last_state.get("situation", "normal") if isinstance(self.last_state, dict) else "normal"

                        objective = self.active_otm.get("objective", {})
                        semantic_kpi = objective.get("kpi", "latency").lower()

                        if "thp" in semantic_kpi or "throughput" in semantic_kpi:
                            proposer_intent = "THR_DL"
                        elif "power" in semantic_kpi or "energy" in semantic_kpi:
                            proposer_intent = "ENERGY_SAVING"
                        else:
                            proposer_intent = "LATENCY_P95"

                        intent_meta = {"intent": proposer_intent, "scope": "GLOBAL"}

                        playbooks = ProposerSampler.sample_contextual_playbooks(
                            action_space=self.action_space,
                            N=CANDIDATE_N,
                            K=PLAYBOOK_K,
                            epsilon=0.3,
                            intent_meta=intent_meta,
                            cache=self.knowledge_base,
                            situation=situation
                        )

                        await self.bus.pub("proposer.candidates", make_msg(
                            "proposer.candidates", "PLAYBOOKS", "playbooks.v1",
                            {"candidates": playbooks}
                        ))

                        self.last_state = None

                    except Exception as e:
                        logger.error(f"[PROPOSER] Error generating playbooks: {e}", exc_info=True)

            await asyncio.sleep(0.1)