import asyncio
import logging
from core.bus.messages import make_msg
from core.common.log_config import should_log, LOG_SCORING, LOG_LEARNING

logger = logging.getLogger(__name__)

class PredictorAgent:
    def __init__(self, bus, predictor):
        self.bus = bus
        self.model = predictor
        self.state = None
        self.pending_candidates = None  # Store candidates if we get them before state

    async def run(self):
        try:
            if should_log(LOG_SCORING):
                logger.info("[SCORING] Initializing predictor agent")
            q_state = await self.bus.sub("kpi.window")
            q_cands = await self.bus.sub("proposer.candidates")

            if should_log(LOG_SCORING):
                logger.info("[SCORING] Subscribed to kpi.window and proposer.candidates")

            # Process messages from both queues independently
            state_task = asyncio.create_task(self._process_state_queue(q_state))
            cands_task = asyncio.create_task(self._process_candidates_queue(q_cands))

            if should_log(LOG_SCORING):
                logger.info("[SCORING] Started processing tasks (state, candidates)")

            # Wait for either task to crash; gather propagates the first exception
            await asyncio.gather(state_task, cands_task)
        except Exception as e:
            logger.error(f"[SCORING] Fatal error in run(): {e}", exc_info=True)
            raise
    
    async def _process_state_queue(self, q_state):
        """Process state window messages."""
        if should_log(LOG_SCORING):
            logger.debug("[SCORING] State queue handler started")
        while True:
            try:
                msg = await q_state.get()
                # Handle both Msg objects and direct payloads
                if hasattr(msg, 'payload'):
                    state = msg.payload.get("state") if isinstance(msg.payload, dict) else msg.payload
                else:
                    state = msg.get("state") if isinstance(msg, dict) else msg
                
                self.state = state
                state_shape = len(self.state) if isinstance(self.state, list) else 'unknown'
                if should_log(LOG_SCORING):
                    logger.debug(f"[SCORING] Received state window (shape: {state_shape})")
                
                # Check if we have pending candidates to score
                if self.pending_candidates is not None:
                    if should_log(LOG_SCORING):
                        logger.debug(f"[SCORING] Have pending candidates ({len(self.pending_candidates)}), scoring now...")
                    await self._score_playbooks(self.pending_candidates)
                    self.pending_candidates = None
            except Exception as e:
                logger.error(f"[SCORING] Error processing state queue: {e}", exc_info=True)
    
    async def _process_candidates_queue(self, q_cands):
        """Process candidate playbook messages."""
        if should_log(LOG_SCORING):
            logger.debug("[SCORING] Started listening for candidate playbooks...")
        while True:
            try:
                msg = await q_cands.get()
                if should_log(LOG_SCORING):
                    logger.debug(f"[SCORING] Got message from candidates queue: type={type(msg)}, has_payload={hasattr(msg, 'payload')}")
                
                # Handle both Msg objects and direct payloads
                if hasattr(msg, 'payload'):
                    payload = msg.payload
                    if should_log(LOG_SCORING):
                        logger.debug(f"[SCORING] Message has payload: type={type(payload)}")
                    if isinstance(payload, dict):
                        candidates = payload.get("candidates", [])
                    else:
                        candidates = []
                else:
                    if isinstance(msg, dict):
                        candidates = msg.get("candidates", [])
                    else:
                        candidates = []
                
                if should_log(LOG_SCORING):
                    logger.debug(f"[SCORING] Received {len(candidates)} candidate playbooks")
                
                # Check if we have state to score with
                if self.state is not None:
                    await self._score_playbooks(candidates)
                    self.pending_candidates = None
                else:
                    if should_log(LOG_SCORING):
                        logger.debug("[SCORING] Received candidates but no state yet, storing for later...")
                    self.pending_candidates = candidates
            except Exception as e:
                logger.error(f"[SCORING] Error processing candidates queue: {e}", exc_info=True)
    
    async def _score_playbooks(self, playbooks):
        """Score playbooks with current state. DQN forward pass runs off the event loop."""
        if not playbooks:
            return

        if should_log(LOG_SCORING):
            logger.debug(f"[SCORING] Scoring {len(playbooks)} playbooks...")

        # Convert state to numpy array if needed
        import numpy as np
        if isinstance(self.state, list):
            state_array = np.array(self.state)
        else:
            state_array = self.state

        try:
            # Run the DQN forward pass in a thread so it doesn't block the event loop
            loop = asyncio.get_running_loop()
            scored = await loop.run_in_executor(
                None, self.model.score_playbooks, state_array, playbooks
            )
            if scored:
                # Check for NaN values (common with untrained models)
                import math
                has_nan = any(math.isnan(q) or not math.isfinite(q) for _, q in scored)

                if has_nan:
                    if should_log(LOG_SCORING):
                        logger.warning("[SCORING] Model returned NaN/infinite Q values (untrained model). Using fallback scoring.")
                    # Fallback: assign random small values for untrained model
                    import random
                    scored = [(pb, random.uniform(-0.1, 0.1)) for pb, _ in scored]

                best_q = max(q for _, q in scored)
                if should_log(LOG_SCORING):
                    logger.info(f"[SCORING] Scored {len(scored)} playbooks, best Q={best_q:.3f}")

                await self.bus.pub("predictor.scored", make_msg(
                    "predictor.scored", "SCORED", "scored.v1",
                    {"scored": [(pb, float(q)) for pb, q in scored]}
                ))
            else:
                if should_log(LOG_SCORING):
                    logger.warning("[SCORING] No scored playbooks returned from model")
        except Exception as e:
            logger.error(f"[SCORING] Error scoring playbooks: {e}", exc_info=True)