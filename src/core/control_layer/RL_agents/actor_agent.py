import asyncio
import logging
from core.bus.messages import make_msg

logger = logging.getLogger(__name__)

class ActorAgent:
    def __init__(self, bus, actor):
        self.bus = bus
        self.actor = actor

    async def run(self):
        q = await self.bus.sub("predictor.scored")
        while True:
            msg = await q.get()
            
            # Extract the best playbook and its score
            best_playbook, best_q = max(msg.payload["scored"], key=lambda t: t[1])
            
            logger.info(f"[ACTOR] 🎬 Executing Playbook (Q-Score: {best_q:.3f})")

            # Publish to actor.apply so the RL Observer knows what action was taken
            await self.bus.pub("actor.apply", make_msg(
                "actor", "APPLY", "v1",
                {"playbook": best_playbook, "q": best_q}
            ))

            try:
                filepath = self.actor.make_and_save(best_playbook)
                logger.debug(f"[ACTOR] Playbook saved to {filepath}")
            except Exception as e:
                logger.error(f"[ACTOR] Failed to serialize/save playbook: {e}", exc_info=True)