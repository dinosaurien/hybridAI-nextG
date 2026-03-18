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

            if hasattr(best_playbook, 'actions'):
                for action in best_playbook.actions:
                    # Skip empty NOOPs so we don't spam the UI
                    if action.type == "REPORTING" and action.params.get("noop"):
                        continue
                    
                    cmd_payload = {
                        "command": action.type,
                        "params": action.params
                    }
                    
                    # Push to the UI
                    await self.bus.pub("command.notify", make_msg("actor", "CMD", "v1", cmd_payload))
            
            try:
                filepath = self.actor.make_and_save(best_playbook)
                logger.debug(f"[ACTOR] Playbook saved to {filepath}")
            except Exception as e:
                logger.error(f"[ACTOR] Failed to serialize/save playbook: {e}", exc_info=True)