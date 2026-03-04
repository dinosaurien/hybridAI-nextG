# src/demo/ain/bus/mem.py
import asyncio
from collections import defaultdict
from typing import Any, Dict, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class MemBus:
    def __init__(self) -> None:
        self._topics: Dict[str, List[asyncio.Queue]] = defaultdict(list)
        self.xapp_server = None

    async def init_xapp_server(self, host: str = "0.0.0.0", port: int = 5000):
        """Initialize TCP server for xApp communication."""
        from .xapp_server import XAppTCPServer
        
        self.xapp_server = XAppTCPServer(host, port)
        self.xapp_server.set_bus(self)  # Give server access to bus
        
        # Register any custom handlers if needed
        self.xapp_server.register_handler("custom_message", self._handle_custom_message)
        
        await self.xapp_server.start()
        logger.info(f"xApp TCP server initialized on {host}:{port}")

    async def _handle_custom_message(self, client_id: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle custom message types from xApp."""
        await self.pub("custom_xapp_message", {
            "client_id": client_id,
            "message": message
        })
        return {"status": "acknowledged"}

    async def pub(self, topic: str, payload: Any) -> None:
        for q in list(self._topics.get(topic, [])):
            await q.put(payload)
            
        # Send relevant updates back to xApp clients
        if self.xapp_server and topic in ["playbook_generated", "optimization_result", "deviation_detected"]:
            notification = {
                "type": "ai_notification",
                "topic": topic,
                "payload": payload,
                "timestamp": datetime.now().isoformat()
            }
            await self.xapp_server.broadcast_to_clients(notification)

    async def sub(self, topic: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._topics[topic].append(q)
        return q

    def unsub(self, topic: str, q: asyncio.Queue) -> None:
        if q in self._topics.get(topic, []):
            self._topics[topic].remove(q)

    async def stop_xapp_server(self):
        """Stop the xApp TCP server."""
        if self.xapp_server:
            await self.xapp_server.stop()
            logger.info("xApp TCP server stopped")
