import asyncio
import json
import logging
from aiohttp import web
from pathlib import Path
from core.bus.mem import MemBus
from core.bus.messages import make_msg

logger = logging.getLogger(__name__)

class WebBusAdapter:
    """Sends user intents from UI to the MemBus."""
    def __init__(self, bus: MemBus):
        self.bus = bus

    async def process_intent(self, text: str):
        logger.info(f"[UI LAYER] Forwarding manual intent: {text}")
        await self.bus.pub("ui.input", make_msg("ui", "REQ", "v1", {"text": text}))


class UnifiedWebServer:
    def __init__(self, bus: MemBus, port=8080):
        self.port = port
        self.bus = bus
        self.app = web.Application()
        self.websockets = set()
        self.bus_adapter = WebBusAdapter(bus)
        
        self.static_dir = Path(__file__).resolve().parent / "static"

        # 2. ROUTES
        self.app.router.add_get('/ws', self.handle_websocket)
        self.app.router.add_post('/api/intent', self.handle_post_intent)
        self.app.router.add_get('/', self.redirect_to_dashboard)
        self.app.router.add_static('/static/', self.static_dir, name='static')
        self.app.router.add_post('/api/cc-callback', self.handle_cc_callback)
            

    async def redirect_to_dashboard(self, request):
        """Redirect root URL to the dashboard HTML file."""
        return web.HTTPFound('/static/dashboard.html')
    
    async def handle_index(self, request):
        # We look for dashboard.html in the static folder
        f = self.static_dir / "dashboard.html"
        if f.exists():
            return web.FileResponse(f)
        else:
            return web.Response(text="dashboard.html not found in static folder", status=404)
    
    async def handle_cc_callback(self, request):
        """Ultra-Loud debug logger for Cognitive Core callbacks."""
        logger.info("📡 [NETWORK] /api/cc-callback endpoint was touched!")
        try:
            # Print headers to see if it's coming through the tunnel
            logger.info(f"Headers: {dict(request.headers)}")
            
            raw_body = await request.text()
            logger.info(f"Raw Body: {raw_body}")
            
            data = json.loads(raw_body)
            logger.info(f"🧠 [COGNITIVE CORE FIRED] {data}")
            
            await self.broadcast("cc_notification", data)
            return web.json_response({"status": "acknowledged"})
        except Exception as e:
            logger.error(f"❌ Error in CC callback: {e}")
            return web.json_response({"status": "error", "reason": str(e)}, status=400)

    async def start(self):
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', self.port)
        await site.start()
        logger.info(f"[UI LAYER] Dashboard available at http://localhost:{self.port}")
        asyncio.create_task(self.bridge_bus_to_ui())

    async def handle_post_intent(self, request):
        try:
            data = await request.json()
            if data.get("text"):
                asyncio.create_task(self.bus_adapter.process_intent(data["text"]))
                return web.json_response({"status": "ok"})
            return web.json_response({"status": "error", "msg": "No text provided"}, status=400)
        except Exception as e:
            return web.json_response({"status": "error", "msg": str(e)}, status=500)

    async def handle_websocket(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.websockets.add(ws)
        try:
            async for _ in ws: pass 
        finally: 
            self.websockets.remove(ws)
        return ws

    async def broadcast(self, msg_type, data):
        msg = json.dumps({"type": msg_type, "data": data})
        for ws in list(self.websockets):
            try: await ws.send_str(msg)
            except: pass

    async def bridge_bus_to_ui(self):
        q_intent = await self.bus.sub("intent.current")
        q_dev = await self.bus.sub("deviation.broadcast")
        q_cmd = await self.bus.sub("command.notify")
        q_kpi = await self.bus.sub("kpi.raw") 

        current_metrics, all_active_ues = {}, {}
        
        while True:
            # Aggregate KPIs
            while not q_kpi.empty():
                payload = (q_kpi.get_nowait()).payload.get("kpi", {})
                current_metrics.update({k: v for k, v in payload.get("CellMetrics", {}).items() if v is not None})
                for ue in payload.get("UEMetrics", []):
                    if uid := ue.get("ue_id"):
                        if uid not in all_active_ues: all_active_ues[uid] = ue
                        else: all_active_ues[uid].update(ue)
                await self.broadcast("kpi", {"cell": current_metrics, "ues": list(all_active_ues.values())})

            while not q_intent.empty(): await self.broadcast("intent", (q_intent.get_nowait()).payload)
            while not q_dev.empty(): await self.broadcast("deviation", (q_dev.get_nowait()).payload)
            while not q_cmd.empty():
                cmd_data = (q_cmd.get_nowait()).payload
                await self.broadcast("command", {"command": cmd_data.get("command"), "params": cmd_data.get("params")})
                
            await asyncio.sleep(0.1)