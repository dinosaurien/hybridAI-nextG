import asyncio
import json
import logging
from aiohttp import web
from pathlib import Path
from core.bus.mem import MemBus
from core.bus.messages import make_msg

logger = logging.getLogger(__name__)

# The front-end HTML and CSS/JS is based on the https://github.com/StartBootstrap/startbootstrap-sb-admin repo
# Although heavily modified to match requirements of this work, the original license is MIT so this is fine for our needs.


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

        self.app.router.add_get('/ws', self.handle_websocket)
        self.app.router.add_post('/api/intent', self.handle_post_intent)
        self.app.router.add_get('/', self.redirect_to_dashboard)
        self.app.router.add_static('/static/', self.static_dir, name='static')
        self.app.router.add_post('/api/cc-callback', self.handle_cc_callback)
        self.app.router.add_post('/api/schedule/cancel', self.handle_cancel_schedule)
            

    async def redirect_to_dashboard(self, request):
        """Redirect root URL to the dashboard HTML file."""
        return web.HTTPFound('/static/dashboard.html')
    
    async def handle_index(self, request):
        f = self.static_dir / "dashboard.html"
        if f.exists():
            return web.FileResponse(f)
        else:
            return web.Response(text="dashboard.html not found in static folder", status=404)
    
    async def handle_cancel_schedule(self, request):
        """Cancel a scheduled intent by ID."""
        try:
            data = await request.json()
            schedule_id = data.get("schedule_id")
            if schedule_id:
                await self.bus.pub("schedule.cancel", make_msg("ui", "CANCEL", "v1", {"schedule_id": schedule_id}))
                return web.json_response({"status": "ok"})
            return web.json_response({"status": "error", "msg": "No schedule_id provided"}, status=400)
        except Exception as e:
            return web.json_response({"status": "error", "msg": str(e)}, status=500)

    async def handle_cc_callback(self, request):
        try:
            
            raw_body = await request.text()
            
            data = json.loads(raw_body)
            
            await self.broadcast("cc_notification", data)
            return web.json_response({"status": "acknowledged"})
        except Exception as e:
            logger.error(f"Error in CC callback: {e}")
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
            async for raw_msg in ws:
                if raw_msg.type == web.WSMsgType.TEXT:
                    try:
                        data = json.loads(raw_msg.data)
                        # Route intent messages from the frontend to the MemBus
                        if data.get("type") == "intent" and data.get("text"):
                            asyncio.create_task(self.bus_adapter.process_intent(data["text"]))
                    except json.JSONDecodeError:
                        logger.debug(f"[UI LAYER] Non-JSON WebSocket message ignored")
        finally:
            self.websockets.discard(ws)
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
        q_schedule = await self.bus.sub("schedule.update")
        q_procedure = await self.bus.sub("procedure.update")

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
            while not q_schedule.empty(): await self.broadcast("schedule", (q_schedule.get_nowait()).payload)
            while not q_procedure.empty(): await self.broadcast("procedure", (q_procedure.get_nowait()).payload)

            await asyncio.sleep(0.1)