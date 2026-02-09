#!/usr/bin/env python3
"""
Integrated AI System: Neuro-Symbolic Logic + xApp Simulation.
Restored CLI interface to match deploy_ai.sh requirements.
"""

from __future__ import annotations
import argparse
import asyncio
import csv
import json
import struct
import sys
import threading
import logging
import os
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from aiohttp import web, WSMsgType

# Add parent directory to path
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent.parent.parent))

from ain.bus.mem import MemBus
from ain.agents.utils import make_msg
from ain.agents.minirocket_agent import MinirocketAgent

# --- NEW IMPORTS ---
from demo.ain.agents.otm_logic import OTMToCommandConverter
from demo.ain.brain.ai_logic import HybridAIController, IntentState

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==========================================
# 1. UNIFIED WEB SERVER (Dashboard + API)
# ==========================================
class UnifiedWebServer:
    def __init__(self, port=8080):
        self.port = port
        self.app = web.Application()
        self.websockets = set()
        self.ai_agent = None 
        self.intent_history = [] 
        
        # Locate Dashboard folder
        candidates = [THIS_DIR.parent / "dashboard" / "static", THIS_DIR / "dashboard"]
        self.static_path = next((p for p in candidates if p.exists()), None)
        if not self.static_path:
            self.static_path = THIS_DIR / "dashboard_missing"
            os.makedirs(self.static_path, exist_ok=True)

        self.app.router.add_get('/', self.handle_index)
        self.app.router.add_static('/static', path=str(self.static_path), name='static')
        self.app.router.add_get('/ws', self.handle_websocket)
        self.app.router.add_post('/api/intent', self.handle_post_intent)

    async def start(self):
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', self.port)
        await site.start()

    async def handle_index(self, request):
        f = self.static_path / "index.html"
        if not f.exists(): f = self.static_path / "dashboard.html"
        return web.FileResponse(f) if f.exists() else web.Response(status=404)

    async def handle_post_intent(self, request):
        data = await request.json()
        if data.get("text") and self.ai_agent:
            asyncio.create_task(self.ai_agent.process_intent(data["text"]))
            return web.json_response({"status": "ok"})
        return web.json_response({"status": "error"}, status=400)

    async def handle_websocket(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.websockets.add(ws)
        await ws.send_json({"type": "connected", "message": "Backend Ready"})
        
        # Send history on connect
        for intent in self.intent_history:
             await ws.send_json({"type": "intent", "data": intent})

        try:
            async for msg in ws: pass 
        finally: self.websockets.remove(ws)
        return ws

    async def broadcast(self, msg_type, data):
        # Save Intent History
        if msg_type == "intent":
            # Simple deduplication by ID or overwrite
            existing = next((i for i, x in enumerate(self.intent_history) if x.get('intent_id') == data.get('intent_id')), None)
            if existing is not None:
                self.intent_history[existing] = data
            else:
                self.intent_history.append(data)
                if len(self.intent_history) > 10: self.intent_history.pop(0)

        msg = json.dumps({"type": msg_type, "data": data})
        for ws in list(self.websockets):
            try: await ws.send_str(msg)
            except: pass

# ==========================================
# 2. AI WRAPPER AGENT (The Stateful Brain)
# ==========================================
class AIWrapperAgent:
    def __init__(self, bus: MemBus, server: UnifiedWebServer, tcp_server: XAppTCPServer):
        self.bus = bus
        self.server = server
        self.tcp_server = tcp_server 
        self.controller = HybridAIController()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.current_metrics = {}
        self.all_active_ues = {}
        self._last_state = IntentState.MONITORING  

    async def run(self):
        q_kpi = await self.bus.sub("kpi.raw")
        q_dev = await self.bus.sub("deviation.detected")
        
        logger.info("[WRAPPER] AI Wrapper Loop Started.")

        while True:
            # --- 0. State Transition Logging ---
            if self.controller.state != self._last_state:
                logger.info(f"[STATE CHANGE] {self._last_state.name} -> {self.controller.state.name}")
                self._last_state = self.controller.state

            # --- 1. Process KPIs (DRAIN THE QUEUE) ---
            # Fix: Process ALL pending messages, not just one.
            while not q_kpi.empty():
                try:
                    msg = q_kpi.get_nowait()
                    payload = msg.payload.get("kpi", {})
                    
                    # Merge Cell Metrics
                    for k, v in payload.get("CellMetrics", {}).items():
                        if v is not None: 
                            self.current_metrics[k] = v
                    
                    # Merge UE Metrics
                    for ue in payload.get("UEMetrics", []):
                        uid = ue.get("ue_id")
                        if uid:
                            # If new UE, add it. If existing, update fields.
                            if uid not in self.all_active_ues: 
                                self.all_active_ues[uid] = ue
                            else:
                                for k, v in ue.items():
                                    if v is not None: self.all_active_ues[uid][k] = v
                except Exception as e:
                    logger.error(f"KPI processing error: {e}")
                    break

            # Broadcast Aggregated State (Sticky)
            # We send this even if no new data came in, to keep the UI alive with last known values
            await self.server.broadcast("kpi", {
                "cell": self.current_metrics, 
                "ues": list(self.all_active_ues.values())
            })

            # --- 2. Process Deviations ---
            while not q_dev.empty():
                try:
                    msg = q_dev.get_nowait()
                    dev = msg.payload
                    
                    # Check Controller Logic
                    accepted = self.controller.process_deviation(dev)
                    
                    if accepted:
                        logger.info(f"[MEMBUS] Anomaly Accepted: {dev['metric']}. Triggering LLM.")
                        await self.server.broadcast("deviation", dev)
                    else:
                        # Only log ignored if we are debugging, otherwise it spams
                        if self.controller.state == IntentState.MONITORING:
                             logger.info(f"[MEMBUS] Anomaly IGNORED (State mismatch): {dev['metric']}")
                except: break

            # --- 3. AI Heartbeat Tick ---
            try:
                obs_copy = dict(self.current_metrics)
                actions = self.controller.step(obs_copy)
                
                for act in actions:
                    a_type = act.get("type")

                    if a_type == "LLM_REQUEST_PLAN":
                        logger.info(f"[WRAPPER] Launching background LLM thread...")
                        asyncio.create_task(self._background_anomaly_task())
                    
                    elif a_type == "INTENT_UPDATE":
                        # Send Dashboard update
                        payload = {
                            "intent_id": "active_intent",
                            "type": act.get('intent_name'),
                            "triggering_intents": act.get('triggering_intents', []),
                            "procedure": act.get('procedure', []),
                            "scope": "CELL"
                        }
                        await self.bus.pub("intent.current", make_msg("intent.current", "INTENT", "v1", payload))
                        await self.server.broadcast("intent", payload)

                    elif a_type == "EXECUTE_CLEANUP":
                        logger.info(f"[WRAPPER] Executing Withdrawal/Cleanup OTM.")
                        await self._execute_otm(act.get('otm'))

                    elif a_type == "UI_CLEAR":
                         await self.server.broadcast("intent", {}) # Clear UI

            except Exception as e:
                logger.error(f"AI Step Error: {e}", exc_info=True)
            
            # Tick Rate: 0.5s is snappier than 1.0s
            await asyncio.sleep(0.5)

    async def _background_anomaly_task(self): # Ensure this has the underscore!
        logger.info("[WRAPPER] Background LLM reasoning task started.")
        try:
            dev = self.controller.current_deviation
            snapshot = dict(self.current_metrics)
            loop = asyncio.get_running_loop()
            
            # 1. Run LLM
            outcome = await loop.run_in_executor(
                self.executor, self.controller.parser.devise_plan, dev['metric'], dev['value'], snapshot
            )
            
            # 2. Process OTM logic (This now looks at the procedure list)
            otm = self.controller.apply_llm_plan(outcome)
            
            # 3. Execute
            await self._execute_otm(otm)
        except Exception as e:
            logger.error(f"Error in background task: {e}", exc_info=True)

    async def _execute_otm(self, otm):
        meid = "gnb:131-133-31000000"
        node_id = 2 
        commands = OTMToCommandConverter.convert(otm, meid, default_node_id=node_id)
        
        if commands:
            logger.info(f"[ACTUATOR] Sending {len(commands)} commands...")
            for client_id in list(self.tcp_server.clients.keys()):
                for cmd in commands:
                    logger.info(f" -> Sending {cmd['cmd']['cmd']} to {client_id}")
                    await self.tcp_server.send_command(client_id, cmd)
                    await asyncio.sleep(0.1)
            # Broadcast back to dashboard
            await self.server.broadcast("command", {"command": "EXECUTE_OTM", "params": commands})
        else:
            logger.warning("[ACTUATOR] No valid commands in OTM.")

    async def process_intent(self, text):
        # ... (Same as previous, just ensure state transition logs happen in main loop) ...
        logger.info(f"[API] Manual Intent: '{text}'")
        loop = asyncio.get_running_loop()
        outcome = await loop.run_in_executor(self.executor, self.controller.parser.parse_user_intent, text)
        otm = self.controller.process_manual_request(outcome, text)
        await self._execute_otm(otm)

# ==========================================
# 3. INFRASTRUCTURE (KPI Adapter & TCP)
# ==========================================
class XAppKPIAdapter:
    @staticmethod
    def convert_xapp_kpi_to_internal(xapp_kpi, meid):
        kpi_data = xapp_kpi.get("kpi", {})
        cell_metrics = {}
        ue_metrics = []
        def clean(n): return n.replace('.', '_').lower()
        
        # Process UE fragments
        for ue in kpi_data.get("ues", []):
            ue_id = ue.get("ue_id") or ue.get("ueId")
            if not ue_id: continue
            m_dict = {"ue_id": str(ue_id), "cell_id": kpi_data.get("cell_id", "CELL_001")}
            for m in ue.get("measurements", []):
                n, v = clean(m.get("name","")), float(m.get("value",0))
                if "delay" in n and "pdcp" in n: m_dict["UE_DRB_PdcpSduDelayDl_UEID"] = v
                elif "throughput" in n or "thp" in n: m_dict["UE_DRB_UEThpDl_UEID"] = v * 1e6
                elif "prb" in n: m_dict["UE_RRU_PrbUsedDl_UEID"] = v
            ue_metrics.append(m_dict)
            
        # Process Cell measurements
        for m in kpi_data.get("measurements", []):
            n, v = clean(m.get("name","")), float(m.get("value",0))
            if "delay" in n and "ue" not in n: cell_metrics["DRB_PdcpSduDelayDl"] = v
            elif "prb" in n: cell_metrics["RRU_PrbUsedDl"] = v
            elif "throughput" in n or "thp" in n: cell_metrics["thr_dl_bps"] = v * 1e6
            elif "active" in n and "ue" in n: cell_metrics["DRB_MeanActiveUeDl"] = v
            
        cell_metrics["cell_id"] = kpi_data.get("cell_id", "CELL_001")
        return {"timestamp": datetime.now(timezone.utc).isoformat(), "CellMetrics": cell_metrics, "UEMetrics": ue_metrics}

class XAppTCPServer:
    def __init__(self, host="0.0.0.0", port=6000, bus=None):
        self.host, self.port, self.bus = host, port, bus
        self.adapter = XAppKPIAdapter()
        self.clients = {}  # FIX: Stores client connections

    async def start(self):
        self.server = await asyncio.start_server(self._handle, self.host, self.port)
        logger.info(f"xApp TCP server started on {self.host}:{self.port}")

    async def stop(self):
        if self.server: self.server.close(); await self.server.wait_closed()

    async def _handle(self, reader, writer):
        # FIX: Register the client so we can send commands back
        addr = writer.get_extra_info('peername')
        client_id = f"{addr[0]}:{addr[1]}"
        self.clients[client_id] = writer
        logger.info(f"[TCP] New xApp Client connected: {client_id}")

        try:
            while True:
                head = await reader.read(4)
                if not head: break
                body_len = struct.unpack("!I", head)[0]
                if body_len == 0: continue
                
                body = await reader.read(body_len)
                msg = json.loads(body.decode("utf-8"))
                
                if msg.get("type") == "kpi":
                    internal = self.adapter.convert_xapp_kpi_to_internal(msg, msg.get("meid"))
                    if self.bus: await self.bus.pub("kpi.raw", make_msg("kpi.raw", "KPI", "v1", {"kpi": internal}))
        except Exception as e:
            logger.error(f"[TCP] Error with client {client_id}: {e}")
        finally: 
            if client_id in self.clients:
                del self.clients[client_id]
            writer.close()
            logger.info(f"[TCP] Client disconnected: {client_id}")

    async def send_command(self, client_id: str, command: Dict[str, Any]) -> bool:
        if client_id not in self.clients: return False
        try:
            writer = self.clients[client_id]
            response_json = json.dumps(command)
            response_bytes = response_json.encode("utf-8")
            length_header = struct.pack("!I", len(response_bytes))
            writer.write(length_header + response_bytes)
            await writer.drain()
            logger.info(f"[TCP] Sent command to {client_id}")
            return True
        except Exception as e:
            logger.error(f"Send Error: {e}")
            return False

# ==========================================
# 4. MAIN ORCHESTRATOR
# ==========================================
async def run_ai_loop_with_membus(tcp_server, target_metric, args):
    bus = MemBus()
    tcp_server.bus = bus
    web_server = UnifiedWebServer(port=args.web_port)
    ai_wrapper = AIWrapperAgent(bus, web_server, tcp_server)
    web_server.ai_agent = ai_wrapper

    # Sensors
    minirocket_agents = []
    if args.minirocket_gnb_model or args.minirocket_model:
        gnb_m = "DRB_PdcpSduDelayDl" if target_metric == "delay_p95_ms" else target_metric
        path = args.minirocket_gnb_model or args.minirocket_model
        minirocket_agents.append(MinirocketAgent(bus, model_path=path, metric=gnb_m, window_size=128))
    
    if args.minirocket_ue_model or args.minirocket_model:
        ue_m = "UE_DRB_PdcpSduDelayDl_UEID" if target_metric in ("delay_p95_ms", "DRB_PdcpSduDelayDl") else target_metric
        path = args.minirocket_ue_model or args.minirocket_model
        minirocket_agents.append(MinirocketAgent(bus, model_path=path, metric=ue_m, window_size=128))

    tasks = [
        asyncio.create_task(tcp_server.start()), 
        asyncio.create_task(web_server.start()),
        asyncio.create_task(ai_wrapper.run())  # <--- FIX: Added AI Wrapper Loop
    ]
    for agent in minirocket_agents: tasks.append(asyncio.create_task(agent.run()))

    logger.info(f"System fully orchestrated. Go to http://localhost:{args.web_port}")
    try: await asyncio.gather(*tasks)
    except KeyboardInterrupt: await tcp_server.stop()

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6000)
    parser.add_argument("--web-port", type=int, default=8080)
    parser.add_argument("--target-metric", default="DRB_PdcpSduDelayDl")
    parser.add_argument("--minirocket-model", default=None)
    parser.add_argument("--minirocket-gnb-model", default=None)
    parser.add_argument("--minirocket-ue-model", default=None)
    parser.add_argument("--log-level", type=str, default="all") 
    
    args = parser.parse_args()
    tcp_server = XAppTCPServer(host=args.host, port=args.port)
    await run_ai_loop_with_membus(tcp_server, args.target_metric, args)

if __name__ == "__main__":
    asyncio.run(main())