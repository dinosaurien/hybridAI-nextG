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
from demo.ain.utils import knowledge_base

# Add parent directory to path
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent.parent.parent))

from ain.bus.mem import MemBus
from ain.bus.messages import make_msg
from ain.agents.minirocket_agent import MinirocketAgent
from demo.ain.agents.cognitive_agent import CognitiveAgent
from demo.ain.agents.orchestrator_agent import OrchestratorAgent
from demo.ain.agents.actuator_agent import ActuatorAgent
from demo.ain.utils.otm_utils import OTMToCommandConverter


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class XAppKPIAdapter:
    """Converts xApp KPI format to internal format expected by RLObserver."""
    
    @staticmethod
    def convert_xapp_kpi_to_internal(xapp_kpi: Dict[str, Any], meid: str) -> Dict[str, Any]:
        """Convert xApp KPI format to internal format."""
        kpi_data = xapp_kpi.get("kpi", {})
        
        cell_metrics = {}
        ue_metrics = []
        
        # Extract from measurements array
        measurements = kpi_data.get("measurements", [])
        ues = kpi_data.get("ues", [])
        
        # Extract UE metrics from ues array
        # Based on ue_kpis.csv: timestamp,meid,cell_id,ue_id,UE_PDCP_Delay_DL_ms,...
        if ues:
            for ue in ues:
                ue_metric = {}
                ue_id = ue.get("ue_id") or ue.get("ueId") or ue.get("id")
                if not ue_id:
                    continue
                
                ue_metric["ue_id"] = str(ue_id)
                ue_metric["cell_id"] = ue.get("cell_id") or ue.get("cellId") or cell_metrics.get("cell_id", "unknown")
                
                # Extract UE-specific metrics from raw fields if available (backward compatibility)
                # But prefer measurements array which has the correct names
                # These checks are for backward compatibility with old format
                if "UE_PDCP_Delay_DL_ms" in ue:
                    ue_metric["UE_DRB_PdcpSduDelayDl_UEID"] = float(ue["UE_PDCP_Delay_DL_ms"])
                if "DRB_EstabSucc_5QI_UEID" in ue:
                    ue_metric["UE_DRB_EstabSucc_5QI_UEID"] = float(ue["DRB_EstabSucc_5QI_UEID"])
                if "TB_TotNbrDlInitial_Qpsk_UEID" in ue:
                    ue_metric["UE_TB_TotNbrDlInitial_Qpsk_UEID"] = int(ue["TB_TotNbrDlInitial_Qpsk_UEID"])
                if "TB_TotNbrDlInitial_64Qam_UEID" in ue:
                    ue_metric["UE_TB_TotNbrDlInitial_64Qam_UEID"] = int(ue["TB_TotNbrDlInitial_64Qam_UEID"])
                if "UE_PRB_Used_DL" in ue:
                    ue_metric["UE_RRU_PrbUsedDl_UEID"] = float(ue["UE_PRB_Used_DL"])
                if "UE_Throughput_DL_Mbps" in ue:
                    ue_metric["UE_DRB_UEThpDl_UEID"] = float(ue["UE_Throughput_DL_Mbps"]) * 1e6  # Convert Mbps to bps
                
                # Extract from nested measurements if available
                # Map to actual CSV column names: UE_DRB_PdcpSduDelayDl_UEID, UE_DRB_UEThpDl_UEID, etc.
                ue_measurements = ue.get("measurements", [])
                for m in ue_measurements:
                    name = m.get("name", "")
                    value = m.get("value", 0)
                    # Convert dot notation to underscores for matching
                    name_normalized = name.replace('.', '_').lower()
                    
                    # Map to actual CSV column names
                    if "drb_pdcpsdudelaydl_ueid" in name_normalized or ("delay" in name_normalized and "pdcp" in name_normalized and "ue" in name_normalized):
                        ue_metric["UE_DRB_PdcpSduDelayDl_UEID"] = float(value)
                    elif "drb_uethpdl_ueid" in name_normalized or ("throughput" in name_normalized and "ue" in name_normalized and "dl" in name_normalized):
                        ue_metric["UE_DRB_UEThpDl_UEID"] = float(value) * 1e6  # Convert Mbps to bps if needed
                    elif "rru_prbuseddl_ueid" in name_normalized or ("prb" in name_normalized and "used" in name_normalized and "ue" in name_normalized):
                        ue_metric["UE_RRU_PrbUsedDl_UEID"] = float(value)
                    elif "drb_blerdl_ueid" in name_normalized or ("bler" in name_normalized and "dl" in name_normalized and "ue" in name_normalized):
                        ue_metric["UE_DRB_BlerDl_UEID"] = float(value)
                    elif "drb_estabsucc_5qi_ueid" in name_normalized or ("estab" in name_normalized and "succ" in name_normalized and "ue" in name_normalized):
                        ue_metric["UE_DRB_EstabSucc_5QI_UEID"] = float(value)
                    elif "tb_totnbrdlinitial_qpsk_ueid" in name_normalized:
                        ue_metric["UE_TB_TotNbrDlInitial_Qpsk_UEID"] = int(value)
                    elif "tb_totnbrdlinitial_16qam_ueid" in name_normalized:
                        ue_metric["UE_TB_TotNbrDlInitial_16Qam_UEID"] = int(value)
                    elif "tb_totnbrdlinitial_64qam_ueid" in name_normalized:
                        ue_metric["UE_TB_TotNbrDlInitial_64Qam_UEID"] = int(value)
                
                if ue_metric:
                    ue_metrics.append(ue_metric)
        
        if measurements:
            for m in measurements:
                name = m.get("name", "")
                value = m.get("value", 0)
                # Convert dot notation to underscores for matching
                name_normalized = name.replace('.', '_').lower()
                
                # Map xApp measurement names to actual CSV column names
                # gNB level: DRB_PdcpSduDelayDl, RRU_PrbUsedDl, DRB_MeanActiveUeDl, etc.
                if "drb_pdcpsdudelaydl" in name_normalized and "ueid" not in name_normalized:
                    cell_metrics["DRB_PdcpSduDelayDl"] = float(value)
                elif "rru_prbuseddl" in name_normalized or ("prb" in name_normalized and "used" in name_normalized and "dl" in name_normalized):
                    cell_metrics["RRU_PrbUsedDl"] = float(value)
                elif "drb_meanactiveuedl" in name_normalized or ("mean" in name_normalized and "active" in name_normalized and "ue" in name_normalized):
                    cell_metrics["DRB_MeanActiveUeDl"] = float(value)
                elif "tb_totnbrdlinitial_qpsk" in name_normalized:
                    cell_metrics["TB_TotNbrDlInitial_Qpsk"] = int(value)
                elif "tb_totnbrdlinitial_16qam" in name_normalized:
                    cell_metrics["TB_TotNbrDlInitial_16Qam"] = int(value)
                elif "tb_totnbrdlinitial_64qam" in name_normalized:
                    cell_metrics["TB_TotNbrDlInitial_64Qam"] = int(value)
                elif "throughput" in name.lower() or "thr" in name.lower():
                    if "dl" in name.lower():
                        cell_metrics["thr_dl_bps"] = float(value) * 1e6
                    elif "ul" in name.lower():
                        cell_metrics["thr_ul_bps"] = float(value) * 1e6
                elif "bler" in name.lower():
                    if "dl" in name.lower():
                        cell_metrics["bler_dl"] = float(value) / 100.0
                    elif "ul" in name.lower():
                        cell_metrics["bler_ul"] = float(value) / 100.0
                elif "cqi" in name.lower():
                    cell_metrics["cqi_avg"] = float(value)
                elif "mcs" in name.lower():
                    if "dl" in name.lower():
                        cell_metrics["mcs_dl_avg"] = int(value)
                    elif "ul" in name.lower():
                        cell_metrics["mcs_ul_avg"] = int(value)
                elif "PRB_Used_DL" in name or ("prb" in name.lower() and "used" in name.lower()):
                    cell_metrics["PRB_Used_DL"] = float(value)
                elif "PRB_Total_DL" in name or ("prb" in name.lower() and ("total" in name.lower() or "avail" in name.lower())):
                    cell_metrics["PRB_Total_DL"] = float(value)
                elif "Mean_Active_UEs_DL" in name or ("active" in name.lower() and "ue" in name.lower()):
                    cell_metrics["Mean_Active_UEs_DL"] = int(value)
                elif "DL_TB_QPSK_Count" in name:
                    cell_metrics["DL_TB_QPSK_Count"] = int(value)
                elif "DL_TB_64QAM_Count" in name:
                    cell_metrics["DL_TB_64QAM_Count"] = int(value)
        
        # Extract from raw fields if available (backward compatibility)
        # Map old names to new CSV column names
        if "UE_PDCP_Delay_DL_ms" in kpi_data:
            cell_metrics["DRB_PdcpSduDelayDl"] = float(kpi_data["UE_PDCP_Delay_DL_ms"])
        if "PRB_Used_DL" in kpi_data:
            cell_metrics["RRU_PrbUsedDl"] = float(kpi_data["PRB_Used_DL"])
        if "PRB_Total_DL" in kpi_data or "PRB_Available_DL" in kpi_data:
            # Store as RRU_PrbUsedDl if we need total, but typically we just use RRU_PrbUsedDl
            pass  # PRB total not in feature list
        if "Mean_Active_UEs_DL" in kpi_data:
            cell_metrics["DRB_MeanActiveUeDl"] = int(kpi_data["Mean_Active_UEs_DL"])
        if "DL_TB_QPSK_Count" in kpi_data:
            cell_metrics["TB_TotNbrDlInitial_Qpsk"] = int(kpi_data["DL_TB_QPSK_Count"])
        if "DL_TB_64QAM_Count" in kpi_data:
            cell_metrics["TB_TotNbrDlInitial_64Qam"] = int(kpi_data["DL_TB_64QAM_Count"])
        
        # Compute PRB_Used_DL_ratio if we have both values
        if "PRB_Used_DL" in cell_metrics and "PRB_Total_DL" not in cell_metrics:
            # Default to 100 RBs if total not available (common default)
            cell_metrics["PRB_Total_DL"] = 100.0
        
        # Normalize and set cell_id
        cell_id_raw = kpi_data.get("cellObjectID") or kpi_data.get("cell_id") or "CELL_001"
        # Normalize cell_id: convert numeric strings to CELL_XXX format
        if cell_id_raw and str(cell_id_raw).isdigit():
            cell_id = f"CELL_{cell_id_raw}"
        elif cell_id_raw and cell_id_raw.startswith("CELL_"):
            cell_id = cell_id_raw
        elif cell_id_raw != "unknown":
            cell_id = f"CELL_{cell_id_raw}" if not cell_id_raw.startswith("CELL_") else cell_id_raw
        else:
            cell_id = "CELL_001"
        if "cell_id" not in cell_metrics:
            cell_metrics["cell_id"] = cell_id
        
        # Extract and store node_id (important for fragment merging)
        node_id = kpi_data.get("node_id") or kpi_data.get("nodeId") or xapp_kpi.get("node_id")
        if node_id is not None:
            cell_metrics["node_id"] = int(node_id)
        # If not in KPI data, try to infer from cell_id pattern
        elif "node_id" not in cell_metrics:
            # Infer from cell_id: CELL_1111 -> node 1, CELL_2222 -> node 2, etc.
            if cell_id.startswith("CELL_"):
                numeric_part = cell_id.replace("CELL_", "")
                if numeric_part.isdigit() and len(numeric_part) > 0:
                    cell_metrics["node_id"] = int(numeric_part[0])  # First digit
                else:
                    cell_metrics["node_id"] = 2  # Default to gNB node
            elif cell_id == "unknown":
                cell_metrics["node_id"] = 2  # Default to gNB node for unknown
            else:
                cell_metrics["node_id"] = 2  # Default to gNB node
        
        # Note: We intentionally do NOT set defaults for missing metrics.
        # The observer will handle missing features using NaN and feature completeness checks.
        
        # Build internal format
        internal_kpi = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "Header": {
                "ric_instance_id": meid,
                "function_id": "kpm_func_v1",
                "kpm_version": "2.0",
                "granularity_period_ms": 1000,
                "window_start": datetime.now(timezone.utc).isoformat(),
                "window_end": datetime.now(timezone.utc).isoformat(),
                "sequence_number": int(datetime.now().timestamp() * 1000) % 1000000,
            },
            "CellMetrics": cell_metrics,
            "UEMetrics": ue_metrics,
        }
        
        return internal_kpi

# This code allows us to accept user input from the UI, similar logic-wise to before but instead of static html server
class UnifiedWebServer:
    def __init__(self, port=8080):
        self.port = port
        self.app = web.Application()
        self.websockets = set()
        self.ai_agent = None 
        self.intent_history = [] 
        
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
        
        for intent in self.intent_history:
             await ws.send_json({"type": "intent", "data": intent})

        try:
            async for msg in ws: pass 
        finally: self.websockets.remove(ws)
        return ws

    async def broadcast(self, msg_type, data):
        # Save Intent History
        if msg_type == "intent":
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


class OTMToCommandConverter:
    """
    Directly converts the declarative OTM structure into xApp-specific 
    control commands using the 'kpi' and 'threshold' fields.
    """
    @staticmethod
    def convert(otm: Dict[str, Any], meid: str, default_node_id: int = 2) -> List[Dict[str, Any]]:
        commands = []
        # The OTM follows the strict Constraint structure
        constraints = otm.get("constraints", [])
        
        for constraint in constraints:
            # We used the 'kpi' field to store the technical parameter name
            kpi_id = constraint.get("kpi")
            val = constraint.get("threshold")
            
            cmd_body = None
            if kpi_id == "dl_mcs_max":
                cmd_body = {"cmd": "set-mcs", "node": default_node_id, "mcs": int(val)}
            elif kpi_id == "tx_power_dbm":
                cmd_body = {"cmd": "set-enb-txpower", "node": default_node_id, "txPowerDbm": float(val)}
            elif kpi_id == "prb_weight":
                # Convert 0.8 weight to '80' bandwidth command
                cmd_body = {"cmd": "set-bandwidth", "node": default_node_id, "bandwidth": int(100 * val)}
            
            if cmd_body:
                commands.append({
                    "type": "control",
                    "meid": meid,
                    "cmd": cmd_body
                })
        return commands
    

class WebBusAdapter:
    # Adapts the WebServer's call to 'self.ai_agent.process_intent(text)' into a Membus publish event.
    def __init__(self, bus: MemBus):
        self.bus = bus

    async def process_intent(self, text: str):
        logger.info(f"[WEB] Forwarding manual intent: {text}")
        await self.bus.pub("ui.input", make_msg("ui", "REQ", "v1", {"text": text}))


class XAppTCPServer:
    """TCP server for xApp communication that publishes to membus."""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 6000, bus: Optional[MemBus] = None, kpi_csv_file: Optional[str] = None, commands_enabled: bool = True):
        self.host = host
        self.port = port
        self.server: Optional[asyncio.Server] = None
        self.clients: Dict[str, asyncio.StreamWriter] = {}
        self.bus = bus
        self.meid_map: Dict[str, str] = {}
        self.kpi_adapter = XAppKPIAdapter()
        self.commands_enabled = commands_enabled
        # UE and node tracking
        self.cell_to_node_map: Dict[str, int] = {}  # cell_id -> node_id
        self.ue_to_node_map: Dict[str, int] = {}  # ue_id -> node_id
        self.ue_to_cell_map: Dict[str, str] = {}  # ue_id -> cell_id
        self.active_ues: Dict[str, Dict[str, Any]] = {}  # ue_id -> latest UE metrics
        self.client_node_map: Dict[str, int] = {}  # client_id -> default node_id
        # CSV logging - default to project root directory
        if kpi_csv_file is None:
            # Find project root (go up from src/demo/ain/RL_demo to project root)
            project_root = THIS_DIR.parent.parent.parent.parent
            self.kpi_csv_file = project_root / "kpms.csv"
        else:
            self.kpi_csv_file = Path(kpi_csv_file)
        self.kpi_csv_initialized = False
        self.kpi_csv_fieldnames = None  # Cached fieldnames to avoid reading file every time
        self.kpi_csv_lock = threading.Lock()  # Lock for thread-safe CSV operations
        self.kpi_csv_rewrite_pending = False  # Flag to track if rewrite is in progress
        self.kpi_csv_write_count = 0  # Counter for periodic flushing
        self.KPI_CSV_FLUSH_INTERVAL = 10  # Flush every 10 writes
        self._init_kpi_csv()
    
    def _init_kpi_csv(self):
        """Initialize KPI CSV file with base headers if it doesn't exist."""
        base_fieldnames = ['timestamp', 'meid', 'cell_id', 'node_id', 'format']
        if not self.kpi_csv_file.exists():
            with open(self.kpi_csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                # Only write base columns - measurement columns will be added dynamically
                writer.writerow(base_fieldnames)
            self.kpi_csv_fieldnames = base_fieldnames
            self.kpi_csv_initialized = True
            logger.info(f"Initialized KPI CSV file: {self.kpi_csv_file}")
        else:
            # Cache existing fieldnames to avoid reading file every time
            try:
                with open(self.kpi_csv_file, 'r') as f:
                    reader = csv.DictReader(f)
                    self.kpi_csv_fieldnames = list(reader.fieldnames) if reader.fieldnames else base_fieldnames
            except Exception as e:
                logger.warning(f"Could not read existing CSV fieldnames: {e}, using defaults")
                self.kpi_csv_fieldnames = base_fieldnames
            self.kpi_csv_initialized = True
    
    def _rewrite_csv_with_new_columns(self, new_fieldnames: List[str]):
        """Background thread function to rewrite CSV with new columns (non-blocking)"""
        try:
            existing_rows = []
            if self.kpi_csv_file.exists():
                with open(self.kpi_csv_file, 'r') as f:
                    reader = csv.DictReader(f)
                    existing_rows = list(reader)
            
            # Rewrite file with new header
            with open(self.kpi_csv_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=new_fieldnames, extrasaction='ignore')
                writer.writeheader()
                for row in existing_rows:
                    writer.writerow(row)
            
            # Update cached fieldnames
            with self.kpi_csv_lock:
                self.kpi_csv_fieldnames = new_fieldnames
                self.kpi_csv_rewrite_pending = False
            logger.debug(f"Completed CSV rewrite with {len(new_fieldnames)} columns")
        except Exception as e:
            logger.error(f"Error rewriting CSV: {e}", exc_info=True)
            with self.kpi_csv_lock:
                self.kpi_csv_rewrite_pending = False
    
    def _write_kpi_to_csv(self, kpi_data: Dict[str, Any], meid: str, cell_id: str, node_id: Optional[int] = None, is_ue_data: bool = False):
        """Write KPI data to CSV file with dynamic column handling.
        
        Args:
            kpi_data: KPI data dictionary
            meid: Management entity ID
            cell_id: Cell ID (or UE ID if is_ue_data=True)
            node_id: Node ID
            is_ue_data: If True, prefix metrics with "UE_" to distinguish from cell-level
        """
        if not self.kpi_csv_initialized:
            self._init_kpi_csv()
        
        try:
            timestamp = datetime.now(timezone.utc).isoformat()
            measurements = kpi_data.get("measurements", [])
            
            # Extract ALL metrics from measurements array (dynamic, like relay server)
            metrics = {}
            for m in measurements:
                name = m.get("name", "")
                if not name:
                    # Try ID if name not available
                    meas_id = m.get("id")
                    if meas_id is not None:
                        name = f"id_{meas_id}"
                    else:
                        continue
                
                value = m.get("value", "")
                # Normalize name (replace dots/spaces with underscores, like relay server)
                csv_name = name.replace(".", "_").replace(" ", "_")
                # Prefix with UE_ if this is UE data
                if is_ue_data and not csv_name.startswith("UE_"):
                    csv_name = f"UE_{csv_name}"
                metrics[csv_name] = value
            
            # Use cached fieldnames (avoid reading file every time)
            with self.kpi_csv_lock:
                base_fieldnames = ['timestamp', 'meid', 'cell_id', 'node_id', 'format']
                existing_fieldnames = self.kpi_csv_fieldnames or base_fieldnames
                
                # Get all metric names (from cached fieldnames + new metrics)
                all_metric_names = set()
                if existing_fieldnames:
                    # Get metric columns (everything except base columns)
                    all_metric_names = set(existing_fieldnames) - set(base_fieldnames)
                
                # Add new metric names
                all_metric_names.update(metrics.keys())
                
                # Sort metric names for consistent column order
                sorted_metric_names = sorted(all_metric_names)
                fieldnames = base_fieldnames + sorted_metric_names
                
                # If we have new columns, start background rewrite
                if set(fieldnames) != set(existing_fieldnames):
                    if not self.kpi_csv_rewrite_pending:
                        self.kpi_csv_rewrite_pending = True
                        # Start background thread for rewrite
                        threading.Thread(
                            target=self._rewrite_csv_with_new_columns,
                            args=(fieldnames,),
                            daemon=True
                        ).start()
                    # Update cached fieldnames immediately (rewrite will complete in background)
                    self.kpi_csv_fieldnames = fieldnames
                
                # Write new row (extrasaction='ignore' will skip new columns until rewrite completes)
                row = {
                    'timestamp': timestamp,
                    'meid': meid,
                    'cell_id': cell_id,
                    'node_id': node_id if node_id is not None else '',
                    'format': kpi_data.get("format", "F1"),
                }
                
                # Add all metrics
                row.update(metrics)
            
            # Write to file (outside lock to minimize lock time, but fieldnames are already cached)
            with open(self.kpi_csv_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.kpi_csv_fieldnames, extrasaction='ignore')
                writer.writerow(row)
                
                # Periodic flush to reduce I/O overhead
                self.kpi_csv_write_count += 1
                if self.kpi_csv_write_count % self.KPI_CSV_FLUSH_INTERVAL == 0:
                    f.flush()
                
        except Exception as e:
            logger.error(f"Error writing KPI to CSV: {e}", exc_info=True)
        
    async def start(self):
        """Start the TCP server."""
        self.server = await asyncio.start_server(
            self._handle_client, self.host, self.port
        )
        logger.info(f"xApp TCP server started on {self.host}:{self.port}")
        
    async def stop(self):
        """Stop the TCP server."""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            
    async def _recv_all(self, reader: asyncio.StreamReader, n: int) -> Optional[bytes]:
        """Receive exactly n bytes."""
        data = b""
        while len(data) < n:
            chunk = await reader.read(n - len(data))
            if not chunk:
                return None
            data += chunk
        return data
            
    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle incoming client connections."""
        client_addr = writer.get_extra_info('peername')
        client_id = f"{client_addr[0]}:{client_addr[1]}"
        
        logger.info(f"xApp client connected: {client_id}")
        self.clients[client_id] = writer
        
        try:
            while True:
                header = await self._recv_all(reader, 4)
                if header is None:
                    logger.info(f"xApp client {client_id} disconnected")
                    break
                    
                length = struct.unpack("!I", header)[0]
                
                if length == 0 or length > 1024 * 1024:
                    logger.error(f"Invalid frame length={length}, closing connection")
                    break
                
                body = await self._recv_all(reader, length)
                if body is None:
                    logger.error("Incomplete frame body")
                    break
                
                try:
                    text = body.decode("utf-8", errors="replace")
                    message = json.loads(text)
                    
                    msg_type = message.get("type", "unknown")
                    # TODO: restore logging with log levels
                    #if should_log(LOG_KPI):
                    #    logger.info(f"[KPI] Received {msg_type} message from {client_id}")
                    
                    if msg_type == "kpi":
                        meid = message.get("meid", "unknown")
                        self.meid_map[client_id] = meid
                        
                        # Extract and track UE/node information
                        kpi_data = message.get("kpi", {})
                        
                        # Debug: Log raw KPI structure (first few messages only to avoid spam)
                        if not hasattr(self, '_kpi_debug_logged'):
                            self._kpi_debug_logged = set()
                        if client_id not in self._kpi_debug_logged:
                            logger.info(f"[DEBUG] Raw KPI message structure from {client_id}:")
                            logger.info(f"  Full message keys: {list(message.keys())}")
                            logger.info(f"  KPI data keys: {list(kpi_data.keys())}")
                            logger.info(f"  KPI data sample (first 500 chars): {str(kpi_data)[:500]}")
                            if "measurements" in kpi_data:
                                logger.info(f"  Measurements array length: {len(kpi_data.get('measurements', []))}")
                                if kpi_data.get("measurements"):
                                    logger.info(f"  First measurement: {kpi_data['measurements'][0]}")
                            self._kpi_debug_logged.add(client_id)
                        cell_id_raw = kpi_data.get("cellObjectID") or kpi_data.get("cell_id") or "unknown"
                        
                        # Normalize cell_id: convert numeric strings to CELL_XXX format
                        # e.g., "1111" -> "CELL_1111", "0000" -> "CELL_0000"
                        if cell_id_raw != "unknown" and cell_id_raw and str(cell_id_raw).isdigit():
                            cell_id = f"CELL_{cell_id_raw}"
                        elif cell_id_raw and cell_id_raw.startswith("CELL_"):
                            cell_id = cell_id_raw
                        elif cell_id_raw != "unknown":
                            cell_id = f"CELL_{cell_id_raw}"  # Prefix if not already prefixed
                        else:
                            cell_id = "unknown"
                        
                        # Extract node_id if available
                        node_id = kpi_data.get("node_id") or kpi_data.get("nodeId") or message.get("node_id")
                        if node_id is not None:
                            node_id = int(node_id)
                            self.client_node_map[client_id] = node_id
                            if cell_id != "unknown":
                                self.cell_to_node_map[cell_id] = node_id
                            # TODO: restore logging with log levels
                            #if should_log(LOG_KPI):
                            #    logger.info(f"[KPI] Extracted node_id={node_id} for cell_id={cell_id}, client={client_id}")
                        else:
                            # Try to infer node_id from cell_id pattern or use default
                            # If cell_id is numeric (like "1111"), try to infer node_id
                            # Common pattern: cell_id "1111" might map to node 1, "0000" to node 0
                            inferred_node_id = None
                            if cell_id != "unknown":
                                # Try existing mapping first
                                if cell_id in self.cell_to_node_map:
                                    inferred_node_id = self.cell_to_node_map[cell_id]
                                    self.client_node_map[client_id] = inferred_node_id
                                    # TODO: restore logging with log levels
                                    #if should_log(LOG_KPI):
                                    #    logger.info(f"[KPI] Using existing node_id={inferred_node_id} for cell_id={cell_id} from mapping")
                                # Try to infer from cell_id pattern (heuristic: extract first digit from numeric part)
                                # Handle both "1111" and "CELL_1111" formats
                                numeric_part = None
                                if cell_id_raw and str(cell_id_raw).isdigit():
                                    numeric_part = str(cell_id_raw)
                                elif cell_id and cell_id.startswith("CELL_"):
                                    # Extract numeric part from "CELL_1111" -> "1111"
                                    numeric_part = cell_id.replace("CELL_", "")
                                    if not numeric_part.isdigit():
                                        numeric_part = None
                                
                                if numeric_part and numeric_part.isdigit():
                                    # Extract first digit: "1111" -> 1, "0000" -> 0, "2222" -> 2
                                    first_digit = int(numeric_part[0])
                                    inferred_node_id = first_digit
                                    self.cell_to_node_map[cell_id] = inferred_node_id
                                    self.client_node_map[client_id] = inferred_node_id
                                    # TODO: restore logging with log levels
                                    #if should_log(LOG_KPI):
                                    #    logger.info(f"[KPI] Inferred node_id={inferred_node_id} for cell_id={cell_id} (extracted from first digit of numeric part '{numeric_part}')")
                                else:
                                    logger.warning(f"No node_id found in KPI for cell_id={cell_id}, client={client_id}. Available mappings: {list(self.cell_to_node_map.keys())}")
                            else:
                                logger.warning(f"No node_id found in KPI for cell_id={cell_id}, client={client_id}. Available mappings: {list(self.cell_to_node_map.keys())}")
                        
                        # Extract and track UE information
                        ues = kpi_data.get("ues", [])
                        for ue in ues:
                            ue_id = ue.get("ue_id") or ue.get("ueId") or ue.get("id")
                            if ue_id:
                                ue_id = str(ue_id)
                                # Map UE to cell
                                ue_cell_id = ue.get("cell_id") or ue.get("cellId") or cell_id
                                if ue_cell_id != "unknown":
                                    self.ue_to_cell_map[ue_id] = ue_cell_id
                                
                                # Extract UE node_id from UE object (xApp sends node_id=3 for UEs)
                                ue_node_id = ue.get("node_id") or ue.get("nodeId")
                                if ue_node_id is not None:
                                    ue_node_id = int(ue_node_id)
                                    self.ue_to_node_map[ue_id] = ue_node_id
                                # Map UE to node (via cell or direct) - fallback if UE object doesn't have node_id
                                elif node_id is not None:
                                    self.ue_to_node_map[ue_id] = node_id
                                elif ue_cell_id in self.cell_to_node_map:
                                    self.ue_to_node_map[ue_id] = self.cell_to_node_map[ue_cell_id]
                                
                                # Store latest UE metrics
                                self.active_ues[ue_id] = {
                                    "ue_id": ue_id,
                                    "cell_id": ue_cell_id,
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                    "metrics": ue
                                }
                        
                        # Write KPI to CSV file
                        # Write cell-level data first
                        inferred_node_id = self.client_node_map.get(client_id) if client_id in self.client_node_map else node_id
                        self._write_kpi_to_csv(kpi_data, meid, cell_id, inferred_node_id)
                        
                        # Write separate rows for each UE with their node_id
                        for ue in ues:
                            ue_id = ue.get("ue_id") or ue.get("ueId") or ue.get("id")
                            if ue_id:
                                ue_node_id = ue.get("node_id") or ue.get("nodeId")
                                if ue_node_id is None:
                                    # Fallback to mapping
                                    ue_node_id = self.ue_to_node_map.get(str(ue_id))
                                if ue_node_id is not None:
                                    # Create a UE-only KPI data structure for CSV writing
                                    ue_kpi_data = {
                                        "format": kpi_data.get("format", "F1"),
                                        "measurements": ue.get("measurements", []),
                                        "ues": []  # Don't include nested UEs
                                    }
                                    ue_cell_id = ue.get("cell_id") or ue.get("cellId") or cell_id
                                    # Write UE row with UE node_id (is_ue_data=True to prefix metrics with UE_)
                                    self._write_kpi_to_csv(ue_kpi_data, meid, f"UE_{ue_id}", int(ue_node_id), is_ue_data=True)
                        
                        # Extract node_id before conversion (needed for adapter)
                        node_id_for_adapter = kpi_data.get("node_id") or kpi_data.get("nodeId") or message.get("node_id")
                        if node_id_for_adapter is None:
                            # Try to infer from cell_id or use mapping
                            if cell_id in self.cell_to_node_map:
                                node_id_for_adapter = self.cell_to_node_map[cell_id]
                            elif client_id in self.client_node_map:
                                node_id_for_adapter = self.client_node_map[client_id]
                            else:
                                node_id_for_adapter = 2  # Default to gNB
                        
                        # Add node_id to message so adapter can extract it
                        if node_id_for_adapter is not None:
                            if "kpi" not in message:
                                message["kpi"] = {}
                            message["kpi"]["node_id"] = int(node_id_for_adapter)
                        
                        # Convert KPI format
                        internal_kpi = self.kpi_adapter.convert_xapp_kpi_to_internal(message, meid)
                        
                        # Publish to membus (KPI Stream → Minirocket & Observer)
                        if self.bus:
                            await self.bus.pub("kpi.raw", make_msg(
                                "kpi.raw", "KPI", "kpi.raw.v1",
                                {
                                    "kpi": internal_kpi,
                                    "client_id": client_id,
                                    "meid": meid
                                }
                            ))
                            logger.debug(f"Published KPI to membus: kpi.raw")
                    else:
                        logger.warning(f"Unknown message type: {msg_type}")
                        
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error: {e}")
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    
        except Exception as e:
            logger.error(f"Connection error with {client_id}: {e}")
        finally:
            if client_id in self.clients:
                del self.clients[client_id]
            if client_id in self.meid_map:
                del self.meid_map[client_id]
            writer.close()
            await writer.wait_closed()
            logger.info(f"xApp connection closed: {client_id}")
            
    async def send_command(self, client_id: str, command: Dict[str, Any]) -> bool:
        """Send a control command to a client."""
        if not self.commands_enabled:
            logger.info(f"[COMMANDS DISABLED] Would send command to {client_id}: {command.get('cmd', {}).get('cmd', 'unknown')}")
            return True  # Return True to indicate "success" (command was processed, just not sent)
        
        if client_id not in self.clients:
            logger.warning(f"Client {client_id} not connected")
            return False
            
        try:
            writer = self.clients[client_id]
            response_json = json.dumps(command)
            response_bytes = response_json.encode("utf-8")
            length_header = struct.pack("!I", len(response_bytes))
            
            writer.write(length_header + response_bytes)
            # Add timeout to prevent hanging
            await asyncio.wait_for(writer.drain(), timeout=2.0)
            logger.info(f"Sent command to {client_id}: {command.get('cmd', {}).get('cmd', 'unknown')}")
            
            # Publish command to membus for dashboard
            await self.bus.pub("command.sent", make_msg(
                "command.sent", "COMMAND", "command.v1", {
                    "command": command.get('cmd', {}).get('cmd', 'unknown'),
                    "params": command.get('cmd', {}),
                    "client_id": client_id
                }
            ))
            
            return True
        except asyncio.TimeoutError:
            logger.error(f"Timeout sending command to {client_id} - connection may be broken")
            # Remove broken client
            if client_id in self.clients:
                try:
                    self.clients[client_id].close()
                except:
                    pass
                del self.clients[client_id]
            return False
        except Exception as e:
            logger.error(f"Failed to send command to {client_id}: {e}")
            # Remove broken client
            if client_id in self.clients:
                try:
                    self.clients[client_id].close()
                except:
                    pass
                del self.clients[client_id]
            return False


async def run_ai_loop_with_membus(tcp_server, target_metric, args):
    bus = MemBus()
    tcp_server.bus = bus
    kb = knowledge_base.KnowledgeBase()
    web_server = UnifiedWebServer(port=args.web_port)
    web_server.ai_agent = WebBusAdapter(bus)

    # Pass KB to agents that need it
    cognitive = CognitiveAgent(bus, kb) 
    orchestrator = OrchestratorAgent(bus)
    actuator = ActuatorAgent(bus, tcp_server, kb)

    minirocket_agents = []
    if args.minirocket_gnb_model or args.minirocket_model:
        gnb_m = "DRB_PdcpSduDelayDl" if target_metric == "delay_p95_ms" else target_metric
        path = args.minirocket_gnb_model or args.minirocket_model
        minirocket_agents.append(MinirocketAgent(bus, model_path=path, metric=gnb_m, window_size=128))
    
    if args.minirocket_ue_model or args.minirocket_model:
        ue_m = "UE_DRB_PdcpSduDelayDl_UEID" if target_metric in ("delay_p95_ms", "DRB_PdcpSduDelayDl") else target_metric
        path = args.minirocket_ue_model or args.minirocket_model
        minirocket_agents.append(MinirocketAgent(bus, model_path=path, metric=ue_m, window_size=128))

    async def bridge_bus_to_ui():
        """
        Listens to the Membus and broadcasts to the Web Dashboard.
        Handles KPI aggregation and UI updates.
        """
        q_intent = await bus.sub("intent.current")
        q_dev = await bus.sub("deviation.broadcast")
        q_cmd = await bus.sub("command.notify")
        q_kpi = await bus.sub("kpi.raw") 

        current_metrics = {}
        all_active_ues = {}
        
        while True:
            # Process KPIs
            while not q_kpi.empty():
                msg = q_kpi.get_nowait()
                payload = msg.payload.get("kpi", {})
                
                # Merge Cell Metrics
                for k, v in payload.get("CellMetrics", {}).items():
                    if v is not None: current_metrics[k] = v
                
                # Merge UE Metrics
                for ue in payload.get("UEMetrics", []):
                    uid = ue.get("ue_id")
                    if uid:
                        if uid not in all_active_ues: all_active_ues[uid] = ue
                        else: all_active_ues[uid].update(ue)
                
                # Broadcast aggregated state
                await web_server.broadcast("kpi", {
                    "cell": current_metrics, 
                    "ues": list(all_active_ues.values())
                })

            # This is mainly leftover logic from before, linked to the UI, maybe refactor is needed as it is not really used
            while not q_intent.empty():
                await web_server.broadcast("intent", q_intent.get_nowait().payload)
            
            # Process Deviations
            while not q_dev.empty():
                await web_server.broadcast("deviation", q_dev.get_nowait().payload)
            
            # Process Commands
            while not q_cmd.empty():
                msg = q_cmd.get_nowait()
                cmd_data = msg.payload
                
                await web_server.broadcast("command", {
                    "command": cmd_data.get("command"),
                    "params": cmd_data.get("params"),
                    "meid": cmd_data.get("meid")
                })
            
            await asyncio.sleep(0.2)

    tasks = [
        asyncio.create_task(tcp_server.start()), 
        asyncio.create_task(web_server.start()),
        asyncio.create_task(cognitive.run()),
        asyncio.create_task(orchestrator.run()),
        asyncio.create_task(actuator.run()),
        asyncio.create_task(bridge_bus_to_ui())
    ]
    
    for agent in minirocket_agents: tasks.append(asyncio.create_task(agent.run()))

    logger.info(f"System fully orchestrated. Dashboard: http://localhost:{args.web_port}")
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