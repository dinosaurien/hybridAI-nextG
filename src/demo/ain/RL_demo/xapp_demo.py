#!/usr/bin/env python3
"""
Demo script for AI system integrated with xApp over TCP.

This script:
1. Starts TCP server on port 5000 to receive KPIs from xApp
2. Converts xApp KPI format to internal format
3. Runs the AI contextual bandit loop
4. Sends control commands back to xApp over TCP
"""

from __future__ import annotations
import argparse
import asyncio
import csv
import json
import signal
import struct
import sys
import time
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

# Add parent directory to path
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent.parent.parent))

from ain.loop.observer_rl import Intent, RLObserver
from ain.loop.predictor import SlateDQNPredictor
from ain.loop.proposer import ActionSpace, ProposerSampler, CacheLibrary, PLAYBOOK_K, CANDIDATE_N, COOLDOWN_STEPS, cooldown_key
from ain.loop.actor import Actor
from ain.brain.llm_reasoner import normalize_deviation, to_proposer_meta, to_rl_intent
from ain.brain.openai_client import fallback_intent_for_deviation
try:
    from llm_logic import generate_biased_gpt2_intent
    HAS_LOCAL_LLM = True
except ImportError:
    HAS_LOCAL_LLM = False
from ain.common.types import ControlAction, Playbook

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class XAppKPIAdapter:
    """Converts xApp KPI format to internal format expected by RLObserver."""
    
    @staticmethod
    def convert_xapp_kpi_to_internal(xapp_kpi: Dict[str, Any], meid: str) -> Dict[str, Any]:
        """
        Convert xApp KPI format to internal format.
        
        xApp format: {"type":"kpi","meid":"...","kpi":{...}}
        Internal format: {"timestamp":"...","Header":{...},"CellMetrics":{...},"UEMetrics":[...]}
        """
        kpi_data = xapp_kpi.get("kpi", {})
        
        # Extract cell metrics from xApp format
        # Based on gnb_kpis.csv: timestamp,meid,cell_id,format,UE_PDCP_Delay_DL_ms,DL_TB_QPSK_Count,DL_TB_16QAM_Count,PRB_Used_DL,Mean_Active_UEs_DL,DL_TB_64QAM_Count
        # Based on ue_kpis.csv: timestamp,meid,cell_id,ue_id,UE_PDCP_Delay_DL_ms,TB_TotNbrDlInitial_Qpsk_UEID,...
        
        cell_metrics = {}
        ue_metrics = []
        
        # Try to extract from measurements array (if xApp sends structured data)
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
                
                # Extract UE-specific metrics
                if "UE_PDCP_Delay_DL_ms" in ue:
                    ue_metric["delay_p95_ms"] = float(ue["UE_PDCP_Delay_DL_ms"])
                if "UE_Throughput_DL_Mbps" in ue:
                    ue_metric["thr_dl_bps"] = float(ue["UE_Throughput_DL_Mbps"]) * 1e6
                if "UE_PRB_Used_DL" in ue:
                    ue_metric["prb_used_dl"] = float(ue["UE_PRB_Used_DL"])
                
                # Extract from nested measurements if available
                ue_measurements = ue.get("measurements", [])
                for m in ue_measurements:
                    name = m.get("name", "").lower()
                    value = m.get("value", 0)
                    if "delay" in name or "latency" in name:
                        ue_metric["delay_p95_ms"] = float(value)
                    elif "throughput" in name or "thr" in name:
                        if "dl" in name:
                            ue_metric["thr_dl_bps"] = float(value) * 1e6
                        elif "ul" in name:
                            ue_metric["thr_ul_bps"] = float(value) * 1e6
                    elif "prb" in name and "used" in name:
                        ue_metric["prb_used_dl"] = float(value)
                    elif "bler" in name:
                        if "dl" in name:
                            ue_metric["bler_dl"] = float(value) / 100.0
                        elif "ul" in name:
                            ue_metric["bler_ul"] = float(value) / 100.0
                    elif "cqi" in name:
                        ue_metric["cqi_avg"] = float(value)
                    elif "mcs" in name:
                        if "dl" in name:
                            ue_metric["mcs_dl_avg"] = int(value)
                        elif "ul" in name:
                            ue_metric["mcs_ul_avg"] = int(value)
                
                if ue_metric:
                    ue_metrics.append(ue_metric)
        
        # If measurements are provided, extract them
        if measurements:
            for m in measurements:
                name = m.get("name", "")
                value = m.get("value", 0)
                
                # Map xApp measurement names to internal format
                if "delay" in name.lower() or "latency" in name.lower():
                    cell_metrics["delay_p95_ms"] = float(value)
                elif "throughput" in name.lower() or "thr" in name.lower():
                    if "dl" in name.lower():
                        cell_metrics["thr_dl_bps"] = float(value) * 1e6  # Convert Mbps to bps
                    elif "ul" in name.lower():
                        cell_metrics["thr_ul_bps"] = float(value) * 1e6
                elif "bler" in name.lower():
                    if "dl" in name.lower():
                        cell_metrics["bler_dl"] = float(value) / 100.0  # Convert % to ratio
                    elif "ul" in name.lower():
                        cell_metrics["bler_ul"] = float(value) / 100.0
                elif "cqi" in name.lower():
                    cell_metrics["cqi_avg"] = float(value)
                elif "mcs" in name.lower():
                    if "dl" in name.lower():
                        cell_metrics["mcs_dl_avg"] = int(value)
                    elif "ul" in name.lower():
                        cell_metrics["mcs_ul_avg"] = int(value)
                elif "prb" in name.lower() and "used" in name.lower():
                    cell_metrics["prb_used_dl"] = float(value)
                elif "active" in name.lower() and "ue" in name.lower():
                    cell_metrics["active_ue_count"] = int(value)
        
        # Extract from raw fields if available (direct field access)
        if "UE_PDCP_Delay_DL_ms" in kpi_data:
            cell_metrics["delay_p95_ms"] = float(kpi_data["UE_PDCP_Delay_DL_ms"])
        if "PRB_Used_DL" in kpi_data:
            cell_metrics["prb_used_dl"] = float(kpi_data["PRB_Used_DL"])
        if "Mean_Active_UEs_DL" in kpi_data:
            cell_metrics["active_ue_count"] = int(kpi_data["Mean_Active_UEs_DL"])
        if "DL_TB_64QAM_Count" in kpi_data:
            # Can use this for MCS estimation if needed
            pass
        
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
        cell_metrics.setdefault("cell_id", cell_id)
        cell_metrics.setdefault("prb_total", 100.0)
        cell_metrics.setdefault("thr_dl_bps", 0.0)
        cell_metrics.setdefault("thr_ul_bps", 0.0)
        cell_metrics.setdefault("bler_dl", 0.0)
        cell_metrics.setdefault("bler_ul", 0.0)
        cell_metrics.setdefault("cqi_avg", 10.0)
        cell_metrics.setdefault("mcs_dl_avg", 15)
        cell_metrics.setdefault("mcs_ul_avg", 15)
        cell_metrics.setdefault("delay_p95_ms", 50.0)
        cell_metrics.setdefault("active_ue_count", 1)
        cell_metrics.setdefault("prb_used_dl", 50.0)
        
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
                "sequence_number": int(time.time() * 1000) % 1000000,
            },
            "CellMetrics": cell_metrics,
            "UEMetrics": ue_metrics,
        }
        
        return internal_kpi


class PlaybookToCommandConverter:
    """Converts playbook actions to xApp control commands."""
    
    @staticmethod
    def playbook_to_commands(
        playbook: Playbook, 
        meid: str, 
        node_id: int = 0,
        cell_to_node_map: Optional[Dict[str, int]] = None,
        ue_to_node_map: Optional[Dict[str, int]] = None,
        ue_to_cell_map: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Convert playbook actions to xApp control commands.
        
        Maps:
        - MCS_CAP -> set-mcs
        - PRB_WEIGHT -> set-bandwidth (approximation)
        - SCHEDULER_POLICY -> (not directly supported, skip or log)
        - SLICE_QOS -> (not directly supported, skip or log)
        - REPORTING -> (no-op, skip)
        
        Args:
            playbook: Playbook with actions
            meid: Management entity ID
            node_id: Default node ID
            cell_to_node_map: Optional mapping from cell_id to node_id
            ue_to_node_map: Optional mapping from ue_id to node_id
            ue_to_cell_map: Optional mapping from ue_id to cell_id
        """
        commands = []
        cell_to_node_map = cell_to_node_map or {}
        ue_to_node_map = ue_to_node_map or {}
        ue_to_cell_map = ue_to_cell_map or {}
        
        for action in playbook.actions:
            cmd = None
            
            # Determine node_id for this action:
            # 1. Check if action.params has explicit "node"
            # 2. For UE-scoped actions: check ue_to_node_map
            # 3. For CELL-scoped actions: check cell_to_node_map
            # 4. Fall back to default node_id
            action_node_id = action.params.get("node", node_id)
            
            if action.scope == "UE" and action.ue_id:
                # UE-scoped action: try to get node_id from UE mapping
                if action.ue_id in ue_to_node_map:
                    action_node_id = ue_to_node_map[action.ue_id]
                    logger.info(f"Using node_id={action_node_id} for UE {action.ue_id} from mapping")
                elif action.ue_id in ue_to_cell_map:
                    # Try via cell mapping
                    cell_id = ue_to_cell_map[action.ue_id]
                    if cell_id in cell_to_node_map:
                        action_node_id = cell_to_node_map[cell_id]
                        logger.info(f"Using node_id={action_node_id} for UE {action.ue_id} via cell {cell_id}")
                else:
                    logger.warning(f"No node mapping found for UE {action.ue_id}, using default node_id={action_node_id}")
            
            elif action.scope == "CELL" and action.cell_id:
                # CELL-scoped action: try to get node_id from cell mapping
                if action.cell_id in cell_to_node_map:
                    action_node_id = cell_to_node_map[action.cell_id]
                    logger.info(f"Using node_id={action_node_id} for cell {action.cell_id} from mapping")
                else:
                    # Fallback: if configured cell_id not found, use the first available cell from KPIs
                    # This handles the case where system is configured with CELL_001 but xApp sends CELL_1111
                    if cell_to_node_map:
                        actual_cell_id = list(cell_to_node_map.keys())[0]
                        action_node_id = cell_to_node_map[actual_cell_id]
                        logger.warning(f"No node mapping found for configured cell {action.cell_id}, using actual cell {actual_cell_id} with node_id={action_node_id}. Available cells: {list(cell_to_node_map.keys())}")
                    else:
                        logger.warning(f"No node mapping found for cell {action.cell_id}, using default node_id={action_node_id}. Available cells: {list(cell_to_node_map.keys())}")
            
            # Log the node_id being used
            logger.info(f"Action {action.type} ({action.scope}) will use node_id={action_node_id}")
            
            if action.type == "MCS_CAP":
                # Map MCS_CAP to set-mcs
                # set-mcs requires node (gNB node with MmWaveEnbNetDevice)
                if action_node_id == 0:
                    logger.warning(f"MCS_CAP requires a valid node_id (gNB node), but got 0. Skipping command.")
                    continue
                
                mcs_value = action.params.get("dl_mcs_max", 18)
                cmd = {
                    "type": "control",
                    "meid": meid,
                    "cmd": {
                        "cmd": "set-mcs",
                        "node": int(action_node_id),  # Required: gNB node
                        "mcs": int(mcs_value)
                    }
                }
                # Add UE ID if this is a UE-scoped action
                if action.scope == "UE" and action.ue_id:
                    cmd["cmd"]["ue_id"] = action.ue_id
                    logger.info(f"MCS_CAP command for UE {action.ue_id} on gNB node {action_node_id}")
                else:
                    logger.info(f"MCS_CAP command for gNB node {action_node_id}, mcs={mcs_value}")
                
            elif action.type == "PRB_WEIGHT":
                # Map PRB_WEIGHT to set-bandwidth (approximate)
                # set-bandwidth: node is optional (if 0 or not provided, searches all nodes)
                weight = action.params.get("weight", 1.0)
                # Convert weight to bandwidth (rough approximation: 1.0 = 100 RBs)
                bandwidth = int(100 * weight)
                cmd = {
                    "type": "control",
                    "meid": meid,
                    "cmd": {
                        "cmd": "set-bandwidth",
                        "bandwidth": bandwidth
                    }
                }
                # Only include node if we have a valid mapping (not 0)
                # If node is 0 or not found, omit it to search all nodes
                if action_node_id != 0:
                    cmd["cmd"]["node"] = int(action_node_id)
                    logger.info(f"PRB_WEIGHT command for gNB node {action_node_id}, bandwidth={bandwidth}")
                else:
                    logger.info(f"PRB_WEIGHT command for all nodes (node not specified), bandwidth={bandwidth}")
                
                # Add UE ID if this is a UE-scoped action
                if action.scope == "UE" and action.ue_id:
                    cmd["cmd"]["ue_id"] = action.ue_id
                    logger.info(f"  → Targeting UE {action.ue_id}")
                
            elif action.type in ("TX_POWER", "POWER_CONTROL"):
                # Map TX_POWER/POWER_CONTROL to set-enb-txpower
                # set-enb-txpower requires node (gNB node with MmWaveEnbNetDevice)
                if action_node_id == 0:
                    logger.warning(f"TX_POWER requires a valid node_id (gNB node), but got 0. Skipping command.")
                    continue
                
                # AI must provide txPowerDbm value - no default, let AI decide
                tx_power_dbm = action.params.get("txPowerDbm") or action.params.get("tx_power_dbm")
                if tx_power_dbm is None:
                    logger.warning(f"{action.type} action missing txPowerDbm parameter, skipping: {action.params}")
                    continue
                cmd = {
                    "type": "control",
                    "meid": meid,
                    "cmd": {
                        "cmd": "set-enb-txpower",
                        "node": int(action_node_id),  # Required: gNB node
                        "txPowerDbm": float(tx_power_dbm)
                    }
                }
                # Add UE ID if this is a UE-scoped action
                if action.scope == "UE" and action.ue_id:
                    cmd["cmd"]["ue_id"] = action.ue_id
                    logger.info(f"TX_POWER command for UE {action.ue_id} on gNB node {action_node_id}, txPower={tx_power_dbm}dBm")
                else:
                    logger.info(f"TX_POWER command for gNB node {action_node_id}, txPower={tx_power_dbm}dBm")
                
            elif action.type == "SCHEDULER_POLICY":
                # Not directly supported, log warning
                logger.warning(f"SCHEDULER_POLICY action not directly supported: {action.params}")
                continue
                
            elif action.type == "SLICE_QOS":
                # Not directly supported, log warning
                logger.warning(f"SLICE_QOS action not directly supported: {action.params}")
                continue
                
            elif action.type == "REPORTING":
                # No-op, skip
                continue
            
            if cmd:
                commands.append(cmd)
        
        return commands


class XAppTCPServer:
    """TCP server for xApp communication."""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 5000, kpi_csv_file: Optional[str] = None, commands_enabled: bool = True):
        self.host = host
        self.port = port
        self.server: Optional[asyncio.Server] = None
        self.clients: Dict[str, asyncio.StreamWriter] = {}
        self.kpi_queue: asyncio.Queue = asyncio.Queue()
        self.meid_map: Dict[str, str] = {}  # Map client_id to meid
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
            
            # Note: UE measurements are now written separately (see caller)
            # This function only handles cell-level data when is_ue_data=False
            
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
                # Read 4-byte header for message length
                header = await self._recv_all(reader, 4)
                if header is None:
                    logger.info(f"xApp client {client_id} disconnected")
                    break
                    
                # Unpack length (big-endian unsigned int)
                length = struct.unpack("!I", header)[0]
                
                if length == 0 or length > 1024 * 1024:  # 1MB max
                    logger.error(f"Invalid frame length={length}, closing connection")
                    break
                
                # Read the message body
                body = await self._recv_all(reader, length)
                if body is None:
                    logger.error("Incomplete frame body")
                    break
                
                # Decode and parse JSON
                try:
                    text = body.decode("utf-8", errors="replace")
                    message = json.loads(text)
                    
                    msg_type = message.get("type", "unknown")
                    #logger.info(f"Received {msg_type} message from {client_id}")
                    
                    if msg_type == "kpi":
                        # Store meid for this client
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
                            logger.info(f"Extracted node_id={node_id} for cell_id={cell_id}, client={client_id}")
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
                                    logger.info(f"Using existing node_id={inferred_node_id} for cell_id={cell_id} from mapping")
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
                                    logger.info(f"Inferred node_id={inferred_node_id} for cell_id={cell_id} (extracted from first digit of numeric part '{numeric_part}')")
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
                                    logger.info(f"Extracted UE node_id={ue_node_id} for UE {ue_id}")
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
                        
                        # Queue KPI for processing
                        await self.kpi_queue.put((client_id, message))
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


class XAppActor(Actor):
    """Extended Actor that sends commands to xApp over TCP."""
    
    def __init__(self, out_dir: str, tcp_server: XAppTCPServer, meid: str = "gnb:131-133-31000000", node_id: int = 0):
        super().__init__(out_dir)
        self.tcp_server = tcp_server
        self.meid = meid
        self.node_id = node_id
        self.converter = PlaybookToCommandConverter()
        self.commands_enabled = getattr(tcp_server, 'commands_enabled', True)
        
    async def apply_async(self, playbook: Playbook, client_id: Optional[str] = None):
        """Apply playbook by sending commands to xApp (async version)."""
        # Save to file (original behavior)
        payload = self.make_payload(playbook)
        self.save_payload(payload)
        logger.info(f"Saved playbook: {payload['playbook_id']}")
        
        # Get mappings from TCP server if available
        cell_to_node_map = {}
        ue_to_node_map = {}
        ue_to_cell_map = {}
        default_node_id = self.node_id
        
        if self.tcp_server:
            cell_to_node_map = getattr(self.tcp_server, 'cell_to_node_map', {})
            ue_to_node_map = getattr(self.tcp_server, 'ue_to_node_map', {})
            ue_to_cell_map = getattr(self.tcp_server, 'ue_to_cell_map', {})
            if client_id and hasattr(self.tcp_server, 'client_node_map'):
                default_node_id = self.tcp_server.client_node_map.get(client_id, self.node_id)
            
            # Log available mappings for debugging
            logger.info(f"Available mappings - cells: {list(cell_to_node_map.keys())}, UEs: {list(ue_to_node_map.keys())[:5]}..., default_node_id: {default_node_id}")
        
        # Convert to commands and send to xApp
        commands = self.converter.playbook_to_commands(
            playbook, 
            self.meid, 
            default_node_id,
            cell_to_node_map=cell_to_node_map,
            ue_to_node_map=ue_to_node_map,
            ue_to_cell_map=ue_to_cell_map
        )
        
        logger.info(f"Converted playbook to {len(commands)} command(s)")
        for i, cmd in enumerate(commands):
            logger.info(f"  Command {i+1}: {cmd.get('cmd', {}).get('cmd', 'unknown')}")
        
        if not self.commands_enabled:
            logger.info(f"[COMMANDS DISABLED] Would send {len(commands)} command(s) but commands are disabled")
            return
        
        if commands:
            # If client_id provided, send to that client, otherwise broadcast
            if client_id:
                if client_id not in self.tcp_server.clients:
                    logger.warning(f"Client {client_id} not connected, cannot send commands")
                else:
                    for i, cmd in enumerate(commands):
                        logger.info(f"Sending command {i+1}/{len(commands)}: {cmd.get('cmd', {}).get('cmd', 'unknown')} to node {cmd.get('cmd', {}).get('node', 'unknown')}")
                        success = await self.tcp_server.send_command(client_id, cmd)
                        if not success:
                            logger.error(f"Failed to send command to {client_id}")
                        # Add 2 second cooldown between commands (except for the last one)
                        if i < len(commands) - 1:
                            logger.info(f"Waiting 2 seconds before next command...")
                            await asyncio.sleep(2.0)
            else:
                # Send to all connected clients
                connected_clients = list(self.tcp_server.clients.keys())
                if not connected_clients:
                    logger.warning("No clients connected, cannot send commands")
                else:
                    for cid in connected_clients:
                        for i, cmd in enumerate(commands):
                            logger.info(f"Sending command {i+1}/{len(commands)}: {cmd.get('cmd', {}).get('cmd', 'unknown')} to node {cmd.get('cmd', {}).get('node', 'unknown')}")
                            success = await self.tcp_server.send_command(cid, cmd)
                            if not success:
                                logger.error(f"Failed to send command to {cid}")
                            # Add 2 second cooldown between commands (except for the last one)
                            if i < len(commands) - 1:
                                logger.info(f"Waiting 2 seconds before next command...")
                                await asyncio.sleep(2.0)
        else:
            logger.warning("No commands generated from playbook - playbook may contain unsupported actions")


async def run_ai_loop(
    tcp_server: XAppTCPServer,
    cells: List[str],
    slices: List[str],
    target_metric: str = "delay_p95_ms",
    target_value: float = 40.0,
    steps: int = 1000,
    offline_model: Optional[str] = None,
    default_node_id: Optional[int] = None,
):
    """Run the AI contextual bandit loop."""
    # Initialize components
    action_space = ActionSpace(cells=cells, slices=slices)
    cache = CacheLibrary(max_per_key=20)
    
    # Create predictor and observer
    tmp_intent = Intent(type="REDUCE_LATENCY", metric=target_metric, target=target_value)
    
    # Create a temporary file for observer (will be updated with real KPIs)
    import tempfile
    temp_kpi_file = Path(tempfile.gettempdir()) / "xapp_kpi_temp.json"
    temp_kpi_file.parent.mkdir(parents=True, exist_ok=True)
    # Initialize with empty KPI stream
    with open(temp_kpi_file, "w") as f:
        json.dump({"kpi_stream": []}, f)
    
    # Create observer with temp file (will be updated when KPIs arrive)
    # We need to create a dummy observer first to get feature count
    dummy_observer = RLObserver(predictor=None, intent=tmp_intent, kpi_file=str(temp_kpi_file), window=12)
    
    # Create predictor with correct feature dimension
    predictor = SlateDQNPredictor(
        action_space, 
        feat_dim=len(dummy_observer.features), 
        seed=0
    )
    
    # Now create the real observer with predictor attached
    observer = RLObserver(predictor=predictor, intent=tmp_intent, kpi_file=str(temp_kpi_file), window=12)
    
    # Load model: Try online checkpoint first, then offline model, then start from scratch
    online_checkpoint = "models/qnet_online.pt"
    model_loaded = False
    
    # Try to load online checkpoint (resume from previous run)
    if predictor.load_checkpoint(online_checkpoint, load_replay_buffer=False):
        logger.info(f"✓ Resumed training from online checkpoint: {online_checkpoint} (step={predictor.steps})")
        model_loaded = True
    # Fallback to offline model if provided
    elif offline_model:
        try:
            predictor.load_offline(offline_model)
            logger.info(f"✓ Loaded pre-trained offline model from {offline_model}")
            model_loaded = True
        except Exception as e:
            logger.warning(f"Could not load pre-trained model ({e}). Starting from scratch.")
    else:
        logger.info("Starting from scratch (no checkpoint or offline model provided)")
    
    # Create actor with TCP server
    # Use provided default_node_id, or 0 as fallback (will be overridden by actual mappings from KPIs)
    actor_node_id = default_node_id if default_node_id is not None else 0
    actor = XAppActor("playbooks", tcp_server, meid="gnb:131-133-31000000", node_id=actor_node_id)
    
    # KPI adapter
    kpi_adapter = XAppKPIAdapter()
    
    # State
    cooldown_clock: Dict = {}
    last_playbook = None
    success_streak = 0
    step = 0
    net_intent = None
    intent_meta = {"intent": "LATENCY_P95", "scope": "GLOBAL"}
    current_client_id = None
    current_meid = "gnb:131-133-31000000"
    
    if model_loaded:
        logger.info("AI loop started with pre-trained model, waiting for KPIs from xApp...")
    else:
        logger.info("AI loop started (training from scratch), waiting for KPIs from xApp...")
    
    # Setup signal handlers to save checkpoint on exit
    online_checkpoint = "models/qnet_online.pt"
    def save_on_exit(signum=None, frame=None):
        """Save checkpoint before exiting."""
        try:
            checkpoint_path = predictor.save_checkpoint(online_checkpoint, save_replay_buffer=False)
            logger.info(f"💾 Saved final checkpoint to {checkpoint_path} (step={predictor.steps})")
        except Exception as e:
            logger.error(f"Failed to save checkpoint on exit: {e}")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, save_on_exit)
    signal.signal(signal.SIGTERM, save_on_exit)
    
    while step < steps:
        # Wait for KPI from xApp
        try:
            client_id, xapp_message = await asyncio.wait_for(tcp_server.kpi_queue.get(), timeout=5.0)
            current_client_id = client_id
            current_meid = tcp_server.meid_map.get(client_id, current_meid)
            actor.meid = current_meid
            
            # Update actor's node_id if available
            if hasattr(tcp_server, 'client_node_map') and client_id in tcp_server.client_node_map:
                actor.node_id = tcp_server.client_node_map[client_id]
                logger.debug(f"Updated actor node_id to {actor.node_id} for client {client_id}")
            
            logger.info(f"Processing KPI from {client_id}, meid={current_meid}, active_ues={len(getattr(tcp_server, 'active_ues', {}))}")
            
            # Convert KPI format
            internal_kpi = kpi_adapter.convert_xapp_kpi_to_internal(xapp_message, current_meid)
            logger.debug(f"Converted KPI: metric={internal_kpi.get('CellMetrics', {}).get('delay_p95_ms', 'N/A')}")
            
        except asyncio.TimeoutError:
            logger.debug("No KPI received, waiting...")
            await asyncio.sleep(0.5)
            continue
        
        # Process KPI - write to temp file for observer to read
        import tempfile
        temp_file = Path(tempfile.gettempdir()) / "xapp_kpi_temp.json"
        with open(temp_file, "w") as f:
            json.dump({"kpi_stream": [internal_kpi]}, f)
        
        # Update observer's kpi_file (must be Path object, not string)
        observer.kpi_file = temp_file
        state = observer.step(last_playbook)
        if state is None:
            logger.warning("Observer returned None state")
            await asyncio.sleep(0.5)
            continue
        
        latest = observer.last_kpi_raw or {}
        cell = (latest or {}).get("CellMetrics", {})
        curr = float(cell.get(observer.intent.metric, 0.0))
        hit = (curr <= observer.intent.target) if observer.intent.direction == "lower_better" else (curr >= observer.intent.target)
        success_streak = success_streak + 1 if hit else 0
        
        # Bootstrap or update intent on step==0 or when needed
        if step == 0 or net_intent is None:
            dev_raw = {
                "source": "xapp_kpi",
                "metric": target_metric,
                "value": curr,
                "target": target_value,
                "direction": "lower_better" if ("delay" in target_metric or "latency" in target_metric) else "higher_better",
                "severity": "medium",
                "scope": {
                    "cell_id": cell.get("cell_id") or "CELL_001",
                    "region": "A",
                    "service": "demo",
                    "tenancy": "prod",
                },
                "evidence_ref": "telemetry://xapp/kpi",
            }
            
            # Print deviation details
            direction_str = dev_raw["direction"]
            is_deviating = (curr > target_value) if direction_str == "lower_better" else (curr < target_value)
            deviation_pct = abs((curr - target_value) / target_value * 100) if target_value > 0 else 0
            logger.info(f"═══════════════════════════════════════════════════════════")
            logger.info(f"📊 DEVIATION DETECTED:")
            logger.info(f"   Metric: {target_metric}")
            logger.info(f"   Current Value: {curr:.2f}")
            logger.info(f"   Target Value: {target_value:.2f}")
            logger.info(f"   Direction: {direction_str}")
            logger.info(f"   Deviation: {deviation_pct:.1f}% {'above' if curr > target_value else 'below'} target")
            logger.info(f"   Status: {'⚠️  DEVIATING' if is_deviating else '✅ OK'}")
            logger.info(f"   Cell ID: {dev_raw['scope']['cell_id']}")
            logger.info(f"   Severity: {dev_raw['severity']}")
            logger.info(f"═══════════════════════════════════════════════════════════")
            
                        
            # --- START LOCAL BIASED LLM LOGIC ---
            if use_llm:
                try:
                    logger.info("Calling Local Biased GPT-2 on AMD GPU...")
                    # This calls the logic from your llm_logic.py
                    from llm_logic import generate_biased_gpt2_intent
                    net_intent = generate_biased_gpt2_intent(dev)
                except Exception as e:
                    logger.error(f"Local LLM Error: {e}. Falling back to hardcoded logic.")
                    net_intent = fallback_intent_for_deviation(dev)
            else:
                net_intent = fallback_intent_for_deviation(dev)
            # --- END LOCAL BIASED LLM LOGIC ---

            intent_meta = to_proposer_meta(net_intent)
            rl_cfg = to_rl_intent(net_intent)
            
            new_intent = Intent(
                type=rl_cfg["type"],
                metric=rl_cfg["metric"],
                target=float(rl_cfg["target"]),
                direction=rl_cfg["direction"],
                action_cost=float(rl_cfg.get("action_cost", 0.01)),
                reward_clip=float(rl_cfg.get("reward_clip", 20.0)),
            )
            observer.intent = new_intent
            logger.info(f"Intent set: {observer.intent} (scope={intent_meta['scope']})")
        
        # Generate candidate playbooks
        logger.info(f"Generating {CANDIDATE_N} candidate playbooks (ε={predictor.epsilon():.3f})...")
        candidates = ProposerSampler.sample_playbooks(
            action_space, N=CANDIDATE_N, K=PLAYBOOK_K,
            cooldown_clock=cooldown_clock,
            cache=cache,
            intent_meta=intent_meta,
            epsilon=predictor.epsilon()
        )
        
        # Score playbooks
        logger.info(f"Evaluating Q-values for {len(candidates)} candidates...")
        scored = predictor.score_playbooks(state, candidates)
        scored.sort(key=lambda x: x[1], reverse=True)
        best_pb, best_q = scored[0]
        logger.info(f"Best Q={best_q:.3f}")
        
        # Update cooldown
        for a in best_pb.actions:
            ck = cooldown_key(a)
            cooldown_clock[ck] = max(cooldown_clock.get(ck, 0), COOLDOWN_STEPS)
        for k in list(cooldown_clock.keys()):
            cooldown_clock[k] -= 1
            if cooldown_clock[k] <= 0:
                cooldown_clock.pop(k, None)
        
        # Cache
        cache.add(intent_meta, best_pb, best_q)
        
        logger.info(f"[t={step:03d}] metric={observer.intent.metric}={curr:.2f} hit={hit} streak={success_streak} "
                   f"eps={predictor.epsilon():.3f} q={best_q:.3f}")
        for i, a in enumerate(best_pb.actions):
            logger.info(f"   • A{i+1}: {a.type} {a.scope} cell={a.cell_id} slice={a.slice_id} params={a.params}")
        
        # Apply playbook (sends commands to xApp)
        logger.info(f"[Step {step}] Applying playbook and sending commands to xApp...")
        logger.debug(f"Current client_id: {current_client_id}, connected clients: {list(tcp_server.clients.keys())}")
        try:
            await actor.apply_async(best_pb, client_id=current_client_id)
            logger.info(f"[Step {step}] Commands sent successfully")
        except Exception as e:
            logger.error(f"[Step {step}] Error applying playbook: {e}", exc_info=True)
            # Continue anyway to avoid getting stuck
        
        last_playbook = best_pb
        predictor.steps += 1
        step += 1
        
        # Save checkpoint periodically (every 50 steps) and on success
        if (step % 50 == 0) or (success_streak >= 4):
            try:
                checkpoint_path = predictor.save_checkpoint(online_checkpoint, save_replay_buffer=False)
                logger.info(f"💾 Saved checkpoint to {checkpoint_path} (step={step}, replay_size={len(predictor.replay)})")
            except Exception as e:
                logger.warning(f"Failed to save checkpoint: {e}")
        
        if success_streak >= 4:
            logger.info(f"Intent achieved for {success_streak} consecutive readings. Resetting streak.")
            success_streak = 0
        
        await asyncio.sleep(0.1)  # Small delay between iterations


async def main():
    parser = argparse.ArgumentParser(description="AI System Demo with xApp TCP Integration")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="TCP server host")
    parser.add_argument("--port", type=int, default=6000, 
                       help="TCP server port (default: 6000 for relay server, use 5000 if connecting directly to xApp)")
    parser.add_argument("--steps", type=int, default=1000, help="Max steps")
    parser.add_argument("--target-metric", type=str, default="delay_p95_ms", help="Target metric")
    parser.add_argument("--target-value", type=float, default=40.0, help="Target value")
    parser.add_argument("--cells", type=str, nargs="+", default=["CELL_001"], help="Cell IDs (will auto-detect from KPIs if not found)")
    parser.add_argument("--slices", type=str, nargs="+", default=["SLICE_A"], help="Slice IDs")
    parser.add_argument("--default-node-id", type=int, default=None, 
                       help="Default node_id for gNB (if xApp doesn't send it). Typically 1 or 2 for gNB nodes. If not specified, will try to infer from KPIs.")
    parser.add_argument("--offline-model", type=str, default=None, 
                       help="Path to pre-trained model (e.g., models/qnet_offline.pt). If not provided, starts training from scratch.")
    parser.add_argument("--commands-enabled", action="store_true", default=True,
                       help="Enable sending control commands to xApp (default: enabled)")
    parser.add_argument("--commands-disabled", action="store_false", dest="commands_enabled",
                       help="Disable sending control commands to xApp (useful for KPI collection only)")
    parser.add_argument("--use-llm", action="store_true", help="Use local biased GPT-2 for reasoning")
    
    args = parser.parse_args()
    
    # Create TCP server (CSV file will be created automatically in project root)
    tcp_server = XAppTCPServer(host=args.host, port=args.port, commands_enabled=args.commands_enabled)
    
    if not args.commands_enabled:
        logger.info("=" * 60)
        logger.info("COMMANDS DISABLED - AI will process KPIs but NOT send control commands")
        logger.info("This is useful for collecting KPIs from simulation without interference")
        logger.info("=" * 60)
    await tcp_server.start()
    
    try:
        # Run AI loop
        await run_ai_loop(
            tcp_server,
            cells=args.cells,
            slices=args.slices,
            target_metric=args.target_metric,
            target_value=args.target_value,
            steps=args.steps,
            offline_model=args.offline_model,
            default_node_id=args.default_node_id,
            use_llm=args.use_llm,
        )
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        await tcp_server.stop()


if __name__ == "__main__":
    asyncio.run(main())

