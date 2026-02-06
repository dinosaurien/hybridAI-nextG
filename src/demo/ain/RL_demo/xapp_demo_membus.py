#!/usr/bin/env python3
"""
Demo script for AI system integrated with xApp over TCP using Membus architecture.

This matches the system design:
- KPI Stream → Minirocket → Reasoner
- SLO Intents → Reasoner
- Reasoner ↔ Knowledge Base ↔ Proposer
- KPI Stream → Loop_Observer → Learner (DQN)
- Proposer ↔ Predictor ↔ Learner (DQN)
- Proposer → Actor → Control Actions
"""

from __future__ import annotations
import argparse
import asyncio
import csv
import json
import signal
import struct
import sys
import tempfile
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

# Add parent directory to path
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent.parent.parent))

from ain.loop.actor import Actor
from ain.bus.mem import MemBus
from ain.agents.utils import make_msg

# Import agents
from ain.agents.minirocket_agent import MinirocketAgent
from ain.agents.actor_agent import ActorAgent

import logging
from ain.common.log_config import (
    set_log_levels, parse_log_levels, should_log, log_if_enabled,
    LOG_LEARNING, LOG_INTENT, LOG_SCORING, LOG_DEVIATION, LOG_OBSERVER, LOG_REWARD, LOG_KPI, LOG_BANDIT,
    LOG_COMMANDS,
    CATEGORY_NAMES
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- START ROCm LLM INJECTION ---
try:
    from llm_logic import generate_biased_gpt2_intent
    import ain.agents.reasoner_agent_enhanced as rae
    
    # Overwrite the default reasoning function with our local biased GPT-2
    rae.create_network_intent_from_deviation = generate_biased_gpt2_intent
    
except Exception as e:
    logger.error(f"[GPT-2] Failed to inject LLM: {e}")
# --- END ROCm LLM INJECTION ---

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
        
        # Debug logging
        if should_log(LOG_KPI):
            logger.debug(f"[KPI] Converted xApp KPI: CellMetrics keys={list(cell_metrics.keys())}, UEMetrics count={len(ue_metrics)}")
        
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
                    if should_log(LOG_KPI):
                        logger.info(f"[KPI] Received {msg_type} message from {client_id}")
                    
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
                            if should_log(LOG_KPI):
                                logger.info(f"[KPI] Extracted node_id={node_id} for cell_id={cell_id}, client={client_id}")
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
                                    if should_log(LOG_KPI):
                                        logger.info(f"[KPI] Using existing node_id={inferred_node_id} for cell_id={cell_id} from mapping")
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
                                    if should_log(LOG_KPI):
                                        logger.info(f"[KPI] Inferred node_id={inferred_node_id} for cell_id={cell_id} (extracted from first digit of numeric part '{numeric_part}')")
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
                                    if should_log(LOG_KPI):
                                        logger.info(f"[KPI] Extracted UE node_id={ue_node_id} for UE {ue_id}")
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


class XAppActorAgent:
    """Actor agent that sends commands to xApp over TCP."""
    
    def __init__(self, bus: MemBus, actor: Actor, tcp_server: XAppTCPServer, 
                 default_meid: str = "gnb:131-133-31000000", 
                 node_id: int = 0):
        self.bus = bus
        self.actor = actor
        self.tcp_server = tcp_server
        # self.converter = converter # PlaybookToCommandConverter is no longer needed
        self.default_meid = default_meid
        self.node_id = node_id
        
    async def run(self):
        """Subscribe to actor events and send commands."""
        #TO-DO: Fix this logic


async def run_ai_loop_with_membus(
    tcp_server: XAppTCPServer,
    target_metric: str = "DRB_PdcpSduDelayDl",  # Use actual CSV metric name
    target_value: float = 40.0,
    offline_model: Optional[str] = None,
    minirocket_model: Optional[str] = None,
    minirocket_gnb_model: Optional[str] = None,
    minirocket_ue_model: Optional[str] = None,
    use_llm: bool = False,
    loss_history_file: Optional[str] = None,
    slo_file: Optional[str] = None,
):
    """Run the AI loop using membus architecture matching system design."""
    # Create membus
    bus = MemBus()
    tcp_server.bus = bus
    
    # Create agents according to system design
    # 1. Minirocket Agents (KPI Stream → Minirocket → deviation.detected)
    # Support both gNB-level and UE-level deviation detection
    minirocket_agents = []
    
    # gNB-level agent (monitors cell-level metrics)
    if minirocket_gnb_model or minirocket_model:
        # Determine gNB metric name from target_metric or use default
        gnb_metric = target_metric
        # Map common metric names to actual CSV column names
        if gnb_metric == "delay_p95_ms":
            gnb_metric = "DRB_PdcpSduDelayDl"  # Use actual column name from CSV
        elif gnb_metric == "thr_dl_bps":
            gnb_metric = "DRB_MeanActiveUeDl"  # Or appropriate gNB metric
        
        gnb_model_path = minirocket_gnb_model or minirocket_model or "models/minirocket_xapp_gnb.joblib"
        minirocket_gnb_agent = MinirocketAgent(
            bus,
            model_path=gnb_model_path,
            metric=gnb_metric,
            window_size=128
        )
        minirocket_agents.append(("gnb", minirocket_gnb_agent))
        logger.info(f"Created gNB-level MinirocketAgent: metric={gnb_metric}, model={gnb_model_path}")
    
    # UE-level agent (monitors UE-level metrics)
    if minirocket_ue_model or minirocket_model:
        # Determine UE metric name from target_metric or use default
        ue_metric = target_metric
        # Map common metric names to actual CSV column names
        if ue_metric == "delay_p95_ms" or ue_metric == "DRB_PdcpSduDelayDl":
            ue_metric = "UE_DRB_PdcpSduDelayDl_UEID"  # Use actual column name from CSV
        elif ue_metric == "thr_dl_bps" or ue_metric == "UE_DRB_UEThpDl_UEID":
            ue_metric = "UE_DRB_UEThpDl_UEID"  # Or appropriate UE metric
        
        ue_model_path = minirocket_ue_model or minirocket_model or "models/minirocket_xapp_ue.joblib"
        minirocket_ue_agent = MinirocketAgent(
            bus,
            model_path=ue_model_path,
            metric=ue_metric,
            window_size=128
        )
        minirocket_agents.append(("ue", minirocket_ue_agent))
        logger.info(f"Created UE-level MinirocketAgent: metric={ue_metric}, model={ue_model_path}")
    
    # Fallback: if no specific models provided, use single agent with default model
    if not minirocket_agents:
        minirocket_agent = MinirocketAgent(
            bus,
            model_path=minirocket_model or "models/minirocket.joblib",
            metric=target_metric
        )
        minirocket_agents = [("default", minirocket_agent)]
        logger.info(f"Created default MinirocketAgent: metric={target_metric}, model={minirocket_model or 'models/minirocket.joblib'}")
    

    # Run all agents
    logger.info("Starting AI loop with membus architecture (matching system design)...")
    
    # Create tasks for all minirocket agents
    minirocket_tasks = [
        asyncio.create_task(agent.run(), name=f"minirocket_agent_{name}")
        for name, agent in minirocket_agents
    ]
    
    # Import and start dashboard
    try:
        from ain.dashboard.dashboard_server import start_dashboard
        from ain.dashboard.http_server import start_http_server
        
        dashboard_static = Path(__file__).parent.parent / "dashboard" / "static"
        
        dashboard_tasks = [
            asyncio.create_task(start_dashboard(bus, port=8081), name="dashboard_websocket"),
            asyncio.create_task(start_http_server(dashboard_static, port=8080), name="dashboard_http"),
        ]
        logger.info("[DASHBOARD] Dashboard enabled - HTTP: http://localhost:8080, WebSocket: ws://localhost:8081")
    except ImportError as e:
        logger.warning(f"[DASHBOARD] Dashboard not available: {e}")
        dashboard_tasks = []
    
    tasks = minirocket_tasks + dashboard_tasks
    
    try:
        # Use return_exceptions=True so one task failure doesn't kill the whole system
        results = await asyncio.gather(*tasks, return_exceptions=True)
        # Check for exceptions in results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                task_name = tasks[i].get_name() if hasattr(tasks[i], 'get_name') else f"task_{i}"
                logger.error(f"Task {task_name} raised exception: {result}", exc_info=result)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")


async def main():
    parser = argparse.ArgumentParser(description="AI System Demo with xApp TCP Integration (Membus)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="TCP server host")
    parser.add_argument("--port", type=int, default=6000, help="TCP server port")
    parser.add_argument("--target-metric", type=str, default="DRB_PdcpSduDelayDl", help="Target metric (use actual CSV column name, e.g., DRB_PdcpSduDelayDl)")
    parser.add_argument("--target-value", type=float, default=40.0, help="Target value")
    parser.add_argument("--offline-model", type=str, default=None, help="Path to pre-trained Q-network model")
    parser.add_argument("--minirocket-model", type=str, default=None, 
                       help="Path to MiniRocket model (legacy, for single model). Use --minirocket-gnb-model and --minirocket-ue-model for separate gNB/UE models.")
    parser.add_argument("--minirocket-gnb-model", type=str, default=None,
                       help="Path to MiniRocket model for gNB-level metrics (e.g., models/minirocket_xapp_gnb.joblib)")
    parser.add_argument("--minirocket-ue-model", type=str, default=None,
                       help="Path to MiniRocket model for UE-level metrics (e.g., models/minirocket_xapp_ue.joblib)")
    parser.add_argument("--use-llm", action="store_true", help="Use LLM for intent reasoning (default: fallback)")
    parser.add_argument("--commands-enabled", action="store_true", default=True,
                       help="Enable sending control commands to xApp (default: enabled)")
    parser.add_argument("--commands-disabled", action="store_false", dest="commands_enabled",
                       help="Disable sending control commands to xApp (useful for KPI collection only)")
    parser.add_argument("--log-level", type=str, default="all",
                       help="Comma-separated log categories to enable: 1=LEARNING, 2=INTENT, 3=SCORING/PLAYBOOK, 4=DEVIATION/COMMANDS, 5=OBSERVER, 6=REWARD, 7=KPI, or 'all' (default: all)")
    
    args = parser.parse_args()
    
    if args.use_llm:
        logger.info("[GPT-2]: INITIALIZING BIASED GPT-2 ON AMD GPU...")
        try:
            # Force path so it finds llm_logic.py in the same folder
            import os
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            
            from llm_logic import generate_biased_gpt2_intent
            import ain.agents.reasoner_agent_enhanced as rae
            
            # Patch the function
            rae.create_network_intent_from_deviation = generate_biased_gpt2_intent
            
            logger.info("SUCCESS: Reasoner Agent patched with local Biased GPT-2")
        except Exception as e:
            logger.error(f"CRITICAL: LLM Injection Failed: {e}")
        print("="*60 + "\n")
    
    # Create TCP server
    tcp_server = XAppTCPServer(host=args.host, port=args.port, commands_enabled=args.commands_enabled)
    
    if not args.commands_enabled:
        logger.info("=" * 60)
        logger.info("COMMANDS DISABLED - AI will process KPIs but NOT send control commands")
        logger.info("This is useful for collecting KPIs from simulation without interference")
        logger.info("=" * 60)
    await tcp_server.start()
    
    try:
        await run_ai_loop_with_membus(
            tcp_server,
            target_metric=args.target_metric,
            target_value=args.target_value,
            offline_model=args.offline_model,
            minirocket_model=args.minirocket_model,
            minirocket_gnb_model=args.minirocket_gnb_model,
            minirocket_ue_model=args.minirocket_ue_model,
            use_llm=args.use_llm
        )
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        await tcp_server.stop()


if __name__ == "__main__":
    asyncio.run(main())
