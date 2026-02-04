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

from ain.loop.observer_rl import Intent, RLObserver
from ain.loop.predictor import SlateDQNPredictor
from ain.loop.proposer import ActionSpace, ProposerSampler, CacheLibrary, PLAYBOOK_K, CANDIDATE_N, COOLDOWN_STEPS, cooldown_key
from ain.loop.actor import Actor
from ain.bus.mem import MemBus
from ain.agents.utils import make_msg

# Import agents
from ain.agents.minirocket_agent import MinirocketAgent
from ain.agents.reasoner_agent_enhanced import EnhancedReasonerAgent
from ain.agents.proposer_agent import ProposerAgent
from ain.agents.predictor_agent import PredictorAgent
from ain.agents.observer_agent import RLObserverAgent
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


class PlaybookToCommandConverter:
    """Converts playbook actions to xApp control commands."""
    
    @staticmethod
    def playbook_to_commands(
        playbook, 
        meid: str, 
        node_id: int = 0,
        cell_to_node_map: Optional[Dict[str, int]] = None,
        ue_to_node_map: Optional[Dict[str, int]] = None,
        ue_to_cell_map: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Convert playbook actions to xApp control commands.
        
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
                    
            elif action.type in ("SCHEDULER_POLICY", "SLICE_QOS"):
                logger.warning(f"{action.type} action not directly supported: {action.params}")
                continue
            elif action.type == "REPORTING":
                continue
            
            if cmd:
                commands.append(cmd)
        
        return commands


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
                 converter: PlaybookToCommandConverter, default_meid: str = "gnb:131-133-31000000", 
                 node_id: int = 0):
        self.bus = bus
        self.actor = actor
        self.tcp_server = tcp_server
        self.converter = converter
        self.default_meid = default_meid
        self.node_id = node_id
        
    async def run(self):
        """Subscribe to scored playbooks and send commands."""
        q_scored = await self.bus.sub("predictor.scored")
        q_intent = await self.bus.sub("intent.current")
        
        active_intent = False
        
        while True:
            done, pending = await asyncio.wait(
                [asyncio.create_task(q_scored.get()), asyncio.create_task(q_intent.get())],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            for t in done:
                msg = t.result()
                
                if msg.topic == "intent.current":
                    # Update active intent state
                    intent = msg.payload
                    # Intent is active if it's a non-empty dictionary
                    was_active = active_intent
                    active_intent = bool(intent and isinstance(intent, dict))
                    if active_intent != was_active:
                        if active_intent:
                            logger.info(f"[ACTOR] Intent ACTIVATED: {intent.get('intent_id')}")
                        else:
                            logger.info(f"[ACTOR] Intent WITHDRAWN - halting command execution")
                    
                elif msg.topic == "predictor.scored":
                    scored = msg.payload.get("scored", [])
                    if not scored:
                        continue
                    
                    # COMMAND GATEWAY: Check if we have an active intent
                    if not active_intent:
                        if should_log(LOG_COMMANDS):
                            logger.warning(f"[ACTOR] Ignoring scored playbook - No active intent")
                        continue
                    
                    # Get best playbook
                    best_pb, best_q = max(scored, key=lambda t: t[1])
                    logger.info(f"Best playbook selected with Q={best_q:.3f}")
                    
                    # Save playbook
                    payload = self.actor.make_payload(best_pb)
                    self.actor.save_payload(payload)
                    logger.info(f"Saved playbook: {payload['playbook_id']}")
                    
                    # Convert to commands with mappings
                    meid = self.default_meid
                    cell_to_node_map = getattr(self.tcp_server, 'cell_to_node_map', {})
                    ue_to_node_map = getattr(self.tcp_server, 'ue_to_node_map', {})
                    ue_to_cell_map = getattr(self.tcp_server, 'ue_to_cell_map', {})
                    default_node_id = self.node_id
                    
                    # Log available mappings for debugging
                    logger.info(f"Available mappings - cells: {list(cell_to_node_map.keys())}, UEs: {list(ue_to_node_map.keys())[:5]}..., default_node_id: {default_node_id}")
                    
                    commands = self.converter.playbook_to_commands(
                        best_pb, 
                        meid, 
                        default_node_id,
                        cell_to_node_map=cell_to_node_map,
                        ue_to_node_map=ue_to_node_map,
                        ue_to_cell_map=ue_to_cell_map
                    )
                    
                    logger.info(f"Converted playbook to {len(commands)} command(s)")
                    
                    if not self.tcp_server.commands_enabled:
                        logger.info(f"[COMMANDS DISABLED] Would send {len(commands)} command(s) but commands are disabled")
                        # Still publish the event for logging/tracking
                        await self.bus.pub("actor.apply", make_msg(
                            "actor.apply", "APPLY", "actor.apply.v1",
                            {"playbook": best_pb, "q": best_q, "commands": commands, "commands_enabled": False}
                        ))
                        # We used to return here, but with logic change we should continue loop
                        continue
                    
                    # Check if intent is MONITORING (idle)
                    # If type is MONITORING or target is 0.0, it means we are in withdrawal/monitoring state
                    if active_intent and (intent.get("type") == "MONITORING" or intent.get("target") == 0.0):
                         if step % 10 == 0:
                             print(f"[AI Loop] In MONITORING state (no active intent). Skipping action generation.")
                         # Still update observer state to keep history for baseline learning
                         time.sleep(1.0) # Sleep and continue
                         step += 1
                         continue

                    # NEW: Enhanced proposer with contextual intelligence
                    # print(f"[Proposer] Generating {CANDIDATE_N} candidate playbooks (ε={predictor.epsilon():.3f})...")
                    if commands:
                        # Send to all connected clients
                        connected_clients = list(self.tcp_server.clients.keys())
                        if not connected_clients:
                            logger.warning("No clients connected, cannot send commands")
                        else:
                            for client_id in connected_clients:
                                # Use meid from map if available
                                meid = self.tcp_server.meid_map.get(client_id, self.default_meid)
                                # Update commands with correct meid
                                for i, cmd in enumerate(commands):
                                    cmd["meid"] = meid
                                    
                                    if should_log(LOG_COMMANDS):
                                        logger.info(f"Sending command {i+1}/{len(commands)}: {cmd.get('cmd', {}).get('cmd', 'unknown')} to node {cmd.get('cmd', {}).get('node', 'unknown')}")
                                    
                                    success = await self.tcp_server.send_command(client_id, cmd)
                                    if not success:
                                        logger.error(f"Failed to send command to {client_id}")
                                    
                                    # Add 2 second cooldown between commands (except for the last one)
                                    if i < len(commands) - 1:
                                        if should_log(LOG_COMMANDS):
                                            logger.info(f"Waiting 5 seconds before next command...")
                                        await asyncio.sleep(10.0)
                    else:
                        logger.warning("No commands generated from playbook")
                    
                    # Publish actor.apply event
                    await self.bus.pub("actor.apply", make_msg(
                        "actor.apply", "APPLY", "actor.apply.v1",
                        {"playbook": best_pb, "q": best_q, "commands": commands}
                    ))
            
            # Cancel pending tasks
            for t in pending:
                t.cancel()


class ObserverBridge:
    """Bridge between membus and RLObserver (file-based)."""
    
    def __init__(self, bus: MemBus, observer: RLObserver, temp_file: Path, knowledge_base: Optional[CacheLibrary] = None):
        self.bus = bus
        self.observer = observer
        self.temp_file = temp_file
        self.knowledge_base = knowledge_base
        self.last_playbook = None
        # Feature accumulation: merge KPIs from fragments using node_id as primary key
        # Key format: "node_{node_id}" for gNB-level, "cell_{cell_id}" for cell-specific, "ue_{ue_id}" for UE-specific
        self.accumulated_kpis: Dict[str, Dict[str, Any]] = {}  # accumulation_key -> accumulated KPI
        self.accumulation_timestamps: Dict[str, float] = {}  # accumulation_key -> last update time
        self.accumulation_timeout = 1.0  # seconds - process after this timeout even if incomplete (reduced for faster response)
        self.last_processed_time: Dict[str, float] = {}  # Track when we last processed to avoid duplicate processing
        self.min_process_interval = 0.2  # Minimum seconds between processing same accumulation (reduced for faster response)
        self.last_processed_metrics: Dict[str, Dict[str, float]] = {}  # Track last processed metric values to detect changes
        
    async def _listen_actor_apply(self, q):
        """Listen for applied playbooks to update last_playbook."""
        while True:
            msg = await q.get()
            playbook = msg.payload.get("playbook")
            if playbook:
                self.last_playbook = playbook
                logger.debug("Updated last_playbook from actor.apply")
        
    async def _periodic_process_accumulated(self):
        """Periodically check and process accumulated KPIs even if no new KPI arrives."""
        while True:
            await asyncio.sleep(0.1)  # 100ms processing interval
            
            # Process accumulated KPIs
            # We iterate over a copy of keys to avoid modification during iteration
            for accumulation_key in list(self.accumulated_kpis.keys()):
                acc_kpi = self.accumulated_kpis[accumulation_key] # Get the actual accumulated KPI
                acc_cell_metrics = acc_kpi["CellMetrics"]
                cell_id = acc_cell_metrics.get('cell_id', 'unknown')
                available_features = [k for k in self.observer.features if k in acc_cell_metrics and acc_cell_metrics[k] is not None]
                completeness = len(available_features) / len(self.observer.features) if self.observer.features else 0.0
                current_time = datetime.now(timezone.utc).timestamp() # Moved inside loop to ensure fresh time for each key
                time_since_start = current_time - self.accumulation_timestamps.get(accumulation_key, current_time)
                last_processed = self.last_processed_time.get(accumulation_key, 0)
                time_since_last_process = current_time - last_processed
                
                has_delay = acc_cell_metrics.get('DRB_PdcpSduDelayDl') is not None or acc_cell_metrics.get('UE_DRB_PdcpSduDelayDl_UEID') is not None
                has_cell_metrics = any(k in acc_cell_metrics and acc_cell_metrics[k] is not None 
                                      for k in ['RRU_PrbUsedDl', 'DRB_MeanActiveUeDl', 'UE_DRB_UEThpDl_UEID'])
                
                should_process = (
                    completeness >= self.observer.min_feature_completeness or
                    (has_delay and has_cell_metrics and time_since_start >= 0.2) or
                    (time_since_start >= self.accumulation_timeout and time_since_last_process >= self.min_process_interval)
                )
                
                if should_process:
                    # Check if metrics have actually changed since last processing
                    current_metrics = {k: v for k, v in acc_cell_metrics.items() 
                                     if k in ['DRB_PdcpSduDelayDl', 'RRU_PrbUsedDl', 'UE_DRB_PdcpSduDelayDl_UEID', 
                                             'UE_DRB_UEThpDl_UEID', 'DRB_MeanActiveUeDl'] and v is not None}
                    last_metrics = self.last_processed_metrics.get(accumulation_key, {})
                    
                    # Check if any tracked metrics have changed
                    metrics_changed = False
                    if not last_metrics:  # First time processing
                        metrics_changed = True
                    else:
                        for key, value in current_metrics.items():
                            if key not in last_metrics or abs(last_metrics[key] - value) > 1e-6:
                                metrics_changed = True
                                break
                    
                    # Skip processing if metrics haven't changed (unless timeout expired)
                    if not metrics_changed and time_since_start < self.accumulation_timeout:
                        continue
                    
                    # Process this accumulated KPI
                    self.last_processed_time[accumulation_key] = current_time
                    self.last_processed_metrics[accumulation_key] = current_metrics.copy()
                    import os
                    temp_file_tmp = self.temp_file.with_suffix(self.temp_file.suffix + ".tmp")
                    with open(temp_file_tmp, "w") as f:
                        json.dump({"kpi_stream": [acc_kpi]}, f)
                        f.flush()
                        os.fsync(f.fileno())
                    os.replace(temp_file_tmp, self.temp_file)
                    try:
                        state = self.observer.step(self.last_playbook)
                        
                        # NEW: Update knowledge base if reward is good
                        last_reward = getattr(self.observer, 'last_reward', None)
                        
                        has_reward = last_reward is not None
                        has_pb = self.last_playbook is not None
                        has_kb = self.knowledge_base is not None

                        if has_reward and has_pb and has_kb:
                            # Threshold for "Good" playbook. 
                            # Rewards are now [-0.3, 1.6]. So > 0.5 is a reasonable threshold for "This actually helped".
                            if last_reward > 0.5:
                                intent_dict = self.observer.intent.__dict__
                                # Also need situation for context
                                situation = getattr(self.observer, 'current_context', {}).get('situation', 'normal')
                                logger.info(f"[CACHE] Adding successful playbook to cache (reward={last_reward:.4f})")
                                self.knowledge_base.add_contextual(intent_dict, self.last_playbook, last_reward, situation)
                            else:
                                 # Diagnostic log (DEBUG level normally, but INFO for now to debug user issue)
                                 # No counter filter now
                                 logger.info(f"[CACHE] Skipped cache update: reward={last_reward:.4f} <= 0.5")
                        else:
                            # Log checking
                            if getattr(self, '_log_missing_counter', 0) % 5 == 0:
                                logger.info(f"[CACHE] Prerequisites missing: Reward={last_reward}, PB={'OK' if has_pb else 'NONE'}, KB={'OK' if has_kb else 'NONE'}")
                            self._log_missing_counter = getattr(self, '_log_missing_counter', 0) + 1

                        if state is not None:
                            import numpy as np
                            state_list = state.tolist() if hasattr(state, 'tolist') else state
                            await self.bus.pub("kpi.window", make_msg(
                                "kpi.window", "STATE_WINDOW", "kpi.window.v1",
                                {"state": state_list, "reward": 0.0}
                            ))
                    except Exception as e:
                        logger.debug(f"Error in periodic processing: {e}")
                        print(f"DEBUG_CACHE ERROR: {e}")
    
    async def run(self):
        """Subscribe to KPIs and update observer, publish state windows."""
        q = await self.bus.sub("kpi.raw")
        q_intent = await self.bus.sub("intent.rl")  # RL intent format
        q_actor_apply = await self.bus.sub("actor.apply")  # Listen for applied playbooks
        
        # Listen for intent updates and actor apply events
        asyncio.create_task(self._listen_intent(q_intent))
        asyncio.create_task(self._listen_actor_apply(q_actor_apply))
        # Start periodic processing task
        asyncio.create_task(self._periodic_process_accumulated())
        
        while True:
            msg = await q.get()
            kpi = msg.payload.get("kpi")
            
            
            if kpi:
                # Accumulate features from fragments using node_id as primary key
                cell_metrics = kpi.get('CellMetrics', {})
                raw_cell_id = cell_metrics.get('cell_id', 'unknown')
                current_time = datetime.now(timezone.utc).timestamp()
                
                # Extract node_id using robust logic
                node_id = 'unknown'
                
                # Extract node_id from KPI (prefer Header, then CellMetrics, then default to 2 for gNB)
                header = kpi.get('Header', {})
                node_id = cell_metrics.get('node_id') or header.get('node_id') or kpi.get('node_id')
                if node_id is None:
                    # Try to infer from cell_id or default to 2 (gNB)
                    if raw_cell_id.startswith('CELL_') or raw_cell_id == 'unknown':
                        node_id = 2  # gNB node
                    elif raw_cell_id.startswith('UE_'):
                        node_id = 3  # UE node (default)
                    else:
                        node_id = 2  # Default to gNB
                
                node_id = int(node_id)
                
                # Determine accumulation key based on node_id (not cell_id or meid)
                # Each node_id gets its own accumulation bucket
                accumulation_key = f"node_{node_id}"
                
                # Update cell_id in accumulated data if we see a known cell (for metadata only)
                if raw_cell_id.startswith('CELL_'):
                    if accumulation_key in self.accumulated_kpis:
                        acc_cell_metrics = self.accumulated_kpis[accumulation_key]['CellMetrics']
                        if acc_cell_metrics.get('cell_id') == 'unknown' or not acc_cell_metrics.get('cell_id'):
                            acc_cell_metrics['cell_id'] = raw_cell_id
                            logger.debug(f"Updated cell_id from 'unknown' to {raw_cell_id} for node_id={node_id} accumulation")
                
                # Initialize or update accumulated KPI for this node_id
                if accumulation_key not in self.accumulated_kpis:
                    # Start new accumulation for this node_id
                    self.accumulated_kpis[accumulation_key] = {
                        "timestamp": kpi.get("timestamp", datetime.now(timezone.utc).isoformat()),
                        "Header": kpi.get("Header", {}),
                        "CellMetrics": cell_metrics.copy(),
                        "UEMetrics": kpi.get("UEMetrics", [])
                    }
                    # Ensure cell_id and node_id are set (use best available)
                    if not self.accumulated_kpis[accumulation_key]["CellMetrics"].get('cell_id') or \
                       self.accumulated_kpis[accumulation_key]["CellMetrics"].get('cell_id') == 'unknown':
                        if raw_cell_id.startswith('CELL_'):
                            self.accumulated_kpis[accumulation_key]["CellMetrics"]['cell_id'] = raw_cell_id
                        else:
                            self.accumulated_kpis[accumulation_key]["CellMetrics"]['cell_id'] = raw_cell_id
                    # Ensure node_id is set
                    if not self.accumulated_kpis[accumulation_key]["CellMetrics"].get('node_id'):
                        self.accumulated_kpis[accumulation_key]["CellMetrics"]['node_id'] = node_id
                    self.accumulation_timestamps[accumulation_key] = current_time
                else:
                    # Merge new features into accumulated KPI for this node_id
                    acc_cell_metrics = self.accumulated_kpis[accumulation_key]["CellMetrics"]
                    
                    # Merge cell metrics for this node_id
                    for key, value in cell_metrics.items():
                        if value is not None:  # Only update with non-None values
                            # Update cell_id if we see a known cell (prefer known over unknown)
                            if key == 'cell_id':
                                if value.startswith('CELL_') and (acc_cell_metrics.get('cell_id') == 'unknown' or not acc_cell_metrics.get('cell_id')):
                                    acc_cell_metrics[key] = value
                                    logger.debug(f"Updated cell_id to {value} for {accumulation_key}")
                                elif not acc_cell_metrics.get('cell_id') or acc_cell_metrics.get('cell_id') == 'unknown':
                                    acc_cell_metrics[key] = value
                            else:
                                acc_cell_metrics[key] = value
                    
                    # Ensure node_id is set
                    if not acc_cell_metrics.get('node_id'):
                        acc_cell_metrics['node_id'] = node_id
                    
                    # Merge UE metrics from this KPI
                    ue_metrics = kpi.get("UEMetrics", [])
                    if ue_metrics:
                        existing_ue_metrics = self.accumulated_kpis[accumulation_key].get("UEMetrics", [])
                        # Add new UE metrics (avoid duplicates)
                        for new_ue in ue_metrics:
                            new_ue_id = new_ue.get("ue_id") or new_ue.get("ueId")
                            if new_ue_id:
                                # Check if this UE already exists
                                existing = next((ue for ue in existing_ue_metrics if (ue.get("ue_id") or ue.get("ueId")) == new_ue_id), None)
                                if existing:
                                    # Merge metrics
                                    for key, value in new_ue.items():
                                        if value is not None:
                                            existing[key] = value
                                else:
                                    existing_ue_metrics.append(new_ue)
                        self.accumulated_kpis[accumulation_key]["UEMetrics"] = existing_ue_metrics
                        
                        # Also aggregate UE-level metrics into cell-level metrics (for global context)
                        # Use actual CSV column names: UE_DRB_PdcpSduDelayDl_UEID, UE_DRB_UEThpDl_UEID, etc.
                        # Aggregate UE delays (use max for delay)
                        ue_delays = [float(ue.get("UE_DRB_PdcpSduDelayDl_UEID", 0)) for ue in ue_metrics if ue.get("UE_DRB_PdcpSduDelayDl_UEID") is not None]
                        if ue_delays and not acc_cell_metrics.get('UE_DRB_PdcpSduDelayDl_UEID'):
                            acc_cell_metrics['UE_DRB_PdcpSduDelayDl_UEID'] = max(ue_delays)
                            if should_log(LOG_OBSERVER):
                                logger.debug(f"Aggregated max UE delay {max(ue_delays):.2f}ms into cell metrics for node_id={node_id}")
                        
                        # Aggregate UE throughputs (sum for throughput)
                        ue_throughputs = [float(ue.get("UE_DRB_UEThpDl_UEID", 0)) for ue in ue_metrics if ue.get("UE_DRB_UEThpDl_UEID") is not None]
                        if ue_throughputs and not acc_cell_metrics.get('UE_DRB_UEThpDl_UEID'):
                            acc_cell_metrics['UE_DRB_UEThpDl_UEID'] = sum(ue_throughputs)
                            if should_log(LOG_OBSERVER):
                                logger.debug(f"Aggregated total UE throughput {sum(ue_throughputs)/1e6:.2f}Mbps into cell metrics for node_id={node_id}")
                        
                        # Aggregate UE PRB usage (sum)
                        ue_prbs = [float(ue.get("UE_RRU_PrbUsedDl_UEID", 0)) for ue in ue_metrics if ue.get("UE_RRU_PrbUsedDl_UEID") is not None]
                        if ue_prbs and not acc_cell_metrics.get('UE_RRU_PrbUsedDl_UEID'):
                            acc_cell_metrics['UE_RRU_PrbUsedDl_UEID'] = sum(ue_prbs)
                            if should_log(LOG_OBSERVER):
                                logger.debug(f"Aggregated total UE PRB usage {sum(ue_prbs):.0f} into cell metrics for node_id={node_id}")
                        
                        # Aggregate UE DRB establishment success (sum)
                        ue_estab = [float(ue.get("UE_DRB_EstabSucc_5QI_UEID", 0)) for ue in ue_metrics if ue.get("UE_DRB_EstabSucc_5QI_UEID") is not None]
                        if ue_estab and not acc_cell_metrics.get('UE_DRB_EstabSucc_5QI_UEID'):
                            acc_cell_metrics['UE_DRB_EstabSucc_5QI_UEID'] = sum(ue_estab)
                            if should_log(LOG_OBSERVER):
                                logger.debug(f"Aggregated UE DRB establishment success {sum(ue_estab):.0f} into cell metrics for node_id={node_id}")
                        
                        # Count active UEs
                        if not acc_cell_metrics.get('DRB_MeanActiveUeDl'):
                            acc_cell_metrics['DRB_MeanActiveUeDl'] = len(ue_metrics)
                
                # Process accumulations for all node_ids (not just gNB)
                # Check if we should process this accumulated KPI
                acc_kpi = self.accumulated_kpis[accumulation_key]
                acc_cell_metrics = acc_kpi["CellMetrics"]
                cell_id = acc_cell_metrics.get('cell_id', 'unknown')  # Use accumulated cell_id for logging
                available_features = [k for k in self.observer.features if k in acc_cell_metrics and acc_cell_metrics[k] is not None]
                completeness = len(available_features) / len(self.observer.features) if self.observer.features else 0.0
                time_since_start = current_time - self.accumulation_timestamps[accumulation_key]  # FIX: use accumulation_key, not cell_id
                
                # Check if we have critical features (delay AND cell metrics) before processing
                has_delay = acc_cell_metrics.get('DRB_PdcpSduDelayDl') is not None or acc_cell_metrics.get('UE_DRB_PdcpSduDelayDl_UEID') is not None
                has_cell_metrics = any(k in acc_cell_metrics and acc_cell_metrics[k] is not None 
                                      for k in ['RRU_PrbUsedDl', 'DRB_MeanActiveUeDl', 'UE_DRB_UEThpDl_UEID'])
                
                # Process accumulations for all node_ids (not just gNB)
                # Process if:
                # 1. We have sufficient completeness, OR
                # 2. We have both delay and cell metrics (even if completeness is low), OR
                # 3. Timeout expired (but only if we haven't processed recently)
                last_processed = self.last_processed_time.get(accumulation_key, 0)
                time_since_last_process = current_time - last_processed
                
                should_process = (
                    completeness >= self.observer.min_feature_completeness or
                    (has_delay and has_cell_metrics and time_since_start >= 0.2) or  # Wait at least 0.2s for fragments to arrive (reduced)
                    (time_since_start >= self.accumulation_timeout and time_since_last_process >= self.min_process_interval)  # Avoid duplicate processing
                )
                
                # Log why we're processing or not processing
                if should_log(LOG_OBSERVER):
                    if should_process:
                        reason = []
                        if completeness >= self.observer.min_feature_completeness:
                            reason.append(f"completeness {completeness:.1%} >= min")
                        if has_delay and has_cell_metrics and time_since_start >= 0.5:
                            reason.append(f"has_delay+cell_metrics after {time_since_start:.1f}s")
                        if time_since_start >= self.accumulation_timeout and time_since_last_process >= 1.0:
                            reason.append(f"timeout {time_since_start:.1f}s >= {self.accumulation_timeout}s")
                        logger.debug(f"[OBSERVER] Will process: {', '.join(reason)}")
                    else:
                        logger.debug(f"[OBSERVER] Skipping: completeness={completeness:.1%}, has_delay={has_delay}, has_cell={has_cell_metrics}, time={time_since_start:.1f}s, last_process={time_since_last_process:.1f}s")
                
                if should_process:
                    # Check if metrics have actually changed since last processing
                    current_metrics = {k: v for k, v in acc_cell_metrics.items() 
                                     if k in ['DRB_PdcpSduDelayDl', 'RRU_PrbUsedDl', 'UE_DRB_PdcpSduDelayDl_UEID', 
                                             'UE_DRB_UEThpDl_UEID', 'DRB_MeanActiveUeDl'] and v is not None}
                    last_metrics = self.last_processed_metrics.get(accumulation_key, {})
                    
                    # Check if any tracked metrics have changed
                    metrics_changed = False
                    if not last_metrics:  # First time processing
                        metrics_changed = True
                    else:
                        for key, value in current_metrics.items():
                            if key not in last_metrics or abs(last_metrics[key] - value) > 1e-6:
                                metrics_changed = True
                                break
                    
                    # Skip processing if metrics haven't changed (unless timeout expired)
                    if not metrics_changed and time_since_start < self.accumulation_timeout:
                        if should_log(LOG_OBSERVER):
                            logger.debug(f"[OBSERVER] Skipping processing: metrics unchanged for {accumulation_key}")
                        continue
                    
                    # Update last processed time and metrics to avoid duplicate processing
                    self.last_processed_time[accumulation_key] = current_time
                    self.last_processed_metrics[accumulation_key] = current_metrics.copy()
                    
                    # Write accumulated KPI to observer's file (atomic write)
                    # Direct pass to observer (skip file I/O for performance)
                    try:
                        # Pass kpi_dict directly to avoid file I/O latency
                        state = self.observer.step(self.last_playbook, kpi_dict=acc_kpi)
                        if state is not None:
                            # Publish state window
                            try:
                                import numpy as np
                                state_list = state.tolist() if hasattr(state, 'tolist') else state
                            except ImportError:
                                state_list = state if isinstance(state, list) else state.tolist() if hasattr(state, 'tolist') else list(state)
                            
                            # Get state shape info
                            if isinstance(state_list, list) and len(state_list) > 0:
                                if isinstance(state_list[0], list):
                                    shape_info = f"[{len(state_list)}, {len(state_list[0])}]"
                                else:
                                    shape_info = f"[{len(state_list)}]"
                            else:
                                shape_info = "unknown"
                            
                            # Extract situation for context-aware agents
                            situation = "normal"
                            curr_ctx = getattr(self.observer, 'current_context', None)
                            if curr_ctx:
                                if isinstance(curr_ctx, dict):
                                    situation = curr_ctx.get('situation', 'normal')
                                else:
                                    situation = getattr(curr_ctx, 'situation', 'normal')

                            await self.bus.pub("kpi.window", make_msg(
                                "kpi.window", "STATE_WINDOW", "kpi.window.v1",
                                {
                                    "state": state_list,
                                    "reward": 0.0,  # Observer computes reward internally
                                    "situation": situation, # Broadcast situation for Proposer
                                }
                            ))
                            if should_log(LOG_OBSERVER):
                                logger.info(f"[OBSERVER] Published state window to membus (shape: {shape_info})")

                            # NEW: Update knowledge base if reward is good (Moved from periodic loop)
                            last_reward = getattr(self.observer, 'last_reward', None)
                            
                            if last_reward is not None and self.last_playbook is not None and self.knowledge_base is not None:
                                # Threshold for "Good" playbook. 
                                # Rewards are now [-0.3, 1.6]. So > 0.5 is a reasonable threshold.
                                if last_reward > 0.5:
                                    intent_dict = self.observer.intent.__dict__.copy() # Use copy to avoid modifying original
                                    # Handle NetworkContext object vs dict
                                    curr_ctx = getattr(self.observer, 'current_context', None)
                                    scope = {}
                                    if isinstance(curr_ctx, dict):
                                        situation = curr_ctx.get('situation', 'normal')
                                        scope = curr_ctx.get('scope', {})
                                    else:
                                        situation = getattr(curr_ctx, 'situation', 'normal')
                                        scope = getattr(curr_ctx, 'scope', {})
                                    
                                    # Fallback: Populate scope from accumulated KPI if missing in context
                                    if not scope and acc_kpi:
                                        # Try to get cell_id from CellMetrics
                                        cell_metrics = acc_kpi.get('CellMetrics', {})
                                        kpi_cell_id = cell_metrics.get('cell_id')
                                        
                                        # Verify this cell_id is valid (starts with CELL_)
                                        if kpi_cell_id and str(kpi_cell_id).startswith('CELL_'):
                                            scope = {'cell_id': kpi_cell_id}
                                            # Optional: Add region if available (hardcoded for now to match reasoner)
                                            # scope['region'] = 'A' 
                                        elif kpi_cell_id == 'unknown' and 'CELL_001' in CELLS: # Fallback to configured cell
                                             # This handles the case where xApp hasn't seen cell ID yet but we know what we are controlling
                                             scope = {'cell_id': CELLS.split(',')[0]}

                                    
                                    # Inject scope into intent_dict for correct key generation
                                    intent_dict['scope'] = scope
                                    
                                    
                                    logger.info(f"[CACHE] Adding successful playbook to cache (reward={last_reward:.4f})")
                                    self.knowledge_base.add_contextual(intent_dict, self.last_playbook, last_reward, situation)
                                else:
                                     # Diagnostic log
                                     if should_log(LOG_BANDIT):
                                         logger.info(f"[CACHE] Skipped cache update: reward={last_reward:.4f} <= 0.5")
                        else:
                            if should_log(LOG_OBSERVER):
                                logger.debug(f"[OBSERVER] Observer returned None state - window may not be full yet (buf size: {len(self.observer.buf)})")
                    except Exception as e:
                        logger.error(f"Error in observer.step(): {e}")
                        import traceback
                        traceback.print_exc()
                    
                    # Clear accumulated KPI after processing
                    if accumulation_key in self.accumulated_kpis:
                        del self.accumulated_kpis[accumulation_key]
                    if accumulation_key in self.accumulation_timestamps:
                        del self.accumulation_timestamps[accumulation_key]
                else:
                    # Still accumulating - log progress
                    logger.debug(f"Accumulating KPI for {cell_id}: {len(available_features)}/{len(self.observer.features)} features, {completeness:.1%} complete, {time_since_start:.2f}s elapsed")
                
                # Clean up old accumulated KPIs (timeout expired) - only for gNB
                # Check timeout for all node_id accumulations
                    expired_keys = [
                        key for key, ts in self.accumulation_timestamps.items()
                        if current_time - ts >= self.accumulation_timeout and key in self.accumulated_kpis
                    ]
                    for key in expired_keys:
                        if key in self.accumulated_kpis:
                            logger.warning(f"Timeout: Processing incomplete KPI for {key} after {self.accumulation_timeout}s")
                            # Process even if incomplete (pass directly)
                            acc_kpi = self.accumulated_kpis[key]
                            try:
                                state = self.observer.step(self.last_playbook, kpi_dict=acc_kpi)
                                if state is not None:
                                    import numpy as np
                                    state_list = state.tolist() if hasattr(state, 'tolist') else state
                                    await self.bus.pub("kpi.window", make_msg(
                                        "kpi.window", "STATE_WINDOW", "kpi.window.v1",
                                        {"state": state_list, "reward": 0.0}
                                    ))
                            except Exception as e:
                                logger.error(f"Error processing expired KPI: {e}")
                            del self.accumulated_kpis[key]
                            if key in self.accumulation_timestamps:
                                del self.accumulation_timestamps[key]
    
    async def _listen_intent(self, q):
        """Listen for RL intent updates."""
        while True:
            msg = await q.get()
            rl_intent = msg.payload
            # Check if intent is MONITORING (idle)
            # If type is MONITORING or target is 0.0, it means we are in withdrawal/monitoring state
            # This check should be done in the main loop that consumes the intent, not here.
            # This method's sole purpose is to update the observer's intent.
            
            self.observer.intent = Intent(
                type=rl_intent.get("type", "REDUCE_LATENCY"),
                metric=rl_intent.get("metric", "DRB_PdcpSduDelayDl"),  # Use actual CSV metric name
                target=float(rl_intent.get("target", 40.0)),
                direction=rl_intent.get("direction", "lower_better"),
                action_cost=float(rl_intent.get("action_cost", 0.01)),
                reward_clip=float(rl_intent.get("reward_clip", 20.0)),
            )
            logger.info(f"Observer intent updated: {self.observer.intent}")


async def run_ai_loop_with_membus(
    tcp_server: XAppTCPServer,
    cells: List[str],
    slices: List[str],
    target_metric: str = "DRB_PdcpSduDelayDl",  # Use actual CSV metric name
    target_value: float = 40.0,
    steps: int = 1000,
    offline_model: Optional[str] = None,
    minirocket_model: Optional[str] = None,
    minirocket_gnb_model: Optional[str] = None,
    minirocket_ue_model: Optional[str] = None,
    use_llm: bool = False,
    model_type: str = "stable",
    loss_history_file: Optional[str] = None,
    slo_file: Optional[str] = None,
):
    """Run the AI loop using membus architecture matching system design."""
    # Create membus
    bus = MemBus()
    tcp_server.bus = bus
    
    # Initialize components
    action_space = ActionSpace(cells=cells, slices=slices)
    knowledge_base = CacheLibrary(max_per_key=20)  # Knowledge Base shared by Reasoner & Proposer
    
    # Create predictor and observer
    tmp_intent = Intent(type="REDUCE_LATENCY", metric=target_metric, target=target_value)
    
    # Create temp file for observer
    temp_kpi_file = Path(tempfile.gettempdir()) / "xapp_kpi_temp.json"
    temp_kpi_file.parent.mkdir(parents=True, exist_ok=True)
    with open(temp_kpi_file, "w") as f:
        json.dump({"kpi_stream": []}, f)
    
    # Create observer with feature completeness threshold (lowered to work with sparse KPIs)
    dummy_observer = RLObserver(
        predictor=None, 
        intent=tmp_intent, 
        kpi_file=str(temp_kpi_file), 
        window=12,
        min_feature_completeness=0.1  # Require at least 10% of features (1 out of 10) to be present
    )
    predictor = SlateDQNPredictor(
        action_space, 
        feat_dim=len(dummy_observer.features), 
        seed=0
    )
    
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
    
    # Load SLO config for multi-metric reward calculation
    slo_config = None
    if slo_file:
        try:
            from ain.agents.reasoner_agent_enhanced import SLOConfig
            slo_config = SLOConfig(slo_file=slo_file)
            logger.info(f"Loaded SLO config from {slo_file} for multi-metric reward calculation")
        except Exception as e:
            logger.warning(f"Could not load SLO config: {e}, using single-metric reward")
    
    observer = RLObserver(
        predictor=predictor, 
        intent=tmp_intent, 
        kpi_file=str(temp_kpi_file), 
        window=12,
        min_feature_completeness=0.1,  # Require at least 10% of features (1 out of 10) to be present
        enable_multi_metric_reward=True,  # Enable multi-metric reward calculation
        slo_config=slo_config  # Pass SLO config for multi-metric rewards
    )
    
    # Create actor
    actor = Actor("playbooks")
    converter = PlaybookToCommandConverter()
    actor_agent = XAppActorAgent(bus, actor, tcp_server, converter)
    
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
    
    # 2. Enhanced Reasoner Agent (deviation.detected + SLO Intents → intent.current)
    reasoner_agent = EnhancedReasonerAgent(
        bus,
        knowledge_base=knowledge_base,
        use_llm=use_llm,
        slo_file=slo_file
    )
    
    # NOTE: Removed initial intent publishing - intents should only be created from deviations
    # The reasoner agent will create intents when deviations are detected by Minirocket
    # 3. Proposer Agent (intent.current + Knowledge Base → proposer.candidates)
    proposer_agent = ProposerAgent(
        bus,
        action_space,
        knowledge_base=knowledge_base
    )
    
    # 4. Predictor Agent (kpi.window + proposer.candidates → predictor.scored)
    predictor_agent = PredictorAgent(bus, predictor)
    
    # 5. Observer Bridge (kpi.raw → observer → kpi.window)
    observer_bridge = ObserverBridge(bus, observer, temp_kpi_file, knowledge_base=knowledge_base)
    
    # Setup checkpoint saving
    online_checkpoint = "models/qnet_online.pt"
    
    # Determine loss history file path
    if loss_history_file:
        loss_history_path = loss_history_file
    else:
        loss_history_path = f"models/loss_history_{model_type}.json"
    
    # Load loss history if it exists to preserve continuity
    if predictor.load_loss_history(loss_history_path):
        logger.info(f"✓ Resumed loss history from {loss_history_path} ({len(predictor.loss_history)} entries)")
    else:
        logger.info(f"Starting new loss history at {loss_history_path}")
    
    async def periodic_checkpoint_saver():
        """Periodically save checkpoint every 50 steps."""
        while True:
            await asyncio.sleep(30)  # Check every 30 seconds
            if predictor.steps > 0 and predictor.steps % 50 == 0:
                try:
                    checkpoint_path = predictor.save_checkpoint(online_checkpoint, save_replay_buffer=False)
                    logger.info(f"💾 Saved periodic checkpoint to {checkpoint_path} (step={predictor.steps}, replay_size={len(predictor.replay)})")
                    # Also save loss history periodically
                    predictor.save_loss_history(loss_history_path)
                except Exception as e:
                    logger.warning(f"Failed to save periodic checkpoint: {e}")
    
    def save_on_exit(signum=None, frame=None):
        """Save checkpoint and loss history before exiting."""
        try:
            checkpoint_path = predictor.save_checkpoint(online_checkpoint, save_replay_buffer=False)
            logger.info(f"💾 Saved final checkpoint to {checkpoint_path} (step={predictor.steps})")
            # Save loss history on exit
            predictor.save_loss_history(loss_history_path)
        except Exception as e:
            logger.error(f"Failed to save checkpoint on exit: {e}")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, save_on_exit)
    signal.signal(signal.SIGTERM, save_on_exit)
    
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
    
    tasks = minirocket_tasks + dashboard_tasks + [
        asyncio.create_task(reasoner_agent.run(), name="reasoner_agent"),
        asyncio.create_task(proposer_agent.run(), name="proposer_agent"),
        asyncio.create_task(predictor_agent.run(), name="predictor_agent"),
        asyncio.create_task(observer_bridge.run(), name="observer_bridge"),
        asyncio.create_task(actor_agent.run(), name="actor_agent"),
        asyncio.create_task(periodic_checkpoint_saver(), name="checkpoint_saver"),
    ]
    
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
    finally:
        # Save final checkpoint and loss history before exiting
        try:
            checkpoint_path = predictor.save_checkpoint(online_checkpoint, save_replay_buffer=False)
            logger.info(f"💾 Saved final checkpoint to {checkpoint_path} (step={predictor.steps})")
            # Save loss history on exit (use the loss_history_path from outer scope)
            predictor.save_loss_history(loss_history_path)
        except Exception as e:
            logger.error(f"Failed to save final checkpoint: {e}")
        
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def main():
    parser = argparse.ArgumentParser(description="AI System Demo with xApp TCP Integration (Membus)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="TCP server host")
    parser.add_argument("--port", type=int, default=6000, help="TCP server port")
    parser.add_argument("--steps", type=int, default=1000, help="Max steps")
    parser.add_argument("--target-metric", type=str, default="DRB_PdcpSduDelayDl", help="Target metric (use actual CSV column name, e.g., DRB_PdcpSduDelayDl)")
    parser.add_argument("--target-value", type=float, default=40.0, help="Target value")
    parser.add_argument("--cells", type=str, nargs="+", default=["CELL_001"], help="Cell IDs")
    parser.add_argument("--slices", type=str, nargs="+", default=["SLICE_A"], help="Slice IDs")
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
    parser.add_argument("--model-type", type=str, default="stable", choices=["stable", "unstable"],
                       help="Model type: 'stable' (uses target network with soft updates) or 'unstable' (no target network or hard updates). Affects loss history filename.")
    parser.add_argument("--loss-history-file", type=str, default=None,
                       help="Path to save loss history JSON file. Default: models/loss_history_{model_type}.json")
    parser.add_argument("--slo-file", type=str, default="configs/slos.json",
                       help="Path to SLO JSON configuration file (default: configs/slos.json)")
    
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
            cells=args.cells,
            slices=args.slices,
            target_metric=args.target_metric,
            target_value=args.target_value,
            steps=args.steps,
            offline_model=args.offline_model,
            minirocket_model=args.minirocket_model,
            minirocket_gnb_model=args.minirocket_gnb_model,
            minirocket_ue_model=args.minirocket_ue_model,
            use_llm=args.use_llm,
            model_type=args.model_type,
            loss_history_file=args.loss_history_file,
            slo_file=args.slo_file,
        )
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        await tcp_server.stop()


if __name__ == "__main__":
    asyncio.run(main())
