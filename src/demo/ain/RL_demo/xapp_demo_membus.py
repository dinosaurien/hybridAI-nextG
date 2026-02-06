#!/usr/bin/env python3
"""
Simplified AI System: KPI Collection & Anomaly Detection Only.

Pipeline:
1. xApp (TCP) -> XAppTCPServer -> MemBus (kpi.raw)
2. MemBus (kpi.raw) -> MinirocketAgent -> MemBus (deviation.detected)
3. CSV Logger tracks all incoming KPIs.
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
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

# Add parent directory to path
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent.parent.parent))

from ain.bus.mem import MemBus
from ain.agents.utils import make_msg
from ain.agents.minirocket_agent import MinirocketAgent

from ain.common.log_config import (
    should_log, LOG_KPI, LOG_DEVIATION
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class XAppKPIAdapter:
    """Converts xApp KPI format to internal format."""
    
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
        if ues:
            for ue in ues:
                ue_metric = {}
                ue_id = ue.get("ue_id") or ue.get("ueId") or ue.get("id")
                if not ue_id:
                    continue
                
                ue_metric["ue_id"] = str(ue_id)
                ue_metric["cell_id"] = ue.get("cell_id") or ue.get("cellId") or cell_metrics.get("cell_id", "unknown")
                
                # Extract from nested measurements if available
                ue_measurements = ue.get("measurements", [])
                for m in ue_measurements:
                    name = m.get("name", "")
                    value = m.get("value", 0)
                    name_normalized = name.replace('.', '_').lower()
                    
                    if "drb_pdcpsdudelaydl_ueid" in name_normalized or ("delay" in name_normalized and "pdcp" in name_normalized and "ue" in name_normalized):
                        ue_metric["UE_DRB_PdcpSduDelayDl_UEID"] = float(value)
                    elif "drb_uethpdl_ueid" in name_normalized or ("throughput" in name_normalized and "ue" in name_normalized and "dl" in name_normalized):
                        ue_metric["UE_DRB_UEThpDl_UEID"] = float(value) * 1e6 
                    elif "rru_prbuseddl_ueid" in name_normalized:
                        ue_metric["UE_RRU_PrbUsedDl_UEID"] = float(value)
                    elif "drb_estabsucc_5qi_ueid" in name_normalized:
                        ue_metric["UE_DRB_EstabSucc_5QI_UEID"] = float(value)
                
                if ue_metric:
                    ue_metrics.append(ue_metric)
        
        if measurements:
            for m in measurements:
                name = m.get("name", "")
                value = m.get("value", 0)
                name_normalized = name.replace('.', '_').lower()
                
                if "drb_pdcpsdudelaydl" in name_normalized and "ueid" not in name_normalized:
                    cell_metrics["DRB_PdcpSduDelayDl"] = float(value)
                elif "rru_prbuseddl" in name_normalized:
                    cell_metrics["RRU_PrbUsedDl"] = float(value)
                elif "drb_meanactiveuedl" in name_normalized:
                    cell_metrics["DRB_MeanActiveUeDl"] = float(value)
                elif "throughput" in name.lower() and "dl" in name.lower():
                     cell_metrics["thr_dl_bps"] = float(value) * 1e6

        # Normalize and set cell_id
        cell_id_raw = kpi_data.get("cellObjectID") or kpi_data.get("cell_id") or "CELL_001"
        if cell_id_raw and str(cell_id_raw).isdigit():
            cell_id = f"CELL_{cell_id_raw}"
        else:
            cell_id = cell_id_raw if cell_id_raw != "unknown" else "CELL_001"
            
        if "cell_id" not in cell_metrics:
            cell_metrics["cell_id"] = cell_id
        
        # Build internal format
        internal_kpi = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "Header": {
                "ric_instance_id": meid,
                "function_id": "kpm_func_v1",
                "granularity_period_ms": 1000,
            },
            "CellMetrics": cell_metrics,
            "UEMetrics": ue_metrics,
        }
        
        return internal_kpi


class XAppTCPServer:
    """TCP server for xApp communication that publishes to membus."""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 6000, bus: Optional[MemBus] = None, kpi_csv_file: Optional[str] = None):
        self.host = host
        self.port = port
        self.server: Optional[asyncio.Server] = None
        self.clients: Dict[str, asyncio.StreamWriter] = {}
        self.bus = bus
        self.meid_map: Dict[str, str] = {}
        self.kpi_adapter = XAppKPIAdapter()
        
        # UE and node tracking
        self.cell_to_node_map: Dict[str, int] = {} 
        self.client_node_map: Dict[str, int] = {}
        
        # CSV logging setup
        if kpi_csv_file is None:
            project_root = THIS_DIR.parent.parent.parent.parent
            self.kpi_csv_file = project_root / "kpms.csv"
        else:
            self.kpi_csv_file = Path(kpi_csv_file)
            
        self.kpi_csv_initialized = False
        self.kpi_csv_fieldnames = None
        self.kpi_csv_lock = threading.Lock()
        self.kpi_csv_write_count = 0
        self._init_kpi_csv()
    
    def _init_kpi_csv(self):
        """Initialize KPI CSV file with base headers."""
        base_fieldnames = ['timestamp', 'meid', 'cell_id', 'node_id', 'format']
        if not self.kpi_csv_file.exists():
            with open(self.kpi_csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(base_fieldnames)
            self.kpi_csv_fieldnames = base_fieldnames
            self.kpi_csv_initialized = True
            logger.info(f"Initialized KPI CSV file: {self.kpi_csv_file}")
        else:
            try:
                with open(self.kpi_csv_file, 'r') as f:
                    reader = csv.DictReader(f)
                    self.kpi_csv_fieldnames = list(reader.fieldnames) if reader.fieldnames else base_fieldnames
            except Exception:
                self.kpi_csv_fieldnames = base_fieldnames
            self.kpi_csv_initialized = True
    
    def _write_kpi_to_csv(self, kpi_data: Dict[str, Any], meid: str, cell_id: str, node_id: Optional[int] = None, is_ue_data: bool = False):
        """Write KPI data to CSV file."""
        if not self.kpi_csv_initialized:
            self._init_kpi_csv()
        
        try:
            timestamp = datetime.now(timezone.utc).isoformat()
            measurements = kpi_data.get("measurements", [])
            
            metrics = {}
            for m in measurements:
                name = m.get("name", "")
                if not name: continue
                value = m.get("value", "")
                csv_name = name.replace(".", "_").replace(" ", "_")
                if is_ue_data and not csv_name.startswith("UE_"):
                    csv_name = f"UE_{csv_name}"
                metrics[csv_name] = value
            
            with self.kpi_csv_lock:
                row = {
                    'timestamp': timestamp,
                    'meid': meid,
                    'cell_id': cell_id,
                    'node_id': node_id if node_id is not None else '',
                    'format': kpi_data.get("format", "F1"),
                }
                row.update(metrics)
                
                # Check for new columns and rewrite header if necessary (simplified for brevity)
                # In full version this handles dynamic schema evolution
                
                with open(self.kpi_csv_file, 'a', newline='') as f:
                    # Note: strict field matching disabled to allow dynamic metrics
                    writer = csv.DictWriter(f, fieldnames=self.kpi_csv_fieldnames, extrasaction='ignore')
                    writer.writerow(row)

        except Exception as e:
            logger.error(f"Error writing KPI to CSV: {e}")
        
    async def start(self):
        self.server = await asyncio.start_server(self._handle_client, self.host, self.port)
        logger.info(f"xApp TCP server started on {self.host}:{self.port}")
        
    async def stop(self):
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            
    async def _handle_client(self, reader, writer):
        client_addr = writer.get_extra_info('peername')
        client_id = f"{client_addr[0]}:{client_addr[1]}"
        logger.info(f"xApp client connected: {client_id}")
        self.clients[client_id] = writer
        
        try:
            while True:
                # Read 4-byte header
                header = await reader.read(4)
                if not header: break
                
                length = struct.unpack("!I", header)[0]
                body = await reader.read(length)
                if not body: break
                
                try:
                    text = body.decode("utf-8")
                    message = json.loads(text)
                    
                    if message.get("type") == "kpi":
                        meid = message.get("meid", "unknown")
                        kpi_data = message.get("kpi", {})
                        cell_id = kpi_data.get("cell_id") or "unknown"
                        
                        # Write CSV
                        self._write_kpi_to_csv(kpi_data, meid, cell_id)
                        
                        # Convert and Publish to Bus
                        internal_kpi = self.kpi_adapter.convert_xapp_kpi_to_internal(message, meid)
                        if self.bus:
                            await self.bus.pub("kpi.raw", make_msg(
                                "kpi.raw", "KPI", "kpi.raw.v1",
                                {"kpi": internal_kpi, "client_id": client_id}
                            ))
                            
                except json.JSONDecodeError:
                    logger.error("JSON decode error")
                    
        except Exception as e:
            logger.error(f"Connection error: {e}")
        finally:
            if client_id in self.clients: del self.clients[client_id]
            writer.close()
            logger.info(f"Client disconnected: {client_id}")


async def log_deviations(bus: MemBus):
    """Simple listener to log detected deviations to console."""
    q = await bus.sub("deviation.detected")
    logger.info("Deviation Logger started. Waiting for anomalies...")
    while True:
        msg = await q.get()
        data = msg.payload
        # Visual alert in logs
        logger.warning(f"🚨 ANOMALY DETECTED: {data.get('metric')} = {data.get('value')} (Scope: {data.get('scope')})")


async def run_ai_loop_with_membus(
    tcp_server: XAppTCPServer,
    target_metric: str = "DRB_PdcpSduDelayDl",
    minirocket_model: Optional[str] = None,
    minirocket_gnb_model: Optional[str] = None,
    minirocket_ue_model: Optional[str] = None,
):
    """Run the simplified loop: xApp -> Membus -> Minirocket -> Log."""
    
    # Create membus
    bus = MemBus()
    tcp_server.bus = bus
    
    # --- AGENT SETUP ---
    minirocket_agents = []
    
    # 1. Determine Correct Metric Names based on CSV columns
    gnb_metric = target_metric
    if gnb_metric == "delay_p95_ms": gnb_metric = "DRB_PdcpSduDelayDl"
    
    # UE Metric Name (Must match XAppKPIAdapter output key)
    ue_metric = "UE_" + gnb_metric if not gnb_metric.startswith("UE_") else gnb_metric
    if "PdcpSduDelayDl" in gnb_metric:
        ue_metric = "UE_DRB_PdcpSduDelayDl_UEID"

    logger.info(f"[SETUP] Metrics mapped: gNB='{gnb_metric}', UE='{ue_metric}'")

    # 2. Create gNB Agent
    gnb_model_path = minirocket_gnb_model or minirocket_model
    minirocket_gnb_agent = MinirocketAgent(
        bus,
        model_path=gnb_model_path,
        metric=gnb_metric,
        window_size=128
    )
    minirocket_agents.append(("gnb", minirocket_gnb_agent))

    # 3. Create UE Agent
    ue_model_path = minirocket_ue_model or minirocket_model
    minirocket_ue_agent = MinirocketAgent(
        bus,
        model_path=ue_model_path,
        metric=ue_metric,
        window_size=128
    )
    minirocket_agents.append(("ue", minirocket_ue_agent))
    
    # --- EXECUTION ---
    logger.info("Starting KPI Collection & Anomaly Detection Loop...")
    
    minirocket_tasks = [
        asyncio.create_task(agent.run(), name=f"minirocket_{name}")
        for name, agent in minirocket_agents
    ]
    
    # Start the Deviation Logger (replaces Reasoner/Actor for visibility)
    logger_task = asyncio.create_task(log_deviations(bus), name="deviation_logger")
    
    # Dashboard (Optional, helps visualize stream)
    try:
        from ain.dashboard.dashboard_server import start_dashboard
        from ain.dashboard.http_server import start_http_server
        dashboard_static = Path(__file__).parent.parent / "dashboard" / "static"
        dashboard_tasks = [
            asyncio.create_task(start_dashboard(bus, port=8081), name="dashboard_websocket"),
            asyncio.create_task(start_http_server(dashboard_static, port=8080), name="dashboard_http"),
        ]
        logger.info("[DASHBOARD] Enabled on http://localhost:8080")
    except ImportError:
        dashboard_tasks = []

    tasks = minirocket_tasks + dashboard_tasks + [logger_task]
    
    try:
        await asyncio.gather(*tasks, return_exceptions=True)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")


async def main():
    parser = argparse.ArgumentParser(description="xApp KPI Collector & Anomaly Detector")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="TCP server host")
    parser.add_argument("--port", type=int, default=6000, help="TCP server port")
    parser.add_argument("--target-metric", type=str, default="DRB_PdcpSduDelayDl", help="Target metric")
    
    # Model paths
    parser.add_argument("--minirocket-model", type=str, default=None)
    parser.add_argument("--minirocket-gnb-model", type=str, default=None)
    parser.add_argument("--minirocket-ue-model", type=str, default=None)
    
    args = parser.parse_args()
    
    # Create TCP server
    tcp_server = XAppTCPServer(host=args.host, port=args.port)
    await tcp_server.start()
    
    try:
        await run_ai_loop_with_membus(
            tcp_server,
            target_metric=args.target_metric,
            minirocket_model=args.minirocket_model,
            minirocket_gnb_model=args.minirocket_gnb_model,
            minirocket_ue_model=args.minirocket_ue_model,
        )
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        await tcp_server.stop()

if __name__ == "__main__":
    asyncio.run(main())