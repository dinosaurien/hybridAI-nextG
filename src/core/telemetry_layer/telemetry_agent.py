import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from collections import deque

from core.bus.messages import make_msg
from core.common.log_config import should_log, LOG_DEVIATION

logger = logging.getLogger(__name__)

try:
    from core.features.minirocket_rt import MiniRocketRT
    MINIROCKET_AVAILABLE = True
except ImportError:
    MINIROCKET_AVAILABLE = False
    logger.warning("[DEVIATION] MiniRocket not available, using fallback threshold detection")


class DeviationMonitor:
    """State management for a single monitored entity (gNB or UE)."""
    
    def __init__(self, entity_id: str, model_path: str, window_size: int, 
                 metric: str, debounce_seconds: float, min_deviation_count: int):
        self.entity_id = entity_id
        self.metric = metric
        self.debounce_seconds = debounce_seconds
        self.min_deviation_count = min_deviation_count
        
        # Debouncing state
        self.last_deviation_time: Optional[datetime] = None
        self.deviation_buffer: deque = deque(maxlen=min_deviation_count)
        self.last_reported_value: Optional[float] = None
        
        # Feature accumulation (if needed per-entity)
        self.accumulated_kpi: Optional[Dict] = None
        self.kpi_timestamp: float = 0.0
        
        # ML Model
        self.minirocket: Optional[MiniRocketRT] = None
        if MINIROCKET_AVAILABLE:
            try:
                self.minirocket = MiniRocketRT(model_path=model_path, win=window_size)
                logger.log(LOG_DEVIATION, f"[DEVIATION] Loaded MiniRocket model for {entity_id} on metric {metric}")
            except Exception as e:
                logger.warning(f"[DEVIATION] Failed to load model for {entity_id}: {e}")
                self.minirocket = None

    def process_value(self, value: float, kpi_context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a new value and return a deviation event if detected."""
        is_deviation = False
        
        if self.minirocket:
            result = self.minirocket.push(value)
            if result and result.get("pred") == 1:
                is_deviation = True
                if should_log(LOG_DEVIATION):
                     logger.info(f"[DEVIATION] {self.entity_id}: ML detected deviation {self.metric}={value:.2f}")
        else:
            # Fallback threshold
            threshold = 40.0 * 1.5
            if value > threshold:
                is_deviation = True

        # Debouncing logic
        self.deviation_buffer.append(is_deviation)
        now = datetime.now(timezone.utc)
        
        should_report = False
        if is_deviation and len(self.deviation_buffer) >= self.min_deviation_count:
            if all(list(self.deviation_buffer)[-self.min_deviation_count:]):
                if self.last_deviation_time is None:
                    should_report = True
                else:
                    time_since_last = (now - self.last_deviation_time).total_seconds()
                    if time_since_last >= self.debounce_seconds:
                        should_report = True
        
        if should_report:
            self.last_deviation_time = now
            self.last_reported_value = value
            return self._create_deviation_event(value, kpi_context)
            
        return None

    def _create_deviation_event(self, value: float, kpi_context: Dict[str, Any]) -> Dict[str, Any]:
        """Create deviation event dict."""
        direction = "lower_better" if ("delay" in self.metric or "latency" in self.metric) else "higher_better"
        
        severity = "medium"
        # Simple severity based on value (could be smarter)
        if direction == "lower_better":
             if value > 100: severity = "critical"
             elif value > 50: severity = "high"
        
        # State target/baseline explicitly so the LLM gets mathematical context for its OTMs
        # 40.0ms for Latency/Delay, 50,000,000 bps (50 Mbps) for Throughput
        target = 40.0 if direction == "lower_better" else 50000000.0
        baseline = target
        
        scope = {
            "cell_id": kpi_context.get("cell_id", "unknown"),
            "region": "A",
        }
        if self.entity_id != "gnb":
            scope["ue_id"] = self.entity_id
            
        return {
            "source": "minirocket",
            "metric": self.metric,
            "value": value,
            "baseline": baseline,
            "target": target,
            "direction": direction,
            "severity": severity,
            "confidence": 0.9,
            "scope": scope,
            "evidence_ref": f"telemetry://minirocket/{self.entity_id}/{self.metric}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


class TelemetryAgent:
    """Detects deviations using MiniRocket ML model (Multi-Entity)."""
    
    def __init__(self, bus, model_path: str = "models/minirocket.joblib", 
                 window_size: int = 128, metric: str = "delay_p95_ms",
                 debounce_seconds: float = 5.0, min_deviation_count: int = 3):
        
        self.bus = bus
        self.model_path = model_path
        self.window_size = window_size
        self.metric = metric
        self.debounce_seconds = debounce_seconds
        self.min_deviation_count = min_deviation_count
        
        self.monitors: Dict[str, DeviationMonitor] = {}
        self.accumulated_kpis: Dict[str, Dict] = {}
        
        if should_log(LOG_DEVIATION):
            logger.info(f"[DEVIATION] MinirocketAgent initialized for {metric}")
    
    def _get_monitor(self, entity_id: str) -> DeviationMonitor:
        """Get or create monitor for an entity."""
        if entity_id not in self.monitors:
            if should_log(LOG_DEVIATION):
                logger.info(f"[DEVIATION] Creating new monitor for {entity_id}")
            self.monitors[entity_id] = DeviationMonitor(
                entity_id, self.model_path, self.window_size, 
                self.metric, self.debounce_seconds, self.min_deviation_count
            )
        return self.monitors[entity_id]

    async def run(self):
        """Subscribe to KPI stream and detect deviations."""
        q = await self.bus.sub("kpi.raw")
        
        while True:
            msg = await q.get()
            kpi = msg.payload.get("kpi", {})
            if not kpi:
                continue
            
            if not self.metric.startswith("UE_"):
                await self._process_gnb_metrics(kpi)
                
            if self.metric.startswith("UE_"):
                await self._process_ue_metrics(kpi)

    async def _process_gnb_metrics(self, kpi: Dict):
        cell_metrics = kpi.get("CellMetrics", {})
        value = cell_metrics.get(self.metric)
        if value is not None:
            # For gNB, we use "gnb" or cell_id as entity ID key
            monitor = self._get_monitor("gnb")
            deviation = monitor.process_value(float(value), cell_metrics)
            if deviation:
                await self._publish_deviation(deviation)

    async def _process_ue_metrics(self, kpi: Dict):
        ue_metrics_list = kpi.get("UEMetrics", [])
        cell_metrics = kpi.get("CellMetrics", {})
        
        for ue_metric_data in ue_metrics_list:
            # UE ID is needed to scope the monitor
            # XAppKPIAdapter puts it in "ue_id" key usually, or embedded in metric name key?
            # Based on code, XAppKPIAdapter populates "ue_id" field.
            ue_id = ue_metric_data.get("ue_id")
            
            # Also can try to extract from metric key per old adapter logic but explicit key is safer
            if not ue_id: 
                continue
                
            # Value for specific metric
            value = ue_metric_data.get(self.metric)
            if value is not None:
                monitor = self._get_monitor(str(ue_id))
                
                # Context includes cell info
                context = cell_metrics.copy()
                context.update(ue_metric_data)
                
                deviation = monitor.process_value(float(value), context)
                if deviation:
                    await self._publish_deviation(deviation)

    async def _publish_deviation(self, deviation: Dict):
        await self.bus.pub("deviation.detected", make_msg(
            "deviation.detected", "DEVIATION", "deviation.v1", deviation
        ))
        if should_log(LOG_DEVIATION):
            entity = deviation["scope"].get("ue_id", "gNB")
            logger.info(f"[DEVIATION] Published deviation for {entity}: {deviation['metric']}={deviation['value']:.2f}")