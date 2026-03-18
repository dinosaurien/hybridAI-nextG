import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, List

@dataclass
class NetworkContext:
    """Rich network context for contextual bandit."""
    # Core RAN metrics
    latency_ms: float
    throughput_dl_mbps: float
    throughput_ul_mbps: float
    bler: float
    cqi: float
    mcs: int
    ue_count: int
    prb_usage: float
    sinr_db: float
    
    # Metadata
    cell_id: str
    timestamp: str
    
    # Derived features
    network_load: float = 0.0  # Computed from PRB usage + UE count
    quality_index: float = 0.0  # Computed from CQI + SINR
    
    def __post_init__(self):
        """Calculate derived features."""
        self.network_load = min(1.0, (self.prb_usage + self.ue_count/100) / 2)
        self.quality_index = min(1.0, (self.cqi/15 + (self.sinr_db + 10)/40) / 2)

class ContextExtractor:
    """Extract structured context from KPI data."""
    
    def __init__(self, window_size: int = 12, feature_count: int = 10):
        self.window_size = window_size
        self.feature_count = feature_count
        
        # Feature normalization bounds
        self.bounds = {
            'latency_ms': (10, 100),
            'throughput_dl_mbps': (0, 1000), 
            'throughput_ul_mbps': (0, 200),
            'bler': (0.0, 0.1),
            'cqi': (0, 15),
            'mcs': (0, 28),
            'ue_count': (0, 100),
            'prb_usage': (0.0, 1.0),
            'sinr_db': (-10, 30),
            'network_load': (0.0, 1.0)
        }
    
    def extract_from_kpi(self, kpi_data: Dict[str, Any]) -> NetworkContext:
        """Extract context from your fake_kpi format."""
        try:
            # Handle your current fake_kpi structure
            if 'CellMetrics' in kpi_data:
                cell_metrics = kpi_data['CellMetrics']
                cell_id = cell_metrics.get('cell_id', 'CELL_001')
                
                context = NetworkContext(
                    latency_ms=cell_metrics.get('delay_p95_ms', 50.0),
                    throughput_dl_mbps=cell_metrics.get('thr_dl_bps', 50000000) / 1e6,
                    throughput_ul_mbps=cell_metrics.get('thr_ul_bps', 10000000) / 1e6,
                    bler=cell_metrics.get('bler_dl', 0.01),
                    cqi=cell_metrics.get('cqi_avg', 10.0),
                    mcs=int(cell_metrics.get('mcs_avg', 15)),
                    ue_count=cell_metrics.get('active_ue_count', 20),
                    prb_usage=cell_metrics.get('prb_used_dl', 50) / 100,
                    sinr_db=cell_metrics.get('sinr_avg', 15.0),
                    cell_id=cell_id,
                    timestamp=kpi_data.get('timestamp', '')
                )
                
                return context
            
            # Fallback for other formats
            return self._create_default_context()
            
        except Exception as e:
            print(f"Context extraction error: {e}")
            return self._create_default_context()
    
    def _create_default_context(self) -> NetworkContext:
        """Create default context when extraction fails."""
        return NetworkContext(
            latency_ms=50.0, throughput_dl_mbps=100.0, throughput_ul_mbps=20.0,
            bler=0.01, cqi=10.0, mcs=15, ue_count=20, prb_usage=0.5,
            sinr_db=15.0, cell_id='CELL_001', timestamp=''
        )
    
    def to_feature_vector(self, context: NetworkContext) -> np.ndarray:
        """Convert context to normalized feature vector."""
        features = []
        
        # Extract feature values
        raw_features = {
            'latency_ms': context.latency_ms,
            'throughput_dl_mbps': context.throughput_dl_mbps,
            'throughput_ul_mbps': context.throughput_ul_mbps,
            'bler': context.bler,
            'cqi': context.cqi,
            'mcs': context.mcs,
            'ue_count': context.ue_count,
            'prb_usage': context.prb_usage,
            'sinr_db': context.sinr_db,
            'network_load': context.network_load
        }
        
        # Normalize each feature to [0, 1]
        for feature_name in list(raw_features.keys())[:self.feature_count]:
            value = raw_features[feature_name]
            min_val, max_val = self.bounds[feature_name]
            normalized = (value - min_val) / (max_val - min_val)
            normalized = np.clip(normalized, 0.0, 1.0)
            features.append(normalized)
        
        return np.array(features, dtype=np.float32)
    
    def to_state_tensor(self, context: NetworkContext) -> np.ndarray:
        """Convert to [W, F] state tensor format."""
        feature_vector = self.to_feature_vector(context)
        
        # For now, repeat current context across window
        # TODO: Implement proper temporal window
        state_tensor = np.tile(feature_vector, (self.window_size, 1))
        
        return state_tensor  # Shape: [W=12, F=10]