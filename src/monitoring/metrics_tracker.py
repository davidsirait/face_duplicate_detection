"""Simple metrics tracking for performance monitoring"""

import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional


class MetricsTracker:
    """Track application performance metrics"""
    
    def __init__(self, metrics_file: str = "metrics.jsonl"):
        self.metrics_file = Path(metrics_file)
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
    
    def log_metric(self, metric_name: str, value: float, metadata: Optional[Dict[str, Any]] = None):
        """Log a single metric to file"""
        try:
            data = {
                "timestamp": datetime.now().isoformat(),
                "metric": metric_name,
                "value": value
            }
            if metadata:
                data.update(metadata)
            
            with open(self.metrics_file, "a") as f:
                f.write(json.dumps(data) + "\n")
        except Exception:
            pass  # Silent fail to not break app
    
    def track_latency(self, operation_name: str):
        """Context manager to track operation latency"""
        return _LatencyTracker(self, operation_name)


class _LatencyTracker:
    """Context manager for tracking latency"""
    
    def __init__(self, tracker: MetricsTracker, operation_name: str):
        self.tracker = tracker
        self.operation_name = operation_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        latency_ms = (time.time() - self.start_time) * 1000
        status = "error" if exc_type else "success"
        self.tracker.log_metric(
            f"{self.operation_name}_latency",
            round(latency_ms, 2),
            {"status": status}
        )
