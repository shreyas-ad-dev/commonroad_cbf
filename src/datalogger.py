# datalogger.py
import json
import numpy as np
from pathlib import Path

class SimulationJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return super().default(obj)

class DataLogger:
    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.file = open(self.output_path, "w")

    def log_step(self, step: int, timestamp: float, payload: dict):
        step_entry = {
            "step": step,
            "timestamp": timestamp,
            "data": payload 
        }
        self.file.write(json.dumps(step_entry, cls=SimulationJSONEncoder) + "\n")

    def close(self):
        self.file.close()
