# src/config.py

from dataclasses import dataclass

@dataclass(frozen=True)
class DataAssociationConfig:
    GATING_PENALTY_COST = float('inf')
    GATING_THRESHOLD = 10.0
    MAX_ASSIGNMENT_DISTANCE = 5.0


DA = DataAssociationConfig()
