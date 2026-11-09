"""Factory anomaly detector: synthetic CNC / robot-cell sensor data,
an Isolation Forest and a NumPy autoencoder, compared on labelled data.
"""

from .generator import generate_dataset, generate_stream, FEATURE_NAMES, FAULT_TYPES
from .detectors import IsolationForestDetector, AutoencoderDetector
from .evaluate import precision_recall_f1, per_fault_recall

__all__ = [
    "generate_dataset", "generate_stream", "FEATURE_NAMES", "FAULT_TYPES",
    "IsolationForestDetector", "AutoencoderDetector",
    "precision_recall_f1", "per_fault_recall",
]
