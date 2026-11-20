"""MR-Sleuthing: reference-free detection and localization of deep-learning MRI reconstruction."""
from .detector import Detection, HallucinationDetector, load_model, save_overlay

__all__ = ["HallucinationDetector", "Detection", "load_model", "save_overlay"]
__version__ = "0.1.0"
