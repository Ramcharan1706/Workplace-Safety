"""Application-wide configuration values."""

from pathlib import Path
import os


BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

IMG_SIZE = (64, 64)
CNN_IMAGE_SIZE = (160, 160)

SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")

FEATURE_VERSION = 1
RANDOM_STATE = 77
TEST_SIZE = 0.2

DECISION_TREE_MAX_DEPTH = 10
CNN_BATCH_SIZE = 16
CNN_EPOCHS = 5

CACHE_X_FILE = MODELS_DIR / "X.npy"
CACHE_Y_FILE = MODELS_DIR / "Y.npy"
CACHE_META_FILE = MODELS_DIR / "cache_meta.json"

CLASSICAL_MODEL_FILES = {
    "SVC": MODELS_DIR / "SVC.joblib",
    "DecisionTree": MODELS_DIR / "DecisionTree.joblib",
    "NearestCentroid": MODELS_DIR / "NearestCentroid.joblib",
    "GaussianProcess": MODELS_DIR / "GaussianProcess.joblib",
}

MODEL_META_SUFFIX = ".meta.json"
CNN_MODEL_FILE = MODELS_DIR / "cnn_mobilenetv2.keras"
CNN_META_FILE = MODELS_DIR / "cnn_mobilenetv2.meta.json"

# Real-time monitoring runtime settings
DEFAULT_MODEL_PATH = os.getenv("WPS_YOLO_MODEL", "yolov8n.pt")

# Comprehensive support for all helmet and vest colors + machinery types
# This includes both custom-trained classes and common PPE model class names
SUPPORTED_CLASSES = {
    # Workers
    "person", "worker",
    # Helmets - all common colors and naming conventions
    "helmet", "hardhat", "hard_hat", "hard hat",
    "yellow_helmet", "white_helmet", "red_helmet", "orange_helmet", 
    "blue_helmet", "pink_helmet", "green_helmet", "black_helmet", "grey_helmet",
    "yellow_hardhat", "white_hardhat", "red_hardhat", "orange_hardhat",
    "blue_hardhat", "green_hardhat", "black_hardhat",
    # Safety Vests - all common colors and naming conventions
    "vest", "safety_vest", "safety vest", "safety_jacket",
    "orange_vest", "yellow_vest", "white_vest", "red_vest", "green_vest",
    "orange_safety_vest", "yellow_safety_vest", "white_safety_vest",
    "reflective_vest", "high_visibility_vest", "high_visibility_jacket",
    "safety_jacket", "reflective_jacket", "hi_vis", "hi-vis",
    # Heavy Machinery
    "machinery", "machine", "equipment",
    "forklift", "truck", "excavator", "vehicle", "car",
    "bulldozer", "loader", "crane", "boom_lift", "boom lift",
    "scissor_lift", "cherry_picker", "man_lift", "pallet_jack",
    "conveyor", "drill_press", "welding_machine", "circular_saw",
    "scaffolding", "ladder", "guardrail", "safety_barrier", "safety barrier",
    # Common PPE detection model class names
    "ppl", "people", "hardhat", "no_hardhat",
    "safety_helmet", "without_helmet", "with_helmet",
    "PPE", "no_PPE", "proper_PPE", "improper_PPE",
}
def _safe_float(key: str, default: str, min_val: float | None = None, max_val: float | None = None) -> float:
    """Safely parse environment variable as float with bounds checking."""
    try:
        value = float(os.getenv(key, default))
        if min_val is not None and value < min_val:
            raise ValueError(f"{key}={value} below minimum {min_val}")
        if max_val is not None and value > max_val:
            raise ValueError(f"{key}={value} exceeds maximum {max_val}")
        return value
    except ValueError as e:
        import logging
        logging.warning(f"Invalid env var {key}: {e}, using default {default}")
        return float(default)

def _safe_int(key: str, default: str, min_val: int | None = None) -> int:
    """Safely parse environment variable as int with bounds checking."""
    try:
        value = int(os.getenv(key, default))
        if min_val is not None and value < min_val:
            raise ValueError(f"{key}={value} below minimum {min_val}")
        return value
    except ValueError as e:
        import logging
        logging.warning(f"Invalid env var {key}: {e}, using default {default}")
        return int(default)

DETECTION_CONFIDENCE = _safe_float("WPS_DET_CONF", "0.45", min_val=0.0, max_val=1.0)
DETECTION_IOU = _safe_float("WPS_DET_IOU", "0.45", min_val=0.0, max_val=1.0)

# Rule engine and association thresholds
PPE_ASSOCIATION_IOU = _safe_float("WPS_PPE_IOU", "0.10", min_val=0.0, max_val=1.0)
PPE_CONTAINMENT_RATIO = _safe_float("WPS_PPE_CONTAIN", "0.30", min_val=0.0, max_val=1.0)
MACHINERY_DANGER_DISTANCE_PX = _safe_float("WPS_MACHINE_DISTANCE", "180", min_val=0.0)

# Confidence thresholds for enhanced rules
HELMET_CONFIDENCE_PASS = _safe_float("WPS_HELMET_CONF_PASS", "0.80", min_val=0.0, max_val=1.0)
HELMET_CONFIDENCE_WARN = _safe_float("WPS_HELMET_CONF_WARN", "0.60", min_val=0.0, max_val=1.0)
VEST_CONFIDENCE_PASS = _safe_float("WPS_VEST_CONF_PASS", "0.80", min_val=0.0, max_val=1.0)
VEST_CONFIDENCE_WARN = _safe_float("WPS_VEST_CONF_WARN", "0.60", min_val=0.0, max_val=1.0)

# Safety scoring configuration
SCORE_HELMET = 40
SCORE_VEST = 30
SCORE_SAFE_DISTANCE = 30
SCORE_PENALTY_WARNING = 15
SCORE_PENALTY_DANGER = 30

# Alerting configuration
ENABLE_SOUND_ALERTS = os.getenv("WPS_SOUND_ALERTS", "0") == "1"

# Streamlit runtime
DEFAULT_STREAM_SECONDS = _safe_int("WPS_STREAM_SECONDS", "15", min_val=1)
DEFAULT_MULTI_CAMERA_COUNT = _safe_int("WPS_MULTI_CAM_COUNT", "2", min_val=1)
EVENTS_CSV = RESULTS_DIR / "violations_log.csv"
SUMMARY_CSV = RESULTS_DIR / "analytics_summary.csv"