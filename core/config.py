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
SUPPORTED_CLASSES = {"person", "helmet", "hardhat", "vest", "safety_vest", "machinery", "forklift", "truck", "excavator"}
DETECTION_CONFIDENCE = float(os.getenv("WPS_DET_CONF", "0.45"))
DETECTION_IOU = float(os.getenv("WPS_DET_IOU", "0.45"))

# Rule engine and association thresholds
PPE_ASSOCIATION_IOU = float(os.getenv("WPS_PPE_IOU", "0.10"))
PPE_CONTAINMENT_RATIO = float(os.getenv("WPS_PPE_CONTAIN", "0.30"))
MACHINERY_DANGER_DISTANCE_PX = float(os.getenv("WPS_MACHINE_DISTANCE", "180"))

# Confidence thresholds for enhanced rules
HELMET_CONFIDENCE_PASS = float(os.getenv("WPS_HELMET_CONF_PASS", "0.80"))
HELMET_CONFIDENCE_WARN = float(os.getenv("WPS_HELMET_CONF_WARN", "0.60"))
VEST_CONFIDENCE_PASS = float(os.getenv("WPS_VEST_CONF_PASS", "0.80"))
VEST_CONFIDENCE_WARN = float(os.getenv("WPS_VEST_CONF_WARN", "0.60"))

# Safety scoring configuration
SCORE_HELMET = 40
SCORE_VEST = 30
SCORE_SAFE_DISTANCE = 30
SCORE_PENALTY_WARNING = 15
SCORE_PENALTY_DANGER = 30

# Alerting configuration
ENABLE_SOUND_ALERTS = os.getenv("WPS_SOUND_ALERTS", "0") == "1"

# Streamlit runtime
DEFAULT_STREAM_SECONDS = int(os.getenv("WPS_STREAM_SECONDS", "15"))
DEFAULT_MULTI_CAMERA_COUNT = int(os.getenv("WPS_MULTI_CAM_COUNT", "2"))
EVENTS_CSV = RESULTS_DIR / "violations_log.csv"
SUMMARY_CSV = RESULTS_DIR / "analytics_summary.csv"