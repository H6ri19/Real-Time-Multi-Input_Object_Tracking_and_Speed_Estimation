import os

class Config:
    # Paths
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    MODEL_PATH = os.path.join(BASE_DIR, "models", "yolov8n.pt")
    OUTPUT_DIR = os.path.join(BASE_DIR, "data", "results")

    # Detection/tracking settings
    MIN_CONF = 0.35
    MAX_AGE = 30
    IOU_THRESHOLD = 0.3

    # Security
    LOG_FILE = os.path.join(BASE_DIR, "logs", "secure_logs.enc")
    ENCRYPTION_KEY_PATH = os.path.join(BASE_DIR, "security", "encryption", "key.key")

    # Speed estimation
    PIXEL_TO_METER = 0.05  # 1 pixel = 0.05 meters, adjust according to your camera setup
