from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT: Path = Path(__file__).parent.parent

# Train dataset folders
DATA_FOLDER: Path = PROJECT_ROOT / "data" / "FaceForensics++_C23"
CSV_FOLDER: Path = DATA_FOLDER / "csv"
FRAME_DIR: Path = DATA_FOLDER / "frames"
FACES_DIR: Path = DATA_FOLDER / "faces"

# Test dataset folders
TEST_DATA_FOLDER: Path = PROJECT_ROOT / "data" / "FaceForensics++_C23_test"
TEST_CSV_FOLDER: Path = TEST_DATA_FOLDER / "csv"
TEST_FRAME_DIR: Path = TEST_DATA_FOLDER / "frames"
TEST_FACES_DIR: Path = TEST_DATA_FOLDER / "faces"

# Output folders
OUTPUT_DIR: Path = PROJECT_ROOT / "output"
MODEL_DIR: Path = OUTPUT_DIR / "models"
LOGS_DIR: Path = PROJECT_ROOT / "logs"

# Logger configuration
LOG_FORMAT: str = "%(asctime)s - %(name)s.%(funcName)s - %(levelname)s - %(message)s - %(lineno)d"
LOG_LEVEL: int = 10  # DEBUG level
LOG_FILE: Path = LOGS_DIR / "app.log"
LOG_DATEFMT: str = "%Y-%m-%d %H:%M:%S"

# Face detection model
FACE_DETECTION_MODEL_DIR: Path = PROJECT_ROOT / "models" / "face_detection"
FACE_DETECTION_MODEL_PATH: Path = FACE_DETECTION_MODEL_DIR / "model.pt"
FACE_DETECTION_HF_REPO_ID: str = "AdamCodd/YOLOv11n-face-detection"
FACE_DETECTION_MODEL_FILENAME: str = "model.pt"
FACE_DETECTION_HF_URL: str = f"https://huggingface.co/{FACE_DETECTION_HF_REPO_ID}/resolve/main/{FACE_DETECTION_MODEL_FILENAME}"

# Deepfake detection model
DEEPFAKE_DETECTION_MODEL_DIR: Path = PROJECT_ROOT / "models" / "deepfake_detection"
DEEPFAKE_DETECTION_MODEL_PATH: Path = DEEPFAKE_DETECTION_MODEL_DIR / "veridisquo_25M.pth"

# Preprocessed dataset folders (balanced and split)
PREPROCESSED_DATASET_DIR: Path = PROJECT_ROOT / "data"

# Published models on Hugging Face fetched for inference engine
HF_REPO_ID: str = "Gazeux33/VeridisQuo"
MODEL_FILENAME: str = "veridisquo_25M.pth"
HF_MODEL_URL: str = f"https://huggingface.co/{HF_REPO_ID}/resolve/main/{MODEL_FILENAME}"

# Preprocessing parameters
IMAGE_SIZE: Tuple[int, int] = (224, 224)
MEAN: List[float] = [0.485, 0.456, 0.406]  # ImageNet mean
STD: List[float] = [0.229, 0.224, 0.225]   # ImageNet std
MIN_FACE_SIZE: int = 40  # Minimum face size for detection
FACE_DETECT_CONFIDENCE_THRESHOLD: float = 0.7  # Confidence threshold for face detection
ONLY_KEEP_TOP_FACE: bool = True  # Whether to keep only the top detected face
FACE_EXTRACT_PADDING: int = 0  # Padding around detected face bounding box
