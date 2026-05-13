from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

DATASET_PATH = DATA_DIR / "tickets.csv"
DATASET_PROCESSED_PATH = DATA_DIR / "tickets_procesado.csv"

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

SVM_MODEL_PATH = MODELS_DIR / "svm_model.joblib"
LOGREG_MODEL_PATH = MODELS_DIR / "logreg_model.joblib"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.joblib"
METADATA_PATH = MODELS_DIR / "metadata.joblib"