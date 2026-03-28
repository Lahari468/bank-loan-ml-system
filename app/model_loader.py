import os
import sys
import logging
import joblib

_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model")
if _MODEL_DIR not in sys.path:
    sys.path.insert(0, _MODEL_DIR)

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "model", "preprocessor.pkl")

_model = None
_preprocessor = None


def load_artifacts():
    """
    Load model and preprocessor from disk into module-level variables.
    Called once at Flask app startup.
    """
    global _model, _preprocessor

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at: {MODEL_PATH}\n"
            "Run `python model/train.py` to generate it."
        )
    if not os.path.exists(PREPROCESSOR_PATH):
        raise FileNotFoundError(
            f"Preprocessor file not found at: {PREPROCESSOR_PATH}\n"
            "Run `python model/train.py` to generate it."
        )

    logger.info(f"Loading model from: {MODEL_PATH}")
    _model = joblib.load(MODEL_PATH)
    logger.info("Model loaded successfully.")

    logger.info(f"Loading preprocessor from: {PREPROCESSOR_PATH}")
    _preprocessor = joblib.load(PREPROCESSOR_PATH)
    logger.info("Preprocessor loaded successfully.")


def get_model():
    """Return the loaded model. Raises if not yet loaded."""
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_artifacts() first.")
    return _model


def get_preprocessor():
    """Return the loaded preprocessor. Raises if not yet loaded."""
    if _preprocessor is None:
        raise RuntimeError("Preprocessor not loaded. Call load_artifacts() first.")
    return _preprocessor
