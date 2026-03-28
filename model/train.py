import os
import sys
import logging
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import DataPreprocessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.pkl")


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    y_pred = model.predict(X_test)
    y_prob = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else None
    )

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    logger.info("=" * 55)
    logger.info("MODEL EVALUATION REPORT")
    logger.info("=" * 55)
    logger.info(f"Accuracy  : {acc:.4f}")
    logger.info(f"Precision : {prec:.4f}")
    logger.info(f"Recall    : {rec:.4f}")
    logger.info(f"F1 Score  : {f1:.4f}")
    logger.info("Confusion Matrix:")
    logger.info(f"TN={cm[0][0]}  FP={cm[0][1]}")
    logger.info(f"FN={cm[1][0]}  TP={cm[1][1]}")
    logger.info("Classification Report:")
    logger.info(
        "\n" + classification_report(y_test, y_pred, target_names=["No Loan", "Loan"])
    )
    logger.info("=" * 55)

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def train():
    logger.info("Loading dataset from: %s", DATA_PATH)

    df = pd.read_csv(DATA_PATH)
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Target distribution:\n{df['y'].value_counts()}")

    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.fit_transform(df, target_col="y")

    logger.info("Training Random Forest classifier...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf_model.fit(X_train, y_train)
    logger.info("Random Forest training complete.")

    logger.info("Training Logistic Regression...")
    lr_model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )
    lr_model.fit(X_train, y_train)

    logger.info("--- Random Forest ---")
    rf_metrics = evaluate_model(rf_model, X_test, y_test)

    logger.info("--- Logistic Regression ---")
    lr_metrics = evaluate_model(lr_model, X_test, y_test)

    if rf_metrics["f1"] >= lr_metrics["f1"]:
        best_model = rf_model
        logger.info("Random Forest selected as best model.")
    else:
        best_model = lr_model
        logger.info("Logistic Regression selected as best model.")

    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
        feat_imp = sorted(
            zip(preprocessor.feature_columns, importances),
            key=lambda x: x[1],
            reverse=True,
        )
        logger.info("Top-10 Feature Importances:")
        for feat, imp in feat_imp[:10]:
            logger.info(f"{feat:<20} {imp:.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    preprocessor.save(PREPROCESSOR_PATH)

    logger.info(f"Model saved to: {MODEL_PATH}")
    logger.info(f"Preprocessor saved to: {PREPROCESSOR_PATH}")
    logger.info("Training pipeline complete!")

    return best_model, preprocessor


if __name__ == "__main__":
    train()