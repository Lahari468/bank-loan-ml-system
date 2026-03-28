import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class DataPreprocessor:

    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.is_fitted = False

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Handling missing values...")

        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if df[col].dtype in ["float64", "int64"]:
                    df[col].fillna(df[col].median(), inplace=True)
                    logger.info(f"Filled numeric column '{col}' with median.")
                else:
                    df[col].fillna(df[col].mode()[0], inplace=True)
                    logger.info(f"Filled categorical column '{col}' with mode.")

        return df

    def _encode_categoricals(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        logger.info("Encoding categorical features...")

        cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

        for col in cat_cols:
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    df[col] = df[col].astype(str).apply(
                        lambda x: x if x in le.classes_ else le.classes_[0]
                    )
                    df[col] = le.transform(df[col])

        return df

    def fit_transform(self, df: pd.DataFrame, target_col: str = "y"):
        logger.info("Starting preprocessing pipeline (fit mode)...")

        df = df.copy()

        initial_rows = len(df)
        df.drop_duplicates(inplace=True)
        logger.info(f"Dropped {initial_rows - len(df)} duplicate rows.")

        df[target_col] = (
            df[target_col].astype(str).str.strip().str.lower()
            .map({"yes": 1, "no": 0})
            .fillna(0)
            .astype(int)
        )

        y = df[target_col].values.astype(int)
        X = df.drop(columns=[target_col])

        X = self._handle_missing_values(X)
        X = self._encode_categoricals(X, fit=True)

        self.feature_columns = X.columns.tolist()
        logger.info(f"Feature columns: {self.feature_columns}")

        X_scaled = self.scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.20, random_state=42, stratify=y
        )

        self.is_fitted = True
        logger.info(f"Preprocessing complete. Train: {X_train.shape}, Test: {X_test.shape}")

        return X_train, X_test, y_train, y_test

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fit before calling transform().")

        df = df.copy()

        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0

        df = df[self.feature_columns]

        df = self._handle_missing_values(df)
        df = self._encode_categoricals(df, fit=False)

        X_scaled = self.scaler.transform(df)

        return X_scaled

    def save(self, path: str = "model/preprocessor.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"Preprocessor saved to {path}")

    @staticmethod
    def load(path: str = "model/preprocessor.pkl") -> "DataPreprocessor":
        preprocessor = joblib.load(path)
        logger.info(f"Preprocessor loaded from {path}")
        return preprocessor