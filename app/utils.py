import logging
from typing import Tuple, Dict, Any

import pandas as pd

logger = logging.getLogger(__name__)
REQUIRED_FIELDS: Dict[str, type] = {
    "age": int,
    "job": str,
    "marital": str,
    "education": str,
    "default": str,
    "balance": (int, float),
    "housing": str,
    "loan": str,
    "contact": str,
    "day": int,
    "month": str,
    "duration": int,
    "campaign": int,
    "pdays": int,
    "previous": int,
    "poutcome": str,
}

VALID_VALUES: Dict[str, list] = {
    "job": [
        "admin.", "blue-collar", "entrepreneur", "housemaid",
        "management", "retired", "self-employed", "services",
        "student", "technician", "unemployed", "unknown",
    ],
    "marital": ["divorced", "married", "single"],
    "education": ["primary", "secondary", "tertiary", "unknown"],
    "default": ["no", "yes"],
    "housing": ["no", "yes"],
    "loan": ["no", "yes"],
    "contact": ["cellular", "telephone", "unknown"],
    "month": [
        "jan", "feb", "mar", "apr", "may", "jun",
        "jul", "aug", "sep", "oct", "nov", "dec",
    ],
    "poutcome": ["failure", "other", "success", "unknown"],
}


def validate_input(data: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate the incoming JSON payload.

    Returns
    -------
    (is_valid: bool, error_message: str)
    """
    if not isinstance(data, dict):
        return False, "Request body must be a JSON object."

    # Check all required fields are present
    for field in REQUIRED_FIELDS:
        if field not in data:
            return False, f"Missing required field: '{field}'"

    # Type checks
    for field, expected_type in REQUIRED_FIELDS.items():
        value = data[field]
        if isinstance(expected_type, tuple):
            if not isinstance(value, expected_type):
                return False, (
                    f"Field '{field}' must be numeric, got {type(value).__name__}."
                )
        else:
            if not isinstance(value, expected_type):
                return False, (
                    f"Field '{field}' must be {expected_type.__name__}, "
                    f"got {type(value).__name__}."
                )

    # Value-range / allowed-value checks
    for field, allowed in VALID_VALUES.items():
        value = str(data[field]).lower()
        if value not in allowed:
            return False, (
                f"Invalid value for '{field}': '{data[field]}'. "
                f"Allowed: {allowed}"
            )

    if not (18 <= data["age"] <= 100):
        return False, "Field 'age' must be between 18 and 100."

    if data["campaign"] < 1:
        return False, "Field 'campaign' must be >= 1."

    return True, ""


def parse_input_to_dataframe(data: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert the validated JSON payload into a single-row DataFrame
    that matches the training data schema.
    """
    # Normalize string fields to lowercase (matches training encoding)
    normalized = {}
    for key, value in data.items():
        normalized[key] = str(value).lower() if isinstance(value, str) else value

    df = pd.DataFrame([normalized])
    logger.debug(f"Parsed input DataFrame:\n{df}")
    return df


def build_response(
    prediction: int,
    probability: float,
    input_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a structured, human-readable API response.

    Parameters
    ----------
    prediction  : 0 or 1
    probability : float probability of class 1 (loan taker)
    input_data  : original request payload (echoed back for traceability)
    """
    label = "Yes - Likely to take loan" if prediction == 1 else "No - Unlikely to take loan"
    risk_band = _risk_band(probability)

    return {
        "prediction": prediction,
        "prediction_label": label,
        "probability_of_loan": round(float(probability), 4),
        "risk_band": risk_band,
        "input_received": input_data,
        "model": "RandomForestClassifier",
        "version": "1.0.0",
    }


def _risk_band(prob: float) -> str:
    """Classify probability into a human-readable risk band."""
    if prob >= 0.75:
        return "HIGH"
    elif prob >= 0.50:
        return "MEDIUM"
    elif prob >= 0.25:
        return "LOW"
    else:
        return "VERY LOW"
