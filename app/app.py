import os
import sys
import time
import logging
from flask import Flask, request, jsonify

# Ensure sibling modules are importable regardless of CWD
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_loader import load_artifacts, get_model, get_preprocessor
from utils import validate_input, parse_input_to_dataframe, build_response

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False  

try:
    load_artifacts()
    logger.info("✅ ML artifacts loaded successfully.")
except FileNotFoundError as e:
    logger.warning(f"⚠️  {e}")
    logger.warning("   The /predict endpoint will return 503 until artifacts exist.")

@app.route("/health", methods=["GET"])
def health():
    """
    Liveness / readiness probe used by Kubernetes.
    Returns 200 if the model is loaded, 503 otherwise.
    """
    try:
        get_model()         
        get_preprocessor()
        return jsonify({"status": "healthy", "model_loaded": True}), 200
    except RuntimeError:
        return jsonify({"status": "unhealthy", "model_loaded": False}), 503


@app.route("/info", methods=["GET"])
def info():
    """Return basic metadata about the deployed model."""
    return jsonify(
        {
            "service": "Bank Loan Prediction API",
            "version": "1.0.0",
            "model": "RandomForestClassifier",
            "description": (
                "Predicts whether a bank customer is likely to take a loan "
                "based on demographic and financial features."
            ),
            "endpoints": {
                "GET /health": "Liveness check",
                "GET /info": "Service metadata",
                "POST /predict": "Run loan-take prediction",
            },
        }
    ), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accept a JSON payload describing a bank customer and return
    a prediction (0/1) with probability and risk band.

    Sample request body:
    {
        "age": 35,
        "job": "management",
        "marital": "married",
        "education": "tertiary",
        "default": "no",
        "balance": 3000,
        "housing": "yes",
        "loan": "no",
        "contact": "cellular",
        "day": 15,
        "month": "may",
        "duration": 300,
        "campaign": 1,
        "pdays": -1,
        "previous": 0,
        "poutcome": "unknown"
    }
    """
    start_time = time.time()
    logger.info(f"POST /predict received from {request.remote_addr}")

    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 415

    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Invalid or empty JSON body"}), 400

    is_valid, error_msg = validate_input(data)
    if not is_valid:
        logger.warning(f"Validation failed: {error_msg}")
        return jsonify({"error": error_msg}), 422

    try:
        model = get_model()
        preprocessor = get_preprocessor()
    except RuntimeError as e:
        logger.error(str(e))
        return jsonify({"error": "Model not available. Please try again later."}), 503

    try:
        input_df = parse_input_to_dataframe(data)
        X = preprocessor.transform(input_df)
    except Exception as e:
        logger.exception("Preprocessing error")
        return jsonify({"error": f"Preprocessing failed: {str(e)}"}), 500

    try:
        prediction = int(model.predict(X)[0])
        probability = float(model.predict_proba(X)[0][1])   # P(class=1)
    except Exception as e:
        logger.exception("Prediction error")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    elapsed_ms = round((time.time() - start_time) * 1000, 2)
    response = build_response(prediction, probability, data)
    response["latency_ms"] = elapsed_ms

    logger.info(
        f"Prediction={prediction} | Probability={probability:.4f} | "
        f"Latency={elapsed_ms}ms"
    )

    return jsonify(response), 200


# ==================================================================
# Error handlers
# ==================================================================

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed"}), 405


@app.errorhandler(500)
def internal_error(e):
    logger.exception("Unhandled internal error")
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"

    logger.info(f"🚀 Starting Bank Loan Prediction API on port {port} ...")
    app.run(host="0.0.0.0", port=port, debug=debug)
