from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "model.joblib"
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(title="Hackathon ML Predictor", version="1.0.0")

# Load once when the server starts. Do NOT reload the model for every prediction.
if not MODEL_PATH.exists():
    raise RuntimeError(f"Model file not found: {MODEL_PATH}")

bundle = joblib.load(MODEL_PATH)

if not isinstance(bundle, dict) or "model" not in bundle:
    raise RuntimeError(
        "model.joblib must contain a dictionary with at least a 'model' key."
    )

model = bundle["model"]
feature_columns = bundle.get("feature_columns", [])
feature_metadata = bundle.get("feature_metadata", {})
problem_type = bundle.get("problem_type", "unknown")
target = bundle.get("target", "prediction")
metrics = bundle.get("test_metrics", {})
training_rows = bundle.get("training_rows")
sklearn_version = bundle.get("sklearn_version")

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def home():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/config")
def get_config():
    """Frontend uses this to dynamically construct the input form."""
    return {
        "feature_columns": feature_columns,
        "feature_metadata": feature_metadata,
        "problem_type": problem_type,
        "target": target,
        "test_metrics": metrics,
        "training_rows": training_rows,
        "sklearn_version": sklearn_version,
    }


def convert_value(name: str, value, metadata: dict):
    """Convert JSON values into types expected by the model."""
    field_type = metadata.get("type", "string")

    if value is None or value == "":
        if metadata.get("required", True):
            raise ValueError(f"{name} is required.")
        return None

    try:
        if field_type == "numeric":
            return float(value)

        if field_type == "integer":
            return int(value)

        # Dates are intentionally kept as strings here. If your trained
        # pipeline expects datetime objects, change this to pd.to_datetime(value).
        if field_type == "date":
            return str(value)

        if field_type == "boolean":
            if isinstance(value, bool):
                return value
            return str(value).lower() in {"true", "1", "yes", "on"}

        return str(value)

    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid value for '{name}'.") from exc


@app.post("/api/predict")
def predict(payload: dict):
    try:
        if not feature_columns:
            raise ValueError("No feature_columns were provided by the model bundle.")

        row = {}

        for feature in feature_columns:
            metadata = feature_metadata.get(feature, {})
            if feature not in payload:
                if metadata.get("required", True):
                    raise ValueError(f"Missing feature: {feature}")
                row[feature] = None
            else:
                row[feature] = convert_value(
                    feature, payload[feature], metadata
                )

        # DataFrame preserves feature names/order and works well with
        # sklearn Pipeline / ColumnTransformer objects.
        X = pd.DataFrame([row], columns=feature_columns)

        prediction = model.predict(X)

        result = prediction[0]

        # Convert numpy scalar types into normal JSON-compatible Python types.
        if hasattr(result, "item"):
            result = result.item()

        response = {
            "prediction": result,
            "target": target,
            "problem_type": problem_type,
        }

        # Some classifiers expose probabilities.
        if problem_type == "classification" and hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X)[0]
            response["probabilities"] = [
                float(x) for x in probabilities
            ]

            if hasattr(model, "classes_"):
                response["classes"] = [
                    c.item() if hasattr(c, "item") else c
                    for c in model.classes_
                ]

        return response

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        # Keep implementation details out of the client response.
        print(f"Prediction error: {exc}")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check the server console for details."
        )


@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": True}
