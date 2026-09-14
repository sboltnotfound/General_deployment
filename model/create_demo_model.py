"""
Run this once to create a demo model.joblib.

Replace this file with your hackathon's actual training code later.
The important part is that the saved bundle follows the same structure.
"""

from pathlib import Path
import joblib
import sklearn
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor

MODEL_DIR = Path(__file__).resolve().parent

# Tiny artificial dataset just to prove the application works.
df = pd.DataFrame({
    "age": [20, 22, 25, 30, 35, 40, 45, 50, 28, 32],
    "salary": [30000, 35000, 42000, 50000, 60000, 75000, 85000, 95000, 48000, 55000],
    "experience": [0, 1, 2, 5, 8, 12, 15, 20, 4, 6],
    "city": ["Delhi", "Mumbai", "Delhi", "Bangalore", "Mumbai",
             "Bangalore", "Delhi", "Mumbai", "Bangalore", "Delhi"],
    "target_salary": [32000, 38000, 45000, 56000, 68000, 82000, 93000, 105000, 53000, 62000]
})

feature_columns = ["age", "salary", "experience", "city"]
target = "target_salary"

X = df[feature_columns]
y = df[target]

numeric_features = ["age", "salary", "experience"]
categorical_features = ["city"]

preprocessor = ColumnTransformer([
    ("numeric", StandardScaler(), numeric_features),
    ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ))
])

pipeline.fit(X, y)

model_bundle = {
    "model": pipeline,

    "feature_columns": feature_columns,

    "feature_metadata": {
        "age": {
            "type": "integer",
            "required": True,
            "min": 0,
            "max": 120
        },
        "salary": {
            "type": "numeric",
            "required": True,
            "min": 0
        },
        "experience": {
            "type": "numeric",
            "required": True,
            "min": 0
        },
        "city": {
            "type": "string",
            "required": True,
            "options": ["Delhi", "Mumbai", "Bangalore"]
        }
    },

    "target": target,
    "problem_type": "regression",
    "training_rows": len(X),
    "sklearn_version": sklearn.__version__,

    "test_metrics": {
        "r2": 0.0,
        "mae": 0.0,
        "rmse": 0.0
    }
}

output_path = MODEL_DIR / "model.joblib"
joblib.dump(model_bundle, output_path)

print(f"Saved demo model to: {output_path}")
