# Generic Hackathon ML Predictor

A reusable FastAPI + plain HTML/CSS/JS prediction app.

The application loads one `model/model.joblib` file and dynamically creates
the frontend form from `feature_metadata`.

## 1. Install

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2. Create the demo model

```powershell
python model/create_demo_model.py
```

This creates:

```text
model/model.joblib
```

## 3. Run

From the project root:

```powershell
uvicorn backend.main:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

## 4. Replacing the model for a hackathon

Train your actual model and save a bundle like:

```python
model_bundle = {
    "model": trained_pipeline,
    "feature_columns": feature_columns,
    "feature_metadata": feature_metadata,
    "target": target,
    "problem_type": "classification",  # or "regression"
    "training_rows": len(X),
    "sklearn_version": sklearn.__version__,
    "test_metrics": {
        "accuracy": float(test_accuracy)
    }
}

joblib.dump(model_bundle, "model/model.joblib")
```

The FastAPI application does not need the training CSV.

## 5. IMPORTANT: save preprocessing with the model

Prefer:

```python
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

pipeline.fit(X_train, y_train)
```

and save `pipeline` inside the bundle.

This ensures the exact transformations used during training are also used
during prediction.

This is especially important for:

- StandardScaler / MinMaxScaler
- OneHotEncoder
- OrdinalEncoder
- date transformations
- text vectorizers
- missing-value imputation
- feature engineering

## 6. KNN

KNN does NOT require the original CSV at prediction time if the fitted KNN
object was saved.

For example:

```python
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", KNeighborsRegressor(n_neighbors=5))
])

pipeline.fit(X_train, y_train)

joblib.dump({
    "model": pipeline,
    ...
}, "model/model.joblib")
```

The fitted KNN stores the training samples it needs for neighbor lookup.

The tradeoff is that KNN model files can become large for large datasets.

## 7. Supported field metadata

Examples:

```python
"age": {
    "type": "integer",
    "required": True,
    "min": 0,
    "max": 120
}
```

```python
"salary": {
    "type": "numeric",
    "required": True
}
```

```python
"joining_date": {
    "type": "date",
    "required": True
}
```

```python
"city": {
    "type": "string",
    "required": True,
    "options": ["Delhi", "Mumbai", "Bangalore"]
}
```

```python
"has_previous_job": {
    "type": "boolean",
    "required": True
}
```

## 8. Architecture

```text
Browser
   |
   | GET /api/config
   v
FastAPI --------------------> model.joblib
   |                              |
   | POST /api/predict            |
   v                              |
DataFrame ----------------> Pipeline
                                  |
                           preprocessing
                                  |
                               ML model
                                  |
                                  v
                              prediction
                                  |
                                  v
                              Browser
```

## Security note

Only load `.joblib` files that your team created or trusts. `joblib.load`
uses Python object deserialization and should not be used on untrusted model
files.
