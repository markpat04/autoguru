# Trained on Python 3.12 (CPU). Model artifacts live in ../models/autogluon/
# and were produced by train.py (AutoGluon TabularPredictor, LightGBM + XGBoost).
FROM python:3.12-slim

WORKDIR /app

# Build deps for lightgbm/xgboost wheels are prebuilt; keep image lean.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# NOTE: the image ships the already-trained model (no training at container start).
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
