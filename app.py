"""AutoGuru FastAPI app.

Serves:
  GET  /health   -> service + model status
  POST /predict  -> {price, interval_low, interval_high, top_factors[]}
  GET  /         -> Gradio UI (mounted)

Run:  uvicorn app:app --host 0.0.0.0 --port 8000
"""
import os

# --- workaround: this VM's NO_PROXY contains bracketed IPv6 literals
# ("[::1]") which break httpx URL parsing at gradio import time.
for _k in ("NO_PROXY", "no_proxy"):
    _v = os.environ.get(_k, "")
    os.environ[_k] = ",".join(p for p in _v.split(",") if not p.strip().startswith("["))

from fastapi import FastAPI
from pydantic import BaseModel, Field

from predictor_service import FEATURES, get_predictor, predict_car

app = FastAPI(title="AutoGuru — Used Car Price Predictor", version="1.0.0")


class CarFeatures(BaseModel):
    brand: str = Field(examples=["Toyota"])
    model: str = Field(examples=["Camry SE"])
    model_year: float = Field(examples=[2019])
    milage: float = Field(examples=[45000])
    fuel_type: str = Field(examples=["gasoline"])
    engine: str = Field(examples=["2.5L 4-Cylinder"])
    transmission: str = Field(examples=["Automatic"])
    ext_col: str = Field(examples=["Black"])
    int_col: str = Field(examples=["Black"])
    accident: str = Field(examples=["none reported"])
    clean_title: str = Field(examples=["yes"])


@app.get("/health")
def health():
    predictor = get_predictor()
    return {
        "status": "ok",
        "model": "autogluon-tabular",
        "label": "price",
        "features": FEATURES,
        "leaderboard_best": predictor.leaderboard(silent=True).iloc[0]["model"],
    }


@app.post("/predict")
def predict(car: CarFeatures):
    return predict_car(car.model_dump())


# Mount the Gradio UI at / (import here so the proxy workaround above runs first)
from ui import build_ui  # noqa: E402

import gradio as gr  # noqa: E402

app = gr.mount_gradio_app(app, build_ui(), path="/")
