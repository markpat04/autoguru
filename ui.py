"""Gradio UI for AutoGuru. Mounted at / by app.py (proxy env is sanitized there
before this module is imported)."""
import gradio as gr
import pandas as pd

from predictor_service import get_feature_values, predict_car


def build_ui() -> gr.Blocks:
    fv = get_feature_values()
    defaults = fv["defaults"]

    def dd(col, label):
        return gr.Dropdown(
            choices=fv[col], value=defaults[col], label=label, allow_custom_value=True
        )

    with gr.Blocks(title="AutoGuru — Used Car Price Predictor") as demo:
        gr.Markdown(
            "# AutoGuru\n"
            "Used-car price predictor (AutoGluon + LightGBM/XGBoost). "
            "Fill in the car's features and get a price estimate with a 90% "
            "prediction interval and the top price-driving factors."
        )
        with gr.Row():
            brand = dd("brand", "Brand")
            model = dd("model", "Model")
            model_year = gr.Number(value=defaults["model_year"], label="Model year")
            milage = gr.Number(value=defaults["milage"], label="Mileage")
        with gr.Row():
            fuel_type = dd("fuel_type", "Fuel type")
            engine = gr.Textbox(value=defaults["engine"], label="Engine")
            transmission = dd("transmission", "Transmission")
        with gr.Row():
            ext_col = dd("ext_col", "Exterior color")
            int_col = dd("int_col", "Interior color")
            accident = dd("accident", "Accident history")
            clean_title = dd("clean_title", "Clean title")
        btn = gr.Button("Predict price", variant="primary")
        price_md = gr.Markdown()
        factors_df = gr.Dataframe(label="Top price-driving factors")

        def on_predict(*vals):
            keys = ["brand", "model", "model_year", "milage", "fuel_type",
                    "engine", "transmission", "ext_col", "int_col",
                    "accident", "clean_title"]
            res = predict_car(dict(zip(keys, vals)))
            md = (
                f"## Estimated price: **${res['price']:,.0f}**\n"
                f"90% prediction interval: ${res['interval_low']:,.0f} – "
                f"${res['interval_high']:,.0f}"
            )
            return md, pd.DataFrame(res["top_factors"])

        btn.click(
            on_predict,
            inputs=[brand, model, model_year, milage, fuel_type, engine,
                    transmission, ext_col, int_col, accident, clean_title],
            outputs=[price_md, factors_df],
        )
        gr.Markdown(
            "_Model trained on a public US used-car dataset (see README). "
            "Estimates are in USD and carry the dataset's US-market bias._"
        )
    return demo
