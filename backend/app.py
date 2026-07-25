
"""
SuperKart Sales Forecasting - Flask Backend API
Serves predictions from the serialized sklearn pipeline (preprocessing + Random Forest model).
"""

from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "superkart_best_model.pkl")
model = joblib.load(MODEL_PATH)

CURRENT_YEAR = 2025

REQUIRED_FIELDS = [
    "Product_Weight",
    "Product_Sugar_Content",
    "Product_Allocated_Area",
    "Product_Type",
    "Product_MRP",
    "Store_Establishment_Year",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
]


def build_features(payload: dict) -> pd.DataFrame:
    """Replicate the notebook's feature engineering for a single incoming record."""
    sugar = payload["Product_Sugar_Content"]
    sugar = "Regular" if sugar == "reg" else sugar

    row = {
        "Product_Weight": float(payload["Product_Weight"]),
        "Product_Sugar_Content": sugar,
        "Product_Allocated_Area": float(payload["Product_Allocated_Area"]),
        "Product_Type": payload["Product_Type"],
        "Product_MRP": float(payload["Product_MRP"]),
        "Store_Size": payload["Store_Size"],
        "Store_Location_City_Type": payload["Store_Location_City_Type"],
        "Store_Type": payload["Store_Type"],
        "Store_Age": CURRENT_YEAR - int(payload["Store_Establishment_Year"]),
        "Product_Id_Category": payload.get("Product_Id_Category", "Food"),
    }
    return pd.DataFrame([row])


@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "SuperKart sales forecasting API is running."})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(force=True)
        missing = [f for f in REQUIRED_FIELDS if f not in payload]
        if missing:
            return jsonify({"error": f"Missing required fields: {missing}"}), 400

        X = build_features(payload)
        pred = model.predict(X)[0]
        return jsonify({"predicted_sales": round(float(pred), 2)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)
