
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

# Load the new 3-feature Random Forest model
model = joblib.load("rootsense_3feature_model.pkl")
label_encoder = joblib.load("rootsense_3feature_label_encoder.pkl")
feature_names = joblib.load("rootsense_3feature_names.pkl")


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "RootSense ML Backend is running",
        "best_model": "Random Forest"
    })


@app.route("/model-info", methods=["GET"])
def model_info():
    return jsonify({
        "best_model": "Random Forest",
        "accuracy": 57.33,
        "features": feature_names,
        "classes": list(label_encoder.classes_)
    })


@app.route("/zones", methods=["GET"])
def zones():
    try:
        csv_path = "RootSense_1500_Root_Health_Dataset.csv"

        if not os.path.exists(csv_path):
            return jsonify({"error": "Dataset CSV file not found"}), 404

        df = pd.read_csv(csv_path)

        required_columns = [
            "Zone", "Soil Moisture (%)", "Soil Temperature (°C)",
            "Soil pH", "Electrical Conductivity (EC)",
            "Nitrogen (N)", "Phosphorus (P)", "Potassium (K)",
            "Air Temperature (°C)", "Humidity (%)"
        ]

        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            return jsonify({
                "error": "Missing columns in dataset",
                "missing_columns": missing
            }), 400

        zone_data = []
        for zone in sorted(df["Zone"].dropna().unique()):
            rows = df[df["Zone"] == zone]
            if rows.empty:
                continue

            row = rows.iloc[0]
            zone_data.append({
                "zone": str(zone),
                "soil_moisture": float(row["Soil Moisture (%)"]),
                "soil_temperature": float(row["Soil Temperature (°C)"]),
                "soil_ph": float(row["Soil pH"]),
                "ec": float(row["Electrical Conductivity (EC)"]),
                "nitrogen": float(row["Nitrogen (N)"]),
                "phosphorus": float(row["Phosphorus (P)"]),
                "potassium": float(row["Potassium (K)"]),
                "air_temperature": float(row["Air Temperature (°C)"]),
                "humidity": float(row["Humidity (%)"])
            })

        return jsonify({"zones": zone_data})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True) or {}

        # Accept the sensor fields sent by the existing Flutter app.
        input_values = {
            "Humidity (%)": data["humidity"],
            "Soil Temperature (°C)": data["soil_temperature"],
            "Soil Moisture (%)": data["soil_moisture"]
        }

        input_data = pd.DataFrame([input_values])
        input_data = input_data[feature_names]

        # The new Random Forest model was trained without a scaler.
        prediction = model.predict(input_data)
        root_health = str(label_encoder.inverse_transform(prediction)[0])

        recommendations = {
            "Healthy": "Maintain current soil conditions and continue regular monitoring.",
            "Stressed": "Check soil moisture and temperature. Adjust irrigation as needed and monitor the plant closely.",
            "Unhealthy": "Inspect the root-zone conditions and check moisture and temperature. Consider soil testing and expert advice."
        }

        return jsonify({
            "root_health": root_health,
            "recommendation": recommendations.get(
                root_health,
                "Monitor soil conditions and inspect the plant."
            )
        })

    except KeyError as e:
        return jsonify({
            "error": f"Missing required sensor value: {e}"
        }), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
