import locale
import pandas as pd
import numpy as np
import pickle

from io import StringIO
import os

# Set locale
locale.setlocale(locale.LC_ALL, 'en_US')

# --- USER INPUT ---
csv_file_path = r'building_data\geometry_data.csv'  # Replace with your actual CSV path
output_csv_path = r'ml_models\usability_predictions.csv'  # Optional: CSV with predictions
scaler_path = r'ml_models\gu_scaler.pkl'
model_path = r'ml_models\usabilitymodel_LR_2405_01.pkl'

try:
    # STEP 1: Load and clean CSV file
    with open(csv_file_path, 'r') as f:
        cleaned_text = "\n".join(
            line.strip().rstrip(",") for line in f.readlines() if line.strip()
        )
    df = pd.read_csv(StringIO(cleaned_text))

    # STEP 2: Encode categorical features
    df["type"] = df["type"].map({"balcony": 0, "courtyard": 1, "terrace": 2})
    df["orientation"] = df["orientation"].map({
        "E": 0, "N": 1, "NE": 2, "NW": 3,
        "S": 4, "SE": 5, "SW": 6, "W": 7
    })
    df["in_out"] = df["in_out"].map({"indoor": 0, "outdoor": 1})

    # STEP 3: Feature selection
    features = [
        "type", "orientation", "area", "level", "open_side", "in_out",
        "compactness", "types touching", "neighbours touching",
        "bounding area ratio", "core distance", "cos(angle_in_radians)",
        "sin(angle_in_radians)", "edge count", "longest edge length"
    ]
    df = df.dropna(subset=features)
    X = df[features].values

    # STEP 4: Load scaler and transform
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    x_scaled = scaler.transform(X)

    # STEP 5: Load model and predict
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    predictions = model.predict(x_scaled)

    # STEP 6: Map predictions to labels
    usability_labels = ["comfortable", "cool_breezy", "warm_sunny"]
    decoded_preds = [usability_labels[i] for i in predictions]

    # STEP 7: Output
    df["usability_prediction"] = decoded_preds
    print(df[["usability_prediction"]])

    # Add O1, O2, O3... keys as the ID column
    df["id"] = [f"O{i+1}" for i in range(len(df))]

    # Move 'id' to the front for readability (optional)
    df = df[["id"] + [col for col in df.columns if col != "id"]]

    # Save to CSV
    df.to_csv(output_csv_path, index=False)
    print(f"✅ Predictions saved to: {output_csv_path}")
except Exception as e:
    print(f"❌ An error occurred: {e}")