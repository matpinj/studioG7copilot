import locale
import pandas as pd
import numpy as np
import pickle
from io import StringIO
import os

# Set locale
locale.setlocale(locale.LC_ALL, 'en_US')

# --- USER INPUT ---
INPUT_CSV = "ml_models\\activity_space_ml.csv"
output_csv_path = 'ml_models\\green_predictions.csv'
scaler_path = 'ml_models\\gu_scaler.pkl'
model_path = 'ml_models\\greenmodel_LR_2405_01.pkl'

try:
    # STEP 1: Load and clean CSV text
    with open(INPUT_CSV, 'r', encoding='utf-8-sig') as f:
        cleaned_text = "\n".join(
            line.strip().rstrip(",") for line in f.readlines() if line.strip()
        )
    df = pd.read_csv(StringIO(cleaned_text))

    # Clean column names (strip whitespace, BOMs)
    df.columns = [col.strip() for col in df.columns]


    #SOME MANUAL CLEANING
     # Drop unwanted column(s) if they exist
    if "bounding area ratio" in df.columns:
        df = df.drop(columns=["bounding area ratio"])
    # Add 'height' column with value 3 for all rows if not present
    if "height" not in df.columns:
        df["height"] = 3



    # Debug: show column names and unique values for critical columns
    print("Columns:", df.columns.tolist())
    print("Unique values in 'type':", df["type"].unique())

    # STEP 2: Encode categorical features (safely)
    type_map = {"balcony": 0, "courtyard": 1, "terrace": 2}
    orientation_map = {
        "E": 0, "N": 1, "NE": 2, "NW": 3,
        "S": 4, "SE": 5, "SW": 6, "W": 7
    }
    in_out_map = {"indoor": 0, "outdoor": 1}

    df["type"] = df["type"].map(type_map)
    df["orientation"] = df["orientation"].map(orientation_map)
    df["in_out"] = df["in_out"].map(in_out_map)

    # Drop rows with unmapped (NaN) values
    df = df.dropna(subset=["type", "orientation", "in_out"])



    # STEP 3: Feature selection
    features = [
        "type", "orientation","height", "area", "level", "open_side", "in_out",
        "compactness", "types touching", "neighbours touching", 
        "core distance", "cos(angle_in_radians)",
        "sin(angle_in_radians)", "edge count", "longest edge length"
    ]
    df = df.dropna(subset=features)
    X = df[features].values

    # STEP 4: Load scaler and transform
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    x_scaled = scaler.transform(X)

    # STEP 5: Load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    predictions = model.predict(x_scaled)

    # STEP 6: Map predictions to labels
    green_labels = ["not_suitable", "suitable"]
    decoded_preds = [green_labels[i] for i in predictions]

    # STEP 7: Output
    df["green_prediction"] = decoded_preds
    print(df[["green_prediction"]])

    # STEP 8: Save predictions
    df["id"] = [f"O{i+1}" for i in range(len(df))]
    df = df[["id"] + [col for col in df.columns if col != "id"]]
    df.to_csv(output_csv_path, index=False)
    print(f"Predictions saved to {output_csv_path}")

except Exception as e:
    import traceback
    traceback.print_exc()
    print("Error:", e)
