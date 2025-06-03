import locale
import pandas as pd
import numpy as np
import pickle
from io import StringIO
import os
import traceback  # [ADDED] for detailed error logs

# Set locale
locale.setlocale(locale.LC_ALL, 'en_US')

# --- USER INPUT ---
csv_file_path = r'ml_models\activity_space_ml.csv'
output_csv_path = r'ml_models\usability_predictions.csv'
scaler_path = r'ml_models\gu_scaler.pkl'
model_path = r'ml_models\usabilitymodel_LR_2405_01.pkl'

try:
    # STEP 1: Load and clean CSV file with UTF-8-SIG to handle BOM
    with open(csv_file_path, 'r', encoding='utf-8-sig') as f:  # [MODIFIED]
        cleaned_text = "\n".join(
            line.strip().rstrip(",") for line in f.readlines() if line.strip()
        )
    df = pd.read_csv(StringIO(cleaned_text))

    # STEP 2: Clean column names (strip whitespace, BOMs, etc.)
    df.columns = [col.strip() for col in df.columns]  # [ADDED]


    #SOME MANUAL CLEANING
     # Drop unwanted column(s) if they exist
    if "bounding area ratio" in df.columns:
        df = df.drop(columns=["bounding area ratio"])
    # Add 'height' column with value 3 for all rows if not present
    if "height" not in df.columns:
        df["height"] = 3

    # Debug print
    print("Columns:", df.columns.tolist())  # [ADDED]
    print("Unique values in 'type':", df["type"].unique())  # [ADDED]

    # STEP 3: Encode categorical features
    df["type"] = df["type"].map({"balcony": 0, "courtyard": 1, "terrace": 2})
    df["orientation"] = df["orientation"].map({
        "E": 0, "N": 1, "NE": 2, "NW": 3,
        "S": 4, "SE": 5, "SW": 6, "W": 7
    })
    df["in_out"] = df["in_out"].map({"indoor": 0, "outdoor": 1})

    # Drop rows with unmapped (NaN) values
    df = df.dropna(subset=["type", "orientation", "in_out"])  # [ADDED]

    # STEP 4: Feature selection
    features = [
        "type", "orientation","height", "area", "level", "open_side", "in_out",
        "compactness", "types touching", "neighbours touching", 
        "core distance", "cos(angle_in_radians)",
        "sin(angle_in_radians)", "edge count", "longest edge length"
    ]
    df = df.dropna(subset=features)
    X = df[features].values

    # STEP 5: Load scaler and transform
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    x_scaled = scaler.transform(X)

    # STEP 6: Load model and predict
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    predictions = model.predict(x_scaled)

    # STEP 7: Map predictions to labels
    usability_labels = ["comfortable", "cool_breezy", "warm_sunny"]
    decoded_preds = [usability_labels[i] for i in predictions]

    # STEP 8: Output
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
    traceback.print_exc()  # [MODIFIED] full traceback for better debugging
    print(f"❌ An error occurred: {e}")
