import pandas as pd
import numpy as np
import pickle
import tensorflow as tf

# --- File paths ---
INPUT_CSV = "ml_models\\activity_space_ml.csv"
OUTPUT_CSV = "ml_models\\threshold_predictions.csv"
SCALER_PATH = "ml_models\\scaler_ann_2405_01.pkl"
MODEL_PATH = "ml_models\\annmodel_2405_01.keras"

# --- Load and clean CSV ---
df = pd.read_csv(INPUT_CSV)

# Encode categorical columns
df["type"] = df["type"].map({"balcony": 0, "courtyard": 1, "terrace": 2})
df["orientation"] = df["orientation"].map({
    "E": 0, "N": 1, "NE": 2, "NW": 3,
    "S": 4, "SE": 5, "SW": 6, "W": 7
})
df["in_out"] = df["in_out"].map({"indoor": 0, "outdoor": 1})

# Required feature columns
features = [
    "type", "orientation", "area", "level", "open_side", "in_out",
    "compactness", "types touching", "neighbours touching",
    "bounding area ratio", "core distance", "cos(angle_in_radians)",
    "sin(angle_in_radians)", "edge count", "longest edge length"
]

# Drop rows with missing values in those columns
df = df.dropna(subset=features)

# --- Feature extraction (use DataFrame to avoid warning) ---
X = df[features]  # ✅ keep as DataFrame instead of using .values

# Load scaler and scale input
with open(SCALER_PATH, 'rb') as f:
    Sscaler = pickle.load(f)
x_scaled = Sscaler.transform(X)

# Load TensorFlow model
model = tf.keras.models.load_model(MODEL_PATH)

# Predict
predictions = model.predict(x_scaled)

# Decode predictions
threshold = 0.5
binary_preds = (predictions > threshold).astype(int)

activity_labels = [
    'Biodiversity balcony', 'Community Pool/BBQ', 'Creative Corridor',
    'Flexible Space', 'Green Corridor', 'Healing Garden', 'Offline Retreat',
    'Outdoor Cinema/Event Space', 'Outdoor Meeting Room', 'Playground',
    'Sitting', 'Sports', 'Storage & Technical Space', 'Sunbath',
    'Urban Agriculture Garden', 'Viewpoint'
]

decoded_preds = []
for row in binary_preds:
    activities = [activity_labels[i] for i, val in enumerate(row) if val == 1]
    decoded_preds.append(activities)

# Add O1, O2, O3... keys as the ID column
df["id"] = [f"O{i+1}" for i in range(len(df))]

# Move 'id' to the front for readability (optional)
df = df[["id"] + [col for col in df.columns if col != "id"]]

# Add predictions to CSV
df["predicted_activities"] = decoded_preds
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Saved predictions to {OUTPUT_CSV}")
