import pandas as pd
import sqlite3
import json

# Connect to your database
conn = sqlite3.connect("sql/gh_data.db")  # Update path if needed

# Load tables from SQL instead of CSV
distances = pd.read_sql_query("SELECT * FROM resident_distances", conn)
personas = pd.read_sql_query("SELECT * FROM personas_assigned", conn)

with open("preset/persona_activity.json") as f:
    persona_activities = json.load(f)

# Clean headers
distances.columns = [col.strip() for col in distances.columns]
personas.columns = [col.strip() for col in personas.columns]

# Define tenant/owner weight multiplier
def status_multiplier(status):
    if status.strip().lower() == "owner":
        return 1.5  # Owner has 1.5x influence
    return 1.0  # Tenant has standard influence

# Create a map from resident_key to (population, persona, status)
resident_map = {
    row["resident_key"]: {
        "population": int(row["resident_population"]),
        "persona": row["resident_persona"].strip(),
        "status": row["tenant/owner"].strip().lower()  # Assuming column name is 'tenant/owner'
    }
    for _, row in personas.iterrows()
}

# Process weights
results = []
for _, row in distances.iterrows():
    space_id = row["Outdoor Space"]
    for resident_key in row.index[1:]:
        value = row[resident_key]
        if pd.isna(value) or resident_key not in resident_map:
            continue
        try:
            distance = float(value)
        except (ValueError, TypeError):
            print(f"Skipping non-numeric value '{value}' for resident_key '{resident_key}' in space '{space_id}'")
            continue

        resident_data = resident_map[resident_key]
        population = resident_data["population"]
        persona = resident_data["persona"]
        status = resident_data["status"]
        multiplier = status_multiplier(status)

        activity_scores = persona_activities.get(persona, {})
        for activity, preference_score in activity_scores.items():
            proximity = 1 / (1 + distance)
            group_weight = 1 + 0.25 * (population - 1)
            weight = round(preference_score * proximity * group_weight * multiplier, 4)

            results.append({
                "resident": resident_key,
                "space": space_id,
                "activity": activity,
                "distance": distance,
                "weight": weight,
                "status": status
            })

# Create and save DataFrame
voting_df = pd.DataFrame(results)
print(voting_df)
voting_df.to_csv("resident_data/voting_weights.csv", index=False)

conn.close()