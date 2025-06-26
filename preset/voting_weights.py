import pandas as pd
import json
import sqlite3
import os

# Load all files
conn = sqlite3.connect('sql/gh_data.db')
distances = pd.read_sql_query("SELECT * FROM resident_distances", conn)
personas = pd.read_sql_query("SELECT * FROM personas_assigned", conn)
conn.close()

with open("preset/persona_activity.json") as f:
    persona_activities = json.load(f)

# Clean headers
distances.columns = [col.strip() for col in distances.columns]
personas.columns = [col.strip() for col in personas.columns]

# Create a map from resident_key to (population, persona)
resident_map = {
    row["resident_key"]: {
        "population": int(row["resident_population"]),
        "persona": row["resident_persona"].strip()
    }
    for _, row in personas.iterrows()
}

# Process weights
results = []
for _, row in distances.iterrows():
    space_id = row["Outdoor Space"]
    # Get distances for all residents for this space
    resident_distances = [
        (resident_key, row[resident_key])
        for resident_key in row.index[1:]
        if pd.notna(row[resident_key]) and resident_key in resident_map
    ]
    # Sort by distance and take the closest 5
    closest_residents = sorted(resident_distances, key=lambda x: x[1])[:5]
    for resident_key, distance in closest_residents:
        resident_data = resident_map[resident_key]
        population = resident_data["population"]
        persona = resident_data["persona"]

        activity_scores = persona_activities.get(persona, {})
        for activity, preference_score in activity_scores.items():
            proximity = 1 / (1 + distance)
            group_weight = 1 + 0.25 * (population - 1)
            weight = round(preference_score * proximity * group_weight, 4)

            results.append({
                "resident": resident_key,
                "space": space_id,
                "activity": activity,
                "distance": distance,
                "weight": weight
            })

# Create and show DataFrame
voting_df = pd.DataFrame(results)
# import ace_tools as tools; tools.display_dataframe_to_user(name="Voting Weights", dataframe=voting_df)
# To display the DataFrame, you can use the following line instead:
print(voting_df)
voting_df.to_csv("preset/voting_weights.csv", index=False)
voting_df.to_csv("resident_data/voting_weights.csv", index=False)

