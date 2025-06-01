import pandas as pd
import json

# Load all files
distances = pd.read_csv("resident_data/resident_distances.csv")
personas = pd.read_csv("resident_data/personas_assigned.csv")
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
    for resident_key in row.index[1:]:
        if pd.isna(row[resident_key]) or resident_key not in resident_map:
            continue

        distance = float(row[resident_key])
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
