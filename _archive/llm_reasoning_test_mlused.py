from server.config import *
import csv
import ast  # to convert string dict to real dict
import json
import pandas as pd
import logging
import re
import os
import requests
import sqlite3



# THIS DEFINITION SET IS FOR LLM REASONING ENGINE TO ASSIGN ACTIVITIES TO OUTDOOR SPACES
# It uses a local LLM server to generate assignments based on activity space data and resident preferences.
EXPLANATION_MODE = True
# True: Answers single query in detail
# False: Assigns activities in a file based on llm reasoning
logging.basicConfig(level=logging.INFO)

def load_csvs():
     # Load activity space geometries
    conn = sqlite3.connect('sql/gh_data.db')  # Use your actual DB path
    geometries = pd.read_sql_query("SELECT * FROM activity_space", conn)
    # match column name in activity_space:
    geometries.rename(columns={"key": "id"}, inplace=True)
    conn.close()
    # Load CSV prediction and resident data
    thresh = pd.read_csv('ml_models/threshold_predictions.csv')
    green = pd.read_csv('ml_models/green_predictions.csv')
    usability = pd.read_csv('ml_models/usability_predictions.csv')
    voting = pd.read_csv('resident_data/voting_weights.csv')
    conn = sqlite3.connect('sql/gh_data.db')  # Use your actual DB path
     # Load resident distances
    distances = pd.read_sql_query("SELECT * FROM resident_distances", conn)
    # match column name in resident_distances:
    distances.rename(columns={"Outdoor Space": "id"}, inplace=True)
    conn.close()
    conn = sqlite3.connect('sql/gh_data.db')  # Use your actual DB path
    # Load personas
    personas = pd.read_sql_query("SELECT * FROM personas_assigned", conn)
    conn.close()
    return geometries, thresh, green, usability, voting, distances, personas

def normalize_ids(dfs):
    for df in dfs:
        df["id"] = df["id"].astype(str)
    return dfs

def make_prompt(row, space_id, activity_scores, residents_summary):
    scores_text = "\n".join([f"- {a}: {round(s, 3)}" for a, s in sorted(activity_scores.items(), key=lambda x: -x[1])])

    if EXPLANATION_MODE:
        return f"""
You are an architecture assistant assigning the best outdoor activity for a given space.

### Outdoor space description:
- ID: {space_id}
- Type: {row['type']}
- Orientation: {row['orientation']}
- Area: {row['area']}
- Open sides: {row['open_side']}
- Compactness: {row['compactness']}

### Threshold-based prediction, for outdoor activities you can only choose from this list for related outdoor spaces:
{row.get('predicted_activities', 'None')}

### Green prediction:
{row.get('green_prediction', 'None')}

### Usability prediction:
{row.get('usability_prediction', 'None')}

### Nearby residents:
{residents_summary}

### Voting-weighted activity preferences:
{scores_text}

Return your reasoning and the **best matching activity which you choose from predicted_activities for related outdoor space {space_id}** in the following JSON format:
```
{{
  "parameters": {{
    "id": "{space_id}",
    "activity": "..."
  }},
  "reasoning": "..."
}}
```
Only output valid JSON, no commentary.
"""
    else:
        return f"""
Only return JSON like below. No explanation:
```
{{
  "parameters": {{
    "id": "{space_id}",
    "activity": "..."
  }}
}}
"""

def call_local_llm(prompt):
    try:
        response = requests.post(
            "http://localhost:1234/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json={
                "model": "local-model",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7
            },
            timeout=30  # <-- Add timeout here
        )
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.Timeout:
        return '{"parameters": {"activity": null}, "reasoning": "LLM request timed out"}'
    except Exception as e:
        return f'{{"parameters": {{"activity": null}}, "reasoning": "LLM error: {str(e)}"}}'

def generate_llm_assignments(output_path="llm_reasoning\llm_assignments.json"):
    geometries, thresh, green, usability, voting, distances, personas = load_csvs()
    geometries, thresh, green, usability = normalize_ids([geometries, thresh, green, usability])

    merged = geometries.copy()
    merged = merged.merge(thresh[["id", "predicted_activities"]], on="id", how="left")
    merged = merged.merge(green[["id", "green_prediction"]], on="id", how="left")
    merged = merged.merge(usability[["id", "usability_prediction"]], on="id", how="left")

    # Ensure 'in_out' is present after merging
    # if 'in_out' not in merged.columns:
    #     merged = merged.merge(geometries[['id', 'in_out']], on='id', how='left')

    if 'resident' in voting.columns:
        voting['resident'] = voting['resident'].astype(str)
        resident_weights = voting.groupby(['space', 'activity'])['weight'].sum().unstack(fill_value=0).reset_index()
        resident_weights['activity_weights'] = resident_weights.drop(columns=['space']).to_dict(orient='records')
        resident_weights = resident_weights[['space', 'activity_weights']]
        merged = merged.merge(resident_weights, left_on="id", right_on="space", how="left")
    else:
        logging.warning("Voting CSV missing 'resident' column.")

    personas["resident_key"] = personas["resident_key"].astype(str)
    results = []

    for _, row in merged.iterrows():
        try:
            space_id = row["id"]
            for col in ["area", "type", "orientation"]:
                if col not in row or pd.isna(row[col]):
                    raise KeyError(f"Missing or NaN column: {col}")

            distance_row = distances[distances["id"] == space_id]
            if distance_row.empty:
                raise ValueError(f"No distances found for space {space_id}")

            resident_distances = distance_row.drop(columns=["id"]).T
            resident_distances.columns = ['distance']
            resident_distances.index.name = 'resident_key'
            resident_distances.reset_index(inplace=True)
            top_residents = resident_distances.sort_values('distance').head(5)

            resident_ids = top_residents["resident_key"].tolist()
            residents_info = personas[personas["resident_key"].isin(resident_ids)]

            residents_summary = "\n".join([
                f"- {r['resident_key']}: {r['resident_persona']} ({r['resident_population']} people)"
                for _, r in residents_info.iterrows()
            ])

            # activity_scores = {} OLD METHOD COUNTING EVERY RESIDENT VOTE
            # for _, res in residents_info.iterrows():
            #     key = res["resident_key"]
            #     pop = float(res["resident_population"])
            #     dist = float(top_residents[top_residents["resident_key"] == key]["distance"].values[0])
            #     weight = pop / ((dist + 1e-5) ** 2) # to increase weight for closer residents
            #     res_votes = voting[(voting["resident"] == key) & (voting["space"] == space_id)]
            #     for _, vote in res_votes.iterrows():
            #         act = vote["activity"]
            #         score = float(vote["weight"])
            #         activity_scores[act] = activity_scores.get(act, 0) + score * weight

            #Calculate activity_scores based on these residents
            activity_scores = {}
            for _, res in residents_info.iterrows():
                key = res["resident_key"]
                pop = float(res["resident_population"])
                # Only consider votes from this resident for this space
                res_votes = voting[(voting["resident"] == key) & (voting["space"] == space_id)]
                for _, vote in res_votes.iterrows():
                    act = vote["activity"]
                    score = float(vote["weight"]) * pop  # or just use vote["weight"] if you don't want to weight by population
                    activity_scores[act] = activity_scores.get(act, 0) + score

            prompt = make_prompt(row, space_id, activity_scores, residents_summary)
            llm_response = call_local_llm(prompt)
            print(f"[DEBUG] Response for {space_id}:\n{llm_response}\n")

            try:
                json_result = json.loads(llm_response)
                if "parameters" not in json_result or "activity" not in json_result["parameters"]:
                    raise ValueError("Missing 'parameters' or 'activity' in response")
            except Exception as e:
                logging.error(f"Invalid JSON from LLM for {space_id}: {e}")
                json_result = {
                    "parameters": {"id": space_id, "activity": None},
                    "reasoning": f"Invalid LLM output: {llm_response}"
                }

            results.append(json_result)

        except Exception as e:
            logging.error(f"LLM failed for {row['id']}: {e}")
            results.append({
                "parameters": {"id": row["id"], "activity": None},
                "reasoning": f"Error: {e}"
            })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logging.info(f"LLM assignments saved to {output_path}")

    summary = [{"space_id": r["parameters"]["id"], "assigned_activity": r["parameters"]["activity"]} for r in results]
    pd.DataFrame(summary).to_csv("llm_reasoning\llm_activity_assignments.csv", index=False)
    logging.info("CSV summary saved to llm_reasoning\llm_activity_assignments.csv")


#THIS BIT OF CODE IS FOR GENERATING ALL ASSIGNMENTS AND SAVING THEM TO A FILE 8COMMENT OUT IF YOU DO NOT WANT TO USE IT)
if __name__ == "__main__":
    generate_llm_assignments()



# THIS DEFINITION SET IS FOR EXPLAINING A SINGLE ACTIVITY FOR A SPACE
def explain_activity_for_space(space_id, question, geometries, thresh, green, usability, voting, distances, personas, assignments_path="llm_reasoning\llm_activity_assignments.csv"):
    # Load assignments and get assigned activity for this space
    assignments = pd.read_csv(assignments_path)
    assigned_row = assignments[assignments['space_id'] == space_id]
    if assigned_row.empty:
        return f"No assigned activity found for space {space_id}."
    activity = assigned_row.iloc[0]['assigned_activity']

    # Find the row for this space
    row = geometries[geometries["id"] == space_id]
    if row.empty:
        return f"No data for space {space_id}"
    row = row.iloc[0]

    # Merge predictions
    for df, col in [(thresh, "predicted_activities"), (green, "green_prediction"), (usability, "usability_prediction")]:
        pred_row = df[df["id"] == space_id]
        if not pred_row.empty:
            row[col] = pred_row.iloc[0][col]
        else:
            row[col] = None

    # Ensure 'in_out'
    # if "in_out" not in row or pd.isna(row["in_out"]):
    #     in_out_row = geometries[geometries["id"] == space_id]
    #     row["in_out"] = in_out_row.iloc[0]["in_out"] if not in_out_row.empty else "unknown"

    # Resident distances
    distance_row = distances[distances["id"] == space_id]
    if distance_row.empty:
        return f"No distances found for space {space_id}"
    resident_distances = distance_row.drop(columns=["id"]).T
    resident_distances.columns = ['distance']
    resident_distances.index.name = 'resident_key'
    resident_distances.reset_index(inplace=True)
    top_residents = resident_distances.sort_values('distance').head(5)
    resident_ids = top_residents["resident_key"].tolist()
    residents_info = personas[personas["resident_key"].isin(resident_ids)]

    residents_summary = "\n".join([
        f"- {r['resident_key']}: {r['resident_persona']} ({r['resident_population']} people)"
        for _, r in residents_info.iterrows()
    ])

    # Voting-weighted activity preferences
    activity_scores = {}
    for _, res in residents_info.iterrows():
        key = res["resident_key"]
        pop = float(res["resident_population"])
        dist = float(top_residents[top_residents["resident_key"] == key]["distance"].values[0])
        weight = pop / (dist + 1e-5)
        res_votes = voting[(voting["resident"] == key) & (voting["space"] == space_id)]
        for _, vote in res_votes.iterrows():
            act = vote["activity"]
            score = float(vote["weight"])
            activity_scores[act] = activity_scores.get(act, 0) + score * weight

    # Build a prompt that asks "Why should this space have its assigned activity?"
    prompt = f"""
You are an architecture assistant. Given the following data, answer the user's question about why the assigned activity for this space is "{activity}". Focus your reasoning on the match between the space and this activity, using all available predictions, resident data, and voting preferences.

### User question:
{question}

### Outdoor space description:
- ID: {space_id}
- Type: {row['type']}
- Orientation: {row['orientation']}
- Area: {row['area']}
- Open sides: {row['open_side']}
- Compactness: {row['compactness']}

### Threshold-based prediction:
{row.get('predicted_activities', 'None')}

### Green prediction:
{row.get('green_prediction', 'None')}

### Usability prediction:
{row.get('usability_prediction', 'None')}

### Nearby residents:
{residents_summary}

### Voting-weighted activity preferences:
{chr(10).join([f"- {a}: {round(s, 3)}" for a, s in sorted(activity_scores.items(), key=lambda x: -x[1])])}

Explain specifically why "{activity}" is a good fit for this space, in the context of the user's question.
"""

    return call_local_llm(prompt)

# THIS DEFINITION SET IS FOR EXPLAINING GENERAL SPACE QUESTIONS
def answer_general_space_question(
    house_key, question, geometries, thresh, green, usability, voting, distances, personas,
    assignments_path="llm_reasoning/llm_activity_assignments.csv"
):
    # Find nearest 5 outdoor spaces
    if house_key not in distances.columns:
        return f"No distances found for house key {house_key}."

    assignments = pd.read_csv(assignments_path)
    nearby = distances[["id", house_key]].rename(columns={house_key: "distance", "id": "space_id"})
    nearby = nearby.sort_values("distance").head(5)

    # For each space, get assigned activity, voting summary, and extra info
    space_summaries = []
    for _, row in nearby.iterrows():
        space_id = row['space_id']
        distance = row['distance']

        # Assigned activity
        assigned_row = assignments[assignments['space_id'] == space_id]
        assigned_activity = assigned_row.iloc[0]['assigned_activity'] if not assigned_row.empty else "Unknown"

        # Geometry info
        geo_row = geometries[geometries["id"] == space_id]
        if not geo_row.empty:
            geo_row = geo_row.iloc[0]
            type_ = geo_row.get("type", "Unknown")
            orientation = geo_row.get("orientation", "Unknown")
            area = geo_row.get("area", "Unknown")
            open_side = geo_row.get("open_side", "Unknown")
            compactness = geo_row.get("compactness", "Unknown")
        else:
            type_ = orientation = area = open_side = compactness = "Unknown"

        # Predictions
        pred_thresh = thresh[thresh["id"] == space_id]["predicted_activities"].values[0] if not thresh[thresh["id"] == space_id].empty else "None"
        pred_green = green[green["id"] == space_id]["green_prediction"].values[0] if not green[green["id"] == space_id].empty else "None"
        pred_usability = usability[usability["id"] == space_id]["usability_prediction"].values[0] if not usability[usability["id"] == space_id].empty else "None"

        # Closest 5 residents and personas
        distance_row = distances[distances["id"] == space_id]
        if not distance_row.empty:
            resident_distances = distance_row.drop(columns=["id"]).T
            resident_distances.columns = ['distance']
            resident_distances.index.name = 'resident_key'
            resident_distances.reset_index(inplace=True)
            top_residents = resident_distances.sort_values('distance').head(5)
            resident_ids = top_residents["resident_key"].tolist()
            residents_info = personas[personas["resident_key"].isin(resident_ids)]
            residents_summary = "; ".join([
                f"{r['resident_key']}: {r['resident_persona']} ({r['resident_population']} people)"
                for _, r in residents_info.iterrows()
            ])
        else:
            residents_summary = "No resident data"

        # Voting summary
        votes = voting[voting['space'] == space_id]
        if not votes.empty:
            top_votes = (
                votes.groupby('activity')['weight']
                .sum()
                .sort_values(ascending=False)
                .head(3)
            )
            voting_summary = "; ".join([f"{act}: {w:.1f}" for act, w in top_votes.items()])
        else:
            voting_summary = "No voting data"

        resident_votes = voting[(voting['space'] == space_id) & (voting['resident'] == house_key)]
        if not resident_votes.empty:
            resident_voting_summary = "; ".join([f"{row['activity']}: {row['weight']:.2f}" for _, row in resident_votes.iterrows()])
        else:
            resident_voting_summary = "No votes from this resident"

        # Compose summary for this space
        space_summaries.append(
            f"- {space_id} ({assigned_activity}, {distance:.1f}m away)\n"
            f"  Type: {type_}, Orientation: {orientation}, Area: {area}, Open sides: {open_side}, Compactness: {compactness}\n"
            f"  Threshold: {pred_thresh} | Green: {pred_green} | Usability: {pred_usability}\n"
            f"  Nearby residents: {residents_summary}\n"
            f"  Voting (all): {voting_summary} | Your votes: {resident_voting_summary}"
        )

    space_summaries_text = "\n\n".join(space_summaries)

    prompt = f"""
You are a community advisor helping a resident understand the outdoor spaces near them.

### Resident info:
- House key: {house_key}

### Question:
{question}

### Nearby spaces, assigned activities, properties, predictions, residents, and voting:
{space_summaries_text}

Outdoor spaces are activity spaces and their keys start with O1, O2, etc. Each space has an assigned activity and a distance from the resident's house.
Apartment keys are residents' house keys and start with H1, H2, etc. The resident has a question about the nearby spaces.
### Your task:
Answer the resident's question based on this information.
Be concise and use plain language.
"""

    return call_local_llm(prompt)

