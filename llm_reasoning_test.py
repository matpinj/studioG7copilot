from server.config import *
import csv
import ast  # to convert string dict to real dict
from server.config import *
import json
import pandas as pd
import logging
import re
import logging
import os
import requests
import sqlite3
import pandas as pd

# THIS DEFINITION SET IS FOR LLM REASONING ENGINE
EXPLANATION_MODE = True
# True: Answers single query in detail
# False: Assigns activities in a file based on llm reasoning

logging.basicConfig(level=logging.INFO)

def load_csvs():
    conn = sqlite3.connect('sql/gh_data.db')  # Use your actual DB path
    geometries = pd.read_sql_query("SELECT * FROM activity_space", conn)
    # match column name in activity_space:
    geometries.rename(columns={"key": "id"}, inplace=True)
    conn.close()
    thresh = pd.read_csv('ml_models/threshold_predictions.csv')
    green = pd.read_csv('ml_models/green_predictions.csv')
    usability = pd.read_csv('ml_models/usability_predictions.csv')
    voting = pd.read_csv('resident_data/voting_weights.csv')
    conn = sqlite3.connect('sql/gh_data.db')  # Use your actual DB path
    distances = pd.read_sql_query("SELECT * FROM resident_distances", conn)
    # match column name in resident_distances:
    distances.rename(columns={"Outdoor Space": "id"}, inplace=True)
    conn.close()
    conn = sqlite3.connect('sql/gh_data.db')  # Use your actual DB path
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
- Indoor/Outdoor: {row.get('in_out', 'unknown')}
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
{scores_text}

Return your reasoning and the **best matching activity** in the following JSON format:
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

def generate_llm_assignments(output_path="llm_assignments.json"):
    geometries, thresh, green, usability, voting, distances, personas = load_csvs()
    geometries, thresh, green, usability = normalize_ids([geometries, thresh, green, usability])

    merged = geometries.copy()
    merged = merged.merge(thresh[["id", "predicted_activities"]], on="id", how="left")
    merged = merged.merge(green[["id", "green_prediction"]], on="id", how="left")
    merged = merged.merge(usability[["id", "usability_prediction"]], on="id", how="left")

    # Ensure 'in_out' is present after merging
    if 'in_out' not in merged.columns:
        merged = merged.merge(geometries[['id', 'in_out']], on='id', how='left')

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
    pd.DataFrame(summary).to_csv("llm_activity_assignments.csv", index=False)
    logging.info("CSV summary saved to llm_activity_assignments.csv")


#THIS CODE IS FOR GENERATING ALL ASSIGNMENTS AND SAVING THEM TO A FILE COMMENT OUT IF YOU DO NOT WANT TO USE IT)
if __name__ == "__main__":
    generate_llm_assignments()

