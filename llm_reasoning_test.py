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

# Global mode toggle
EXPLANATION_MODE = True  # Set to True for detailed reasoning, False for simple activity assignment.

logging.basicConfig(level=logging.INFO)

def load_csvs():
    geometries = pd.read_csv('gh_data/geometry_data_with_id.csv')
    thresh = pd.read_csv('ml_models/threshold_predictions.csv')
    green = pd.read_csv('ml_models/green_predictions.csv')
    usability = pd.read_csv('ml_models/usability_predictions.csv')
    voting = pd.read_csv('preset/voting_weights.csv')
    distances = pd.read_csv('resident_data/resident_distances.csv')
    personas = pd.read_csv('resident_data/personas_assigned.csv')
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
- Indoor/Outdoor: {row['in_out']}
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
    response = requests.post(
        "http://localhost:1234/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json={
            "model": "local-model",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7
        }
    )
    return response.json()["choices"][0]["message"]["content"]

def generate_llm_assignments(output_path="llm_assignments.json"):
    geometries, thresh, green, usability, voting, distances, personas = load_csvs()
    geometries, thresh, green, usability = normalize_ids([geometries, thresh, green, usability])

    merged = geometries.copy()
    merged = merged.merge(thresh[["id", "predicted_activities"]], on="id", how="left")
    merged = merged.merge(green[["id", "green_prediction"]], on="id", how="left")
    merged = merged.merge(usability[["id", "usability_prediction"]], on="id", how="left")

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

            distance_row = distances[distances["Outdoor Space"] == space_id]
            if distance_row.empty:
                raise ValueError(f"No distances found for space {space_id}")

            resident_distances = distance_row.drop(columns=["Outdoor Space"]).T
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

##THIS CODE GIVES AUTO ACTIVITIES WITH EXPLANATIONS AND WRITES TO CSV
# logging.basicConfig(level=logging.INFO)

# EXPLANATION_MODE = True # Set to True only if you want LLM to include reasoning

# ------------------------
# Load all CSV inputs
# ------------------------
# def load_csvs():
#     geometries = pd.read_csv('gh_data/geometry_data_with_id.csv')
#     thresh = pd.read_csv('ml_models/threshold_predictions.csv')
#     green = pd.read_csv('ml_models/green_predictions.csv')
#     usability = pd.read_csv('ml_models/usability_predictions.csv')
#     voting = pd.read_csv('preset/voting_weights.csv')
#     distances = pd.read_csv('resident_data/resident_distances.csv')
#     personas = pd.read_csv('resident_data/personas_assigned.csv')
#     return geometries, thresh, green, usability, voting, distances, personas

# # ------------------------
# # Normalize IDs to string
# # ------------------------
# def normalize_ids(dfs):
#     for df in dfs:
#         df["id"] = df["id"].astype(str).str.strip()
#     return dfs

# # ------------------------
# # Build prompts
# # ------------------------
# def make_simple_prompt(row, space_id):
#     return f"""
# You are an architecture assistant. Based on the description below, return only the best matching activity.

# ### Outdoor space description:
# - ID: {space_id}
# - Type: {row['type']}
# - Orientation: {row['orientation']}
# - Area: {row['area']}
# - Open sides: {row['open_side']}
# - Indoor/Outdoor: {row['in_out']}
# - Compactness: {row['compactness']}

# Only return valid JSON in the following format. Do not explain or comment.
# {{
#   "parameters": {{
#     "id": "{space_id}",
#     "activity": "..."
#   }}
# }}
# """

# def make_prompt_with_reasoning(row, space_id, activity_scores, residents_summary):
#     scores_text = "\n".join([f"- {a}: {round(s, 3)}" for a, s in sorted(activity_scores.items(), key=lambda x: -x[1])])
#     return f"""
# You are an architecture assistant assigning the best outdoor activity for a given space.

# ### Outdoor space description:
# - ID: {space_id}
# - Type: {row['type']}
# - Orientation: {row['orientation']}
# - Area: {row['area']}
# - Open sides: {row['open_side']}
# - Indoor/Outdoor: {row['in_out']}
# - Compactness: {row['compactness']}

# ### Threshold-based prediction:
# {row.get('predicted_activities', 'None')}

# ### Green prediction:
# {row.get('green_prediction', 'None')}

# ### Usability prediction:
# {row.get('usability_prediction', 'None')}

# ### Nearby residents:
# {residents_summary}

# ### Voting-weighted activity preferences:
# {scores_text}

# Return your reasoning and the **best matching activity** in the following JSON format:
# ```
# {{
#   "parameters": {{
#     "id": "{space_id}",
#     "activity": "..."
#   }},
#   "reasoning": "..."
# }}
# ```
# Only output valid JSON, no commentary.
# """

# # ------------------------
# # Call LLM
# # ------------------------
# def call_local_llm(prompt):
#     response = requests.post(
#         "http://localhost:1234/v1/chat/completions",
#         headers={"Content-Type": "application/json"},
#         json={
#             "model": "local-model",
#             "messages": [
#                 {"role": "system", "content": "You only output JSON objects as requested."},
#                 {"role": "user", "content": prompt}
#             ],
#             "temperature": 0.7
#         }
#     )
#     return response.json()["choices"][0]["message"]["content"]

# # ------------------------
# # Main function
# # ------------------------
# def generate_llm_assignments(output_path="llm_assignments.json"):
#     geometries, thresh, green, usability, voting, distances, personas = load_csvs()
#     geometries, thresh, green, usability = normalize_ids([geometries, thresh, green, usability])

#     merged = geometries.copy()
#     merged = merged.merge(thresh[["id", "predicted_activities"]], on="id", how="left")
#     merged = merged.merge(green[["id", "green_prediction"]], on="id", how="left")
#     merged = merged.merge(usability[["id", "usability_prediction"]], on="id", how="left")

#     voting['resident'] = voting['resident'].astype(str)
#     personas["resident_key"] = personas["resident_key"].astype(str)

#     results = []

#     for _, row in merged.iterrows():
#         try:
#             space_id = row["id"]

#             if EXPLANATION_MODE:
#                 # Get nearest residents and weights
#                 distance_row = distances[distances["Outdoor Space"] == space_id]
#                 if distance_row.empty:
#                     raise ValueError(f"No distances for {space_id}")
#                 resident_distances = distance_row.drop(columns=["Outdoor Space"]).T
#                 resident_distances.columns = ['distance']
#                 resident_distances.index.name = 'resident_key'
#                 resident_distances.reset_index(inplace=True)
#                 top_residents = resident_distances.sort_values('distance').head(5)
#                 resident_ids = top_residents["resident_key"].tolist()
#                 residents_info = personas[personas["resident_key"].isin(resident_ids)]
#                 residents_summary = "\n".join([
#                     f"- {r['resident_key']}: {r['resident_persona']} ({r['resident_population']} people)"
#                     for _, r in residents_info.iterrows()
#                 ])

#                 activity_scores = {}
#                 for _, res in residents_info.iterrows():
#                     key = res["resident_key"]
#                     pop = float(res["resident_population"])
#                     dist = float(top_residents[top_residents["resident_key"] == key]["distance"].values[0])
#                     weight = pop / (dist + 1e-5)
#                     res_votes = voting[(voting["resident"] == key) & (voting["space"] == space_id)]
#                     for _, vote in res_votes.iterrows():
#                         act = vote["activity"]
#                         score = float(vote["weight"])
#                         activity_scores[act] = activity_scores.get(act, 0) + score * weight

#                 prompt = make_prompt_with_reasoning(row, space_id, activity_scores, residents_summary)
#             else:
#                 prompt = make_simple_prompt(row, space_id)

#             llm_response = call_local_llm(prompt)
#             json_result = json.loads(llm_response)
#             results.append(json_result)

#         except Exception as e:
#             logging.error(f"LLM failed for {row['id']}: {e}")
#             results.append({
#                 "parameters": {"id": row["id"], "activity": None},
#                 "reasoning": f"Error: {e}"
#             })

#     with open(output_path, "w", encoding="utf-8") as f:
#         json.dump(results, f, indent=2)
#     logging.info(f"LLM assignments saved to {output_path}")

#     summary = [{"space_id": r["parameters"]["id"], "assigned_activity": r["parameters"]["activity"]} for r in results]
#     pd.DataFrame(summary).to_csv("llm_activity_assignments.csv", index=False)
#     logging.info("CSV summary saved to llm_activity_assignments.csv")

# # Run
# if __name__ == "__main__":
#     generate_llm_assignments()

# logging.basicConfig(level=logging.INFO)
# # ------------------------
# # Load all CSV inputs
# # ------------------------
# def load_csvs():
#     geometries = pd.read_csv('gh_data/geometry_data_with_id.csv')
#     thresh = pd.read_csv('ml_models/threshold_predictions.csv')
#     green = pd.read_csv('ml_models/green_predictions.csv')
#     usability = pd.read_csv('ml_models/usability_predictions.csv')
#     voting = pd.read_csv('preset/voting_weights.csv')
#     distances = pd.read_csv('resident_data/resident_distances.csv')
#     personas = pd.read_csv('resident_data/personas_assigned.csv')
#     return geometries, thresh, green, usability, voting, distances, personas

# # ------------------------
# # Normalize IDs to string
# # ------------------------
# def normalize_ids(dfs):
#     for df in dfs:
#         df["id"] = df["id"].astype(str).str.strip()
#     return dfs

# # ------------------------
# # Build prompt for LLM
# # ------------------------
# def make_prompt(row, space_id, activity_scores, residents_summary):
#     scores_text = "\n".join([f"- {a}: {round(s, 3)}" for a, s in sorted(activity_scores.items(), key=lambda x: -x[1])])
#     prompt = f"""
# You are an architecture assistant assigning the best outdoor activity for a given space.
# You must return only a valid JSON response. Do not ask questions or provide explanations.

# ### Outdoor space description:
# - ID: {space_id}
# - Type: {row['type']}
# - Orientation: {row['orientation']}
# - Area: {row['area']}
# - Open sides: {row['open_side']}
# - Indoor/Outdoor: {row['in_out']}
# - Compactness: {row['compactness']}

# ### Threshold-based prediction:
# {row.get('predicted_activities', 'None')}

# ### Green prediction:
# {row.get('green_prediction', 'None')}

# ### Usability prediction:
# {row.get('usability_prediction', 'None')}

# ### Nearby residents:
# {residents_summary}

# ### Voting-weighted activity preferences:
# {scores_text}

# Respond with the best activity in this JSON format and nothing else:
# {{
#   "parameters": {{
#     "id": "{space_id}",
#     "activity": "..."
#   }},
#   "reasoning": "..."
# }}
# """
#     return prompt.strip()

# # ------------------------
# # Send prompt to local LLM (LM Studio)
# # ------------------------
# def call_local_llm(prompt):
#     try:
#         response = requests.post(
#             "http://localhost:1234/v1/chat/completions",
#             headers={"Content-Type": "application/json"},
#             json={
#                 "model": "local-model",
#                 "messages": [
#                     {"role": "system", "content": "You are a helpful assistant that returns only valid JSON and never asks questions."},
#                     {"role": "user", "content": prompt}
#                 ],
#                 "temperature": 0.4,
#                 "max_tokens": 200
#             },
#             timeout=30
#         )
#         return response.json()["choices"][0]["message"]["content"]
#     except Exception as e:
#         logging.error(f"Failed to call LLM: {e}")
#         return ""

# # ------------------------
# # Main function
# # ------------------------
# def generate_llm_assignments(output_path="llm_assignments.json"):
#     geometries, thresh, green, usability, voting, distances, personas = load_csvs()
#     geometries, thresh, green, usability = normalize_ids([geometries, thresh, green, usability])

#     merged = geometries.copy()
#     merged = merged.merge(thresh[["id", "predicted_activities"]], on="id", how="left")
#     merged = merged.merge(green[["id", "green_prediction"]], on="id", how="left")
#     merged = merged.merge(usability[["id", "usability_prediction"]], on="id", how="left")

#     if 'resident' in voting.columns:
#         voting['resident'] = voting['resident'].astype(str)
#         resident_weights = voting.groupby(['space', 'activity'])['weight'].sum().unstack(fill_value=0).reset_index()
#         resident_weights['activity_weights'] = resident_weights.drop(columns=['space']).to_dict(orient='records')
#         resident_weights = resident_weights[['space', 'activity_weights']]
#         merged = merged.merge(resident_weights, left_on="id", right_on="space", how="left")
#     else:
#         logging.warning("Voting CSV missing 'resident' column.")

#     personas["resident_key"] = personas["resident_key"].astype(str)
#     results = []

#     for _, row in merged.iterrows():
#         try:
#             space_id = row["id"]
#             for col in ["area", "type", "orientation"]:
#                 if col not in row or pd.isna(row[col]):
#                     raise KeyError(f"Missing or NaN column: {col}")

#             distance_row = distances[distances["Outdoor Space"] == space_id]
#             if distance_row.empty:
#                 raise ValueError(f"No distances found for space {space_id}")

#             resident_distances = distance_row.drop(columns=["Outdoor Space"]).T
#             resident_distances.columns = ['distance']
#             resident_distances.index.name = 'resident_key'
#             resident_distances.reset_index(inplace=True)
#             top_residents = resident_distances.sort_values('distance').head(5)

#             resident_ids = top_residents["resident_key"].tolist()
#             residents_info = personas[personas["resident_key"].isin(resident_ids)]

#             residents_summary = "\n".join([
#                 f"- {r['resident_key']}: {r['resident_persona']} ({r['resident_population']} people)"
#                 for _, r in residents_info.iterrows()
#             ])

#             activity_scores = {}
#             for _, res in residents_info.iterrows():
#                 key = res["resident_key"]
#                 pop = float(res["resident_population"])
#                 dist = float(top_residents[top_residents["resident_key"] == key]["distance"].values[0])
#                 weight = pop / (dist + 1e-5)

#                 res_votes = voting[(voting["resident"] == key) & (voting["space"] == space_id)]
#                 for _, vote in res_votes.iterrows():
#                     act = vote["activity"]
#                     score = float(vote["weight"])
#                     activity_scores[act] = activity_scores.get(act, 0) + score * weight

#             prompt = make_prompt(row, space_id, activity_scores, residents_summary)
#             llm_response = call_local_llm(prompt)
#             print(f"[DEBUG] Response for {space_id}:\n{llm_response}\n")

#             try:
#                 json_result = json.loads(llm_response)
#                 if "parameters" not in json_result or "activity" not in json_result["parameters"]:
#                     raise ValueError("Missing 'parameters' or 'activity' in response")
#             except Exception as e:
#                 logging.error(f"Invalid JSON from LLM for {space_id}: {e}")
#                 json_result = {
#                     "parameters": {"id": space_id, "activity": None},
#                     "reasoning": f"Invalid LLM output: {llm_response}"
#                 }

#             results.append(json_result)

#         except Exception as e:
#             logging.error(f"LLM failed for {row['id']}: {e}")
#             results.append({
#                 "parameters": {"id": row["id"], "activity": None},
#                 "reasoning": f"Error: {e}"
#             })

#     with open(output_path, "w", encoding="utf-8") as f:
#         json.dump(results, f, indent=2)
#     logging.info(f"LLM assignments saved to {output_path}")

#     summary = [{"space_id": r["parameters"]["id"], "assigned_activity": r["parameters"]["activity"]} for r in results]
#     pd.DataFrame(summary).to_csv("llm_activity_assignments.csv", index=False)
#     logging.info("CSV summary saved to llm_activity_assignments.csv")

# # Run
# if __name__ == "__main__":
#     generate_llm_assignments()

# Example usage
# print(ask_about_assignment("001", "Why did you pick Yoga over Play?"))


# #Load CSVs
# geometries = pd.read_csv('gh_data\geometry_data_with_id.csv')
# thresh = pd.read_csv('ml_models/threshold_predictions.csv')
# green = pd.read_csv('ml_models/green_predictions.csv')
# usability = pd.read_csv('ml_models/usability_predictions.csv')
# voting = pd.read_csv('preset/voting_weights.csv')
# distances = pd.read_csv('resident_data/resident_distances.csv')
# personas = pd.read_csv('resident_data\personas_assigned.csv')

# # Normalize IDs
# for df in [geometries, thresh, green, usability]:
#     if "id" not in df.columns:
#         df.index.name = "id"
#         df.reset_index(inplace=True)
# for df in [geometries, thresh, green, usability]:
#     df["id"] = df["id"].astype(str)


# #Merge all data
# merged = geometries.merge(thresh, on="id")
# merged = merged.merge(green[["id", "green_prediction"]], on="id")
# merged = merged.merge(usability[["id", "usability_prediction"]], on="id")
# merged = merged.merge(thresh[["id", "predicted_activities"]], on="id")

# #Group community voting preference weights
# prefs_grouped = voting.groupby("space", group_keys=False).apply(
#     lambda df: df.groupby("activity")["weight"].sum().to_dict()
# ).reset_index().rename(columns={0: "activity_weights"})

# merged = merged.merge(prefs_grouped, left_on="id", right_on="space", how="left")


# #LLM Prompt Builder
# def make_prompt(row):
#     prompt = f"""
# You are evaluating a shared outdoor space with the following characteristics:
# - Space ID: {row['id']}
# - Area: {row['area']} m², Type: {row['type']}, Orientation: {row['orientation']}, Open Sides: {row.get('open_side', 'N/A')}
# - Comfort: {row['usability_prediction']}, Green Suitability: {row['green_prediction']}
# - ML Suggests: {row['predicted_activities']}
# - Community Activity Preference Weights: {row['activity_weights']}

# - You also have access to distance data from houses to outdoor spaces in "resident_distances.csv". Each row is a resident's distance to each outdoor space.
# - You also have access to "resident_data/personas_assigned.csv" Which contains user resident_key as H1, H2, etc. ; resident_population(how many people lives in that house)
# and their assigned personas resident_persona (e.g. young_entrepreneurs etc.)

# TASK:
# Assign the most suitable activity and explain why it suits this space using all available data including comfort, green score, orientation, ML, and proximity and resident`s distances
# and residents_persona, resident`s population, residents activity weights.
# Return valid JSON like this:
# {{
#     "parameters": {{
#         "id": "{row['id']}",
#         "activity": "Flexible Space"
#     }},
#     "reasoning": "This space has high comfort, SE orientation, and strong community weight for flexible usage."
# }}
# """
#     return prompt.strip()


# # --- Generate assignment for all spaces ---
# def generate_llm_assignments(output_path="llm_assignments.json"):
#     results = []

#     for _, row in merged.iterrows():
#         try:
#             # Use the predicted activities from ML
#             predicted = ast.literal_eval(row["predicted_activities"]) if isinstance(row["predicted_activities"], str) else []

#             # Use voting weights to prioritize community interest
#             weights = row.get("activity_weights", {})
#             if isinstance(weights, str):
#                 weights = ast.literal_eval(weights)

#             # Score each predicted activity based on weight + usability + green score
#             best_activity = None
#             best_score = -float("inf")
#             for activity in predicted:
#                 weight = weights.get(activity, 0)
#                 score = (
#                     weight +
#                     float(row.get("usability_prediction", 0)) +
#                     float(row.get("green_prediction", 0))
#                 )
#                 if score > best_score:
#                     best_score = score
#                     best_activity = activity

#             if not best_activity and predicted:
#                 best_activity = predicted[0]  # fallback

#             reasoning = f"Activity '{best_activity}' chosen based on high predicted likelihood, community preference weight, and comfort (usability={row.get('usability_prediction')}) and green score ({row.get('green_prediction')})."

#             results.append({
#                 "id": row["id"],
#                 "proposed_activity": best_activity,
#                 "reasoning": reasoning
#             })

#         except Exception as e:
#             results.append({
#                 "id": row["id"],
#                 "proposed_activity": None,
#                 "reasoning": f"Error during reasoning: {e}"
#             })

#     with open(output_path, "w", encoding="utf-8") as f:
#         json.dump(results, f, indent=2)
#     print(f"✅ Local assignments saved to {output_path}")




# # --- Generate assignment for a single space ID ---
# def generate_llm_assignment_for_id(space_id, assignment_path="llm_assignments.json"):
#     """Returns the LLM assignment from precomputed file."""
#     try:
#         with open(assignment_path, "r", encoding="utf-8") as f:
#             assignments = json.load(f)
#         match = next((a for a in assignments if a["id"] == space_id), None)
#         if not match:
#             return {
#                 "id": space_id,
#                 "proposed_activity": None,
#                 "reasoning": "No assignment found for this space ID."
#             }
#         return match
#     except Exception as e:
#         return {
#             "id": space_id,
#             "proposed_activity": None,
#             "reasoning": f"Error loading assignments: {e}"
#         }


# # --- Utility: Top N spaces near resident ---
# def top_spaces_near_resident(resident_id, num=3, assignment_path="llm_assignments.json"):
#     """Returns detailed info for top N outdoor spaces closest to a resident,
#     including activity and reasoning from precomputed assignments."""

#     # Check if resident ID exists in distances
#     if resident_id not in distances.columns:
#         return {
#             "resident_id": resident_id,
#             "error": "Resident ID not found in distance data."
#         }

#     # Load assignments
#     try:
#         with open(assignment_path, "r", encoding="utf-8") as f:
#             assignments = json.load(f)
#         assignments_dict = {a["id"]: a for a in assignments}
#     except Exception as e:
#         return {
#             "resident_id": resident_id,
#             "error": f"Failed to load LLM assignments: {e}"
#         }

#     # Get closest spaces
#     near_df = distances[["Outdoor Space", resident_id]].dropna().sort_values(by=resident_id)
#     top_spaces = near_df.head(num)["Outdoor Space"].tolist()

#     # Build results
#     results = []
#     for space_id in top_spaces:
#         space_row = merged[merged["id"] == space_id]
#         if space_row.empty:
#             results.append({
#                 "id": space_id,
#                 "proposed_activity": None,
#                 "reasoning": "No data found for this space."
#             })
#             continue

#         assignment = assignments_dict.get(space_id)
#         if not assignment:
#             results.append({
#                 "id": space_id,
#                 "proposed_activity": None,
#                 "reasoning": "No LLM assignment found for this space."
#             })
#             continue

#         # You can use make_prompt(space_row.iloc[0]) for deeper context if needed
#         results.append({
#             "id": space_id,
#             "proposed_activity": assignment["proposed_activity"],
#             "reasoning": assignment["reasoning"]
#         })

#     return {
#         "resident_id": resident_id,
#         "top_spaces": results
#     }



# # --- CLI entry ---
# if __name__ == "__main__":
#     generate_llm_assignments()








def read_activity_weights(csv_path=r'C:\Users\nseda\Documents\GitHub\aia25-studio-agent\user_activiy_weights.csv'):
    """Reads CSV and returns a dict mapping profiles to activity weight dicts."""
    profile_map = {}
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            profile = row['profiles'].strip()
            weights = ast.literal_eval(row['activity_weights'])
            profile_map[profile] = weights
    return profile_map

def assign_activity(message, user_profile="young_entrepreneurs"):
    weights_map = read_activity_weights()
    weights = weights_map.get(user_profile, {})

    # Format the weights as a readable string for LLM
    weights_text = ", ".join([f"{k}: {v}" for k, v in weights.items()])

    # Construct full user input message
    full_message = f"""
    USER PROFILE: {user_profile}
    ACTIVITY WEIGHTS: {weights_text}

    USER REQUEST:
    {message}
    """

    # Send to LLM
    response = client.chat.completions.create(
        model=completion_model,
        messages=[
            {
                "role": "system",
                "content": """
Below I describe how you need to assign activities based on user input:

- **Possible activities**:
Sitting, Offline Retreat, Sunbath, Healing Garden, Playground, Sports, Outdoor Cinema/Event Space, Community Pool/BBQ, Flexible Space, Creative Corridor, Outdoor Meeting Room, Green Corridor, Biodiversity balcony, Urban Agriculture Garden, Viewpoint, Storage & Technical Space

### Your task:
Based on a user's design request, profile, and activity weights, assign **one activity** and explain your reasoning.

### Format your response like this:
{
  "parameters": {
    "activity": "Flexible Space"
  },
  "reasoning": "Explain your choice based on the weights and design intent."
}

Only return a valid JSON object. No markdown, no commentary.
"""
            },
            {
                "role": "user",
                "content": full_message
            }
        ]
    )
    return response.choices[0].message.content
