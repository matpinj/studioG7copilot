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
from functools import lru_cache
import time

# THIS DEFINITION SET IS FOR LLM REASONING ENGINE TO ASSIGN ACTIVITIES TO OUTDOOR SPACES
# It uses a local LLM server to generate assignments based on activity space data and resident preferences.
EXPLANATION_MODE = True
# True: Answers single query in detail
# False: Assigns activities in a file based on llm reasoning
logging.basicConfig(level=logging.INFO)

# ============================================================================
# CACHING AND OPTIMIZATION
# ============================================================================

# Global cache for data to avoid reloading
_reasoning_cache = {}
_cache_initialized = False

def initialize_reasoning_cache():
    """Initialize cache for reasoning functions - called once"""
    global _reasoning_cache, _cache_initialized
    
    if _cache_initialized:
        return
        
    print("🧠 Initializing reasoning cache...")
    start_time = time.time()
    
    try:
        # Load from database first
        conn = sqlite3.connect('sql/gh_data.db')
        
        # Load activity space geometries
        geometries = pd.read_sql_query("SELECT * FROM activity_space", conn)
        geometries.rename(columns={"key": "id"}, inplace=True)
        geometries["id"] = geometries["id"].apply(lambda x: f"O{x}" if not str(x).startswith("O") else str(x))
        
        # Load personas
        personas = pd.read_sql_query("SELECT * FROM personas_assigned", conn)
        personas["resident_key"] = personas["resident_key"].astype(str).str.strip()
        
        # Load resident distances
        distances = pd.read_sql_query("SELECT * FROM resident_distances", conn)
        if "Outdoor Space" in distances.columns:
            distances = distances.rename(columns={"Outdoor Space": "id"})
        
        conn.close()
        
        # Load CSV data
        multi = pd.read_csv('gh_data/gh_data_multiple_activities.csv')
        multi = multi.rename(columns={'key': 'id'})
        thresh = multi[['id', 'activity']].rename(columns={'activity': 'predicted_activities'})
        green = multi[['id', 'green_suitability']].rename(columns={'green_suitability': 'green_prediction'})
        usability = multi[['id', 'usability']].rename(columns={'usability': 'usability_prediction'})
        
        # Load voting weights
        voting = pd.read_csv('resident_data/voting_weights.csv')
        
        # Store in cache
        _reasoning_cache['geometries'] = geometries
        _reasoning_cache['thresh'] = thresh
        _reasoning_cache['green'] = green
        _reasoning_cache['usability'] = usability
        _reasoning_cache['voting'] = voting
        _reasoning_cache['distances'] = distances
        _reasoning_cache['personas'] = personas
        
        # Create lookup dictionaries for faster access
        _reasoning_cache['thresh_lookup'] = thresh.set_index('id')['predicted_activities'].to_dict()
        _reasoning_cache['green_lookup'] = green.set_index('id')['green_prediction'].to_dict()
        _reasoning_cache['usability_lookup'] = usability.set_index('id')['usability_prediction'].to_dict()
        
        _cache_initialized = True
        print(f"✅ Reasoning cache initialized in {time.time() - start_time:.2f}s")
        
    except Exception as e:
        print(f"❌ Error initializing reasoning cache: {e}")
        raise

def get_reasoning_data(key):
    """Get cached reasoning data"""
    if not _cache_initialized:
        initialize_reasoning_cache()
    return _reasoning_cache.get(key)

@lru_cache(maxsize=128)
def get_space_geometry(space_id):
    """Cached lookup for space geometry"""
    geometries = get_reasoning_data('geometries')
    if geometries is None:
        return None
    row = geometries[geometries["id"] == space_id]
    return row.iloc[0] if not row.empty else None

@lru_cache(maxsize=128)
def get_space_distances(space_id):
    """Cached lookup for space distances"""
    distances = get_reasoning_data('distances')
    if distances is None:
        return None
    distance_row = distances[distances["id"] == space_id]
    return distance_row if not distance_row.empty else None

# ============================================================================
# OPTIMIZED UTILITY FUNCTIONS
# ============================================================================

def clean_llm_json(text):
    """Clean LLM response to extract JSON"""
    # Remove markdown code block markers if present
    text = re.sub(r"```json|```", "", text).strip()
    # Try to extract the first {...} block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    return text  # fallback

def load_csvs():
    """Legacy function for backward compatibility - now uses cache"""
    if not _cache_initialized:
        initialize_reasoning_cache()
        
    return (
        get_reasoning_data('geometries'),
        get_reasoning_data('thresh'),
        get_reasoning_data('green'),
        get_reasoning_data('usability'),
        get_reasoning_data('voting'),
        get_reasoning_data('distances'),
        get_reasoning_data('personas')
    )

def normalize_ids(dfs):
    """Normalize IDs in dataframes"""
    for df in dfs:
        df["id"] = df["id"].astype(str)
    return dfs

def make_prompt(row, space_id, activity_scores, residents_summary):
    """Create optimized prompt for LLM"""
    # Limit activity scores to top 5 for cleaner prompt
    top_scores = sorted(activity_scores.items(), key=lambda x: -x[1])[:5]
    scores_text = "\n".join([f"- {a}: {round(s, 3)}" for a, s in top_scores])

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

### Available activities (choose ONLY from this list):
{row.get('predicted_activities', 'None')}

### Green suitability: {row.get('green_prediction', 'None')}
### Usability score: {row.get('usability_prediction', 'None')}

### Nearby residents: {residents_summary}

### Top activity preferences:
{scores_text}

Return your reasoning and the **best matching activity** in JSON format:
```json
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
```json
{{
  "parameters": {{
    "id": "{space_id}",
    "activity": "..."
  }}
}}
```"""

def call_local_llm(prompt):
    """Optimized LLM call with better error handling"""
    try:
        response = requests.post(
            "http://localhost:1234/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json={
                "model": "local-model",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 300  # Limit tokens for faster response
            },
            timeout=15  # Reduced timeout
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f'{{"parameters": {{"activity": null}}, "reasoning": "LLM HTTP error: {response.status_code}"}}'
            
    except requests.exceptions.Timeout:
        return '{"parameters": {"activity": null}, "reasoning": "LLM request timed out"}'
    except Exception as e:
        return f'{{"parameters": {{"activity": null}}, "reasoning": "LLM error: {str(e)}"}}'

# ============================================================================
# OPTIMIZED MAIN FUNCTIONS
# ============================================================================

def generate_llm_assignments(output_path=None):
    """Generate LLM assignments with optimized data loading"""
    if not _cache_initialized:
        initialize_reasoning_cache()
        
    # Set output directory and file paths
    output_dir = os.path.join(os.path.dirname(__file__), "llm_reasoning")
    os.makedirs(output_dir, exist_ok=True)
    if output_path is None:
        output_path = os.path.join(output_dir, "llm_assignments.json")
    csv_path = os.path.join(output_dir, "llm_activity_assignments.csv")

    # Get cached data
    geometries = get_reasoning_data('geometries')
    thresh = get_reasoning_data('thresh')
    green = get_reasoning_data('green')
    usability = get_reasoning_data('usability')
    voting = get_reasoning_data('voting')
    distances = get_reasoning_data('distances')
    personas = get_reasoning_data('personas')

    geometries, thresh, green, usability = normalize_ids([geometries, thresh, green, usability])

    # Merge data efficiently
    merged = geometries.copy()
    merged = merged.merge(thresh[["id", "predicted_activities"]], on="id", how="left")
    merged = merged.merge(green[["id", "green_prediction"]], on="id", how="left")
    merged = merged.merge(usability[["id", "usability_prediction"]], on="id", how="left")

    # Process voting weights
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

    # Get lookup dictionaries from cache for faster processing
    thresh_lookup = get_reasoning_data('thresh_lookup')
    green_lookup = get_reasoning_data('green_lookup')
    usability_lookup = get_reasoning_data('usability_lookup')

    print(f"🔄 Processing {len(merged)} spaces...")
    
    for idx, row in merged.iterrows():
        if idx % 10 == 0:
            print(f"  Processed {idx}/{len(merged)} spaces...")
            
        try:
            space_id = row["id"]
            
            # Validate required columns
            for col in ["area", "type", "orientation"]:
                if col not in row or pd.isna(row[col]):
                    raise KeyError(f"Missing or NaN column: {col}")

            # Get distances using cached lookup
            distance_data = get_space_distances(space_id)
            if distance_data is None:
                raise ValueError(f"No distances found for space {space_id}")

            # Process resident distances efficiently
            resident_distances = distance_data.drop(columns=["id"]).T
            resident_distances.columns = ['distance']
            resident_distances.index.name = 'resident_key'
            resident_distances.reset_index(inplace=True)
            top_residents = resident_distances.sort_values('distance').head(5)

            # Get resident info
            resident_ids = top_residents["resident_key"].tolist()
            residents_info = personas[personas["resident_key"].isin(resident_ids)]

            # Create concise residents summary
            residents_summary = "; ".join([
                f"{r['resident_key']}: {r['resident_persona']} ({r['resident_population']}p)"
                for _, r in residents_info.iterrows()
            ])

            # Calculate activity scores efficiently
            activity_scores = {}
            for _, res in residents_info.iterrows():
                key = res["resident_key"]
                pop = float(res["resident_population"])
                res_votes = voting[(voting["resident"] == key) & (voting["space"] == space_id)]
                for _, vote in res_votes.iterrows():
                    act = vote["activity"]
                    score = float(vote["weight"]) * pop
                    activity_scores[act] = activity_scores.get(act, 0) + score

            # Create and send prompt
            prompt = make_prompt(row, space_id, activity_scores, residents_summary)
            llm_response = call_local_llm(prompt)

            # Process LLM response
            try:
                llm_response_clean = clean_llm_json(llm_response)
                json_result = json.loads(llm_response_clean)
                params = json_result.get("parameters", {})
                if not isinstance(params.get("id", None), str):
                    params["id"] = str(space_id)
                activity = params.get("activity", None)
                if activity is not None and not isinstance(activity, str):
                    params["activity"] = str(activity)
                json_result["parameters"] = params
                if "reasoning" not in json_result:
                    json_result["reasoning"] = ""
                results.append(json_result)
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

    # Save results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logging.info(f"LLM assignments saved to {output_path}")

    summary = [{"space_id": r["parameters"]["id"], "assigned_activity": r["parameters"]["activity"]} for r in results]
    pd.DataFrame(summary).to_csv(csv_path, index=False)
    logging.info(f"CSV summary saved to {csv_path}")

def explain_activity_for_space(space_id, question, geometries=None, thresh=None, green=None, usability=None, voting=None, distances=None, personas=None, assignments_path="llm_reasoning/llm_activity_assignments.csv"):
    """Optimized explanation function with caching"""
    
    # Initialize cache if not already done
    if not _cache_initialized:
        initialize_reasoning_cache()
    
    # Use cached data if parameters are None
    if geometries is None:
        geometries = get_reasoning_data('geometries')
    if thresh is None:
        thresh = get_reasoning_data('thresh')
    if green is None:
        green = get_reasoning_data('green')
    if usability is None:
        usability = get_reasoning_data('usability')
    if voting is None:
        voting = get_reasoning_data('voting')
    if distances is None:
        distances = get_reasoning_data('distances')
    if personas is None:
        personas = get_reasoning_data('personas')
    
    # Load assignments and get assigned activity for this space
    assignments = pd.read_csv(assignments_path)
    assignments['space_id'] = assignments['space_id'].astype(str).str.strip()
    space_id = str(space_id).strip()
    assigned_row = assignments[assignments['space_id'] == space_id]
    if assigned_row.empty:
        return f"No assigned activity found for space {space_id}."
    activity = assigned_row.iloc[0]['assigned_activity']

    # Get space geometry using cached lookup
    row = get_space_geometry(space_id)
    if row is None:
        return f"No data for space {space_id}"

    # Use cached lookups for predictions
    thresh_lookup = get_reasoning_data('thresh_lookup')
    green_lookup = get_reasoning_data('green_lookup')
    usability_lookup = get_reasoning_data('usability_lookup')
    
    row = row.copy()  # Make a copy to avoid modifying cached data
    row['predicted_activities'] = thresh_lookup.get(space_id, 'None')
    row['green_prediction'] = green_lookup.get(space_id, 'None')
    row['usability_prediction'] = usability_lookup.get(space_id, 'None')

    # Get resident distances using cached lookup
    distance_data = get_space_distances(space_id)
    if distance_data is None:
        return f"No distances found for space {space_id}"
        
    resident_distances = distance_data.drop(columns=["id"]).T
    resident_distances.columns = ['distance']
    resident_distances.index.name = 'resident_key'
    resident_distances.reset_index(inplace=True)
    top_residents = resident_distances.sort_values('distance').head(5)
    
    # Get resident info
    resident_ids = top_residents["resident_key"].tolist()
    residents_info = personas[personas["resident_key"].isin(resident_ids)]

    residents_summary = "\n".join([
        f"- {r['resident_key']}: {r['resident_persona']} ({r['resident_population']} people)"
        for _, r in residents_info.iterrows()
    ])

    # Calculate voting-weighted activity preferences efficiently
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

    # Create optimized prompt focused on explanation
    top_scores = sorted(activity_scores.items(), key=lambda x: -x[1])[:5]
    scores_text = "\n".join([f"- {a}: {round(s, 3)}" for a, s in top_scores])

    prompt = f"""
You are an architecture assistant. Explain why "{activity}" is assigned to outdoor space {space_id}.

### User question: {question}

### Space details:
- ID: {space_id}
- Type: {row['type']}, Orientation: {row['orientation']}, Area: {row['area']}
- Open sides: {row['open_side']}, Compactness: {row['compactness']}

### Predictions:
- Available activities: {row.get('predicted_activities', 'None')}
- Green suitability: {row.get('green_prediction', 'None')}
- Usability: {row.get('usability_prediction', 'None')}

### Nearby residents: {residents_summary}

### Top activity preferences: {scores_text}

Explain conversationally why "{activity}" is the best fit for this space, considering the user's question.
"""

    return call_local_llm(prompt)

def answer_general_space_question(house_key, question, geometries=None, thresh=None, green=None, usability=None, voting=None, distances=None, personas=None, assignments_path="llm_reasoning/llm_activity_assignments.csv"):
    """Optimized general space question answering with caching"""
    
    # Initialize cache if not already done
    if not _cache_initialized:
        initialize_reasoning_cache()
    
    # Use cached data if parameters are None
    if geometries is None:
        geometries = get_reasoning_data('geometries')
    if thresh is None:
        thresh = get_reasoning_data('thresh')
    if green is None:
        green = get_reasoning_data('green')
    if usability is None:
        usability = get_reasoning_data('usability')
    if voting is None:
        voting = get_reasoning_data('voting')
    if distances is None:
        distances = get_reasoning_data('distances')
    if personas is None:
        personas = get_reasoning_data('personas')

    # Detect if the question is about a specific activity
    activity_keywords = ["sports", "playground", "sunbath", "social", "recreation", "exercise"]
    activity_mentioned = None
    for act in activity_keywords:
        if act.lower() in question.lower():
            activity_mentioned = act
            break

    assignments = pd.read_csv(assignments_path)
    assignments['space_id'] = assignments['space_id'].astype(str).str.strip()
    assignments['assigned_activity'] = assignments['assigned_activity'].astype(str).str.strip()

    if activity_mentioned:
        filtered = assignments[assignments['assigned_activity'].str.lower() == activity_mentioned.lower()]
        if filtered.empty:
            return f"No outdoor spaces are assigned to {activity_mentioned}."
        space_list = filtered['space_id'].tolist()
        return f"The following outdoor spaces are assigned to {activity_mentioned}: {', '.join(space_list)}"

    # Find nearest 5 outdoor spaces
    if house_key not in distances.columns:
        return f"No distances found for house key {house_key}."

    nearby = distances[["id", house_key]].rename(columns={house_key: "distance", "id": "space_id"})
    nearby = nearby.sort_values("distance").head(5)

    # Get cached lookups for faster processing
    thresh_lookup = get_reasoning_data('thresh_lookup')
    green_lookup = get_reasoning_data('green_lookup')
    usability_lookup = get_reasoning_data('usability_lookup')

    # Process spaces efficiently
    space_summaries = []
    for _, row in nearby.iterrows():
        space_id = row['space_id']
        distance = row['distance']

        # Get assigned activity
        assigned_row = assignments[assignments['space_id'] == str(space_id).strip()]
        assigned_activity = assigned_row.iloc[0]['assigned_activity'] if not assigned_row.empty else "Unknown"
        
        # Get geometry info using cached lookup
        geo_data = get_space_geometry(space_id)
        if geo_data is not None:
            type_ = geo_data.get("type", "Unknown")
            orientation = geo_data.get("orientation", "Unknown")
            area = geo_data.get("area", "Unknown")
        else:
            type_ = orientation = area = "Unknown"

        # Get predictions using cached lookups
        pred_thresh = thresh_lookup.get(space_id, "None")
        pred_green = green_lookup.get(space_id, "None")
        pred_usability = usability_lookup.get(space_id, "None")

        # Get closest residents efficiently
        distance_data = get_space_distances(space_id)
        if distance_data is not None:
            resident_distances = distance_data.drop(columns=["id"]).T
            resident_distances.columns = ['distance']
            resident_distances.index.name = 'resident_key'
            resident_distances.reset_index(inplace=True)
            top_residents = resident_distances.sort_values('distance').head(3)  # Reduced to 3 for speed
            resident_ids = top_residents["resident_key"].tolist()
            residents_info = personas[personas["resident_key"].isin(resident_ids)]
            residents_summary = "; ".join([
                f"{r['resident_key']}: {r['resident_persona']}"
                for _, r in residents_info.iterrows()
            ])
        else:
            residents_summary = "No resident data"

        # Simplified voting summary (top 2 activities only)
        votes = voting[voting['space'] == space_id]
        if not votes.empty:
            top_votes = votes.groupby('activity')['weight'].sum().sort_values(ascending=False).head(2)
            voting_summary = "; ".join([f"{act}: {w:.1f}" for act, w in top_votes.items()])
        else:
            voting_summary = "No voting data"

        # Compose concise summary for this space
        space_summaries.append(
            f"- {space_id} ({assigned_activity}, {distance:.1f}m)\n"
            f"  {type_}, {orientation}, Area: {area}\n"
            f"  Predictions: {pred_thresh} | Green: {pred_green}\n"
            f"  Residents: {residents_summary}\n"
            f"  Top votes: {voting_summary}"
        )

    space_summaries_text = "\n\n".join(space_summaries)

    # Create concise prompt for natural conversation
    prompt = f"""
You are a friendly community advisor helping resident {house_key}.

Question: {question}

Nearby spaces:
{space_summaries_text}

Answer naturally and conversationally, focusing on what's most relevant to their question. Be concise and helpful.
"""

    return call_local_llm(prompt)

@lru_cache(maxsize=64)
def get_spaces_with_assigned_activity(activity_name, assignments_path="llm_reasoning/llm_activity_assignments.csv"):
    """Cached lookup for spaces with specific activities"""
    assignments = pd.read_csv(assignments_path)
    assignments['assigned_activity'] = assignments['assigned_activity'].astype(str).str.strip()
    filtered = assignments[assignments['assigned_activity'].str.lower() == activity_name.lower()]
    return filtered['space_id'].tolist()

# ============================================================================
# LEGACY COMPATIBILITY
# ============================================================================

# This section maintains backward compatibility with the original code
if __name__ == "__main__":
    # Initialize cache
    initialize_reasoning_cache()
    
    # Run original debugging code
    geometries, thresh, green, usability, voting, distances, personas = load_csvs()
    print("geometries['id'] sample:", geometries['id'].head(10).tolist())
    print("thresh['id'] sample:", thresh['id'].head(10).tolist())
    print("distances['id'] sample:", distances['id'].head(10).tolist())
    print("Missing in distances:", set(geometries['id']) - set(distances['id']))
    
    # Uncomment to generate assignments
    # generate_llm_assignments()