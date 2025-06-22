from server.config import *
import json
import pandas as pd
import logging
import sqlite3


# This module handles the negotiation process using LLMs to suggest actions based on user requests.

# Data loading utility
def load_csvs():
    conn = sqlite3.connect('sql/gh_data.db')
    geometries = pd.read_sql_query("SELECT * FROM activity_space", conn)
    geometries.rename(columns={"key": "id"}, inplace=True)
    conn.close()
    thresh = pd.read_csv('ml_models/threshold_predictions.csv')
    green = pd.read_csv('ml_models/green_predictions.csv')
    usability = pd.read_csv('ml_models/usability_predictions.csv')
    voting = pd.read_csv('resident_data/voting_weights.csv')
    conn = sqlite3.connect('sql/gh_data.db')
    distances = pd.read_sql_query("SELECT * FROM resident_distances", conn)
    distances.rename(columns={"Outdoor Space": "id"}, inplace=True)
    conn.close()
    conn = sqlite3.connect('sql/gh_data.db')
    personas = pd.read_sql_query("SELECT * FROM personas_assigned", conn)
    conn.close()
    return geometries, thresh, green, usability, voting, distances, personas

# ACTIONS LIST 

def change_geometry(params):
    """Suggest a new geometry for a given space."""
    geometries, *_ = load_csvs()
    outdoor_id = params.get("outdoor_id") or params.get("id")
    if not outdoor_id:
        return {"error": "No outdoor_id provided."}
    space = geometries[geometries["id"] == outdoor_id]
    if space.empty:
        return {"error": f"No space found with id {outdoor_id}."}
    area = space.iloc[0]["area"]
    new_area = area * 1.1
    return {
        "result": f"Suggested new area for space {outdoor_id}: {new_area:.2f} (was {area})",
        "old_area": area,
        "new_area": new_area,
        "params": params
    }

def get_nearby_activities(params):
    """Return the 10 nearest outdoor spaces for a user, including assigned activity, area, and distance."""
    geometries, _, _, _, voting, distances, personas = load_csvs()
    user_id = params.get("user_id")
    if not user_id:
        return {"error": "No user_id provided."}
    if user_id not in distances.columns:
        return {"error": f"No distances found for user {user_id}."}
    # Load assignments for assigned activities
    try:
        assignments = pd.read_csv('llm_reasoning/llm_activity_assignments.csv')
    except Exception:
        assignments = None
    # Get 10 nearest spaces
    nearby = distances[["id", user_id]].rename(columns={user_id: "distance"})
    nearby = nearby.sort_values("distance").head(10)
    results = []
    for _, row in nearby.iterrows():
        space_id = row["id"]
        dist = row["distance"]
        # Get area from geometries
        space_row = geometries[geometries["id"] == space_id]
        area = float(space_row["area"].iloc[0]) if not space_row.empty else None
        # Get assigned activity from assignments
        assigned_activity = None
        if assignments is not None:
            assign_row = assignments[assignments["space_id"] == space_id]
            if not assign_row.empty:
                assigned_activity = assign_row.iloc[0]["assigned_activity"]
        # Fix: convert NaN to None for JSON
        if pd.isna(assigned_activity):
            assigned_activity = None
        results.append({
            "space_id": space_id,
            "distance": dist,
            "area": area,
            "assigned_activity": assigned_activity
        })
    return {"result": results, "params": params}

def propose_activity_change(params):
    """Suggest negotiation with other residents for activity change."""
    # Add: check voting, find residents, check weights, check geometry suitability, explain process
    user_id = params.get("user_id")
    space_id = params.get("space_id")
    desired = params.get("desired_activity")
    current = params.get("current_activity")
    if not user_id or not desired or not current or not space_id:
        return {"error": "Missing user_id, space_id, desired_activity, or current_activity."}
    # Clean up activity names
    def clean_activity_name(name):
        if not name:
            return ""
        name = name.strip()
        if name.lower().startswith("for "):
            name = name[4:]
        return name.strip().title()
    desired_clean = clean_activity_name(desired)
    current_clean = clean_activity_name(current)
    # Analyze voting
    _, _, _, _, voting, _, _ = load_csvs()
    print(f"[DEBUG] Loaded voting file shape: {voting.shape} from resident_data/voting_weights.csv")
    bbq_voters = voting[(voting["space"] == space_id) & (voting["activity"].str.strip().str.title() == current_clean)]
    sports_voters = voting[(voting["space"] == space_id) & (voting["activity"].str.strip().str.title() == desired_clean)]
    # Find residents who like both
    bbq_residents = set(bbq_voters["resident"])
    sports_residents = set(sports_voters["resident"])
    overlap = bbq_residents & sports_residents
    explanation = f"Residents who voted for {current_clean} in {space_id}: {list(bbq_residents)}. Residents who voted for {desired_clean}: {list(sports_residents)}. Overlap: {list(overlap)}."
    # Check geometry suitability (dummy logic)
    geometries, *_ = load_csvs()
    space = geometries[geometries["id"] == space_id]
    if not space.empty:
        area = space.iloc[0]["area"]
        explanation += f" Current area: {area}. (Dummy: check if suitable for {desired_clean}.)"
    # Decision
    if overlap:
        explanation += f" Activity change is possible. Do you want to proceed?"
        return {"result": explanation, "can_proceed": True, "params": params}
    else:
        explanation += f" No overlap found. Activity change not possible."
        return {"result": explanation, "can_proceed": False, "params": params}

def find_profile_swap(params):
    """Suggest possible apartment (house) swaps based on preferences and proximity to desired activities."""
    user_id = params.get("user_id")
    desired = params.get("desired_activity", "Sports")
    geometries, _, _, _, voting, distances, personas = load_csvs()
    try:
        assignments = pd.read_csv('llm_reasoning/llm_activity_assignments.csv')
    except Exception:
        return {"error": "Could not load activity assignments."}

    # 1. Find all outdoor spaces assigned to the desired activity
    spaces_with_desired = assignments[assignments['assigned_activity'].str.lower() == desired.lower()]
    swap_candidates = []

    # 2. For each such space, find the closest resident (excluding the requesting user)
    for _, space_row in spaces_with_desired.iterrows():
        space_id = space_row['space_id']
        # Find closest resident to this space
        if space_id in distances['id'].values:
            dists_row = distances[distances['id'] == space_id]
            if not dists_row.empty:
                dists_row = dists_row.iloc[0]
                # All residents except 'id' and the requesting user
                resident_dists = [(col, dists_row[col]) for col in distances.columns if col != 'id' and col != user_id]
                resident_dists = sorted(resident_dists, key=lambda x: x[1])
                for resident, dist in resident_dists[:3]:  # Only top 3 closest
                    # Find the resident's current space (the one they're closest to)
                    if resident in distances.columns:
                        closest_space_row = distances[['id', resident]].sort_values(resident).head(1)
                        if not closest_space_row.empty:
                            their_space = closest_space_row.iloc[0]['id']
                            # What activity is assigned to their space?
                            their_activity = None
                            row = assignments[assignments['space_id'] == their_space]
                            if not row.empty:
                                their_activity = row.iloc[0]['assigned_activity']
                            # What are their top preferences?
                            their_votes = voting[voting['resident'] == resident]
                            top_their_prefs = their_votes.groupby('activity')['weight'].sum().sort_values(ascending=False)
                            # What are your top preferences?
                            your_votes = voting[voting['resident'] == user_id]
                            top_your_prefs = your_votes.groupby('activity')['weight'].sum().sort_values(ascending=False)
                            # If the resident's top preference matches your current nearby activity, and your top matches theirs, suggest swap
                            if not top_their_prefs.empty and not top_your_prefs.empty:
                                their_top = top_their_prefs.index[0]
                                your_top = top_your_prefs.index[0]
                                if (their_top.lower() == their_activity.lower() and
                                    your_top.lower() == desired.lower()):
                                    swap_candidates.append({
                                        "swap_with": resident,
                                        "their_apartment": resident,
                                        "their_current_space": their_space,
                                        "their_current_activity": their_activity,
                                        "distance_to_desired_space": dist,
                                        "your_top_preference": your_top,
                                        "their_top_preference": their_top,
                                        "desired_space": space_id,
                                        "desired_activity": desired
                                    })
    if swap_candidates:
        explanation = "Possible apartment swaps found:\n"
        for cand in swap_candidates:
            explanation += (
                f"- Swap with {cand['swap_with']} (currently near {cand['their_current_space']} assigned as {cand['their_current_activity']}). "
                f"Your top preference: {cand['your_top_preference']}, their top: {cand['their_top_preference']}. "
                f"Distance to desired space: {cand['distance_to_desired_space']:.2f}m.\n"
            )
        return {"result": explanation, "params": params, "candidates": swap_candidates}
    else:
        return {"result": "No suitable apartment swaps found based on current assignments and preferences.", "params": params}

def process_booking(params):
    """Book an activity for a user."""
    user_id = params.get("user_id")
    desired = params.get("desired_activity")
    space_id = params.get("space_id")
    # Dummy: check if slot is available (always true)
    if not user_id or not desired or not space_id:
        return {"error": "Missing user_id, desired_activity, or space_id."}
    explanation = f"Booked {desired} in {space_id} for user {user_id}. (Dummy: slot available, booking confirmed.)\nDo you want to finalize this booking?"
    return {"result": explanation, "can_proceed": True, "params": params}

def summarize_preferences(params):
    """Summarize a user's preferences."""
    user_id = params.get("user_id")
    if not user_id:
        return {"error": "No user_id provided."}
    _, _, _, _, voting, _, _ = load_csvs()
    user_votes = voting[voting["resident"] == user_id]
    summary = user_votes.groupby("activity")["weight"].sum().sort_values(ascending=False).to_dict()
    return {
        "result": f"Summary of preferences for user {user_id}: {summary}",
        "params": params
    }

def assign_activity(params):
    """
    Assigns an activity to a space based on the parameters provided.
    params: dict, contains 'id' or 'space_id' and 'activity' to assign.
    Returns a confirmation message.
    """
    space_id = params.get("space_id") or params.get("id")
    activity = params.get("activity")
    if not space_id or not activity:
        return {"error": "Missing space_id or activity."}
    # Example: Just return a confirmation
    return {
        "result": f"Activity '{activity}' assigned to space '{space_id}'!",
        "params": params
    }



#ACTIONS DICTIONARY
ACTION_DISPATCHER = {
    "change_geometry": change_geometry,
    "get_nearby_activities": get_nearby_activities,
    "propose_activity_change": propose_activity_change,
    "find_profile_swap": find_profile_swap,
    "process_booking": process_booking,
    "summarize_preferences": summarize_preferences,
    "assign_activity": assign_activity,
}

#ROUTE FOR SUGGESTING ACTIONS

def route_action(llm_json):
    """
    Routes the action(s) suggested by the LLM to the appropriate function(s).
    llm_json: dict, parsed from LLM output.
    Returns a list of results or a single result.
    """
    results = []
    # Handle single action
    if "action" in llm_json:
        action = llm_json["action"]
        params = llm_json.get("parameters", llm_json)
        func = ACTION_DISPATCHER.get(action)
        if func:
            results.append(func(params))
        else:
            results.append({"error": f"Unknown action: {action}"})
    # Handle multiple actions
    elif "actions" in llm_json:
        params = llm_json.get("parameters", llm_json)
        for action in llm_json["actions"]:
            func = ACTION_DISPATCHER.get(action)
            if func:
                results.append(func(params))
            else:
                results.append({"error": f"Unknown action: {action}"})
    else:
        results.append({"error": "No action found in LLM response."})
    return results if len(results) > 1 else results[0]

# --- Example usage ---

if __name__ == "__main__":
    # Example LLM output (as string)
    llm_response = '''
    {
        "action": "change_geometry",
        "outdoor_id": "O1",
        "reasoning": "User requested to modify the geometry of O1."
    }
    '''
    llm_json = json.loads(llm_response)
    result = route_action(llm_json)
    print(result)

    # Example with multiple actions
    llm_response_multi = '''
    {
        "actions": ["get_nearby_activities"],
        "user_id": "H5"
    }
    '''
    llm_json_multi = json.loads(llm_response_multi)
    result_multi = route_action(llm_json_multi)
    print(result_multi)
    
def suggest_actions_from_request(message):
    response = client.chat.completions.create(
        model=completion_model,
        messages=[
            {
                "role": "system",
                "content": """
You are an assistant that interprets user requests and suggests high-level actions for a smart architecture system.

Given the user's request, output a JSON object with an "action" field (or "actions" if multiple) and any other necessary parameters.

Use only these action names:
- "change_geometry": For requests about changing, enlarging, or modifying a space's physical properties.
- "get_nearby_activities": For requests about what activities are available nearby, their distance, size, and details.
- "propose_activity_change": For requests about changing an activity in a space, or negotiating with other residents.
- "find_profile_swap": For requests about swapping/switching apartments/houses with someone whose preferences match better.
- "process_booking": For requests about booking a space or activity for a certain time.
- "assign_activity": For confirming or finalizing an activity assignment.
- "summarize_preferences": For summarizing the user's activity or space preferences.

Respond ONLY with valid JSON. No extra text.

### Examples:

User: "I would like to have a larger space, what do you suggest?"
Response:
{
  "action": "change_geometry"
}

User: "Suggest me what are the activities around me that I can enjoy better, how far they are to my house, how big they are and what are their activities?"
Response:
{
  "action": "get_nearby_activities",
    "parameters": {
    "user_id": "H5",  # Example user ID
    "current_activity": "Viewpoint",  # Example current activity
    "desired_activity": ["Sports", "Playground"],  # Example activities
    "distances": ["20m", "50m"],  # Example distances
    "sizes": ["100m²", "200m²"],  # Example sizes
    },
  "reasoning": "User wants to know about nearby activities, their distance, size, and type."
}

User: "I'm insistent to have activity X instead of activity Y, what do you suggest me to do?"
Response:
{
  "action": "propose_activity_change",
  "parameters": {
    "user_id": "H5",  # Example user ID
    "current_activity": "Sunbath",    # Example current activity
    "desired_activity": "Viewpoint",  # Example desired activity
  },
  "reasoning": "User wants to change the activity in their space."
}

User: "I want this activity and bigger area and south sun and more green. Is there any other people who would like to have my apartment and we can swap our houses?"
Response:
{
  "action": "find_profile_swap",
  "parameters": {
    "user_id": "H5",  # Example user ID
    "potential_swap_ids": ["H6", "H7"],  # Example potential swap user IDs
    "potential_swap_activities": ["Sunbath", "Green Corridor"],  # Example activities of potential swaps
    "current_activity": "Viewpoint",    # Example current activity
    "desired_features": ["bigger area", "south sun", "more green", "activity X"]
}

User: "I want activity X instead of Y but I don't want to move or change my apartment, what are my options?"
Response:
{
  "action": "process_booking",
  "parameters": {
    "user_id": "H5",  # Example user ID
    "desired_activity": "X",
    "current_activity": "Y",
}

User: "I am convinced or my problem is solved!"
Response:
{
  "action": "assign_activity",
  "parameters": {
    "user_id": "H5",  # Example user ID
    "space_id": "O1",  # Example outdoor space ID
    "activity": "Sunbath"  # Example activity to assign
}

User: "Can you summarize my choices for activities?"
Response:
{
  "action": "summarize_preferences",
  "parameters": {
    "user_id": "H5",  # Example user ID
}

Important: Return only a valid JSON object. No extra text.
""",
            },
            {
                "role": "user",
                "content": message,
            },
        ],
    )
    return response.choices[0].message.content

def handle_user_request(message):
    try:
        action_json_str = suggest_actions_from_request(message)
        print("[DEBUG] Suggested JSON:\n", action_json_str)
        action_json = json.loads(action_json_str)
        result = route_action(action_json)
        return result
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON from LLM: {e}")
        return {"error": "Invalid LLM output format"}
    except Exception as e:
        logging.error(f"Failed to handle user request: {e}")
        return {"error": str(e)}

def negotiation_flow(user_query, user_id=None, last_context=None):
    import re
    context = {}
    suggestions = []
    cycling_phrases = ["no", "another suggestion", "next", "not this", "different", "something else"]
    reset_phrases = ["reset", "clear", "start over", "restart", "new negotiation", "begin again"]
    user_query_lower = user_query.lower().strip()

    # --- Reset negotiation if requested ---
    if any(phrase in user_query_lower for phrase in reset_phrases):
        return {
            "context": {},
            "suggestions": [{
                "action": "reset_negotiation",
                "explanation": "Negotiation has been reset. You can start a new request.",
                "parameters": {"user_id": user_id}
            }]
        }

    cycling_mode = any(phrase in user_query_lower for phrase in cycling_phrases)
    if cycling_mode and last_context:
        prev_suggestions = last_context.get("all_suggestions", [])
        suggestion_idx = last_context.get("suggestion_idx", 0) + 1
        if prev_suggestions and suggestion_idx < len(prev_suggestions):
            next_suggestion = prev_suggestions[suggestion_idx]
            context = last_context.get("context", {})
            context["last_action"] = next_suggestion["action"]
            context["last_params"] = next_suggestion["parameters"]
            context["all_suggestions"] = prev_suggestions
            context["suggestion_idx"] = suggestion_idx
            return {"context": context, "suggestions": [next_suggestion]}
        else:
            context = last_context.get("context", {})
            return {"context": context, "suggestions": [{
                "action": "summarize_preferences",
                "explanation": "No more alternative suggestions available for this negotiation. Please rephrase your request or try a different activity/space.",
                "parameters": {"user_id": user_id}
            }]}
    if not user_id:
        context["nearby_activities"] = "User ID not provided."
        context["preferences"] = "User ID not provided."
        return {"context": context, "suggestions": suggestions}

    # 1. Gather context
    context["nearby_activities"] = get_nearby_activities({"user_id": user_id})
    context["preferences"] = summarize_preferences({"user_id": user_id})

    # 2. Load data
    geometries, _, _, _, voting, distances, _ = load_csvs()
    try:
        assignments = pd.read_csv('llm_reasoning/llm_activity_assignments.csv')
    except Exception:
        assignments = None

    # 3. Parse user query for intent (robust extraction)
    user_query_lower = user_query.lower().strip()
    # If user says 'yes' or 'proceed', try to finalize last negotiation
    if user_query_lower in ["yes", "proceed", "confirm", "ok", "sure"] and last_context:
        last_action = last_context.get("last_action")
        last_params = last_context.get("last_params", {})
        if last_action == "propose_activity_change" and last_params:
            # TODO: Implement activity change confirmation logic here
            pass
        if last_action == "process_booking" and last_params:
            # TODO: Implement booking confirmation logic here
            pass
        if last_action == "find_profile_swap" and last_params:
            suggestions.append({
                "action": "swap_apartment",
                "explanation": f"Searching for apartments near outdoor spaces with {last_params.get('desired_activity')} for you. (This is a placeholder for the actual swap logic.)",
                "parameters": last_params
            })
            return {"context": context, "suggestions": suggestions}
        suggestions.append({
            "action": "summarize_preferences",
            "explanation": "Summarize your preferences for negotiation.",
            "parameters": {"user_id": user_id}
        })
        return {"context": context, "suggestions": suggestions}

    # 4. Extract desired activity and space from query (robust)
    desired_activity = None
    space_id = None
    # Try all reasonable patterns
    patterns = [
        r"assign ([A-Za-z0-9_\- ]+) to (O\d+)",
        r"i would like to assign ([A-Za-z0-9_\- ]+) to (O\d+)",
        r"i want ([A-Za-z0-9_\- ]+) in (O\d+)",
        r"i would like (O\d+) (?:to be|to be for|for|to have|as|to) ([A-Za-z0-9_\- ]+)",
        r"i want (O\d+) (?:for|as) ([A-Za-z0-9_\- ]+)",
        r"(O\d+) (?:to be|to be for|for|to have|as|to) ([A-Za-z0-9_\- ]+)",
        r"i would like (O\d+) be for ([A-Za-z0-9_\- ]+)",
        r"i would like (O\d+) be ([A-Za-z0-9_\- ]+)",
        r"i want (O\d+) be for ([A-Za-z0-9_\- ]+)",
        r"i want (O\d+) be ([A-Za-z0-9_\- ]+)",
        r"([A-Za-z0-9_\- ]+) (?:to|in) (O\d+)"
    ]
    for pat in patterns:
        m = re.search(pat, user_query, re.IGNORECASE)
        if m:
            if len(m.groups()) == 2:
                g1, g2 = m.group(1).strip(), m.group(2).strip()
                # Heuristic: whichever looks like Oxx is space, the other is activity
                if g1.upper().startswith("O"):
                    space_id, desired_activity = g1, g2
                elif g2.upper().startswith("O"):
                    space_id, desired_activity = g2, g1
                else:
                    desired_activity, space_id = g1, g2
            break
    # Fallback: look for 'to O10' or 'in O10' and use previous word as activity
    if not desired_activity or not space_id:
        m2 = re.search(r"([A-Za-z0-9_\- ]+) (?:to|in) (O\d+)", user_query, re.IGNORECASE)
        if m2:
            desired_activity = m2.group(1).strip()
            space_id = m2.group(2).strip()
    # Clean up desired_activity
    if desired_activity:
        desired_activity = re.sub(r"^(assigned as|as|for|to|with)\s+", "", desired_activity, flags=re.IGNORECASE).strip().title()
    # If space_id found, look up current activity
    current_activity = None
    if assignments is not None and space_id:
        row = assignments[assignments["space_id"] == space_id]
        if not row.empty:
            current_activity = row.iloc[0]["assigned_activity"]
    # Fallback: use top activity from preferences
    if not desired_activity:
        prefs = context["preferences"]["result"] if isinstance(context["preferences"], dict) else ""
        if isinstance(prefs, dict) and prefs:
            desired_activity = list(prefs.keys())[0]
    # Fallback: use top activity in nearby activities
    if not desired_activity and isinstance(context["nearby_activities"], dict):
        nearby = context["nearby_activities"].get("result", [])
        if nearby and "top_activities" in nearby[0]:
            desired_activity = list(nearby[0]["top_activities"].keys())[0]
    # Fallback: use any activity from assignments
    if not desired_activity and assignments is not None:
        acts = assignments["assigned_activity"].unique().tolist()
        if acts:
            desired_activity = acts[0]
    # Try to find a space_id if not found
    if not space_id and assignments is not None:
        if user_id in distances.columns:
            nearby = distances[["id", user_id]].rename(columns={user_id: "distance"})
            nearby = nearby.sort_values("distance").head(1)
            if not nearby.empty:
                space_id = nearby.iloc[0]["id"]
    # Try to find current activity in that space
    if not current_activity and assignments is not None and space_id:
        row = assignments[assignments["space_id"] == space_id]
        if not row.empty:
            current_activity = row.iloc[0]["assigned_activity"]

    # --- Swap/Move intent detection (MUST be before any early return) ---
    swap_phrases = [
        "move to another house", "move to another apartment", "move apartment", "move house",
        "swap", "switch", "exchange", "find another house", "find another apartment", "swap apartment", "switch apartment"
    ]
    if any(phrase in user_query_lower for phrase in swap_phrases):
        # Try to extract desired activity
        m = re.search(r"(?:for|with|to)?\s*(sports|playground|bbq|cinema|retreat|viewpoint|garden|corridor|sunbath|meeting room|outdoor [a-z]+)", user_query_lower)
        if m:
            desired_activity = m.group(1).strip().title()
        else:
            # fallback to top preference
            prefs = context["preferences"]["result"] if isinstance(context["preferences"], dict) else ""
            if isinstance(prefs, dict) and prefs:
                desired_activity = list(prefs.keys())[0]
            else:
                desired_activity = "Sports"
        swap_params = {"user_id": user_id, "desired_activity": desired_activity}
        suggestions.append({
            "action": "find_profile_swap",
            "explanation": f"Suggesting possible apartment swaps for you to be closer to an outdoor space with {desired_activity}.",
            "parameters": swap_params
        })
        # Save context for cycling
        context["last_action"] = "find_profile_swap"
        context["last_params"] = swap_params
        context["all_suggestions"] = suggestions
        context["suggestion_idx"] = 0
        return {"context": context, "suggestions": suggestions}

    # If missing info, try to use last_context for multi-turn negotiation
    if (not space_id or not desired_activity or not current_activity) and last_context:
        last_action = last_context.get("last_action")
        last_params = last_context.get("last_params", {})
        # Use last known params if available
        space_id = space_id or last_params.get("space_id")
        desired_activity = desired_activity or last_params.get("desired_activity")
        current_activity = current_activity or last_params.get("current_activity")
        # If still missing, fallback to summarize_preferences
        if not space_id or not desired_activity or not current_activity:
            suggestions.append({
                "action": "summarize_preferences",
                "explanation": "Could not determine enough details from your request. Please specify the space and activity.",
                "parameters": {"user_id": user_id}
            })
            context["error"] = "Not enough info for negotiation."
            return {"context": context, "suggestions": suggestions}

    if not current_activity:
        suggestions.append({
            "action": "summarize_preferences",
            "explanation": f"Could not determine the current activity for {space_id}. Please check the data.",
            "parameters": {"user_id": user_id, "space_id": space_id}
        })
        context["error"] = f"No current activity found for {space_id}."
        return {"context": context, "suggestions": suggestions}
    # If current activity is already the desired one
    if current_activity.lower() == desired_activity.lower():
        suggestions.append({
            "action": "summarize_preferences",
            "explanation": f"{space_id} is already assigned to {desired_activity}. No change needed.",
            "parameters": {"user_id": user_id, "space_id": space_id, "activity": desired_activity}
        })
        context["info"] = f"{space_id} is already assigned to {desired_activity}."
        return {"context": context, "suggestions": suggestions}
    # Otherwise, suggest negotiation/booking
    suggestions.append({
        "action": "propose_activity_change",
        "explanation": f"Negotiate with residents who voted for {current_activity} in {space_id}. If some also like {desired_activity}, you may be able to swap the activity.",
        "parameters": {"space_id": space_id, "current_activity": current_activity, "desired_activity": desired_activity, "user_id": user_id}
    })
    suggestions.append({
        "action": "process_booking",
        "explanation": f"You can book {space_id} for {desired_activity} if the slot is available and other residents agree.",
        "parameters": {"user_id": user_id, "space_id": space_id, "activity": desired_activity}
    })
    # Save last action/params for multi-turn and all suggestions for cycling
    if suggestions:
        context["last_action"] = suggestions[0]["action"]
        context["last_params"] = suggestions[0]["parameters"]
        context["all_suggestions"] = suggestions
        context["suggestion_idx"] = 0
    return {"context": context, "suggestions": suggestions}

