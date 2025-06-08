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
    """Return nearby activities for a user, including area and desired activities."""
    geometries, _, _, _, voting, distances, personas = load_csvs()
    user_id = params.get("user_id")
    desired_activities = params.get("desired_activity", [])
    if not user_id:
        return {"error": "No user_id provided."}
    if user_id not in distances.columns:
        return {"error": f"No distances found for user {user_id}."}
    # Get 3 nearest spaces
    nearby = distances[["id", user_id]].rename(columns={user_id: "distance"})
    nearby = nearby.sort_values("distance").head(3)
    results = []
    for _, row in nearby.iterrows():
        space_id = row["id"]
        dist = row["distance"]
        # Get area from geometries
        space_row = geometries[geometries["id"] == space_id]
        area = float(space_row["area"].iloc[0]) if not space_row.empty else None
        # Get top activities by voting
        votes = voting[voting["space"] == space_id]
        top_activities = votes.groupby("activity")["weight"].sum().sort_values(ascending=False)
        # If desired_activities is provided, filter or highlight them
        if desired_activities:
            desired_found = {act: top_activities.get(act, 0) for act in desired_activities}
        else:
            desired_found = {}
        # Prepare result
        results.append({
            "space_id": space_id,
            "distance": dist,
            "area": area,
            "top_activities": top_activities.head(3).to_dict(),
            "desired_activities_weights": desired_found
        })
    return {"result": results, "params": params}

def propose_activity_change(params):
    """Suggest negotiation with other residents for activity change."""
    user_id = params.get("user_id")
    desired = params.get("desired_activity")
    current = params.get("current_activity")
    if not user_id or not desired or not current:
        return {"error": "Missing user_id, desired_activity, or current_activity."}
    # Example: Just return a negotiation suggestion
    return {
        "result": f"To change from {current} to {desired}, you may need to negotiate with other residents.",
        "params": params
    }

def find_profile_swap(params):
    """Suggest possible apartment (house) swaps based on preferences."""
    _, _, _, _, voting, _, personas = load_csvs()
    desired_features = params.get("desired_features", [])
    # Example: Just return a static suggestion
    return {
        "result": f"Suggested swaps for features {desired_features}: H20, H50 (example)",
        "params": params
    }

def process_booking(params):
    """Book an activity for a user."""
    user_id = params.get("user_id")
    desired = params.get("desired_activity")
    if not user_id or not desired:
        return {"error": "Missing user_id or desired_activity."}
    # Example: Just return a booking confirmation
    return {
        "result": f"Booked activity {desired} for user {user_id} (example).",
        "params": params
    }

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
- "find_profile_swap": For requests about swapping apartments/houses with someone whose preferences match better.
- "process_booking": For requests about booking a space or activity for a certain time.
- "assign_activity": For confirming or finalizing an activity assignment.
- "summarize_preferences": For summarizing the user's activity or space preferences.

Respond ONLY with valid JSON. No extra text.

### Examples:

User: "I would like to have a larger space, what do you suggest?"
Response:
{
  "action": "change_geometry",
  "reasoning": "User wants a larger space, so geometry change is suggested."
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
    },
  "reasoning": "User wants to swap apartments with someone whose preferences match."
}

User: "I want activity X instead of Y but I don't want to move or change my apartment, what are my options?"
Response:
{
  "action": "process_booking",
  "parameters": {
    "user_id": "H5",  # Example user ID
    "desired_activity": "X",
    "current_activity": "Y",
  },
  "reasoning": "User wants to book a different activity for their current space."
}

User: "I am convinced or my problem is solved!"
Response:
{
  "action": "assign_activity",
  "parameters": {
    "user_id": "H5",  # Example user ID
    "space_id": "O1",  # Example outdoor space ID
    "activity": "Sunbath"  # Example activity to assign
  },
  "reasoning": "User is satisfied and wants to finalize the activity assignment."
}

User: "Can you summarize my choices for activities?"
Response:
{
  "action": "summarize_preferences",
  "parameters": {
    "user_id": "H5",  # Example user ID
    "reasoning": "User wants a summary of their activity preferences."
  }
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

