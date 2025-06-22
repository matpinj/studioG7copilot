from flask import Blueprint, request, jsonify
# import requests # No longer directly calling GH via HTTP from here in this model
import json
from geometry_mod.geometry_orchestrator import get_intelligent_geometric_suggestions # Import the orchestrator

geometry_tab = Blueprint("geometry_tab", __name__)

# Global variable to store the latest payload for Grasshopper
latest_gh_payload = None
latest_gh_result = None # Optional: to store results from GH

@geometry_tab.route("/initiate_gh_workflow", methods=["POST"])
def initiate_gh_workflow():
    global latest_gh_payload, latest_gh_result # Ensure latest_gh_result is accessible
    data = request.get_json()

    # Extract inputs (sent from the UI or Grasshopper)
    space_id = data.get("space_id")
    resident_key = data.get("resident_key")
    question = data.get("question")
    desired_activity = data.get("desired_activity")

    if not all([space_id, resident_key]):
        return jsonify({"error": "Missing required fields: space_id and resident_key"}), 400

    # Store the payload for Grasshopper to fetch
    latest_gh_payload = {
        "space_id": space_id,
        "resident_key": resident_key,
        "user_question": question,
        "desired_activity": desired_activity,
        "triggered": True # Indicate new data is available
        # This payload is still stored for GH if it wants the original inputs
    }
    print(f"Stored payload for GH: {latest_gh_payload}")

    # Directly call the orchestrator to get LLM suggestion
    try:
        llm_suggestion_data = get_intelligent_geometric_suggestions(space_id, resident_key, question, desired_activity)
        latest_gh_result = llm_suggestion_data # Store the result (dict or error dict)
        if "error" in llm_suggestion_data:
            print(f"Error from orchestrator during UI workflow: {llm_suggestion_data}")
            return jsonify(latest_gh_result), 500 # Return the actual error data from orchestrator
        print(f"LLM suggestion processed and stored: {latest_gh_result}")
        return jsonify(latest_gh_result), 200 # Return the actual LLM suggestion data
    except Exception as e:
        latest_gh_result = {"error": f"Failed to process LLM suggestion in workflow: {str(e)}"}
        print(f"Exception during LLM orchestration in UI workflow: {e}")
        return jsonify(latest_gh_result), 500 # Return the error details

@geometry_tab.route("/get_gh_input", methods=["GET"])
def get_gh_input():
    global latest_gh_payload
    if latest_gh_payload and latest_gh_payload.get("triggered"):
        # Return the data and then mark it as fetched (or implement a more robust queue)
        data_to_send = latest_gh_payload.copy()
        latest_gh_payload["triggered"] = False # So GH doesn't process the same data multiple times
        return jsonify(data_to_send), 200
    return jsonify({"message": "No new data for Grasshopper."}), 204 # No content

# Endpoint for GH to post results
@geometry_tab.route("/submit_gh_result", methods=["POST"])
def submit_gh_result():
    global latest_gh_result
    latest_gh_result = request.get_json()
    print(f"Received result from GH: {latest_gh_result}")
    return jsonify({"message": "Result received from Grasshopper."}), 200

# Endpoint for UI to fetch results
@geometry_tab.route("/get_gh_result", methods=["GET"])
def get_gh_result():
    global latest_gh_result
    if latest_gh_result:
        return jsonify(latest_gh_result), 200
    return jsonify({"message": "No result from Grasshopper yet or result has been cleared."}), 204
