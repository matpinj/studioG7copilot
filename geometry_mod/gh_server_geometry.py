import sys
import os

# Add the project root directory to sys.path
# This allows absolute imports from the project root (e.g., 'server.config')
# __file__ is d:\01_IAAC\03_aia studio\studioG7copilot\geometry_mod\gh_server_geometry.py
# os.path.dirname(__file__) is d:\01_IAAC\03_aia studio\studioG7copilot\geometry_mod
# os.path.join(os.path.dirname(__file__), '..') is d:\01_IAAC\03_aia studio\studioG7copilot
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from flask import Flask, request, jsonify
from server.config import *  # Changed from relative to absolute
from llm_calls import *      # Assuming llm_calls.py is at the project root
import json
from llm_reasoning_test import * # Assuming llm_reasoning_test.py is at the project root
import re # Import the regular expression module
from geometry_mod.geometry_orchestrator import get_intelligent_geometric_suggestions, process_natural_language_to_sql_answer 
from geometry_mod.geometry_tab_handler import geometry_tab # Import the blueprint

app = Flask(__name__)
app.register_blueprint(geometry_tab) # Register the blueprint

import socket

def send_udp_command(message, host="127.0.0.1", port=9000):  # Set port to your gHowl UDP receiver port!
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(message.encode(), (host, port))

@app.route('/initiate_gh_workflow', methods=['POST'])
def initiate_gh_workflow():
    return suggest_geometric_variations_route()

@app.route('/llm_call', methods=['POST'])
def llm_call():
    data = request.get_json()
    user_input = data.get('input', '')
    user_profile = data.get('profile', 'young_entrepreneurs')  # default if not passed


    answer = classify_input(user_input)
    return jsonify({'response': answer})


@app.route('/llm_nearby_space_qna', methods=['POST'])
def llm_nearby_space_qna():
    data = request.get_json()
    house_key = data.get("house_key")
    question = data.get("question")

    if not house_key or not question:
        return jsonify({"error": "Missing 'house_key' or 'question' in request."}), 400

    try:
        # Load pre-generated activity assignments
        assignments = pd.read_csv("llm_activity_assignments.csv")  # space_id, assigned_activity
        distances = pd.read_csv("resident_data/resident_distances.csv")  # wide format

        if house_key not in distances.columns:
            return jsonify({"error": f"No distances found for house key {house_key}."}), 404

        # Find nearest 5 outdoor spaces
        nearby = distances[["Outdoor Space", house_key]].rename(columns={house_key: "distance"})
        nearby = nearby.sort_values("distance").head(5)

        # Join with activity assignments
        nearby = nearby.merge(assignments, left_on="Outdoor Space", right_on="space_id", how="left")

        space_summaries = "\n".join([
            f"- {row['Outdoor Space']} ({row['assigned_activity']}): {row['distance']:.1f}m away"
            for _, row in nearby.iterrows()
        ])

        prompt = f"""
You are a community advisor helping a resident understand the outdoor spaces near them.

### Resident info:
- House key: {house_key}

### Question:
{question}

### Nearby spaces and assigned uses:
{space_summaries}

Answer the resident’s question based on this information.
Be concise and use plain language.
"""

        # Send to local LLM (LM Studio)
        response = requests.post(
            "http://localhost:1234/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json={
                "model": "local-model",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7
            }
        )

        reply = response.json()["choices"][0]["message"]["content"]
        return jsonify({"response": reply})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/llm_space_assignment', methods=['POST'])
def llm_space_assignment():
    data = request.get_json()
    space_id = data.get("space_id")
    if not space_id:
        return jsonify({"error": "Missing 'space_id' in request."}), 400

    result = generate_llm_assignment_for_id(space_id)
    return jsonify(result)

# ---------------------------------------------------------------------------
# 1.  Fake engine – ALWAYS returns a well-formed dict in the schema you gave
# ---------------------------------------------------------------------------
def get_intelligent_geometric_suggestions(
        space_id: str,
        resident_key: str,
        user_question: str,
        desired_activity: str):
    """
    Fake version for local testing.
    Ignores the inputs and returns a deterministic payload.
    """
    return {
        "space_id":        "O2",
        "space_details":   "balcony",
        "user_profile":    "travelers/expats",
        "user_question":   "Who else benefits?",
        "desired_activity": "Sunbath",
        "resident_distance": 60.42,
        "current_activity": "Sunbath",
        "usability_prediction": "",
        "suggestions": [
            {
                "variation_type": "Add Wall",
                "variation_name": "Low wall for wind/privacy",
                "description":
                    "Adds a low wall to provide wind and privacy while still allowing sunlight.",
                "reason_for_profile":
                    "Suitable for travelers/expats who value sunbathing and relaxation.",
                "optimal_time_impact": "+1 hour of usable time",
                "profile_suitability_notes":
                    "This suggestion is suitable for the traveler/expat profile as it provides a comfortable and private space for sunbathing.",
                "suitability_%_increase": 20,
                "comfort_usability_impact":
                    "Improved comfort and usability due to added wind protection and privacy.",
                "other_beneficiaries": {"H8": "Sunbath", "H67": "Sunbath"},
                "wall_height": 0.8,
                "slab_extension_sqm": 2,
                "louvre_height": 0.5,
                "other_activities_benefit": []
            }
        ],
        "summary_reasoning":
            "A low parapet mitigates cross-winds and visual exposure, extending comfortable "
            "sun-hours and benefiting neighbouring residents who also sunbathe.",
        "householder_reasoning": {
            "H8": "Gains calmer sunbathing conditions.",
            "H67": "Enjoys extra privacy and lower wind chill."
        }
    }


# ---------------------------------------------------------------------------
# 2.  Patch the existing route so it sends ONE clean JSON string via UDP
# ---------------------------------------------------------------------------
@app.route("/suggest_geometric_variations", methods=["POST"])
def suggest_geometric_variations_route():
    print("Endpoint /suggest_geometric_variations called")
    data = request.get_json()

    if not data:
        return jsonify({"error": "Request body must be JSON."}), 400

    try:
        suggestions_data = get_intelligent_geometric_suggestions(
            data.get("space_id"),
            data.get("resident_key"),
            data.get("question"),
            data.get("desired_activity")
        )

        # Send the entire object over UDP as a single JSON payload
        udp_msg = json.dumps(suggestions_data, ensure_ascii=False)
        print(f"[OK] Sending UDP to 127.0.0.1:9000:\n{udp_msg}")
        send_udp_command(udp_msg, host="127.0.0.1", port=9000)

        return jsonify(suggestions_data), 200

    except Exception as e:
        app.logger.error(f"Unexpected error in /suggest_geometric_variations: {e}")
        return jsonify({"error": f"Failed to suggest geometric variations: {e}"}), 500

 
##~~GEOMETRIC VARIATIONS FROM LLM AND SQL~~##



if __name__ == '__main__':
    app.run(debug=False, port=5004)