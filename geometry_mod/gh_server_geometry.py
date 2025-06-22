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

@app.route('/llm_general_call', methods=['POST'])
def llm_general_call():
    data = request.get_json()
    user_input = data.get('input', '')
    user_profile = data.get('profile', 'young_entrepreneurs')  # default if not passed

    answer = answer_general_questions(user_input, user_profile)
    return jsonify({'response': answer})


##~~GEOMETRIC VARIATIONS FROM LLM AND SQL~~##

@app.route('/suggest_geometric_variations', methods=['POST'])
def suggest_geometric_variations_route():
    data = request.get_json()

    if not data:
        return jsonify({"error": "Request body must be JSON."}), 400

    space_id = data.get('space_id')
    resident_key = data.get('resident_key') 
    user_question = data.get('question') # This can be a specific question or a general request for suggestions
    desired_activity = data.get('desired_activity') # New: Get desired activity

    # Path 1: Geometric Suggestions (if space_id and resident_key are provided)
    # The user_question, if present, will be used to tailor these suggestions.
    if space_id and resident_key:
        try:
            # Orchestrator now returns a dictionary (parsed JSON or error dict)
            suggestions_data = get_intelligent_geometric_suggestions(space_id, resident_key, user_question, desired_activity)
            if "error" in suggestions_data:
                # Log the error if needed, using app.logger if Flask logging is configured
                app.logger.error(f"Error from orchestrator for space_id {space_id}, resident {resident_key}: {suggestions_data.get('details', '')}. Raw: {suggestions_data.get('raw_response', '')}")
                return jsonify(suggestions_data), 500 # Propagate error
            return jsonify(suggestions_data), 200
        except Exception as e:
            app.logger.error(f"Unexpected error in /suggest_geometric_variations for space_id {space_id}, resident {resident_key}: {str(e)}")
            return jsonify({"error": f"Failed to suggest geometric variations: {str(e)}"}), 500

    # Path 2: General SQL Query (if only user_question is provided, and space_id/resident_key are missing for geometric path)
    elif user_question: # and not (space_id and resident_key)
        try:
            result = process_natural_language_to_sql_answer(user_question)
            if "error" in result:
                 # Assuming 500 for critical issues from orchestrator, 400 for bad input/query.
                status_code = 500 if "Could not determine" not in result["error"] and "not found in database schema" not in result["error"] else 400
                return jsonify(result), status_code
            return jsonify(result), 200
        except Exception as e:
            app.logger.error(f"Error processing natural language question to SQL: {str(e)}")
            return jsonify({"error": f"Failed to process question for SQL: {str(e)}"}), 500
    else:
        # Path 3: Insufficient information for either route
        return jsonify({"error": "Invalid request. Provide 'question' for a general query, or 'space_id', 'resident_key' (and optionally 'question' and 'desired_activity') for geometric suggestions."}), 400

 
##~~GEOMETRIC VARIATIONS FROM LLM AND SQL~~##



if __name__ == '__main__':
    app.run(debug=True, port=5004)