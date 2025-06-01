from flask import Flask, request, jsonify
from server.config import *
from llm_calls import *
import json
from llm_reasoning_test import *
import re # Import the regular expression module
from geometry_orchestrator import get_intelligent_geometric_suggestions, process_natural_language_to_sql_answer # Ensure this import is present

app = Flask(__name__)




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

#llm_nearby_space_qna
#region
# @app.route('/llm_nearby_space_qna', methods=['POST'])
# def llm_nearby_space_qna():
#     data = request.get_json()
#     house_key = data.get("house_key")
#     question = data.get("question")

#     if not house_key or not question:
#         return jsonify({"error": "Missing 'house_key' or 'question' in request."}), 400

#     response = ask_about_nearby_spaces(house_key, question)
#     return jsonify({"response": response})




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
  

    space_id = data.get('space_id')
    # Default to 'default_profile' if not provided, or make it mandatory
    user_profile = data.get('user_profile', 'young_entrepreneurs') 

    if not data:
        return jsonify({"error": "Request body must be JSON."}), 400

    user_question = data.get('question')
    # space_id is already fetched above

    if user_question:
        # Path 1: Process natural language question for SQL
        # This functionality comes from process_natural_language_to_sql_answer in geometry_orchestrator.py
        result = process_natural_language_to_sql_answer(user_question)
        if "error" in result:
            # process_natural_language_to_sql_answer should ideally log detailed errors.
            # The client receives the error message from the result.
            # We return 500 if it's an internal/unexpected error, otherwise 400 for bad input/query.
            # For simplicity here, we'll use 500 if an error key is present,
            # assuming the orchestrator flags critical issues.
            return jsonify(result), 500 
        return jsonify(result), 200

    elif space_id:
        # Path 2: Suggest geometric variations
        # user_profile is already fetched above with a default
        # This functionality comes from get_intelligent_geometric_suggestions in geometry_orchestrator.py
        try:
            suggestions_json_str = get_intelligent_geometric_suggestions(space_id, user_profile)
            
            # Attempt to extract JSON if wrapped in markdown or has leading/trailing text
            cleaned_json_str = suggestions_json_str # Default to original string
            # First, try to find JSON wrapped in ```json ... ```
            match_markdown = re.search(r'```json\s*(\{[\s\S]*?\})\s*```', suggestions_json_str, re.DOTALL)
            if match_markdown:
                cleaned_json_str = match_markdown.group(1)
            else:
                # If not in markdown, try to find the first occurrence of a JSON object structure
                # This looks for the first '{' and the last '}' assuming it's a single JSON object
                match_object = re.search(r'(\{[\s\S]*\})', suggestions_json_str, re.DOTALL)
                if match_object:
                    cleaned_json_str = match_object.group(1)
            
            suggestions_data = json.loads(cleaned_json_str) # Parse the cleaned or original string
            app.logger.info(f"Successfully parsed JSON for space_id {space_id}")
            return jsonify(suggestions_data), 200
        except json.JSONDecodeError as e:
            # Log the raw string and the cleaned attempt
            app.logger.error(f"JSONDecodeError for space_id {space_id}: {e}. Raw LLM response: >>>{suggestions_json_str}<<< Cleaned attempt: >>>{cleaned_json_str if 'cleaned_json_str' in locals() and cleaned_json_str != suggestions_json_str else 'N/A (used raw)'}<<<")
            return jsonify({"error": "Failed to parse LLM response for geometric variations. Output was not valid JSON."}), 500
        except Exception as e:
            app.logger.error(f"Error in geometric suggestions for space_id {space_id}: {str(e)}. Raw LLM response if available: >>>{suggestions_json_str if 'suggestions_json_str' in locals() else 'Not available'}<<< Cleaned attempt if available: >>>{cleaned_json_str if 'cleaned_json_str' in locals() and cleaned_json_str != suggestions_json_str else 'N/A'}<<<")
            return jsonify({"error": f"Failed to suggest geometric variations: {str(e)}"}), 500
    else:
        # Neither 'question' nor 'space_id' was provided
        return jsonify({"error": "Invalid request. Provide 'question' for SQL query, or 'space_id' (and optionally 'user_profile') for geometric suggestions."}), 400

 
##~~GEOMETRIC VARIATIONS FROM LLM AND SQL~~##



if __name__ == '__main__':
    app.run(debug=True)