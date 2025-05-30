from flask import Flask, request, jsonify
from server.config import *
from llm_calls import *
from llm_general_calls import *
from llm_reasoning import *
import json
import urllib.request
from llm_reasoning_test import *

app = Flask(__name__)



@app.route('/llm_call', methods=['POST'])
def llm_call():
    data = request.get_json()
    user_input = data.get('input', '')
    user_profile = data.get('profile', 'young_entrepreneurs')  # default if not passed

    answer = assign_activity(user_input, user_profile)
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

if __name__ == '__main__':
    app.run(debug=True)