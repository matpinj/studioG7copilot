from flask import Flask, request, jsonify
from server.config import *
from llm_calls import *
import json
import urllib.request
#import calls for REASONING ENGINE TESTING
from llm_reasoning_test import *
#import calls for GENERAL QUERIES USING SQL
from sql_gh import *
app = Flask(__name__)


# THIS CALL IS FOR GENERAL QUERIES USING SQL FROM SQL_GH.PY(like an llmcalls file)
@app.route('/sql_gh', methods=['POST'])
def handle_grasshopper_input():
    data = request.get_json()
    user_question = data.get('question', '')
    answer = answer_user_question(user_question, db_path="sql/gh_data.db")  # <-- specify correct db
    return jsonify({'response': answer})

# THIS CALL IS A TEST CALL FOR THE LLM CALLS FROM LLM_CALLS.PY
# @app.route('/llm_call', methods=['POST'])
# def llm_call():
#     data = request.get_json()
#     user_input = data.get('input', '')
#     user_profile = data.get('profile', 'young_entrepreneurs')  # default if not passed

#     answer = assign_activity(user_input, user_profile)
#     return jsonify({'response': answer})

#THIS CALL IS FOR RESIDENTS SPECIFIC QUERIES ABOUT NEARBY SPACES AFTER THE LLM HAS BEEN TRAINED ON THE ACTIVITY ASSIGNMENTS
@app.route('/llm_nearby_space_qna', methods=['POST'])
def llm_nearby_space_qna():
    data = request.get_json()
    house_key = data.get("house_key")
    question = data.get("question")

    if not house_key or not question:
        return jsonify({"error": "Missing 'house_key' or 'question' in request."}), 400

    try:
        # Load all data fresh for each request
        conn = sqlite3.connect('sql/gh_data.db')
        activity_space = pd.read_sql_query("SELECT * FROM activity_space", conn)
        distances = pd.read_sql_query("SELECT * FROM resident_distances", conn)
        conn.close()
        voting = pd.read_csv('resident_data/voting_weights.csv')
        assignments = pd.read_csv('llm_activity_assignments.csv')  # <-- Load assignments

        # Find nearest 5 outdoor spaces
        if house_key not in distances.columns:
            return jsonify({"error": f"No distances found for house key {house_key}."}), 404

        nearby = distances[["Outdoor Space", house_key]].rename(columns={house_key: "distance"})
        nearby = nearby.sort_values("distance").head(5)

        # For each space, get assigned activity from CSV and voting summary
        space_summaries = []
        for _, row in nearby.iterrows():
            space_id = row['Outdoor Space']
            distance = row['distance']

            # Assigned activity from llm_activity_assignments.csv
            assigned_row = assignments[assignments['space_id'] == space_id]
            if not assigned_row.empty:
                assigned_activity = assigned_row.iloc[0]['assigned_activity']
            else:
                assigned_activity = "Unknown"

                       # Voting summary for this space (all residents)
            votes = voting[voting['space'] == space_id]
            if not votes.empty:
                top_votes = (
                    votes.groupby('activity')['weight']
                    .sum()
                    .sort_values(ascending=False)
                    .head(3)
                )
                voting_summary = "; ".join([f"{act}: {w:.1f}" for act, w in top_votes.items()])
            else:
                voting_summary = "No voting data"

            # Voting summary for this resident
            resident_votes = voting[(voting['space'] == space_id) & (voting['resident'] == house_key)]
            if not resident_votes.empty:
                resident_voting_summary = "; ".join([f"{row['activity']}: {row['weight']:.2f}" for _, row in resident_votes.iterrows()])
            else:
                resident_voting_summary = "No votes from this resident"

            space_summaries.append(
                f"- {space_id} ({assigned_activity}): {distance:.1f}m away | Voting (all): {voting_summary} | Your votes: {resident_voting_summary}"
            )

        space_summaries_text = "\n".join(space_summaries)

        prompt = f"""
You are a community advisor helping a resident understand the outdoor spaces near them.

### Resident info:
- House key: {house_key}

### Question:
{question}

### Nearby spaces, assigned activities, and voting preferences for each activity per outdoor space and weights:
{space_summaries_text}

Outdoor spaces are activity spaces and their keys start with O1, O2, etc. Each space has an assigned activity and a distance from the resident's house.
Apartment keys is residents house key and start with H1, H2, etc. The resident has a question about the nearby spaces.
### Your task:
Answer the resident's question based on this information.
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


# @app.route('/llm_space_assignment', methods=['POST'])
# def llm_space_assignment():
#     data = request.get_json()
#     space_id = data.get("space_id")
#     if not space_id:
#         return jsonify({"error": "Missing 'space_id' in request."}), 400

#     result = generate_llm_assignment_for_id(space_id)
#     return jsonify(result)

# @app.route('/llm_general_call', methods=['POST'])
# def llm_general_call():
#     data = request.get_json()
#     user_input = data.get('input', '')
#     user_profile = data.get('profile', 'young_entrepreneurs')  # default if not passed

#     answer = answer_general_questions(user_input, user_profile)
#     return jsonify({'response': answer})

if __name__ == '__main__':
    app.run(debug=True)