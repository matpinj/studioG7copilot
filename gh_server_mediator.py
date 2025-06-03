from flask import Flask, request, jsonify
from server.config import *
from llm_calls import *
#import calls for REASONING ENGINE TESTING
from llm_reasoning_test import *
import re
# import calls for GENERAL QUERIES USING SQL
from sql_gh import *
from llm_negotiation import *
app = Flask(__name__)


# THIS CALL IS FOR GENERAL QUERIES USING SQL FROM SQL_GH.PY(like an llmcalls file)
@app.route('/sql_gh', methods=['POST'])
def handle_grasshopper_input():
    data = request.get_json()
    user_question = data.get('question', '')
    answer = answer_user_question(user_question, db_path="sql/gh_data.db")  # <-- specify correct db
    return jsonify({'response': answer})


# THIS CALL IS FOR TRANSFERING JSON FILES AFTER LLM ASSIGNS NEW ACTIVITIES TO 3D GRAPH IN GH
JSON_FILE = "llm_reasoning/llm_assignments.json"
@app.route('/get_json', methods=['GET'])
def get_json():
    try:
        with open(JSON_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)})

# THIS CALL IS A TEST CALL FOR THE LLM NEGOTIATION CALLS FROM LLM_NEGOTIATION.PY
#This one returns  action but not working atm
# @app.route('/llm_negotiate', methods=['POST'])
# def llm_negotiate():
#     data = request.get_json()
#     house_key = data.get('house_key', '')
#     query = data.get('query', '')

#     # Combine house_key and query for context
#     if house_key:
#         llm_input = f"House key: {house_key}\n{query}"
#     else:
#         llm_input = query

#     result = handle_user_request(llm_input)
#     return jsonify({'response': result})

# if __name__ == '__main__':
#     app.run(debug=True)

@app.route('/llm_negotiate', methods=['POST'])
def llm_negotiate():
    data = request.get_json()
    house_key = data.get('house_key', '')
    query = data.get('query', '')

    if house_key:
        llm_input = f"House key: {house_key}\n{query}"
    else:
        llm_input = query

    # Get the raw LLM suggestion as a string
    action_json_str = suggest_actions_from_request(llm_input)
    try:
        action_json = json.loads(action_json_str)
    except Exception:
        action_json = {"error": "Invalid LLM output", "raw": action_json_str}

    result = route_action(action_json)

    # Prepare outputs
    result_text = result.get('result', '')
    params_text = result.get('params', '')
    return jsonify({
        'result': result_text,
        'params': params_text,
        'llm_suggestion': action_json  # <-- Add this line
    })

#THIS CALL IS FOR RESIDENTS SPECIFIC QUERIES ABOUT NEARBY SPACES AFTER THE LLM HAS BEEN TRAINED ON THE ACTIVITY ASSIGNMENTS
@app.route('/llm_nearby_space_qna', methods=['POST'])
def llm_nearby_space_qna():
    data = request.get_json()
    house_key = data.get("house_key")
    question = data.get("question", "")

    if not house_key or not question:
        return jsonify({"error": "Missing 'house_key' or 'question' in request."}), 400

    try:
        # 1. If user asks "why" or "reason" for a specific space/activity
        match = re.search(r'(?:why|reason).*?(O\d+)', question, re.IGNORECASE)
        if match:
            space_id = match.group(1)
            geometries, thresh, green, usability, voting, distances, personas = load_csvs()
            reasoning = explain_activity_for_space(
                space_id, question, geometries, thresh, green, usability, voting, distances, personas
            )
            return jsonify({"response": reasoning})

        # 2. If user asks for closest/nearest outdoor spaces or spaces on my floor
        if re.search(r'(closest|nearest|nearby).*outdoor', question, re.IGNORECASE) or \
           re.search(r'outdoor.*(spaces|areas).*on my floor', question, re.IGNORECASE):
            conn = sqlite3.connect('sql/gh_data.db')
            distances = pd.read_sql_query("SELECT * FROM resident_distances", conn)
            conn.close()
            assignments = pd.read_csv('llm_reasoning\llm_activity_assignments.csv')
            voting = pd.read_csv('resident_data/voting_weights.csv')

            if house_key not in distances.columns:
                return jsonify({"error": f"No distances found for house key {house_key}."}), 404

            nearby = distances[["Outdoor Space", house_key]].rename(columns={house_key: "distance"})
            nearby = nearby.sort_values("distance").head(5)

            space_summaries = []
            for _, row in nearby.iterrows():
                space_id = row['Outdoor Space']
                distance = row['distance']
                assigned_row = assignments[assignments['space_id'] == space_id]
                assigned_activity = assigned_row.iloc[0]['assigned_activity'] if not assigned_row.empty else "Unknown"
                space_summaries.append(
                    f"- {space_id} ({assigned_activity}): {distance:.1f}m away"
                )
            response_text = "Nearest outdoor spaces:\n" + "\n".join(space_summaries)
            return jsonify({"response": response_text})

        # 3. Otherwise, use the general LLM prompt (main logic)
        # (This is your original code, unchanged)
        conn = sqlite3.connect('sql/gh_data.db')
        activity_space = pd.read_sql_query("SELECT * FROM activity_space", conn)
        distances = pd.read_sql_query("SELECT * FROM resident_distances", conn)
        conn.close()
        voting = pd.read_csv('resident_data/voting_weights.csv')
        assignments = pd.read_csv('llm_reasoning\llm_activity_assignments.csv')

        if house_key not in distances.columns:
            return jsonify({"error": f"No distances found for house key {house_key}."}), 404

        nearby = distances[["Outdoor Space", house_key]].rename(columns={house_key: "distance"})
        nearby = nearby.sort_values("distance").head(5)

        space_summaries = []
        for _, row in nearby.iterrows():
            space_id = row['Outdoor Space']
            distance = row['distance']

            assigned_row = assignments[assignments['space_id'] == space_id]
            if not assigned_row.empty:
                assigned_activity = assigned_row.iloc[0]['assigned_activity']
            else:
                assigned_activity = "Unknown"

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

### Your task:
- Use all the information above to answer the resident's question, whether it is about preferences, assignments, reasoning, or general context.
- If the question is about why a space does not match preferences, or why an activity is assigned, explain using the voting data and assignments.
- If the question is about the list, provide the list.
- If the question is about how decisions are made, explain the process using the data.
- If the question is about preferences, summarize the relevant voting or assignment data.
- If the question is something else, use your best judgment to answer using all the context above.
Be concise and use plain language.
"""

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

if __name__ == '__main__':
    app.run(debug=True)