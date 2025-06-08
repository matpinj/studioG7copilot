from flask import Flask, request, jsonify
from server.config import *
from llm_calls import *
#import calls for REASONING ENGINE TESTING
from llm_reasoning_test import *
import re
import sys
import threading
# import calls for GENERAL QUERIES USING SQL
from sql_gh import *
from llm_negotiation import *
app = Flask(__name__)
from PyQt5.QtWidgets import QApplication
import json
import requests
import pandas as pd
from llm_reasoning_test import load_csvs, explain_activity_for_space


#THIS CALLS AND DEFINITIONS ARE FOR PYQT5 TO SEND AND GET DATA
stored_data = None
concept_data = None

@app.route('/get_from_grasshopper', methods=['POST', 'GET'])
def get_from_grasshopper():
    global stored_data
    if request.method == 'POST':
        data = request.get_json()
        stored_data = data['input']
    return jsonify(stored_data)

@app.route('/send_to_grasshopper', methods=['POST', 'GET'])
def send_to_grasshopper():
    global concept_data
    
    if request.method == 'POST':
        data = request.get_json()
        concept_data = data['concept_text']
    
    return jsonify(concept_data)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    house_key = data.get('house_key', '')
    message = data.get('message', '')

    # 1. Ask the LLM to classify the intent
    llm_prompt = f"""
You are a smart assistant for a residential community.  
Your job is to classify the user's message into one of these actions, based on their intent:

- "llm_nearby_space_qna": For questions about nearby outdoor spaces, such as:
    - "What are the closest outdoor spaces to my apartment?"
    - "Why was activity X assigned to space O1?"
    - "What are the voting results for the playground?"
    - "Which spaces are on my floor?"
    - "Why can't I have my preferred activity in O2?"
    - Any question about reasoning, voting, assignments, or explanations about spaces/activities.

- "llm_negotiate": For negotiation, booking, assignment, or preference changes, such as:
    - "I want to book the yoga area for tomorrow."
    - "Can I swap my apartment with someone who prefers more green?"
    - "Assign sunbathing to my balcony."
    - "I want a bigger space."
    - "Suggest activities I might enjoy."
    - "Summarize my preferences."
    - Any request to change, book, assign, or negotiate about spaces or activities.

- "sql_query": For direct database or technical queries, such as:
    - "Show me all activities assigned to O1."
    - "List all residents."
    - "How many votes did activity X get?"
    - "Query the database for all playgrounds."
    - Any technical or data retrieval question.

- "other": For anything else, such as:
    - Greetings, small talk, or off-topic questions.

**Instructions:**
- Carefully read the user's message.
- Choose the most appropriate action from the list above.
- Return a JSON object with "action" and a brief "reasoning" field explaining your choice.

**Examples:**

User: "Why was yoga assigned to O2?"
Response:
{"action": "llm_nearby_space_qna", "reasoning": "User is asking for the reasoning behind an assignment."}

User: "I want to book the playground for Saturday."
Response:
{"action": "llm_negotiate", "reasoning": "User wants to book a space."}

User: "List all outdoor spaces."
Response:
{"action": "sql_query", "reasoning": "User is requesting a database listing."}

User: "Hello, how are you?"
Response:
{"action": "other", "reasoning": "Greeting, not a functional request."}

Return a JSON as identified per action like: {{"action": "...", "reasoning": "..."}}
User message: "{message}"
"""
    # Call your LLM (OpenAI, local, etc.)
    action_json_str = suggest_actions_from_request(llm_prompt)
    try:
        action_json = json.loads(action_json_str)
        action = action_json.get("action", "other")
    except Exception:
        action = "other"

    # 2. Route based on LLM's suggestion
    if action == "llm_nearby_space_qna":
        # Call your QnA logic
        result = llm_nearby_space_qna(house_key, message)
        return jsonify({'result': result, 'params': ''})
    elif action == "llm_negotiate":
        # Call your negotiation logic
        llm_input = f"House key: {house_key}\n{message}" if house_key else message
        action_json_str = suggest_actions_from_request(llm_input)
        try:
            action_json = json.loads(action_json_str)
        except Exception:
            action_json = {"error": "Invalid LLM output", "raw": action_json_str}
        result = route_action(action_json)
        response_text = result.get('result', '')
        params_text = result.get('params', '')
        return jsonify({'result': response_text, 'params': params_text})
    elif action == "sql_query":
        answer = answer_user_question(message, db_path="sql/gh_data.db")
        return jsonify({'result': answer, 'params': ''})
    else:
        return jsonify({'result': "Sorry, I didn't understand your request.", 'params': ''})

def run_flask():
    app.run(debug=False, use_reloader=False)  # Run Flask server in a separate thread

if __name__ == '__main__':
    # Start Flask server in a separate thread
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    # Start PyQt application
    app = QApplication(sys.argv)
    app.setStyleSheet("QWidget { font-size: 14px; }") 
    # Import or define FlaskClientChatUI before using it
    from ui_pyqt_spaceqna import FlaskClientChatUI   # Make sure this import points to the correct file/module

    window = FlaskClientChatUI()
    window.show()
    sys.exit(app.exec_())




#Functions
# Update llm_nearby_space_qna to accept house_key and question as arguments

def llm_nearby_space_qna(house_key, question):
    if not house_key or not question:
        return "Missing 'house_key' or 'question' in request."
    try:
        # 1. If user asks "why" or "reason" for a specific space/activity
        match = re.search(r'(?:why|reason).*?(O\d+)', question, re.IGNORECASE)
        if match:
            space_id = match.group(1)
            geometries, thresh, green, usability, voting, distances, personas = load_csvs()
            reasoning = explain_activity_for_space(
                space_id, question, geometries, thresh, green, usability, voting, distances, personas
            )
            return reasoning
        # 2. If user asks for closest/nearest outdoor spaces or spaces on my floor
        if re.search(r'(closest|nearest|nearby).*outdoor', question, re.IGNORECASE) or \
           re.search(r'outdoor.*(spaces|areas).*on my floor', question, re.IGNORECASE):
            import sqlite3
            import pandas as pd
            conn = sqlite3.connect('sql/gh_data.db')
            distances = pd.read_sql_query("SELECT * FROM resident_distances", conn)
            conn.close()
            assignments = pd.read_csv('llm_reasoning/llm_activity_assignments.csv')
            voting = pd.read_csv('resident_data/voting_weights.csv')
            if house_key not in distances.columns:
                return f"No distances found for house key {house_key}."
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
            return response_text
        # 3. Otherwise, use the general LLM prompt (main logic)
        import sqlite3
        import pandas as pd
        conn = sqlite3.connect('sql/gh_data.db')
        activity_space = pd.read_sql_query("SELECT * FROM activity_space", conn)
        distances = pd.read_sql_query("SELECT * FROM resident_distances", conn)
        conn.close()
        voting = pd.read_csv('resident_data/voting_weights.csv')
        assignments = pd.read_csv('llm_reasoning/llm_activity_assignments.csv')
        if house_key not in distances.columns:
            return f"No distances found for house key {house_key}."
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
        return reply
    except Exception as e:
        return f"Error: {str(e)}"