from flask import Flask, request, jsonify
from server.config import *
from llm_calls import *
from utils.rag_utils import answer_from_knowledge
from sql_main import answer_sql_question
from question_router import route_question
from llm_reasoning_test import *
from llm_negotiation import *
import json
import time
import re
import os
import pandas as pd
import sqlite3

app = Flask(__name__)

# --- Utility: General Q&A logic ---
conversation_history = []

geometry_command = {"geometry_command": 0}
geometry_all_visible = False

def print_answer(label, answer):
    print(f"\n[{label.upper()} RESULT]")
    print(answer)
    print("-" * 50)

def answer_general_question(user_message, conversation_history=None):
    if conversation_history is None:
        conversation_history = []
    routed_parts = route_question(user_message)
    combined_answer = ""
    for part in routed_parts:
        destination = part["destination"]
        part_text = part["text"]

        if destination == "sql":
            answer = answer_sql_question(part_text)
            combined_answer += f"\n(SQL Answer for: \"{part_text}\")\n{answer}"
        elif destination == "knowledge":
            embedding_file = part.get("embedding_file")
            answer = answer_from_knowledge(part_text, embedding_file, conversation_history)
            combined_answer += f"\n(Knowledge Answer for: \"{part_text}\")\n{answer}"
        else:
            fallback = "Sorry, I couldn't understand that part."
            combined_answer += f"\n(Error for: \"{part_text}\")\n{fallback}"
    return combined_answer.strip()

# ---- Flask API ----
app = Flask(__name__)

@app.route('/general_question', methods=['POST'])
def handle_general_question():
    try:
        start = time.time()
        print("Received question request")
        data = request.get_json()
        print("Data received:", data)
        user_message = data.get('question', '')
        conv_hist = data.get('conversation_history', [])
        answer = answer_general_question(user_message, conv_hist)

        # --- User-friendly error handling ---
        friendly_message = "I'm sorry but I was not able to find any relevant information to answer your question. Please, try again."
        # You can add more patterns as needed
        if (
            not answer.strip() or
            "no such table" in answer.lower() or
            "error" in answer.lower() or
            "exception" in answer.lower() or
            "traceback" in answer.lower() or
            "None" == answer.strip()
        ):
            answer = friendly_message

        conv_hist.append({"role": "user", "content": user_message})
        conv_hist.append({"role": "assistant", "content": answer})
        print("Returning response, elapsed:", time.time() - start, "seconds")
        return jsonify({'response': answer, 'conversation_history': conv_hist})
    except Exception as e:
        print("Error:", e)
        return jsonify({'response': f"Server error: {e}", 'conversation_history': []}), 500

@app.route('/set_geometry', methods=['POST'])
def set_geometry():
    global geometry_command, geometry_all_visible
    data = request.get_json()
    if data.get("geometry_command") == "toggle_all":
        geometry_all_visible = not geometry_all_visible
        geometry_command = {"geometry_command": "show_all" if geometry_all_visible else "hide_all"}
        return jsonify({"status": "ok", "visible": geometry_all_visible})
    elif data.get("geometry_command") == "show_all":
        geometry_all_visible = True
        geometry_command = {"geometry_command": "show_all"}
        return jsonify({"status": "ok", "visible": True})
    elif data.get("geometry_command") == "hide_all":
        geometry_all_visible = False
        geometry_command = {"geometry_command": "hide_all"}
        return jsonify({"status": "ok", "visible": False})
    else:
        # This is for specific geometry (level, space_info, apt_info)
        geometry_command = data
        return jsonify({"status": "ok"})

@app.route('/get_geometry', methods=['GET'])
def get_geometry():
    return jsonify(geometry_command)







# --- Geometry endpoints ---
last_geometry_key = {"key": None}
geometry_command = {"geometry_command": 0}
geometry_all_visible = False


@app.route('/show_geometry_by_key', methods=['POST'])
def show_geometry_by_key():
    data = request.get_json()
    key = data.get("key")
    if isinstance(key, str):
        key = key.strip()
    else:
        key = None
    if not key:
        return jsonify({"error": "No key provided"}), 400
    last_geometry_key["key"] = key
    return jsonify({"status": "ok", "key": key})

@app.route('/hide_geometry_by_key', methods=['POST'])
def hide_geometry_by_key():
    last_geometry_key["key"] = None
    return jsonify({"status": "ok", "key": None})

@app.route('/get_geometry_key', methods=['GET'])
def get_geometry_key():
    return jsonify({"key": last_geometry_key["key"]})

# --- JSON transfer for Grasshopper (if needed) ---
JSON_FILE = "llm_reasoning/llm_assignments.json"
@app.route('/get_json', methods=['GET', 'POST'])
def get_json():
    file_path = JSON_FILE
    if request.method == 'POST':
        data = request.get_json()
        file_path = data.get('file_path', JSON_FILE)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)})

# --- Negotiation endpoints ---
@app.route('/llm_negotiate', methods=['POST'])
def llm_negotiate():
    data = request.get_json()
    house_key = data.get('house_key', '')
    query = data.get('query', '')
    last_context = data.get('last_context', None)
    negotiation_result = negotiation_flow(query, house_key, last_context)
    return jsonify(negotiation_result)

@app.route('/llm_negotiate_action', methods=['POST'])
def llm_negotiate_action():
    data = request.get_json()
    action_name = data.get('action')
    parameters = data.get('parameters', {})
    house_key = data.get('house_key', '')
    query = data.get('query', '')
    last_context = data.get('last_context', None)
    result = route_action({'action': action_name, 'parameters': parameters, 'house_key': house_key, 'query': query})
    result_text = result.get('result', '')
    params_text = result.get('params', '')
    context = data.get('last_context', {})
    return jsonify({
        'result': result_text,
        'params': params_text,
        'context': context
    })
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# --- Robust Nearby Space Q&A endpoint (for Q&A tab, with context, history, and LLM prompt logic) ---
conversation_histories = {}
last_contexts = {}
@app.route('/llm_nearby_space_qna', methods=['POST'])
def llm_nearby_space_qna():
    data = request.get_json()
    house_key = data.get("house_key")
    question = data.get("question", "")
    if not house_key or not question:
        return jsonify({"error": "Missing 'house_key' or 'question' in request."}), 400


    #DISTANCE RELATED QUESTIONS
    distances_all = pd.read_csv('resident_data/resident_distances_all.csv')
    distances_all.columns = [c.strip() for c in distances_all.columns]
    distances_all['Source Node'] = distances_all['Source Node'].astype(str).str.strip()
    # If you use SQLite, do the same:
    conn = sqlite3.connect('sql/gh_data.db')
    distances = pd.read_sql_query("SELECT * FROM resident_distances", conn)
    distances['Outdoor Space'] = distances['Outdoor Space'].astype(str).str.strip()
    conn.close()

    if re.search(r"\b(how far|distance from my house|distance to my house|distance)\b", question.lower()):
        response_text = (
            "To check the distance from your house to any outdoor space, "
            "please refer to the file: resident_data/resident_distances_all.csv"
        )
        return jsonify({"response": response_text})


    #ASSIGNED ACTIVITIES BY LLM
    assignments = pd.read_csv('llm_reasoning/llm_activity_assignments.csv')
    assignments['assigned_activity'] = assignments['assigned_activity'].astype(str).str.strip().str.lower()
    assignments['space_id'] = assignments['space_id'].astype(str).str.strip()




    # Try to find an activity mentioned in the question (case-insensitive)
    activity_names = assignments['assigned_activity'].unique()
    question_lower = question.lower()
    activity_mentioned = None
    for act in activity_names:
        if act in question_lower and re.search(r"\b(which|what|list|show)\b.*\b(space|area|building)", question_lower):
            activity_mentioned = act
            break

    if activity_mentioned:
        filtered = assignments[assignments['assigned_activity'] == activity_mentioned]
        # Only keep space_ids that start with 'O'
        filtered = filtered[filtered['space_id'].str.startswith('O')]
        if filtered.empty:
            response_text = f"No outdoor spaces are assigned to {activity_mentioned.title()}."
        else:
            # Build list with distances for each assigned space
            house_col = house_key.strip()
            space_distances = []
            for space_id in filtered['space_id']:
                space_id_str = str(space_id).strip()
                row = distances_all[distances_all['Source Node'] == space_id_str]
                if not row.empty and house_col in distances_all.columns:
                    dist = row.iloc[0][house_col]
                    try:
                        dist = float(dist)
                        dist_str = f"{dist:.1f}m"
                    except Exception:
                        dist_str = str(dist)
                else:
                    dist_str = "unknown"
                # This will always show O5, O10, etc.
                space_distances.append(f"{space_id_str} ({dist_str})")
            response_text = f"The following outdoor spaces are assigned to {activity_mentioned.title()}: {', '.join(space_distances)}"
        return jsonify({"response": response_text})






        # --- Neighbors Q&A logic ---
    if re.search(r"\b(neighbor|neighbour|closest houses|who lives near|nearby residents|best matching neighbors|neighbors like me|similar neighbors|matching neighbors)\b", question.lower()):
        conn = sqlite3.connect('sql/gh_data.db')
        personas = pd.read_sql_query("SELECT * FROM personas_assigned", conn)
        conn.close()
        personas['resident_key'] = personas['resident_key'].astype(str).str.strip()
        house_key_str = str(house_key).strip()

        distances = pd.read_csv('resident_data/resident_distances_all.csv')
        distances.columns = [c.strip() for c in distances.columns]
        distances['Source Node'] = distances['Source Node'].astype(str).str.strip()

        if re.search(r"\b(best matching neighbors|neighbors like me|similar neighbors|matching neighbors)\b", question.lower()):
            # --- Best Matching Neighbors (by Persona) ---
            user_row = personas[personas['resident_key'] == house_key_str]
            if user_row.empty:
                return jsonify({"response": f"Cannot find your persona for house {house_key_str}."})
            
            user_persona = user_row.iloc[0]['resident_persona']
            matching_neighbors = personas[
                (personas['resident_persona'] == user_persona) &
                (personas['resident_key'] != house_key_str) &
                (personas['resident_key'].str.startswith('H'))  # 🔑 Only houses
            ]

            # Find row of distances for house_key
            if house_key_str not in distances['Source Node'].values:
                return jsonify({"response": f"Cannot find distances for your house ({house_key_str})."})

            dist_row = distances[distances['Source Node'] == house_key_str].iloc[0]

            neighbor_distances = []
            for _, row in matching_neighbors.iterrows():
                neighbor_key = str(row['resident_key']).strip()
                if not neighbor_key.startswith('H'):
                    continue  # 🧱 Skip non-house keys
                dist_val = dist_row.get(neighbor_key, "unknown")

                pop = row.get('resident_population', 'unknown')
                age = row.get('age', 'unknown')
                status = row.get('tenant/owner', 'unknown')

                neighbor_distances.append(f"{neighbor_key}: {dist_val}m away | Population: {pop} | Age: {age} | {status}")
            
            if not neighbor_distances:
                response_text = "No matching neighbors found."
            else:
                response_text = f"Neighbors with the same persona ({user_persona}):\n" + "\n".join(neighbor_distances)

            return jsonify({"response": response_text})
        
        else:
            # --- Nearest Physical Neighbors ---
            if house_key_str not in distances['Source Node'].values:
                return jsonify({"response": f"Cannot find distances for your house ({house_key_str})."})

            row = distances[distances['Source Node'] == house_key_str].iloc[0]

            neighbor_distances = []
            for col in distances.columns:
                if col == 'Source Node' or col == house_key_str or not col.startswith('H'):
                    continue  # 🧱 Skip self and non-house columns
                dist = row[col]
                neighbor_key = col
                persona_row = personas[personas['resident_key'] == neighbor_key]
                if not persona_row.empty:
                    persona = persona_row.iloc[0].get('resident_persona', 'unknown')
                    pop = persona_row.iloc[0].get('resident_population', 'unknown')
                    age = persona_row.iloc[0].get('age', 'unknown')
                    status = persona_row.iloc[0].get('tenant/owner', 'unknown')
                else:
                    persona, pop, age, status = "unknown", "unknown", "unknown", "unknown"
                neighbor_distances.append({
                    "neighbor_key": neighbor_key,
                    "dist": dist,
                    "persona": persona,
                    "pop": pop,
                    "age": age,
                    "status": status
                })

            # Sort and show closest 5
            neighbor_distances = sorted(neighbor_distances, key=lambda x: float(x["dist"]) if str(x["dist"]).replace('.','',1).isdigit() else float('inf'))
            neighbor_info = [
                f"{n['neighbor_key']}: {n['dist']}m away | Persona: {n['persona']} | Population: {n['pop']} | Age: {n['age']} | {n['status']}"
                for n in neighbor_distances[:5]
            ]
            response_text = "Closest 5 neighbors:\n" + "\n".join(neighbor_info)
            return jsonify({"response": response_text})






    # --- Conversation history logic ---
    history = conversation_histories.setdefault(house_key, [])
    history.append({"role": "user", "content": question})
    # --- Conversational context resolution ---
    last_space = None
    last_activity = None
    for msg in reversed(history[:-1]):
        content = msg.get("content", "")
        space_match = re.search(r"O\d+", content)
        if space_match and not last_space:
            last_space = space_match.group(0)
        activity_match = re.search(r"activity\s*:?\s*([A-Za-z0-9_\- ]+)", content)
        if activity_match and not last_activity:
            last_activity = activity_match.group(1).strip()
        if last_space and last_activity:
            break
    last_contexts[house_key] = {"space": last_space, "activity": last_activity}
    resolved_question = question
    if last_space:
        resolved_question = re.sub(r"\bthis space\b", last_space, resolved_question, flags=re.IGNORECASE)
    if last_activity:
        resolved_question = re.sub(r"\bthis activity\b", last_activity, resolved_question, flags=re.IGNORECASE)
    question = resolved_question




    # --- Handle resident-specific voting preferences directly ---
    match_weights = re.search(r'my\s+(weights?|preferences|votes?)\s+for\s+(O\d+)', question, re.IGNORECASE)
    if match_weights:
        space_id = match_weights.group(2)
        try:
            voting = pd.read_csv('resident_data/voting_weights.csv')
            if not set(['resident','space','activity','weight']).issubset(voting.columns):
                raise ValueError('Missing expected columns')
        except Exception:
            voting = pd.read_csv('resident_data/voting_weights.csv', header=None)
            voting.columns = ['resident','space','activity','distance','weight','role']
        resident_votes = voting[(voting['space'] == space_id) & (voting['resident'] == house_key)]
        if not resident_votes.empty:
            resident_votes = resident_votes[['activity','weight']].sort_values('activity')
            weights_list = [f"- {row['activity']}: {row['weight']}" for _, row in resident_votes.iterrows()]
            weights_text = "\n".join(weights_list)
            response_text = f"Your preferences for {space_id} are:\n{weights_text}"
        else:
            response_text = f"No voting preferences found for you ({house_key}) in {space_id}."
        history.append({"role": "assistant", "content": response_text})
        return jsonify({"response": response_text})
    try:
        match = re.search(r'(?:why|reason).*?(O\d+)', question, re.IGNORECASE)
        if match:
            space_id = match.group(1)
            geometries, thresh, green, usability, voting, distances, personas = load_csvs()
            reasoning = explain_activity_for_space(
                space_id, question, geometries, thresh, green, usability, voting, distances, personas
            )
            history.append({"role": "assistant", "content": reasoning})
            return jsonify({"response": reasoning})
        if re.search(r'(closest|nearest|nearby).*outdoor', question, re.IGNORECASE) or \
           re.search(r'outdoor.*(spaces|areas).*on my floor', question, re.IGNORECASE):
            conn = sqlite3.connect('sql/gh_data.db')
            distances = pd.read_sql_query("SELECT * FROM resident_distances", conn)
            conn.close()
            assignments = pd.read_csv('llm_reasoning/llm_activity_assignments.csv')
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
            history.append({"role": "assistant", "content": response_text})
            return jsonify({"response": response_text})
        conn = sqlite3.connect('sql/gh_data.db')
        activity_space = pd.read_sql_query("SELECT * FROM activity_space", conn)
        distances = pd.read_sql_query("SELECT * FROM resident_distances", conn)
        conn.close()
        voting = pd.read_csv('resident_data/voting_weights.csv')
        assignments = pd.read_csv('llm_reasoning/llm_activity_assignments.csv')

        conn = sqlite3.connect('sql/gh_data.db')  # Use your actual DB path
        # Load personas
        personas = pd.read_sql_query("SELECT * FROM personas_assigned", conn)
        conn.close()


        if house_key not in distances.columns:
            return jsonify({"error": f"No distances found for house key {house_key}."}), 404
        user_persona = None
        user_persona_details = None
        if personas is not None and 'resident_key' in personas.columns and 'resident_persona' in personas.columns:
            persona_row = personas[personas['resident_key'].astype(str) == str(house_key)]
            if not persona_row.empty:
                user_persona = persona_row.iloc[0]['resident_persona']
                user_persona_details = persona_row.iloc[0].to_dict()
        persona_activities = []
        if user_persona is not None and 'assigned_activity' in assignments.columns and 'resident_persona' in assignments.columns:
            persona_activities = assignments[assignments['resident_persona'] == user_persona]['assigned_activity'].unique().tolist()
        elif user_persona is not None and 'assigned_activity' in assignments.columns:
            persona_activities = assignments['assigned_activity'].unique().tolist()
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
        persona_activities_text = ", ".join(persona_activities) if persona_activities else "No data"
        persona_details_text = "No data"
        if user_persona_details:
            persona_details_text = ", ".join([f"{k}: {v}" for k, v in user_persona_details.items()])
        all_space_activities = []
        for _, row in assignments.iterrows():
            space_id = row['space_id'] if 'space_id' in row else row.get('id', None)
            assigned_activity = row['assigned_activity'] if 'assigned_activity' in row else row.get('activity', None)
            if space_id and assigned_activity:
                all_space_activities.append(f"| {space_id} | {assigned_activity} |")
        all_space_activities_text = "\n".join(["| Space ID | Assigned Activity |", "|----------|-------------------|"] + all_space_activities) if all_space_activities else "No data"
        messages = history.copy()
        messages.append({
            "role": "system",
            "content": f"""
You are a community advisor helping a resident understand the outdoor spaces near them.

### Resident info:
- House key: {house_key}
- Persona: {user_persona or 'Unknown'}
- Persona details: {persona_details_text}

### Question:
{question}

### Nearby spaces, assigned activities, and voting preferences for each activity per outdoor space and weights:
{space_summaries_text}

### Activities assigned to other spaces for this user's persona:
{persona_activities_text}

### All outdoor spaces and their assigned activities (full list):
{all_space_activities_text}

### Your task:
- Use all the information above to answer the resident's question, whether it is about preferences, assignments, reasoning, or general context.
- If the question is about why a space does not match preferences, or why an activity is assigned, explain using the voting data and assignments.
- If the question is about the list, provide the list.
- If the question is about how decisions are made, explain the process using the data.
- If the question is about preferences, summarize the relevant voting or assignment data.
- If the question is about spaces with a specific activity (e.g., 'Sports'), go through the full list of assigned activities in the table above and extract every space ID that has the assigned activity exactly matching 'Sports'. Then list them all. Do not guess.
- If the question is something else, use your best judgment to answer using all the context above.
Be concise and use plain language.
"""
        })
        response = requests.post(
            "http://localhost:1234/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json={
                "model": "local-model",
                "messages": messages,
                "temperature": 0.7
            }
        )
        reply = response.json()["choices"][0]["message"]["content"]
        history.append({"role": "assistant", "content": reply})
        return jsonify({"response": reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(port=5000, debug=True)
