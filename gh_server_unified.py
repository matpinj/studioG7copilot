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

import random
from functools import lru_cache
import threading
import requests


# flask_server_process_geometry = None
# flask_server_process_unified = None

# --- Utility: General Q&A logic ---
conversation_history = []

# Global variables for cached data
_data_cache = {}
_cache_lock = threading.Lock()

def initialize_data_cache():
    """Load all data once at startup and cache it globally"""
    global _data_cache
    
    print("🚀 Loading data cache at startup...")
    start_time = time.time()
    
    try:
        # Load database data once (prefer DB over CSV for consistency)
        print("  🗄️  Loading database tables...")
        conn = sqlite3.connect('sql/gh_data.db')
        
        # Replace: distances_all = pd.read_csv('resident_data/resident_distances_all.csv')
        _data_cache['distances_all'] = pd.read_sql_query("SELECT * FROM resident_distances_all", conn)
        _data_cache['distances_all'].columns = [c.strip() for c in _data_cache['distances_all'].columns]
        _data_cache['distances_all']['Source Node'] = _data_cache['distances_all']['Source Node'].astype(str).str.strip()
        
        # Load other distances table
        _data_cache['distances'] = pd.read_sql_query("SELECT * FROM resident_distances", conn)
        _data_cache['distances']['Outdoor Space'] = _data_cache['distances']['Outdoor Space'].astype(str).str.strip()
        
        # Load activity space geometries (needed by llm_reasoning_test.py)
        _data_cache['geometries'] = pd.read_sql_query("SELECT * FROM activity_space", conn)
        _data_cache['geometries'].rename(columns={"key": "id"}, inplace=True)
        _data_cache['geometries']["id"] = _data_cache['geometries']["id"].apply(lambda x: f"O{x}" if not str(x).startswith("O") else str(x))
        
        # Load personas
        _data_cache['personas'] = pd.read_sql_query("SELECT * FROM personas_assigned", conn)
        _data_cache['personas']['resident_key'] = _data_cache['personas']['resident_key'].astype(str).str.strip()
        
        # Load activity space
        _data_cache['activity_space'] = pd.read_sql_query("SELECT * FROM activity_space", conn)
        
        conn.close()
        
        # Load CSV data that doesn't have database tables yet
        print("  📊 Loading remaining CSV files...")
        
        _data_cache['assignments'] = pd.read_csv('llm_reasoning/llm_activity_assignments.csv')
        _data_cache['assignments']['assigned_activity'] = _data_cache['assignments']['assigned_activity'].astype(str).str.strip().str.lower()
        _data_cache['assignments']['space_id'] = _data_cache['assignments']['space_id'].astype(str).str.strip()
        
        _data_cache['voting'] = pd.read_csv('resident_data/voting_weights.csv')
        
        # Load ML prediction data (needed by llm_reasoning_test.py)
        try:
            multi = pd.read_csv('gh_data/gh_data_multiple_activities.csv')
            multi = multi.rename(columns={'key': 'id'})
            _data_cache['thresh'] = multi[['id', 'activity']].rename(columns={'activity': 'predicted_activities'})
            _data_cache['green'] = multi[['id', 'green_suitability']].rename(columns={'green_suitability': 'green_prediction'})
            _data_cache['usability'] = multi[['id', 'usability']].rename(columns={'usability': 'usability_prediction'})
        except FileNotFoundError:
            # Fallback: create from geometries if multi CSV doesn't exist
            print("  ⚠️  gh_data_multiple_activities.csv not found, creating from geometries...")
            _data_cache['thresh'] = _data_cache['geometries'][['id', 'activity']].rename(columns={'activity': 'predicted_activities'})
            _data_cache['green'] = _data_cache['geometries'][['id', 'green_suitability']].rename(columns={'green_suitability': 'green_prediction'})
            _data_cache['usability'] = _data_cache['geometries'][['id', 'usability']].rename(columns={'usability': 'usability_prediction'})
        
        # Pre-process activity names for faster lookup
        activity_names = _data_cache['assignments']['assigned_activity'].unique()
        _data_cache['activity_names_set'] = set(activity_names)
        
        print(f"✅ Data cache loaded successfully in {time.time() - start_time:.2f}s")
        
        # Initialize sub-module caches
        try:
            from llm_reasoning_test import initialize_reasoning_cache
            initialize_reasoning_cache()
        except:
            print("  ⚠️  Could not initialize reasoning cache")
            
        try:
            from llm_negotiation import initialize_negotiation_cache
            initialize_negotiation_cache()
        except:
            print("  ⚠️  Could not initialize negotiation cache")
        
    except Exception as e:
        print(f"❌ Error loading data cache: {e}")
        raise

def get_cached_data(key):
    """Thread-safe access to cached data"""
    with _cache_lock:
        return _data_cache.get(key)

@lru_cache(maxsize=128)
def get_resident_persona(house_key_str):
    """Cached lookup for resident persona"""
    personas = get_cached_data('personas')
    if personas is None:
        return None, None
        
    user_row = personas[personas['resident_key'] == house_key_str]
    if user_row.empty:
        return None, None
    
    user_persona = user_row.iloc[0]['resident_persona']
    user_persona_details = user_row.iloc[0].to_dict()
    return user_persona, user_persona_details

@lru_cache(maxsize=256)
def get_resident_distances(house_key_str):
    """Cached lookup for resident distances"""
    distances_all = get_cached_data('distances_all')
    if distances_all is None or house_key_str not in distances_all['Source Node'].values:
        return None
    return distances_all[distances_all['Source Node'] == house_key_str].iloc[0]

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

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

def create_natural_llm_prompt(question, house_key, context_data):
    """Create a more natural, conversational LLM prompt"""
    
    # Extract only the most relevant data
    nearby_spaces = context_data.get('nearby_spaces', [])
    user_persona = context_data.get('user_persona', 'Unknown')
    
    # Create a concise, natural prompt
    prompt = f"""You are a friendly community advisor helping {house_key}, a resident with the {user_persona} personality type.

Their question: "{question}"

Here's what I know about their area:
"""
    
    # Add only top 3-5 most relevant spaces
    if nearby_spaces:
        prompt += "Nearby outdoor spaces:\n"
        for space in nearby_spaces[:5]:  # Limit to 5 spaces
            prompt += f"• {space['space_id']} ({space['activity']}): {space['distance']:.1f}m away\n"
    
    prompt += f"""
Respond naturally and conversationally, like you're chatting with a neighbor. Be helpful and friendly. If they're asking about specific spaces or activities, focus on those. Keep your response concise but warm.

If they mention neighbors, talk about community and getting to know people. If they ask about distances or activities, give practical advice.

Don't list data - have a natural conversation!"""

    return prompt

# ============================================================================
# FLASK ROUTES
# ============================================================================

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
        geometry_command = data
        return jsonify({"status": "ok"})

@app.route('/get_geometry', methods=['GET'])
def get_geometry():
    return jsonify(geometry_command)

# Geometry endpoints
last_geometry_key = {"key": None}

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

# JSON transfer for Grasshopper
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

# Negotiation endpoints
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
    
    # Execute the action
    result = route_action({'action': action_name, 'parameters': parameters, 'house_key': house_key, 'query': query})
    
    # Helper function to convert numpy/pandas types to Python native types
    def convert_to_serializable(obj):
        if hasattr(obj, 'item'):  # numpy types
            return obj.item()
        elif hasattr(obj, 'tolist'):  # numpy arrays
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        else:
            return obj
    
    # Convert result to be JSON serializable
    result = convert_to_serializable(result)
    
    # Handle geometry changes
    if isinstance(result, dict) and result.get('geometry_changes_needed'):
        geometry_changes = result.get('geometry_changes', {})
        current_geometry = result.get('current_geometry', {})
        
        response_text = result.get('result', '') + "\n\nGeometry changes are needed. Would you like to proceed to the Geometry tab to make these changes?"
        
        return jsonify({
            'result': response_text,
            'params': str(parameters),
            'context': last_context or {},
            'geometry_changes_needed': True,
            'geometry_changes': convert_to_serializable(geometry_changes),
            'current_geometry': convert_to_serializable(current_geometry),
            'redirect_to_geometry': True
        })
    
    # Handle swap recommendations
    elif isinstance(result, dict) and 'swap_candidates' in result:
        return jsonify({
            'result': result.get('result', ''),
            'params': str(parameters),
            'context': last_context or {},
            'swap_candidates': convert_to_serializable(result.get('swap_candidates', []))
        })
    
    # Handle booking confirmations
    elif isinstance(result, dict) and 'booking_details' in result:
        return jsonify({
            'result': result.get('result', ''),
            'params': str(parameters),
            'context': last_context or {},
            'booking_details': convert_to_serializable(result.get('booking_details', {}))
        })
    
    # Standard response
    else:
        result_text = result.get('result', '') if isinstance(result, dict) else str(result)
        params_text = result.get('params', '') if isinstance(result, dict) else ''
        
        return jsonify({
            'result': result_text,
            'params': str(params_text),
            'context': last_context or {}
        })

# ============================================================================
# OPTIMIZED NEARBY SPACE Q&A ENDPOINT (COMPLETE VERSION)
# ============================================================================

conversation_histories = {}
last_contexts = {}

@app.route('/llm_nearby_space_qna', methods=['POST'])
def llm_nearby_space_qna():
    start_time = time.time()
    data = request.get_json()
    house_key = data.get("house_key")
    question = data.get("question", "")
    
    if not house_key or not question:
        return jsonify({"error": "Missing 'house_key' or 'question' in request."}), 400

    house_key_str = str(house_key).strip()
    
    try:
        # Quick distance check using cached data
        if re.search(r"\b(how far|distance from my house|distance to my house|distance)\b", question.lower()):
            return jsonify({
                "response": "To check the distance from your house to any outdoor space, please refer to the database table: resident_distances_all"
            })

        # Get cached data
        assignments = get_cached_data('assignments')
        distances_all = get_cached_data('distances_all')
        personas = get_cached_data('personas')
        voting = get_cached_data('voting')
        distances = get_cached_data('distances')
        
        print(f"⏱️  Data loading time: {time.time() - start_time:.3f}s")

        # ACTIVITY ASSIGNMENT QUERIES (optimized)
        activity_names_set = get_cached_data('activity_names_set')
        question_lower = question.lower()
        activity_mentioned = None
        
        for act in activity_names_set:
            if act in question_lower and re.search(r"\b(which|what|list|show)\b.*\b(space|area|building)", question_lower):
                activity_mentioned = act
                break

        if activity_mentioned:
            filtered = assignments[
                (assignments['assigned_activity'] == activity_mentioned) & 
                (assignments['space_id'].str.startswith('O'))
            ]
            
            if filtered.empty:
                response_text = f"No outdoor spaces are assigned to {activity_mentioned.title()}."
            else:
                house_col = house_key_str
                space_distances = []
                
                if house_col in distances_all.columns:
                    for space_id in filtered['space_id']:
                        space_id_str = str(space_id).strip()
                        row = distances_all[distances_all['Source Node'] == space_id_str]
                        if not row.empty:
                            dist = row.iloc[0][house_col]
                            try:
                                dist = float(dist)
                                dist_str = f"{dist:.1f}m"
                            except:
                                dist_str = str(dist)
                        else:
                            dist_str = "unknown"
                        space_distances.append(f"{space_id_str} ({dist_str})")
                    
                    response_text = f"The following outdoor spaces are assigned to {activity_mentioned.title()}: {', '.join(space_distances)}"
                else:
                    space_list = ', '.join(filtered['space_id'])
                    response_text = f"The following outdoor spaces are assigned to {activity_mentioned.title()}: {space_list}"
            
            return jsonify({"response": response_text})

    # --- NEW: Direct count for outdoor spaces ---
        if re.search(r'how many (outdoor )?spaces', question.lower()):
            try:
                conn = sqlite3.connect('sql/gh_data.db')
                df = pd.read_sql_query("SELECT * FROM activity_space", conn)
                conn.close()
                df['id'] = df['key'].apply(lambda x: f"O{x}" if not str(x).startswith("O") else str(x))
                outdoor_spaces = df[df['id'].str.startswith('O')]
                count = len(outdoor_spaces)
                example_list = []
                for _, row in outdoor_spaces.head(3).iterrows():
                    name = row.get('name') or row.get('activity') or "Outdoor Space"
                    example_list.append(f"{row['id']} ({name})")
                example_str = ", ".join(example_list)
                # Compose a context string for the LLM
                context = (
                    f"There are {count} outdoor spaces in the building. "
                    f"Examples include: {example_str}. "
                    "Here is the full list of outdoor spaces:\n" +
                    "\n".join([f"{row['id']}: {row.get('name') or row.get('activity') or 'Outdoor Space'}" for _, row in outdoor_spaces.iterrows()])
                )
                # Compose the LLM prompt
                messages = [
                    {"role": "system", "content": "You are a friendly community assistant who answers questions about the building using the provided data."},
                    {"role": "user", "content": f"{question}\n\nHere is some data you can use:\n{context}\n\nPlease answer the question in a warm, conversational way, using the data above. Do not make up numbers."}
                ]
                response = requests.post(
                    "http://localhost:1234/v1/chat/completions",
                    headers={"Content-Type": "application/json"},
                    json={
                        "model": "local-model",
                        "messages": messages,
                        "temperature": 0.7,
                        "max_tokens": 200
                    },
                    timeout=15
                )
                if response.status_code == 200:
                    reply = response.json()["choices"][0]["message"]["content"]
                else:
                    reply = f"There are {count} outdoor spaces in the building. For example: {example_str}."
                return jsonify({"response": reply})
            except Exception as e:
                return jsonify({"error": f"Failed to count outdoor spaces: {e}"})

        # --- LLM-powered: Count residents, conversational ---
        if re.search(r'how many residents', question.lower()):
            try:
                conn = sqlite3.connect('sql/gh_data.db')
                df = pd.read_sql_query("SELECT * FROM personas_assigned", conn)
                conn.close()
                count = df['resident_key'].nunique()
                # List a few example residents
                example_list = []
                for _, row in df.head(3).iterrows():
                    name = row.get('resident_name') or row.get('resident_key')
                    persona = row.get('resident_persona', '')
                    if persona:
                        example_list.append(f"{name} ({persona})")
                    else:
                        example_list.append(str(name))
                example_str = ", ".join(example_list)
                # Calculate persona percentages
                persona_counts = df['resident_persona'].value_counts()
                persona_percentages = [
                    f"{persona}: {count_} ({(count_/len(df))*100:.1f}%)"
                    for persona, count_ in persona_counts.items()
                ]
                persona_percent_str = "; ".join(persona_percentages)
                # Compose context for LLM
                context = (
                    f"There are {count} residents living in the building. "
                    f"Some of your neighbors include: {example_str}. "
                    f"Here's the breakdown by personality type: {persona_percent_str}."
                )
                # Compose LLM prompt
                messages = [
                    {"role": "system", "content": "You are a friendly community assistant who answers questions about the building using the provided data."},
                    {"role": "user", "content": f"{question}\n\nHere is some data you can use:\n{context}\n\nPlease answer the question in a warm, conversational way, using the data above. Do not make up numbers."}
                ]
                response = requests.post(
                    "http://localhost:1234/v1/chat/completions",
                    headers={"Content-Type": "application/json"},
                    json={
                        "model": "local-model",
                        "messages": messages,
                        "temperature": 0.7,
                        "max_tokens": 200
                    },
                    timeout=15
                )
                if response.status_code == 200:
                    reply = response.json()["choices"][0]["message"]["content"]
                else:
                    reply = context
                return jsonify({"response": reply})
            except Exception as e:
                return jsonify({"error": f"Failed to count residents: {e}"})






        # NEIGHBOR QUERIES (optimized with full original logic)
        if re.search(r"\b(neighbor|neighbour|closest houses|who lives near|nearby residents|best matching neighbors|neighbors like me|similar neighbors|matching neighbors)\b", question.lower()):
            print(f"🔍 Processing neighbor question: {question}")
            
            neighbor_keys_found = []

            if re.search(r"\b(best matching neighbors|neighbors like me|similar neighbors|matching neighbors)\b", question.lower()):
                print("🔍 DEBUG: Processing PERSONA-BASED neighbors")
                
                # Get user's persona
                user_persona, user_persona_details = get_resident_persona(house_key_str)
                if user_persona is None:
                    return jsonify({"response": f"I couldn't find your persona information for house {house_key_str}."})
                
                matching_neighbors = personas[
                    (personas['resident_persona'] == user_persona) &
                    (personas['resident_key'] != house_key_str) &
                    (personas['resident_key'].str.startswith('H'))
                ]
                
                if house_key_str not in distances_all['Source Node'].values:
                    return jsonify({"response": f"I couldn't find distance data for your house ({house_key_str})."})

                dist_row = distances_all[distances_all['Source Node'] == house_key_str].iloc[0]

                neighbor_list = []
                for _, row in matching_neighbors.iterrows():
                    neighbor_key = str(row['resident_key']).strip()
                    if not neighbor_key.startswith('H'):
                        continue
                        
                    neighbor_keys_found.append(neighbor_key)
                    dist_val = dist_row.get(neighbor_key, "unknown")
                    
                    neighbor_list.append({
                        'house': neighbor_key,
                        'distance': dist_val,
                        'persona': row.get('resident_persona', 'unknown'),
                        'population': row.get('resident_population', 'unknown'),
                        'age': row.get('age', 'unknown'),
                        'status': row.get('tenant/owner', 'unknown')
                    })
                
                neighbor_data = {
                    'type': 'persona_matching',
                    'user_persona': user_persona,
                    'neighbors': neighbor_list
                }
            
            else:
                print("🔍 DEBUG: Processing PHYSICAL proximity neighbors")
                
                if house_key_str not in distances_all['Source Node'].values:
                    return jsonify({"response": f"I couldn't find distance data for your house ({house_key_str})."})

                row = distances_all[distances_all['Source Node'] == house_key_str].iloc[0]

                neighbor_list = []
                for col in distances_all.columns:
                    if col == 'Source Node' or col == house_key_str or not col.startswith('H'):
                        continue
                        
                    try:
                        dist = row[col]
                        neighbor_key = col
                        neighbor_keys_found.append(neighbor_key)
                        
                        persona_row = personas[personas['resident_key'] == neighbor_key]
                        
                        if not persona_row.empty:
                            persona = persona_row.iloc[0].get('resident_persona', 'unknown')
                            pop = persona_row.iloc[0].get('resident_population', 'unknown')
                            age = persona_row.iloc[0].get('age', 'unknown')
                            status = persona_row.iloc[0].get('tenant/owner', 'unknown')
                        else:
                            persona, pop, age, status = "unknown", "unknown", "unknown", "unknown"
                            
                        try:
                            dist_float = float(dist)
                        except (ValueError, TypeError):
                            dist_float = float('inf')
                            
                        neighbor_list.append({
                            'house': neighbor_key,
                            'distance': dist_float,
                            'persona': persona,
                            'population': pop,
                            'age': age,
                            'status': status
                        })
                    except Exception as e:
                        print(f"🔍 Error processing neighbor {col}: {e}")
                        continue

                # Sort by distance and take closest 5
                neighbor_list = sorted(neighbor_list, key=lambda x: x['distance'])[:5]
                neighbor_keys_found = [n['house'] for n in neighbor_list]
                
                neighbor_data = {
                    'type': 'physical_proximity',
                    'neighbors': neighbor_list
                }

            # Create natural LLM response
            history = conversation_histories.setdefault(house_key, [])
            history.append({"role": "user", "content": question})
            
            # Format neighbor data for LLM
            if neighbor_data['type'] == 'persona_matching':
                neighbors_context = f"""
**You asked about neighbors who share your personality type.**

Your persona: {neighbor_data['user_persona']}

Neighbors with the same persona ({neighbor_data['user_persona']}):
"""
                for neighbor in neighbor_data['neighbors']:
                    neighbors_context += f"\n- **{neighbor['house']}**: {neighbor['distance']}m away, {neighbor['population']} people, age {neighbor['age']}, {neighbor['status']}"
                    
                if not neighbor_data['neighbors']:
                    neighbors_context += f"\nNo neighbors found with your exact persona type ({neighbor_data['user_persona']})."
                    
            else:  # physical_proximity
                neighbors_context = f"""
**You asked about your nearest neighbors.**

Your 5 closest neighbors:
"""
                for neighbor in neighbor_data['neighbors']:
                    dist_str = f"{neighbor['distance']:.1f}" if isinstance(neighbor['distance'], (int, float)) and neighbor['distance'] != float('inf') else "unknown"
                    neighbors_context += f"\n- **{neighbor['house']}**: {dist_str}m away, persona: {neighbor['persona']}, {neighbor['population']} people, age {neighbor['age']}, {neighbor['status']}"

            # Create LLM prompt
            messages = history.copy()
            messages.append({
                "role": "system", 
                "content": f"""
You are a friendly community advisor helping a resident understand their neighbors. Respond conversationally and helpfully.

**About the resident asking:** House {house_key}

**Their question:** "{question}"

**Neighbor Information:**
{neighbors_context}

**Instructions:**
1. Answer their question in a warm, conversational tone
2. Explain what the data means (distance, persona types, demographics)
3. If they asked about matching personas, explain what that persona type typically likes and why you'd get along
4. If they asked about closest neighbors, give them a sense of their immediate community diversity
5. Be encouraging about community building and getting to know neighbors
6. Use "you" and "your" to make it personal
7. Suggest ways they might connect with these neighbors if appropriate
8. Keep it natural and conversational, not like a data report

**Tone:** Friendly and informative, like a helpful neighbor who knows everyone in the community.
"""
            })

            try:
                response = requests.post(
                    "http://localhost:1234/v1/chat/completions",
                    headers={"Content-Type": "application/json"},
                    json={
                        "model": "local-model",
                        "messages": messages,
                        "temperature": 0.7,
                        "max_tokens": 200  # Reduced for faster response
                    },
                    timeout=15  # Reduced timeout
                )
                
                if response.status_code == 200:
                    reply = response.json()["choices"][0]["message"]["content"]
                else:
                    # Quick fallback without LLM
                    if neighbor_data['type'] == 'persona_matching':
                        reply = f"You share the '{neighbor_data['user_persona']}' personality type with some neighbors nearby! This means you likely have similar interests and could get along well."
                    else:
                        reply = f"Your closest neighbors are {', '.join([n['house'] for n in neighbor_data['neighbors'][:3]])}. They're a diverse group with different personalities and backgrounds."
                        
            except Exception as e:
                print(f"🔍 LLM failed: {e}")
                # Quick fallback
                if neighbor_data['type'] == 'persona_matching':
                    reply = f"You share the '{neighbor_data['user_persona']}' personality with some neighbors!"
                else:
                    reply = f"Your closest neighbors: {', '.join([n['house'] for n in neighbor_data['neighbors'][:3]])}"

            # Add key detection
            if neighbor_keys_found:
                detected_keys = "|".join(neighbor_keys_found)
                reply += f"\n**Detected keys:** {detected_keys}"

            history.append({"role": "assistant", "content": reply})
            return jsonify({"response": reply})

        # CONVERSATION HISTORY LOGIC (from original)
        history = conversation_histories.setdefault(house_key, [])
        history.append({"role": "user", "content": question})
        
        # Conversational context resolution
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

        # VOTING PREFERENCES (optimized)
        match_weights = re.search(r'my\s+(weights?|preferences|votes?)\s+for\s+(O\d+)', question, re.IGNORECASE)
        if match_weights:
            space_id = match_weights.group(2)
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

        # EXPLAIN ACTIVITY FOR SPACE (using optimized reasoning)
        match = re.search(r'(?:why|reason).*?(O\d+)', question, re.IGNORECASE)
        if match:
            space_id = match.group(1)
            geometries = get_cached_data('geometries')
            thresh = get_cached_data('thresh') 
            green = get_cached_data('green')
            usability = get_cached_data('usability')
            voting = get_cached_data('voting')
            distances = get_cached_data('distances')
            personas = get_cached_data('personas')
            
            reasoning = explain_activity_for_space(
                space_id, question, geometries, thresh, green, usability, voting, distances, personas
            )
            history.append({"role": "assistant", "content": reasoning})
            return jsonify({"response": reasoning})

        # CLOSEST/NEAREST OUTDOOR SPACES (from original)
        if re.search(r'(closest|nearest|nearby).*outdoor', question, re.IGNORECASE) or \
           re.search(r'outdoor.*(spaces|areas).*on my floor', question, re.IGNORECASE):
            
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

        # GENERAL QUERIES (complete original logic with optimization)
        print(f"⏱️  Processing time before LLM: {time.time() - start_time:.3f}s")
        
        if house_key not in distances.columns:
            return jsonify({"error": f"No distances found for house key {house_key}."}), 404

        # Get user persona and details
        user_persona, user_persona_details = get_resident_persona(house_key_str)
        
        # Get persona activities
        persona_activities = []
        if user_persona is not None and 'assigned_activity' in assignments.columns and 'resident_persona' in assignments.columns:
            persona_activities = assignments[assignments['resident_persona'] == user_persona]['assigned_activity'].unique().tolist()
        elif user_persona is not None and 'assigned_activity' in assignments.columns:
            persona_activities = assignments['assigned_activity'].unique().tolist()

        # Get nearby spaces with full details (from original)
        nearby = distances[["Outdoor Space", house_key]].rename(columns={house_key: "distance"})
        nearby = nearby.sort_values("distance").head(5)
        
        space_summaries = []
        for _, row in nearby.iterrows():
            space_id = row['Outdoor Space']
            distance = row['distance']

            # Get assigned activity
            assigned_row = assignments[assignments['space_id'] == space_id]
            if not assigned_row.empty:
                assigned_activity = assigned_row.iloc[0]['assigned_activity']
            else:
                assigned_activity = "Unknown"

            # Get voting summary
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

            # Get resident's votes
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

        # Get all space activities (from original)
        all_space_activities = []
        for _, row in assignments.iterrows():
            space_id = row['space_id'] if 'space_id' in row else row.get('id', None)
            assigned_activity = row['assigned_activity'] if 'assigned_activity' in row else row.get('activity', None)
            if space_id and assigned_activity:
                all_space_activities.append(f"| {space_id} | {assigned_activity} |")
        
        all_space_activities_text = "\n".join(["| Space ID | Assigned Activity |", "|----------|-------------------|"] + all_space_activities) if all_space_activities else "No data"

        # Create comprehensive LLM prompt (from original)
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

        try:
            response = requests.post(
                "http://localhost:1234/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                json={
                    "model": "local-model",
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 300
                },
                timeout=15
            )
            
            if response.status_code == 200:
                reply = response.json()["choices"][0]["message"]["content"]
            else:
                reply = "I'm having trouble processing your question right now. Could you try rephrasing it?"
                
        except Exception as e:
            print(f"LLM request failed: {e}")
            reply = "Sorry, I'm experiencing some technical difficulties. Please try again."

        # Store in conversation history
        history.append({"role": "assistant", "content": reply})
        
        # Keep history manageable
        if len(history) > 20:
            history = history[-20:]
            conversation_histories[house_key] = history

        print(f"⏱️  Total processing time: {time.time() - start_time:.3f}s")
        return jsonify({"response": reply})

    except Exception as e:
        print(f"Error in llm_nearby_space_qna: {e}")
        return jsonify({"error": str(e)}), 500

# ============================================================================
# STARTUP AND COMPATIBILITY
# ============================================================================

# Load data once at startup - keep this for backward compatibility with original structure
try:
    print("DEBUG: Loading CSV data for backward compatibility...")
    # This preserves the original try/catch pattern from your code
    # but now uses cached data from initialize_data_cache()
    geometries = None  # Will be loaded in cache
    thresh = None      # Will be loaded in cache  
    green = None       # Will be loaded in cache
    usability = None   # Will be loaded in cache
    voting_data = None # Will be loaded in cache
    distances_data = None # Will be loaded in cache
    personas = None    # Will be loaded in cache
    print("DEBUG: Successfully loaded CSV data at startup")
except Exception as e:
    print(f"DEBUG: Failed to load CSV data at startup: {e}")
    geometries = thresh = green = usability = voting_data = distances_data = personas = None


# def _get_server_script_path_geometry():
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     return os.path.join(current_dir, "geometry_mod", "gh_server_geometry.py")

# def _get_server_script_path_unified():
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     return os.path.join(current_dir, "gh_server_unified.py")

# def start_flask_servers():
#     global flask_server_process_geometry, flask_server_process_unified
#     geometry_script = _get_server_script_path_geometry()
#     unified_script = _get_server_script_path_unified()
#     if os.path.exists(geometry_script):
#         flask_server_process_geometry = subprocess.Popen([sys.executable, geometry_script])
#         print(f"Started geometry server: {geometry_script} (PID: {flask_server_process_geometry.pid})")
#     else:
#         print(f"Geometry server script not found at {geometry_script}")
#     if os.path.exists(unified_script):
#         flask_server_process_unified = subprocess.Popen([sys.executable, unified_script])
#         print(f"Started unified server: {unified_script} (PID: {flask_server_process_unified.pid})")
#     else:
#         print(f"Unified server script not found at {unified_script}")

# def stop_flask_servers():
#     global flask_server_process_geometry, flask_server_process_unified
#     for proc in [flask_server_process_geometry, flask_server_process_unified]:
#         if proc:
#             print(f"Stopping Flask server with PID: {proc.pid}...")
#             proc.terminate()
#             proc.wait(timeout=60)
#             if proc.poll() is None:
#                 print("Server did not terminate gracefully, killing...")
#                 proc.kill()
#             print("Flask server stopped.")

# # Call this before QApplication is created:
# start_flask_servers()
# atexit.register(stop_flask_servers)


if __name__ == '__main__':
    # Initialize data cache before starting the server
    initialize_data_cache()
    print("🚀 Starting Flask server...")
    app.run(port=5000, debug=True, use_reloader=False)
