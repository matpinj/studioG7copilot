import sys
import os
import demjson3
import re
import json
import pandas as pd
import string

# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sql_calls import get_space_details_as_string, get_dB_schema, format_dB_context, fetch_sql, execute_sql_query
from llm_calls import suggest_geometric_variations, generate_sql_query, build_answer
from utils.rag_utils import sql_rag_call

# Define paths for data sources
GH_DATA_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "sql", "gh_data_for_geometry.db")
LLM_ACTIVITY_ASSIGNMENTS_CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "llm_reasoning", "llm_activity_assignments.csv")
VOTING_WEIGHTS_CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "resident_data", "voting_weights.csv")
ML_ACTIVITY_LOGIC_CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "preset", "ml_activity_logic.csv")  # <-- Add this line

# Cache for loaded CSV data
_loaded_llm_activity_assignments_df = None
_loaded_voting_weights_df = None

def _load_csv_data(csv_path, df_cache_attr_name):
    """Helper function to load and cache a CSV file."""
    global _loaded_llm_activity_assignments_df, _loaded_voting_weights_df
    df_cache = globals()[df_cache_attr_name]
    if df_cache is None:
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                if 'space_id' in df.columns:
                    df['space_id'] = df['space_id'].astype(str)
                globals()[df_cache_attr_name] = df
                print(f"Loaded data from {csv_path}")
            except Exception as e:
                print(f"Error loading {csv_path}: {e}")
                globals()[df_cache_attr_name] = pd.DataFrame()
        else:
            print(f"Warning: CSV not found at {csv_path}")
            globals()[df_cache_attr_name] = pd.DataFrame()
    return globals()[df_cache_attr_name]

def load_llm_activity_assignments_df():
    return _load_csv_data(LLM_ACTIVITY_ASSIGNMENTS_CSV_PATH, "_loaded_llm_activity_assignments_df")

def load_voting_weights_df():
    return _load_csv_data(VOTING_WEIGHTS_CSV_PATH, "_loaded_voting_weights_df")

def get_all_table_names(db_path):
    query = "SELECT name FROM sqlite_master WHERE type='table';"
    result = execute_sql_query(db_path, query, ())
    return [row[0] for row in result] if result else []

def get_relevant_db_context(space_id, resident_key, other_resident_keys):
    """
    Returns a string with:
    - All rows from activity_space (excluding unwanted columns)
    - All rows from personas_assigned
    - Only relevant rows from resident_distances_all (for the given space and residents)
    """
    # List of columns to keep from activity_space (matching your schema)
    activity_space_columns = [
        "space_id", "orientation", "area", "level", "open_sides",
        "wind_exposure", "usability", "privacy_score",
        "slab_extension_limit_sqm", "longest edge length"
    ]
    activity_space_cols_str = ", ".join([f'"{col}"' for col in activity_space_columns])
    activity_space_rows = execute_sql_query(
        GH_DATA_DB_PATH, f"SELECT {activity_space_cols_str} FROM activity_space;", ()
    )

    personas_rows = execute_sql_query(GH_DATA_DB_PATH, "SELECT * FROM personas_assigned;", ())

    # Get relevant distances
    relevant_keys = [resident_key] + list(other_resident_keys)
    placeholders = ",".join([f'"{k}"' for k in relevant_keys])
    distance_query = f'SELECT "Source Node", {placeholders} FROM resident_distances_all WHERE "Source Node" = ?'
    distance_rows = execute_sql_query(GH_DATA_DB_PATH, distance_query, (space_id,))
    context = []
    context.append(f"Table: activity_space\nRows:\n{activity_space_rows}\n")
    context.append(f"Table: personas_assigned\nRows:\n{personas_rows}\n")
    context.append(f"Table: resident_distances_all (filtered)\nRows:\n{distance_rows}\n")
    return "\n".join(context)
def get_resident_distance_from_db(space_id, resident_key):
    """
    Fetch the distance from a resident to a space using the space's id.
    """
    query = f"""
        SELECT "{resident_key}"
        FROM resident_distances_all
        WHERE "Source Node" = ?
    """
    result = execute_sql_query(GH_DATA_DB_PATH, query, (space_id,))
    if result and result[0] and pd.notna(result[0][0]):
        return result[0][0]
    return "N/A"

def get_resident_status_from_db(resident_key):
    """
    Fetch the resident's status (owner/tenant) for a given space from the personas_assigned table.
    Adjust the column name if needed.
    """
    query = """
        SELECT [tenant/owner]
        FROM personas_assigned
        WHERE resident_key = ? 
    """
    result = execute_sql_query(GH_DATA_DB_PATH, query, (resident_key,))
    if result and result[0]:
        return result[0][0]
    return None

def get_resident_persona_from_db(resident_key):
    query = """
        SELECT resident_persona
        FROM personas_assigned
        WHERE resident_key = ?
    """
    result = execute_sql_query(GH_DATA_DB_PATH, query, (resident_key,))
    if result and result[0]:
        return result[0][0]
    return None

def get_current_activity_in_space(space_id):
    """
    Fetch the current assigned activity for a space from llm_activity_assignments.csv.
    """
    df = load_llm_activity_assignments_df()
    if df is not None and not df.empty:
        row = df[df['space_id'] == str(space_id)]
        if not row.empty:
            return row.iloc[0]['assigned_activity']
    return None

def get_activity_logic_for_activity(activity_name):
    df = pd.read_csv(ML_ACTIVITY_LOGIC_CSV_PATH)
    filtered = df[df['activity'].str.lower() == activity_name.lower()]
    if filtered.empty:
        return "No ML activity logic found for this activity."
    return filtered.to_string(index=False)



def get_intelligent_geometric_suggestions(
    space_id: str,
    resident_key: str,
    user_question: str = None,
    desired_activity_for_space: str = None
) -> str:
    resident_key_str = str(resident_key)
    space_id_str = str(space_id)
    voting_df = load_voting_weights_df()

    # Always define space_details_str before any return or use
    space_details_str = get_space_details_as_string(
        GH_DATA_DB_PATH,
        space_id_str,
        "activity_space",
        "key"
    )

    # Ensure voting_df is a DataFrame
    if voting_df is None or not isinstance(voting_df, pd.DataFrame):
        voting_df = pd.DataFrame()

    # Debug: print columns
    print(f"DEBUG: voting_df.columns = {voting_df.columns}")

    # --- Permission Check: Use DB for owner/tenant status ---
    resident_status = get_resident_status_from_db(resident_key_str)
    can_suggest_changes = resident_status is not None and resident_status.strip().lower() == "owner"

    activity_weights_for_resident_str = "No specific preferences found for this space."
    resident_voting_data_for_space = pd.DataFrame()

    required_cols = {'resident', 'space', 'activity', 'weight'}
    if set(required_cols).issubset(set(voting_df.columns)):
        resident_voting_data_for_space = voting_df[
            (voting_df['resident'] == resident_key_str) &
            (voting_df['space'] == space_id_str)
        ]
        if not resident_voting_data_for_space.empty:
            prefs = dict(zip(resident_voting_data_for_space['activity'], resident_voting_data_for_space['weight']))
            if prefs:
                activity_weights_for_resident_str = ", ".join([f"{act}: {w:.2f}" for act, w in prefs.items()])
    else:
        print("ERROR: voting_weights.csv is missing required columns or is empty.")

    print(f"DEBUG: resident_voting_data_for_space for resident={resident_key_str}, space={space_id_str}:\n{resident_voting_data_for_space}")

    if not can_suggest_changes:
        return json.dumps({
            "error": f"Resident {resident_key_str} is not allowed to change the geometry of space {space_id_str}. "
                     f"Reason: Resident must have 'owner' status for this space to suggest changes."
        })

    # --- Prepare summary for other residents ---
    other_residents_benefit_summary = "No specific data on other highly interested residents."
    primary_benefited_activity = desired_activity_for_space
    other_resident_keys = []

    if primary_benefited_activity and primary_benefited_activity != "Not specified" and not voting_df.empty:
        relevant_votes = voting_df[
            (voting_df['space'] == space_id_str) &
            (voting_df['activity'] == primary_benefited_activity) &
            (voting_df['resident'] != resident_key_str)
        ]
        if not relevant_votes.empty:
            potential_beneficiaries_info = []
            for _, vote_row in relevant_votes.iterrows():
                other_resident_id = str(vote_row['resident'])
                vote_weight = vote_row['weight']
                # Get distance from DB
                distance = get_resident_distance_from_db(space_id_str, other_resident_id)
                try:
                    distance_val = float(distance)
                except Exception:
                    distance_val = None
                if distance_val is not None:
                    score = (vote_weight * 10) / (distance_val + 1)
                    potential_beneficiaries_info.append({
                        "resident": other_resident_id,
                        "preference_weight": vote_weight,
                        "distance": distance_val,
                        "score": score
                    })
            if potential_beneficiaries_info:
                sorted_beneficiaries = sorted(potential_beneficiaries_info, key=lambda x: x['score'], reverse=True)
                top_n = 3
                summary_parts = []
                for ben_info in sorted_beneficiaries[:top_n]:
                    other_resident_keys.append(ben_info['resident'])
                    dist_desc = "close by" if ben_info['distance'] < 15 else "nearby" if ben_info['distance'] < 40 else "further away"
                    pref_desc = "strong preference" if ben_info['preference_weight'] > 0.6 else "good preference" if ben_info['preference_weight'] > 0.3 else "some preference"
                    summary_parts.append(
                        f"{ben_info['resident']} (who is {dist_desc} and has a {pref_desc} for '{primary_benefited_activity}')"
                    )
                if summary_parts:
                    other_residents_benefit_summary = (
                        "Other residents who might particularly benefit include: " + "; ".join(summary_parts) + "."
                    )
            else:
                other_residents_benefit_summary = "No distance data found in DB to assess other beneficiaries."
    # --- End summary ---

    # You may need to define or fetch these variables as appropriate for your context
    resident_persona = get_resident_persona_from_db(resident_key_str)

    distance = get_resident_distance_from_db(space_id, resident_key)
    current_activity = get_current_activity_in_space(space_id)
    activity_logic = get_activity_logic_for_activity(desired_activity_for_space)
    activity_logic_str = str(activity_logic)
    all_db_data = get_relevant_db_context(space_id, resident_key, other_resident_keys)

    # If allowed, proceed to get suggestions
    suggestions_json_str = suggest_geometric_variations(
        space_id=space_id,
        resident_persona=resident_persona,
        space_context=space_details_str,
        distance_to_space=str(distance),
        activity_weights_for_resident=activity_weights_for_resident_str,
        activity_logic_context=activity_logic_str,
        current_activity_in_space=current_activity,
        user_question_for_suggestion=user_question if user_question else "General suggestions requested.",
        desired_activity_for_space=desired_activity_for_space if desired_activity_for_space else "Not specified",
        other_residents_summary=other_residents_benefit_summary,
        full_db_context=all_db_data,
    )

    try:
        cleaned_json_str = extract_largest_json_object(suggestions_json_str)
    except Exception:
        # fallback: try to strip markdown and explanations, then extract again
        cleaned_json_str = strip_markdown_and_explanations(suggestions_json_str)
        try:
            cleaned_json_str = extract_largest_json_object(cleaned_json_str)
        except Exception:
            # fallback: just use whatever is left
            pass

    cleaned_json_str = remove_control_chars(cleaned_json_str)
    cleaned_json_str = cleaned_json_str.replace('\\ n', '\\n')
    cleaned_json_str = re.sub(r'\\(?![bfnrtu"\\\/])', '', cleaned_json_str)
    cleaned_json_str = repair_llm_json(cleaned_json_str)
    try:
        suggestions_data = json.loads(cleaned_json_str)
        return suggestions_data
    except json.JSONDecodeError as e:
        try:
            suggestions_data = demjson3.decode(cleaned_json_str)
            return suggestions_data
        except Exception as e2:
            print(f"demjson3 also failed: {e2}")
            print(f"JSONDecodeError in orchestrator for space_id {space_id}, resident {resident_key}: {e}. Raw: >>>{suggestions_json_str}<<< Cleaned: >>>{cleaned_json_str}<<<")
            return {
                "error": "Failed to parse LLM response for geometric variations.",
                "details": str(e),
                "raw_response": suggestions_json_str,
                "cleaned_response": cleaned_json_str
            }

def strip_markdown_and_explanations(s):
    # Remove markdown code blocks (``` and content between)
    s = re.sub(r'```(?:json)?[\s\S]*?```', '', s)
    # Remove everything before the first '{'
    first_brace = s.find('{')
    if first_brace != -1:
        s = s[first_brace:]
    # Remove everything after the last '}'
    last_brace = s.rfind('}')
    if last_brace != -1:
        s = s[:last_brace+1]
    return s

def repair_llm_json(s):
    import re
    # Fix wall_height: 0.8-1.3 → 0.8
    s = re.sub(r'("wall_height"\s*:\s*)([0-9.]+)\s*-\s*[0-9.]+', r'\1\2', s)
    # Add missing comma before "summary_reasoning"
    s = re.sub(r'(\}\s*\])\s*"summary_reasoning"', r'\1,\n  "summary_reasoning"', s)
    # Remove leading space and quote in summary_reasoning
    s = re.sub(r'"summary_reasoning":\s*"\s*"', '"summary_reasoning": ""', s)
    # Remove leading space and quote after colon (fixes: "summary_reasoning": " "Text"")
    s = re.sub(r'("summary_reasoning"\s*:\s*)"\s+"', r'\1"', s)
    # Remove double quotes at the end (fixes: ... "Text""})
    s = re.sub(r'("summary_reasoning"\s*:\s*".*?)""(\s*})', r'\1"\2', s)
    # Remove trailing text after the last closing bracket
    last_brace = s.rfind('}')
    if last_brace != -1:
        s = s[:last_brace+1]
    # If suggestions array is not closed, close it
    if s.count('[') > s.count(']'):
        s += ']'
    # If object is not closed, close it
    if s.count('{') > s.count('}'):
        s += '}'
    return s

def remove_control_chars(s):
    # Remove ASCII control characters except for \n, \t, \r
    return ''.join(ch for ch in s if ch in string.printable or ch in '\n\r\t')

def extract_largest_json_object(text):
    """
    Extracts the largest JSON object from a string.
    Returns the JSON string or raises ValueError if not found.
    """
    stack = []
    start = None
    for i, c in enumerate(text):
        if c == '{':
            if not stack:
                start = i
            stack.append('{')
        elif c == '}':
            if stack:
                stack.pop()
                if not stack and start is not None:
                    candidate = text[start:i+1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except Exception:
                        continue
    raise ValueError("No valid JSON object found.")

TABLE_DESCRIPTIONS_PATH = os.path.join(os.path.dirname(__file__), "knowledge", "table_descriptions.json")
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "sql", "example.db")

def process_natural_language_to_sql_answer(user_question: str) -> dict:
    """
    Processes a natural language question, converts it to SQL, queries the database,
    and formulates a natural language answer.
    Returns a dictionary with 'answer' or 'error'.
    """
    try:
        db_schema = get_dB_schema(DB_PATH)
        table_names = list(db_schema.keys())
        explicit_table = None
        table_description = ""

        table_names_sorted = sorted(table_names, key=len, reverse=True)
        clean_question = re.sub(r"[\"']", "", user_question.lower())

        for tname in table_names_sorted:
            if re.search(rf"\b{re.escape(tname.lower())}\b", clean_question):
                explicit_table = tname
                break

        if explicit_table:
            relevant_table = explicit_table
            print(f"Explicit table found in question: {relevant_table}")
            if os.path.exists(TABLE_DESCRIPTIONS_PATH):
                try:
                    with open(TABLE_DESCRIPTIONS_PATH, 'r', encoding='utf-8') as f:
                        all_descriptions = json.load(f)
                    table_description = all_descriptions.get(relevant_table, "")
                    if not table_description:
                        print(f"No specific description found for explicit table '{relevant_table}' in {TABLE_DESCRIPTIONS_PATH}.")
                    else:
                        print(f"Loaded description for explicit table '{relevant_table}'.")
                except Exception as e:
                    print(f"Error loading or parsing {TABLE_DESCRIPTIONS_PATH}: {e}. Proceeding without explicit table description.")
        else:
            if not os.path.exists(TABLE_DESCRIPTIONS_PATH):
                return {"error": f"Table descriptions file not found at {TABLE_DESCRIPTIONS_PATH}"}
            rag_result = sql_rag_call(user_question, TABLE_DESCRIPTIONS_PATH, n_results=1)
            if not rag_result or not rag_result[0]:
                return {"error": "Could not determine a relevant table for the question using RAG."}
            relevant_table, table_description = rag_result
            relevant_table = relevant_table.split()[0].strip()
            print(f"Most relevant table via RAG: {relevant_table}")

        if not relevant_table:
            return {"error": "No relevant table could be identified for the question."}

        table_schema = db_schema.get(relevant_table)
        if table_schema is None:
            return {"error": f"Table '{relevant_table}' not found in database schema."}

        filtered_schema = {relevant_table: table_schema}
        db_context = format_dB_context(DB_PATH, filtered_schema)

        current_question_for_llm = user_question

        sql_query = generate_sql_query(db_context, table_description, current_question_for_llm)
        print(f"Generated SQL Query: \n {sql_query}")

        if "No information" in sql_query or not sql_query.strip():
            return {"answer": "I'm sorry, but this database does not seem to contain enough information to answer that question, or I could not formulate a query."}

        sql_query, query_result = fetch_sql(sql_query, db_context, user_question, DB_PATH)

        if not query_result or query_result == "Failed to generate a correct SQL query after multiple attempts...":
            return {"answer": "I tried to query the database, but I couldn't find the specific information or the query failed. Please try rephrasing your question."}

        final_answer = build_answer(sql_query, query_result, user_question)
        print(f"Final Answer: \n {final_answer}")
        return {"answer": final_answer, "sql_query_executed": sql_query}

    except Exception as e:
        print(f"Error in process_natural_language_to_sql_answer: {str(e)}")
        return {"error": f"An unexpected error occurred while processing your question."}