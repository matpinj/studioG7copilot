# d:\01_IAAC\03_aia studio\studioG7copilot\geometry_orchestrator.py
import sys # Add sys import
import os # Add os import

# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sql_calls import get_space_details_as_string, get_dB_schema, format_dB_context, fetch_sql, execute_sql_query # Now an absolute import
from llm_calls import suggest_geometric_variations, generate_sql_query, build_answer # Now an absolute import
# Ensure the RAG utility is correctly imported based on your project structure.
# If 'utils' is a direct subdirectory of 'studioG7copilot', this should be:
# from .utils.rag_utils import sql_rag_call
# If 'utils' is at the same level as 'studioG7copilot' and your execution path handles it:
from utils.rag_utils import sql_rag_call # Now an absolute import
import re
import json 
import os
import pandas as pd # Added for CSV handling

# Define paths for ML predictions CSVs
GREEN_PREDICTIONS_CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "ml_models", "green_predictions.csv")
THRESHOLD_PREDICTIONS_CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "ml_models", "threshold_predictions.csv")
USABILITY_PREDICTIONS_CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "ml_models", "usability_predictions.csv")
LLM_ACTIVITY_ASSIGNMENTS_CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "llm_reasoning", "llm_activity_assignments.csv") # Corrected path assuming it's in llm_reasoning
VOTING_WEIGHTS_CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "resident_data", "voting_weights.csv") # type: ignore
STUDIO_EXPORT_CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "gh_data", "studio_export_ml.csv") # Path to studio_export_ml.csv in gh_data folder
RESIDENT_DISTANCES_CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "resident_data", "resident_distances.csv")



# Database path for resident-specific data like distances, if different from general DB_PATH
# Assuming gh_data.db is in the project root's sql directory
GH_DATA_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "sql", "gh_data.db")

# Cache for loaded CSV data
_loaded_green_predictions_df = None
_loaded_threshold_predictions_df = None
_loaded_usability_predictions_df = None
_loaded_llm_activity_assignments_df = None
_loaded_voting_weights_df = None # type: ignore
_loaded_studio_export_df = None
_loaded_resident_distances_df = None

def _load_csv_data(csv_path, df_cache_attr_name):
    """Helper function to load and cache a CSV file."""
    # Ensure all global df caches are accessible
    global _loaded_green_predictions_df, _loaded_threshold_predictions_df, _loaded_usability_predictions_df, _loaded_llm_activity_assignments_df, _loaded_voting_weights_df, _loaded_studio_export_df, _loaded_resident_distances_df

    
    df_cache = globals()[df_cache_attr_name]
    if df_cache is None:
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                # Normalize ID to string for 'id' or 'space_id' columns
                if 'id' in df.columns: 
                    df['id'] = df['id'].astype(str) # Normalize ID to string
                elif 'space_id' in df.columns: # For llm_activity_assignments.csv
                    df['space_id'] = df['space_id'].astype(str)
                globals()[df_cache_attr_name] = df
                print(f"Loaded predictions from {csv_path}")
            except Exception as e:
                print(f"Error loading {csv_path}: {e}")
                globals()[df_cache_attr_name] = pd.DataFrame() # Empty DataFrame on error
        else:
            print(f"Warning: Predictions CSV not found at {csv_path}")
            globals()[df_cache_attr_name] = pd.DataFrame() # Empty DataFrame if not found
    return globals()[df_cache_attr_name]

def load_green_predictions_df():
    return _load_csv_data(GREEN_PREDICTIONS_CSV_PATH, "_loaded_green_predictions_df")

def load_threshold_predictions_df():
    return _load_csv_data(THRESHOLD_PREDICTIONS_CSV_PATH, "_loaded_threshold_predictions_df")

def load_usability_predictions_df():
    return _load_csv_data(USABILITY_PREDICTIONS_CSV_PATH, "_loaded_usability_predictions_df")

def load_llm_activity_assignments_df():
    return _load_csv_data(LLM_ACTIVITY_ASSIGNMENTS_CSV_PATH, "_loaded_llm_activity_assignments_df")

def load_voting_weights_df():
    return _load_csv_data(VOTING_WEIGHTS_CSV_PATH, "_loaded_voting_weights_df")

def load_studio_export_df():
    return _load_csv_data(STUDIO_EXPORT_CSV_PATH, "_loaded_studio_export_df")

def load_resident_distances_df():
    return _load_csv_data(RESIDENT_DISTANCES_CSV_PATH, "_loaded_resident_distances_df")

def _get_prediction_from_df(df, space_id_str, column_name, default_value="N/A"):
    if not df.empty and 'id' in df.columns and column_name in df.columns:
        space_data = df[df['id'] == space_id_str]
        if not space_data.empty and pd.notna(space_data.iloc[0][column_name]):
            return space_data.iloc[0][column_name]
    return default_value

def get_intelligent_geometric_suggestions(space_id: str, resident_key: str, user_question: str = None, desired_activity_for_space: str = None) -> str: # type: ignore
    """
    Orchestrates fetching space details from SQL and then getting geometric
    suggestions from the LLM, personalized for a resident.
    """
    # Step 1: Fetch relevant details for the space_id from the SQL database.
    # We need to know which table and column identify the space.
    # For this example, let's assume a table 'architectural_spaces' and id column 'identifier'.
    # You'll need to adjust this based on your actual database schema.
    space_details_str = get_space_details_as_string(
        db_path=DB_PATH, # Use defined DB_PATH
        space_id=space_id,
        table_name="activity_space", # Example table name
        id_column_name="key"    # Example ID column name
    )

    if not space_details_str:
        space_details_str = "No specific details found for this space in the database."

    # Load ML prediction DataFrames
    green_df = load_green_predictions_df()
    thresh_df = load_threshold_predictions_df()
    usability_df = load_usability_predictions_df()
    studio_export_df = load_studio_export_df() # Load the studio export CSV
    resident_distances_df = load_resident_distances_df() # Load resident distances CSV


    space_id_str = str(space_id) # Ensure space_id is string for lookup
    resident_key_str = str(resident_key)

    # Retrieve ML predictions for the space
    green_pred_val = _get_prediction_from_df(green_df, space_id_str, 'green_prediction')
    threshold_pred_val = _get_prediction_from_df(thresh_df, space_id_str, 'predicted_activities')
    usability_pred_val = _get_prediction_from_df(usability_df, space_id_str, 'usability_prediction')

    # Retrieve details from studio_export.csv
    studio_export_details_str = "N/A"
    if not studio_export_df.empty and 'space_id' in studio_export_df.columns:
        # Ensure space_id_str is compared with string version of 'space_id' column
        space_export_data = studio_export_df[studio_export_df['space_id'].astype(str) == space_id_str]
        if not space_export_data.empty:
            # Convert the row to a JSON string or key-value pairs
            studio_export_details_str = space_export_data.iloc[0].to_json()
            # Or: studio_export_details_str = "; ".join([f"{col}: {val}" for col, val in space_export_data.iloc[0].items() if pd.notna(val)])

    # Load resident-specific data
    voting_df = load_voting_weights_df()
    llm_assignments_df = load_llm_activity_assignments_df()


    # Get resident persona from gh_data.db
    resident_persona = "Unknown"
    try:
        # Confirm your actual table and column names for personas
        # Example: table 'personas_table', columns 'resident_key', 'persona_value'
        query_persona = "SELECT resident_persona FROM personas_assigned WHERE resident_key = ?" # Assuming table name is 'personas_assigned'
        persona_result = execute_sql_query(GH_DATA_DB_PATH, query_persona, (resident_key_str,))
        if persona_result and persona_result[0] and pd.notna(persona_result[0][0]):
            resident_persona = persona_result[0][0]
    except Exception as e:
        print(f"Error fetching resident persona from {GH_DATA_DB_PATH}: {e}")

    # Get current activity in space
    current_activity_in_space = "Unknown"
    if not llm_assignments_df.empty and 'space_id' in llm_assignments_df.columns and 'assigned_activity' in llm_assignments_df.columns:
        activity_data = llm_assignments_df[llm_assignments_df['space_id'] == space_id_str]
        if not activity_data.empty:
            current_activity_in_space = activity_data.iloc[0]['assigned_activity']
            if pd.isna(current_activity_in_space): # Handle potential NaN
                current_activity_in_space = "Unknown"
    
    
    # Get distance to space for resident
    distance_to_space = "N/A"
    if not resident_distances_df.empty:
        # Assuming resident_distances_df has 'Outdoor_Space' (or 'id') as index or column for space_id,
        # and columns named after resident_keys (e.g., H1, H2) for distances.
        # Adjust column names as per your CSV structure.
        # The resident_distances.csv uses 'Outdoor Space' as the column for space IDs, and _load_csv_data does not rename it to 'id'.
        space_row = resident_distances_df[resident_distances_df['Outdoor Space'].astype(str) == space_id_str]
        if not space_row.empty and resident_key_str in space_row.columns:
            distance_val = space_row.iloc[0][resident_key_str]
            if pd.notna(distance_val):
                if isinstance(distance_val, (int, float)):
                    distance_to_space = f"{distance_val:.1f}"
                else:
                    distance_to_space = str(distance_val)


    # Voting Weights & Permission Check
    can_suggest_changes = False
    activity_weights_for_resident_str = "No specific preferences found for this space."

    if not voting_df.empty and 'resident' in voting_df.columns and 'space' in voting_df.columns and \
       'status' in voting_df.columns and 'activity' in voting_df.columns and 'weight' in voting_df.columns:
        
        resident_voting_data_for_space = voting_df[
            (voting_df['resident'] == resident_key_str) & 
            (voting_df['space'] == space_id_str)
        ]
        if not resident_voting_data_for_space.empty:
            if 'owner' in resident_voting_data_for_space['status'].unique():
                can_suggest_changes = True
            
            prefs = dict(zip(resident_voting_data_for_space['activity'], resident_voting_data_for_space['weight']))
            if prefs:
                activity_weights_for_resident_str = ", ".join([f"{act}: {w:.2f}" for act, w in prefs.items()])

    if not can_suggest_changes:
        return json.dumps({"error": f"Resident {resident_key_str} is not allowed to change the geometry of space {space_id_str}. Reason: Resident must have 'owner' status for this space to suggest changes."})

    # --- START: New section to prepare summary for other residents ---
    other_residents_benefit_summary = "No specific data on other highly interested residents."
    # This is a simplified example. Actual logic would involve:
    # 1. Determining the primary activity benefited by a potential suggestion type.
    #    This is tricky as the suggestion isn't made yet.
    #    Alternatively, focus on the `desired_activity_for_space` if provided.
    primary_benefited_activity = desired_activity_for_space # Or infer from suggestion type

    if primary_benefited_activity and primary_benefited_activity != "Not specified" and not voting_df.empty and not resident_distances_df.empty:
        relevant_votes = voting_df[
            (voting_df['space'] == space_id_str) &
            (voting_df['activity'] == primary_benefited_activity) &
            (voting_df['resident'] != resident_key_str) # Exclude the primary resident
        ]

        if not relevant_votes.empty:
            # Get distances for the current space_id_str
            # Ensure 'Outdoor Space' column exists and is used for matching space_id_str
            if 'Outdoor Space' in resident_distances_df.columns and \
               space_id_str in resident_distances_df['Outdoor Space'].astype(str).values:
                
                space_specific_distances_series = resident_distances_df[
                    resident_distances_df['Outdoor Space'].astype(str) == space_id_str
                ].iloc[0] # Get the row as a Series

                potential_beneficiaries_info = []
                for _, vote_row in relevant_votes.iterrows():
                    other_resident_id = str(vote_row['resident']) # Ensure resident ID is string
                    vote_weight = vote_row['weight']
                    
                    if other_resident_id in space_specific_distances_series.index:
                        distance = space_specific_distances_series[other_resident_id]
                        if pd.notna(distance) and isinstance(distance, (int, float)):
                            # Simple scoring: higher vote weight, lower distance = better
                            score = (vote_weight * 10) / (distance + 1) # Emphasize vote weight, avoid div by zero
                            potential_beneficiaries_info.append({
                                "resident": other_resident_id,
                                "preference_weight": vote_weight,
                                "distance": distance,
                                "score": score
                            })
                
                if potential_beneficiaries_info:
                    sorted_beneficiaries = sorted(potential_beneficiaries_info, key=lambda x: x['score'], reverse=True)
                    top_n = 3 # Consider top 3 other beneficiaries
                    summary_parts = []
                    for ben_info in sorted_beneficiaries[:top_n]:
                        dist_desc = "close by" if ben_info['distance'] < 15 else "nearby" if ben_info['distance'] < 40 else "further away"
                        pref_desc = "strong preference" if ben_info['preference_weight'] > 0.6 else "good preference" if ben_info['preference_weight'] > 0.3 else "some preference"
                        summary_parts.append(f"{ben_info['resident']} (who is {dist_desc} and has a {pref_desc} for '{primary_benefited_activity}')")
                    
                    if summary_parts:
                        other_residents_benefit_summary = "Other residents who might particularly benefit include: " + "; ".join(summary_parts) + "."
            else:
                other_residents_benefit_summary = f"Distance data for space {space_id_str} not found to assess other beneficiaries."
    # --- END: New section ---

    # If allowed, proceed to get suggestions
    suggestions_json_str = suggest_geometric_variations(
        space_id=space_id,
        resident_persona=resident_persona, # Pass resident's actual persona
        space_context=space_details_str,
        green_prediction=green_pred_val,
        threshold_prediction=threshold_pred_val,
        usability_prediction=usability_pred_val,
        distance_to_space=str(distance_to_space), # Ensure it's a string
        activity_weights_for_resident=activity_weights_for_resident_str, # type: ignore
        current_activity_in_space=current_activity_in_space,
        studio_export_details=studio_export_details_str,
        user_question_for_suggestion=user_question if user_question else "General suggestions requested.",
        desired_activity_for_space=desired_activity_for_space if desired_activity_for_space else "Not specified",
        other_residents_summary=other_residents_benefit_summary # Pass the new summary
    )

    # Clean and parse the JSON string response from LLM
    cleaned_json_str = suggestions_json_str
    # First, try to find JSON within markdown-style code blocks
    match_markdown = re.search(r'```json\s*(\{[\s\S]*?\})\s*```', cleaned_json_str, re.DOTALL)
    if match_markdown:
        cleaned_json_str = match_markdown.group(1)
    else:
        # If not found, try to find the first '{' and last '}' to extract the main JSON object
        # This is more robust against leading/trailing non-JSON text.
        match_object = re.search(r'^\s*.*?(\{[\s\S]*\})\s*.*?$', cleaned_json_str, re.DOTALL)
        if match_object:
            cleaned_json_str = match_object.group(1)
    cleaned_json_str = cleaned_json_str.replace('\\ n', '\\n')
    # Attempt to fix common LLM escape issues more carefully
    cleaned_json_str = re.sub(r'\\(?![bfnrtu"\\\/])', '', cleaned_json_str) # Remove backslashes not part of valid escapes

    try:
        suggestions_data = json.loads(cleaned_json_str)
        return suggestions_data
    except json.JSONDecodeError as e:
        # Log this error appropriately in a real application
        print(f"JSONDecodeError in orchestrator for space_id {space_id}, resident {resident_key}: {e}. Raw: >>>{suggestions_json_str}<<< Cleaned: >>>{cleaned_json_str}<<<")
        return {"error": "Failed to parse LLM response for geometric variations.", "details": str(e), "raw_response": suggestions_json_str, "cleaned_response": cleaned_json_str}


TABLE_DESCRIPTIONS_PATH = os.path.join(os.path.dirname(__file__), "knowledge", "table_descriptions.json")
# Assuming example.db is in the project root's sql directory
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "sql", "example.db") # Define DB_PATH consistently

#region
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
        table_description = ""  # Initialize table_description

        # Sort table names by length (longest first) to avoid partial matches
        table_names_sorted = sorted(table_names, key=len, reverse=True)
        # Clean question for safer regex matching
        clean_question = re.sub(r"[\"']", "", user_question.lower())

        for tname in table_names_sorted:
            if re.search(rf"\b{re.escape(tname.lower())}\b", clean_question):
                explicit_table = tname
                break

        if explicit_table:
            relevant_table = explicit_table
            print(f"Explicit table found in question: {relevant_table}")
            # Try to load its description from TABLE_DESCRIPTIONS_PATH
            if os.path.exists(TABLE_DESCRIPTIONS_PATH):
                try:
                    with open(TABLE_DESCRIPTIONS_PATH, 'r', encoding='utf-8') as f:
                        all_descriptions = json.load(f) # Assumes JSON is a dict {table_name: description}
                    table_description = all_descriptions.get(relevant_table, "")
                    if not table_description:
                        print(f"No specific description found for explicit table '{relevant_table}' in {TABLE_DESCRIPTIONS_PATH}.")
                    else:
                        print(f"Loaded description for explicit table '{relevant_table}'.")
                except Exception as e:
                    print(f"Error loading or parsing {TABLE_DESCRIPTIONS_PATH}: {e}. Proceeding without explicit table description.")
                    # table_description remains ""
            else:
                print(f"Table descriptions file not found at {TABLE_DESCRIPTIONS_PATH}, cannot load description for explicit table.")
        else:
            if not os.path.exists(TABLE_DESCRIPTIONS_PATH):
                return {"error": f"Table descriptions file not found at {TABLE_DESCRIPTIONS_PATH}"}
            
            # sql_rag_call is expected to return a tuple (relevant_table_name_str, table_description_str)
            # or (None, None) if no good match.
            rag_result = sql_rag_call(user_question, TABLE_DESCRIPTIONS_PATH, n_results=1)
            if not rag_result or not rag_result[0]: # Check if rag_result itself or its first element is None/empty
                 return {"error": "Could not determine a relevant table for the question using RAG."}
            relevant_table, table_description = rag_result # table_description is set by RAG
            relevant_table = relevant_table.split()[0].strip() # Assuming format "table_name description..."
            print(f"Most relevant table via RAG: {relevant_table}")

        if not relevant_table: # Should be caught by RAG check, but as a safeguard
            return {"error": "No relevant table could be identified for the question."}

        table_schema = db_schema.get(relevant_table)
        if table_schema is None:
            return {"error": f"Table '{relevant_table}' not found in database schema."}
        
        filtered_schema = {relevant_table: table_schema}
        db_context = format_dB_context(DB_PATH, filtered_schema)

        current_question_for_llm = user_question # Keep original question for LLM context

        sql_query = generate_sql_query(db_context, table_description, current_question_for_llm)
        print(f"Generated SQL Query: \n {sql_query}")

        if "No information" in sql_query or not sql_query.strip():
            return {"answer": "I'm sorry, but this database does not seem to contain enough information to answer that question, or I could not formulate a query."}

        # fetch_sql handles execution and self-debugging
        sql_query, query_result = fetch_sql(sql_query, db_context, user_question, DB_PATH)

        if not query_result or query_result == "Failed to generate a correct SQL query after multiple attempts...":
            return {"answer": "I tried to query the database, but I couldn't find the specific information or the query failed. Please try rephrasing your question."}

        final_answer = build_answer(sql_query, query_result, user_question)
        print(f"Final Answer: \n {final_answer}")
        return {"answer": final_answer, "sql_query_executed": sql_query}

    except Exception as e:
        print(f"Error in process_natural_language_to_sql_answer: {str(e)}")
        # In a production environment, you might want to log the full traceback
        return {"error": f"An unexpected error occurred while processing your question."}
    
#endregion
