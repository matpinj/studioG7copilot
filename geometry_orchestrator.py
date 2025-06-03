# d:\01_IAAC\03_aia studio\studioG7copilot\geometry_orchestrator.py
from sql_calls import get_space_details_as_string, get_dB_schema, format_dB_context, fetch_sql, execute_sql_query
from llm_calls import suggest_geometric_variations, generate_sql_query, build_answer
# Ensure the RAG utility is correctly imported based on your project structure.
# If 'utils' is a direct subdirectory of 'studioG7copilot', this should be:
# from .utils.rag_utils import sql_rag_call
# If 'utils' is at the same level as 'studioG7copilot' and your execution path handles it:
from utils.rag_utils import sql_rag_call
import re
import json 
import os
import pandas as pd # Added for CSV handling

# Define paths for ML predictions CSVs
GREEN_PREDICTIONS_CSV_PATH = os.path.join(os.path.dirname(__file__), "ml_models", "green_predictions.csv")
THRESHOLD_PREDICTIONS_CSV_PATH = os.path.join(os.path.dirname(__file__), "ml_models", "threshold_predictions.csv")
USABILITY_PREDICTIONS_CSV_PATH = os.path.join(os.path.dirname(__file__), "ml_models", "usability_predictions.csv")
PERSONAS_ASSIGNED_CSV_PATH = os.path.join(os.path.dirname(__file__), "resident_data", "personas_assigned.csv")
LLM_ACTIVITY_ASSIGNMENTS_CSV_PATH = os.path.join(os.path.dirname(__file__), "llm_activity_assignments.csv") # Assuming it's in the same directory
VOTING_WEIGHTS_CSV_PATH = os.path.join(os.path.dirname(__file__), "resident_data", "voting_weights.csv")

# Database path for resident-specific data like distances, if different from general DB_PATH
GH_DATA_DB_PATH = os.path.join(os.path.dirname(__file__), "sql", "gh_data.db")

# Cache for loaded CSV data
_loaded_green_predictions_df = None
_loaded_threshold_predictions_df = None
_loaded_usability_predictions_df = None
_loaded_personas_assigned_df = None
_loaded_llm_activity_assignments_df = None

_loaded_voting_weights_df = None

def _load_csv_data(csv_path, df_cache_attr_name):
    """Helper function to load and cache a CSV file."""
    global _loaded_green_predictions_df, _loaded_threshold_predictions_df, _loaded_usability_predictions_df, \
           _loaded_personas_assigned_df, _loaded_voting_weights_df, _loaded_llm_activity_assignments_df

    
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

def load_personas_assigned_df():
    return _load_csv_data(PERSONAS_ASSIGNED_CSV_PATH, "_loaded_personas_assigned_df")

def load_llm_activity_assignments_df():
    return _load_csv_data(LLM_ACTIVITY_ASSIGNMENTS_CSV_PATH, "_loaded_llm_activity_assignments_df")


def load_voting_weights_df():
    return _load_csv_data(VOTING_WEIGHTS_CSV_PATH, "_loaded_voting_weights_df")

def _get_prediction_from_df(df, space_id_str, column_name, default_value="N/A"):
    if not df.empty and 'id' in df.columns and column_name in df.columns:
        space_data = df[df['id'] == space_id_str]
        if not space_data.empty:
            return space_data.iloc[0][column_name]
    return default_value

def get_intelligent_geometric_suggestions(space_id: str, resident_key: str) -> str:
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

    space_id_str = str(space_id) # Ensure space_id is string for lookup
    resident_key_str = str(resident_key)

    # Retrieve ML predictions for the space
    green_pred_val = _get_prediction_from_df(green_df, space_id_str, 'green_prediction')
    threshold_pred_val = _get_prediction_from_df(thresh_df, space_id_str, 'predicted_activities')
    usability_pred_val = _get_prediction_from_df(usability_df, space_id_str, 'usability_prediction')

    # Load resident-specific data
    personas_df = load_personas_assigned_df()
    voting_df = load_voting_weights_df()
    llm_assignments_df = load_llm_activity_assignments_df()


    # Get resident persona
    resident_persona = "Unknown"
    if not personas_df.empty and 'resident_key' in personas_df.columns:
        persona_data = personas_df[personas_df['resident_key'] == resident_key_str]
        if not persona_data.empty:
            resident_persona = persona_data.iloc[0].get('resident_persona', "Unknown")

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
    try:
        # Ensure resident_key_str is a valid column name (alphanumeric, underscores)
        # This is a simple check; more robust validation might be needed if keys can be arbitrary.
        if re.match(r"^[a-zA-Z0-9_]+$", resident_key_str):
            # Construct query to select the column named after the resident_key
            # The table name is 'resident_distances', column for space IDs is 'Outdoor_Space'
            query = f'SELECT "{resident_key_str}" FROM resident_distances WHERE "Outdoor_Space" = ?'
            result = execute_sql_query(GH_DATA_DB_PATH, query, (space_id_str,))
            if result and result[0] and pd.notna(result[0][0]):
                distance_val = result[0][0]
                if isinstance(distance_val, (int, float)):
                    distance_to_space = f"{distance_val:.1f}"
                else:
                    distance_to_space = str(distance_val) 
        else:
            print(f"Warning: Invalid resident_key format for SQL query: {resident_key_str}")
    except Exception as e:
        print(f"Error querying resident distance from {GH_DATA_DB_PATH}: {e}")
        # distance_to_space remains "N/A"

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


    # If allowed, proceed to get suggestions
    suggestions_json_str = suggest_geometric_variations(
        space_id=space_id,
        resident_persona=resident_persona, # Pass resident's actual persona
        space_context=space_details_str,
        green_prediction=green_pred_val,
        threshold_prediction=threshold_pred_val,
        usability_prediction=usability_pred_val,
        distance_to_space=str(distance_to_space), # Ensure it's a string
        activity_weights_for_resident=activity_weights_for_resident_str,
        current_activity_in_space=current_activity_in_space
    )

    return suggestions_json_str

TABLE_DESCRIPTIONS_PATH = os.path.join(os.path.dirname(__file__), "knowledge", "table_descriptions.json")
DB_PATH = os.path.join(os.path.dirname(__file__), "sql", "example.db") # Define DB_PATH consistently

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
