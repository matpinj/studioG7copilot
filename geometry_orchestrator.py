# d:\01_IAAC\03_aia studio\studioG7copilot\geometry_orchestrator.py
from sql_calls import get_space_details_as_string, get_dB_schema, format_dB_context, fetch_sql
from llm_calls import suggest_geometric_variations, generate_sql_query, build_answer
# Ensure the RAG utility is correctly imported based on your project structure.
# If 'utils' is a direct subdirectory of 'studioG7copilot', this should be:
# from .utils.rag_utils import sql_rag_call
# If 'utils' is at the same level as 'studioG7copilot' and your execution path handles it:
from utils.rag_utils import sql_rag_call
import re
import os

def get_intelligent_geometric_suggestions(space_id: str, user_profile: str) -> str:
    """
    Orchestrates fetching space details from SQL and then getting geometric
    suggestions from the LLM.
    """
    # Step 1: Fetch relevant details for the space_id from the SQL database.
    # We need to know which table and column identify the space.
    # For this example, let's assume a table 'architectural_spaces' and id column 'identifier'.
    # You'll need to adjust this based on your actual database schema.
    space_details_str = get_space_details_as_string(
        db_path="sql/example.db",
        space_id=space_id,
        table_name="activity_space", # Example table name
        id_column_name="key"    # Example ID column name
    )

    if not space_details_str:
        space_details_str = "No specific details found for this space in the database."

    # Step 2: Call the LLM with the space_id, user_profile, and the fetched details.
    # The suggest_geometric_variations function in llm_calls.py will need to be updated
    # to accept and use this additional 'space_context'.
    suggestions_json_str = suggest_geometric_variations(
        space_id=space_id,
        user_profile=user_profile,
        space_context=space_details_str
    )

    return suggestions_json_str

TABLE_DESCRIPTIONS_PATH = os.path.join(os.path.dirname(__file__), "knowledge", "table_descriptions.json")

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
            table_description = ""  # Optionally, load description if needed for explicit match
            print(f"Explicit table found in question: {relevant_table}")
        else:
            if not os.path.exists(TABLE_DESCRIPTIONS_PATH):
                return {"error": f"Table descriptions file not found at {TABLE_DESCRIPTIONS_PATH}"}
            
            # sql_rag_call is expected to return a tuple (relevant_table_name_str, table_description_str)
            # or (None, None) if no good match.
            rag_result = sql_rag_call(user_question, TABLE_DESCRIPTIONS_PATH, n_results=1)
            if not rag_result or not rag_result[0]: # Check if rag_result itself or its first element is None/empty
                 return {"error": "Could not determine a relevant table for the question using RAG."}
            relevant_table, table_description = rag_result
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
