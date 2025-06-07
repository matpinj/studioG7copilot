from server.config import *
from llm_calls import *
from sql_calls import *
from utils.rag_utils import sql_rag_call
import re

def answer_sql_question(user_question):

    # --- Load SQL Database ---
    db_path = "sql/gh_data.db"
    db_schema = get_dB_schema(db_path)
    print("Tables found in schema:", list(db_schema.keys()))
    for table, columns in db_schema.items():
        print(f"Table '{table}' columns: {columns}")
    # --- Try to extract table name directly from question ---
    table_names = list(db_schema.keys())
    explicit_table = None

    # Sort table names by length (longest first) to avoid partial matches like 'level1' in 'level==1'
    table_names_sorted = sorted(table_names, key=len, reverse=True)

    clean_question = re.sub(r"[\"']", "", user_question.lower())

    for tname in table_names_sorted:
        # Match as whole word, or inside quotes, or after 'table'
        if re.search(rf"\b{re.escape(tname.lower())}\b", clean_question):
            explicit_table = tname
            break

    if explicit_table:
        relevant_table = explicit_table
        table_description = ""  # Optionally, fetch description from your descriptions file
        print(f"Explicit table found in question: {relevant_table}")
    else:
        # --- Retrieve most relevant table ---
        table_descriptions_path = "knowledge/table_descriptions.json"
        relevant_table, table_description = sql_rag_call(
            user_question, table_descriptions_path, n_results=1
        )
        relevant_table = relevant_table.split()[0].strip()
        print(f"Most relevant table: {relevant_table}")

    if relevant_table:
        print(f"Most relevant table: {relevant_table}")
    else:
        print("No relevant table found.")
        exit()

    # --- Filter Schema to relevant table ---
    table_schema = db_schema.get(relevant_table)
    if table_schema is None:
        print(f"Table '{relevant_table}' not found in database schema.")
        exit()
    filtered_schema = {relevant_table: table_schema}
    db_context = format_dB_context(db_path, filtered_schema)

    # --- Try to extract column name from the question ---
    column_names = db_schema[relevant_table]
    explicit_column = None
    for cname in column_names:
        if cname.lower() in user_question.lower():
            explicit_column = cname
            break

    if explicit_column:
        print(f"Explicit column found in question: {explicit_column}")
        # Optionally, you can pass this as an extra hint to your LLM
        user_question = f"{user_question} (Focus only on column: {explicit_column})"

    # --- Generate SQL query from LLM ---
    sql_query = generate_sql_query(db_context, table_description, user_question)
    print(f"SQL Query: \n {sql_query}")

    # --- LLM says insufficient info ---
    if "No information" in sql_query:
        print("I'm sorry, but this database does not contain enough information to answer that question.")
        exit()

    # --- Execute SQL with a self-debbuging feature ---
    sql_query, query_result = fetch_sql(sql_query, db_context, user_question, db_path)

    # -- If self-debugging failed after max_retries we give up
    if not query_result:
        print("SQL query failed or returned no data.")
        print("I'm sorry but I was not able to find any relevant information to answer your question. Please, try again.")
        exit()

    # --- Build natural language answer to user ---
    final_answer = build_answer(sql_query, query_result, user_question)
    return final_answer
