from server.config import *
from llm_calls import *
from sql_calls import *
from utils.rag_utils import sql_rag_call

def answer_user_question(user_question: str, db_path: str = "sql/gh_data.db") -> str:
    db_schema = get_dB_schema(db_path)
    print("Tables found in schema:", list(db_schema.keys()))
    for table, columns in db_schema.items():
        print(f"Table '{table}' columns: {columns}")

    table_names = list(db_schema.keys())
    explicit_table = None

    table_names_sorted = sorted(table_names, key=len, reverse=True)
    clean_question = re.sub(r"[\"']", "", user_question.lower())

    for tname in table_names_sorted:
        if re.search(rf"\b{re.escape(tname.lower())}\b", clean_question):
            explicit_table = tname
            break

    if explicit_table:
        relevant_table = explicit_table
        table_description = ""
        print(f"Explicit table found in question: {relevant_table}")
    else:
        table_descriptions_path = "knowledge/table_descriptions.json"
        relevant_table, table_description = sql_rag_call(
            user_question, table_descriptions_path, n_results=1
        )
        relevant_table = relevant_table.split()[0].strip()
        print(f"Most relevant table: {relevant_table}")

    if not relevant_table:
        print("No relevant table found.")
        return "No relevant table found."

    table_schema = db_schema.get(relevant_table)
    if table_schema is None:
        print(f"Table '{relevant_table}' not found in database schema.")
        return f"Table '{relevant_table}' not found in database schema."
    filtered_schema = {relevant_table: table_schema}
    db_context = format_dB_context(db_path, filtered_schema)

    column_names = db_schema[relevant_table]
    explicit_column = None
    for cname in column_names:
        if cname.lower() in user_question.lower():
            explicit_column = cname
            break

    if explicit_column:
        print(f"Explicit column found in question: {explicit_column}")
        user_question = f"{user_question} (Focus only on column: {explicit_column})"

    sql_query = generate_sql_query(db_context, table_description, user_question)
    print(f"SQL Query: \n {sql_query}")

    if "No information" in sql_query:
        print("I'm sorry, but this database does not contain enough information to answer that question.")
        return "I'm sorry, but this database does not contain enough information to answer that question."

    sql_query, query_result = fetch_sql(sql_query, db_context, user_question, db_path)

    if not query_result:
        print("SQL query failed or returned no data.")
        return "I'm sorry but I was not able to find any relevant information to answer your question. Please, try again."

    final_answer = build_answer(sql_query, query_result, user_question)
    print(f"Final Answer: \n {final_answer}")
    return final_answer

# Example usage (remove or comment out when importing in gh_server.py)
if __name__ == "__main__":
    question = "How many gardener households are on level1?"
    answer = answer_user_question(question)
    print("Returned Answer:", answer)