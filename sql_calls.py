import sqlite3
import pandas as pd
import re
from llm_calls import fix_sql_query

# Get the schema (tables and properties) of the SQL database
def get_dB_schema(dB_path):
    conn = sqlite3.connect(dB_path)
    cursor = conn.cursor()
    schema_info = {}
    # Get a list of all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    table_names = cursor.fetchall()
    
    for table in table_names:
        table_name = table[0]
        column_names = []
        
        # Get the schema for the specific table
        cursor.execute(f"PRAGMA table_info({table_name});")
        schema = cursor.fetchall()
        
        for column in schema:
            column_names.append(column[1]) 

        schema_info[table_name] = column_names
    
    conn.close()
    return schema_info

# Format dB schema into LLM prompt format
def format_dB_context(ifc_sql_dB, filtered_dB_schema: str) -> str:

    def fetch_example_rows(db_path, table_name):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        query = f"SELECT * FROM {table_name} ORDER BY RANDOM() LIMIT 3"
        cursor.execute(query)
        rows = cursor.fetchall()
        
        conn.close()
        return rows

    chunks = []
    for table_name in filtered_dB_schema:
        properties_names = filtered_dB_schema[table_name]
        formatted_string = ', '.join(f'"{property}"' for property in properties_names)
        example_rows = fetch_example_rows(ifc_sql_dB, table_name)
        df = pd.DataFrame(example_rows, columns=properties_names)

        chunk = f"""CREATE TABLE "{table_name}" ({formatted_string})
        /*
        {df.to_string()}
        */
        \n
        """
        chunks.append(chunk)
    chunks = "\n".join(chunks)
    return chunks

# Run an SQL query against the database
def execute_sql_query(dB_path, sql_query, params=None):
    # Connect to the SQLite database
    print(f"[DEBUG] Connecting to database: {dB_path}")
    print(f"[DEBUG] SQL Query: {sql_query}")
    if params:
        print(f"[DEBUG] Query Parameters: {params}")
    else:
        print("[DEBUG] No query parameters.")

    conn = sqlite3.connect(dB_path)
    cursor = conn.cursor()

    try:
        # Execute the SQL query
        if params:
            cursor.execute(sql_query, params)
        else:
            cursor.execute(sql_query)
        result = cursor.fetchall()
        print(f"[DEBUG] Raw SQL Result: {result}")
    except Exception as e:
        print(f"[ERROR] Exception during SQL execution: {e}")
        raise
    finally:
        # Close the connection
        conn.close()
        print("[DEBUG] Database connection closed.")

    return result

# Execute and self-debug sql queries
def fetch_sql(sql_query, dB_context, user_question, dB_path):
    attempt = 1
    max_retries = 3
    atempted_queries = []
    exceptions = []

    while attempt <= max_retries:
        try:
            print("____________________")
            print(f"[DEBUG] Execute Attempt {attempt}/{max_retries}")
            print(f"[DEBUG] Attempting SQL Query: {sql_query}")
            sql_result = execute_sql_query(dB_path, sql_query)

            # If query returns empty because of wrong property name
            if not sql_result or str(sql_result) == "[(0,)]":
                sql_exception = "The query returned empty. You should try either looking at a different table or at other properties in the same table."
                atempted_queries.append(sql_query)
                exceptions.append(sql_exception)
                print(f"[DEBUG] Query result: EMPTY. Attempted queries so far: {atempted_queries}")
                print(f"[DEBUG] Exceptions so far: {exceptions}")

                sql_query = fix_sql_query(dB_context, user_question, atempted_queries, exceptions)
                print(f"[DEBUG] Trying a new query: \n{sql_query}")
                attempt += 1
                continue

            # Exit if we got a result
            else:
                print(f"[DEBUG] This SQL query had a valid result!")
                print(f"[DEBUG] Final SQL Query: {sql_query}")
                print(f"[DEBUG] Final SQL Result: {sql_result}")
                return sql_query, sql_result

        # When the table name is wrong
        except Exception as sql_exception:
            print(f"[ERROR] Exception during SQL execution: {sql_exception}")
            attempt += 1
            atempted_queries.append(sql_query)
            exceptions.append(str(sql_exception))
            print(f"[DEBUG] Attempted queries so far: {atempted_queries}")
            print(f"[DEBUG] Exceptions so far: {exceptions}")

            sql_query = fix_sql_query(dB_context, user_question, atempted_queries, exceptions)
            print(f"[DEBUG] Trying a new query: \n{sql_query}")
            continue

    # Exit if we didnt manage to get a result after max tries
    if attempt == max_retries:
        print("[ERROR] Failed to generate a correct SQL query after multiple attempts.")
        sql_query = None
        sql_result = "Failed to generate a correct SQL query after multiple attempts..."

    print(f"[DEBUG] Returning after {max_retries} attempts. SQL Query: {sql_query}, SQL Result: {sql_result}")
    return sql_query, sql_result

def get_space_details_as_string(db_path: str, space_id: str, table_name: str, id_column_name: str) -> str | None: # type: ignore
    """
    Fetches all details for a specific space_id from a given table and returns them as a formatted string.

    Args:
        db_path (str): The path to the SQLite database file.
        space_id (str): The ID of the space to fetch details for.
        table_name (str): The name of the table where space details are stored.
        id_column_name (str): The name of the column in table_name that holds the space_id.

    Returns:
        str | None: A string containing the space details, or None if not found or an error occurs.
    """
    conn = None
    # The try-finally structure is kept to ensure the database connection is always closed
    # if it was successfully opened. The 'except' block is removed.
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        query = f"SELECT * FROM {table_name} WHERE {id_column_name} = ?"
        cursor.execute(query, (space_id,))
        row = cursor.fetchone()

        if row:
            column_names = [description[0] for description in cursor.description]
            details = [f"{col_name}: {value}" for col_name, value in zip(column_names, row)]
            return "\n".join(details)
        return None  # Space ID not found
    # If a sqlite3.Error occurs in the try block, it will propagate up,
    # and the function will terminate before reaching 'return None' here.
    finally:
        if conn:
            conn.close()
