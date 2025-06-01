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
def execute_sql_query(dB_path, sql_query):
    # Connect to the SQLite database
    conn = sqlite3.connect(dB_path)
    cursor = conn.cursor()

    # Execute the SQL query
    cursor.execute(sql_query)
    result = cursor.fetchall()

    # Close the connection
    conn.close()

    return result

# Execute and self-debug sql queries
def fetch_sql(sql_query, dB_context, user_question, dB_path):
    sql_result = None  # <-- Ensure it's always defined
    attempt = 1
    max_retries = 3
    atempted_queries = []
    exceptions = []

    while attempt <= max_retries:
        try:
            print("____________________")
            print(f"Execute Attempt {attempt}/{max_retries}")
            sql_result = execute_sql_query(dB_path, sql_query)

            # If query returns empty because of wrong property name
            if not sql_result or str(sql_result) == "[(0,)]":
                sql_exception = "The query returned empty. You should try either looking at a different table or at other properties in the same table."
                atempted_queries.append(sql_query)
                exceptions.append(sql_exception)
            
                sql_query = fix_sql_query(dB_context, user_question, atempted_queries, exceptions)
                #print(f"Attempt {attempt}/{max_retries}")
                print(f"Query result: EMPTY. \nTrying a new query: \n {sql_query}")
                attempt += 1
                continue
            
            # Exit if we got a result
            else:
                print(f"This SQL query had a valid result!")
                return sql_query, sql_result
            
        # When the table name is wrong
        except Exception as sql_exception:
            attempt += 1
            atempted_queries.append(sql_query)
            exceptions.append(sql_exception)

            sql_query = fix_sql_query(dB_context, user_question, atempted_queries, exceptions)
            print(f"Query result: INVALID. \nTrying a new query: \n {sql_query}")
            continue

    # Exit if we didnt manage to get a result after max tries
    if attempt == max_retries:
        sql_query = None
        sql_result = "Failed to generate a correct SQL query after multiple attempts..."
        
    return sql_query, sql_result

