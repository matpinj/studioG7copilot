import sqlite3
import pandas as pd

excel_file_path = 'gh_data/studio_gh_data.xlsx'
conn = sqlite3.connect('sql/gh_data.db')
cursor = conn.cursor()

def drop_all_tables(cursor):
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    for table_name in tables:
        cursor.execute(f"DROP TABLE IF EXISTS {table_name[0]}")

drop_all_tables(cursor)
conn.commit()
print("Existing tables have been dropped.")

# Load all sheets into a dictionary of DataFrames
sheets_dict = pd.read_excel(excel_file_path, sheet_name=None)

for sheet_name, df in sheets_dict.items():
    # Optional: Strip column names whitespace just in case
    df.columns = df.columns.str.strip()
    
    # Insert DataFrame into SQLite table
    # Use if_exists='replace' to recreate table with updated columns and schema
    df.to_sql(sheet_name, conn, if_exists='replace', index=False)
    
    print(f"Data from sheet '{sheet_name}' inserted into table '{sheet_name}' (table replaced).")

# Close the connection properly
conn.close()
print("Database connection closed.")
