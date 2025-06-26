import sqlite3
conn = sqlite3.connect('D:/01_IAAC/03_aia studio/studioG7copilot/sql/ml_activity_logic.db')
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print(cursor.fetchall())
conn.close()