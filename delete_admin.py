import sqlite3

DB_PATH = "production_app.db"

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.execute("DELETE FROM users WHERE username = ?", ("admin",))
conn.commit()
conn.close()

print("✅ Old admin account deleted.")
