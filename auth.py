# auth.py
import bcrypt
from db import get_connection

def create_user(username, email, password):
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO users (username,email,password) VALUES (?,?,?)",
            (username, email, hashed)
        )
        conn.commit()
        return True
    except:
        return False
    finally:
        conn.close()

def login_user(email, password):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT id,password FROM users WHERE email=?",
        (email,)
    )
    row = cur.fetchone()
    conn.close()

    if row and bcrypt.checkpw(password.encode(), row[1]):
        return row[0]
    return None
