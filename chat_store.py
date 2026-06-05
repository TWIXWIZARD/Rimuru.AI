# chat_store.py
from db import get_connection

def create_session(user_id, title="New Chat", mode="general"):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO sessions (user_id,title,mode) VALUES (?,?,?)",
        (user_id, title, mode)
    )
    sid = cur.lastrowid
    conn.commit()
    conn.close()
    return sid

def get_sessions(user_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT id,title,mode,created_at FROM sessions WHERE user_id=? ORDER BY created_at DESC",
        (user_id,)
    )
    rows = cur.fetchall()
    conn.close()
    return rows

def delete_session(session_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM sessions WHERE id=?", (session_id,))
    conn.commit()
    conn.close()

def rename_session(session_id, new_title):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("UPDATE sessions SET title=? WHERE id=?", (new_title, session_id))
    conn.commit()
    conn.close()

def get_or_create_default_session(user_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT id FROM sessions WHERE user_id=? ORDER BY created_at LIMIT 1",
        (user_id,)
    )
    row = cur.fetchone()
    if row:
        conn.close()
        return row[0]

    cur.execute(
        "INSERT INTO sessions (user_id,title,mode) VALUES (?,?,?)",
        (user_id, "Default Chat", "general")
    )
    sid = cur.lastrowid
    conn.commit()
    conn.close()
    return sid

def save_message(session_id, role, content):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO messages (session_id,role,content) VALUES (?,?,?)",
        (session_id, role, content)
    )
    conn.commit()
    conn.close()

def load_messages(session_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT role,content,created_at FROM messages WHERE session_id=? ORDER BY created_at",
        (session_id,)
    )
    rows = cur.fetchall()
    conn.close()
    return rows

def get_message_count(session_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM messages WHERE session_id=?", (session_id,))
    count = cur.fetchone()[0]
    conn.close()
    return count

def get_all_user_messages(user_id):
    """For analytics: get all messages across all sessions"""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT m.role, m.content, m.created_at, s.title
        FROM messages m
        JOIN sessions s ON m.session_id = s.id
        WHERE s.user_id=?
        ORDER BY m.created_at
    """, (user_id,))
    rows = cur.fetchall()
    conn.close()
    return rows

# ---- Graph storage ----

def save_graph(session_id, image_data, question, title="Chart"):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO graphs (session_id,title,image_data,question) VALUES (?,?,?,?)",
        (session_id, title, image_data, question)
    )
    gid = cur.lastrowid
    conn.commit()
    conn.close()
    return gid

def load_graphs(session_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT id,title,image_data,question,created_at FROM graphs WHERE session_id=? ORDER BY created_at DESC",
        (session_id,)
    )
    rows = cur.fetchall()
    conn.close()
    return rows

def load_all_user_graphs(user_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT g.id, g.title, g.image_data, g.question, g.created_at, s.title as session_title
        FROM graphs g
        JOIN sessions s ON g.session_id = s.id
        WHERE s.user_id=?
        ORDER BY g.created_at DESC
    """, (user_id,))
    rows = cur.fetchall()
    conn.close()
    return rows

def delete_graph(graph_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM graphs WHERE id=?", (graph_id,))
    conn.commit()
    conn.close()

# ---- Notes ----

def save_note(user_id, title, content):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO notes (user_id,title,content) VALUES (?,?,?)",
        (user_id, title, content)
    )
    nid = cur.lastrowid
    conn.commit()
    conn.close()
    return nid

def load_notes(user_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT id,title,content,created_at FROM notes WHERE user_id=? ORDER BY created_at DESC",
        (user_id,)
    )
    rows = cur.fetchall()
    conn.close()
    return rows

def delete_note(note_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM notes WHERE id=?", (note_id,))
    conn.commit()
    conn.close()
