import sqlite3

def init_db():
    conn = sqlite3.connect('memories.db')
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS memories (
                   vector TEXT PRIMARY KEY,
                   value TEXT
        )
    ''')

    conn.commit()
    conn.close()

def add_memory(vector: list, value):
    conn = sqlite3.connect('memories.db')
    cursor = conn.cursor()

    string_vector = ", ".join(vector)

    cursor.execute('''
        INSERT INTO memories
            (vector, value)
            VALUE (?, ?)
    ''',
    string_vector,
    value
    )

    conn.commit()
    conn.close()

def delete_memory(vector: list):
    conn = sqlite3.connect('memories.db')
    cursor = conn.cursor()

    string_vector = ", ".join(vector)

    cursor.execute('''
        DELETE FROM memories
        WHERE vector = ?
    ''',
    string_vector,
    )

    conn.commit()
    conn.close()

def update_memory(vector: list, value):
    conn = sqlite3.connect('memories.db')
    cursor = conn.cursor()

    string_vector = ", ".join(vector)

    cursor.execute('''
        UPDATE memories
        SET
            value = ?
        WHERE vector = ?
    ''',
    value,
    string_vector
    )

    conn.commit()
    conn.close()

def get_memory(vector: list, value):
    conn = sqlite3.connect('memories.db')
    cursor = conn.cursor()

    string_vector = ", ".join(vector)

    cursor.execute('''
        SELECT memories
        FROM events
        WHERE vector = ?
    ''', 
    string_vector,
    )

    result = cursor.fetchall()

    conn.close()

    return result