import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'emotion_data.db')

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            emotion TEXT NOT NULL,
            confidence REAL NOT NULL,
            probabilities TEXT NOT NULL,
            face_count INTEGER DEFAULT 1
        )
    ''')
    conn.commit()
    conn.close()

def save_detection(emotion, confidence, probabilities, face_count=1):
    conn = get_connection()
    conn.execute(
        'INSERT INTO detections (timestamp, emotion, confidence, probabilities, face_count) VALUES (?, ?, ?, ?, ?)',
        (datetime.now().isoformat(), emotion, confidence, probabilities, face_count)
    )
    conn.commit()
    conn.close()

def get_recent_detections(limit=50):
    conn = get_connection()
    rows = conn.execute(
        'SELECT * FROM detections ORDER BY timestamp DESC LIMIT ?', (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_emotion_stats():
    conn = get_connection()
    rows = conn.execute('''
        SELECT emotion, COUNT(*) as count, AVG(confidence) as avg_confidence
        FROM detections
        GROUP BY emotion
        ORDER BY count DESC
    ''').fetchall()
    conn.close()
    return [dict(r) for r in rows]
