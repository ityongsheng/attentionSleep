"""
Database manager for logging fatigue detection events
"""
import sqlite3
from datetime import datetime
import os
from pathlib import Path
import cv2


class DatabaseManager:
    """SQLite database manager for fatigue detection logs"""

    def __init__(self, db_path='database/fatigue_logs.db'):
        """
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        # Initialize database
        self._init_database()

    def _init_database(self):
        """Create database tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS fatigue_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                fatigue_class TEXT NOT NULL,
                confidence REAL NOT NULL,
                ear REAL,
                mar REAL,
                duration INTEGER,
                screenshot_path TEXT,
                notes TEXT
            )
        ''')

        # Create sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS monitoring_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_time TEXT NOT NULL,
                end_time TEXT,
                total_events INTEGER DEFAULT 0,
                normal_count INTEGER DEFAULT 0,
                drowsy_count INTEGER DEFAULT 0,
                very_drowsy_count INTEGER DEFAULT 0
            )
        ''')

        conn.commit()
        conn.close()

    def log_event(self, fatigue_class, confidence, ear, mar, screenshot=None):
        """
        Log a fatigue detection event

        Args:
            fatigue_class: Detected fatigue class
            confidence: Model confidence
            ear: Eye Aspect Ratio
            mar: Mouth Aspect Ratio
            screenshot: Image frame (optional)

        Returns:
            Event ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        timestamp = datetime.now().isoformat()
        screenshot_path = None

        # Save screenshot if provided
        if screenshot is not None:
            screenshot_dir = 'logs/screenshots'
            os.makedirs(screenshot_dir, exist_ok=True)
            screenshot_path = os.path.join(
                screenshot_dir,
                f"event_{timestamp.replace(':', '-')}.jpg"
            )
            cv2.imwrite(screenshot_path, screenshot)

        cursor.execute('''
            INSERT INTO fatigue_events
            (timestamp, fatigue_class, confidence, ear, mar, screenshot_path)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (timestamp, fatigue_class, confidence, ear, mar, screenshot_path))

        event_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return event_id

    def start_session(self):
        """Start a new monitoring session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        start_time = datetime.now().isoformat()
        cursor.execute('''
            INSERT INTO monitoring_sessions (start_time)
            VALUES (?)
        ''', (start_time,))

        session_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return session_id

    def end_session(self, session_id):
        """End a monitoring session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        end_time = datetime.now().isoformat()

        # Count events during this session
        cursor.execute('''
            SELECT fatigue_class, COUNT(*)
            FROM fatigue_events
            WHERE timestamp >= (
                SELECT start_time FROM monitoring_sessions WHERE id = ?
            )
            GROUP BY fatigue_class
        ''', (session_id,))

        counts = {'Normal': 0, 'Drowsy': 0, 'Very Drowsy': 0}
        for row in cursor.fetchall():
            counts[row[0]] = row[1]

        total_events = sum(counts.values())

        cursor.execute('''
            UPDATE monitoring_sessions
            SET end_time = ?,
                total_events = ?,
                normal_count = ?,
                drowsy_count = ?,
                very_drowsy_count = ?
            WHERE id = ?
        ''', (end_time, total_events, counts['Normal'],
              counts['Drowsy'], counts['Very Drowsy'], session_id))

        conn.commit()
        conn.close()

    def get_recent_events(self, limit=100):
        """Get recent fatigue events"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT * FROM fatigue_events
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (limit,))

        events = cursor.fetchall()
        conn.close()

        return events

    def get_session_summary(self, session_id):
        """Get summary of a monitoring session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT * FROM monitoring_sessions
            WHERE id = ?
        ''', (session_id,))

        session = cursor.fetchone()
        conn.close()

        return session

    def clear_old_events(self, days=30):
        """Delete events older than specified days"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cutoff_date = datetime.now().timestamp() - (days * 24 * 3600)

        cursor.execute('''
            DELETE FROM fatigue_events
            WHERE timestamp < datetime(?, 'unixepoch')
        ''', (cutoff_date,))

        conn.commit()
        conn.close()


if __name__ == "__main__":
    # Test database
    db = DatabaseManager()
    session_id = db.start_session()
    print(f"Started session: {session_id}")

    # Log test event
    event_id = db.log_event('Normal', 0.95, 0.3, 0.4)
    print(f"Logged event: {event_id}")

    # Get recent events
    events = db.get_recent_events(10)
    print(f"Recent events: {len(events)}")
