#!/usr/bin/env python3
"""
queue_db.py — Knowledge-Web Queue Database Manager

Manages the SQLite processing queue for video ingestion.
Provides CRUD operations for the queue and channels_watched tables.

Usage:
    python queue_db.py --init          # Create database and tables
    python queue_db.py --status        # Show queue status summary
    python queue_db.py --list          # List all queue entries
    python queue_db.py --list-channels # List watched channels
"""

import argparse
import os
import sqlite3
import sys
from datetime import datetime, timezone

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS queue (
    id TEXT PRIMARY KEY,                    -- YouTube video ID
    url TEXT NOT NULL,
    title TEXT,
    channel TEXT,
    status TEXT DEFAULT 'pending',          -- pending | processing | complete | failed
    source TEXT,                            -- manual | watch | batch | discover
    added_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    started_at DATETIME,
    completed_at DATETIME,
    error TEXT,
    retry_count INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS channels_watched (
    channel_id TEXT PRIMARY KEY,
    channel_name TEXT,
    last_checked DATETIME,
    last_video_id TEXT                      -- most recent video we've seen
);

CREATE INDEX IF NOT EXISTS idx_queue_status ON queue(status);
CREATE INDEX IF NOT EXISTS idx_queue_source ON queue(source);
CREATE INDEX IF NOT EXISTS idx_queue_added_at ON queue(added_at);
"""


class QueueDB:
    """Interface for the Knowledge-Web processing queue."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn = None

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
        return self._conn

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    def init_db(self):
        """Create tables and indexes."""
        conn = self._connect()
        conn.executescript(SCHEMA_SQL)
        conn.commit()
        print(f"Database initialized at {self.db_path}")

    # ── Queue Operations ──

    def add_to_queue(
        self,
        video_id: str,
        url: str,
        title: str | None = None,
        channel: str | None = None,
        source: str = "manual",
    ) -> bool:
        """Add a video to the processing queue. Returns True if added, False if duplicate."""
        conn = self._connect()
        try:
            conn.execute(
                """INSERT INTO queue (id, url, title, channel, source)
                   VALUES (?, ?, ?, ?, ?)""",
                (video_id, url, title, channel, source),
            )
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def get_pending(self, limit: int = 10) -> list[dict]:
        """Get pending queue entries, oldest first."""
        conn = self._connect()
        rows = conn.execute(
            """SELECT * FROM queue
               WHERE status = 'pending'
               ORDER BY added_at ASC
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_failed(self, max_retries: int = 3) -> list[dict]:
        """Get failed entries eligible for retry."""
        conn = self._connect()
        rows = conn.execute(
            """SELECT * FROM queue
               WHERE status = 'failed' AND retry_count < ?
               ORDER BY added_at ASC""",
            (max_retries,),
        ).fetchall()
        return [dict(r) for r in rows]

    def mark_processing(self, video_id: str) -> None:
        """Mark a queue entry as currently processing."""
        conn = self._connect()
        conn.execute(
            """UPDATE queue
               SET status = 'processing', started_at = ?
               WHERE id = ?""",
            (datetime.now(timezone.utc).isoformat(), video_id),
        )
        conn.commit()

    def mark_complete(self, video_id: str) -> None:
        """Mark a queue entry as successfully completed."""
        conn = self._connect()
        conn.execute(
            """UPDATE queue
               SET status = 'complete', completed_at = ?
               WHERE id = ?""",
            (datetime.now(timezone.utc).isoformat(), video_id),
        )
        conn.commit()

    def mark_failed(self, video_id: str, error: str) -> None:
        """Mark a queue entry as failed with error details."""
        conn = self._connect()
        conn.execute(
            """UPDATE queue
               SET status = 'failed', error = ?, retry_count = retry_count + 1
               WHERE id = ?""",
            (error, video_id),
        )
        conn.commit()

    def reset_for_retry(self, video_id: str) -> None:
        """Reset a failed entry back to pending for retry."""
        conn = self._connect()
        conn.execute(
            """UPDATE queue
               SET status = 'pending', error = NULL, started_at = NULL
               WHERE id = ?""",
            (video_id,),
        )
        conn.commit()

    def get_status_counts(self) -> dict:
        """Get count of entries by status."""
        conn = self._connect()
        rows = conn.execute(
            """SELECT status, COUNT(*) as count
               FROM queue
               GROUP BY status"""
        ).fetchall()
        return {row["status"]: row["count"] for row in rows}

    def get_entry(self, video_id: str) -> dict | None:
        """Get a single queue entry by video ID."""
        conn = self._connect()
        row = conn.execute(
            "SELECT * FROM queue WHERE id = ?", (video_id,)
        ).fetchone()
        return dict(row) if row else None

    def exists(self, video_id: str) -> bool:
        """Check if a video ID is already in the queue."""
        conn = self._connect()
        row = conn.execute(
            "SELECT 1 FROM queue WHERE id = ?", (video_id,)
        ).fetchone()
        return row is not None

    def list_all(self, status: str | None = None, limit: int = 50) -> list[dict]:
        """List queue entries, optionally filtered by status."""
        conn = self._connect()
        if status:
            rows = conn.execute(
                """SELECT * FROM queue
                   WHERE status = ?
                   ORDER BY added_at DESC LIMIT ?""",
                (status, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM queue ORDER BY added_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Channel Watch Operations ──

    def add_channel(self, channel_id: str, channel_name: str) -> bool:
        """Add a channel to the watch list. Returns True if added."""
        conn = self._connect()
        try:
            conn.execute(
                """INSERT INTO channels_watched (channel_id, channel_name)
                   VALUES (?, ?)""",
                (channel_id, channel_name),
            )
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def remove_channel(self, channel_id: str) -> bool:
        """Remove a channel from the watch list."""
        conn = self._connect()
        cursor = conn.execute(
            "DELETE FROM channels_watched WHERE channel_id = ?",
            (channel_id,),
        )
        conn.commit()
        return cursor.rowcount > 0

    def get_watched_channels(self) -> list[dict]:
        """Get all watched channels."""
        conn = self._connect()
        rows = conn.execute(
            "SELECT * FROM channels_watched ORDER BY channel_name"
        ).fetchall()
        return [dict(r) for r in rows]

    def update_channel_check(
        self, channel_id: str, last_video_id: str | None = None
    ) -> None:
        """Update the last checked time (and optionally last video ID) for a channel."""
        conn = self._connect()
        now = datetime.now(timezone.utc).isoformat()
        if last_video_id:
            conn.execute(
                """UPDATE channels_watched
                   SET last_checked = ?, last_video_id = ?
                   WHERE channel_id = ?""",
                (now, last_video_id, channel_id),
            )
        else:
            conn.execute(
                """UPDATE channels_watched
                   SET last_checked = ?
                   WHERE channel_id = ?""",
                (now, channel_id),
            )
        conn.commit()


def print_status(db: QueueDB) -> None:
    """Print a summary of queue status."""
    counts = db.get_status_counts()
    total = sum(counts.values())

    print("=" * 40)
    print("Knowledge-Web Queue Status")
    print("=" * 40)
    print(f"  Pending:    {counts.get('pending', 0):4d}")
    print(f"  Processing: {counts.get('processing', 0):4d}")
    print(f"  Complete:   {counts.get('complete', 0):4d}")
    print(f"  Failed:     {counts.get('failed', 0):4d}")
    print(f"  Total:      {total:4d}")
    print()

    channels = db.get_watched_channels()
    print(f"  Watched channels: {len(channels)}")
    print("=" * 40)


def print_list(db: QueueDB, status: str | None = None) -> None:
    """Print queue entries."""
    entries = db.list_all(status=status)
    if not entries:
        print("Queue is empty.")
        return

    print(f"{'ID':13s} {'Status':11s} {'Source':8s} {'Title'}")
    print("-" * 70)
    for e in entries:
        title = (e.get("title") or "untitled")[:40]
        print(f"{e['id']:13s} {e['status']:11s} {e.get('source', '?'):8s} {title}")


def print_channels(db: QueueDB) -> None:
    """Print watched channels."""
    channels = db.get_watched_channels()
    if not channels:
        print("No watched channels.")
        return

    print(f"{'Channel ID':26s} {'Name':30s} {'Last Checked'}")
    print("-" * 70)
    for c in channels:
        last = c.get("last_checked") or "never"
        print(f"{c['channel_id']:26s} {c['channel_name']:30s} {last}")


def main():
    parser = argparse.ArgumentParser(
        description="Knowledge-Web Queue Database Manager"
    )
    parser.add_argument("--init", action="store_true", help="Create database and tables")
    parser.add_argument("--status", action="store_true", help="Show queue status summary")
    parser.add_argument("--list", action="store_true", help="List all queue entries")
    parser.add_argument("--list-channels", action="store_true", help="List watched channels")
    parser.add_argument(
        "--filter",
        choices=["pending", "processing", "complete", "failed"],
        help="Filter --list by status",
    )
    parser.add_argument("--db-path", default=None, help="Path to queue.db")

    args = parser.parse_args()

    # Determine DB path
    if args.db_path:
        db_path = args.db_path
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(script_dir, "_system", "queue.db")

    db = QueueDB(db_path)

    try:
        if args.init:
            db.init_db()

        if not any([args.init, args.status, args.list, args.list_channels]):
            parser.print_help()
            sys.exit(1)

        if args.status:
            print_status(db)
        if args.list:
            print_list(db, status=args.filter)
        if args.list_channels:
            print_channels(db)
    finally:
        db.close()


if __name__ == "__main__":
    main()
