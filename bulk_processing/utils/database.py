# utils/database.py
import sqlite3
import json
from typing import Dict, List, Any, Optional
from contextlib import contextmanager


class JobDatabase:
    def __init__(self, db_path: str = "batch_jobs.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize database with tables"""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL DEFAULT 'created',
                    config TEXT,  -- JSON string
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    total_tasks INTEGER DEFAULT 0,
                    completed_tasks INTEGER DEFAULT 0,
                    failed_tasks INTEGER DEFAULT 0,
                    processing_tasks INTEGER DEFAULT 0
                );
                
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    request_data TEXT NOT NULL,  -- JSON string
                    result_data TEXT,  -- JSON string
                    error_message TEXT,
                    attempts INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    processing_time_ms INTEGER,
                    
                    FOREIGN KEY (job_id) REFERENCES jobs (job_id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_tasks_job_id ON tasks(job_id);
                CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
                CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
            """)

    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Access columns by name
        try:
            yield conn
        finally:
            conn.close()

    def create_job(self, job_id: str, requests: List[Dict], config: Dict) -> Dict:
        """Create a new batch job"""
        with self.get_connection() as conn:
            # Insert job
            conn.execute(
                """
                INSERT INTO jobs (job_id, config, total_tasks)
                VALUES (?, ?, ?)
            """,
                (job_id, json.dumps(config), len(requests)),
            )

            # Insert tasks
            tasks_data = []
            for i, request in enumerate(requests):
                task_id = f"{job_id}_{i:06d}"
                tasks_data.append((task_id, job_id, json.dumps(request)))

            conn.executemany(
                """
                INSERT INTO tasks (task_id, job_id, request_data)
                VALUES (?, ?, ?)
            """,
                tasks_data,
            )

            conn.commit()

        return self.get_job(job_id)

    def get_job(self, job_id: str) -> Optional[Dict]:
        """Get job with current statistics"""
        with self.get_connection() as conn:
            job_row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()

            if not job_row:
                return None

            # Get current task counts
            stats = conn.execute(
                """
                SELECT 
                    status,
                    COUNT(*) as count
                FROM tasks 
                WHERE job_id = ? 
                GROUP BY status
            """,
                (job_id,),
            ).fetchall()

            stats_dict = {row["status"]: row["count"] for row in stats}

            return {
                "job_id": job_row["job_id"],
                "status": job_row["status"],
                "config": json.loads(job_row["config"]),
                "created_at": job_row["created_at"],
                "updated_at": job_row["updated_at"],
                "stats": {
                    "total": job_row["total_tasks"],
                    "pending": stats_dict.get("pending", 0),
                    "processing": stats_dict.get("processing", 0),
                    "completed": stats_dict.get("completed", 0),
                    "failed": stats_dict.get("failed", 0),
                },
            }

    def update_task_status(
        self,
        job_id: str,
        task_id: str,
        status: str,
        result: Any = None,
        error: str = None,
    ) -> bool:
        """Update task status atomically"""
        with self.get_connection() as conn:
            # Build update query dynamically
            set_clauses = ["status = ?", "updated_at = CURRENT_TIMESTAMP"]
            params = [status]

            if status == "processing":
                set_clauses.append("started_at = CURRENT_TIMESTAMP")
            elif status == "completed":
                set_clauses.extend(
                    [
                        "completed_at = CURRENT_TIMESTAMP",
                        "processing_time_ms = (julianday(CURRENT_TIMESTAMP) - julianday(started_at)) * 86400000",
                    ]
                )
                if result is not None:
                    set_clauses.append("result_data = ?")
                    params.append(json.dumps(result))
            elif status == "failed":
                set_clauses.append("attempts = attempts + 1")
                if error:
                    set_clauses.append("error_message = ?")
                    params.append(error)

            params.append(task_id)

            # Update task
            cursor = conn.execute(
                f"""
                UPDATE tasks 
                SET {", ".join(set_clauses)}
                WHERE task_id = ?
            """,
                params,
            )

            if cursor.rowcount == 0:
                return False

            # Update job statistics and status
            conn.execute(
                """
                UPDATE jobs SET
                    completed_tasks = (SELECT COUNT(*) FROM tasks WHERE job_id = ? AND status = 'completed'),
                    failed_tasks = (SELECT COUNT(*) FROM tasks WHERE job_id = ? AND status = 'failed'),
                    processing_tasks = (SELECT COUNT(*) FROM tasks WHERE job_id = ? AND status = 'processing'),
                    updated_at = CURRENT_TIMESTAMP,
                    status = CASE 
                        WHEN (SELECT COUNT(*) FROM tasks WHERE job_id = ? AND status IN ('completed', 'failed')) = total_tasks
                        THEN 'completed'
                        WHEN (SELECT COUNT(*) FROM tasks WHERE job_id = ? AND status = 'processing') > 0
                        THEN 'processing'
                        ELSE 'created'
                    END
                WHERE job_id = ?
            """,
                (job_id, job_id, job_id, job_id, job_id, job_id),
            )

            conn.commit()
            return True

    def get_pending_tasks(self, job_id: str, max_retries: int = 3) -> List[Dict]:
        """Get pending tasks including failed tasks under retry limit"""
        with self.get_connection() as conn:
            rows = conn.execute(
                """
                SELECT task_id, request_data, attempts
                FROM tasks 
                WHERE job_id = ? 
                AND (status = 'pending' OR (status = 'failed' AND attempts < ?))
                ORDER BY created_at
            """,
                (job_id, max_retries),
            ).fetchall()

            return [
                {
                    "task_id": row["task_id"],
                    "request": json.loads(row["request_data"]),
                    "attempts": row["attempts"],
                }
                for row in rows
            ]

    def get_job_analytics(self, job_id: str) -> Dict:
        """Get detailed analytics for a job"""
        with self.get_connection() as conn:
            analytics = conn.execute(
                """
                SELECT 
                    AVG(processing_time_ms) as avg_processing_time,
                    MIN(processing_time_ms) as min_processing_time,
                    MAX(processing_time_ms) as max_processing_time,
                    COUNT(CASE WHEN attempts > 1 THEN 1 END) as retry_count,
                    MAX(attempts) as max_attempts
                FROM tasks 
                WHERE job_id = ? AND status = 'completed'
            """,
                (job_id,),
            ).fetchone()

            return dict(analytics) if analytics else {}
