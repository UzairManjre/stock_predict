import os
import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager

# Manual .env loader to avoid third-party dependencies
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))
if os.path.exists(env_path):
    try:
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, val = line.split('=', 1)
                    os.environ[key.strip()] = val.strip()
    except Exception as e:
        print(f"⚠️ Warning: Failed to parse .env file: {e}")

# Read from environment variables, fallback to defaults
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("DB_NAME", "stock_predict")
DB_USER = os.environ.get("DB_USER", "postgres")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "postgres")

def get_db_connection():
    """Establishes a connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        return conn
    except Exception as e:
        print(f"❌ Error connecting to database ({DB_USER}@{DB_HOST}:{DB_PORT}/{DB_NAME}): {e}")
        raise

@contextmanager
def db_cursor(commit=True):
    """
    Context manager to easily get a cursor, handle exceptions,
    and automatically close/commit.
    """
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    try:
        yield cursor
        if commit:
            conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"⚠️ Database transaction rolled back due to error: {e}")
        raise
    finally:
        cursor.close()
        conn.close()
