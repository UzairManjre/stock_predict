import os
import sys
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# Add root folder to sys.path so we can import from Scaper
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Scaper.utils.db_connector import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD, get_db_connection

def create_database_if_not_exists():
    """Connect to template/postgres database and create the target database if missing."""
    print(f"Connecting to postgres server to verify '{DB_NAME}' database exists...")
    try:
        # Connect to 'postgres' system database first
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database="postgres",
            user=DB_USER,
            password=DB_PASSWORD
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Check if DB exists
        cursor.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s", (DB_NAME,))
        exists = cursor.fetchone()
        
        if not exists:
            print(f"[INFO] Database '{DB_NAME}' does not exist. Creating...")
            cursor.execute(f'CREATE DATABASE "{DB_NAME}"')
            print(f"[SUCCESS] Database '{DB_NAME}' created successfully.")
        else:
            print(f"[SUCCESS] Database '{DB_NAME}' already exists.")
            
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"[ERROR] Failed to verify/create database: {e}")
        sys.exit(1)

def run_schema_sql():
    """Reads schema.sql and runs it against the target database."""
    schema_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'schema.sql'))
    if not os.path.exists(schema_path):
        print(f"[ERROR] schema.sql not found at {schema_path}")
        sys.exit(1)
        
    print(f"Running schema from {schema_path}...")
    try:
        with open(schema_path, 'r', encoding='utf-8') as f:
            sql_script = f.read()
            
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Split and run the SQL commands (except TimescaleDB extension, which we handle carefully)
        # Note: psycopg2 can execute multi-statement strings, but TimescaleDB extension setup
        # might fail if TimescaleDB library is not loaded.
        try:
            # Try to run the entire script
            cursor.execute(sql_script)
            conn.commit()
            print("[SUCCESS] Main SQL Schema executed successfully.")
        except Exception as schema_err:
            conn.rollback()
            print(f"[WARNING] Direct execution of schema.sql failed: {schema_err}")
            print("[INFO] Executing schema commands individually, ignoring TimescaleDB setup if it fails...")
            
            # Executing line by line / block by block
            # We will run statements separated by semicolon, but clean up comments and empty statements.
            statements = [s.strip() for s in sql_script.split(';') if s.strip()]
            for stmt in statements:
                if "timescaledb" in stmt.lower():
                    # Attempt TimescaleDB extension activation
                    try:
                        cursor.execute(stmt)
                        conn.commit()
                        print("[SUCCESS] TimescaleDB extension created successfully.")
                    except Exception as ts_err:
                        conn.rollback()
                        print(f"[WARNING] TimescaleDB extension not supported or library missing: {ts_err}")
                else:
                    try:
                        cursor.execute(stmt)
                        conn.commit()
                    except Exception as stmt_err:
                        conn.rollback()
                        print(f"[ERROR] Failed to run SQL statement: {stmt}\nError: {stmt_err}")
                        
        # 3. Create Hypertable for TimescaleDB if possible
        try:
            # Check if timescaledb extension exists in pg_extension
            cursor.execute("SELECT 1 FROM pg_extension WHERE extname = 'timescaledb'")
            has_timescaledb = cursor.fetchone()
            
            if has_timescaledb:
                print("Converting 'market_data_daily' to a hypertable...")
                cursor.execute("SELECT create_hypertable('market_data_daily', 'date', if_not_exists => TRUE);")
                conn.commit()
                print("[SUCCESS] Successfully converted 'market_data_daily' into a TimescaleDB hypertable!")
            else:
                print("[INFO] TimescaleDB is not enabled. 'market_data_daily' will remain a standard table.")
        except Exception as hyper_err:
            conn.rollback()
            print(f"[INFO] Could not convert to hypertable (TimescaleDB not loaded/supported): {hyper_err}")
            print("   'market_data_daily' will run as a standard table.")
            
        cursor.close()
        conn.close()
        print("[SUCCESS] Database setup complete.")
        
    except Exception as e:
        print(f"[ERROR] Failed to setup database schema: {e}")
        sys.exit(1)

def test_connection():
    """Verify that we can establish a connection to the database."""
    print("Testing connection parameters...")
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        ver = cursor.fetchone()
        print(f"[SUCCESS] Connection successful! PostgreSQL Version: {ver[0]}")
        cursor.close()
        conn.close()
        sys.exit(0)
    except Exception as e:
        print(f"[ERROR] Connection test failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    if "--test-connection" in sys.argv:
        test_connection()
        
    create_database_if_not_exists()
    run_schema_sql()
