import os
import logging
from dotenv import load_dotenv
import pyodbc

load_dotenv()

SQL_DRIVER = "ODBC Driver 17 for SQL Server"
SQL_SERVER = os.getenv("SQL_SERVER")
SQL_DATABASE = os.getenv("SQL_DATABASE")
SQL_USER = os.getenv("SQL_USER")
SQL_SECRET = os.getenv("SQL_SECRET")

connection_string = f"driver={SQL_DRIVER};server={SQL_SERVER};database={SQL_DATABASE};UID={SQL_USER};PWD={SQL_SECRET}"


def get_connection():
    try:
        conn = pyodbc.connect(connection_string)
        logging.info("Conexión exitosa a la base de datos")
        return conn
    except pyodbc.Error as e:
        logging.error(f"Error db_connection 22: {e}")
        raise
