import sqlite3
from pathlib import Path

import pandas as pd


def jsonl_to_sqlite(
    jsonl_file: str | Path, db_file: str | Path = "data.db", table_name: str = "data_table", overwrite: bool = False
) -> str:
    """
    Load a JSONL file into a SQLite database and save it to disk.

    Args:
        jsonl_file (str | Path): Path to the JSONL file.
        db_file (str | Path): Path to the SQLite database file.
        table_name (str): Name of the table to create.
        overwrite (bool): Whether to overwrite the table if it already exists.

    Returns:
        str: Path to the SQLite database file.
    """
    try:
        df = pd.read_json(jsonl_file, lines=True)
        conn = sqlite3.connect(db_file)
        try:
            df.to_sql(table_name, conn, if_exists="replace" if overwrite else "fail", index=False)
        except ValueError as exc:
            # Error is expected when overwrite=False and table exists -- we just want to ignore that
            # to avoid regenerting the table when it is not needed (overwrite=false)
            if overwrite:
                raise ValueError("Fout bij overschrijven van tabel in SQLite.") from exc
        conn.close()
        return str(db_file)
    except Exception as exc:
        raise ConnectionError(f"Fout bij laden JSONL naar SQLite: {exc}") from exc
