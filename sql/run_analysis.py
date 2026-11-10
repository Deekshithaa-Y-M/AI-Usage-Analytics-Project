"""Run the DuckDB analysis script without SQLTools."""

from pathlib import Path

import duckdb


SQL_FILE = Path(__file__).with_name("analysis.sql")
DATABASE_FILE = Path(__file__).with_name("ai_usage.duckdb")


def main() -> None:
    connection = duckdb.connect(str(DATABASE_FILE))
    try:
        connection.execute(SQL_FILE.read_text(encoding="utf-8"))
        row_count = connection.execute(
            "SELECT COUNT(*) FROM fact_master"
        ).fetchone()[0]
    finally:
        connection.close()

    print(f"Analysis completed: {DATABASE_FILE}")
    print(f"fact_master rows: {row_count}")


if __name__ == "__main__":
    main()
