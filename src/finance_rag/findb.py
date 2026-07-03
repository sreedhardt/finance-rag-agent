"""Structured financial data in SQLite, exposed to the agent through a
guard-railed, read-only SQL tool. Numbers are consistent with the sample
documents so the agent can cross-check unstructured claims against the ledger."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from . import config

SCHEMA_DESCRIPTION = """\
Table quarterly_financials(fiscal_quarter TEXT e.g. 'FY2025-Q1', segment TEXT
  in ('Data Center','Automotive','Client'), revenue_usd_m REAL, cogs_usd_m REAL,
  opex_usd_m REAL) -- one row per quarter per segment, FY2025.
Table supplier_spend(fiscal_quarter TEXT, supplier TEXT, spend_usd_m REAL,
  category TEXT) -- quarterly spend by supplier, FY2025."""

_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|vacuum|replace|reindex)\b",
    re.IGNORECASE,
)

_QUARTERLY = [
    # quarter, segment, revenue, cogs, opex  (USD millions)
    ("FY2025-Q1", "Data Center", 1120.0, 470.0, 205.0),
    ("FY2025-Q1", "Automotive", 310.0, 168.0, 74.0),
    ("FY2025-Q1", "Client", 410.0, 232.0, 88.0),
    ("FY2025-Q2", "Data Center", 1265.0, 512.0, 214.0),
    ("FY2025-Q2", "Automotive", 322.0, 171.0, 76.0),
    ("FY2025-Q2", "Client", 423.0, 236.0, 90.0),
    ("FY2025-Q3", "Data Center", 1490.0, 585.0, 228.0),
    ("FY2025-Q3", "Automotive", 345.0, 180.0, 79.0),
    ("FY2025-Q3", "Client", 455.0, 249.0, 93.0),
    ("FY2025-Q4", "Data Center", 1710.0, 651.0, 241.0),
    ("FY2025-Q4", "Automotive", 368.0, 189.0, 82.0),
    ("FY2025-Q4", "Client", 482.0, 259.0, 96.0),
]

_SUPPLIER_SPEND = [
    ("FY2025-Q1", "Helios Foundry", 270.0, "Wafer fabrication"),
    ("FY2025-Q2", "Helios Foundry", 290.0, "Wafer fabrication"),
    ("FY2025-Q3", "Helios Foundry", 315.0, "Wafer fabrication"),
    ("FY2025-Q4", "Helios Foundry", 345.0, "Wafer fabrication"),
    ("FY2025-Q1", "Meridian Substrates", 62.0, "Advanced packaging"),
    ("FY2025-Q2", "Meridian Substrates", 66.0, "Advanced packaging"),
    ("FY2025-Q3", "Meridian Substrates", 71.0, "Advanced packaging"),
    ("FY2025-Q4", "Meridian Substrates", 78.0, "Advanced packaging"),
]


def init_financials_db(db_path: Path | None = None) -> Path:
    db_path = db_path or config.FINANCE_DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript("""
            DROP TABLE IF EXISTS quarterly_financials;
            DROP TABLE IF EXISTS supplier_spend;
            CREATE TABLE quarterly_financials (
                fiscal_quarter TEXT NOT NULL,
                segment TEXT NOT NULL,
                revenue_usd_m REAL NOT NULL,
                cogs_usd_m REAL NOT NULL,
                opex_usd_m REAL NOT NULL
            );
            CREATE TABLE supplier_spend (
                fiscal_quarter TEXT NOT NULL,
                supplier TEXT NOT NULL,
                spend_usd_m REAL NOT NULL,
                category TEXT NOT NULL
            );
        """)
        conn.executemany(
            "INSERT INTO quarterly_financials VALUES (?, ?, ?, ?, ?)", _QUARTERLY
        )
        conn.executemany(
            "INSERT INTO supplier_spend VALUES (?, ?, ?, ?)", _SUPPLIER_SPEND
        )
    return db_path


def validate_sql(sql: str) -> str:
    """Reject anything that isn't a single read-only SELECT/CTE statement."""
    cleaned = sql.strip().rstrip(";").strip()
    if not cleaned:
        raise ValueError("empty query")
    if ";" in cleaned:
        raise ValueError("only a single SQL statement is allowed")
    if not cleaned.lower().startswith(("select", "with")):
        raise ValueError("only SELECT queries are allowed")
    match = _FORBIDDEN.search(cleaned)
    if match:
        raise ValueError(f"forbidden keyword: {match.group(0)}")
    return cleaned


def run_query(sql: str, db_path: Path | None = None, max_rows: int = 200) -> dict:
    db_path = db_path or config.FINANCE_DB_PATH
    cleaned = validate_sql(sql)
    # Read-only URI connection: defense in depth on top of validate_sql.
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        cursor = conn.execute(cleaned)
        columns = [c[0] for c in cursor.description]
        rows = cursor.fetchmany(max_rows)
        return {"columns": columns, "rows": [list(r) for r in rows]}
    finally:
        conn.close()
