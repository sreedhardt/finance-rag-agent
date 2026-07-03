import pytest

from finance_rag import findb


@pytest.fixture
def db_path(tmp_path):
    return findb.init_financials_db(tmp_path / "finance.db")


def test_seed_data_reconciles_with_documents(db_path):
    # Totals asserted here are the same figures quoted in the sample documents.
    result = findb.run_query(
        "SELECT SUM(revenue_usd_m) FROM quarterly_financials", db_path
    )
    assert result["rows"][0][0] == pytest.approx(8700.0)

    result = findb.run_query(
        "SELECT SUM(spend_usd_m) FROM supplier_spend WHERE supplier = 'Helios Foundry'",
        db_path,
    )
    assert result["rows"][0][0] == pytest.approx(1220.0)  # ≥ $1,200M commitment


def test_select_returns_columns_and_rows(db_path):
    result = findb.run_query(
        "SELECT fiscal_quarter, revenue_usd_m FROM quarterly_financials "
        "WHERE segment = 'Data Center' ORDER BY fiscal_quarter",
        db_path,
    )
    assert result["columns"] == ["fiscal_quarter", "revenue_usd_m"]
    assert len(result["rows"]) == 4


@pytest.mark.parametrize("bad_sql", [
    "DROP TABLE quarterly_financials",
    "INSERT INTO supplier_spend VALUES ('x','y',1,'z')",
    "SELECT 1; DELETE FROM supplier_spend",
    "PRAGMA table_info(quarterly_financials)",
    "UPDATE quarterly_financials SET revenue_usd_m = 0",
    "",
])
def test_guard_rejects_non_select(db_path, bad_sql):
    with pytest.raises(ValueError):
        findb.run_query(bad_sql, db_path)


def test_cte_is_allowed(db_path):
    result = findb.run_query(
        "WITH dc AS (SELECT revenue_usd_m FROM quarterly_financials "
        "WHERE segment='Data Center') SELECT COUNT(*) FROM dc",
        db_path,
    )
    assert result["rows"][0][0] == 4
