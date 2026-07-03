"""Airflow DAG wrapping the ingestion pipeline stages as individual tasks.

The pipeline stages in finance_rag.pipeline are plain functions, so each maps
1:1 onto a task — validate acts as a quality gate that fails the run before
anything reaches the index. Requires `pip install apache-airflow` in the
scheduler environment; the app itself does not depend on Airflow.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator


def _extract_and_validate(**_):
    from finance_rag.pipeline import stage_extract, stage_validate
    docs = stage_extract()
    stage_validate(docs)  # raises → task fails → downstream never runs


def _index(**_):
    from finance_rag.embedder import GeminiEmbedder
    from finance_rag.indexer import VectorIndex
    from finance_rag.pipeline import stage_extract, stage_index, stage_validate
    docs = stage_extract()
    chunked = stage_validate(docs)
    stage_index(docs, chunked, VectorIndex(GeminiEmbedder()))


def _graph_and_db(**_):
    from finance_rag import findb
    from finance_rag.pipeline import stage_extract, stage_graph
    stage_graph(stage_extract())
    findb.init_financials_db()


with DAG(
    dag_id="finance_rag_ingestion",
    description="Ingest financial documents into the RAG index with a quality gate",
    schedule="0 6 * * *",  # daily, after upstream document drops land
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args={"retries": 2, "retry_delay": timedelta(minutes=5)},
) as dag:
    quality_gate = PythonOperator(
        task_id="extract_and_validate", python_callable=_extract_and_validate
    )
    build_index = PythonOperator(task_id="embed_and_index", python_callable=_index)
    build_graph_db = PythonOperator(
        task_id="graph_and_structured_db", python_callable=_graph_and_db
    )

    quality_gate >> build_index >> build_graph_db
