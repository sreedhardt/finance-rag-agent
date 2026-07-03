"""Central configuration. Every knob is overridable via environment variables
so the same code runs locally, in Docker, and inside an orchestrator task."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = Path(os.getenv("RAW_DATA_DIR", PROJECT_ROOT / "data" / "raw"))
PROCESSED_DIR = Path(os.getenv("PROCESSED_DIR", PROJECT_ROOT / "data" / "processed"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", PROCESSED_DIR / "chroma"))
GRAPH_PATH = Path(os.getenv("GRAPH_PATH", PROCESSED_DIR / "knowledge_graph.json"))
FINANCE_DB_PATH = Path(os.getenv("FINANCE_DB_PATH", PROCESSED_DIR / "finance.db"))

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))

CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", "1400"))
CHUNK_OVERLAP_CHARS = int(os.getenv("CHUNK_OVERLAP_CHARS", "200"))

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "finance_docs")
MAX_AGENT_TURNS = int(os.getenv("MAX_AGENT_TURNS", "8"))
