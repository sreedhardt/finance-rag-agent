"""Tool surface exposed to the agent: semantic search, full-document fetch,
knowledge-graph lookup, and guard-railed SQL over the financials database."""

from __future__ import annotations

from google.genai import types

from . import findb
from .graph import KnowledgeGraph
from .indexer import VectorIndex

TOOL_DECLARATIONS = [
    types.FunctionDeclaration(
        name="search_documents",
        description=(
            "Semantic search over ingested financial documents (SEC filings, "
            "supplier contracts, tax reports). Returns the most relevant chunks "
            "with chunk_ids for citation. Use this first for any question about "
            "document content."
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "query": types.Schema(type=types.Type.STRING,
                                      description="Natural-language search query."),
                "doc_type": types.Schema(
                    type=types.Type.STRING,
                    description="Optional filter: sec_filing | contract | tax_report.",
                ),
            },
            required=["query"],
        ),
    ),
    types.FunctionDeclaration(
        name="get_document",
        description="Fetch the full text of one document by doc_id (from search or graph results).",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "doc_id": types.Schema(type=types.Type.STRING),
            },
            required=["doc_id"],
        ),
    ),
    types.FunctionDeclaration(
        name="graph_lookup",
        description=(
            "Look up an entity (company, supplier, agreement, business segment) in the "
            "knowledge graph. Returns its relations and the documents that mention it. "
            "Use it to discover connections across documents before searching."
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "entity": types.Schema(type=types.Type.STRING,
                                       description="Entity name or partial name."),
            },
            required=["entity"],
        ),
    ),
    types.FunctionDeclaration(
        name="query_financials",
        description=(
            "Run a single read-only SELECT against the structured financials database "
            "(SQLite). Use this for exact numbers, aggregations, and cross-checks "
            "against document claims.\nSchema:\n" + findb.SCHEMA_DESCRIPTION
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "sql": types.Schema(type=types.Type.STRING,
                                    description="A single SELECT statement."),
            },
            required=["sql"],
        ),
    ),
]


def _schema_to_json(schema: types.Schema) -> dict:
    """Convert a Gemini types.Schema to plain JSON Schema (OpenAI tool format)."""
    out: dict = {}
    if schema.type is not None:
        out["type"] = schema.type.name.lower()
    if schema.description:
        out["description"] = schema.description
    if schema.properties:
        out["properties"] = {k: _schema_to_json(v) for k, v in schema.properties.items()}
    if schema.required:
        out["required"] = list(schema.required)
    return out


def openai_tool_declarations() -> list[dict]:
    """The same tool surface in OpenAI/Groq chat-completions format, derived
    from TOOL_DECLARATIONS so the two providers can never drift apart."""
    return [
        {
            "type": "function",
            "function": {
                "name": decl.name,
                "description": decl.description,
                "parameters": _schema_to_json(decl.parameters),
            },
        }
        for decl in TOOL_DECLARATIONS
    ]


class AgentTools:
    """Binds tool names to implementations over the live index/graph/db."""

    def __init__(self, index: VectorIndex, graph: KnowledgeGraph):
        self.index = index
        self.graph = graph

    def dispatch(self, name: str, args: dict) -> dict:
        try:
            handler = getattr(self, f"_tool_{name}", None)
            if handler is None:
                return {"error": f"unknown tool: {name}"}
            return handler(**args)
        except Exception as exc:  # surfaced to the model so it can recover
            return {"error": f"{type(exc).__name__}: {exc}"}

    def _tool_search_documents(self, query: str, doc_type: str | None = None) -> dict:
        hits = self.index.search(query, k=5, doc_type=doc_type or None)
        return {"results": hits} if hits else {"results": [], "note": "no matches"}

    def _tool_get_document(self, doc_id: str) -> dict:
        text = self.index.get_document_text(doc_id)
        if text is None:
            known = [d["doc_id"] for d in self.index.list_documents()]
            return {"error": f"doc_id '{doc_id}' not found", "available": known}
        return {"doc_id": doc_id, "text": text}

    def _tool_graph_lookup(self, entity: str) -> dict:
        relations = self.graph.neighbors(entity)
        if not relations:
            return {"relations": [], "known_entities": self.graph.entity_names()}
        return {"relations": relations}

    def _tool_query_financials(self, sql: str) -> dict:
        return findb.run_query(sql)
