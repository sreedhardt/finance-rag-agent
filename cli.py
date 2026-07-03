"""CLI entry points:
    uv run python cli.py ingest            # build index, graph, and db
    uv run python cli.py ask "question"    # one-shot agentic Q&A
"""

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Finance RAG Agent")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ingest", help="Run the full ingestion pipeline")
    ask = sub.add_parser("ask", help="Ask the agent a question")
    ask.add_argument("question")
    ask.add_argument("--show-steps", action="store_true", help="Print each tool call")
    args = parser.parse_args()

    if args.command == "ingest":
        from finance_rag.pipeline import run_ingestion
        run_ingestion()
        return

    from finance_rag.agent import FinanceAgent
    from finance_rag.embedder import GeminiEmbedder
    from finance_rag.graph import KnowledgeGraph
    from finance_rag.indexer import VectorIndex
    from finance_rag.tools import AgentTools

    index = VectorIndex(GeminiEmbedder())
    if index.count() == 0:
        sys.exit("Index is empty — run `python cli.py ingest` first.")
    agent = FinanceAgent(AgentTools(index, KnowledgeGraph.load()))
    result = agent.ask(args.question)

    if args.show_steps:
        for i, step in enumerate(result.steps, 1):
            print(f"--- step {i}: {step.tool}({step.args})")
    print()
    print(result.answer)


if __name__ == "__main__":
    main()
