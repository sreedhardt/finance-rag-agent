"""Agentic loop: Gemini function calling with manual dispatch.

The loop is written by hand (no framework) so every tool call is observable,
loggable, and gateable — the model proposes tool calls, we execute them, feed
results back, and repeat until it produces a final grounded answer."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from google import genai
from google.genai import types

from . import config
from .tools import TOOL_DECLARATIONS, AgentTools

SYSTEM_PROMPT = """\
You are a financial document intelligence agent for Orion Semiconductor's
finance team. You answer questions using ONLY evidence gathered through your
tools — never from memory.

Available evidence sources:
- search_documents / get_document: unstructured documents (10-K excerpt,
  supplier contract, tax report).
- query_financials: the structured ledger (exact revenue, cost, supplier spend).
- graph_lookup: the knowledge graph linking entities to documents and each other.

Method:
1. Break the question down. For cross-source questions (e.g. comparing a
   contract term against actual spend), gather each piece separately.
2. Use graph_lookup to discover which documents/entities are relevant when the
   question involves relationships.
3. Use query_financials for any exact figure or aggregation; prefer it over
   numbers quoted in prose when they conflict, and flag the conflict.
4. Cite every factual claim: [chunk_id] for document evidence (e.g.
   [orion_10k_fy2025_excerpt#3]) and [db:table_name] for SQL results.
5. If the evidence is insufficient, say exactly what is missing. Do not guess.

Keep answers concise and structured. End with a "Sources" line listing the
citations used."""


@dataclass
class AgentStep:
    tool: str
    args: dict
    result: dict


@dataclass
class AgentResult:
    answer: str
    steps: list[AgentStep] = field(default_factory=list)


class FinanceAgent:
    def __init__(self, tools: AgentTools, client: genai.Client | None = None,
                 model: str = config.GEMINI_MODEL):
        self._tools = tools
        self._client = client or genai.Client()
        self._model = model

    def ask(self, question: str, history: list[types.Content] | None = None) -> AgentResult:
        contents: list[types.Content] = list(history or [])
        contents.append(types.Content(role="user", parts=[types.Part(text=question)]))
        gen_config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=[types.Tool(function_declarations=TOOL_DECLARATIONS)],
            temperature=0.2,
        )

        steps: list[AgentStep] = []
        for _ in range(config.MAX_AGENT_TURNS):
            response = self._client.models.generate_content(
                model=self._model, contents=contents, config=gen_config,
            )
            calls = response.function_calls
            if not calls:
                return AgentResult(answer=response.text or "(no answer)", steps=steps)

            contents.append(response.candidates[0].content)
            result_parts = []
            for call in calls:
                args = dict(call.args or {})
                result = self._tools.dispatch(call.name, args)
                steps.append(AgentStep(tool=call.name, args=args, result=result))
                result_parts.append(types.Part.from_function_response(
                    name=call.name,
                    response={"result": json.dumps(result, default=str)},
                ))
            contents.append(types.Content(role="user", parts=result_parts))

        return AgentResult(
            answer="Stopped: reached the maximum number of tool-use turns "
                   f"({config.MAX_AGENT_TURNS}) without a final answer.",
            steps=steps,
        )
