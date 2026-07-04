"""Groq-backed agent: the same agentic loop as agent.FinanceAgent, driven
through Groq's OpenAI-compatible chat-completions tool calling. Tool
declarations are derived from the single Gemini-format source of truth in
tools.py, so the providers share one tool surface."""

from __future__ import annotations

import json

from groq import Groq

from . import config
from .agent import SYSTEM_PROMPT, AgentResult, AgentStep
from .tools import AgentTools, openai_tool_declarations


class GroqFinanceAgent:
    def __init__(self, tools: AgentTools, client: Groq | None = None,
                 model: str = config.GROQ_MODEL):
        self._tools = tools
        self._client = client or Groq()  # reads GROQ_API_KEY from env
        self._model = model

    def ask(self, question: str, history: list[dict] | None = None) -> AgentResult:
        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(history or [])
        messages.append({"role": "user", "content": question})

        steps: list[AgentStep] = []
        for _ in range(config.MAX_AGENT_TURNS):
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                tools=openai_tool_declarations(),
                tool_choice="auto",
                temperature=0.2,
            )
            message = response.choices[0].message
            if not message.tool_calls:
                return AgentResult(answer=message.content or "(no answer)", steps=steps)

            messages.append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [tc.model_dump() for tc in message.tool_calls],
            })
            for call in message.tool_calls:
                try:
                    args = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = self._tools.dispatch(call.function.name, args)
                steps.append(AgentStep(tool=call.function.name, args=args, result=result))
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result, default=str),
                })

        return AgentResult(
            answer="Stopped: reached the maximum number of tool-use turns "
                   f"({config.MAX_AGENT_TURNS}) without a final answer.",
            steps=steps,
        )
