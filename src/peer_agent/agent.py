from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai import Agent as PydanticAgent
from pydantic_ai import Tool as PaiTool
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import UsageLimits

from peer_agent.types import Verdict

_VERDICT_RE = re.compile(r"VERDICT:\s*([A-Z_]+)")


class OverBudget(Exception):
    pass


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict
    result: object


@dataclass
class Trajectory:
    calls: list[ToolCall] = field(default_factory=list)
    answer: str = ""
    done: bool = False
    verdict: Verdict | None = None


class Agent:
    def __init__(self, llm: Any, tools: list, max_turns: int = 8):
        self.llm = llm
        self.tools = tools
        self.max_turns = max_turns

    @classmethod
    def from_env(cls) -> Agent:
        raise NotImplementedError("phase 2: real model wiring")

    def review(self, case: Any) -> Trajectory:
        calls: list[ToolCall] = []

        def bind(tool: Any) -> Any:
            def bound(**kwargs: Any) -> Any:
                result = tool.fn(case.df)
                calls.append(ToolCall(tool.name, kwargs, result))
                return result

            return bound

        pai_tools = [
            PaiTool.from_schema(
                function=bind(t),
                name=t.name,
                description=t.description,
                json_schema=t.schema,
                takes_ctx=False,
            )
            for t in self.tools
        ]

        def adapter(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            del info
            turn = self.llm(messages, self.tools)
            if isinstance(turn, str):
                return ModelResponse(parts=[TextPart(turn)])
            return ModelResponse(parts=[ToolCallPart(tool_name=n, args=a) for n, a in turn])

        pai_agent = PydanticAgent(FunctionModel(adapter), tools=pai_tools)
        try:
            result = pai_agent.run_sync(
                "Review this experiment.",
                usage_limits=UsageLimits(request_limit=self.max_turns),
            )
        except UsageLimitExceeded as exc:
            raise OverBudget(str(exc)) from exc

        answer = result.output
        match = _VERDICT_RE.search(answer.upper())
        verdict = (
            Verdict[match.group(1)]
            if match and match.group(1) in Verdict.__members__
            else None
        )
        return Trajectory(calls=calls, answer=answer, done=True, verdict=verdict)
