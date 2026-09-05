from __future__ import annotations

import dataclasses
import functools
import json
import os
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from importlib import resources
from pathlib import Path
from typing import Any

import logfire
from pydantic_ai import Agent as PydanticAgent
from pydantic_ai import Tool as PaiTool
from pydantic_ai.exceptions import ModelRetry, UsageLimitExceeded
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models import Model
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.usage import UsageLimits
from pydantic_ai_harness import Skills
from pydantic_ai_harness.filesystem import FileSystem

from peer_agent.sandbox import Sandbox
from peer_agent.tools import analysis_tools, design_tools
from peer_agent.types import DesignSpec, Verdict

_VERDICT_RE = re.compile(r"VERDICT:\s*([A-Z_]+)")
_NUMBER_RE = re.compile(r"-?\d+\.?\d*")
_PROMPT = resources.files("peer_agent").joinpath("prompt.md").read_text()
_SKILLS_DIR = Path(str(resources.files("peer_agent").joinpath("skills")))
_REPORTS_DIR = Path("reports")
_PREREG_PREAMBLE = (
    "Pre-registered plan for this experiment. These choices were fixed before "
    "anyone saw the data — follow them, and say so plainly if you depart from any."
)


def _files_capability() -> FileSystem:
    """
    Lets the model write its own review/design write-up to disk when asked
    ("save this as a report") — pydantic-ai-harness's own FileSystem tool,
    scoped to reports/ so it can't touch anything else in the repo.
    """
    _REPORTS_DIR.mkdir(exist_ok=True)
    return FileSystem(root_dir=_REPORTS_DIR)


class OverBudget(Exception):
    pass


_logfire_configured = False


def configure_logfire() -> None:
    """
    Idempotent: safe to call from both cli.py's main() and Agent.from_env(), the
    two process-start points a real run comes through. Zero-config locally
    (`send_to_logfire='if-token-present'`); the moment LOGFIRE_TOKEN is set in
    .env, the same code starts shipping traces to a hosted project.
    """
    global _logfire_configured
    if _logfire_configured:
        return
    logfire.configure(send_to_logfire="if-token-present")
    logfire.instrument_pydantic_ai()
    _logfire_configured = True


def json_default(obj: Any) -> Any:
    """Shared by the JSONL run log and evals.py's results.json — dataclasses and
    enums (SRMResult, Verdict, ...) show up as tool-call results in both."""
    if isinstance(obj, Enum):
        return obj.name
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    return str(obj)


def _log_trajectory(kind: str, traj: Trajectory, *, case_id: str | None = None) -> None:
    """
    Append one JSON line per completed review()/design() call to a local log,
    independent of Logfire — grep-able with no network, no token, no browser.
    """
    path = Path(os.environ.get("PEER_LOG_PATH", "logs/runs.jsonl"))
    path.parent.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "ts": datetime.now(UTC).isoformat(),
        "kind": kind,
        "calls": [{"name": c.name, "args": c.args, "result": c.result} for c in traj.calls],
        "answer": traj.answer,
    }
    if case_id is not None:
        record["case_id"] = case_id
    if traj.verdict is not None:
        record["verdict"] = traj.verdict.name
    if traj.spec is not None:
        record["spec"] = traj.spec
    with path.open("a") as f:
        f.write(json.dumps(record, default=json_default) + "\n")


def _srm_broken(calls: list[ToolCall]) -> bool:
    return any(
        c.name == "check_srm" and getattr(c.result, "mismatch", False) for c in calls
    )


class _RecordingCall(functools.partial):
    """
    A tool call bound to its fixed leading args (e.g. the case dataframe), kept
    as a real functools.partial so pydantic-ai still sees fn's true signature
    and validates/coerces args against it — a plain **kwargs wrapper would hide
    the real parameter types and skip that validation entirely.
    """

    def __call__(self, /, **kwargs: Any) -> Any:
        try:
            result = super().__call__(**kwargs)
        except ValueError as exc:
            # e.g. a stats check needing a column this case doesn't have — let the
            # model see the failure and adjust, instead of crashing the whole run.
            raise ModelRetry(str(exc)) from exc
        self._calls.append(ToolCall(self._tool_name, kwargs, result))
        return result


def _spec_defaults(spec: DesignSpec | None) -> dict[str, dict[str, Any]]:
    """
    The analysis choices a pre-registration pins down, per tool. Bound as partial
    keywords, so pydantic-ai reports them to the model as the *defaults* — it can
    still override one, and the override lands in the trajectory where a reader
    can see the departure from plan.

    `check_srm` and `check_guardrails` keep their own alphas: a split check and a
    safety check answer different questions from the primary metric, and reusing
    the primary alpha for them would loosen both.
    """
    if spec is None:
        return {}
    common = {"control": spec.control_value}
    primary = common | {"alpha": spec.alpha}
    return {
        "check_srm": common,
        "check_novelty": common,
        "check_guardrails": common,
        "scan_segments": primary,
        "analyze": primary | {"covariate": spec.covariate},
        "sequential": primary | {"looks": spec.looks},
    }


def _bind(tool: Any, calls: list[ToolCall], *bound_args: Any, **bound_kwargs: Any) -> Any:
    call = _RecordingCall(tool.fn, *bound_args, **bound_kwargs)
    call.__name__ = tool.fn.__name__
    call.__qualname__ = tool.fn.__qualname__
    call.__doc__ = tool.fn.__doc__
    call._calls = calls
    call._tool_name = tool.name
    return call


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
    spec: DesignSpec | None = None


class Agent:
    def __init__(self, llm: Any, tools: list, max_turns: int = 16):
        self.llm = llm
        self.tools = tools
        self.max_turns = max_turns

    @classmethod
    def from_env(cls) -> Agent:
        configure_logfire()
        required = ("LLM_MODEL", "LLM_API_KEY", "LLM_BASE_URL")
        missing = [name for name in required if not os.environ.get(name)]
        if missing:
            raise RuntimeError(f"{', '.join(missing)} not set — see .env.example")
        provider = OpenAIProvider(
            base_url=os.environ["LLM_BASE_URL"],
            api_key=os.environ["LLM_API_KEY"],
        )
        # This endpoint 500s if a conversation-storage request is left at its default.
        settings = OpenAIChatModelSettings(openai_store=False)
        model = OpenAIChatModel(
            os.environ["LLM_MODEL"], provider=provider, settings=settings
        )
        return cls(model, tools=analysis_tools(Sandbox()))

    def _model(self, adapter: Any) -> Any:
        return self.llm if isinstance(self.llm, Model) else FunctionModel(adapter)

    def chat_agent(self, df: Any) -> PydanticAgent:
        """
        The same tool registry, prompt, and skill review() uses, bound to df the
        same way, minus the Trajectory-capturing wrapper — for `peer chat`/`peer web`.
        """
        pai_tools = [PaiTool(_bind(t, [], df), takes_ctx=False) for t in self.tools]
        return PydanticAgent(
            self.llm,
            tools=pai_tools,
            system_prompt=_PROMPT,
            capabilities=[
                Skills(_SKILLS_DIR, include=["review-protocol"]),
                _files_capability(),
            ],
        )

    def review(
        self,
        case: Any,
        spec: DesignSpec | None = None,
        question: str = "Review this experiment.",
    ) -> Trajectory:
        calls: list[ToolCall] = []
        defaults = _spec_defaults(spec)
        pai_tools = [
            PaiTool(_bind(t, calls, case.df, **defaults.get(t.name, {})), takes_ctx=False)
            for t in self.tools
        ]
        if spec is not None:
            question = f"{_PREREG_PREAMBLE}\n\n{spec}\n\n{question}"

        def adapter(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            del info
            turn = self.llm(messages, self.tools)
            if isinstance(turn, str):
                return ModelResponse(parts=[TextPart(turn)])
            return ModelResponse(parts=[ToolCallPart(tool_name=n, args=a) for n, a in turn])

        pai_agent = PydanticAgent(
            self._model(adapter),
            tools=pai_tools,
            system_prompt=_PROMPT,
            capabilities=[
                Skills(_SKILLS_DIR, include=["review-protocol"]),
                _files_capability(),
            ],
        )
        try:
            result = pai_agent.run_sync(
                question,
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
        if _srm_broken(calls):
            # Whether the model actually withholds every figure describing a broken
            # split is a prompting question, not a guarantee — enforce it here so the
            # invariant holds regardless of what the model decides to write.
            answer = _NUMBER_RE.sub("[redacted]", answer)
        traj = Trajectory(calls=calls, answer=answer, done=True, verdict=verdict, spec=spec)
        _log_trajectory("review", traj)
        return traj

    def design(self, brief: str) -> Trajectory:
        calls: list[ToolCall] = []
        tools = design_tools()
        pai_tools = [PaiTool(_bind(t, calls), takes_ctx=False) for t in tools]

        def adapter(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            del info
            turn = self.llm(messages, tools)
            if isinstance(turn, str):
                return ModelResponse(parts=[TextPart(turn)])
            return ModelResponse(parts=[ToolCallPart(tool_name=n, args=a) for n, a in turn])

        pai_agent = PydanticAgent(
            self._model(adapter),
            tools=pai_tools,
            output_type=DesignSpec | str,
            system_prompt=_PROMPT,
            capabilities=[
                Skills(_SKILLS_DIR, include=["design-protocol"]),
                _files_capability(),
            ],
        )
        try:
            result = pai_agent.run_sync(
                brief, usage_limits=UsageLimits(request_limit=self.max_turns)
            )
        except UsageLimitExceeded as exc:
            raise OverBudget(str(exc)) from exc

        if isinstance(result.output, DesignSpec):
            traj = Trajectory(calls=calls, spec=result.output, done=True)
        else:
            traj = Trajectory(calls=calls, answer=str(result.output), done=True)
        _log_trajectory("design", traj)
        return traj
