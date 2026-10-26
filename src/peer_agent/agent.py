from __future__ import annotations

import dataclasses
import functools
import json
import math
import os
import re
import threading
import time
from collections.abc import Callable
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
# Thousands separators have to be part of the literal. Without them "50,000"
# reads as a 50 (too small to check) beside a 000, so a sample size invented
# whole sails through, and "1,234,567" gets reported as the figures 234 and 567.
_LITERAL_RE = re.compile(r"(?<![\w.,])(-?\d{1,3}(?:,\d{3})+|-?\d+)(\.\d+)?\s*(%?)")
_LIST_MARKER_RE = re.compile(r"^(\s*)\d+\.(\s)", re.MULTILINE)
_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
# Significance levels, confidence levels and even splits turn up in any write-up
# and belong to the protocol rather than to this dataset.
_CONVENTIONAL = frozenset(
    {0.0, 1.0, 5.0, 50.0, 95.0, 99.0, 100.0, 0.01, 0.05, 0.1, 0.5, 0.8, 0.9, 0.95, 0.99}
)
_REWRITES_ALLOWED = 1  # one chance to fix its own write-up before we redact it
_PROMPT = resources.files("peer_agent").joinpath("prompt.md").read_text()
_SKILLS_DIR = Path(str(resources.files("peer_agent").joinpath("skills")))
_REPORTS_DIR = Path("reports")
_PREREG_PREAMBLE = (
    "Pre-registered plan for this experiment. These choices were fixed before "
    "anyone saw the data — follow them, and say so plainly if you depart from any."
)
ROWS_ARE_LOADED = (
    "The experiment rows are already loaded. The tools read them. "
    "Do not ask for a file. Call the tools, then give a VERDICT."
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


def request_timeout_s() -> float:
    """Wall-clock limit for one model request (HTTP layer). Default three minutes."""
    return float(os.environ.get("PEER_REQUEST_TIMEOUT_S", "180"))


def openai_provider(base_url: str, api_key: str) -> OpenAIProvider:
    import httpx

    client = httpx.AsyncClient(timeout=httpx.Timeout(request_timeout_s()))
    return OpenAIProvider(base_url=base_url, api_key=api_key, http_client=client)


def chat_settings(base_url: str) -> OpenAIChatModelSettings:
    """Leave OpenAI's `store` field off the request when the host is OpenRouter.

    OpenRouter returns 400 for parameters the model page does not list. Other
    compatible endpoints 500 when conversation storage stays at its default, so
    those keep `store` off.
    """
    if "openrouter.ai" in base_url:
        return OpenAIChatModelSettings()
    return OpenAIChatModelSettings(openai_store=False)


_logfire_configured = False
_LOG_LOCK = threading.Lock()


def configure_logfire() -> None:
    """
    Idempotent: safe to call from both cli.py's main() and Agent.from_env(), the
    two process-start points a real run comes through. Zero-config locally
    (`send_to_logfire='if-token-present'`); the moment LOGFIRE_TOKEN is set in
    .env, the same code starts shipping traces to a hosted project.

    Set OTEL_EXPORTER_OTLP_ENDPOINT to also export spans over OTLP/HTTP. Logfire
    reads that variable through the OpenTelemetry exporter; no other setting is
    required.
    """
    global _logfire_configured
    if _logfire_configured:
        return
    processors: list[Any] = []
    if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        processors.append(SimpleSpanProcessor(OTLPSpanExporter()))
    logfire.configure(
        send_to_logfire="if-token-present",
        console=False,
        additional_span_processors=processors,
    )
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


def _failed_trajectory(
    kind: str,
    calls: list[ToolCall],
    *,
    started: float,
    exc: Exception,
    spec: DesignSpec | None = None,
) -> Trajectory:
    traj = Trajectory(
        calls=calls,
        done=False,
        error=f"{type(exc).__name__}: {exc}",
        spec=spec,
        latency_s=time.perf_counter() - started,
    )
    _log_trajectory(kind, traj)
    return traj


def _log_trajectory(kind: str, traj: Trajectory, *, case_id: str | None = None) -> None:
    """
    Append one JSON line per completed review()/design() call to a local log,
    independent of Logfire — grep-able with no network, no token, no browser.
    """
    path = Path(os.environ.get("PEER_LOG_PATH") or "logs/runs.jsonl")
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
    line = json.dumps(record, default=json_default) + "\n"
    with _LOG_LOCK, path.open("a") as f:
        f.write(line)


def _srm_broken(calls: list[ToolCall]) -> bool:
    return any(
        c.name == "check_srm" and getattr(c.result, "mismatch", False) for c in calls
    )


def _numbers_in(obj: Any, found: set[float]) -> None:
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, int | float):
        found.add(float(obj))
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        _numbers_in(dataclasses.asdict(obj), found)
    elif isinstance(obj, dict):
        for value in obj.values():
            _numbers_in(value, found)
    elif isinstance(obj, list | tuple | set):
        for value in obj:
            _numbers_in(value, found)


def _computed(calls: list[ToolCall]) -> set[float]:
    """
    Every number the toolbox touched. Arguments count alongside results: an alpha
    the model passed in is as traceable as a p-value it got back, and excluding
    them would flag "we tested at the pre-registered alpha" as an invention.
    """
    found: set[float] = set()
    for call in calls:
        _numbers_in(call.args, found)
        _numbers_in(call.result, found)
    return found


def _agrees(written: float, computed: float, decimals: int) -> bool:
    """Does `computed`, rounded or truncated the way the author wrote it, land on
    `written`? Sign is ignored — "down 7%" is a fair reading of -0.0695."""
    scale = 10**decimals
    for value in (computed, -computed):
        truncated = math.trunc(value * scale) / scale
        if abs(round(value, decimals) - written) < 1e-9 or abs(truncated - written) < 1e-9:
            return True
    return False


def unsupported_numbers(traj: Trajectory) -> tuple[str, ...]:
    """
    Numeric literals in the answer that no tool call accounts for, in the order
    they were written.

    A literal is checked when it carries a decimal point, wears a percent sign,
    or is at least 100 — a bare small integer is prose ("two arms", "14 days",
    an ordered-list marker), not a statistic, and checking those is all false
    positives. It is supported when some number the toolbox produced or was
    given reproduces it at the precision the author chose to write it at, either
    as-is or scaled by 100 so a lift of -0.0695 covers "6.95%" and "7%".

    The rule is deliberately generous: this is a guard against figures invented
    whole, not an audit of arithmetic.
    """
    computed = _computed(traj.calls)
    prose = _DATE_RE.sub("", _LIST_MARKER_RE.sub(r"\1#\2", traj.answer))
    unsupported = []
    for whole, fraction, percent in _LITERAL_RE.findall(prose):
        written = float(whole.replace(",", "") + fraction)
        grouped = "," in whole
        checkable = fraction or percent or grouped or abs(written) >= 100
        a_year = not fraction and not percent and 1900 <= written <= 2100
        if not checkable or a_year or abs(written) in _CONVENTIONAL:
            continue
        decimals = len(fraction.lstrip("."))
        if not any(
            _agrees(abs(written), value * scale, decimals)
            for value in computed
            for scale in (1, 100)
        ):
            unsupported.append(whole + fraction + percent)
    return tuple(unsupported)


def _cite_the_toolbox(calls: list[ToolCall]) -> Callable[[str], str]:
    """
    An output validator: sends the model back to fix a write-up that quotes a
    figure nothing computed, and strikes the figure out if it does it again.
    """
    rewrites = 0

    def validate(output: str) -> str:
        nonlocal rewrites
        invented = unsupported_numbers(Trajectory(calls=calls, answer=output))
        if not invented:
            return output
        rewrites += 1
        if rewrites > _REWRITES_ALLOWED:
            return _redact(output, invented)
        raise ModelRetry(
            f"No tool produced these figures in your answer: {', '.join(invented)}. "
            "Measure them or drop them — do not estimate a number you have not "
            "computed."
        )

    return validate


def _verdict_error(answer: str, verdict: Verdict | None) -> str | None:
    if verdict is not None:
        return None
    return (
        "empty_model_response"
        if not str(answer).strip()
        else "missing_or_invalid_verdict_line"
    )


def _redact(answer: str, invented: tuple[str, ...]) -> str:
    """Last resort, once the model has had its chance: strike the figures out
    rather than hand a reader a number nothing stands behind."""
    for literal in invented:
        number = re.escape(literal.rstrip("%"))
        answer = re.sub(rf"(?<![\w.]){number}(?![\d.])", "[unsupported]", answer, count=1)
    return answer


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
        "check_srm": common | {"ratio": spec.ratio},
        "check_novelty": common,
        "check_guardrails": common | {"guardrails": spec.guardrails},
        "scan_segments": primary,
        "analyze": primary
        | {"covariate": spec.covariate, "mde": spec.mde, "unit": spec.unit_column},
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


@dataclass(frozen=True)
class Usage:
    """Token counts from pydantic-ai's RunUsage. Missing fields stay None."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    requests: int | None = None


def _read_usage(result: Any) -> Usage | None:
    raw_usage = getattr(result, "usage", None)
    if raw_usage is None:
        return None
    raw = raw_usage() if callable(raw_usage) else raw_usage
    return Usage(
        input_tokens=getattr(raw, "input_tokens", None),
        output_tokens=getattr(raw, "output_tokens", None),
        requests=getattr(raw, "requests", None),
    )


@dataclass
class Trajectory:
    calls: list[ToolCall] = field(default_factory=list)
    answer: str = ""
    done: bool = False
    verdict: Verdict | None = None
    spec: DesignSpec | None = None
    usage: Usage | None = None
    latency_s: float = 0.0
    error: str | None = None


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
        base_url = os.environ["LLM_BASE_URL"]
        provider = openai_provider(base_url, os.environ["LLM_API_KEY"])
        model = OpenAIChatModel(
            os.environ["LLM_MODEL"], provider=provider, settings=chat_settings(base_url)
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
        question: str | None = None,
    ) -> Trajectory:
        calls: list[ToolCall] = []
        defaults = _spec_defaults(spec)
        pai_tools = [
            PaiTool(_bind(t, calls, case.df, **defaults.get(t.name, {})), takes_ctx=False)
            for t in self.tools
        ]
        if question is None:
            question = f"Review this experiment.\n\n{ROWS_ARE_LOADED}"
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
        pai_agent.output_validator(_cite_the_toolbox(calls))
        started = time.perf_counter()
        try:
            result = pai_agent.run_sync(
                question,
                usage_limits=UsageLimits(request_limit=self.max_turns),
            )
        except UsageLimitExceeded as exc:
            raise OverBudget(str(exc)) from exc
        except Exception as exc:
            return _failed_trajectory("review", calls, spec=spec, started=started, exc=exc)
        latency_s = time.perf_counter() - started

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
        traj = Trajectory(
            calls=calls,
            answer=answer,
            done=True,
            verdict=verdict,
            spec=spec,
            usage=_read_usage(result),
            latency_s=latency_s,
            error=_verdict_error(answer, verdict),
        )
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
        started = time.perf_counter()
        try:
            result = pai_agent.run_sync(
                brief, usage_limits=UsageLimits(request_limit=self.max_turns)
            )
        except UsageLimitExceeded as exc:
            raise OverBudget(str(exc)) from exc
        except Exception as exc:
            return _failed_trajectory("design", calls, started=started, exc=exc)
        latency_s = time.perf_counter() - started
        usage = _read_usage(result)

        if isinstance(result.output, DesignSpec):
            traj = Trajectory(
                calls=calls, spec=result.output, done=True, usage=usage, latency_s=latency_s
            )
        else:
            traj = Trajectory(
                calls=calls,
                answer=str(result.output),
                done=True,
                usage=usage,
                latency_s=latency_s,
            )
        _log_trajectory("design", traj)
        return traj
