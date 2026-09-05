"""
Fixtures for the suite.

Everything here talks to peer_agent through its public functions. If I rewrite the
internals, these files should not move.

The map of the codebase:

    types.py    Verdict, DesignSpec, and the frozen result dataclasses
    sim.py      make_case - simulated experiments with known truth
    stats.py    the six analyses, plain functions over a dataframe
    design.py   power_analysis and simulate_design
    sandbox.py  Sandbox - runs the agent's code in a container
    tools.py    the registry that hands stats.py to the model
    agent.py    the loop
"""

from __future__ import annotations

import pytest
from peer_agent.agent import Agent
from peer_agent.sandbox import Sandbox
from peer_agent.sim import NOVELTY, SIMPSON, SRM, make_case
from peer_agent.tools import default_tools

# --------------------------------------------------------------------------- #
# Fake model
# --------------------------------------------------------------------------- #


class FakeLLM:
    """
    Stands in for the model so the loop can be tested offline.

    Pass a list of turns to replay them in order, or a function to decide each
    turn from the messages so far. A turn is either a list of (name, args)
    tool calls, or a string to finish.
    """

    def __init__(self, turns=None, react=None):
        self.turns = list(turns or [])
        self.react = react
        self.seen = []

    def __call__(self, messages, tools):
        self.seen.append(messages)
        if self.react:
            return self.react(messages)
        return self.turns.pop(0) if self.turns else "done"


# --------------------------------------------------------------------------- #
# Reading a trajectory
# --------------------------------------------------------------------------- #


def names(traj):
    return [c.name for c in traj.calls]


def called_before(traj, first, second):
    n = names(traj)
    return first in n and (second not in n or n.index(first) < n.index(second))


def args_for(traj, name):
    return [c.args for c in traj.calls if c.name == name]


def quotes_number(traj, value, tol=0.05):
    """Did the answer put this figure in front of the reader?"""
    import re

    found = (float(m) for m in re.findall(r"-?\d+\.?\d*", traj.answer))
    return any(abs(f - value) <= abs(value) * tol for f in found)


# --------------------------------------------------------------------------- #
# Cases
# --------------------------------------------------------------------------- #


@pytest.fixture
def clean():
    """Nothing wrong, nothing happened. The agent has to say so."""
    return make_case(n=40_000, lift=0.0, seed=1)


@pytest.fixture
def real_win():
    return make_case(n=80_000, lift=0.06, seed=2)


@pytest.fixture
def srm():
    """Broken split hiding a fake +2%."""
    return make_case(n=50_000, lift=0.0, flaws=[SRM], seed=3)


@pytest.fixture
def novelty():
    return make_case(n=60_000, lift=0.0, flaws=[NOVELTY], days=14, seed=4)


@pytest.fixture
def simpson():
    return make_case(n=60_000, lift=0.0, flaws=[SIMPSON], seed=5)


@pytest.fixture
def sandbox():
    with Sandbox(timeout=10, memory_mb=512) as sb:
        yield sb


@pytest.fixture
def agent(sandbox):
    def build(llm, **kw):
        return Agent(llm, tools=default_tools(sandbox), **kw)

    return build


@pytest.fixture
def sandbox_free_agent():
    """
    Phase 1 has no sandbox yet, so the loop is built with only the stats tools.
    Kept afterwards because most loop tests don't need a container.
    """

    def build(llm, **kw):
        return Agent(llm, tools=default_tools(), **kw)

    return build


@pytest.fixture
def live_agent():
    return Agent.from_env()


def pytest_configure(config):
    for marker in [
        "slow: hundreds of simulated runs, minutes not seconds",
        "live: needs a real model and network",
        "docker: needs a container runtime",
    ]:
        config.addinivalue_line("markers", marker)
