"""
The sandbox boundary.

The agent writes code, so this is the only thing between a hallucinated
shutil.rmtree and my laptop. Tested adversarially: half these tests try to
break out.

sandbox.run(code, data) returns a Result with ok, stdout, error and artifacts.
It never raises for anything the code does; a crash inside the container is a
result, not an exception.
"""

from __future__ import annotations

import time

import pytest

pytestmark = pytest.mark.docker


# --------------------------------------------------------------------------- #
# It works
# --------------------------------------------------------------------------- #


def test_it_runs_ordinary_analysis_code(sandbox, clean):
    result = sandbox.run(
        "import pandas as pd; print(len(pd.read_parquet('/data/exp.parquet')))",
        clean.df,
    )
    assert result.ok
    assert result.stdout.strip().isdigit()


def test_the_scientific_stack_is_there(sandbox, clean):
    result = sandbox.run(
        "import numpy, pandas, scipy, statsmodels, matplotlib; print('ok')", clean.df
    )
    assert result.ok


def test_plots_come_back_as_artifacts(sandbox, clean):
    result = sandbox.run(
        "import matplotlib; matplotlib.use('Agg')\n"
        "import matplotlib.pyplot as plt\n"
        "plt.plot([1, 2, 3]); plt.savefig('/out/trend.png')",
        clean.df,
    )
    assert result.ok
    assert "trend.png" in result.artifacts


def test_a_syntax_error_is_a_result_not_an_exception(sandbox, clean):
    result = sandbox.run("this is not python", clean.df)
    assert not result.ok
    assert "SyntaxError" in result.error


# --------------------------------------------------------------------------- #
# It contains
# --------------------------------------------------------------------------- #


def test_the_network_is_unreachable(sandbox, clean):
    result = sandbox.run(
        "import urllib.request; urllib.request.urlopen('https://example.com', timeout=5)",
        clean.df,
    )
    assert not result.ok


def test_the_dataset_is_read_only(sandbox, clean):
    result = sandbox.run("open('/data/exp.parquet', 'w').write('x')", clean.df)
    assert not result.ok
    assert "read-only" in result.error.lower() or "permission" in result.error.lower()


def test_my_files_are_not_in_there(sandbox, clean):
    result = sandbox.run("import os; print(os.listdir('/'))", clean.df)
    assert "abcode" not in result.stdout


def test_an_infinite_loop_is_killed_on_time(sandbox, clean):
    started = time.monotonic()
    result = sandbox.run("while True: pass", clean.df)
    assert not result.ok and "timeout" in result.error.lower()
    assert time.monotonic() - started < sandbox.timeout + 5


def test_a_memory_bomb_is_killed(sandbox, clean):
    assert not sandbox.run("x = bytearray(4 * 1024**3)", clean.df).ok


def test_it_does_not_run_as_root(sandbox, clean):
    assert sandbox.run("import os; print(os.getuid())", clean.df).stdout.strip() != "0"


def test_nothing_survives_between_runs(sandbox, clean):
    sandbox.run("secret = 1234", clean.df)
    result = sandbox.run("print(secret)", clean.df)
    assert not result.ok and "NameError" in result.error


def test_a_flood_of_output_is_truncated(sandbox, clean):
    """Otherwise one print() eats the whole context window."""
    result = sandbox.run("print('x' * 10_000_000)", clean.df)
    assert len(result.stdout) < 200_000
    assert "truncated" in result.stdout.lower()
