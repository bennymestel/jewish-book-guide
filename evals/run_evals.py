"""
Eval runner for the Jewish Book Guide agent.

Usage:
    python -m evals.run_evals

Each case is checked deterministically:
  - Tool trajectory: at least one required tool was called
  - Grounding:       every book title in the reply exists in the local DB
  - Constraint:      no recommended title exceeds max_difficulty (if set)

Exit code 1 if any case fails.
"""
from __future__ import annotations

import asyncio
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.table import Table
from rich import box

from evals.harness import build_eval_graph, run_message
from evals.checks import (
    assert_tool_used,
    tools_called,
    extract_titles,
    assert_grounded,
    assert_difficulty_max,
)
from evals.cases import CASES

logger = logging.getLogger(__name__)
console = Console()


def _tick(passed: bool) -> str:
    return "[green]PASS[/green]" if passed else "[red]FAIL[/red]"


async def run_evals() -> bool:
    logger.info("Building agent graph")
    graph = await build_eval_graph()
    logger.info("Running %d cases", len(CASES))

    table = Table(box=box.SIMPLE_HEAVY, show_lines=True)
    table.add_column("Case", style="cyan", no_wrap=True)
    table.add_column("Tools", justify="center")
    table.add_column("Grounded", justify="center")
    table.add_column("Constraint", justify="center")
    table.add_column("Result", justify="center")

    all_passed = True
    passed_count = 0

    for case in CASES:
        logger.info("Running case: %s", case["id"])
        reply, messages = await run_message(graph, case["input"])

        # Tool trajectory check
        if case.get("required_tools"):
            tools_ok = assert_tool_used(messages, case["required_tools"])
            if not tools_ok:
                called = {tc["name"] for tc in tools_called(messages)}
                logger.warning(
                    "[%s] tool check failed: called=%s, required one of %s",
                    case["id"], called, case["required_tools"],
                )
        else:
            tools_ok = True

        # Grounding check
        if case.get("expect_grounded", True):
            titles = extract_titles(reply)
            grounded_ok, unresolved = assert_grounded(titles)
            if not grounded_ok:
                logger.warning("[%s] grounding failed: unresolved titles=%s", case["id"], unresolved)
        else:
            grounded_ok = True

        # Difficulty constraint check
        if case.get("max_difficulty") is not None:
            titles = extract_titles(reply)
            constraint_ok, violations = assert_difficulty_max(titles, case["max_difficulty"])
            if violations:
                logger.warning("[%s] difficulty constraint failed: %s", case["id"], violations)
        else:
            constraint_ok = True

        passed = tools_ok and grounded_ok and constraint_ok
        if passed:
            passed_count += 1
        else:
            all_passed = False

        tools_cell = _tick(tools_ok) if case.get("required_tools") else "[dim]n/a[/dim]"
        grounded_cell = _tick(grounded_ok) if case.get("expect_grounded", True) else "[dim]skipped[/dim]"
        constraint_cell = _tick(constraint_ok) if case.get("max_difficulty") is not None else "[dim]n/a[/dim]"

        table.add_row(case["id"], tools_cell, grounded_cell, constraint_cell, _tick(passed))

    console.print(table)
    total = len(CASES)
    color = "green" if all_passed else "red"
    console.print(f"[bold {color}]{passed_count}/{total} cases passed[/bold {color}]")
    return all_passed


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    all_passed = asyncio.run(run_evals())
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
