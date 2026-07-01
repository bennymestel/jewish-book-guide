"""
Eval runner for the Jewish Book Guide agent.

Usage:
    python -m evals.run_evals [--mode simple|multi|both]

Each case is checked with up to five passes:
  - Tools:      at least one required tool was called
  - Args:       tool argument predicate (e.g. both seeds present)
  - Grounded:   every book title in the reply exists in the local DB; also enforces
                min_titles (if set) to guard against vacuous passes on empty replies
  - Constraint: no recommended title exceeds max_difficulty (if set)
  - Quality:    LLM-as-judge for scope / faithfulness / responsiveness (if rubric set)

Exit code 1 if any case fails.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.table import Table
from rich import box

from evals.harness import build_eval_graph, run_message, run_conversation
from evals.checks import (
    assert_tool_used,
    assert_tool_args,
    tools_called,
    extract_titles,
    assert_grounded,
    assert_difficulty_max,
    extract_tool_context,
)
from evals.judge import judge_reply
from evals.cases import CASES

logger = logging.getLogger(__name__)
console = Console()


def _tick(passed: bool) -> str:
    return "[green]PASS[/green]" if passed else "[red]FAIL[/red]"


async def run_evals(mode: str = "simple") -> tuple[bool, int, int]:
    """Run the full case suite against one graph mode ("simple" or "multi").
    Returns (all_passed, passed_count, total)."""
    logger.info("Building agent graph (mode=%s)", mode)
    graph = await build_eval_graph(mode)
    flatten_subagents = mode == "multi"
    logger.info("Running %d cases", len(CASES))

    table = Table(title=f"mode={mode}", box=box.SIMPLE_HEAVY, show_lines=True)
    table.add_column("Case", style="cyan", no_wrap=True)
    table.add_column("Tools", justify="center")
    table.add_column("Args", justify="center")
    table.add_column("Grounded", justify="center")
    table.add_column("Constraint", justify="center")
    table.add_column("Quality", justify="center")
    table.add_column("Result", justify="center")

    all_passed = True
    passed_count = 0

    for case in CASES:
        logger.info("Running case: %s", case["id"])

        # Support both single-turn ("input") and multi-turn ("inputs") cases.
        if "inputs" in case:
            reply, messages = await run_conversation(
                graph, case["inputs"], flatten_subagents=flatten_subagents
            )
            user_input = " → ".join(case["inputs"])
        else:
            reply, messages = await run_message(
                graph, case["input"], flatten_subagents=flatten_subagents
            )
            user_input = case["input"]

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

        # Tool argument check
        if case.get("tool_arg_check"):
            args_ok, args_reason = assert_tool_args(messages, case["tool_arg_check"])
            if not args_ok:
                logger.warning("[%s] tool arg check failed: %s", case["id"], args_reason)
        else:
            args_ok = True
            args_reason = ""

        # Grounding check
        if case.get("expect_grounded", True):
            titles = extract_titles(reply)
            grounded_ok, unresolved = assert_grounded(titles)
            min_titles = case.get("min_titles", 0)
            if len(titles) < min_titles:
                grounded_ok = False
                logger.warning(
                    "[%s] grounding failed: expected >=%d titles, extracted %d",
                    case["id"], min_titles, len(titles),
                )
            elif not grounded_ok:
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

        # LLM-as-judge quality check
        if case.get("judge"):
            tool_context = extract_tool_context(messages)
            quality_ok, quality_reason = await judge_reply(
                input=user_input,
                reply=reply,
                rubric=case["judge"],
                context=tool_context,
            )
            if not quality_ok:
                logger.warning("[%s] quality check failed: %s", case["id"], quality_reason)
            else:
                logger.info("[%s] quality check passed: %s", case["id"], quality_reason)
        else:
            quality_ok = True
            quality_reason = ""

        passed = tools_ok and args_ok and grounded_ok and constraint_ok and quality_ok
        if passed:
            passed_count += 1
        else:
            all_passed = False

        tools_cell = _tick(tools_ok) if case.get("required_tools") else "[dim]n/a[/dim]"
        args_cell = _tick(args_ok) if case.get("tool_arg_check") else "[dim]n/a[/dim]"
        grounded_cell = _tick(grounded_ok) if case.get("expect_grounded", True) else "[dim]skipped[/dim]"
        constraint_cell = _tick(constraint_ok) if case.get("max_difficulty") is not None else "[dim]n/a[/dim]"
        quality_cell = _tick(quality_ok) if case.get("judge") else "[dim]n/a[/dim]"

        table.add_row(
            case["id"], tools_cell, args_cell, grounded_cell, constraint_cell, quality_cell, _tick(passed)
        )

    console.print(table)
    total = len(CASES)
    color = "green" if all_passed else "red"
    console.print(f"[bold {color}]{passed_count}/{total} cases passed[/bold {color}]")
    return all_passed, passed_count, total


async def run_evals_both() -> bool:
    """Run the suite against both graphs and print a side-by-side comparison."""
    simple_passed, simple_count, total = await run_evals("simple")
    multi_passed, multi_count, _ = await run_evals("multi")
    console.print(
        f"\n[bold]simple: {simple_count}/{total} passed   "
        f"multi: {multi_count}/{total} passed[/bold]"
    )
    return simple_passed and multi_passed


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["simple", "multi", "both"], default="simple")
    args = parser.parse_args()

    if args.mode == "both":
        all_passed = asyncio.run(run_evals_both())
    else:
        all_passed, _, _ = asyncio.run(run_evals(args.mode))
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
