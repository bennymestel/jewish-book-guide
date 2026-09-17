"""
Run the eval suite against several models and print a comparison table.

Usage:
    python -m evals.model_sweep --model MODEL_ID [--model MODEL_ID ...]
        [--mode simple|multi] [-k SUBSTRING] [--markdown]

Local only — costs OpenRouter credit (and Gemini quota if gemini-* is one of --model).
Refuses to run if a --model shares JUDGE_MODEL's family, since a judge grading its own
family would bias the comparison it's supposed to produce.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import statistics
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()  # must run before `import config` so JUDGE_MODEL/AGENT_MODEL see .env

from rich.console import Console
from rich.table import Table
from rich import box

import config
from evals.harness import build_eval_graph
from evals.run_evals import _evaluate_case, select_cases

logger = logging.getLogger(__name__)
console = Console()


def _family(model_id: str) -> str:
    return model_id.split("/")[0].lower() if "/" in model_id else "google"


def _gate_score(results: list[dict], cases: list[dict], gate: str, applies) -> str:
    """Fraction of cases where `gate` passed, counted only among cases `applies(case)` is true."""
    applicable = [(r, c) for r, c in zip(results, cases) if applies(c)]
    if not applicable:
        return "n/a"
    passed = sum(1 for r, _ in applicable if r.get(gate))
    return f"{passed}/{len(applicable)}"


async def _run_one_model(model: str, mode: str, cases: list[dict]) -> dict:
    graph = await build_eval_graph(mode, model=model)
    flatten_subagents = mode == "multi"
    results = []
    for case in cases:
        try:
            results.append(await _evaluate_case(case, graph, flatten_subagents))
        except Exception:
            logger.exception("[%s/%s] case errored", model, case["id"])
            results.append({"id": case["id"], "passed": False, "elapsed": 0.0})
    return {
        "model": model,
        "passed": sum(r["passed"] for r in results),
        "total": len(results),
        "tools": _gate_score(results, cases, "tools_ok", lambda c: c.get("required_tools")),
        "grounded": _gate_score(results, cases, "grounded_ok", lambda c: c.get("expect_grounded", True)),
        "quality": _gate_score(results, cases, "quality_ok", lambda c: c.get("judge")),
        "median_s": statistics.median(r["elapsed"] for r in results) if results else 0.0,
    }


async def run_sweep(models: list[str], mode: str, cases: list[dict]) -> list[dict]:
    judge_family = _family(config.JUDGE_MODEL)
    clashes = [m for m in models if _family(m) == judge_family]
    if clashes:
        console.print(
            f"[red]Refusing to run: {clashes} share the judge's family "
            f"({config.JUDGE_MODEL}). Use a held-out JUDGE_MODEL for a fair comparison.[/red]"
        )
        sys.exit(2)

    rows = []
    for model in models:
        logger.info("Running suite for model=%s", model)
        rows.append(await _run_one_model(model, mode, cases))
    return rows


def _print_table(rows: list[dict], mode: str) -> None:
    table = Table(title=f"Model comparison (mode={mode})", box=box.SIMPLE_HEAVY)
    table.add_column("Model", style="cyan")
    table.add_column("Pass", justify="center")
    table.add_column("Tools", justify="center")
    table.add_column("Grounded", justify="center")
    table.add_column("Quality", justify="center")
    table.add_column("Median time", justify="center")
    for row in rows:
        table.add_row(
            row["model"], f"{row['passed']}/{row['total']}", row["tools"],
            row["grounded"], row["quality"], f"{row['median_s']:.1f}s",
        )
    console.print(table)
    console.print(f"[dim]Judge: {config.JUDGE_MODEL} (held out — not among the models compared)[/dim]")


def _print_markdown(rows: list[dict], mode: str) -> None:
    print(f"\n**Model comparison** (mode={mode})\n")
    print("| Model | Pass | Tools | Grounded | Quality | Median time |")
    print("|---|---|---|---|---|---|")
    for row in rows:
        print(
            f"| {row['model']} | {row['passed']}/{row['total']} | {row['tools']} | "
            f"{row['grounded']} | {row['quality']} | {row['median_s']:.1f}s |"
        )
    print(f"\n_Judge: {config.JUDGE_MODEL} (held out — not among the models compared)_")


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True, metavar="MODEL_ID",
                         help="model to include in the sweep (repeatable)")
    parser.add_argument("--mode", choices=["simple", "multi"], default="simple")
    parser.add_argument("-k", dest="keyword", metavar="SUBSTRING",
                         help="run only cases whose id contains this substring")
    parser.add_argument("--markdown", action="store_true",
                         help="print a markdown table instead of a rich table")
    args = parser.parse_args()

    cases = select_cases(None, args.keyword)
    if not cases:
        console.print("[red]No cases matched the filter.[/red]")
        sys.exit(2)

    rows = asyncio.run(run_sweep(args.model, args.mode, cases))
    if args.markdown:
        _print_markdown(rows, args.mode)
    else:
        _print_table(rows, args.mode)


if __name__ == "__main__":
    main()
