"""
LangSmith evaluation path for the Jewish Book Guide agent.

Usage:
    python -m evals.langsmith_eval [--mode simple|multi]

Syncs CASES to a LangSmith dataset and runs a named experiment with five evaluators
(tools, args, grounded, difficulty, quality) that reuse the same logic as the local
run_evals.py runner. The agent runs exactly once per case — no extra Gemini cost.

Requires LANGCHAIN_API_KEY. For an offline run with no LangSmith account:
    python -m evals.run_evals
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from langsmith import Client, aevaluate
import langsmith.utils

from evals.harness import build_eval_graph, run_message, run_conversation
from evals.checks import (
    tools_called,
    extract_titles,
    extract_tool_context,
    assert_tool_used,
    assert_tool_args,
    assert_grounded,
    assert_difficulty_max,
)
from evals.judge import judge_reply
from evals.cases import CASES

DATASET_NAME = "jewish-book-guide-agent"
EXPERIMENT_PREFIX = "jewish-book-guide-agent"

logger = logging.getLogger(__name__)

# CASES keyed by id so evaluators can look up non-serializable lambdas (tool_arg_check).
_cases_by_id: dict[str, dict] = {c["id"]: c for c in CASES}

# Module-level graph + mode — built once before aevaluate kicks off.
_graph = None
_flatten_subagents = False


# ── Dataset sync ──────────────────────────────────────────────────────────────

def _sync_dataset(client: Client) -> None:
    """Create (or recreate) the LangSmith dataset from CASES."""
    try:
        dataset = client.read_dataset(dataset_name=DATASET_NAME)
        logger.info("Found existing dataset '%s' — clearing examples for resync", DATASET_NAME)
        example_ids = [ex.id for ex in client.list_examples(dataset_id=dataset.id)]
        if example_ids:
            client.delete_examples(example_ids)
    except langsmith.utils.LangSmithNotFoundError:
        dataset = client.create_dataset(
            DATASET_NAME,
            description="End-to-end agent eval cases for the Jewish Book Guide agent.",
        )
        logger.info("Created dataset '%s'", DATASET_NAME)

    examples = []
    for case in CASES:
        inputs = {"inputs": case["inputs"]} if "inputs" in case else {"input": case["input"]}
        # Store case metadata as reference_outputs so evaluators receive it via the
        # standard (outputs, reference_outputs) evaluator signature.
        # tool_arg_check is a lambda — not JSON-serialisable; evaluators resolve it
        # from _cases_by_id using the case id.
        reference_outputs = {
            "id": case["id"],
            "required_tools": list(case["required_tools"]) if case.get("required_tools") else None,
            "max_difficulty": case.get("max_difficulty"),
            "expect_grounded": case.get("expect_grounded", True),
            "min_titles": case.get("min_titles", 0),
            "has_tool_arg_check": bool(case.get("tool_arg_check")),
            "judge": case.get("judge"),
        }
        examples.append({"inputs": inputs, "outputs": reference_outputs})

    client.create_examples(dataset_id=dataset.id, examples=examples)
    logger.info("Synced %d examples to dataset '%s'", len(examples), DATASET_NAME)


# ── Target function ───────────────────────────────────────────────────────────

async def _target(inputs: dict) -> dict:
    """
    Run one eval case through the agent graph. Pre-extracts tool calls, titles,
    and tool context so evaluators receive JSON-serialisable outputs.
    """
    if "inputs" in inputs:
        reply, messages = await run_conversation(
            _graph, inputs["inputs"], flatten_subagents=_flatten_subagents
        )
    else:
        reply, messages = await run_message(
            _graph, inputs["input"], flatten_subagents=_flatten_subagents
        )

    return {
        "reply": reply,
        "tool_calls": tools_called(messages),   # [{name, args}, ...]
        "titles": list(extract_titles(reply)),   # candidate book titles parsed from reply
        "tool_context": extract_tool_context(messages),  # concatenated ToolMessage outputs
    }


# ── Evaluators ────────────────────────────────────────────────────────────────
# Each evaluator returns {"key": str, "score": 0|1, "comment": str}.
# Signature: (outputs: dict, reference_outputs: dict) — LangSmith matches by name.

def tools_evaluator(outputs: dict, reference_outputs: dict) -> dict:
    required = reference_outputs.get("required_tools")
    if not required:
        return {"key": "tools", "score": 1, "comment": "n/a"}
    called_names = {tc["name"] for tc in outputs.get("tool_calls", [])}
    passed = bool(called_names & set(required))
    return {
        "key": "tools",
        "score": int(passed),
        "comment": f"called={sorted(called_names)}, required one of {sorted(required)}",
    }


def args_evaluator(outputs: dict, reference_outputs: dict) -> dict:
    if not reference_outputs.get("has_tool_arg_check"):
        return {"key": "args", "score": 1, "comment": "n/a"}
    case = _cases_by_id.get(reference_outputs.get("id", ""), {})
    predicate = case.get("tool_arg_check")
    if not predicate:
        return {"key": "args", "score": 1, "comment": "n/a (predicate not found)"}
    passed, reason = predicate(outputs.get("tool_calls", []))
    return {"key": "args", "score": int(passed), "comment": reason}


def grounded_evaluator(outputs: dict, reference_outputs: dict) -> dict:
    if not reference_outputs.get("expect_grounded", True):
        return {"key": "grounded", "score": 1, "comment": "skipped"}
    titles = set(outputs.get("titles", []))
    min_titles = reference_outputs.get("min_titles", 0)
    if len(titles) < min_titles:
        return {
            "key": "grounded",
            "score": 0,
            "comment": f"expected >={min_titles} titles, extracted {len(titles)}",
        }
    passed, unresolved = assert_grounded(titles)
    comment = f"unresolved={unresolved}" if not passed else f"all {len(titles)} titles resolved"
    return {"key": "grounded", "score": int(passed), "comment": comment}


def difficulty_evaluator(outputs: dict, reference_outputs: dict) -> dict:
    max_diff = reference_outputs.get("max_difficulty")
    if max_diff is None:
        return {"key": "difficulty", "score": 1, "comment": "n/a"}
    titles = set(outputs.get("titles", []))
    passed, violations = assert_difficulty_max(titles, max_diff)
    comment = f"violations={violations}" if not passed else f"all titles ≤ difficulty {max_diff}"
    return {"key": "difficulty", "score": int(passed), "comment": comment}


async def quality_evaluator(outputs: dict, reference_outputs: dict) -> dict:
    rubric = reference_outputs.get("judge")
    if not rubric:
        return {"key": "quality", "score": 1, "comment": "n/a"}
    case = _cases_by_id.get(reference_outputs.get("id", ""), {})
    user_input = " → ".join(case["inputs"]) if "inputs" in case else case.get("input", "")
    passed, reason = await judge_reply(
        input=user_input,
        reply=outputs.get("reply", ""),
        rubric=rubric,
        context=outputs.get("tool_context", ""),
    )
    return {"key": "quality", "score": int(passed), "comment": reason}


# ── Runner ────────────────────────────────────────────────────────────────────

async def run_langsmith_evals(mode: str = "simple") -> None:
    if not os.environ.get("LANGCHAIN_API_KEY"):
        print(
            "LANGCHAIN_API_KEY is not set — cannot run LangSmith evaluation.\n"
            "Add it to your .env file (see .env.example).\n\n"
            "To run evals locally without LangSmith:  python -m evals.run_evals"
        )
        sys.exit(1)

    client = Client()
    _sync_dataset(client)

    global _graph, _flatten_subagents
    logger.info("Building agent graph (mode=%s)", mode)
    _graph = await build_eval_graph(mode)
    _flatten_subagents = mode == "multi"

    logger.info(
        "Starting LangSmith experiment (max_concurrency=0 — sequential, "
        "num_repetitions=1 — Gemini-quota-safe)"
    )
    results = await aevaluate(
        _target,
        data=DATASET_NAME,
        evaluators=[
            tools_evaluator,
            args_evaluator,
            grounded_evaluator,
            difficulty_evaluator,
            quality_evaluator,
        ],
        experiment_prefix=f"{EXPERIMENT_PREFIX}-{mode}",
        description=(
            "End-to-end agent evals: tool trajectory, tool args, grounding, "
            f"difficulty constraints, LLM-as-judge quality. (mode={mode})"
        ),
        max_concurrency=0,   # sequential — protects Gemini free-tier quota
        num_repetitions=1,
        client=client,
        blocking=True,
    )

    print(f"\nExperiment complete: {results.experiment_name}")
    print("View results at https://smith.langchain.com")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["simple", "multi"], default="simple")
    args = parser.parse_args()
    asyncio.run(run_langsmith_evals(args.mode))


if __name__ == "__main__":
    main()
