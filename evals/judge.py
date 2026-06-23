"""
LLM-as-judge for agent evals.

Grades scope, instruction-following, faithfulness, and responsiveness —
NOT factual correctness (that's what the deterministic checks in checks.py are for).
"""
from __future__ import annotations

import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

import config

logger = logging.getLogger(__name__)

_JUDGE_PROMPT = """\
You are a strict binary evaluator for an AI assistant.

USER REQUEST:
{input}

{context_block}ASSISTANT REPLY:
{reply}

RUBRIC:
{rubric}

Evaluate whether the assistant reply satisfies the rubric.
Reply with ONLY valid JSON in this exact format — nothing else:
{{"pass": true, "reason": "<one sentence>"}}
"""


def _build_judge_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=config.JUDGE_MODEL,
        google_api_key=os.environ["GOOGLE_API_KEY"],
        # temperature=0 for deterministic grading; same-model self-preference bias is
        # further reduced by using binary scope/faithfulness rubrics, not open-ended quality.
        temperature=0,
    )


async def judge_reply(
    input: str,
    reply: str,
    rubric: str,
    context: str = "",
) -> tuple[bool, str]:
    """
    Ask the judge LLM whether `reply` satisfies `rubric` for the given `input`.
    For faithfulness rubrics, pass `context` (the tool output) so the judge can
    check consistency against what the agent actually received.

    Returns (passed, reason).
    """
    context_block = f"CONTEXT (tool output the agent received):\n{context}\n\n" if context else ""
    prompt = _JUDGE_PROMPT.format(
        input=input,
        context_block=context_block,
        reply=reply,
        rubric=rubric,
    )

    llm = _build_judge_llm()
    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        content = response.content
        if isinstance(content, list):
            content = "".join(block["text"] for block in content if isinstance(block, dict) and "text" in block)
        raw = content.strip()
        # Strip markdown code fences if the model wrapped the JSON
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        result = json.loads(raw)
        passed = bool(result.get("pass", False))
        reason = str(result.get("reason", ""))
        return passed, reason
    except Exception as exc:
        logger.warning("[judge] failed to parse response: %s", exc)
        return False, f"judge error: {exc}"
