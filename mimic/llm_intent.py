"""OPTIONAL: let Claude phrase / sanity-check the learned intent in plain language.

The symbolic engine in intent.py is what drives the robot. This just adds a human-readable
"why" and can flag if the inferred goal looks odd. Needs: pip install anthropic, and ANTHROPIC_API_KEY set.
"""
import json
import os

from .models import Goal


def explain_with_llm(goal: Goal, model: str = None) -> str:
    import anthropic
    client = anthropic.Anthropic()
    summary = json.dumps(goal.to_dict(), indent=1)
    msg = client.messages.create(
        model=model or os.getenv("MIMIC_LLM_MODEL", "claude-sonnet-5-5"),
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": "A robot watched a human rearrange objects once and inferred this goal as object relations:\n"
                       f"{summary}\n\nIn 2 sentences, state the human's most likely intention in plain language, "
                       "then name one alternative interpretation the robot should double-check.",
        }],
    )
    return "".join(b.text for b in msg.content if b.type == "text")
