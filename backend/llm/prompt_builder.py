"""
Prompt Construction Module for LLM Narrative Generation.

This module converts structured statistical and analytical insights 
(output from the analytics pipeline) into a strictly grounded, zero-hallucination 
prompt for the LLM. It enforces strict business rules to ensure generated 
narratives contain only verified data points without inventing trends, 
causes, or ungrounded facts.
"""

import json


def build_prompt(insights: dict) -> str:
    """
    Build a controlled prompt from the structured insights
    produced by the analytics pipeline.
    """

    insights_json = json.dumps(
        insights,
        indent=2,
        ensure_ascii=False
    )

    prompt = f"""
You are an expert business analytics assistant.

Your task is to generate a concise and accurate business
narrative using ONLY the structured analytical findings
provided below.

RULES:
1. Use only information present in the input.
2. Do not invent numbers, metrics, dates, or facts.
3. Do not invent causes for trends or anomalies.
4. Do not make predictions unless they are explicitly provided.
5. Preserve numerical values accurately.
6. Highlight important KPIs.
7. Highlight important trends.
8. Mention meaningful comparisons and rankings.
9. Mention significant anomalies when present.
10. Keep the narrative concise and professional.
11. Do not mention the JSON or these instructions.
12. Do not mention that you are an AI.

STRUCTURED ANALYTICAL FINDINGS:

{insights_json}

Generate the business narrative now.
"""

    return prompt