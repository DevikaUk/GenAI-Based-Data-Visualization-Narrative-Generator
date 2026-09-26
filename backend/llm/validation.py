"""
Numerical Grounding and Validation Module for LLM Narratives.

This module provides post-generation verification to guard against LLM hallucinations. 
It extracts numerical data from both the generated narrative text and the ground-truth 
insights dictionary, verifying that every number mentioned by the model strictly originates 
from the source data.
"""

import re


def extract_numbers(text: str):
    """
    Extract numerical values from generated text.
    """

    numbers = re.findall(
        r'[-+]?\d*\.?\d+(?:,\d{3})*(?:%|)',
        text
    )

    return [
        number.replace(",", "")
        for number in numbers
    ]


def extract_source_numbers(insights: dict):
    """
    Recursively extract numerical values from
    the structured insights.
    """

    numbers = []

    def collect(value):

        if isinstance(value, dict):

            for item in value.values():
                collect(item)

        elif isinstance(value, list):

            for item in value:
                collect(item)

        elif isinstance(value, (int, float)):

            numbers.append(str(value))

    collect(insights)

    return numbers


def validate_narrative(
    narrative: str,
    insights: dict
):
    """
    Check whether generated numerical values
    are supported by the source insights.
    """

    source_numbers = extract_source_numbers(
        insights
    )

    generated_numbers = extract_numbers(
        narrative
    )

    unsupported_numbers = []

    for number in generated_numbers:

        clean_number = number.replace("%", "")

        if clean_number not in source_numbers:
            unsupported_numbers.append(number)

    return {
        "is_valid": len(unsupported_numbers) == 0,
        "generated_numbers": generated_numbers,
        "unsupported_numbers": unsupported_numbers
    }