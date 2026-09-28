"""
Chart Summary Engine.

Converts underlying chart data and analytical metrics into factual, objective
metadata summaries suitable for consumption by downstream LLM prompt builders.
"""

from typing import Union, Optional
import pandas as pd
import numpy as np

from backend.schemas.insight_schema import VisualizationSummary


class ChartSummaryGenerator:
    """
    Generates structured factual summaries from chart data and analytical outcomes.
    """

    @staticmethod
    def summarize_line_chart(
        title: str,
        metric: str,
        dimension: str,
        direction: str,
        change_pct: float,
        start_val: Optional[Union[int, float]] = None,
        end_val: Optional[Union[int, float]] = None,
        peak_label: Optional[str] = None,
        peak_val: Optional[Union[int, float]] = None
    ) -> VisualizationSummary:
        """
        Generate summary for a time-series line chart.
        """
        if direction == "stable":
            summary = f"{metric} remained approximately stable across the reporting period with a minor change of {change_pct}%."
        else:
            summary = f"{metric} {direction} consistently during the reporting period, moving by {change_pct}%."

        if peak_label and peak_val is not None:
            summary += f" The metric reached its peak at {peak_label} ({peak_val})."

        return VisualizationSummary(
            title=title,
            type="line",
            metric=metric,
            dimension=dimension,
            summary=summary
        )

    @staticmethod
    def summarize_bar_chart(
        title: str,
        metric: str,
        dimension: str,
        highest_name: str,
        highest_val: Union[int, float],
        lowest_name: Optional[str] = None,
        lowest_val: Optional[Union[int, float]] = None
    ) -> VisualizationSummary:
        """
        Generate summary for a category or regional bar chart.
        """
        summary = f"{highest_name} generated the highest {metric.lower()} ({highest_val})."
        if lowest_name and lowest_val is not None:
            summary += f" In contrast, {lowest_name} recorded the lowest ({lowest_val})."

        return VisualizationSummary(
            title=title,
            type="bar",
            metric=metric,
            dimension=dimension,
            summary=summary
        )

    @staticmethod
    def summarize_distribution_chart(
        title: str,
        metric: str,
        dimension: str,
        mean_val: float,
        median_val: float,
        min_val: float,
        max_val: float
    ) -> VisualizationSummary:
        """
        Generate summary for a distribution/histogram chart.
        """
        summary = (
            f"{metric} distribution ranges from a minimum of {min_val} to a maximum of {max_val}, "
            f"with a median of {median_val} and an average of {mean_val}."
        )

        return VisualizationSummary(
            title=title,
            type="histogram",
            metric=metric,
            dimension=dimension,
            summary=summary
        )
