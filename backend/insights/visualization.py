"""
Visualization Module for Analytics and Insights.

Supports both static export (via Matplotlib) for notebooks and reports,
and Plotly-compatible JSON structures for interactive dashboard rendering
in the React frontend.
"""

from typing import Dict, Any, List, Optional, Union
import io
import base64
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt

# Professional color palette
PALETTE = {
    "primary": "#2563EB",     # Royal Blue
    "secondary": "#7C3AED",   # Purple
    "accent": "#059669",      # Emerald
    "danger": "#DC2626",      # Red
    "warning": "#D97706",     # Amber
    "text": "#1F2937",        # Slate Gray
    "grid": "#E5E7EB",        # Light Gray
    "background": "#F9FAFB"   # Off-white
}


class VisualizationEngine:
    """
    Engine for generating charts, KPI card representations, and Plotly-compatible JSON.
    """

    def __init__(self):
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # ==========================================
    # Matplotlib Static Rendering
    # ==========================================

    def plot_time_series(
        self,
        df: pd.DataFrame,
        date_col: str,
        metric_col: str,
        title: str = "Time Series Analysis",
        save_path: Optional[str] = None
    ) -> matplotlib.figure.Figure:
        """
        Generate a publication-grade time-series line chart.
        """
        clean_df = df[[date_col, metric_col]].dropna().copy()
        clean_df[date_col] = pd.to_datetime(clean_df[date_col])
        clean_df = clean_df.sort_values(by=date_col)

        fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
        ax.plot(
            clean_df[date_col],
            clean_df[metric_col],
            color=PALETTE["primary"],
            linewidth=2.5,
            marker="o",
            markersize=5,
            label=metric_col.title()
        )

        ax.set_title(title, fontsize=14, fontweight="bold", pad=15, color=PALETTE["text"])
        ax.set_xlabel(date_col.title(), fontsize=11, labelpad=10, color=PALETTE["text"])
        ax.set_ylabel(metric_col.title(), fontsize=11, labelpad=10, color=PALETTE["text"])
        ax.grid(True, linestyle="--", alpha=0.6, color=PALETTE["grid"])
        ax.tick_params(colors=PALETTE["text"])

        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")

        return fig

    def plot_category_bar(
        self,
        df: pd.DataFrame,
        category_col: str,
        metric_col: str,
        title: str = "Category Comparison",
        save_path: Optional[str] = None,
        top_n: int = 10
    ) -> matplotlib.figure.Figure:
        """
        Generate a category or regional comparison bar chart.
        """
        agg_df = (
            df.groupby(category_col)[metric_col]
            .sum()
            .reset_index()
            .sort_values(by=metric_col, ascending=True)
            .tail(top_n)
        )

        fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
        bars = ax.barh(
            agg_df[category_col].astype(str),
            agg_df[metric_col],
            color=PALETTE["primary"],
            edgecolor="none",
            height=0.65
        )

        # Highlight maximum value
        if len(bars) > 0:
            bars[-1].set_color(PALETTE["secondary"])

        ax.set_title(title, fontsize=14, fontweight="bold", pad=15, color=PALETTE["text"])
        ax.set_xlabel(metric_col.title(), fontsize=11, labelpad=10, color=PALETTE["text"])
        ax.grid(True, axis="x", linestyle="--", alpha=0.6, color=PALETTE["grid"])
        ax.tick_params(colors=PALETTE["text"])

        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.annotate(
                f"{width:,.0f}",
                xy=(width, bar.get_y() + bar.get_height() / 2),
                xytext=(5, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=9,
                color=PALETTE["text"]
            )

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")

        return fig

    def plot_distribution(
        self,
        df: pd.DataFrame,
        metric_col: str,
        title: str = "Metric Distribution",
        save_path: Optional[str] = None,
        bins: int = 20
    ) -> matplotlib.figure.Figure:
        """
        Generate a distribution histogram.
        """
        clean_vals = df[metric_col].dropna().astype(float)

        fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
        ax.hist(
            clean_vals,
            bins=bins,
            color=PALETTE["primary"],
            edgecolor="white",
            alpha=0.85
        )

        mean_val = float(clean_vals.mean())
        median_val = float(clean_vals.median())

        ax.axvline(mean_val, color=PALETTE["danger"], linestyle="--", linewidth=1.8, label=f"Mean: {mean_val:,.1f}")
        ax.axvline(median_val, color=PALETTE["accent"], linestyle=":", linewidth=2, label=f"Median: {median_val:,.1f}")

        ax.set_title(title, fontsize=14, fontweight="bold", pad=15, color=PALETTE["text"])
        ax.set_xlabel(metric_col.title(), fontsize=11, labelpad=10, color=PALETTE["text"])
        ax.set_ylabel("Frequency", fontsize=11, labelpad=10, color=PALETTE["text"])
        ax.grid(True, linestyle="--", alpha=0.6, color=PALETTE["grid"])
        ax.legend(frameon=True)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")

        return fig

    @staticmethod
    def figure_to_base64(fig: matplotlib.figure.Figure) -> str:
        """Convert a Matplotlib figure to a base64 encoded PNG string."""
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)
        return encoded

    # ==========================================
    # Plotly-Compatible JSON Generation
    # ==========================================

    def create_plotly_line_chart(
        self,
        df: pd.DataFrame,
        date_col: str,
        metric_col: str,
        title: str = "Time Series Line Chart"
    ) -> Dict[str, Any]:
        """
        Produce a Plotly-compatible JSON spec for interactive React frontend line charts.
        """
        clean_df = df[[date_col, metric_col]].dropna().copy()
        clean_df[date_col] = pd.to_datetime(clean_df[date_col]).dt.strftime("%Y-%m-%d")
        clean_df = clean_df.sort_values(by=date_col)

        return {
            "data": [
                {
                    "x": clean_df[date_col].tolist(),
                    "y": clean_df[metric_col].tolist(),
                    "type": "scatter",
                    "mode": "lines+markers",
                    "name": metric_col.title(),
                    "line": {"color": PALETTE["primary"], "width": 2.5},
                    "marker": {"size": 6, "color": PALETTE["primary"]}
                }
            ],
            "layout": {
                "title": {"text": title},
                "xaxis": {"title": date_col.title()},
                "yaxis": {"title": metric_col.title()},
                "autosize": True,
                "hovermode": "closest",
                "template": "plotly_white"
            }
        }

    def create_plotly_bar_chart(
        self,
        df: pd.DataFrame,
        category_col: str,
        metric_col: str,
        title: str = "Category Comparison Bar Chart",
        top_n: int = 10
    ) -> Dict[str, Any]:
        """
        Produce a Plotly-compatible JSON spec for interactive React frontend bar charts.
        """
        agg_df = (
            df.groupby(category_col)[metric_col]
            .sum()
            .reset_index()
            .sort_values(by=metric_col, ascending=False)
            .head(top_n)
        )

        return {
            "data": [
                {
                    "x": agg_df[category_col].astype(str).tolist(),
                    "y": agg_df[metric_col].tolist(),
                    "type": "bar",
                    "name": metric_col.title(),
                    "marker": {"color": PALETTE["primary"]}
                }
            ],
            "layout": {
                "title": {"text": title},
                "xaxis": {"title": category_col.title()},
                "yaxis": {"title": metric_col.title()},
                "autosize": True,
                "template": "plotly_white"
            }
        }

    # ==========================================
    # KPI Card Metadata
    # ==========================================

    @staticmethod
    def generate_kpi_cards(metrics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format metrics into frontend-friendly KPI cards.
        """
        cards = []
        for m in metrics:
            val = m.get("value", 0)
            unit = m.get("unit", "")
            formatted_val = f"{val:,.2f}" if isinstance(val, float) else f"{val:,}"

            cards.append({
                "title": m.get("name"),
                "value": val,
                "formatted_value": f"{formatted_val} {unit}".strip(),
                "unit": unit,
                "description": m.get("description", "")
            })
        return cards
