"""
Structured Insight Generation Pipeline (Primary Orchestrator).

Combines trend detection, anomaly detection, categorical comparisons,
distribution statistics, and chart summaries to generate the canonical
Structured Insights JSON contract required by the LLM prompt builder.
"""

from typing import List, Dict, Any, Optional, Union
import json
import pandas as pd
import numpy as np

from backend.schemas.insight_schema import (
    StructuredInsights,
    DatasetMetadata,
    ColumnMetadata,
    Metric,
    Trend,
    Comparison,
    ComparisonExtremum,
    Anomaly,
    Distribution,
    VisualizationSummary
)
from .trend_detector import TrendDetector
from .anomaly_detector import AnomalyDetector
from .chart_summary import ChartSummaryGenerator
from .visualization import VisualizationEngine


class InsightEngine:
    """
    Orchestrates analytical findings and visual metadata into the unified
    structured insights contract.
    """

    def __init__(
        self,
        trend_detector: Optional[TrendDetector] = None,
        anomaly_detector: Optional[AnomalyDetector] = None,
        visualization_engine: Optional[VisualizationEngine] = None
    ):
        self.trend_detector = trend_detector or TrendDetector()
        self.anomaly_detector = anomaly_detector or AnomalyDetector()
        self.visualization_engine = visualization_engine or VisualizationEngine()

    def generate_insights(
        self,
        df: pd.DataFrame,
        dataset_name: str = "dataset.csv",
        domain: str = "business",
        temporal_col: Optional[str] = "date",
        numeric_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
        identifier_cols: Optional[List[str]] = None,
        unit_currency: str = "INR",
        external_metrics: Optional[List[Metric]] = None
    ) -> StructuredInsights:
        """
        Process the dataset and produce fully-grounded structured insights.
        """
        # 1. Infer/clean column classifications
        if numeric_cols is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if categorical_cols is None:
            categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
            if temporal_col in categorical_cols:
                categorical_cols.remove(temporal_col)
        if identifier_cols is None:
            identifier_cols = [c for c in df.columns if "id" in c.lower()]

        dataset_meta = DatasetMetadata(
            name=dataset_name,
            domain=domain,
            record_count=len(df),
            columns=ColumnMetadata(
                numeric=numeric_cols,
                categorical=categorical_cols,
                temporal=[temporal_col] if temporal_col and temporal_col in df.columns else [],
                identifier=identifier_cols
            )
        )

        # 2. Extract or Compute Headline Metrics
        metrics: List[Metric] = []
        if external_metrics:
            metrics.extend(external_metrics)
        else:
            metrics = self._calculate_default_metrics(df, numeric_cols, unit_currency)

        # 3. Detect Trends
        trends: List[Trend] = []
        if temporal_col and temporal_col in df.columns:
            for num_col in numeric_cols[:2]:  # Focus on key metrics
                trend = self.trend_detector.detect_trend(df, metric=num_col, dimension=temporal_col)
                if trend:
                    trends.append(trend)

        # 4. Compute Categorical Comparisons
        comparisons: List[Comparison] = []
        primary_metric = numeric_cols[0] if numeric_cols else None
        if primary_metric:
            for cat_col in categorical_cols[:2]:
                comp = self._compute_comparison(df, category_col=cat_col, metric_col=primary_metric)
                if comp:
                    comparisons.append(comp)

        # 5. Detect Statistical Anomalies
        anomalies: List[Anomaly] = []
        if primary_metric and temporal_col and temporal_col in df.columns:
            anomalies = self.anomaly_detector.detect_anomalies(
                df, metric=primary_metric, dimension=temporal_col, max_anomalies=5
            )

        # 6. Compute Distributions
        distributions: List[Distribution] = []
        for num_col in numeric_cols[:2]:
            dist = self._compute_distribution(df, metric_col=num_col)
            if dist:
                distributions.append(dist)

        # 7. Generate Visualizations & Chart Summaries
        visualizations: List[VisualizationSummary] = []
        if temporal_col and temporal_col in df.columns and primary_metric:
            trend_item = trends[0] if trends else None
            direction = trend_item.direction if trend_item else "stable"
            change_pct = trend_item.change if trend_item else 0.0

            line_summary = ChartSummaryGenerator.summarize_line_chart(
                title=f"Monthly {primary_metric.title()}",
                metric=primary_metric.title(),
                dimension=temporal_col,
                direction=direction,
                change_pct=change_pct
            )
            visualizations.append(line_summary)

        if categorical_cols and primary_metric:
            cat_col = categorical_cols[0]
            comp_item = comparisons[0] if comparisons else None
            if comp_item:
                bar_summary = ChartSummaryGenerator.summarize_bar_chart(
                    title=f"{primary_metric.title()} by {cat_col.title()}",
                    metric=primary_metric.title(),
                    dimension=cat_col,
                    highest_name=comp_item.highest.name,
                    highest_val=comp_item.highest.value,
                    lowest_name=comp_item.lowest.name,
                    lowest_val=comp_item.lowest.value
                )
                visualizations.append(bar_summary)

        return StructuredInsights(
            dataset=dataset_meta,
            metrics=metrics,
            trends=trends,
            comparisons=comparisons,
            anomalies=anomalies,
            distributions=distributions,
            visualizations=visualizations
        )

    def _calculate_default_metrics(
        self,
        df: pd.DataFrame,
        numeric_cols: List[str],
        unit_currency: str
    ) -> List[Metric]:
        """Compute standard baseline business metrics."""
        metrics = []
        for col in numeric_cols[:3]:
            total_val = float(df[col].sum())
            metrics.append(
                Metric(
                    name=f"Total {col.title()}",
                    value=round(total_val, 2),
                    unit=unit_currency if "profit" in col.lower() or "sales" in col.lower() or "revenue" in col.lower() else "units",
                    description=f"Total {col} across all records"
                )
            )

        if len(numeric_cols) > 0:
            avg_val = float(df[numeric_cols[0]].mean())
            metrics.append(
                Metric(
                    name=f"Average {numeric_cols[0].title()}",
                    value=round(avg_val, 2),
                    unit=unit_currency,
                    description=f"Average {numeric_cols[0]} per record"
                )
            )

        return metrics

    def _compute_comparison(
        self,
        df: pd.DataFrame,
        category_col: str,
        metric_col: str
    ) -> Optional[Comparison]:
        """Compute top and bottom values for a category."""
        if df.empty or category_col not in df.columns or metric_col not in df.columns:
            return None

        agg = df.groupby(category_col)[metric_col].sum().reset_index()
        if agg.empty:
            return None

        agg = agg.sort_values(by=metric_col, ascending=False)
        highest_row = agg.iloc[0]
        lowest_row = agg.iloc[-1]

        return Comparison(
            dimension=category_col,
            metric=metric_col.title(),
            highest=ComparisonExtremum(
                name=str(highest_row[category_col]),
                value=round(float(highest_row[metric_col]), 2)
            ),
            lowest=ComparisonExtremum(
                name=str(lowest_row[category_col]),
                value=round(float(lowest_row[metric_col]), 2)
            )
        )

    def _compute_distribution(
        self,
        df: pd.DataFrame,
        metric_col: str
    ) -> Optional[Distribution]:
        """Compute distribution statistics for a numeric metric."""
        if df.empty or metric_col not in df.columns:
            return None

        vals = df[metric_col].dropna().astype(float)
        if len(vals) == 0:
            return None

        return Distribution(
            metric=metric_col.title(),
            mean=round(float(vals.mean()), 2),
            median=round(float(vals.median()), 2),
            minimum=round(float(vals.min()), 2),
            maximum=round(float(vals.max()), 2),
            std_dev=round(float(vals.std()), 2)
        )
