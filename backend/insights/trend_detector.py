"""
Trend Detection Engine.

Analyzes time-series data to detect directional trends, compute growth/decline rates,
and assess analytical significance according to documented thresholds.
"""

from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np

from backend.schemas.insight_schema import Trend, TrendPeriod


class TrendDetector:
    """
    Detects and classifies trends in temporal business metrics.
    """

    def __init__(self, threshold_stable_pct: float = 3.0, threshold_high_pct: float = 10.0):
        """
        Args:
            threshold_stable_pct: Percentage change below which a trend is considered 'stable'.
            threshold_high_pct: Percentage change above which significance is considered 'high'.
        """
        self.threshold_stable_pct = threshold_stable_pct
        self.threshold_high_pct = threshold_high_pct

    def detect_trend(
        self,
        df: pd.DataFrame,
        metric: str,
        dimension: str = "date",
        freq: Optional[str] = None
    ) -> Optional[Trend]:
        """
        Detect trend for a single metric over a temporal dimension.

        Args:
            df: DataFrame containing the data.
            metric: Numeric column to evaluate.
            dimension: Temporal column (e.g., date).
            freq: Optional pandas resampling frequency (e.g., 'ME', 'W', 'D').

        Returns:
            Trend object or None if insufficient data.
        """
        if df.empty or metric not in df.columns or dimension not in df.columns:
            return None

        clean_df = df[[dimension, metric]].dropna().copy()
        if len(clean_df) < 2:
            return None

        clean_df[dimension] = pd.to_datetime(clean_df[dimension])
        clean_df = clean_df.sort_values(by=dimension)

        if freq:
            resampled = clean_df.set_index(dimension).resample(freq).sum().reset_index()
            clean_df = resampled.dropna()
            if len(clean_df) < 2:
                return None

        start_row = clean_df.iloc[0]
        end_row = clean_df.iloc[-1]

        start_val = float(start_row[metric])
        end_val = float(end_row[metric])

        start_date = start_row[dimension].strftime("%Y-%m-%d")
        end_date = end_row[dimension].strftime("%Y-%m-%d")

        if start_val == 0:
            change_pct = 0.0 if end_val == 0 else 100.0
        else:
            change_pct = ((end_val - start_val) / abs(start_val)) * 100.0

        change_pct_rounded = round(change_pct, 2)

        # Direction classification
        if change_pct_rounded > self.threshold_stable_pct:
            direction = "increasing"
        elif change_pct_rounded < -self.threshold_stable_pct:
            direction = "decreasing"
        else:
            direction = "stable"

        # Significance classification
        abs_change = abs(change_pct_rounded)
        if abs_change >= self.threshold_high_pct:
            significance = "high"
        elif abs_change >= self.threshold_stable_pct:
            significance = "medium"
        else:
            significance = "low"

        return Trend(
            metric=metric,
            dimension=dimension,
            direction=direction,
            change=change_pct_rounded,
            unit="percent",
            period=TrendPeriod(start=start_date, end=end_date),
            significance=significance
        )

    def detect_multiple_trends(
        self,
        df: pd.DataFrame,
        metrics: List[str],
        dimension: str = "date",
        freq: Optional[str] = None
    ) -> List[Trend]:
        """Detect trends for multiple metrics."""
        trends = []
        for m in metrics:
            trend = self.detect_trend(df, metric=m, dimension=dimension, freq=freq)
            if trend:
                trends.append(trend)
        return trends
