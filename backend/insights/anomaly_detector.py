"""
Statistical Anomaly Detection Engine.

Detects outliers and abnormal data spikes using documented statistical techniques
(Interquartile Range - IQR, and Z-Score method). Outputs structured anomaly findings
strictly grounded in the underlying data.
"""

from typing import List, Optional
import pandas as pd
import numpy as np

from backend.schemas.insight_schema import Anomaly, AnomalyExpectedRange


class AnomalyDetector:
    """
    Detects statistical anomalies across metrics and dimensions.
    """

    def __init__(self, method: str = "iqr", iqr_factor: float = 1.5, z_threshold: float = 2.5):
        """
        Args:
            method: 'iqr' or 'zscore'.
            iqr_factor: Multiplier for IQR (default 1.5 for Tukey's fences).
            z_threshold: Number of standard deviations from mean (default 2.5).
        """
        self.method = method.lower()
        self.iqr_factor = iqr_factor
        self.z_threshold = z_threshold

    def detect_anomalies(
        self,
        df: pd.DataFrame,
        metric: str,
        dimension: str = "date",
        max_anomalies: int = 5
    ) -> List[Anomaly]:
        """
        Detect anomalies for a given metric and dimension using configured method.
        """
        if self.method == "zscore":
            return self.detect_anomalies_zscore(df, metric, dimension, max_anomalies=max_anomalies)
        return self.detect_anomalies_iqr(df, metric, dimension, max_anomalies=max_anomalies)

    def detect_anomalies_iqr(
        self,
        df: pd.DataFrame,
        metric: str,
        dimension: str = "date",
        max_anomalies: int = 5
    ) -> List[Anomaly]:
        """
        Detect anomalies using Interquartile Range (IQR).
        """
        if df.empty or metric not in df.columns or dimension not in df.columns:
            return []

        clean_df = df[[dimension, metric]].dropna().copy()
        if len(clean_df) < 4:
            return []

        values = clean_df[metric].astype(float)
        q1 = float(np.percentile(values, 25))
        q3 = float(np.percentile(values, 75))
        iqr = q3 - q1

        if iqr == 0:
            return []

        lower_bound = round(q1 - (self.iqr_factor * iqr), 2)
        upper_bound = round(q3 + (self.iqr_factor * iqr), 2)

        anomalies: List[Anomaly] = []

        for _, row in clean_df.iterrows():
            val = float(row[metric])
            obs_raw = row[dimension]
            obs_str = (
                obs_raw.strftime("%Y-%m-%d")
                if isinstance(obs_raw, (pd.Timestamp, pd.DatetimeIndex))
                else str(obs_raw)
            )

            if val > upper_bound:
                excess_ratio = (val - upper_bound) / iqr
                severity = "high" if excess_ratio > 1.5 else ("medium" if excess_ratio > 0.5 else "low")
                anomalies.append(
                    Anomaly(
                        metric=metric,
                        dimension=dimension,
                        observation=obs_str,
                        value=round(val, 2),
                        expected_range=AnomalyExpectedRange(lower=lower_bound, upper=upper_bound),
                        type="high",
                        severity=severity
                    )
                )
            elif val < lower_bound:
                excess_ratio = (lower_bound - val) / iqr
                severity = "high" if excess_ratio > 1.5 else ("medium" if excess_ratio > 0.5 else "low")
                anomalies.append(
                    Anomaly(
                        metric=metric,
                        dimension=dimension,
                        observation=obs_str,
                        value=round(val, 2),
                        expected_range=AnomalyExpectedRange(lower=lower_bound, upper=upper_bound),
                        type="low",
                        severity=severity
                    )
                )

        # Sort anomalies by severity/distance from boundary
        anomalies.sort(
            key=lambda a: abs(a.value - (upper_bound if a.type == "high" else lower_bound)),
            reverse=True
        )

        return anomalies[:max_anomalies]

    def detect_anomalies_zscore(
        self,
        df: pd.DataFrame,
        metric: str,
        dimension: str = "date",
        max_anomalies: int = 5
    ) -> List[Anomaly]:
        """
        Detect anomalies using Z-score method.
        """
        if df.empty or metric not in df.columns or dimension not in df.columns:
            return []

        clean_df = df[[dimension, metric]].dropna().copy()
        if len(clean_df) < 4:
            return []

        values = clean_df[metric].astype(float)
        mean = float(np.mean(values))
        std = float(np.std(values))

        if std == 0:
            return []

        lower_bound = round(mean - (self.z_threshold * std), 2)
        upper_bound = round(mean + (self.z_threshold * std), 2)

        anomalies: List[Anomaly] = []

        for _, row in clean_df.iterrows():
            val = float(row[metric])
            obs_raw = row[dimension]
            obs_str = (
                obs_raw.strftime("%Y-%m-%d")
                if isinstance(obs_raw, (pd.Timestamp, pd.DatetimeIndex))
                else str(obs_raw)
            )

            z = (val - mean) / std

            if abs(z) > self.z_threshold:
                anom_type = "high" if z > 0 else "low"
                severity = "high" if abs(z) > (self.z_threshold + 1.0) else ("medium" if abs(z) > self.z_threshold else "low")
                anomalies.append(
                    Anomaly(
                        metric=metric,
                        dimension=dimension,
                        observation=obs_str,
                        value=round(val, 2),
                        expected_range=AnomalyExpectedRange(lower=lower_bound, upper=upper_bound),
                        type=anom_type,
                        severity=severity
                    )
                )

        anomalies.sort(key=lambda a: abs(a.value - mean), reverse=True)
        return anomalies[:max_anomalies]
