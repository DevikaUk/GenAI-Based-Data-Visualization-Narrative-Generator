"""
Unit tests for TrendDetector.
"""

import unittest
import pandas as pd
from backend.insights.trend_detector import TrendDetector


class TestTrendDetector(unittest.TestCase):

    def setUp(self):
        self.detector = TrendDetector(threshold_stable_pct=3.0, threshold_high_pct=10.0)

    def test_increasing_trend(self):
        data = {
            "date": pd.date_range(start="2026-01-01", periods=5, freq="D"),
            "sales": [100, 110, 120, 130, 150]
        }
        df = pd.DataFrame(data)
        trend = self.detector.detect_trend(df, metric="sales", dimension="date")

        self.assertIsNotNone(trend)
        self.assertEqual(trend.direction, "increasing")
        self.assertEqual(trend.change, 50.0)
        self.assertEqual(trend.significance, "high")
        self.assertEqual(trend.period.start, "2026-01-01")
        self.assertEqual(trend.period.end, "2026-01-05")

    def test_decreasing_trend(self):
        data = {
            "date": pd.date_range(start="2026-01-01", periods=4, freq="D"),
            "sales": [200, 180, 160, 150]
        }
        df = pd.DataFrame(data)
        trend = self.detector.detect_trend(df, metric="sales", dimension="date")

        self.assertIsNotNone(trend)
        self.assertEqual(trend.direction, "decreasing")
        self.assertEqual(trend.change, -25.0)
        self.assertEqual(trend.significance, "high")

    def test_stable_trend(self):
        data = {
            "date": pd.date_range(start="2026-01-01", periods=4, freq="D"),
            "sales": [100, 101, 100, 101]
        }
        df = pd.DataFrame(data)
        trend = self.detector.detect_trend(df, metric="sales", dimension="date")

        self.assertIsNotNone(trend)
        self.assertEqual(trend.direction, "stable")
        self.assertEqual(trend.change, 1.0)
        self.assertEqual(trend.significance, "low")

    def test_insufficient_data(self):
        df_empty = pd.DataFrame()
        self.assertIsNone(self.detector.detect_trend(df_empty, metric="sales"))

        df_single = pd.DataFrame({"date": ["2026-01-01"], "sales": [100]})
        self.assertIsNone(self.detector.detect_trend(df_single, metric="sales"))


if __name__ == "__main__":
    unittest.main()
