"""
Unit tests for AnomalyDetector.
"""

import unittest
import pandas as pd
from backend.insights.anomaly_detector import AnomalyDetector


class TestAnomalyDetector(unittest.TestCase):

    def setUp(self):
        self.detector = AnomalyDetector(method="iqr", iqr_factor=1.5)

    def test_detect_high_anomaly_iqr(self):
        # A normal series with one massive spike
        dates = [f"2026-01-0{i}" for i in range(1, 9)]
        sales = [100, 105, 95, 102, 98, 104, 101, 1000]  # 1000 is an obvious outlier
        df = pd.DataFrame({"date": dates, "sales": sales})

        anomalies = self.detector.detect_anomalies_iqr(df, metric="sales", dimension="date")

        self.assertGreaterEqual(len(anomalies), 1)
        top_anomaly = anomalies[0]
        self.assertEqual(top_anomaly.type, "high")
        self.assertEqual(top_anomaly.observation, "2026-01-08")
        self.assertEqual(top_anomaly.value, 1000.0)
        self.assertGreater(top_anomaly.value, top_anomaly.expected_range.upper)

    def test_detect_zscore_anomaly(self):
        detector_z = AnomalyDetector(method="zscore", z_threshold=2.0)
        dates = [f"2026-01-0{i}" for i in range(1, 9)]
        sales = [50, 52, 48, 51, 49, 53, 50, 500]
        df = pd.DataFrame({"date": dates, "sales": sales})

        anomalies = detector_z.detect_anomalies_zscore(df, metric="sales", dimension="date")

        self.assertGreaterEqual(len(anomalies), 1)
        self.assertEqual(anomalies[0].type, "high")
        self.assertEqual(anomalies[0].value, 500.0)

    def test_no_anomaly(self):
        dates = [f"2026-01-0{i}" for i in range(1, 9)]
        sales = [100, 102, 101, 99, 100, 103, 101, 102]
        df = pd.DataFrame({"date": dates, "sales": sales})

        anomalies = self.detector.detect_anomalies_iqr(df, metric="sales", dimension="date")
        self.assertEqual(len(anomalies), 0)


if __name__ == "__main__":
    unittest.main()
