"""
Unit tests for VisualizationEngine.
"""

import unittest
import pandas as pd
from backend.insights.visualization import VisualizationEngine


class TestVisualizationEngine(unittest.TestCase):

    def setUp(self):
        self.engine = VisualizationEngine()
        self.df = pd.DataFrame({
            "date": pd.date_range("2026-01-01", periods=6, freq="D"),
            "sales": [1000, 1200, 900, 1500, 1800, 2000],
            "category": ["A", "B", "A", "C", "B", "C"]
        })

    def test_plot_time_series_matplotlib(self):
        fig = self.engine.plot_time_series(self.df, date_col="date", metric_col="sales")
        self.assertIsNotNone(fig)
        b64 = self.engine.figure_to_base64(fig)
        self.assertTrue(len(b64) > 100)

    def test_plot_category_bar_matplotlib(self):
        fig = self.engine.plot_category_bar(self.df, category_col="category", metric_col="sales")
        self.assertIsNotNone(fig)
        b64 = self.engine.figure_to_base64(fig)
        self.assertTrue(len(b64) > 100)

    def test_plot_distribution_matplotlib(self):
        fig = self.engine.plot_distribution(self.df, metric_col="sales")
        self.assertIsNotNone(fig)
        b64 = self.engine.figure_to_base64(fig)
        self.assertTrue(len(b64) > 100)

    def test_create_plotly_line_chart(self):
        spec = self.engine.create_plotly_line_chart(self.df, date_col="date", metric_col="sales")
        self.assertIn("data", spec)
        self.assertIn("layout", spec)
        self.assertEqual(len(spec["data"]), 1)
        self.assertEqual(spec["data"][0]["type"], "scatter")

    def test_create_plotly_bar_chart(self):
        spec = self.engine.create_plotly_bar_chart(self.df, category_col="category", metric_col="sales")
        self.assertIn("data", spec)
        self.assertIn("layout", spec)
        self.assertEqual(len(spec["data"]), 1)
        self.assertEqual(spec["data"][0]["type"], "bar")

    def test_generate_kpi_cards(self):
        metrics = [
            {"name": "Total Sales", "value": 8400, "unit": "INR", "description": "Sum of sales"}
        ]
        cards = self.engine.generate_kpi_cards(metrics)
        self.assertEqual(len(cards), 1)
        self.assertEqual(cards[0]["title"], "Total Sales")
        self.assertIn("8,400", cards[0]["formatted_value"])


if __name__ == "__main__":
    unittest.main()
