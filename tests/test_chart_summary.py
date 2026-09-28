"""
Unit tests for ChartSummaryGenerator.
"""

import unittest
from backend.insights.chart_summary import ChartSummaryGenerator


class TestChartSummaryGenerator(unittest.TestCase):

    def test_line_chart_summary_increasing(self):
        summary = ChartSummaryGenerator.summarize_line_chart(
            title="Monthly Revenue",
            metric="Revenue",
            dimension="date",
            direction="increasing",
            change_pct=15.2,
            peak_label="2026-03-31",
            peak_val=95000
        )
        self.assertEqual(summary.title, "Monthly Revenue")
        self.assertEqual(summary.type, "line")
        self.assertIn("increasing", summary.summary)
        self.assertIn("15.2%", summary.summary)
        self.assertIn("95000", summary.summary)

    def test_bar_chart_summary(self):
        summary = ChartSummaryGenerator.summarize_bar_chart(
            title="Revenue by Category",
            metric="Revenue",
            dimension="category",
            highest_name="Electronics",
            highest_val=450000,
            lowest_name="Furniture",
            lowest_val=180000
        )
        self.assertEqual(summary.title, "Revenue by Category")
        self.assertEqual(summary.type, "bar")
        self.assertIn("Electronics", summary.summary)
        self.assertIn("450000", summary.summary)
        self.assertIn("Furniture", summary.summary)
        self.assertIn("180000", summary.summary)

    def test_distribution_chart_summary(self):
        summary = ChartSummaryGenerator.summarize_distribution_chart(
            title="Order Value Distribution",
            metric="Order Value",
            dimension="order_id",
            mean_val=2450.5,
            median_val=2100.0,
            min_val=150.0,
            max_val=18500.0
        )
        self.assertEqual(summary.title, "Order Value Distribution")
        self.assertEqual(summary.type, "histogram")
        self.assertIn("2450.5", summary.summary)
        self.assertIn("18500.0", summary.summary)


if __name__ == "__main__":
    unittest.main()
