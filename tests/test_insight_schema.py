"""
Unit tests for Structured Insight Schema.
"""

import unittest
from backend.schemas.insight_schema import (
    StructuredInsights,
    DatasetMetadata,
    ColumnMetadata,
    Metric,
    Trend,
    TrendPeriod,
    Comparison,
    ComparisonExtremum,
    Anomaly,
    AnomalyExpectedRange,
    Distribution,
    VisualizationSummary
)


class TestInsightSchema(unittest.TestCase):

    def test_schema_serialization_contract(self):
        insights = StructuredInsights(
            dataset=DatasetMetadata(
                name="sales.csv",
                domain="sales",
                record_count=12500,
                columns=ColumnMetadata(
                    numeric=["sales", "profit"],
                    categorical=["category"],
                    temporal=["date"],
                    identifier=["order_id"]
                )
            ),
            metrics=[
                Metric(
                    name="Total Revenue",
                    value=1250000,
                    unit="INR",
                    description="Total revenue"
                )
            ],
            trends=[
                Trend(
                    metric="Revenue",
                    dimension="date",
                    direction="increasing",
                    change=12.4,
                    unit="percent",
                    period=TrendPeriod(start="2026-01-01", end="2026-03-31"),
                    significance="high"
                )
            ],
            comparisons=[
                Comparison(
                    dimension="category",
                    metric="Revenue",
                    highest=ComparisonExtremum(name="Electronics", value=450000),
                    lowest=ComparisonExtremum(name="Furniture", value=180000)
                )
            ],
            anomalies=[
                Anomaly(
                    metric="Revenue",
                    dimension="date",
                    observation="2026-02-15",
                    value=95000,
                    expected_range=AnomalyExpectedRange(lower=40000, upper=65000),
                    type="high",
                    severity="medium"
                )
            ],
            distributions=[
                Distribution(
                    metric="Order Value",
                    mean=2450.5,
                    median=2100.0,
                    minimum=150.0,
                    maximum=18500.0
                )
            ],
            visualizations=[
                VisualizationSummary(
                    title="Monthly Revenue",
                    type="line",
                    metric="Revenue",
                    dimension="date",
                    summary="Revenue increased consistently."
                )
            ]
        )

        d = insights.to_contract_dict()

        # Validate top-level keys match Section 4 contract exactly
        self.assertIn("dataset", d)
        self.assertIn("metrics", d)
        self.assertIn("trends", d)
        self.assertIn("comparisons", d)
        self.assertIn("anomalies", d)
        self.assertIn("distributions", d)
        self.assertIn("visualizations", d)

        # Validate values
        self.assertEqual(d["dataset"]["name"], "sales.csv")
        self.assertEqual(d["metrics"][0]["value"], 1250000)
        self.assertEqual(d["trends"][0]["change"], 12.4)
        self.assertEqual(d["comparisons"][0]["highest"]["name"], "Electronics")
        self.assertEqual(d["anomalies"][0]["value"], 95000)
        self.assertEqual(d["distributions"][0]["mean"], 2450.5)


if __name__ == "__main__":
    unittest.main()
