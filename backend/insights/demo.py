"""
Demo & Integration Script for the Visualization & Insight Engine.

Demonstrates:
1. Loading the business dataset.
2. Generating the complete Structured Insights JSON matching the schema contract.
3. Rendering and exporting charts (Time-series, Bar, Distribution).
4. Generating Plotly-compatible JSON structures.
5. Verifying interoperability with the downstream prompt builder and validator.
"""

import os
import json
import pandas as pd

from backend.insights import (
    InsightEngine,
    VisualizationEngine
)
from backend.llm.prompt_builder import build_prompt
from backend.llm.validation import extract_source_numbers


def run_demo():
    print("=" * 60)
    print("VISUALIZATION & INSIGHT ENGINE DEMO")
    print("=" * 60)

    # 1. Load sample dataset
    data_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "sample_sales.csv")
    output_json_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "sample_insights.json")
    output_charts_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "charts")

    os.makedirs(output_charts_dir, exist_ok=True)

    print(f"\n[1] Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"    Loaded {len(df)} records with columns: {list(df.columns)}")

    # 2. Initialize InsightEngine
    engine = InsightEngine()
    print("\n[2] Executing InsightEngine (Trends, Comparisons, Anomalies, Distributions, Chart Summaries)...")

    insights_obj = engine.generate_insights(
        df=df,
        dataset_name="sales.csv",
        domain="sales",
        temporal_col="date",
        numeric_cols=["sales", "profit", "quantity"],
        categorical_cols=["category", "region", "product"],
        identifier_cols=["order_id"],
        unit_currency="INR"
    )

    insights_dict = insights_obj.to_contract_dict()

    # 3. Save Structured Insights JSON
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(insights_dict, f, indent=2, ensure_ascii=False)
    print(f"\n[3] Saved Structured Insights JSON to: {output_json_path}")
    print(f"    - Metrics count: {len(insights_dict['metrics'])}")
    print(f"    - Trends count: {len(insights_dict['trends'])}")
    print(f"    - Comparisons count: {len(insights_dict['comparisons'])}")
    print(f"    - Anomalies count: {len(insights_dict['anomalies'])}")
    print(f"    - Distributions count: {len(insights_dict['distributions'])}")
    print(f"    - Visualizations count: {len(insights_dict['visualizations'])}")

    # 4. Generate & Save Static Visualizations (Matplotlib)
    vis = VisualizationEngine()
    print("\n[4] Rendering and saving sample charts...")

    fig_line = vis.plot_time_series(
        df,
        date_col="date",
        metric_col="sales",
        title="Monthly Sales Trend",
        save_path=os.path.join(output_charts_dir, "monthly_sales_trend.png")
    )
    print(f"    - Saved: {os.path.join(output_charts_dir, 'monthly_sales_trend.png')}")

    fig_bar = vis.plot_category_bar(
        df,
        category_col="category",
        metric_col="sales",
        title="Sales by Category",
        save_path=os.path.join(output_charts_dir, "sales_by_category.png")
    )
    print(f"    - Saved: {os.path.join(output_charts_dir, 'sales_by_category.png')}")

    fig_dist = vis.plot_distribution(
        df,
        metric_col="sales",
        title="Sales Value Distribution",
        save_path=os.path.join(output_charts_dir, "sales_distribution.png")
    )
    print(f"    - Saved: {os.path.join(output_charts_dir, 'sales_distribution.png')}")

    # 5. Generate Plotly JSON for Frontend
    print("\n[5] Generating Plotly-compatible JSON specs for React frontend...")
    plotly_line = vis.create_plotly_line_chart(df, date_col="date", metric_col="sales", title="Interactive Sales Trend")
    print(f"    - Plotly Line Spec keys: {list(plotly_line.keys())}, points: {len(plotly_line['data'][0]['x'])}")

    # 6. Verify Interoperability with LLM Prompt Builder & Validator
    print("\n[6] Verifying interoperability with LLM Prompt Builder...")
    prompt = build_prompt(insights_dict)
    print("    - Prompt built successfully! Character length:", len(prompt))

    source_numbers = extract_source_numbers(insights_dict)
    print(f"    - Validator successfully extracted {len(source_numbers)} ground-truth numbers.")

    print("\n" + "=" * 60)
    print("ALL PIPELINE STEPS COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()
