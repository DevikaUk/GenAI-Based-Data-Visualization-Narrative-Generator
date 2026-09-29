# Visualization & Insight Engine

This module implements the **Visualization & Insight Engine** for the GenAI-Based Data Visualization Narrative Generator.

## Responsibilities
- **Consumes:** Cleaned data and statistical metrics from the Data Analytics Pipeline.
- **Produces:** Structured Insights JSON contract (consumed by the LLM prompt builder) and visualization artifacts (consumed by dashboards and reports).

---

## Directory Structure
```
backend/
├── schemas/
│   ├── __init__.py
│   └── insight_schema.py      # Pydantic models enforcing Section 4 JSON contract
└── insights/
    ├── __init__.py
    ├── trend_detector.py      # Directional trend detection & significance classification
    ├── anomaly_detector.py    # IQR & Z-score statistical outlier detection
    ├── chart_summary.py       # Factual, unopinionated chart metadata summaries
    ├── visualization.py       # Matplotlib static exports & Plotly-compatible JSON specs
    ├── insight_engine.py      # Central orchestrator generating StructuredInsights
    ├── demo.py                # End-to-end verification and integration script
    └── README.md              # Technical documentation
```

---

## Key Modules & Rules

### 1. Trend Detection (`trend_detector.py`)
- **Metric Growth Calculation:**
  $$\Delta \% = \left(\frac{V_{\text{end}} - V_{\text{start}}}{|V_{\text{start}}|}\right) \times 100$$
- **Direction Classification:**
  - `increasing`: $\Delta \% > +3.0\%$
  - `decreasing`: $\Delta \% < -3.0\%$
  - `stable`: $-3.0\% \le \Delta \% \le +3.0\%$
- **Significance Classification:**
  - `high`: $|\Delta \%| \ge 10.0\%$
  - `medium`: $3.0\% \le |\Delta \%| < 10.0\%$
  - `low`: $|\Delta \%| < 3.0\%$

### 2. Anomaly Detection (`anomaly_detector.py`)
- **Interquartile Range (IQR) Method:**
  - $IQR = Q_3 - Q_1$
  - $\text{Lower Bound} = Q_1 - 1.5 \times IQR$
  - $\text{Upper Bound} = Q_3 + 1.5 \times IQR$
  - Outliers flagged with type (`high` / `low`) and severity (`high`, `medium`, `low` based on excess ratio).
- **Z-Score Method:**
  - $Z = \frac{x - \mu}{\sigma}$
  - Outliers flagged when $|Z| > 2.5$.

### 3. Chart Summaries (`chart_summary.py`)
- Produces strictly grounded, non-hallucinatory textual summaries for time-series, categorical comparisons, and distributions without postulating external causes.

### 4. Visualization Engine (`visualization.py`)
- **Matplotlib Renderer:** Exports high-resolution PNG charts and base64 strings for static reporting.
- **Plotly JSON Generator:** Emits standard `{"data": [...], "layout": {...}}` dictionaries for interactive React charts without requiring heavy frontend computation.

---

## How to Run Tests
```bash
python3 -m unittest discover -s tests -v
```

## How to Run Demo
```bash
PYTHONPATH=. python3 backend/insights/demo.py
```
