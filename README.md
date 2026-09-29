# GenAI-Based Data Visualization & Narrative Generator

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E.svg)](https://huggingface.co/)
[![PEFT](https://img.shields.io/badge/PEFT-QLoRA%204--bit-orange.svg)](https://github.com/huggingface/peft)
[![Pydantic](https://img.shields.io/badge/Pydantic-v2.0-e92063.svg)](https://docs.pydantic.dev/)
[![Plotly](https://img.shields.io/badge/Visualization-Plotly%20%7C%20Matplotlib-3F4F75.svg)](https://plotly.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Amrita Vishwa Vidyapeetham — Semester 7 GenAI Final Case Study**  
> An end-to-end automated analytics and narrative generation system that transforms tabular enterprise datasets into interactive visualizations, rigorous statistical findings, and zero-hallucination, executive-ready business narratives using fine-tuned open-source Large Language Models (Qwen 2.5 + QLoRA).

---

## Table of Contents

- [Executive Summary](#executive-summary)
- [System Architecture](#system-architecture)
- [Key Features](#key-features)
- [Team Roles & Modules](#team-roles--modules)
  - [Member 1 — Data Preprocessing & Analytics](#member-1--data-preprocessing--analytics)
  - [Member 2 — Visualization & Insight Engine](#member-2--visualization--insight-engine)
  - [Member 3 — GenAI Narrative Engine & Validation](#member-3--genai-narrative-engine--validation)
- [Data Contract & Schema Specification](#data-contract--schema-specification)
- [QLoRA Fine-Tuning & Evaluation](#qlora-fine-tuning--evaluation)
- [Directory Structure](#directory-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Pipeline](#running-the-pipeline)
  - [Running Tests](#running-tests)
- [Tech Stack](#tech-stack)
- [Authors & Acknowledgments](#authors--acknowledgments)

---

## Executive Summary

Modern enterprise business intelligence tools produce charts and tabular dashboards, yet business leaders rely on written narratives for context, strategic meaning, and decision-making. Standard generative LLMs applied to raw data frequently suffer from **numerical hallucinations**, **unsupported causal inferences**, and **lack of strict analytical grounding**.

This platform solves these challenges through a strict, multi-stage architecture:
1. **Deterministic Analytics**: Cleans raw tabular records and computes headline KPIs, growth trends, comparisons, and distributions.
2. **Statistical Insight & Visualization Engine**: Employs IQR and Z-score methods to isolate anomalies, classifies growth significance, generates factual chart summaries, and prepares both static and Plotly-compatible visual specifications.
3. **Controlled GenAI & Hallucination Guardrails**: Feeds structured JSON data into a strictly constrained 12-rule prompt template, queries a QLoRA fine-tuned `Qwen/Qwen2.5-1.5B-Instruct` model, and runs an automated numerical verification engine to guarantee zero hallucinated figures.

---

## System Architecture

```mermaid
flowchart TD
    A[Raw CSV Dataset] --> B[Data Preprocessing & Validation]
    B --> C[KPI & Statistical Analytics Engine]
    C --> D[Member 1: Analytics JSON]
    
    D --> E[Member 2: Insight Engine]
    E --> F[Trend & Significance Detector]
    E --> G[Anomaly Detector IQR & Z-Score]
    E --> H[Visualization Engine Matplotlib & Plotly]
    E --> I[Chart Summary Generator]
    
    F --> J[Pydantic Structured Insights Contract]
    G --> J
    H --> J
    I --> J
    
    J --> K[Member 3: LLM Prompt Builder]
    K --> L[12-Rule Constrained Prompt]
    L --> M[QLoRA Fine-Tuned Model Qwen 2.5-1.5B]
    M --> N[Generated Business Narrative]
    
    N --> O[Numerical Grounding & Hallucination Validator]
    J --> O
    O -->|Validated| P[Executive Narrative Report & Visual Dashboard]
    O -->|Discrepancy Found| Q[Validation Alert / Rejection]
```

---

## Key Features

- **Strict Zero-Hallucination Pipeline**: All natural language claims are rooted directly in verified statistical findings; no external causes or unsupported numbers are admitted.
- **Automated Business KPIs**: Calculates Total Revenue, Total Profit, Order Volumes, Average Order Value (AOV), Profit Margins, and Period-over-Period (PoP) metrics.
- **Dual Anomaly Detection**: Combines Interquartile Range (IQR fences) and Z-score testing ($|Z| > 2.5$) with severity classifications (`low`, `medium`, `high`).
- **Interactive & Static Visualizations**: Generates high-res Matplotlib chart images alongside frontend-ready Plotly JSON specifications.
- **Parameter-Efficient Fine-Tuning (PEFT / QLoRA)**: Adapts `Qwen/Qwen2.5-1.5B-Instruct` in 4-bit NormalFloat (NF4) with double quantization, trained via TRL `SFTTrainer`.
- **Post-Generation Number Verification**: Regex-based extraction audits every numerical entity in the generated text against the source JSON payload.

---

## Team Roles & Modules

```
Member 1: Data & Analytics  ──▶  Member 2: Insights & Charts  ──▶  Member 3: LLM & Validation
```

### Member 1 — Data Preprocessing & Analytics
- **Primary Package:** [`backend/analytics/`](backend/analytics/)
- **Core Files:**
  - [`preprocessing.py`](backend/analytics/preprocessing.py): Loads raw CSVs, performs type inference, cleans missing values, and adds derived calendar dimensions (`year`, `quarter`, `month`, `profit_margin`).
  - [`kpis.py`](backend/analytics/kpis.py): Computes core business metrics (Revenue, Profit, Orders, Quantity, AOV, Profit Margin, YoY/PoP revenue changes).
  - [`trends.py`](backend/analytics/trends.py): Performs linear regression on time aggregates to determine trajectory and identifies category extrema.
  - [`analytics_builder.py`](backend/analytics/analytics_builder.py): Orchestrates Member 1 steps and writes the baseline analytics JSON.
  - [`notebooks/member1_eda.ipynb`](notebooks/member1_eda.ipynb): Comprehensive exploratory data analysis on the sales data.

### Member 2 — Visualization & Insight Engine
- **Primary Package:** [`backend/insights/`](backend/insights/) & [`backend/schemas/`](backend/schemas/)
- **Core Files:**
  - [`schemas/insight_schema.py`](backend/schemas/insight_schema.py): Pydantic v2 contract formalizing the `StructuredInsights` schema.
  - [`trend_detector.py`](backend/insights/trend_detector.py): Computes growth percentage $\Delta \% = \left(\frac{V_{\text{end}} - V_{\text{start}}}{|V_{\text{start}}|}\right) \times 100$ and classifies trends as `increasing` (> +3%), `decreasing` (< -3%), or `stable`, with 3-tier significance rating (`low`, `medium`, `high`).
  - [`anomaly_detector.py`](backend/insights/anomaly_detector.py): Detects statistical outliers using IQR fences ($Q_1 - 1.5 \times IQR$, $Q_3 + 1.5 \times IQR$) and Z-score thresholding ($|Z| > 2.5$).
  - [`chart_summary.py`](backend/insights/chart_summary.py): Generates factual, unopinionated narrative summaries for time-series, category bars, and distributions.
  - [`visualization.py`](backend/insights/visualization.py): Renders Matplotlib static PNG figures, base64 strings, and Plotly-compatible JSON structures.
  - [`insight_engine.py`](backend/insights/insight_engine.py): Central orchestrator producing the final contract JSON.
  - [`demo.py`](backend/insights/demo.py): End-to-end integration demo verifying data generation through Member 3 interoperability.

### Member 3 — GenAI Narrative Engine & Validation
- **Primary Package:** [`backend/llm/`](backend/llm/) & [`training/`](training/)
- **Core Files:**
  - [`prompt_builder.py`](backend/llm/prompt_builder.py): Formulates a 12-rule controlled prompt preventing the LLM from inventing causes, dates, ungrounded numbers, or speculative forecasts.
  - [`inference.py`](backend/llm/inference.py): Loads the 4-bit NF4 quantized `Qwen/Qwen2.5-1.5B-Instruct` model with PEFT LoRA adapters, performing greedy decoding (`do_sample=False`) for deterministic business reporting.
  - [`validation.py`](backend/llm/validation.py): Audits the generated output by extracting numbers via regex and cross-referencing against source insight values.
  - [`training/train_qlora.ipynb`](training/train_qlora.ipynb): Google Colab notebook for QLoRA fine-tuning with TRL `SFTTrainer`.
  - [`training/evaluation.ipynb`](training/evaluation.ipynb): Comparative evaluation notebook benchmarking base Qwen vs QLoRA fine-tuned weights on narrative quality and grounding.

---

## Data Contract & Schema Specification

The pipeline relies on a unified Pydantic contract ([`backend/schemas/insight_schema.py`](backend/schemas/insight_schema.py)) that standardizes communication between components:

```json
{
  "dataset": {
    "name": "sales.csv",
    "domain": "sales",
    "record_count": 9994,
    "columns": {
      "numeric": ["sales", "profit", "quantity"],
      "categorical": ["category", "region", "product"],
      "temporal": ["date"],
      "identifier": ["order_id"]
    }
  },
  "metrics": [
    {
      "name": "Total Revenue",
      "value": 2297200.86,
      "unit": "INR",
      "description": "Sum of all order sales"
    }
  ],
  "trends": [
    {
      "metric": "sales",
      "dimension": "date",
      "direction": "increasing",
      "change": 18.4,
      "unit": "percent",
      "period": { "start": "2023-01-01", "end": "2024-12-31" },
      "significance": "high"
    }
  ],
  "comparisons": [
    {
      "dimension": "category",
      "metric": "sales",
      "highest": { "name": "Technology", "value": 836154.03 },
      "lowest": { "name": "Office Supplies", "value": 719047.03 }
    }
  ],
  "anomalies": [
    {
      "metric": "sales",
      "dimension": "date",
      "observation": "2024-03-18",
      "value": 22638.48,
      "expected_range": { "lower": 12.5, "upper": 3200.0 },
      "type": "high",
      "severity": "high"
    }
  ],
  "distributions": [
    {
      "metric": "sales",
      "mean": 229.85,
      "median": 54.49,
      "minimum": 0.44,
      "maximum": 22638.48,
      "std_dev": 623.24
    }
  ],
  "visualizations": [
    {
      "title": "Monthly Sales Trend",
      "type": "line",
      "metric": "sales",
      "dimension": "date",
      "summary": "Monthly sales displayed an upward trend from Jan 2023 to Dec 2024."
    }
  ]
}
```

---

## QLoRA Fine-Tuning & Evaluation

### Training Setup
- **Base Model:** `Qwen/Qwen2.5-1.5B-Instruct`
- **Quantization:** 4-bit NormalFloat4 (NF4), double quantization enabled, compute dtype `torch.float16` via `bitsandbytes`.
- **LoRA Configuration:**
  - Rank ($r$): `16`
  - Alpha ($\alpha$): `32`
  - Dropout: `0.05`
  - Target Modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
  - Task Type: `CAUSAL_LM`
- **Trainer:** HuggingFace `trl.SFTTrainer` with `SFTConfig`

### Evaluation
The model is benchmarked against the base model using automated text similarity and hallucination tracking in [`training/evaluation.ipynb`](training/evaluation.ipynb):
- **ROUGE-1 / ROUGE-2 / ROUGE-L** & **BLEU** against gold-standard business summaries.
- **Numerical Accuracy Rate**: Verified via [`validation.py`](backend/llm/validation.py), ensuring that all numbers in the output match source insights.

---

## Directory Structure

```
GenAI-Based-Data-Visualization-Narrative-Generator/
├── backend/
│   ├── analytics/                     # Member 1: Preprocessing & Analytics
│   │   ├── __init__.py
│   │   ├── analytics_builder.py       # Member 1 pipeline builder
│   │   ├── kpis.py                    # KPI calculation functions
│   │   ├── preprocessing.py           # Data cleaning & type conversion
│   │   └── trends.py                  # Linear trends & category comparisons
│   ├── insights/                      # Member 2: Visualization & Insight Engine
│   │   ├── __init__.py
│   │   ├── anomaly_detector.py        # IQR & Z-Score anomaly detectors
│   │   ├── chart_summary.py           # Grounded chart caption generation
│   │   ├── demo.py                    # End-to-end integration demo script
│   │   ├── insight_engine.py          # Unified insight orchestrator
│   │   ├── trend_detector.py          # Percentage growth & significance
│   │   ├── visualization.py           # Matplotlib & Plotly JSON generator
│   │   └── README.md                  # Module-specific documentation
│   ├── llm/                           # Member 3: GenAI & Validation
│   │   ├── __init__.py
│   │   ├── inference.py               # Local 4-bit QLoRA inference loader
│   │   ├── prompt_builder.py          # 12-rule constrained prompt builder
│   │   └── validation.py              # Numerical grounding validator
│   ├── schemas/                       # Shared Schema Contracts
│   │   ├── __init__.py
│   │   └── insight_schema.py          # Pydantic v2 data models
│   └── test_prompt.py                 # Prompt generator test script
├── data/
│   ├── charts/                        # Generated static visualization PNGs
│   │   ├── monthly_sales_trend.png
│   │   ├── sales_by_category.png
│   │   └── sales_distribution.png
│   ├── raw/                           # Raw input datasets (e.g. Superstore)
│   │   └── Sample - Superstore.csv
│   ├── adapt_superstore.py            # Superstore dataset adaptation script
│   ├── analytics_output.json          # Precomputed analytics output (Member 1)
│   ├── generate_sample_data.py        # Synthetic dataset generator
│   ├── sales.csv                      # Primary sales dataset
│   ├── sample_insights.json           # Sample structured insights (Member 2)
│   └── sample_sales.csv               # Sample sales subset for demos
├── notebooks/
│   └── member1_eda.ipynb              # Member 1 Exploratory Data Analysis
├── tests/                             # Unified Unit Test Suite
│   ├── __init__.py
│   ├── test_anomaly_detector.py       # Tests for IQR and Z-Score detection
│   ├── test_chart_summary.py          # Tests for chart narrative summaries
│   ├── test_insight_schema.py         # Tests for Pydantic schema contracts
│   ├── test_member1.py                # Tests for Member 1 analytics & KPIs
│   ├── test_trend_detector.py         # Tests for trend & growth calculations
│   └── test_visualization.py          # Tests for Matplotlib & Plotly exports
├── training/                          # Member 3 Training & Evaluation
│   ├── evaluation.ipynb               # QLoRA vs Base evaluation notebook
│   └── train_qlora.ipynb              # QLoRA fine-tuning notebook
├── .gitignore
├── requirements.txt                   # Complete project dependencies
└── README.md                          # Repository documentation
```

---

## Getting Started

### Prerequisites
- Python 3.10 or 3.11
- Git
- NVIDIA GPU with CUDA support (recommended for local QLoRA inference; CPU/Colab supported)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/DevikaUk/GenAI-Based-Data-Visualization-Narrative-Generator.git
   cd GenAI-Based-Data-Visualization-Narrative-Generator
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # On macOS/Linux:
   python3 -m venv venv
   source venv/bin/activate

   # On Windows:
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Pipeline

1. **Generate or adapt sales dataset:**
   ```bash
   python data/generate_sample_data.py
   # Or adapt the included Superstore dataset:
   python data/adapt_superstore.py
   ```

2. **Run Member 1 Analytics Pipeline:**
   ```bash
   python backend/analytics/analytics_builder.py --csv data/sales.csv --out data/analytics_output.json
   ```

3. **Run Member 2 Insight & Visualization Demo:**
   ```bash
   python backend/insights/demo.py
   ```
   *This outputs `data/sample_insights.json` and static charts in `data/charts/`.*

4. **Test Member 3 Prompt Builder:**
   ```bash
   python backend/test_prompt.py
   ```

5. **Run Local QLoRA Inference (Requires trained adapter or checkpoint):**
   ```python
   from backend.llm.inference import generate_narrative
   from backend.llm.validation import validate_narrative
   import json

   with open("data/sample_insights.json") as f:
       insights = json.load(f)

   narrative = generate_narrative(insights)
   print("Generated Narrative:\n", narrative)

   report = validate_narrative(narrative, insights)
   print("\nValidation Status:", "PASSED" if report["is_valid"] else "FAILED")
   ```

### Running Tests

Execute the comprehensive test suite across all modules:

```bash
# Using pytest:
pytest tests/ -v

# Using Python unittest:
python -m unittest discover -s tests -v
```

---

## Tech Stack

| Layer | Technologies |
|---|---|
| **Language & Runtime** | Python 3.10+, Jupyter Notebooks |
| **Data & Computation** | Pandas, NumPy, SciPy, Statsmodels |
| **Data Validation** | Pydantic v2 |
| **Data Visualization** | Plotly, Matplotlib |
| **GenAI / LLM** | Qwen 2.5 (1.5B Instruct), Hugging Face Transformers |
| **Model Adaptation** | PEFT, QLoRA (4-bit NF4 Quantization), BitsAndBytes, TRL |
| **Testing & CI** | Pytest, Unittest |

---

## Authors & Acknowledgments

Developed as part of the **Amrita Vishwa Vidyapeetham Semester 7 GenAI Final Case Study**:

- **Member 1:** Data Pipeline, Preprocessing, KPI Computation, Trend & Linear Regressions, EDA.
- **Member 2:** Statistical Insight Engine, Anomaly Detection (IQR/Z-Score), Visualization Engine, Pydantic Contract.
- **Member 3:** Controlled Prompting, QLoRA Fine-Tuning, PEFT Inference Engine, Numerical Grounding & Hallucination Auditing.
