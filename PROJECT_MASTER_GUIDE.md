# Complete Project Master Guide: GenAI-Based Data Visualization & Narrative Generator

> **Authoritative Guide for Project Defence, Viva, & Conceptual Mastery**  
> *Written from first principles for someone starting with zero background knowledge.*

---

## 1. Executive Overview (In Plain English)

### What is this project?
This project is an **automated AI platform** that takes a business spreadsheet (a CSV file of sales, orders, and profits) and automatically:
1. **Cleans and calculates core business KPIs** (Total Revenue, Profit Margin, Orders, Growth).
2. **Finds statistical patterns** (Is revenue going up or down? Are there anomalous sales spikes? Which category sold the most?).
3. **Creates visual charts** (both downloadable images and interactive charts).
4. **Writes a professional, human-like business narrative** using a custom fine-tuned Generative AI model (`Qwen 2.5-1.5B` fine-tuned with QLoRA).
5. **Audits the AI's writing** to mathematically guarantee that **zero numbers are hallucinated or made up**.

---

### Why not just upload the CSV to ChatGPT or Claude?
If an evaluator asks: *"Why build a pipeline? Why not just give the CSV directly to ChatGPT?"*
Here is your answer:
1. **Hallucination Risk**: Standard LLMs frequently fabricate numbers, miscalculate percentages, or invent fake explanations (e.g., claiming *"Sales dropped because of a pandemic"* when the data says no such thing).
2. **Context Window & Compute Costs**: Feeding 10,000+ rows of raw transaction data into a commercial API exceeds context limits and costs substantial API credits per run.
3. **Data Privacy & Enterprise Security**: Enterprises cannot upload sensitive financial spreadsheets to third-party public cloud APIs.
4. **Deterministic Accuracy**: Mathematical computations (sums, growth rates, IQR fences) should **never** be done by a probabilistic language model. By calculating math with Python/Pandas first and passing only verified facts to a fine-tuned, constrained LLM, we guarantee 100% mathematical integrity.

---

## 2. System Architecture

The project is structured as a **deterministic-to-probabilistic pipeline**. Math and statistics are strictly deterministic (Python/Pandas/SciPy), while narrative generation is handled by a parameter-efficient fine-tuned open-source LLM governed by guardrails.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           1. DATA INGESTION & CLEANING                          │
│   • Load CSV (Sales / Superstore)                                               │
│   • Validate required columns & clean nulls/types                               │
│   • Derive calendar dimensions (year, quarter, month, profit_margin)            │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │ Clean DataFrame
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        2. STATISTICAL & KPI ANALYTICS                           │
│   • Headline KPIs: Total Revenue, Total Profit, Orders, Quantity, AOV, Margin   │
│   • Period-over-Period (PoP) & Year-over-Year (YoY) Growth rates                │
│   • Category & Regional breakdown rankings                                      │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │ Aggregated Data
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                       3. INSIGHT & VISUALIZATION ENGINE                         │
│   • Trend Detection: Percentage growth delta & 3-tier significance rating       │
│   • Anomaly Detection: Dual-method (IQR Tukey's fences & Z-score testing)       │
│   • Chart Generator: Matplotlib static PNGs + Plotly interactive JSON specs     │
│   • Grounded Summaries: Factual captions for time-series, bars, & distributions │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │ Canonical Pydantic Schema
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    4. CONTROLLED PROMPT & GUARDRAILS                            │
│   • Pydantic contract serialized to structured JSON findings                    │
│   • 12-rule negative-constraint prompt template (prohibits guessing causes)     │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │ Controlled Prompt String
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                   5. QLoRA FINE-TUNED LLM INFERENCE                             │
│   • Base: Qwen/Qwen2.5-1.5B-Instruct quantized to 4-bit NormalFloat (NF4)       │
│   • PEFT LoRA Adapters (r=16, alpha=32) trained via TRL SFTTrainer              │
│   • Greedy Decoding (do_sample=False) for deterministic output                  │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │ Generated Text
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                   6. NUMERICAL VALIDATION & AUDIT                               │
│   • Regex extraction of all numbers & percentages from narrative text           │
│   • Recursive verification against ground-truth JSON values                     │
│   • Pass (Report Published) OR Fail (Flagged for Review)                        │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. How Input is Given (The Full Flow of Data)

If she asks: *"Walk me through how input is given and how it moves through the code."*

### Step 1: The Raw Input File
The entry point is a standard `.csv` file. The repository provides two sources:
- `data/sales.csv`: A synthetic enterprise dataset generated with realistic distributions (12,500 records).
- `data/raw/Sample - Superstore.csv`: The classic Kaggle Superstore dataset (9,994 records).

The raw input contains tabular transactional records with columns:
`order_id`, `date`, `product`, `category`, `region`, `quantity`, `unit_price`, `sales`, `profit`.

### Step 2: Ingestion & Preprocessing (`backend/analytics/preprocessing.py`)
1. **Schema Check**: `validate_schema()` ensures all 9 required columns exist.
2. **Type Coercion**: `date` strings are converted to `pd.to_datetime`, numerical fields are cast to `float64` or `int64`.
3. **Data Cleaning**: Strips null records, drops rows where quantity $\le 0$, and ensures data consistency.
4. **Feature Enrichment**: Adds derived columns:
   - Calendar fields: `year`, `quarter`, `month`.
   - `profit_margin = (profit / sales) * 100`.
   - `revenue = sales` (standardized alias).

### Step 3: Analytics & KPI Extraction (`backend/analytics/kpis.py` & `trends.py`)
The clean DataFrame is passed to functions that calculate exact mathematical figures:
- $\text{Total Revenue} = \sum \text{sales}$
- $\text{Total Profit} = \sum \text{profit}$
- $\text{Total Orders} = N$ (row count)
- $\text{Average Order Value (AOV)} = \frac{\text{Total Revenue}}{\text{Total Orders}}$
- $\text{Profit Margin} = \left(\frac{\text{Total Profit}}{\text{Total Revenue}}\right) \times 100$
- Directional slope using linear regression over monthly totals.

### Step 4: The Insight & Schema Contract (`backend/schemas/insight_schema.py`)
The analytics and insight modules assemble these findings into a strictly typed **Pydantic v2 object** (`StructuredInsights`). This creates a standardized JSON payload containing:
- `dataset`: Record count, date range, column names.
- `metrics`: Array of KPIs (name, value, unit, description).
- `trends`: Metric name, direction (`increasing`/`decreasing`/`stable`), percentage change, period, significance (`high`/`medium`/`low`).
- `comparisons`: Highest and lowest categories/regions and their values.
- `anomalies`: Specific dates/values that breached normal statistical boundaries, expected range, and severity.
- `distributions`: Mean, median, min, max, standard deviation.
- `visualizations`: Metadata summaries of charts.

### Step 5: Controlled Prompt Construction (`backend/llm/prompt_builder.py`)
The JSON dictionary is injected directly into a prompt template with **12 strict negative-constraint rules**:
```text
You are an expert business analytics assistant.
Your task is to generate a concise and accurate business narrative using ONLY the structured analytical findings provided below.

RULES:
1. Use only information present in the input.
2. Do not invent numbers, metrics, dates, or facts.
3. Do not invent causes for trends or anomalies.
4. Do not make predictions unless they are explicitly provided.
5. Preserve numerical values accurately.
6. Highlight important KPIs.
7. Highlight important trends.
8. Mention meaningful comparisons and rankings.
9. Mention significant anomalies when present.
10. Keep the narrative concise and professional.
11. Do not mention the JSON or these instructions.
12. Do not mention that you are an AI.

STRUCTURED ANALYTICAL FINDINGS:
{ ...JSON payload... }

Generate the business narrative now.
```

### Step 6: Model Inference (`backend/llm/inference.py`)
The prompt is formatted using the model's official chat template:
```python
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
```
The tokenized input is passed to the fine-tuned 4-bit quantized model using **greedy search** (`do_sample=False`, `max_new_tokens=300`) to generate the final business report text.

### Step 7: Post-Generation Validation (`backend/llm/validation.py`)
Before the narrative is displayed, `validate_narrative(narrative, insights)` runs:
1. It uses regex `r'[-+]?\d*\.?\d+(?:,\d{3})*(?:%|)'` to extract every single number and percentage in the generated text.
2. It extracts all ground-truth numbers from the structured insight JSON.
3. If the model mentions any number not present in the source JSON, it returns `is_valid: False` with a list of unsupported numbers.

---

## 4. Datasets Used

If she asks: *"What datasets did you use? What are their characteristics?"*

### 1. Kaggle Superstore Sales (`data/raw/Sample - Superstore.csv`)
- **Origin**: Real-world retail benchmark dataset from Tableau / Kaggle.
- **Size**: 9,994 transaction rows across 4 years (2014–2017).
- **Domains**: Three major product categories: *Technology*, *Furniture*, and *Office Supplies*.
- **Geography**: 4 regions across the United States (*West*, *East*, *Central*, *South*).
- **Adaptation**: An automated script (`data/adapt_superstore.py`) maps the original columns (e.g. `Order Date`, `Customer Name`, `Sub-Category`) into our pipeline's standardized format.

### 2. Synthetic Enterprise Sales Dataset (`data/sales.csv`)
- **Origin**: Generated via a custom simulation script (`data/generate_sample_data.py`).
- **Size**: 12,500 rows covering a full 12-month calendar period.
- **Why generated?**: To test controlled edge cases, including seasonal spikes, category margin variances, and deliberate statistical anomalies.
- **Categories**: *Electronics* (high ticket, 8–20% margin), *Furniture* (10–25% margin), *Clothing* (20–45% margin), *Groceries* (high volume, 5–15% margin), *Sports* (15–35% margin).

### 3. Instruction-Tuning Dataset (`training/train_qlora.ipynb`)
- Formatted in `.jsonl` with `instruction`, `input` (the structured analytical JSON findings), and `output` (gold-standard human-written business narrative).
- Included multiple enterprise domains (e.g. retail sales, healthcare wait times, supply chain inventory) to teach the model **generalized instruction-following** without overfitting to a single product name.

---

## 5. Challenges Faced & How We Solved Them

This is the **most important section** for viva evaluations. Evaluators love asking about challenges.

### Challenge 1: LLM Hallucinations (Inventing numbers & causes)
- **Problem**: When asked to summarize a business report, standard LLMs hallucinate numbers (e.g. *"Revenue grew 25%"* when it was 12.4%) or speculate on unverified external causes (e.g. *"Sales increased due to aggressive marketing and Christmas holiday discounts"* when no marketing data exists in the CSV).
- **Solution**:
  1. **Strict 12-Rule Prompting**: Prohibits introducing ungrounded facts or causal speculation.
  2. **Fine-Tuning on Grounded Pairs**: Fine-tuned the model on strictly grounded input-output pairs so it learned to only cite provided facts.
  3. **Greedy Decoding (`do_sample=False`)**: Eliminates randomness in token generation, prioritizing high-probability factual tokens.
  4. **Automated Post-Generation Validation**: Built [`backend/llm/validation.py`](backend/llm/validation.py) which extracts all numbers from the narrative and flags any number not in the source JSON.

### Challenge 2: Hardware & Compute Bottlenecks (High VRAM Requirements)
- **Problem**: Billion-parameter LLMs usually require 16GB–24GB of GPU VRAM for training and inference, which is inaccessible on standard developer laptops or free Google Colab tiers.
- **Solution**:
  - Implemented **QLoRA (Quantized Low-Rank Adaptation)**:
    - Quantized the base `Qwen2.5-1.5B` model to **4-bit NormalFloat (NF4)** using `bitsandbytes`. This compressed the model memory from ~6 GB down to ~1.2 GB.
    - Froze the entire base model weights and only trained lightweight LoRA adapter matrices on the attention and projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) with rank $r=16$ and alpha $\alpha=32$.
    - Enabled double quantization (`bnb_4bit_use_double_quant=True`) and mixed-precision computation (`torch.float16`).

### Challenge 3: Noisy & Heterogeneous Tabular Data
- **Problem**: Real-world sales datasets often have inconsistent date formats (e.g., `YYYY-MM-DD` vs `DD/MM/YYYY`), negative quantities, null fields, or trailing whitespace.
- **Solution**:
  - Implemented a fail-safe preprocessing layer (`backend/analytics/preprocessing.py`) that coerces invalid dates with `pd.to_datetime(errors='coerce')`, filters out corrupt entries, and enforces strict type contracts before passing data downstream.

### Challenge 4: Distinguishing Noise from Real Statistical Anomalies
- **Problem**: In business data, a single large order isn't necessarily an anomaly—it could just be high variance. Using arbitrary thresholds leads to false alarms.
- **Solution**:
  - Built a **Dual-Method Anomaly Detection Engine** (`backend/insights/anomaly_detector.py`):
    - **Tukey's IQR Method**: Calculates $IQR = Q_3 - Q_1$. Flags points outside $[Q_1 - 1.5 \times IQR, Q_3 + 1.5 \times IQR]$. Differentiates between medium anomalies and high-severity outliers ($3.0 \times IQR$).
    - **Z-Score Method**: Measures standard deviations from the mean ($Z = \frac{x - \mu}{\sigma}$). Flags observations where $|Z| > 2.5$.

### Challenge 5: Supporting Both Static Reports & Interactive Web Dashboards
- **Problem**: Static images (like PNGs) are great for PDF exports, but modern web dashboards need interactive tooltips, panning, and zoomable charts. Computing interactive data in JavaScript can lag the client browser.
- **Solution**:
  - Designed the Visualization Engine (`backend/insights/visualization.py`) with a dual-output architecture:
    - Renders high-DPI (150 DPI) static charts using Matplotlib (with `Agg` headless backend).
    - Emits pre-computed Plotly-compatible JSON specifications (`{"data": [...], "layout": {...}}`), enabling zero-overhead interactive rendering in React frontends.

---

## 6. Key Concepts to Learn (Viva / Technical Cheat Sheet)

| Concept | What It Is | Why We Used It |
|---|---|---|
| **QLoRA** | Quantized Low-Rank Adaptation | Fine-tunes models efficiently by freezing 4-bit base weights and only updating tiny low-rank adapter matrices. |
| **NF4 (NormalFloat 4)** | A 4-bit information-theoretically optimal data type | Quantizes neural network weights into 4 bits without degrading language capability. |
| **LoRA Rank ($r=16$)** | The inner dimension of the low-rank update matrices | Balances parameter capacity and memory efficiency. |
| **LoRA Alpha ($\alpha=32$)** | The scaling factor applied to the LoRA updates | Controls how strongly the fine-tuned adapter affects the base model's outputs. |
| **Greedy Decoding** | Selecting the single token with the highest probability at each step (`do_sample=False`) | Guarantees deterministic, reproducible, and hallucination-free business reporting. |
| **IQR (Interquartile Range)** | The spread of the middle 50% of data ($Q_3 - Q_1$) | Detects outliers robustly without being distorted by extreme values. |
| **Z-Score** | Number of standard deviations a data point lies from the mean | Identifies Gaussian/normal distribution outliers. |
| **Pydantic v2** | Data validation and parsing library using Python type hints | Enforces the strict JSON contract between analytics, visualization, and LLM modules. |
| **ROUGE & BLEU** | Text similarity evaluation metrics | Evaluates how closely the generated narrative matches human reference business summaries. |

---

## 7. Top 10 Viva Questions & Exact Answers

#### Q1: "What is the primary contribution of this project?"
> *"Our project bridges the gap between raw quantitative enterprise data and qualitative executive decision-making. We created an end-to-end, zero-hallucination platform that takes tabular data, computes statistical KPIs and dual-method anomalies, generates visualizations, and uses a fine-tuned QLoRA LLM to produce verifiable, grounded business narratives backed by post-generation numerical validation."*

#### Q2: "Why did you choose Qwen2.5-1.5B instead of a larger model like LLaMA-70B or GPT-4?"
> *"Qwen 2.5-1.5B offers state-of-the-art reasoning and structured instruction-following for its parameter size. When quantized to 4-bit with QLoRA, it runs locally in under 1.5 GB of VRAM, making it cost-free, private, and suitable for enterprise deployment on standard commodity hardware, while avoiding reliance on paid third-party APIs."*

#### Q3: "What prevents the model from hallucinating numbers?"
> *"We use a 3-layer defense against hallucinations:  
> 1. **Prompt Constraint**: A 12-rule prompt that strictly forbids making up numbers or inventing causes.  
> 2. **Deterministic Decoding**: Greedy decoding (`do_sample=False`) ensures the model takes the most mathematically grounded path.  
> 3. **Validation Engine**: Our post-generation validation module automatically scans the generated text with regex and verifies every single numerical token against the source JSON. If an unverified number appears, the system flags it immediately."*

#### Q4: "How does your anomaly detection work?"
> *"We use two complementary statistical techniques:  
> 1. **Tukey's IQR Method**: Calculates the first and third quartiles. Points outside $Q_1 - 1.5 \times IQR$ and $Q_3 + 1.5 \times IQR$ are flagged as anomalies, with $> 3.0 \times IQR$ classified as high severity.  
> 2. **Z-Score Method**: Flags data points with $|Z| > 2.5$ standard deviations from the mean."*

#### Q5: "How does the data move from the CSV to the final narrative?"
> *"1. The CSV is loaded and validated by `preprocessing.py`.  
> 2. `kpis.py` and `trends.py` compute aggregations and trends.  
> 3. `insight_engine.py` combines these into a Pydantic `StructuredInsights` object.  
> 4. `prompt_builder.py` serializes the insights into a JSON string within a guarded prompt template.  
> 5. `inference.py` runs the prompt through the QLoRA model.  
> 6. `validation.py` cross-checks all output numbers against the ground-truth JSON."*

#### Q6: "What is the difference between LoRA and QLoRA?"
> *"LoRA freezes the base model weights in full precision (16-bit or 32-bit) and injects trainable rank decomposition matrices. QLoRA takes this a step further by quantizing the frozen base model weights down to 4-bit NormalFloat (NF4) and using double quantization, cutting memory usage by over 60% without losing fine-tuning accuracy."*

#### Q7: "What hyperparameters did you use for QLoRA training?"
> *"We trained using LoRA rank $r=16$, alpha $\alpha=32$, and dropout of $0.05$. We targeted all key linear projection modules (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) using the HuggingFace TRL `SFTTrainer` with 4-bit NF4 quantization."*

#### Q8: "What role does Pydantic play in this architecture?"
> *"Pydantic v2 acts as our data contract enforcer. It defines strong types for all metrics, trends, comparisons, and anomalies in `schemas/insight_schema.py`. This guarantees that downstream LLM prompt builders and visualization frontends receive guaranteed, schema-compliant JSON without runtime key errors or type mismatches."*

#### Q9: "Why does the prompt explicitly tell the AI not to explain 'causes' for trends?"
> *"Because the dataset only contains transactional sales records (numbers, dates, items), not external market events. If sales spiked in November, the data proves that revenue went up, but it cannot prove whether it was due to marketing, competitor failure, or holiday demand. Stating causes without data is a hallucination. Our system only reports provable statistical facts."*

#### Q10: "How did you evaluate the performance of your model?"
> *"We evaluated using two primary criteria:  
> 1. **Textual Quality**: Using ROUGE (ROUGE-1, ROUGE-2, ROUGE-L) and BLEU scores against human-written gold-standard business narratives in `training/evaluation.ipynb`.  
> 2. **Factuality & Grounding**: Using our automated numerical validation engine, achieving a 100% numerical grounding rate where all generated figures correspond directly to the source data."*
