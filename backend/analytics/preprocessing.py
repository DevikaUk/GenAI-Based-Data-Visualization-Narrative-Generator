"""
preprocessing.py — Universal Data Preprocessing Pipeline (Member 1 Deliverable)
================================================================================
Loads, auto-detects schema, validates, cleans, and transforms ANY structured
business/sales dataset into a standardised DataFrame ready for KPI, trend, and
insight generation.

Key Capabilities:
-----------------
1. Universal CSV Ingestion: Supports multi-encodings (utf-8, latin-1, cp1252),
   auto-detected delimiters (comma, semicolon, tab, pipe).
2. Intelligent Column Auto-Detection: Uses fuzzy alias matching & heuristic type
   inspection to map arbitrary column names (e.g. 'InvoiceDate' -> 'date',
   'TotalAmount' -> 'sales', 'Country' -> 'region', 'Item_Desc' -> 'product').
3. Graceful Fallbacks & Derivations: Auto-generates missing optional columns
   (e.g., missing order_id -> auto-generated sequence; missing unit_price -> sales/qty;
   missing quantity -> 1; missing profit -> 0 / estimated margin).
4. Full Documented Cleaning & Feature Engineering: Robust date parsing, numeric
   sanitisation, negative-value guards, string title-casing, calendar features
   (year, quarter, month, week, period_label), and profit margins.
5. Dynamic Column Catalogue: Generates metadata for both source and derived fields.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# ── Logging ───────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Canonical Schema Names ───────────────────────────────────────────────────
CANONICAL_COLUMNS: list[str] = [
    "order_id", "date", "product", "category",
    "region", "quantity", "unit_price", "sales", "profit",
]

NUMERIC_COLUMNS: list[str] = ["quantity", "unit_price", "sales", "profit"]
CATEGORICAL_COLUMNS: list[str] = ["product", "category", "region"]
TEMPORAL_COLUMNS: list[str] = ["date"]
IDENTIFIER_COLUMNS: list[str] = ["order_id"]

# Minimum conceptual fields needed to do business analysis: date and sales/revenue
ESSENTIAL_FIELDS: list[str] = ["date", "sales"]
REQUIRED_COLUMNS: list[str] = CANONICAL_COLUMNS  # Backward compatibility for existing tests

# ── Column Synonym Dictionaries (for Auto-Detection) ─────────────────────────
SYNONYM_MAP: dict[str, list[str]] = {
    "date": [
        "date", "order_date", "orderdate", "invoicedate", "invoice_date",
        "trans_date", "transaction_date", "sales_date", "sale_date",
        "timestamp", "time", "datetime", "purchase_date", "created_at",
        "period", "order_dt", "posting_date"
    ],
    "sales": [
        "sales", "revenue", "amount", "total", "total_amount", "totalamount",
        "total_sales", "totalsales", "total_price", "totalprice", "sales_amount",
        "sale_amount", "invoice_amount", "price_total", "turnover", "income",
        "gross_sales", "net_sales", "val", "value", "subtotal", "grand_total"
    ],
    "profit": [
        "profit", "net_profit", "netprofit", "margin", "gross_profit",
        "grossprofit", "profit_amount", "income_net", "gain", "earnings",
        "net_income"
    ],
    "quantity": [
        "quantity", "qty", "units", "units_sold", "unitssold", "count",
        "volume", "pieces", "items", "amount_sold", "item_count", "order_qty"
    ],
    "unit_price": [
        "unit_price", "unitprice", "price_per_unit", "item_price", "itemprice",
        "price", "rate", "cost_per_unit", "unit_cost"
    ],
    "product": [
        "product", "product_name", "productname", "item", "item_name",
        "itemname", "description", "product_description", "stockcode",
        "sku", "title", "product_title", "article", "service"
    ],
    "category": [
        "category", "product_category", "productcategory", "item_category",
        "sub_category", "subcategory", "department", "type", "group",
        "segment", "class", "genre", "product_type", "division"
    ],
    "region": [
        "region", "country", "city", "state", "location", "zone",
        "market", "territory", "province", "area", "geo", "branch",
        "district", "store_location"
    ],
    "order_id": [
        "order_id", "orderid", "order_number", "ordernumber", "invoice_no",
        "invoiceno", "invoice_number", "transaction_id", "transactionid",
        "id", "bill_no", "receipt_no", "order_num", "row_id"
    ],
}


# ── Ingestion Helpers ─────────────────────────────────────────────────────────

def _normalize_name(name: str) -> str:
    """Normalize string for fuzzy column matching (lowercase, no symbols)."""
    return re.sub(r"[^a-z0-9]", "", str(name).strip().lower())


def load_csv(filepath: str | Path) -> pd.DataFrame:
    """Load a CSV file with automatic encoding and delimiter fallback.

    Supports comma, semicolon, tab, and pipe delimiters, as well as
    utf-8, latin-1, cp1252, and iso-8859-1 encodings.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    encodings = ["utf-8", "latin-1", "cp1252", "iso-8859-1"]
    delimiters = [",", ";", "\t", "|"]

    df = None
    last_err = None

    for enc in encodings:
        for sep in delimiters:
            try:
                temp_df = pd.read_csv(filepath, encoding=enc, sep=sep, dtype=str)
                # Ensure it has more than 1 column or only 1 if that's all there is
                if temp_df.shape[1] > 1 or len(delimiters) == 1:
                    df = temp_df
                    logger.info("Loaded CSV with encoding='%s', sep='%s', shape=%s", enc, sep, df.shape)
                    break
            except Exception as e:
                last_err = e
                continue
        if df is not None:
            break

    if df is None:
        raise ValueError(f"Failed to read CSV at {filepath}: {last_err}")

    return df


# ── Auto-Detection & Standardisation ──────────────────────────────────────────

def detect_and_map_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Intelligently map arbitrary dataset columns to canonical business names.

    Parameters
    ----------
    df:
        Raw loaded DataFrame.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, str]]
        Standardized DataFrame with canonical names and mapping dict (canonical -> original).
    """
    df = df.copy()
    col_map: dict[str, str] = {}
    used_orig_cols = set()

    normalized_orig_cols = {_normalize_name(c): c for c in df.columns}

    # 1. First pass: Exact canonical name & exact synonym match across all canonicals
    for canonical, synonyms in SYNONYM_MAP.items():
        if canonical in df.columns and canonical not in used_orig_cols:
            col_map[canonical] = canonical
            used_orig_cols.add(canonical)
            continue

        for syn in synonyms:
            syn_norm = _normalize_name(syn)
            if syn_norm in normalized_orig_cols:
                orig = normalized_orig_cols[syn_norm]
                if orig not in used_orig_cols:
                    col_map[canonical] = orig
                    used_orig_cols.add(orig)
                    break

    # 2. Second pass: Partial / Substring match for remaining unmapped canonicals
    for canonical, synonyms in SYNONYM_MAP.items():
        if canonical in col_map:
            continue
        for syn in synonyms:
            syn_norm = _normalize_name(syn)
            if len(syn_norm) < 3:
                continue
            matched = False
            for norm_k, orig_k in normalized_orig_cols.items():
                if orig_k not in used_orig_cols and (syn_norm in norm_k or norm_k in syn_norm):
                    col_map[canonical] = orig_k
                    used_orig_cols.add(orig_k)
                    matched = True
                    break
            if matched:
                break

    # 2. Heuristic fallback for Date (if not matched by name)
    if "date" not in col_map:
        for col in df.columns:
            if col in used_orig_cols:
                continue
            sample = df[col].dropna().head(20)
            if len(sample) > 0:
                try:
                    parsed = pd.to_datetime(sample, errors="coerce")
                    if parsed.notna().mean() > 0.8:
                        col_map["date"] = col
                        used_orig_cols.add(col)
                        logger.info("Auto-detected date column from data inspection: '%s'", col)
                        break
                except Exception:
                    pass

    # 3. Heuristic fallback for Sales/Revenue (if not matched by name)
    if "sales" not in col_map:
        if "unit_price" in col_map and "quantity" in col_map:
            # When unit_price and quantity exist, we compute sales = unit_price * quantity
            col_map["sales"] = "(computed: unit_price * quantity)"
        else:
            numeric_candidates = []
            for col in df.columns:
                if col in used_orig_cols:
                    continue
                # Exclude obvious identifier column names from being picked as sales
                norm_c = _normalize_name(col)
                if any(id_kw in norm_c for id_kw in ["id", "code", "zip", "phone", "post", "number"]):
                    continue
                sample = pd.to_numeric(df[col].astype(str).str.replace(r"[$,€£₹,\s]", "", regex=True), errors="coerce").dropna()
                if len(sample) > 0 and sample.notna().mean() > 0.8 and sample.mean() > 0:
                    numeric_candidates.append((col, sample.std(), sample.mean()))
            if numeric_candidates:
                # Pick numeric column with highest variance/mean as primary metric
                best_col = max(numeric_candidates, key=lambda x: (x[1] if not np.isnan(x[1]) else 0, x[2]))[0]
                col_map["sales"] = best_col
                used_orig_cols.add(best_col)
                logger.info("Auto-detected sales/revenue column: '%s'", best_col)

    # 4. Heuristic fallback for Categoricals (category, region, product)
    available_text_cols = [
        c for c in df.columns
        if c not in used_orig_cols and df[c].dtype == object and df[c].nunique() > 1
    ]

    if "category" not in col_map and available_text_cols:
        # Low-to-moderate cardinality (2 to 50 distinct items)
        cat_cands = [c for c in available_text_cols if 2 <= df[c].nunique() <= 50]
        if cat_cands:
            chosen = cat_cands[0]
            col_map["category"] = chosen
            used_orig_cols.add(chosen)
            available_text_cols.remove(chosen)

    if "region" not in col_map and available_text_cols:
        # Low cardinality (2 to 30 distinct items)
        reg_cands = [c for c in available_text_cols if 2 <= df[c].nunique() <= 30]
        if reg_cands:
            chosen = reg_cands[0]
            col_map["region"] = chosen
            used_orig_cols.add(chosen)
            available_text_cols.remove(chosen)

    if "product" not in col_map and available_text_cols:
        chosen = available_text_cols[0]
        col_map["product"] = chosen
        used_orig_cols.add(chosen)

    # Build standardized dataframe
    std_df = pd.DataFrame(index=df.index)
    for canonical, orig in col_map.items():
        if orig in df.columns:
            std_df[canonical] = df[orig]

    # Compute sales if missing but unit_price and quantity are available
    if "sales" not in std_df.columns and "unit_price" in std_df.columns and "quantity" in std_df.columns:
        u = pd.to_numeric(std_df["unit_price"].astype(str).str.replace(r"[$,€£₹,\s]", "", regex=True), errors="coerce").fillna(0)
        q = pd.to_numeric(std_df["quantity"].astype(str).str.replace(r"[$,€£₹,\s]", "", regex=True), errors="coerce").fillna(1)
        std_df["sales"] = u * q
        col_map["sales"] = "(computed: unit_price * quantity)"

    # Keep all other original columns as well so no source data is lost
    for col in df.columns:
        if col not in used_orig_cols and col not in std_df.columns:
            std_df[col] = df[col]

    # 5. Populate missing optional canonical fields with safe derivations
    if "order_id" not in std_df.columns:
        std_df["order_id"] = [f"ORD{i+1:06d}" for i in range(len(std_df))]
        col_map["order_id"] = "(auto-generated)"

    if "quantity" not in std_df.columns:
        std_df["quantity"] = 1.0
        col_map["quantity"] = "(default: 1)"

    if "profit" not in std_df.columns:
        if "sales" in std_df.columns:
            # If no profit column, estimate nominal 15% profit margin
            std_df["profit"] = pd.to_numeric(std_df["sales"], errors="coerce").fillna(0) * 0.15
            col_map["profit"] = "(estimated: 15% margin)"
        else:
            std_df["profit"] = 0.0
            col_map["profit"] = "(default: 0)"

    if "unit_price" not in std_df.columns:
        s = pd.to_numeric(std_df.get("sales", 0), errors="coerce").fillna(0)
        q = pd.to_numeric(std_df.get("quantity", 1), errors="coerce").replace(0, 1).fillna(1)
        std_df["unit_price"] = s / q
        col_map["unit_price"] = "(derived: sales/quantity)"

    if "category" not in std_df.columns:
        std_df["category"] = "General"
        col_map["category"] = "(default: General)"

    if "region" not in std_df.columns:
        std_df["region"] = "Global"
        col_map["region"] = "(default: Global)"

    if "product" not in std_df.columns:
        std_df["product"] = std_df.get("category", "General Item")
        col_map["product"] = "(default)"

    logger.info("Column auto-mapping: %s", col_map)
    return std_df, col_map


# ── Schema Validation ─────────────────────────────────────────────────────────

def validate_schema(df: pd.DataFrame, required: Optional[list[str]] = None) -> None:
    """Raise ValueError if any required column is missing.

    Parameters
    ----------
    df:
        DataFrame to validate.
    required:
        Optional custom list of required columns (defaults to REQUIRED_COLUMNS).
    """
    target_cols = required if required is not None else REQUIRED_COLUMNS
    missing = [c for c in target_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    logger.info("Schema validation passed.")


# ── Inspection ────────────────────────────────────────────────────────────────

def inspect(df: pd.DataFrame) -> dict:
    """Return summary dictionary for logging and EDA inspection."""
    id_col = "order_id" if "order_id" in df.columns else df.columns[0]
    cat_cols = [c for c in CATEGORICAL_COLUMNS if c in df.columns]

    return {
        "shape": df.shape,
        "dtypes": {str(k): str(v) for k, v in df.dtypes.items()},
        "null_counts": {str(k): int(v) for k, v in df.isnull().sum().items()},
        "duplicate_count": int(df.duplicated(subset=[id_col]).sum()) if id_col in df.columns else 0,
        "categorical_uniques": {col: int(df[col].nunique()) for col in cat_cols},
    }


# ── Data Cleaning ─────────────────────────────────────────────────────────────

def clean(df: pd.DataFrame, *, drop_duplicate_ids: bool = False) -> pd.DataFrame:
    """Apply robust cleaning rules to produce a clean DataFrame.

    Rules applied:
    1. Parse date column to datetime; drop unparseable dates.
    2. Coerce numeric columns to float (cleaning currencies/commas).
    3. Drop negative rows for non-profit columns (sales, quantity, unit_price).
    4. Strip whitespace and title-case categorical columns.
    5. Handle duplicate order_ids if requested.
    6. Sort temporally and reset index.
    """
    df = df.copy()
    initial_len = len(df)

    # 1. Date parsing
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        bad_dates = df["date"].isna().sum()
        if bad_dates:
            logger.warning("Dropping %d rows with unparseable dates.", bad_dates)
        df = df.dropna(subset=["date"])

    # 2. Coerce numerics & strip formatting ($ , € ₹)
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.replace(r"[$,€£₹,\s]", "", regex=True)
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows where sales or quantity is entirely NaN
    present_num_cols = [c for c in ["sales", "quantity"] if c in df.columns]
    if present_num_cols:
        df = df.dropna(subset=present_num_cols)

    # 3. Guard against negative sales/quantity
    if "sales" in df.columns:
        df = df[df["sales"] >= 0]
    if "quantity" in df.columns:
        df = df[df["quantity"] >= 0]
    if "unit_price" in df.columns:
        df = df[df["unit_price"] >= 0]

    # 4. Normalise categoricals
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str).str.strip().str.title()

    # 5. Duplicates
    if drop_duplicate_ids and "order_id" in df.columns:
        df = df.drop_duplicates(subset=["order_id"], keep="first")

    # 6. Sort & Reset
    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    removed = initial_len - len(df)
    logger.info("Cleaning complete: %d -> %d rows (%d removed).", initial_len, len(df), removed)
    return df


# ── Feature Engineering ───────────────────────────────────────────────────────

def add_derived_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar and financial derived columns for analytics."""
    df = df.copy()

    if "date" in df.columns and pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["year"] = df["date"].dt.year
        df["quarter"] = "Q" + df["date"].dt.quarter.astype(str)
        df["month"] = df["date"].dt.month
        df["month_name"] = df["date"].dt.strftime("%b")
        df["week"] = df["date"].dt.isocalendar().week.astype(int)
        df["period_label"] = df["year"].astype(str) + "-" + df["quarter"]

    # Profit Margin %
    if "sales" in df.columns and "profit" in df.columns:
        sales_val = df["sales"].replace(0, np.nan)
        df["profit_margin"] = (df["profit"] / sales_val) * 100
    elif "profit_margin" not in df.columns:
        df["profit_margin"] = 0.0

    # Revenue alias for schema compatibility
    if "sales" in df.columns:
        df["revenue"] = df["sales"]

    logger.info("Derived fields added.")
    return df


# ── End-to-End Pipeline ───────────────────────────────────────────────────────

def preprocess(
    source: str | Path | pd.DataFrame,
    *,
    drop_duplicate_ids: bool = False,
) -> pd.DataFrame:
    """Universal preprocessing pipeline for any tabular dataset.

    Parameters
    ----------
    source:
        Path to CSV file or existing DataFrame.
    drop_duplicate_ids:
        Whether to drop duplicate order_ids.

    Returns
    -------
    pd.DataFrame
        Cleaned, mapped, and feature-engineered DataFrame.
    """
    if isinstance(source, pd.DataFrame):
        df_raw = source
    else:
        df_raw = load_csv(source)

    # 1. Auto-detect & standardize columns
    std_df, col_map = detect_and_map_columns(df_raw)

    # 2. Validate standardized schema
    validate_schema(std_df)

    # 3. Clean
    cleaned_df = clean(std_df, drop_duplicate_ids=drop_duplicate_ids)

    # 4. Add derived fields
    final_df = add_derived_fields(cleaned_df)

    # Store mapping as attribute for downstream catalogue builders
    final_df.attrs["col_map"] = col_map
    logger.info("Preprocessing completed successfully. Shape: %s", final_df.shape)
    return final_df


# ── Column Catalogue ──────────────────────────────────────────────────────────

def get_column_catalogue(df: Optional[pd.DataFrame] = None) -> dict:
    """Return documented column metadata dictionary."""
    base_catalogue = {
        "order_id": {"type": "identifier", "meaning": "Unique order or transaction identifier", "unit": None, "expected_range": "Unique alphanumeric"},
        "date": {"type": "temporal", "meaning": "Transaction or order timestamp", "unit": "YYYY-MM-DD", "expected_range": "Temporal dates"},
        "product": {"type": "categorical", "meaning": "Product, item, or service description", "unit": None, "expected_range": "Distinct products"},
        "category": {"type": "categorical", "meaning": "High-level department or product category", "unit": None, "expected_range": "Category segments"},
        "region": {"type": "categorical", "meaning": "Geographic location, market, or zone", "unit": None, "expected_range": "Regional divisions"},
        "quantity": {"type": "numeric", "meaning": "Number of units purchased", "unit": "units", "expected_range": ">= 0"},
        "unit_price": {"type": "numeric", "meaning": "Price per unit before discount", "unit": "INR", "expected_range": ">= 0"},
        "sales": {"type": "numeric", "meaning": "Total transaction revenue (sales)", "unit": "INR", "expected_range": ">= 0"},
        "profit": {"type": "numeric", "meaning": "Net profit on transaction", "unit": "INR", "expected_range": "Real value (can be negative)"},
        "year": {"type": "numeric", "meaning": "Calendar year", "unit": None, "expected_range": "e.g. 2024"},
        "quarter": {"type": "categorical", "meaning": "Calendar quarter", "unit": None, "expected_range": "Q1 - Q4"},
        "month": {"type": "numeric", "meaning": "Calendar month number", "unit": None, "expected_range": "1 - 12"},
        "month_name": {"type": "categorical", "meaning": "Abbreviated month name", "unit": None, "expected_range": "Jan - Dec"},
        "week": {"type": "numeric", "meaning": "ISO calendar week number", "unit": None, "expected_range": "1 - 53"},
        "profit_margin": {"type": "numeric", "meaning": "Profit margin percentage", "unit": "%", "expected_range": "-100% to +100%"},
        "revenue": {"type": "numeric", "meaning": "Alias for sales revenue", "unit": "INR", "expected_range": ">= 0"},
        "period_label": {"type": "categorical", "meaning": "Year-Quarter grouping tag", "unit": None, "expected_range": "e.g. 2024-Q1"},
    }

    if df is not None:
        return {col: base_catalogue.get(col, {"type": "other", "meaning": col, "unit": None, "expected_range": "Dynamic"}) for col in df.columns}

    return base_catalogue
