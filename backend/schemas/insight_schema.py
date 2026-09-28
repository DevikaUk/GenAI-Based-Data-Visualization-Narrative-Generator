"""
Structured Insight Schema Definition.

Defines the common JSON contract agreed upon across team members (Section 4).
Member 2 produces this schema, which Member 3 consumes for prompt construction
and numerical validation.
"""

from typing import List, Dict, Optional, Any, Union
from pydantic import BaseModel, Field


class ColumnMetadata(BaseModel):
    numeric: List[str] = Field(default_factory=list)
    categorical: List[str] = Field(default_factory=list)
    temporal: List[str] = Field(default_factory=list)
    identifier: List[str] = Field(default_factory=list)


class DatasetMetadata(BaseModel):
    name: str
    domain: str
    record_count: int
    columns: ColumnMetadata


class Metric(BaseModel):
    name: str
    value: Union[int, float]
    unit: str
    description: str


class TrendPeriod(BaseModel):
    start: str
    end: str


class Trend(BaseModel):
    metric: str
    dimension: str
    direction: str  # "increasing", "decreasing", "stable"
    change: float   # percentage change
    unit: str = "percent"
    period: TrendPeriod
    significance: str = "medium"  # "high", "medium", "low"


class ComparisonExtremum(BaseModel):
    name: str
    value: Union[int, float]


class Comparison(BaseModel):
    dimension: str
    metric: str
    highest: ComparisonExtremum
    lowest: ComparisonExtremum


class AnomalyExpectedRange(BaseModel):
    lower: float
    upper: float


class Anomaly(BaseModel):
    metric: str
    dimension: str
    observation: str
    value: Union[int, float]
    expected_range: AnomalyExpectedRange
    type: str  # "high", "low"
    severity: str  # "high", "medium", "low"


class Distribution(BaseModel):
    metric: str
    mean: float
    median: float
    minimum: float
    maximum: float
    std_dev: Optional[float] = None


class VisualizationSummary(BaseModel):
    title: str
    type: str  # "line", "bar", "histogram", "pie"
    metric: str
    dimension: str
    summary: str


class StructuredInsights(BaseModel):
    dataset: DatasetMetadata
    metrics: List[Metric] = Field(default_factory=list)
    trends: List[Trend] = Field(default_factory=list)
    comparisons: List[Comparison] = Field(default_factory=list)
    anomalies: List[Anomaly] = Field(default_factory=list)
    distributions: List[Distribution] = Field(default_factory=list)
    visualizations: List[VisualizationSummary] = Field(default_factory=list)

    def to_contract_dict(self) -> Dict[str, Any]:
        """Convert to the standard dictionary format matching project contract."""
        return self.model_dump(exclude_none=True)
