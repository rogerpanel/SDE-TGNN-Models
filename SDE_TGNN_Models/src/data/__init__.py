"""Data loading, preprocessing, and feature engineering for SDE-TGNN.

This subpackage provides utilities for:
- Loading and preprocessing six network intrusion detection datasets
- Constructing temporal graphs from network flow data
- Feature harmonization across heterogeneous data domains
- PyTorch Dataset and DataLoader wrappers
"""

from src.data.preprocessing import DataPreprocessor
from src.data.dataset import SecurityFlowDataset, TemporalGraphDataset, MultiDomainDataLoader
from src.data.feature_engineering import FeatureHarmonizer, GraphConstructor, TemporalEncoder

__all__ = [
    "DataPreprocessor",
    "SecurityFlowDataset",
    "TemporalGraphDataset",
    "MultiDomainDataLoader",
    "FeatureHarmonizer",
    "GraphConstructor",
    "TemporalEncoder",
]
