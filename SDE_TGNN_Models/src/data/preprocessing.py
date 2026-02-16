"""Data preprocessing pipelines for six network intrusion detection datasets.

Handles loading, cleaning, encoding, normalization, and splitting for:
1. Microsoft Azure Cloud Malware Dataset
2. Edge-IIoTset Federated Learning Dataset
3. Kubernetes vs Docker Container Dataset
4. CSE-CIC-IDS2018 Dataset
5. UNB-CIC-IoT2023 Dataset
6. UNSW-NB15 Dataset
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Unified preprocessing pipeline for multi-domain intrusion detection datasets.

    This class provides methods to load, clean, encode, and normalize each of
    the six target datasets, producing standardized feature matrices and label
    vectors ready for downstream model consumption.

    Attributes:
        data_root: Root directory containing raw dataset folders.
        normalize_method: Normalization strategy ('standard', 'minmax', 'robust').
        test_split: Fraction of data reserved for testing.
        val_split: Fraction of training data reserved for validation.
        random_state: Random seed for reproducibility.
        label_encoders: Dictionary mapping dataset names to fitted LabelEncoders.
        scalers: Dictionary mapping dataset names to fitted feature scalers.
    """

    # Canonical column names used across all datasets after harmonization
    CANONICAL_COLUMNS = [
        "duration", "src_bytes", "dst_bytes", "src_pkts", "dst_pkts",
        "protocol", "src_port", "dst_port", "tcp_flags", "flow_iat_mean",
        "flow_iat_std", "fwd_iat_mean", "bwd_iat_mean", "pkt_len_mean",
        "pkt_len_std", "fin_flag_cnt", "syn_flag_cnt", "rst_flag_cnt",
        "psh_flag_cnt", "ack_flag_cnt", "header_len", "fwd_pkt_len_mean",
        "bwd_pkt_len_mean", "flow_bytes_per_s", "flow_pkts_per_s",
        "down_up_ratio", "avg_pkt_size", "fwd_seg_size_avg",
        "bwd_seg_size_avg", "subflow_fwd_pkts", "subflow_bwd_pkts",
        "init_fwd_win_bytes", "init_bwd_win_bytes",
    ]

    def __init__(
        self,
        data_root: str,
        normalize_method: str = "standard",
        test_split: float = 0.2,
        val_split: float = 0.1,
        random_state: int = 42,
    ) -> None:
        """Initialize the preprocessor.

        Args:
            data_root: Root directory containing raw dataset folders.
            normalize_method: Normalization strategy ('standard', 'minmax', 'robust').
            test_split: Fraction of data reserved for testing.
            val_split: Fraction of training data reserved for validation.
            random_state: Random seed for reproducibility.
        """
        self.data_root = data_root
        self.normalize_method = normalize_method
        self.test_split = test_split
        self.val_split = val_split
        self.random_state = random_state
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scalers: Dict[str, Union[StandardScaler, MinMaxScaler, RobustScaler]] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_scaler(self) -> Union[StandardScaler, MinMaxScaler, RobustScaler]:
        """Return a fresh scaler instance based on the configured method."""
        if self.normalize_method == "minmax":
            return MinMaxScaler()
        if self.normalize_method == "robust":
            return RobustScaler()
        return StandardScaler()

    @staticmethod
    def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Remove NaN, infinity, and duplicate rows from a DataFrame.

        Args:
            df: Raw pandas DataFrame.

        Returns:
            Cleaned DataFrame with no missing or infinite values.
        """
        initial_rows = len(df)
        # Replace infinities with NaN then drop
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        df = df.drop_duplicates()
        removed = initial_rows - len(df)
        if removed > 0:
            logger.info("Removed %d dirty rows (%.2f%%)", removed, 100.0 * removed / max(initial_rows, 1))
        return df.reset_index(drop=True)

    @staticmethod
    def _encode_categorical(
        df: pd.DataFrame,
        columns: List[str],
    ) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
        """Label-encode specified categorical columns in place.

        Args:
            df: DataFrame with categorical columns.
            columns: List of column names to encode.

        Returns:
            Tuple of (encoded DataFrame, dict of column -> fitted LabelEncoder).
        """
        encoders: Dict[str, LabelEncoder] = {}
        for col in columns:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                encoders[col] = le
        return df, encoders

    def _normalize_and_split(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        dataset_name: str,
    ) -> Dict[str, np.ndarray]:
        """Normalize features and split into train / val / test.

        Args:
            features: Feature matrix of shape (N, D).
            labels: Label vector of shape (N,).
            dataset_name: Identifier used to store the fitted scaler.

        Returns:
            Dictionary with keys 'X_train', 'X_val', 'X_test',
            'y_train', 'y_val', 'y_test'.
        """
        # Train / temp split
        X_train, X_temp, y_train, y_temp = train_test_split(
            features, labels,
            test_size=self.test_split + self.val_split,
            random_state=self.random_state,
            stratify=labels,
        )

        # Val / test split from temp
        relative_val = self.val_split / (self.test_split + self.val_split)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=1.0 - relative_val,
            random_state=self.random_state,
            stratify=y_temp,
        )

        # Fit scaler on train only
        scaler = self._get_scaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        self.scalers[dataset_name] = scaler

        logger.info(
            "%s splits  train=%d  val=%d  test=%d",
            dataset_name, len(y_train), len(y_val), len(y_test),
        )

        return {
            "X_train": X_train.astype(np.float32),
            "X_val": X_val.astype(np.float32),
            "X_test": X_test.astype(np.float32),
            "y_train": y_train.astype(np.int64),
            "y_val": y_val.astype(np.int64),
            "y_test": y_test.astype(np.int64),
        }

    # ------------------------------------------------------------------
    # Per-dataset preprocessing
    # ------------------------------------------------------------------

    def preprocess_microsoft_cloud(
        self,
        filename: str = "microsoft_cloud_malware.csv",
    ) -> Dict[str, np.ndarray]:
        """Load and preprocess the Microsoft Azure Cloud Malware dataset.

        Expected CSV columns include a mixture of metadata (machine ID,
        timestamps), categorical features (product name, engine version,
        OS platform), and numeric counters (census fields).  The label
        column is ``HasDetections`` (binary).

        Args:
            filename: CSV filename inside ``data_root/microsoft_cloud/``.

        Returns:
            Dictionary of train / val / test arrays.
        """
        path = os.path.join(self.data_root, "microsoft_cloud", filename)
        logger.info("Loading Microsoft Cloud dataset from %s", path)
        df = pd.read_csv(path, low_memory=False)
        df = self._clean_dataframe(df)

        # Identify label
        label_col = "HasDetections"
        if label_col not in df.columns:
            # Fallback: last column
            label_col = df.columns[-1]

        # Separate features and labels
        feature_cols = [c for c in df.columns if c != label_col]

        # Detect categorical columns
        cat_cols = [c for c in feature_cols if df[c].dtype == object]
        df, cat_encoders = self._encode_categorical(df, cat_cols)
        self.label_encoders["microsoft_cloud_cat"] = cat_encoders  # type: ignore[assignment]

        # Encode label
        le = LabelEncoder()
        labels = le.fit_transform(df[label_col].values)
        self.label_encoders["microsoft_cloud"] = le

        features = df[feature_cols].values.astype(np.float64)
        return self._normalize_and_split(features, labels, "microsoft_cloud")

    def preprocess_edge_iiot(
        self,
        filename: str = "edge_iiot.csv",
    ) -> Dict[str, np.ndarray]:
        """Load and preprocess the Edge-IIoTset Federated Learning dataset.

        The dataset covers 14 IoT/IIoT attack types captured from real
        edge and fog layer devices.  Typical columns include frame-level
        and flow-level statistics.  The label column is ``Attack_type``.

        Args:
            filename: CSV filename inside ``data_root/edge_iiot/``.

        Returns:
            Dictionary of train / val / test arrays.
        """
        path = os.path.join(self.data_root, "edge_iiot", filename)
        logger.info("Loading Edge-IIoTset dataset from %s", path)
        df = pd.read_csv(path, low_memory=False)
        df = self._clean_dataframe(df)

        label_col = "Attack_type"
        if label_col not in df.columns:
            label_col = df.columns[-1]

        feature_cols = [c for c in df.columns if c != label_col]

        cat_cols = [c for c in feature_cols if df[c].dtype == object]
        df, _ = self._encode_categorical(df, cat_cols)

        le = LabelEncoder()
        labels = le.fit_transform(df[label_col].values)
        self.label_encoders["edge_iiot"] = le

        features = df[feature_cols].values.astype(np.float64)
        return self._normalize_and_split(features, labels, "edge_iiot")

    def preprocess_kubernetes_docker(
        self,
        filename: str = "kubernetes_docker.csv",
    ) -> Dict[str, np.ndarray]:
        """Load and preprocess the Kubernetes vs Docker Container dataset.

        This dataset contrasts normal container orchestration traffic
        against various attack patterns observed in Kubernetes and
        Docker environments.  The label column is ``label``.

        Args:
            filename: CSV filename inside ``data_root/kubernetes_docker/``.

        Returns:
            Dictionary of train / val / test arrays.
        """
        path = os.path.join(self.data_root, "kubernetes_docker", filename)
        logger.info("Loading Kubernetes/Docker dataset from %s", path)
        df = pd.read_csv(path, low_memory=False)
        df = self._clean_dataframe(df)

        label_col = "label"
        if label_col not in df.columns:
            label_col = df.columns[-1]

        feature_cols = [c for c in df.columns if c != label_col]

        cat_cols = [c for c in feature_cols if df[c].dtype == object]
        df, _ = self._encode_categorical(df, cat_cols)

        le = LabelEncoder()
        labels = le.fit_transform(df[label_col].values)
        self.label_encoders["kubernetes_docker"] = le

        features = df[feature_cols].values.astype(np.float64)
        return self._normalize_and_split(features, labels, "kubernetes_docker")

    def preprocess_cic_ids2018(
        self,
        filename: str = "cic_ids2018.csv",
    ) -> Dict[str, np.ndarray]:
        """Load and preprocess the CSE-CIC-IDS2018 dataset.

        CIC-IDS2018 contains 80+ bidirectional flow features extracted
        via CICFlowMeter.  Attack categories include brute-force, DoS,
        DDoS, web attacks, infiltration, and botnet.  The label column
        is ``Label``.

        Args:
            filename: CSV filename inside ``data_root/cic_ids2018/``.

        Returns:
            Dictionary of train / val / test arrays.
        """
        path = os.path.join(self.data_root, "cic_ids2018", filename)
        logger.info("Loading CIC-IDS2018 dataset from %s", path)
        df = pd.read_csv(path, low_memory=False)

        # Strip whitespace from column names (CIC convention)
        df.columns = df.columns.str.strip()
        df = self._clean_dataframe(df)

        label_col = "Label"
        if label_col not in df.columns:
            label_col = df.columns[-1]

        # Drop non-numeric metadata if present
        drop_cols = ["Flow ID", "Src IP", "Dst IP", "Timestamp"]
        feature_cols = [c for c in df.columns if c != label_col and c not in drop_cols]

        cat_cols = [c for c in feature_cols if df[c].dtype == object]
        df, _ = self._encode_categorical(df, cat_cols)

        le = LabelEncoder()
        labels = le.fit_transform(df[label_col].values)
        self.label_encoders["cic_ids2018"] = le

        features = df[feature_cols].values.astype(np.float64)
        return self._normalize_and_split(features, labels, "cic_ids2018")

    def preprocess_cic_iot2023(
        self,
        filename: str = "cic_iot2023.csv",
    ) -> Dict[str, np.ndarray]:
        """Load and preprocess the UNB-CIC-IoT2023 dataset.

        CIC-IoT2023 provides 46 extracted features from IoT device
        traffic, covering 33 attack types organized in 7 categories
        (DDoS, DoS, Recon, Web, BruteForce, Spoofing, Mirai).  The
        label column is ``label``.

        Args:
            filename: CSV filename inside ``data_root/cic_iot2023/``.

        Returns:
            Dictionary of train / val / test arrays.
        """
        path = os.path.join(self.data_root, "cic_iot2023", filename)
        logger.info("Loading CIC-IoT2023 dataset from %s", path)
        df = pd.read_csv(path, low_memory=False)
        df.columns = df.columns.str.strip()
        df = self._clean_dataframe(df)

        label_col = "label"
        if label_col not in df.columns:
            label_col = df.columns[-1]

        drop_cols = ["flow_id", "src_ip", "dst_ip", "timestamp"]
        feature_cols = [
            c for c in df.columns
            if c != label_col and c.lower() not in [d.lower() for d in drop_cols]
        ]

        cat_cols = [c for c in feature_cols if df[c].dtype == object]
        df, _ = self._encode_categorical(df, cat_cols)

        le = LabelEncoder()
        labels = le.fit_transform(df[label_col].values)
        self.label_encoders["cic_iot2023"] = le

        features = df[feature_cols].values.astype(np.float64)
        return self._normalize_and_split(features, labels, "cic_iot2023")

    def preprocess_unsw_nb15(
        self,
        filename: str = "unsw_nb15.csv",
    ) -> Dict[str, np.ndarray]:
        """Load and preprocess the UNSW-NB15 dataset.

        UNSW-NB15 features 49 flow-level attributes and nine attack
        categories (Fuzzers, Analysis, Backdoors, DoS, Exploits,
        Generic, Reconnaissance, Shellcode, Worms).  The label column
        is ``attack_cat`` (multi-class) or ``label`` (binary).

        Args:
            filename: CSV filename inside ``data_root/unsw_nb15/``.

        Returns:
            Dictionary of train / val / test arrays.
        """
        path = os.path.join(self.data_root, "unsw_nb15", filename)
        logger.info("Loading UNSW-NB15 dataset from %s", path)
        df = pd.read_csv(path, low_memory=False)
        df = self._clean_dataframe(df)

        # Use multi-class label when available
        if "attack_cat" in df.columns:
            label_col = "attack_cat"
        elif "label" in df.columns:
            label_col = "label"
        else:
            label_col = df.columns[-1]

        drop_cols = ["id", "label"] if label_col == "attack_cat" else ["id"]
        feature_cols = [
            c for c in df.columns
            if c != label_col and c not in drop_cols
        ]

        cat_cols = [c for c in feature_cols if df[c].dtype == object]
        df, _ = self._encode_categorical(df, cat_cols)

        le = LabelEncoder()
        labels = le.fit_transform(df[label_col].values)
        self.label_encoders["unsw_nb15"] = le

        features = df[feature_cols].values.astype(np.float64)
        return self._normalize_and_split(features, labels, "unsw_nb15")

    # ------------------------------------------------------------------
    # Cross-domain harmonization
    # ------------------------------------------------------------------

    def harmonize_features(
        self,
        datasets: Dict[str, Dict[str, np.ndarray]],
        target_dim: Optional[int] = None,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Standardize feature dimensions across all domains.

        When feature counts differ between datasets, this method pads
        smaller feature sets with zeros (or truncates larger ones) so
        that every dataset shares the same dimensionality.

        Args:
            datasets: Mapping of dataset_name -> split arrays from
                the individual ``preprocess_*`` methods.
            target_dim: Desired unified feature dimension.  If *None*,
                the maximum dimension across datasets is used.

        Returns:
            Updated datasets dictionary with aligned feature matrices.
        """
        if target_dim is None:
            target_dim = max(
                d["X_train"].shape[1] for d in datasets.values()
            )

        harmonized: Dict[str, Dict[str, np.ndarray]] = {}
        for name, splits in datasets.items():
            harmonized[name] = {}
            for key in ("X_train", "X_val", "X_test"):
                arr = splits[key]
                current_dim = arr.shape[1]
                if current_dim < target_dim:
                    padding = np.zeros(
                        (arr.shape[0], target_dim - current_dim),
                        dtype=arr.dtype,
                    )
                    arr = np.concatenate([arr, padding], axis=1)
                elif current_dim > target_dim:
                    arr = arr[:, :target_dim]
                harmonized[name][key] = arr

            # Labels pass through unchanged
            for key in ("y_train", "y_val", "y_test"):
                harmonized[name][key] = splits[key]

            logger.info(
                "Harmonized %s: %d -> %d features",
                name, splits["X_train"].shape[1], target_dim,
            )

        return harmonized

    # ------------------------------------------------------------------
    # Convenience: preprocess all datasets
    # ------------------------------------------------------------------

    def preprocess_all(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Run preprocessing for all six datasets and harmonize features.

        Returns:
            Dictionary mapping dataset name -> split arrays.
        """
        datasets: Dict[str, Dict[str, np.ndarray]] = {}

        loaders = {
            "microsoft_cloud": self.preprocess_microsoft_cloud,
            "edge_iiot": self.preprocess_edge_iiot,
            "kubernetes_docker": self.preprocess_kubernetes_docker,
            "cic_ids2018": self.preprocess_cic_ids2018,
            "cic_iot2023": self.preprocess_cic_iot2023,
            "unsw_nb15": self.preprocess_unsw_nb15,
        }

        for name, loader_fn in loaders.items():
            try:
                datasets[name] = loader_fn()
            except FileNotFoundError:
                logger.warning("Dataset %s not found, skipping.", name)
            except Exception as exc:
                logger.error("Error preprocessing %s: %s", name, exc)

        if datasets:
            datasets = self.harmonize_features(datasets)

        return datasets
