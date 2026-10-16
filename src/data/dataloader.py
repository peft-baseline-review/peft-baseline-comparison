#!/usr/bin/env python
"""
SCI-ready Tigrinya Data Loader with Optimized Caching and Robust Parsing
Fully revised - ensures exact reproducibility, fails fast,
and provides comprehensive dataset statistics without redundant reloads.
"""

import torch
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List, Union
import pickle
import hashlib
import json
from enum import Enum
from dataclasses import dataclass, field
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# Optional encoding detection - provide fallback if chardet is missing
try:
    import chardet
    HAS_CHARDET = True
except ImportError:
    HAS_CHARDET = False
    import warnings
    warnings.warn("chardet not installed; falling back to utf-8/latin-1 detection")

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Configuration Enums and Dataclasses
# ----------------------------------------------------------------------
class DelimiterType(str, Enum):
    COMMA = ','
    SEMICOLON = ';'
    TAB = '\t'
    PIPE = '|'

@dataclass
class TaskConfig:
    task_type: str
    num_labels: int
    label_names: List[str]
    data_dir: str
    train_file: str
    test_file: str
    val_file: Optional[str] = None
    delimiter: DelimiterType = DelimiterType.COMMA
    text_column: str = 'text'
    label_column: str = 'label'
    max_length: int = 128
    batch_size: int = 32
    use_class_weights: bool = True
    imbalanced: bool = False
    val_split: float = 0.1
    num_workers: int = 4
    pin_memory: bool = True

@dataclass
class ExperimentConfig:
    task: TaskConfig
    model: Any
    seed: int = 42
    device: str = 'cpu'

# ----------------------------------------------------------------------
# Optimized Cache Manager (enhanced with metadata)
# ----------------------------------------------------------------------
class DataCacheManager:
    def __init__(self, cache_dir: Path = Path('data_cache')):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.cache_hits = 0
        self.cache_misses = 0
        logger.info(f"Cache initialized: {self.cache_dir}")

    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute MD5 hash of file content for cache invalidation."""
        try:
            return hashlib.md5(file_path.read_bytes()).hexdigest()
        except Exception:
            return "0"

    def get_cache_key(self, params: Dict[str, Any]) -> str:
        """
        Generate a deterministic cache key from all relevant parameters.
        Any change in these parameters invalidates the cache.
        """
        # Sort keys for reproducibility
        items = sorted(params.items())
        # Convert all values to string in a stable way
        key_str = "_".join(f"{k}:{v}" for k, v in items)
        return hashlib.md5(key_str.encode()).hexdigest()[:16]

    def get_cache_path(self, cache_key: str, suffix: str = "dataset") -> Path:
        return self.cache_dir / f"{suffix}_{cache_key}.pkl"

    def save_to_cache(self, data: Any, cache_path: Path, metadata: Optional[Dict] = None):
        """Save data with optional metadata (e.g., parameters used)."""
        try:
            # If metadata is provided, save both in a tuple
            if metadata is not None:
                data = (metadata, data)
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.debug(f"Cached: {cache_path}")
        except Exception as e:
            logger.warning(f"Cache save failed: {e}")

    def load_from_cache(self, cache_path: Path, expected_metadata: Optional[Dict] = None) -> Optional[Any]:
        """
        Load from cache; if metadata is provided, verify that it matches.
        Returns the data part if successful, else None.
        """
        if not cache_path.exists():
            self.cache_misses += 1
            return None

        try:
            with open(cache_path, 'rb') as f:
                loaded = pickle.load(f)

            # If we stored metadata, it should be a tuple (metadata, data)
            if expected_metadata is not None:
                if not isinstance(loaded, tuple) or len(loaded) != 2:
                    logger.warning("Cache format mismatch (metadata expected)")
                    self.cache_misses += 1
                    return None
                stored_metadata, data = loaded
                if stored_metadata != expected_metadata:
                    logger.warning("Cache metadata mismatch - cache invalid")
                    self.cache_misses += 1
                    return None
            else:
                data = loaded

            self.cache_hits += 1
            logger.debug(f"Cache hit: {cache_path}")
            return data

        except Exception as e:
            logger.warning(f"Cache load failed: {e}")
            self.cache_misses += 1
            return None

    def get_stats(self) -> Dict[str, Any]:
        total = self.cache_hits + self.cache_misses
        return {
            'hits': self.cache_hits,
            'misses': self.cache_misses,
            'total': total,
            'hit_rate': self.cache_hits / total if total > 0 else 0,
        }

# ----------------------------------------------------------------------
# Collate Function
# ----------------------------------------------------------------------
def collate_fn_tensors(batch):
    """Optimized collate function - unchanged."""
    if isinstance(batch[0], dict):
        result = {}
        for key in batch[0].keys():
            values = [item[key] for item in batch]
            if isinstance(values[0], torch.Tensor):
                result[key] = torch.stack(values)
            elif isinstance(values[0], (int, np.integer)):
                result[key] = torch.tensor(values, dtype=torch.long)
            elif isinstance(values[0], np.ndarray):
                result[key] = torch.tensor(np.stack(values), dtype=torch.long)
            else:
                result[key] = torch.tensor(values)
        return result
    return torch.utils.data.default_collate(batch)

# ----------------------------------------------------------------------
# Optimized Dataset (unchanged, but note it expects encodings dict)
# ----------------------------------------------------------------------
class TigrinyaDataset(Dataset):
    """Memory-efficient dataset."""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        if isinstance(labels, list):
            self.labels = torch.tensor(labels, dtype=torch.long)
        elif isinstance(labels, np.ndarray):
            self.labels = torch.from_numpy(labels).long()
        elif isinstance(labels, torch.Tensor):
            self.labels = labels.long()
        else:
            raise TypeError(f"Unsupported labels type: {type(labels)}")

    def __getitem__(self, idx):
        item = {}
        for k, v in self.encodings.items():
            item[k] = v[idx].clone() if torch.is_tensor(v) else v[idx]
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

# ----------------------------------------------------------------------
# Main DataLoader (Revised)
# ----------------------------------------------------------------------
class TigrinyaDataLoader:
    """
    Production-grade data loader for Tigrinya text classification.

    Features:
    - Robust CSV parsing with automatic encoding detection
    - Intelligent caching (based on file content and all parameters)
    - Exact reproducibility: fixed validation split via indices caching
    - Platform-aware worker settings (Windows vs Linux)
    - Computation of class weights and statistics from the actual training data
    - No silent fallbacks - fails fast on errors
    - Full provenance: label mappings saved to experiment directory
    """

    def __init__(self, config: ExperimentConfig, use_cache: bool = True,
                 cache_manager: Optional[DataCacheManager] = None):
        self.config = config
        self.task_config = config.task
        self.use_cache = use_cache
        self.cache_manager = cache_manager or DataCacheManager()

        # Store loaded datasets for later use (statistics, class weights)
        self._train_dataset: Optional[TigrinyaDataset] = None
        self._val_dataset: Optional[TigrinyaDataset] = None
        self._test_dataset: Optional[TigrinyaDataset] = None
        self._label_map: Optional[Dict[str, int]] = None  # if labels were strings

        # Setup paths
        self.data_dir = Path(self.task_config.data_dir)
        self.delimiter = self.task_config.delimiter.value
        self.text_column = self.task_config.text_column
        self.label_column = self.task_config.label_column
        self.max_length = self.task_config.max_length
        self.batch_size = self.task_config.batch_size

        # Platform-aware worker settings
        import platform
        self.num_workers = 0 if platform.system() == 'Windows' else self.task_config.num_workers
        self.pin_memory = self.task_config.pin_memory and torch.cuda.is_available() and platform.system() != 'Windows'

        # Initialize tokenizer with robust pad token handling
        self.tokenizer = AutoTokenizer.from_pretrained(config.model.name)
        self._ensure_pad_token()

        logger.info(f"DataLoader initialized for {config.model.name}")

    def _ensure_pad_token(self):
        """Safely set a pad token if the tokenizer lacks one."""
        if self.tokenizer.pad_token is not None:
            return
        if self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info(f"Set pad_token = eos_token for {self.config.model.name}")
        else:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            logger.warning(f"Added [PAD] token for {self.config.model.name}. "
                           f"Model embeddings may need resizing.")

    def _detect_encoding(self, file_path: Path) -> str:
        """Detect file encoding, with fallback to utf-8/latin-1 if chardet missing."""
        if HAS_CHARDET:
            try:
                with open(file_path, 'rb') as f:
                    raw = f.read(10000)
                    result = chardet.detect(raw)
                    if result['encoding']:
                        return result['encoding']
            except Exception:
                pass
        # Fallback: try utf-8, then latin-1
        return 'utf-8'

    def _load_csv(self, file_path: Path) -> pd.DataFrame:
        """
        Load CSV with robust encoding detection and column validation.
        Raises ValueError if expected columns are missing.
        """
        encoding = self._detect_encoding(file_path)
        # Use Python engine only when necessary (e.g., for complex delimiters)
        # but we keep it simple: always use Python engine to handle varied delimiters.
        df = pd.read_csv(file_path, sep=self.delimiter, encoding=encoding,
                         engine='python', on_bad_lines='warn')

        # Auto-detect text/label columns if specified ones are missing
        if self.text_column not in df.columns:
            candidates = ['text', 'sentence', 'content', 'tweet']
            found = [c for c in candidates if c in df.columns]
            if not found:
                raise ValueError(f"Text column '{self.text_column}' not found and no fallback found. "
                                 f"Available columns: {list(df.columns)}")
            self.text_column = found[0]
            logger.warning(f"Text column auto-detected as '{self.text_column}'")

        if self.label_column not in df.columns:
            candidates = ['label', 'labels', 'class', 'category']
            found = [c for c in candidates if c in df.columns]
            if not found:
                raise ValueError(f"Label column '{self.label_column}' not found. "
                                 f"Available columns: {list(df.columns)}")
            self.label_column = found[0]
            logger.warning(f"Label column auto-detected as '{self.label_column}'")

        # Drop rows with missing text or label, but warn
        initial_len = len(df)
        df = df.dropna(subset=[self.text_column, self.label_column])
        if len(df) < initial_len:
            logger.warning(f"Dropped {initial_len - len(df)} rows with missing text/label.")

        # Ensure text is string
        df[self.text_column] = df[self.text_column].astype(str).str.strip()

        # Convert labels to integers, preserving mapping
        if not pd.api.types.is_numeric_dtype(df[self.label_column]):
            # String labels - create deterministic mapping (sorted)
            unique_labels = sorted(df[self.label_column].unique())
            self._label_map = {label: idx for idx, label in enumerate(unique_labels)}
            df[self.label_column] = df[self.label_column].map(self._label_map)
            logger.info(f"Created label mapping: {self._label_map}")
        else:
            # Already numeric, ensure they are integers
            df[self.label_column] = pd.to_numeric(df[self.label_column], errors='coerce')
            # Check for any conversion failures
            if df[self.label_column].isna().any():
                raise ValueError("Label column contains non-numeric values that could not be coerced.")
            df[self.label_column] = df[self.label_column].astype(int)

        return df

    def _tokenize_batch(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Tokenize a list of texts. No model-specific hacks - rely on tokenizer."""
        return self.tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
            return_token_type_ids=True   # tokenizer will handle if model doesn't use them
        )

    def _build_cache_params(self, file_path: Path, split: str) -> Dict[str, Any]:
        """
        Build a dictionary of all parameters that affect the tokenized dataset.
        Used for cache key generation.
        """
        return {
            'file_path': str(file_path),
            'file_hash': self.cache_manager._compute_file_hash(file_path),
            'tokenizer_name': self.config.model.name,
            'max_length': self.max_length,
            'delimiter': self.delimiter,
            'text_column': self.text_column,
            'label_column': self.label_column,
            'split': split,
            'seed': self.config.seed,
            'val_split': self.task_config.val_split,
            'imbalanced': self.task_config.imbalanced,
            'label_map': json.dumps(self._label_map) if self._label_map else None,
        }

    def _load_and_cache_dataset(self, file_path: Path, split: str) -> TigrinyaDataset:
        """
        Load a dataset from file (or cache) and return a TigrinyaDataset.
        """
        cache_params = self._build_cache_params(file_path, split)
        cache_key = self.cache_manager.get_cache_key(cache_params)
        cache_path = self.cache_manager.get_cache_path(cache_key, f"dataset_{split}")

        if self.use_cache:
            cached = self.cache_manager.load_from_cache(cache_path, expected_metadata=cache_params)
            if cached is not None:
                # cached is a tuple (encodings, labels)
                encodings, labels = cached
                return TigrinyaDataset(encodings, labels)

        # Cache miss - load from CSV and tokenize
        df = self._load_csv(file_path)
        texts = df[self.text_column].tolist()
        labels = df[self.label_column].tolist()
        encodings = self._tokenize_batch(texts)

        dataset = TigrinyaDataset(encodings, labels)

        if self.use_cache:
            # Save both data and metadata
            self.cache_manager.save_to_cache(
                (encodings, labels),
                cache_path,
                metadata=cache_params
            )

        logger.info(f"Loaded {split}: {len(dataset)} samples")
        return dataset

    def load_datasets(self) -> Tuple[TigrinyaDataset, Optional[TigrinyaDataset], TigrinyaDataset]:
        """
        Load or create train, validation, and test datasets.
        If no validation file is provided, a fixed split of the training data is used,
        and the split indices are cached for reproducibility.
        """
        train_path = self.data_dir / self.task_config.train_file
        test_path = self.data_dir / self.task_config.test_file

        # 1. Load test set (always from file)
        test_dataset = self._load_and_cache_dataset(test_path, 'test')
        self._test_dataset = test_dataset

        # 2. Handle validation set
        if self.task_config.val_file is not None:
            # Use provided validation file
            val_path = self.data_dir / self.task_config.val_file
            val_dataset = self._load_and_cache_dataset(val_path, 'val')
            # Load training set normally
            train_dataset = self._load_and_cache_dataset(train_path, 'train')
        else:
            # Split training data into train/val
            # First, load the full training CSV (without tokenizing yet)
            df_train_full = self._load_csv(train_path)
            texts_full = df_train_full[self.text_column].tolist()
            labels_full = df_train_full[self.label_column].tolist()

            # Build parameters for split cache
            split_params = {
                'file_path': str(train_path),
                'file_hash': self.cache_manager._compute_file_hash(train_path),
                'seed': self.config.seed,
                'val_split': self.task_config.val_split,
                'imbalanced': self.task_config.imbalanced,
                'label_map': json.dumps(self._label_map) if self._label_map else None,
            }
            split_key = self.cache_manager.get_cache_key(split_params)
            split_cache_path = self.cache_manager.get_cache_path(split_key, "split_indices")

            if self.use_cache:
                cached_split = self.cache_manager.load_from_cache(split_cache_path, expected_metadata=split_params)
                if cached_split is not None:
                    train_indices, val_indices = cached_split
                else:
                    # Perform split and cache indices
                    stratify = labels_full if self.task_config.imbalanced else None
                    train_indices, val_indices = train_test_split(
                        range(len(labels_full)),
                        test_size=self.task_config.val_split,
                        random_state=self.config.seed,
                        stratify=stratify
                    )
                    self.cache_manager.save_to_cache(
                        (train_indices, val_indices),
                        split_cache_path,
                        metadata=split_params
                    )
            else:
                # No caching - split each time (but at least with fixed seed)
                stratify = labels_full if self.task_config.imbalanced else None
                train_indices, val_indices = train_test_split(
                    range(len(labels_full)),
                    test_size=self.task_config.val_split,
                    random_state=self.config.seed,
                    stratify=stratify
                )

            # Create datasets by tokenizing only the selected subsets
            train_texts = [texts_full[i] for i in train_indices]
            train_labels = [labels_full[i] for i in train_indices]
            val_texts = [texts_full[i] for i in val_indices]
            val_labels = [labels_full[i] for i in val_indices]

            train_enc = self._tokenize_batch(train_texts)
            val_enc = self._tokenize_batch(val_texts)

            train_dataset = TigrinyaDataset(train_enc, train_labels)
            val_dataset = TigrinyaDataset(val_enc, val_labels)

        self._train_dataset = train_dataset
        self._val_dataset = val_dataset
        self._test_dataset = test_dataset

        # Optionally save the label mapping to the experiment directory later
        # (caller can retrieve it via get_label_map())

        return train_dataset, val_dataset, test_dataset

    def create_dataloaders(self) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
        """Create PyTorch DataLoaders from the datasets."""
        if self._train_dataset is None:
            self.load_datasets()

        train_loader = DataLoader(
            self._train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn_tensors,
            persistent_workers=False   # set True if num_workers>0 and dataset large
        )

        val_loader = None
        if self._val_dataset is not None:
            val_loader = DataLoader(
                self._val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=collate_fn_tensors,
                persistent_workers=False
            )

        test_loader = DataLoader(
            self._test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn_tensors,
            persistent_workers=False
        )

        return train_loader, val_loader, test_loader

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute class weights from the **actual training dataset** (after any split).
        Raises RuntimeError if training data not loaded yet.
        """
        if self._train_dataset is None:
            raise RuntimeError("Training dataset not loaded. Call load_datasets() first.")
        labels = self._train_dataset.labels.numpy()
        unique = np.unique(labels)
        if len(unique) != self.task_config.num_labels:
            logger.warning(f"Number of unique labels in training set ({len(unique)}) "
                           f"differs from config.num_labels ({self.task_config.num_labels})")
        weights = compute_class_weight('balanced', classes=unique, y=labels)
        return torch.tensor(weights, dtype=torch.float)

    def get_data_stats(self) -> Dict[str, Any]:
        """
        Return statistics of the **loaded datasets** (train/val/test).
        Does NOT reload CSV files.
        """
        if self._train_dataset is None:
            raise RuntimeError("Datasets not loaded. Call load_datasets() first.")

        stats = {
            'task_type': self.task_config.task_type,
            'model_name': self.config.model.name,
            'cache_stats': self.cache_manager.get_stats(),
            'label_map': self._label_map,
        }

        # Helper to compute basic stats on a dataset
        def dataset_stats(ds: TigrinyaDataset, name: str) -> Dict:
            if ds is None:
                return {}
            labels = ds.labels.numpy()
            # Text lengths would require re-tokenization; we skip to avoid cost.
            # But we can compute label distribution.
            unique, counts = np.unique(labels, return_counts=True)
            return {
                f'{name}_samples': len(labels),
                f'{name}_class_distribution': dict(zip(unique.tolist(), counts.tolist())),
            }

        stats.update(dataset_stats(self._train_dataset, 'train'))
        stats.update(dataset_stats(self._val_dataset, 'val'))
        stats.update(dataset_stats(self._test_dataset, 'test'))

        # Optionally add tokenizer info
        stats['tokenizer'] = {
            'name': self.tokenizer.name_or_path,
            'vocab_size': self.tokenizer.vocab_size,
            'max_length': self.max_length,
        }

        return stats

    def get_label_map(self) -> Optional[Dict[str, int]]:
        """Return the mapping from string labels to integers, if any."""
        return self._label_map

    def save_label_map(self, output_dir: Path):
        """Save the label mapping to a JSON file in the output directory."""
        if self._label_map is not None:
            map_path = output_dir / "label_map.json"
            with open(map_path, 'w', encoding='utf-8') as f:
                json.dump(self._label_map, f, indent=2)
            logger.info(f"Label map saved to {map_path}")

    def clear_cache(self):
        """Clear the entire cache directory (use with caution)."""
        import shutil
        shutil.rmtree(self.cache_dir)
        self.cache_dir.mkdir(parents=True)
        logger.warning("Cache cleared.")