"""
Scientific Evaluator for BOPET Experiments v2.3 - Optimized & DRY
Complete evaluation with statistical rigor, power analysis, and inference metrics
"""

from matplotlib import pyplot as plt
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from sklearn.metrics import balanced_accuracy_score
import torch
import torch.nn as nn
import logging
from pathlib import Path
from datetime import datetime
import time
import psutil
import gc
from scipy import stats
from dataclasses import dataclass
import json

from .energy_meter import EnergyMeter   # adjust import path as needed

# Import sklearn metrics with fallback
try:
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score,
        recall_score, confusion_matrix, classification_report,
        cohen_kappa_score, matthews_corrcoef
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available - limited metrics")

# Configuration imports with fallback
try:
    from config.experiment import ExperimentConfig
    from config.evaluation import EvaluationConfig
except ImportError:
    @dataclass
    class EvaluationConfig:
        metrics: List[str] = None
        confidence_level: float = 0.95
        compute_ci: bool = True
        bootstrap_samples: int = 1000
        composite_alpha: float = 0.01
        power_analysis_alpha: float = 0.05
        power_threshold: float = 0.8
        inference_batch_size: int = 1
        num_inference_runs: int = 100

    @dataclass
    class ExperimentConfig:
        evaluation: EvaluationConfig = EvaluationConfig()
        task: Any = None

logger = logging.getLogger(__name__)


class ScientificEvaluator:
    """
    Comprehensive evaluator for PEFT experiments v2.3 (optimized) with:
    - Statistical metrics with confidence intervals for ALL metrics
    - Per-class power analysis and statistical power calculations
    - FLOPs/MACs computation with theoretical/actual measurements
    - Inference time, memory, and energy/power metrics
    - Scientifically rigorous composite scoring
    - Publication-ready reports
    """

    def __init__(self, config: ExperimentConfig):
        """Initialize with experiment configuration"""
        self.config = config
        self.evaluation_config = config.evaluation
        self.rng = np.random.default_rng(42)

        # Set evaluation parameters
        self.metrics = self.evaluation_config.metrics or [
            'accuracy', 'f1_weighted', 'f1_macro',
            'precision_macro', 'recall_macro', 'cohen_kappa', 'mcc'
        ]
        self.confidence_level = self.evaluation_config.confidence_level
        self.bootstrap_samples = self.evaluation_config.bootstrap_samples
        self.power_alpha = self.evaluation_config.power_analysis_alpha
        self.power_threshold = self.evaluation_config.power_threshold

        self.carbon_intensity = getattr(self.evaluation_config, 'carbon_intensity_kg_per_kwh', 0.233)
        self.energy_per_flop_pj = getattr(self.evaluation_config, 'energy_per_flop_pj', 20.0)
        self.memory_overhead = getattr(self.evaluation_config, 'memory_overhead_factor', 2.5)
        self.enable_power_measurement = getattr(self.evaluation_config, 'enable_power_measurement', True)
        self.energy_meter = None
        self._sample_batch = None   # optional: store a batch for later

        logger.info(f"🧪 Evaluator v2.3 initialized with {len(self.metrics)} metrics")

    # ----------------------------------------------------------------------
    # Helper functions (DRY)
    # ----------------------------------------------------------------------
    def _safe_get(self, d: Dict, *keys, default='N/A'):
        """Safely traverse nested dict."""
        for key in keys:
            if isinstance(d, dict) and key in d:
                d = d[key]
            else:
                return default
        return d

    def _safe_format(self, value, format_spec='.2f'):
        """Safely format numeric values."""
        if value == 'N/A' or value is None:
            return 'N/A'
        if isinstance(value, (int, float, np.number)):
            try:
                return format(value, format_spec)
            except (ValueError, TypeError):
                return str(value)
        return str(value)

    def _sanitize_inputs(self, y_true, y_pred):
        """Atomic input normalization."""
        if y_true is None or y_pred is None:
            return [], []

        if torch.is_tensor(y_true):
            y_true = y_true.detach().cpu().numpy()
        if torch.is_tensor(y_pred):
            y_pred = y_pred.detach().cpu().numpy()

        y_true = np.asarray(y_true).reshape(-1)
        y_pred = np.asarray(y_pred).reshape(-1)

        if len(y_true) != len(y_pred):
            min_len = min(len(y_true), len(y_pred))
            y_true = y_true[:min_len]
            y_pred = y_pred[:min_len]

        return y_true.astype(np.int64), y_pred.astype(np.int64)

    def _compute_core_metrics(self, y_true, y_pred):
        """Compute all classification metrics at once."""
        if not SKLEARN_AVAILABLE:
            return {'accuracy': float(np.mean(y_true == y_pred)) if len(y_true) else 0.0}

        metrics = {}
        try:
            metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
            metrics['f1_weighted'] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
            metrics['f1_macro'] = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
            metrics['precision_weighted'] = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
            metrics['recall_weighted'] = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
            metrics['precision_macro'] = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
            metrics['recall_macro'] = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
            metrics['cohen_kappa'] = float(cohen_kappa_score(y_true, y_pred))
            metrics['mcc'] = float(matthews_corrcoef(y_true, y_pred))
            metrics['balanced_accuracy'] = float(balanced_accuracy_score(y_true, y_pred))
        except Exception as e:
            logger.warning(f"Core metric computation failed: {e}")
        return metrics

    # ----------------------------------------------------------------------
    # Main evaluation
    # ----------------------------------------------------------------------
    def evaluate_classification(
        self,
        y_true,
        y_pred,
        y_prob=None,
        label_names=None,
        task_name="classification",
        inference_metrics: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Enhanced classification evaluation with statistical rigor.
        Fully atomic — NO model forward passes allowed.
        """
        # Sanitize
        try:
            y_true, y_pred = self._sanitize_inputs(y_true, y_pred)
        except Exception as e:
            logger.warning(f"Sanitize inputs failed: {e}")
            return {}

        if y_true is None or y_pred is None or len(y_true) == 0:
            logger.warning("Empty evaluation inputs received.")
            return {}

        metrics = self._compute_core_metrics(y_true, y_pred)

        # Calibration metrics
        if y_prob is not None:
            try:
                calibration_metrics = self._compute_calibration_metrics(y_true, y_prob)
                metrics.update(calibration_metrics)
                ace_metrics = self._compute_adaptive_calibration_error(y_true, y_prob)
                metrics.update(ace_metrics)
            except Exception as e:
                logger.warning(f"Calibration metrics failed: {e}")

        # Bootstrap confidence intervals
        if self.evaluation_config.compute_ci and len(y_true) >= 10:
            ci_metrics = self._bootstrap_confidence_intervals_comprehensive(y_true, y_pred)
            metrics.update(ci_metrics)

        # Statistical power analysis
        if len(set(y_true)) > 1:
            power_metrics = self._compute_per_class_power_analysis(y_true, y_pred)
            metrics.update(power_metrics)

        # Detailed reports
        if label_names and len(set(y_true)) > 1:
            try:
                report = classification_report(
                    y_true, y_pred,
                    target_names=label_names,
                    output_dict=True,
                    zero_division=0
                )
                metrics['classification_report'] = report

                cm = confusion_matrix(y_true, y_pred)
                metrics['confusion_matrix'] = cm.tolist()
                row_sums = cm.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1
                metrics['confusion_matrix_normalized'] = (cm / row_sums).tolist()
            except Exception as e:
                logger.warning(f"Detailed reporting failed: {e}")

        # Statistical significance
        if len(y_true) >= 30:
            significance_metrics = self._compute_statistical_significance_comprehensive(y_true, y_pred)
            metrics.update(significance_metrics)

        # Attach inference metrics if provided
        if inference_metrics is not None:
            metrics["inference_metrics"] = inference_metrics

        # Defaults
        metrics.setdefault("accuracy", 0.0)
        metrics.setdefault("f1_weighted", 0.0)

        return metrics

    # ----------------------------------------------------------------------
    # Calibration helpers
    # ----------------------------------------------------------------------
    def _compute_calibration_metrics(self, y_true, y_prob, n_bins: int = 15) -> Dict[str, float]:
        """Compute ECE, MCE, Brier score."""
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)

        if y_prob.ndim != 2:
            raise ValueError("y_prob must be shape (N, num_classes)")

        n_samples = y_prob.shape[0]
        confidences = np.max(y_prob, axis=1)
        predictions = np.argmax(y_prob, axis=1)
        correctness = (predictions == y_true).astype(float)

        bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        mce = 0.0
        bin_acc, bin_conf, bin_counts = [], [], []

        for i in range(n_bins):
            lower, upper = bin_boundaries[i], bin_boundaries[i + 1]
            mask = (confidences > lower) & (confidences <= upper)
            count = np.sum(mask)
            if count == 0:
                continue
            acc = np.mean(correctness[mask])
            conf = np.mean(confidences[mask])
            gap = abs(acc - conf)
            ece += (count / n_samples) * gap
            mce = max(mce, gap)
            bin_acc.append(float(acc))
            bin_conf.append(float(conf))
            bin_counts.append(int(count))

        # Brier score
        one_hot = np.zeros_like(y_prob)
        one_hot[np.arange(n_samples), y_true] = 1.0
        brier_score = np.mean(np.sum((y_prob - one_hot) ** 2, axis=1))

        return {
            "ece": float(ece),
            "mce": float(mce),
            "brier_score": float(brier_score),
            "calibration_bins": {
                "accuracy": bin_acc,
                "confidence": bin_conf,
                "counts": bin_counts,
                "n_bins": n_bins
            }
        }

    def _compute_adaptive_calibration_error(self, y_true, y_prob, n_bins: int = 15) -> Dict[str, Any]:
        """Research-grade ACE + Classwise ECE."""
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)

        if y_prob.ndim != 2:
            raise ValueError("y_prob must be (N, C)")

        n_samples, n_classes = y_prob.shape
        confidences = np.max(y_prob, axis=1)
        predictions = np.argmax(y_prob, axis=1)
        correctness = (predictions == y_true).astype(float)

        # ACE
        sorted_idx = np.argsort(confidences)
        conf_sorted = confidences[sorted_idx]
        corr_sorted = correctness[sorted_idx]
        ace = 0.0
        bin_size = max(1, n_samples // n_bins)
        for i in range(n_bins):
            start = i * bin_size
            end = (i + 1) * bin_size if i < n_bins - 1 else n_samples
            if end > start:
                acc = np.mean(corr_sorted[start:end])
                conf = np.mean(conf_sorted[start:end])
                ace += np.abs(acc - conf) * ((end - start) / n_samples)

        # Classwise ECE
        classwise_ece = {}
        ece_vals = []
        for cls in range(n_classes):
            cls_probs = y_prob[:, cls]
            cls_true = (y_true == cls).astype(float)
            ece_cls = 0.0
            bin_bounds = np.linspace(0, 1, n_bins + 1)
            for i in range(n_bins):
                lower, upper = bin_bounds[i], bin_bounds[i + 1]
                mask = (cls_probs > lower) & (cls_probs <= upper)
                count = np.sum(mask)
                if count == 0:
                    continue
                acc = np.mean(cls_true[mask])
                conf = np.mean(cls_probs[mask])
                ece_cls += (count / n_samples) * abs(acc - conf)
            classwise_ece[f"class_{cls}"] = float(ece_cls)
            ece_vals.append(ece_cls)
        mean_classwise_ece = float(np.mean(ece_vals)) if ece_vals else 0.0

        return {
            "adaptive_calibration_error": float(ace),
            "classwise_ece_mean": mean_classwise_ece,
            "classwise_ece": classwise_ece,
            "calibration_method": "ACE + ClasswiseECE"
        }
    
    def _ensure_energy_meter(self, model: nn.Module, sample_batch: Dict):
        if self.energy_meter is None:
            device = next(model.parameters()).device
            self.energy_meter = EnergyMeter(
                model=model,
                device=device,
                carbon_intensity_kg_per_kwh=self.carbon_intensity,
                energy_per_flop_pj=self.energy_per_flop_pj,
                memory_overhead_factor=self.memory_overhead,
                enable_power_measurement=self.enable_power_measurement
            )
        return self.energy_meter

    # ----------------------------------------------------------------------
    # Bootstrap confidence intervals (refactored)
    # ----------------------------------------------------------------------
    def _bootstrap_confidence_intervals_comprehensive(self, y_true, y_pred) -> Dict[str, float]:
        """Stratified bootstrap CI for all metrics."""
        n_samples = len(y_true)
        if n_samples < 10:
            return {}

        n_boot = min(self.bootstrap_samples, max(1000, n_samples * 10))
        rng = self.rng
        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)
        unique_classes = np.unique(y_true_arr)

        # List of metrics to compute (same as in _compute_core_metrics)
        metric_names = ['accuracy', 'f1_weighted', 'f1_macro', 'precision_macro',
                        'recall_macro', 'cohen_kappa', 'mcc', 'balanced_accuracy']
        boot_vals = {name: [] for name in metric_names}

        for _ in range(n_boot):
            # Stratified resampling
            if len(unique_classes) > 1:
                indices = []
                for cls in unique_classes:
                    cls_indices = np.where(y_true_arr == cls)[0]
                    sampled = rng.choice(cls_indices, size=len(cls_indices), replace=True)
                    indices.extend(sampled)
                indices = np.array(indices)
            else:
                indices = rng.choice(n_samples, size=n_samples, replace=True)

            yt_boot = y_true_arr[indices]
            yp_boot = y_pred_arr[indices]

            try:
                # Compute all metrics at once using core function
                m = self._compute_core_metrics(yt_boot, yp_boot)
                for name in metric_names:
                    if name in m:
                        boot_vals[name].append(m[name])
            except Exception:
                continue

        # Empirical values
        emp = self._compute_core_metrics(y_true_arr, y_pred_arr)

        ci_results = {}
        alpha = 1 - self.confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        for name in metric_names:
            vals = boot_vals.get(name, [])
            if len(vals) < 10:
                continue
            emp_val = emp.get(name, 0.0)
            # Bias-corrected accelerated (simplified - just percentiles for now)
            ci_lower = float(np.percentile(vals, lower_percentile))
            ci_upper = float(np.percentile(vals, upper_percentile))
            ci_results[f'{name}_ci_lower'] = ci_lower
            ci_results[f'{name}_ci_upper'] = ci_upper
            ci_results[f'{name}_ci_width'] = ci_upper - ci_lower
            ci_results[f'{name}_standard_error'] = float(np.std(vals))

        ci_results['confidence_level'] = float(self.confidence_level)
        ci_results['bootstrap_samples_used'] = len(boot_vals.get('accuracy', []))
        ci_results['bootstrap_method'] = 'percentile_bootstrap'

        return ci_results

    # ----------------------------------------------------------------------
    # Power analysis (refactored)
    # ----------------------------------------------------------------------
    def _per_class_power(self, cls, n_samples, n_correct, p_null):
        """Calculate power for one class."""
        if n_samples == 0:
            return None
        p_observed = n_correct / n_samples
        # Effect size Cohen's h
        h = 2 * (np.arcsin(np.sqrt(p_observed)) - np.arcsin(np.sqrt(p_null)))
        # Standard error under null
        se_null = np.sqrt(max(p_null * (1 - p_null), 1e-12) / n_samples)
        z_observed = (p_observed - p_null) / se_null if se_null > 0 else 0
        z_critical = stats.norm.ppf(1 - self.power_alpha)
        power = float(np.clip(1 - stats.norm.cdf(z_critical - z_observed), 0.0, 1.0))

        # Minimum sample size for 80% power
        z_beta = stats.norm.ppf(self.power_threshold)
        delta = max(abs(p_observed - p_null), 1e-6)
        n_min = ((z_critical * np.sqrt(p_null * (1 - p_null)) +
                  z_beta * np.sqrt(max(p_observed * (1 - p_observed), 1e-8))) / delta) ** 2

        return {
            'samples': int(n_samples),
            'correct': int(n_correct),
            'accuracy': float(p_observed),
            'null_accuracy': float(p_null),
            'effect_size_cohens_h': float(h),
            'z_score': float(z_observed),
            'statistical_power': power,
            'power_classification': 'adequate' if power >= self.power_threshold else 'underpowered',
            'minimum_samples_for_power': int(np.ceil(n_min)) if n_min > 0 else 0
        }

    def _compute_per_class_power_analysis(self, y_true, y_pred) -> Dict[str, Any]:
        """Compute statistical power for each class."""
        n_classes = len(set(y_true))
        if n_classes < 2:
            return {}

        from sklearn.metrics import precision_recall_fscore_support
        _, _, _, support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)

        power_analysis = {
            'per_class_power': {},
            'overall_power_metrics': {}
        }

        powers = []
        for cls in range(n_classes):
            n_samp = support[cls]
            n_correct = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
            p_null = 1 / n_classes
            res = self._per_class_power(cls, n_samp, n_correct, p_null)
            if res:
                power_analysis['per_class_power'][f'class_{cls}'] = res
                powers.append(res['statistical_power'])

        if powers:
            power_analysis['overall_power_metrics'] = {
                'mean_power': float(np.mean(powers)),
                'median_power': float(np.median(powers)),
                'min_power': float(np.min(powers)),
                'max_power': float(np.max(powers)),
                'proportion_adequate_power': float(
                    sum(1 for p in powers if p >= self.power_threshold) / len(powers)
                ),
                'recommended_sample_size_multiplier': float(
                    1 / min(max(0.1, np.mean(powers)), 0.99)
                )
            }

        return power_analysis

    # ----------------------------------------------------------------------
    # Statistical significance (refactored - minor)
    # ----------------------------------------------------------------------
    def _compute_statistical_significance_comprehensive(self, y_true, y_pred) -> Dict[str, Any]:
        """Comprehensive significance analysis."""
        n_correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        n_total = len(y_true)
        if n_total == 0:
            return {}

        significance = {}
        p_hat = n_correct / n_total
        z = stats.norm.ppf(1 - (1 - self.confidence_level) / 2)

        # Wilson score interval
        denominator = 1 + (z ** 2) / n_total
        center = (p_hat + (z ** 2) / (2 * n_total)) / denominator
        margin = (z * np.sqrt(
            (p_hat * (1 - p_hat) / n_total) +
            (z ** 2) / (4 * n_total ** 2)
        )) / denominator

        significance['accuracy_interval'] = {
            'estimate': float(p_hat),
            'standard_error': float(np.sqrt(p_hat * (1 - p_hat) / n_total)),
            'confidence_interval': [float(max(0, center - margin)), float(min(1, center + margin))],
            'interval_width': float(2 * margin),
            'method': 'Wilson_score_interval'
        }

        # Kappa bootstrap (simplified)
        kappa = cohen_kappa_score(y_true, y_pred)
        try:
            from sklearn.utils import resample
            kappas = []
            for _ in range(200):
                yt, yp = resample(y_true, y_pred, random_state=self.rng.integers(1e9))
                kappas.append(cohen_kappa_score(yt, yp))
            lower = np.percentile(kappas, (1 - self.confidence_level) / 2 * 100)
            upper = np.percentile(kappas, (1 + self.confidence_level) / 2 * 100)
            significance['cohens_kappa_interval'] = {
                'estimate': float(kappa),
                'confidence_interval': [float(lower), float(upper)],
                'method': 'bootstrap_ci'
            }
        except Exception:
            significance['cohens_kappa_interval'] = {'estimate': float(kappa)}

        # Permutation test for F1
        if n_total >= 50:
            observed_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            perm_f1s = []
            for _ in range(1000):
                y_perm = self.rng.permutation(y_true)
                perm_f1s.append(f1_score(y_true, y_perm, average='weighted', zero_division=0))
            p_val = (sum(1 for f1 in perm_f1s if f1 >= observed_f1) + 1) / (len(perm_f1s) + 1)
            significance['permutation_test'] = {
                'observed_f1': float(observed_f1),
                'permutation_mean_f1': float(np.mean(perm_f1s)),
                'p_value': float(p_val),
                'significant_at_alpha_0.05': p_val < 0.05,
                'test': 'permutation_test_1000_iterations'
            }

        return significance

    # ----------------------------------------------------------------------
    # Inference metrics (from trainer)
    # ----------------------------------------------------------------------
    def build_inference_metrics_from_trainer(
        self,
        inference_stats,
        num_samples: int,
        evaluation_time_seconds: float,
        model: Optional[nn.Module] = None,
        sample_batch: Optional[Dict] = None
    ) -> Dict[str, float]:
        metrics = {
            "latency_mean_ms": 0.0,
            "latency_std_ms": 0.0,
            "throughput_samples_per_sec": 0.0,
            "num_batches": 0,
            "peak_memory_mb": 0.0,
            "peak_memory_gb": 0.0,
            "per_param_kb": 0.0,
            "energy_per_inference_joules": 0.0,
            "energy_per_sample_joules": 0.0,
            "samples_per_kwh": 0.0,
            "carbon_per_million_inferences_kg": 0.0,
            "flops": 0.0,
            "measurement_method": "none"
        }

        if not inference_stats or num_samples <= 0 or evaluation_time_seconds <= 0:
            return metrics

        # Extract per‑batch latencies (in seconds)
        latencies = [x["latency"] for x in inference_stats if x]
        if not latencies:
            return metrics

        # Determine batch size from the first batch (all batches should have same size)
        batch_size = inference_stats[0].get("batch_size", 1)

        # Per‑sample latency in milliseconds
        per_sample_latency_ms = (np.mean(latencies) / batch_size) * 1000.0
        per_sample_latency_std_ms = (np.std(latencies) / batch_size) * 1000.0

        metrics["latency_mean_ms"] = float(per_sample_latency_ms)
        metrics["latency_std_ms"] = float(per_sample_latency_std_ms)
        metrics["throughput_samples_per_sec"] = float(num_samples / evaluation_time_seconds)
        metrics["num_batches"] = len(latencies)

        if model is not None:
            device = next(model.parameters()).device
            if torch.cuda.is_available() and device.type == "cuda":
                peak_mem_bytes = torch.cuda.max_memory_allocated(device)
                metrics["peak_memory_mb"] = float(peak_mem_bytes / 1024**2)
                metrics["peak_memory_gb"] = float(peak_mem_bytes / 1024**3)
            else:
                process = psutil.Process()
                peak_mem_bytes = process.memory_info().rss
                metrics["peak_memory_mb"] = float(peak_mem_bytes / 1024**2)
                metrics["peak_memory_gb"] = float(peak_mem_bytes / 1024**3)

            total_params = sum(p.numel() for p in model.parameters())
            metrics["per_param_kb"] = float(metrics["peak_memory_mb"] * 1024 / max(total_params, 1))

            # Energy measurement using the EnergyMeter
            if sample_batch is not None:
                # Ensure sample_batch is on the correct device
                sample_batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in sample_batch.items()}
                meter = self._ensure_energy_meter(model, sample_batch)
                batch_size_from_batch = next(iter(sample_batch.values())).shape[0]

                # Get the measure_power flag from evaluation config
                measure_power = getattr(self.evaluation_config, 'measure_power', False)

                energy_info = meter.measure_energy(
                    sample_batch,
                    num_runs=100,
                    batch_size=batch_size_from_batch,
                    measure_power=measure_power   # set to True if you want real power
                )
                metrics.update(energy_info)
            else:
                logger.warning("No sample batch provided; energy estimates will be zero.")

        return metrics

    # ----------------------------------------------------------------------
    # Efficiency metrics (unified)
    # ----------------------------------------------------------------------
    def compute_efficiency_metrics(
        self,
        model: nn.Module,
        model_metrics: Dict[str, Any] = None,
        inference_metrics: Dict[str, Any] = None
    ) -> Dict[str, float]:
        """Enhanced efficiency metrics with scientific rigor."""
        accuracy = model_metrics.get('accuracy', 0.0) if model_metrics else 0.0
        accuracy_ci_lower = model_metrics.get('accuracy_ci_lower', accuracy) if model_metrics else accuracy

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        trainable_percentage = (trainable_params / total_params * 100) if total_params > 0 else 0
        trainable_safe = max(trainable_params, 1)

        efficiency = {
            'total_params': float(total_params),
            'trainable_params': float(trainable_params),
            'trainable_percentage': float(trainable_percentage),
            'frozen_params': float(total_params - trainable_params),
            'params_per_accuracy_point': float(trainable_safe / (accuracy * 100 + 1e-6)),
            'accuracy_per_million_params': float(accuracy * 1e6 / trainable_safe),
            'conservative_params_per_accuracy': float(trainable_safe / (accuracy_ci_lower * 100 + 1e-6)),
            'compression_ratio': float(total_params / trainable_safe),
            'sparsity_ratio': float(1 - (trainable_safe / total_params)) if total_params > 0 else 0,
            'bits_per_accuracy': float(np.log2(total_params) / (accuracy + 1e-6)),
            'accuracy_density': float(accuracy / np.log1p(total_params)),
        }

        # Merge inference metrics
        if inference_metrics:
            # Common keys to extract
            mapping = {
                'inference_time_mean_ms': 'latency_mean_ms',
                'throughput_samples_per_second': 'throughput_samples_per_sec',
                'inference_time_cv_percent': lambda: 100 * inference_metrics.get('latency_std_ms', 0) / max(inference_metrics.get('latency_mean_ms', 1e-6), 1e-6),
                'peak_memory_mb': 'peak_memory_mb',
                'peak_memory_gb': 'peak_memory_gb',
                'per_param_kb': 'per_param_kb',
                'energy_per_sample_joules': 'energy_per_sample_joules',
                'samples_per_kwh': 'samples_per_kwh',
                'carbon_per_million_inferences_kg': 'carbon_per_million_inferences_kg',
                'flops': 'flops',
                'measurement_method': 'measurement_method',
            }
            for target, source in mapping.items():
                if callable(source):
                    efficiency[target] = source()
                else:
                    if source in inference_metrics:
                        efficiency[target] = inference_metrics[source]

            # Computational complexity
            if 'computational_complexity' in inference_metrics:
                comp = inference_metrics['computational_complexity']
                efficiency.update({
                    'total_flops': comp.get('flops', 0),
                    'total_macs': comp.get('macs', 0),
                    'flops_per_param': comp.get('flops_per_param', 0),
                    'flops_per_accuracy': comp.get('flops', 0) / (accuracy + 1e-6),
                    'compute_efficiency': comp.get('actual_vs_theoretical_ratio', 0),
                })

            # Energy
            if 'energy_consumption' in inference_metrics:
                energy = inference_metrics['energy_consumption']
                efficiency.update({
                    'energy_per_inference_joules': energy.get('energy_per_inference_joules', 0),
                    'energy_per_sample_joules': energy.get('energy_per_sample_joules', 0),
                    'samples_per_kwh': energy.get('samples_per_kwh', 0),
                })

        efficiency['composite_score'] = self._compute_inferential_composite_score(
            accuracy_point=accuracy,
            accuracy_ci_lower=accuracy_ci_lower,
            trainable_percentage=trainable_percentage,
            efficiency_metrics=efficiency
        )

        efficiency['pareto_efficiency'] = self._calculate_pareto_efficiency(
            accuracy=accuracy,
            trainable_params=trainable_params,
            inference_time=efficiency.get('inference_time_mean_ms', 100),
            memory_usage=efficiency.get('peak_memory_mb', 100)
        )

        return efficiency

    def _compute_inferential_composite_score(
    self, accuracy_point, accuracy_ci_lower, trainable_percentage, efficiency_metrics
) -> float:
        acc = max(accuracy_ci_lower, 1e-6)
        trainable_ratio = trainable_percentage / 100.0
        eff = max(1.0 - trainable_ratio, 1e-6)

        # Use the correct key for latency (per‑sample ms)
        latency_ms = efficiency_metrics.get('inference_time_mean_ms', 100.0)
        latency_ms = max(latency_ms, 1e-6)
        speed_factor = 1.0 / (1.0 + latency_ms)

        memory_gb = efficiency_metrics.get('peak_memory_gb', 10.0)
        memory_gb = max(memory_gb, 1e-6)
        memory_factor = 1.0 / (1.0 + memory_gb)

        composite = acc * eff * speed_factor * memory_factor

        efficiency_metrics['composite_score_weights'] = {
            'accuracy_lower': float(acc),
            'efficiency_ratio': float(eff),
            'speed_factor': float(speed_factor),
            'memory_factor': float(memory_factor),
            'method': 'product_v1.0'
        }
        return float(composite)

    def _calculate_pareto_efficiency(self, accuracy, trainable_params, inference_time, memory_usage) -> Dict:
        """Pareto efficiency with dynamic references."""
        acc_norm = float(np.clip(accuracy, 0.0, 1.0))
        params_safe = max(trainable_params, 1)
        latency_ms = max(inference_time, 1e-6)
        memory_mb = max(memory_usage * 1024.0 if memory_usage < 64 else memory_usage, 1e-6)

        latency_ref = getattr(self, "latency_reference_ms", 5.0 + 10.0 * np.sqrt(params_safe / 1e6))
        memory_ref = getattr(self, "memory_reference_mb", (params_safe * 4) / (1024 ** 2) * 2.5 + 200.0)

        eff_norm = 1.0 / np.log1p(params_safe)
        speed_norm = 1.0 / (1.0 + latency_ms / latency_ref)
        mem_norm = 1.0 / (1.0 + memory_mb / memory_ref)

        eff_norm = float(np.clip(eff_norm, 0.0, 1.0))
        speed_norm = float(np.clip(speed_norm, 0.0, 1.0))
        mem_norm = float(np.clip(mem_norm, 0.0, 1.0))

        distance = np.sqrt((1 - acc_norm) ** 2 + (1 - eff_norm) ** 2 + (1 - speed_norm) ** 2 + (1 - mem_norm) ** 2)
        hypervolume = acc_norm * eff_norm * speed_norm * mem_norm

        baseline = [0.70, 1.0 / np.log1p(10e6), 1.0 / (1.0 + latency_ref * 2 / latency_ref), 1.0 / (1.0 + memory_ref * 1.5 / memory_ref)]
        current = [acc_norm, eff_norm, speed_norm, mem_norm]
        dominance = sum(1 for c, b in zip(current, baseline) if c > b) / len(baseline)

        return {
            "distance_to_ideal": float(distance),
            "hypervolume_indicator": float(hypervolume),
            "dominance_ratio": float(dominance),
            "pareto_classification": "efficient" if dominance >= 0.75 else "inefficient",
            "improvement_potential": float(1.0 - hypervolume),
            "radar_axes": {
                "accuracy": acc_norm,
                "parameter_efficiency": eff_norm,
                "speed": speed_norm,
                "memory_efficiency": mem_norm
            },
            "dynamic_references": {
                "latency_reference_ms": float(latency_ref),
                "memory_reference_mb": float(memory_ref)
            }
        }

    # ----------------------------------------------------------------------
    # Report generation (unified)
    # ----------------------------------------------------------------------
    def generate_scientific_report(
        self,
        model_metrics: Dict[str, Any],
        efficiency_metrics: Dict[str, Any],
        experiment_name: str,
        config: Optional[Dict] = None,
        additional_info: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive scientific report."""
        report = {
            'experiment_info': {
                'name': experiment_name,
                'timestamp': datetime.now().isoformat(),
                'evaluation_method': 'scientific_bopet_v3.0',
                'version': '3.0',
                'statistical_rigor': 'enhanced',
                'confidence_level': self.confidence_level
            },
            'performance_metrics': model_metrics,
            'efficiency_metrics': efficiency_metrics, 
            'visualizations': model_metrics.get('visualizations', {}),
            'inference_analysis': {
                'inference_time_ms': efficiency_metrics.get('inference_time_mean_ms', 'N/A'),
                'throughput_sps': efficiency_metrics.get('throughput_samples_per_second', 'N/A'),
                'energy_per_sample_joules': efficiency_metrics.get('energy_per_sample_joules', 'N/A'),
                'measurement_source': efficiency_metrics.get('measurement_source', 'unknown')
            },
            'power_analysis': model_metrics.get('power_analysis', model_metrics.get('per_class_power_analysis', {})),
            'statistical_summary': self._create_enhanced_statistical_summary(model_metrics, efficiency_metrics),
            'interpretation': self._interpret_enhanced_results(model_metrics, efficiency_metrics),
            'additional_info': additional_info or {}
        }
        if config:
            report['configuration'] = config
        if additional_info and 'rank_history' in additional_info:
            report['rank_history'] = additional_info['rank_history']   # duplicate for plotting

        accuracy = model_metrics.get('accuracy', 0)
        accuracy_ci_lower = model_metrics.get('accuracy_ci_lower', accuracy)
        efficiency_pct = efficiency_metrics.get('trainable_percentage', 100)

        report['overall_scores'] = {
            'accuracy_point': float(accuracy),
            'accuracy_conservative': float(accuracy_ci_lower),
            'efficiency_percentage': float(efficiency_pct),
            'composite_score': float(efficiency_metrics.get('composite_score', 0) or 0),
            'composite_score_weights': efficiency_metrics.get('composite_score_weights', {}),
            'f1_weighted': model_metrics.get('f1_weighted', 0),
            'training_efficiency': float(accuracy_ci_lower * efficiency_pct / 100),
            'pareto_efficiency': efficiency_metrics.get('pareto_efficiency', {})
        }
        report["efficiency_results"] = {
            experiment_name: efficiency_metrics
        }
        return report

    def _create_enhanced_statistical_summary(self, model_metrics, efficiency_metrics) -> Dict:
        return {
            'primary_metrics': {
                'accuracy': {
                    'value': float(model_metrics.get('accuracy', 0)),
                    'interpretation': self._interpret_accuracy(model_metrics.get('accuracy', 0))
                },
                'f1_weighted': {
                    'value': float(model_metrics.get('f1_weighted', 0)),
                    'interpretation': self._interpret_f1(model_metrics.get('f1_weighted', 0))
                }
            },
            'efficiency_analysis': {
                'trainable_percentage': efficiency_metrics.get('trainable_percentage', 0),
                'relative_efficiency': efficiency_metrics.get('compression_ratio', 1),
                'interpretation': self._interpret_efficiency(efficiency_metrics)
            },
            'inference_performance': {
                'time_ms': efficiency_metrics.get('inference_time_mean_ms', 'N/A'),
                'throughput': efficiency_metrics.get('throughput_samples_per_second', 'N/A'),
                'flops': efficiency_metrics.get('total_flops', 'N/A'),
                'energy_per_sample': efficiency_metrics.get('energy_per_sample_joules', 'N/A')
            },
            'statistical_power': self._safe_get(model_metrics, 'power_analysis', 'overall_power_metrics', default={})
        }

    def _interpret_enhanced_results(self, model_metrics, efficiency_metrics) -> Dict:
        interpretations = {
            'performance': self._classify_performance(model_metrics.get('accuracy', 0)),
            'f1_score': self._classify_f1(model_metrics.get('f1_weighted', 0)),
            'efficiency': self._classify_efficiency(efficiency_metrics.get('trainable_percentage', 100)),
            'overall': self._provide_overall_recommendation(
                model_metrics.get('accuracy', 0),
                efficiency_metrics.get('trainable_percentage', 100),
                efficiency_metrics.get('composite_score', 0)
            ),
            'inference_speed': self._classify_inference_speed(efficiency_metrics.get('inference_time_mean_ms', 100)),
            'energy_efficiency': self._classify_energy_efficiency(efficiency_metrics.get('energy_per_sample_joules', 0)),
            'statistical_power': self._interpret_power_adequacy(model_metrics.get('power_analysis', {})),
            'computational_efficiency': self._classify_computational_efficiency(efficiency_metrics)
        }
        return interpretations

    # ----------------------------------------------------------------------
    # Report export (unified)
    # ----------------------------------------------------------------------
    def export_report(
        self,
        report: Dict[str, Any],
        output_dir: Path,
        formats: List[str] = ['json', 'md', 'txt', 'csv']
    ) -> Dict[str, Path]:
        """Export report in multiple formats."""
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_paths = {}
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = f"{report['experiment_info']['name']}_{timestamp}"

        if 'json' in formats:
            json_path = output_dir / f"{base_name}_report.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            saved_paths['json'] = json_path

        if 'md' in formats:
            md_path = output_dir / f"{base_name}_report.md"
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(self._generate_markdown_report(report, enhanced=False))
            saved_paths['markdown'] = md_path
            # Also generate enhanced version
            enhanced_md_path = output_dir / f"{base_name}_report_v2.3.md"
            with open(enhanced_md_path, 'w', encoding='utf-8') as f:
                f.write(self._generate_markdown_report(report, enhanced=True))
            saved_paths['markdown_v2.3'] = enhanced_md_path

        if 'txt' in formats:
            txt_path = output_dir / f"{base_name}_summary.txt"
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(self._generate_text_summary(report))
            saved_paths['text'] = txt_path

        if 'csv' in formats:
            csv_path = output_dir / f"{base_name}_metrics.csv"
            self._export_metrics_to_csv(report, csv_path)
            saved_paths['csv'] = csv_path

        # Save inference and power details
        if 'inference_metrics' in report:
            inf_path = output_dir / f"{base_name}_inference_metrics.json"
            with open(inf_path, 'w') as f:
                json.dump(report['inference_metrics'], f, indent=2)
            saved_paths['inference_json'] = inf_path

        if 'power_analysis' in report:
            power_path = output_dir / f"{base_name}_power_analysis.json"
            with open(power_path, 'w') as f:
                json.dump(report['power_analysis'], f, indent=2)
            saved_paths['power_json'] = power_path

        # Save visualizations if present
        if 'visualizations' in report and report['visualizations']:
            vis_dir = output_dir / "figures"
            vis_dir.mkdir(exist_ok=True)
            for name, path in report['visualizations'].items():
                if path:
                    saved_paths[f'figure_{name}'] = Path(path)

        logger.info(f"Reports exported to {output_dir}")
        return saved_paths

    def _generate_markdown_report(self, report: Dict[str, Any], enhanced: bool = False) -> str:
        """Generate markdown report, optionally with enhanced v2.3 sections."""
        exp = report['experiment_info']
        perf = report['performance_metrics']
        eff = report['efficiency_metrics']
        interp = report.get('interpretation', {})
        overall = report.get('overall_scores', {})

        md = f"""# BOPET Scientific Experiment Report {"(Enhanced v2.3)" if enhanced else ""}

## Experiment Information
- Name: {exp.get('name')}
- Version: {exp.get('version')}
- Timestamp: {exp.get('timestamp')}

## Performance Metrics
| Metric | Value | Interpretation |
|--------|-------|----------------|
| Accuracy | {perf.get('accuracy', 0):.4f} | {interp.get('performance','N/A')} |
| F1 Weighted | {perf.get('f1_weighted', 0):.4f} | {interp.get('f1_score','N/A')} |
"""

        # Calibration
        if enhanced:
            ace = perf.get('adaptive_calibration_error')
            cw_ece = perf.get('classwise_ece')
            md += "\n### Calibration Metrics\n"
            md += f"- ACE: {self._safe_format(ace, '.6f')}\n"
            if isinstance(cw_ece, dict):
                md += "- Classwise ECE:\n"
                for cls, val in cw_ece.items():
                    md += f"    - {cls}: {self._safe_format(val, '.6f')}\n"
            else:
                md += f"- Classwise ECE: {self._safe_format(cw_ece, '.6f')}\n"

        md += f"""
## Efficiency
| Metric | Value |
|--------|-------|
| Trainable % | {eff.get('trainable_percentage', 0):.2f}% |
| Compression Ratio | {eff.get('compression_ratio', 0):.2f}x |
| Composite Score | {overall.get('composite_score', 0):.4f} |

## Overall Assessment
**{interp.get('overall','N/A')}**
"""

        if 'accuracy_ci_lower' in perf:
            ci_l = perf.get('accuracy_ci_lower', 0)
            ci_u = perf.get('accuracy_ci_upper', ci_l)
            cl = perf.get('confidence_level', 0.95)
            md += f"""
## Confidence Interval
- Accuracy {int(cl*100)}% CI: [{ci_l:.4f}, {ci_u:.4f}]
"""

        if enhanced:
            # Add inference, power, pareto sections
            md += self._generate_enhanced_sections(report)

        md += "\n---\n*Generated by BOPET Scientific Evaluator v2.3*\n"
        return md

    def _generate_enhanced_sections(self, report: Dict) -> str:
        """Helper to add enhanced metrics to markdown."""
        eff = report['efficiency_metrics']
        perf = report['performance_metrics']
        inference = report.get('inference_metrics', {})
        power = report.get('power_analysis', {})
        pareto = eff.get('pareto_efficiency', {})

        sections = "\n## Enhanced Metrics (v2.3)\n"

        # Inference
        sections += "\n### Inference Performance\n"
        sections += f"| Metric | Value |\n|--------|-------|\n"
        sections += f"| Mean Inference Time | {self._safe_format(eff.get('inference_time_mean_ms'), '.2f')} ms |\n"
        sections += f"| Throughput | {self._safe_format(eff.get('throughput_samples_per_second'), '.0f')} samples/sec |\n"

        # Computational Complexity
        comp = inference.get('computational_complexity', {})
        if comp:
            sections += "\n### Computational Complexity\n"
            sections += f"| Metric | Value |\n|--------|-------|\n"
            sections += f"| Total FLOPs | {self._safe_format(comp.get('flops'), '.2e')} |\n"
            sections += f"| FLOPs per Param | {self._safe_format(comp.get('flops_per_param'), '.2f')} |\n"
            sections += f"| Compute Efficiency | {self._safe_format(comp.get('actual_vs_theoretical_ratio'), '.3f')} |\n"

        # Memory
        sections += "\n### Memory Usage\n"
        sections += f"| Metric | Value |\n|--------|-------|\n"
        sections += f"| Peak Memory | {self._safe_format(eff.get('peak_memory_mb'), '.1f')} MB |\n"
        sections += f"| Memory per Param | {self._safe_format(eff.get('per_param_kb'), '.2f')} KB |\n"

        # Energy
        sections += "\n### Energy Efficiency\n"
        sections += f"| Metric | Value |\n|--------|-------|\n"
        sections += f"| Energy per Sample | {self._safe_format(eff.get('energy_per_sample_joules'), '.2e')} J |\n"
        sections += f"| Samples per kWh | {self._safe_format(eff.get('samples_per_kwh'), ',.0f')} |\n"

        # Power
        power_ov = power.get('overall_power_metrics', {})
        if power_ov:
            sections += "\n### Statistical Power\n"
            sections += f"| Metric | Value |\n|--------|-------|\n"
            sections += f"| Mean Power | {self._safe_format(power_ov.get('mean_power'), '.3f')} |\n"
            sections += f"| Proportion Adequate | {self._safe_format(power_ov.get('proportion_adequate_power'), '.1%')} |\n"

        # Pareto
        if pareto:
            sections += "\n### Pareto Efficiency\n"
            sections += f"| Metric | Value |\n|--------|-------|\n"
            sections += f"| Distance to Ideal | {self._safe_format(pareto.get('distance_to_ideal'), '.3f')} |\n"
            sections += f"| Hypervolume | {self._safe_format(pareto.get('hypervolume_indicator'), '.3f')} |\n"
            sections += f"| Dominance Ratio | {self._safe_format(pareto.get('dominance_ratio'), '.3f')} |\n"
            sections += f"| Classification | {pareto.get('pareto_classification', 'N/A')} |\n"

        return sections

    def _generate_text_summary(self, report: Dict[str, Any]) -> str:
        """Generate concise text summary."""
        perf = report['performance_metrics']
        eff = report['efficiency_metrics']
        overall = report.get('overall_scores', {})
        interp = report.get('interpretation', {})
        return f"""BOPET EXPERIMENT SUMMARY
=======================

EXPERIMENT: {report['experiment_info']['name']}
TIMESTAMP: {report['experiment_info']['timestamp']}

PERFORMANCE
-----------
Accuracy: {perf.get('accuracy', 0):.4f} ({interp.get('performance', 'N/A')})
F1 Score: {perf.get('f1_weighted', 0):.4f} ({interp.get('f1_score', 'N/A')})

EFFICIENCY
----------
Trainable Params: {eff.get('trainable_params', 0):,}
Trainable %: {eff.get('trainable_percentage', 0):.2f}% ({interp.get('efficiency', 'N/A')})

OVERALL
-------
Composite Score: {overall.get('composite_score', 0):.4f}
Recommendation: {interp.get('overall', 'N/A')}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    def _export_metrics_to_csv(self, report: Dict[str, Any], csv_path: Path):
        """Export metrics to CSV."""
        import pandas as pd
        metrics_data = {}
        for key, value in report['performance_metrics'].items():
            if isinstance(value, (int, float)):
                metrics_data[f'performance_{key}'] = value
        for key, value in report['efficiency_metrics'].items():
            if isinstance(value, (int, float)):
                metrics_data[f'efficiency_{key}'] = value
        for key, value in report.get('overall_scores', {}).items():
            if isinstance(value, (int, float)):
                metrics_data[f'overall_{key}'] = value
        metrics_data['experiment_name'] = report['experiment_info']['name']
        metrics_data['timestamp'] = report['experiment_info']['timestamp']
        df = pd.DataFrame([metrics_data])
        df.to_csv(csv_path, index=False)

    # ----------------------------------------------------------------------
    # Visualization (with safe plot helper)
    # ----------------------------------------------------------------------
    def _safe_plot(self, plot_func, save_dir, filename, *args, **kwargs):
        """Execute a plotting function and save if directory given."""
        try:
            fig = plot_func(*args, **kwargs)
            if save_dir:
                save_path = Path(save_dir) / filename
                fig.savefig(save_path, dpi=300, bbox_inches="tight")
                plt.close(fig)
                return str(save_path)
            plt.close(fig)
            return None
        except Exception as e:
            logger.warning(f"Plot {filename} failed: {e}")
            return None

    def generate_scientific_visualizations(
        self, report: Dict[str, Any], save_dir: Optional[str] = None
    ) -> Dict[str, str]:
        """Generate and save scientific plots."""
        import matplotlib.pyplot as plt

        figures = {}
        perf = report['performance_metrics']

        # Reliability diagram
        if 'calibration_bins' in perf:
            def plot_reliability():
                bins = perf['calibration_bins']
                conf = np.array(bins['confidence'])
                acc = np.array(bins['accuracy'])
                fig, ax = plt.subplots()
                ax.plot(conf, acc, marker='o')
                ax.plot([0,1],[0,1], 'k--')
                ax.set_title("Reliability Diagram")
                ax.set_xlabel("Confidence")
                ax.set_ylabel("Accuracy")
                return fig
            figures['reliability_diagram'] = self._safe_plot(
                plot_reliability, save_dir, "reliability_diagram.png"
            )

        # Pareto radar
        eff_results = report.get("efficiency_results", {})
        if eff_results:
            def plot_radar():
                import numpy as np
                fig, ax = plt.subplots(figsize=(8,8), subplot_kw=dict(polar=True))
                angles = np.linspace(0, 2*np.pi, 4, endpoint=False)
                angles = np.concatenate([angles, [angles[0]]])
                labels = ["Accuracy","Param Eff","Speed","Memory"]
                ax.set_theta_offset(np.pi/2)
                ax.set_theta_direction(-1)
                ax.set_thetagrids(angles[:-1]*180/np.pi, labels)
                ax.set_ylim(0,1)
                for name, metrics in eff_results.items():
                    pareto = metrics.get("pareto_efficiency", {})
                    axes = pareto.get("radar_axes")
                    if axes:
                        vals = [axes.get("accuracy",0), axes.get("parameter_efficiency",0),
                                axes.get("speed",0), axes.get("memory_efficiency",0)]
                        vals = np.concatenate([vals, [vals[0]]])
                        ax.plot(angles, vals, linewidth=2, label=name)
                        ax.fill(angles, vals, alpha=0.15)
                ax.legend(loc="upper right", bbox_to_anchor=(1.25,1.1))
                ax.set_title("Scientific Efficiency Radar", fontsize=14, pad=20)
                return fig
            figures['pareto_radar'] = self._safe_plot(
                plot_radar, save_dir, "scientific_pareto_radar.png"
            )

        # Efficiency scatter
        def plot_scatter():
            acc = perf.get('accuracy',0)
            eff = 1 - report['efficiency_metrics'].get('trainable_percentage',100)/100
            fig, ax = plt.subplots()
            ax.scatter([eff],[acc])
            ax.set_xlabel("Efficiency (1 - trainable %)")
            ax.set_ylabel("Accuracy")
            ax.set_title("Efficiency vs Accuracy")
            return fig
        figures['efficiency_scatter'] = self._safe_plot(
            plot_scatter, save_dir, "efficiency_scatter.png"
        )

        # Training stability
        history = self._safe_get(report, 'additional_info', 'training_history', default={})
        loss = history.get('train_loss')
        if loss:
            def plot_stability():
                fig, ax = plt.subplots()
                ax.plot(loss)
                ax.set_title("Training Stability")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Loss")
                return fig
            figures['training_stability'] = self._safe_plot(
                plot_stability, save_dir, "training_stability.png"
            )

        # Rank evolution
        rank_hist = report.get("rank_history", [])
        if rank_hist:
            def plot_rank():
                import numpy as np
                epochs = [e["epoch"] for e in rank_hist]
                # collect module names
                mods = set()
                for e in rank_hist:
                    mods.update(e["allocation"].keys())
                mods = sorted(mods)
                rank_mat = []
                for e in rank_hist:
                    row = [e["allocation"].get(m,0) for m in mods]
                    rank_mat.append(row)
                rank_mat = np.array(rank_mat)
                fig, ax = plt.subplots(figsize=(10,6))
                for i, m in enumerate(mods):
                    ax.plot(epochs, rank_mat[:,i], alpha=0.4)
                avg = rank_mat.mean(axis=1)
                ax.plot(epochs, avg, linewidth=3, label="Average Rank")
                ax.set_title("Progressive PEPT Rank Evolution")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Allocated Rank")
                ax.legend()
                ax.grid(True)
                return fig
            figures['rank_evolution'] = self._safe_plot(
                plot_rank, save_dir, "rank_evolution.png"
            )

        return figures

    # ----------------------------------------------------------------------
    # Interpretation helpers (unchanged logic)
    # ----------------------------------------------------------------------
    def _interpret_accuracy(self, acc: float) -> str:
        if acc >= 0.95: return "Excellent performance"
        elif acc >= 0.90: return "Very good performance"
        elif acc >= 0.80: return "Good performance"
        elif acc >= 0.70: return "Acceptable performance"
        else: return "Needs improvement"

    def _interpret_f1(self, f1: float) -> str:
        if f1 >= 0.90: return "Excellent balance"
        elif f1 >= 0.80: return "Good balance"
        elif f1 >= 0.70: return "Acceptable balance"
        else: return "Imbalanced performance"

    def _interpret_efficiency(self, eff: Dict) -> str:
        pct = eff.get('trainable_percentage', 100)
        if pct < 5: return "Highly efficient PEFT"
        elif pct < 20: return "Moderately efficient"
        else: return "Standard efficiency"

    def _classify_performance(self, acc: float) -> str:
        if acc >= 0.95: return "State-of-the-art"
        elif acc >= 0.90: return "Competitive"
        elif acc >= 0.80: return "Good"
        elif acc >= 0.70: return "Baseline"
        else: return "Below baseline"

    def _classify_f1(self, f1: float) -> str:
        if f1 >= 0.95: return "Excellent"
        elif f1 >= 0.90: return "Very Good"
        elif f1 >= 0.85: return "Good"
        elif f1 >= 0.80: return "Acceptable"
        else: return "Poor"

    def _classify_efficiency(self, pct: float) -> str:
        if pct < 5: return "Highly Efficient (PEFT optimized)"
        elif pct < 15: return "Efficient"
        elif pct < 30: return "Moderate"
        elif pct < 50: return "Standard"
        else: return "Inefficient (near full fine-tuning)"

    def _provide_overall_recommendation(self, acc: float, pct: float, comp: float) -> str:
        if comp > 0.03:
            return "Excellent trade-off – highly recommended"
        elif comp > 0.02:
            return "Good trade-off – recommended"
        elif comp > 0.01:
            return "Moderate trade-off – acceptable"
        else:
            return "Poor trade-off – consider alternatives"

    def _classify_inference_speed(self, time_ms: float) -> str:
        if time_ms < 10: return "Real-time capable"
        elif time_ms < 100: return "Fast inference"
        elif time_ms < 1000: return "Moderate speed"
        else: return "Slow inference"

    def _classify_energy_efficiency(self, energy_j: float) -> str:
        if energy_j < 1e-6: return "Extremely energy efficient"
        elif energy_j < 1e-3: return "Energy efficient"
        elif energy_j < 1: return "Moderate energy use"
        else: return "Energy intensive"

    def _interpret_power_adequacy(self, power_analysis: Dict) -> str:
        if not power_analysis:
            return "Power analysis not available"
        mean_power = self._safe_get(power_analysis, 'overall_power_metrics', 'mean_power', default=0)
        if mean_power >= 0.8: return "Adequate statistical power"
        elif mean_power >= 0.5: return "Moderate statistical power"
        else: return "Underpowered study - consider more samples"

    def _classify_computational_efficiency(self, eff: Dict) -> str:
        fpp = eff.get('flops_per_param', 0)
        ce = eff.get('compute_efficiency', 0)
        if fpp < 10 and ce > 0.5: return "Computationally efficient"
        elif fpp < 100: return "Moderate compute use"
        else: return "Computationally intensive"


# Alias for backward compatibility
BOPETEvaluator = ScientificEvaluator


def create_evaluator_from_config(config: ExperimentConfig) -> ScientificEvaluator:
    return ScientificEvaluator(config)