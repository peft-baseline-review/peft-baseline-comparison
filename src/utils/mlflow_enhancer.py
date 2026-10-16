"""
Enhanced MLflow tracking with v2.2 metrics, scientific rigour, and BOPET integration
Production-ready with progressive PEPT support
"""

import mlflow
import mlflow.pytorch
import psutil
import torch
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
import json
import yaml
import logging
import numpy as np
import time
import gc
import shutil

logger = logging.getLogger(__name__)


class MLflowEnhancer:
    """
    Enhanced MLflow tracking for scientific experiments v2.2
    """

    def __init__(self, config, run_name: Optional[str] = None):
        self.config = config
        self.run_name = run_name or self._generate_run_name()
        self.run_id = None
        self.run = None

        # Tracking state
        self.start_time = datetime.now()
        self.metrics_buffer = {}
        self.last_flush_time = time.time()
        self.flush_interval = 60

        # Resource monitoring
        self.peak_gpu_memory_gb = 0
        self.total_energy_joules = 0
        self.batch_metrics_history = []

        # Progressive PEPT tracking
        self.rank_allocation_history = []
        self.importance_scores_history = []

        # Setup MLflow with error handling
        self._setup_mlflow()

    def _safe_str(self, val) -> str:
        """Convert any value to string safely."""
        if val is None:
            return 'unknown'
        if isinstance(val, (list, dict, tuple)):
            return str(val)
        return str(val)

    def _generate_run_name(self) -> str:
        """Generate informative run name."""
        timestamp = datetime.now().strftime('%H%M%S')
        if hasattr(self.config, 'experiment_name'):
            base_name = self.config.experiment_name.replace(' ', '_')
            return f"{base_name}_{timestamp}"
        model_name = getattr(self.config.model, 'name', 'unknown').split('/')[-1]
        peft_method = self._safe_str(getattr(self.config.peft, 'method', 'unknown'))
        return f"{model_name}_{peft_method}_{timestamp}"

    def _setup_mlflow(self):
        """Enhanced MLflow setup with error handling."""
        if not getattr(self.config, 'use_mlflow', True):
            logger.info("📊 MLflow tracking disabled by configuration")
            return

        try:
            # Set tracking URI with fallback
            tracking_uri = getattr(self.config, 'mlflow_tracking_uri', None)
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
                logger.info(f"MLflow tracking URI: {tracking_uri}")

            # Create or get experiment
            experiment_name = getattr(self.config, 'mlflow_experiment_name', 'bopet_research')
            try:
                experiment = mlflow.get_experiment_by_name(experiment_name)
                if experiment is None:
                    mlflow.create_experiment(experiment_name)
                    logger.info(f"Created new experiment: {experiment_name}")
            except Exception as e:
                logger.warning(f"Experiment creation failed: {e}")
                experiment_name = 'Default'

            mlflow.set_experiment(experiment_name)

            # Prepare tags
            tags = self._prepare_tags()

            # Start run
            self.run = mlflow.start_run(
                run_name=self.run_name,
                tags=tags,
                description=self._generate_run_description()
            )
            self.run_id = self.run.info.run_id

            # Log initial parameters
            self.log_experiment_params()

            logger.info(f"📊 MLflow run started: {self.run_id}")
            logger.info(f"   Experiment: {experiment_name}")
            logger.info(f"   Run name: {self.run_name}")

        except Exception as e:
            logger.error(f"❌ Failed to initialize MLflow: {e}")
            logger.info("🔄 Continuing without MLflow tracking")
            self.config.use_mlflow = False

    def _prepare_tags(self) -> Dict[str, str]:
        """Prepare comprehensive tags for the run with safe string conversion."""
        tags = {
            "project": "BOPET",
            "study_id": self._safe_str(getattr(self.config, 'study_id', None)),
            "hypothesis": self._safe_str(getattr(self.config, 'hypothesis', None))[:100],
            "model": self._safe_str(getattr(self.config.model, 'name', None)),
            "model_architecture": self._safe_str(getattr(self.config.model, 'architecture', None)),
            "peft_method": self._safe_str(getattr(self.config.peft, 'method', None)),
            "task": self._safe_str(getattr(self.config.task, 'task_type', None)),
            "num_classes": self._safe_str(getattr(self.config.task, 'num_labels', 0)),
            "timestamp": datetime.now().isoformat(),
            "mlflow_version": mlflow.__version__,
            "bopet_version": "2.2",
        }

        # Add platform info
        import platform
        tags.update({
            "platform": platform.system(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "cuda_available": self._safe_str(torch.cuda.is_available()),
        })

        # Add progressive PEPT tags if applicable
        if hasattr(self.config, 'peft'):
            method = self._safe_str(getattr(self.config.peft, 'method', ''))
            if 'progressive' in method.lower():
                tags.update({
                    "progressive_pept": "true",
                    "base_method": self._safe_str(getattr(self.config.peft, 'base_method', 'lora')),
                    "rank_schedule": self._safe_str(getattr(self.config.peft, 'rank_schedule', 'linear')),
                    "allocation_strategy": self._safe_str(getattr(self.config.peft, 'allocation_strategy', 'proportional')),
                })

        return tags

    def _generate_run_description(self) -> str:
        """Generate detailed run description."""
        return f"BOPET experiment run {self.run_name}"

    def log_experiment_params(self, additional_params: Optional[Dict] = None,
                              data_stats: Optional[Dict[str, Any]] = None):
        """Log comprehensive experiment parameters."""
        if not getattr(self.config, 'use_mlflow', True) or not self.run:
            return

        try:
            self._log_config_params()
            self._log_system_parameters()
            self._log_v2_2_parameters()
            if data_stats:
                self._log_data_statistics(data_stats)
            if additional_params:
                for key, value in additional_params.items():
                    if value is not None:
                        mlflow.log_param(f"additional_{key}", self._safe_str(value))
        except Exception as e:
            logger.warning(f"Failed to log parameters: {e}")

    def _log_config_params(self):
        """Log configuration parameters safely."""
        # Model parameters
        mlflow.log_params({
            "model_name": self._safe_str(getattr(self.config.model, 'name', None)),
            "model_architecture": self._safe_str(getattr(self.config.model, 'architecture', None)),
            "model_hidden_size": getattr(self.config.model, 'hidden_size', 0),
            "model_num_layers": getattr(self.config.model, 'num_hidden_layers', 0),
        })

        # PEFT parameters
        peft_params = {
            "peft_method": self._safe_str(getattr(self.config.peft, 'method', None)),
            "peft_rank": getattr(self.config.peft, 'rank', 0),
            "peft_alpha": getattr(self.config.peft, 'alpha', 0),
            "peft_dropout": getattr(self.config.peft, 'dropout', 0),
        }

        # Add progressive PEPT parameters
        if hasattr(self.config, 'peft') and 'progressive' in self._safe_str(getattr(self.config.peft, 'method', '')):
            peft_params.update({
                "progressive_base_method": self._safe_str(getattr(self.config.peft, 'base_method', 'lora')),
                "progressive_initial_rank": getattr(self.config.peft, 'initial_rank', 4),
                "progressive_final_rank": getattr(self.config.peft, 'final_rank', 16),
                "progressive_rank_schedule": self._safe_str(getattr(self.config.peft, 'rank_schedule', 'linear')),
                "progressive_allocation_strategy": self._safe_str(getattr(self.config.peft, 'allocation_strategy', 'proportional')),
                "progressive_importance_method": self._safe_str(getattr(self.config.peft, 'importance_method', 'gradient_norm')),
            })

        mlflow.log_params(peft_params)

        # Task parameters
        mlflow.log_params({
            "task_type": self._safe_str(getattr(self.config.task, 'task_type', None)),
            "num_classes": getattr(self.config.task, 'num_labels', 0),
            "batch_size": getattr(self.config.task, 'batch_size', 0),
            "max_length": getattr(self.config.task, 'max_length', 128),
        })

        # Training parameters
        mlflow.log_params({
            "training_epochs": getattr(self.config.training, 'num_epochs', 0),
            "learning_rate": getattr(self.config.training, 'learning_rate', 0),
            "weight_decay": getattr(self.config.training, 'weight_decay', 0),
            "warmup_ratio": getattr(self.config.training, 'warmup_ratio', 0),
            "lr_scheduler": getattr(self.config.training, 'lr_scheduler', 'linear'),
            "gradient_accumulation": getattr(self.config.training, 'gradient_accumulation_steps', 1),
        })

        # Evaluation parameters
        if hasattr(self.config, 'evaluation'):
            mlflow.log_params({
                "eval_confidence_level": getattr(self.config.evaluation, 'confidence_level', 0.95),
                "eval_bootstrap_samples": getattr(self.config.evaluation, 'bootstrap_samples', 1000),
                "eval_composite_alpha": getattr(self.config.evaluation, 'composite_alpha', 0.01),
                "eval_power_alpha": getattr(self.config.evaluation, 'power_analysis_alpha', 0.05),
                "eval_power_threshold": getattr(self.config.evaluation, 'power_threshold', 0.8),
            })

    def _log_system_parameters(self):
        """Log comprehensive system parameters."""
        import platform
        mlflow.log_params({
            "system_platform": platform.system(),
            "system_release": platform.release(),
            "system_version": platform.version(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "cpu_cores_physical": psutil.cpu_count(logical=False) or 1,
            "cpu_cores_logical": psutil.cpu_count(logical=True) or 1,
            "system_memory_total_gb": psutil.virtual_memory().total / 1e9,
            "system_memory_available_gb": psutil.virtual_memory().available / 1e9,
        })

        if torch.cuda.is_available():
            mlflow.log_param("gpu_count", torch.cuda.device_count())
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                mlflow.log_params({
                    f"gpu_{i}_name": props.name,
                    f"gpu_{i}_memory_gb": props.total_memory / 1e9,
                    f"gpu_{i}_compute": f"{props.major}.{props.minor}",
                    f"gpu_{i}_multiprocessors": props.multi_processor_count,
                })
        else:
            mlflow.log_param("gpu_count", 0)

    def _log_v2_2_parameters(self):
        """Log v2.2 specific parameters."""
        if hasattr(self.config, 'evaluation'):
            eval_config = self.config.evaluation
            mlflow.log_params({
                "inference_batch_size": getattr(eval_config, 'inference_batch_size', 1),
                "num_inference_runs": getattr(eval_config, 'num_inference_runs', 100),
                "measure_energy": getattr(eval_config, 'measure_energy', False),
                "measure_memory": getattr(eval_config, 'measure_memory', True),
                "statistical_tests": self._safe_str(getattr(eval_config, 'statistical_tests', [])),
                "multiple_comparison_correction": getattr(eval_config, 'multiple_comparison_correction', 'holm'),
                "min_samples_for_analysis": getattr(eval_config, 'min_samples_for_analysis', 10),
            })

    def _log_data_statistics(self, data_stats: Dict[str, Any]):
        """Log dataset statistics safely."""
        try:
            if not isinstance(data_stats, dict):
                return
            for key, value in data_stats.items():
                if isinstance(value, (int, float, str, bool)):
                    mlflow.log_param(f"data_{key}", value)
        except Exception as e:
            logger.debug(f"Failed to log data statistics: {e}")

    def log_training_metrics(self, history: Dict[str, List[float]], final_epoch: int):
        """Enhanced training metrics logging with v2.2 support."""
        if not getattr(self.config, 'use_mlflow', True) or not self.run:
            return

        try:
            metrics_to_log = {}
            metrics_to_log["final_epoch"] = final_epoch

            for key, values in history.items():
                if values and isinstance(values, list) and len(values) > 0:
                    final_value = values[-1]
                    metrics_to_log[f"final_{key}"] = float(final_value)

                    if key.endswith('_loss'):
                        metrics_to_log.update({
                            f"{key}_min": float(np.min(values)),
                            f"{key}_max": float(np.max(values)),
                            f"{key}_mean": float(np.mean(values)),
                            f"{key}_std": float(np.std(values)) if len(values) > 1 else 0.0,
                            f"{key}_final": float(final_value),
                        })

                    if key in ['train_loss', 'val_loss', 'train_accuracy', 'val_accuracy', 'val_f1']:
                        sample_rate = max(1, len(values) // 20)
                        for epoch, value in enumerate(values):
                            if epoch % sample_rate == 0 or epoch == len(values) - 1:
                                mlflow.log_metric(f"{key}_epoch", float(value), step=epoch)

            if 'train_loss' in history and history['train_loss']:
                train_losses = history['train_loss']
                if len(train_losses) > 1:
                    convergence_rate = train_losses[0] / train_losses[-1] if train_losses[-1] > 0 else 0
                    metrics_to_log["convergence_rate"] = float(convergence_rate)

            mlflow.log_metrics(metrics_to_log)

            training_duration = (datetime.now() - self.start_time).total_seconds()
            mlflow.log_metric("training_duration_seconds", training_duration)

            logger.debug(f"📊 Logged {len(metrics_to_log)} training metrics")

        except Exception as e:
            logger.warning(f"Failed to log training metrics: {e}")

    def log_model_metrics_v2_2(self, results: Dict[str, Any]):
        """Enhanced model metrics logging with full v2.2 integration."""
        if not getattr(self.config, 'use_mlflow', True) or not self.run:
            return

        try:
            model_metrics = results.get("results", {}).get("model_metrics", {})
            efficiency_metrics = results.get("results", {}).get("efficiency_metrics", {})
            inference_metrics = results.get("results", {}).get("model_metrics", {}).get("inference_metrics", {})
            power_analysis = results.get("results", {}).get("model_metrics", {}).get("power_analysis", {})

            all_metrics = {}

            # 1. Core Classification Metrics
            core_metrics = ['accuracy', 'f1_weighted', 'f1_macro', 'precision_macro',
                           'recall_macro', 'cohen_kappa', 'mcc', 'balanced_accuracy']
            for metric in core_metrics:
                if metric in model_metrics:
                    value = model_metrics[metric]
                    if isinstance(value, (int, float)):
                        all_metrics[metric] = float(value)

            # 2. Confidence Intervals
            for key in model_metrics.keys():
                if key.endswith('_ci_lower') or key.endswith('_ci_upper') or key.endswith('_ci_width'):
                    value = model_metrics[key]
                    if isinstance(value, (int, float)):
                        all_metrics[key] = float(value)

            # 3. Efficiency Metrics
            for key, value in efficiency_metrics.items():
                if isinstance(value, (int, float)):
                    all_metrics[f"eff_{key}"] = float(value)

            # 4. Inference Metrics
            if inference_metrics:
                if 'inference_time' in inference_metrics:
                    for subkey, subvalue in inference_metrics['inference_time'].items():
                        if isinstance(subvalue, (int, float)):
                            all_metrics[f"inf_time_{subkey}"] = float(subvalue)
                if 'computational_complexity' in inference_metrics:
                    for subkey, subvalue in inference_metrics['computational_complexity'].items():
                        if isinstance(subvalue, (int, float)):
                            all_metrics[f"inf_comp_{subkey}"] = float(subvalue)
                if 'memory_usage' in inference_metrics:
                    for subkey, subvalue in inference_metrics['memory_usage'].items():
                        if isinstance(subvalue, (int, float)):
                            all_metrics[f"inf_mem_{subkey}"] = float(subvalue)
                if 'energy_consumption' in inference_metrics:
                    for subkey, subvalue in inference_metrics['energy_consumption'].items():
                        if isinstance(subvalue, (int, float)):
                            all_metrics[f"inf_energy_{subkey}"] = float(subvalue)

            # 5. Power Analysis Metrics
            if power_analysis and 'overall_power_metrics' in power_analysis:
                for key, value in power_analysis['overall_power_metrics'].items():
                    if isinstance(value, (int, float)):
                        all_metrics[f"power_{key}"] = float(value)

            # 6. Statistical Significance
            if 'significance_level' in model_metrics:
                sig_metrics = model_metrics['significance_level']
                if isinstance(sig_metrics, dict):
                    for key, value in sig_metrics.items():
                        if isinstance(value, (int, float)):
                            all_metrics[f"sig_{key}"] = float(value)

            # 7. Composite Score
            if 'composite_score' in efficiency_metrics:
                all_metrics['composite_score'] = float(efficiency_metrics['composite_score'])

            # 8. Pareto Efficiency
            if 'pareto_efficiency' in efficiency_metrics:
                pareto_metrics = efficiency_metrics['pareto_efficiency']
                if isinstance(pareto_metrics, dict):
                    for key, value in pareto_metrics.items():
                        if isinstance(value, (int, float)):
                            all_metrics[f"pareto_{key}"] = float(value)

            # Batch logging
            batch_size = 100
            metrics_items = list(all_metrics.items())
            for i in range(0, len(metrics_items), batch_size):
                batch = dict(metrics_items[i:i+batch_size])
                mlflow.log_metrics(batch)
                time.sleep(0.1)

            logger.info(f"📊 Logged {len(all_metrics)} v2.2 metrics to MLflow")

            # Log experiment report as artifact
            if 'experiment_report' in results.get("results", {}):
                report = results["results"]["experiment_report"]
                report_path = Path("temp_report.json")
                with open(report_path, 'w') as f:
                    json.dump(report, f, indent=2)
                mlflow.log_artifact(str(report_path), artifact_path="reports")
                report_path.unlink()

        except Exception as e:
            logger.error(f"❌ Failed to log v2.2 metrics: {e}")
            import traceback
            logger.debug(traceback.format_exc())

    def log_progressive_pept_metrics(self, rank_allocation: Dict[str, Any], epoch: int):
        """Log progressive PEPT rank allocation metrics."""
        if not getattr(self.config, 'use_mlflow', True) or not self.run:
            return

        try:
            metrics = {
                f"progressive_epoch_{epoch}_total_rank": rank_allocation.get('total_rank', 0),
                f"progressive_epoch_{epoch}_active_modules": rank_allocation.get('active_modules', 0),
                f"progressive_epoch_{epoch}_rank_change_count": rank_allocation.get('rank_change_count', 0),
            }

            if 'importance_scores' in rank_allocation:
                scores = rank_allocation['importance_scores']
                if isinstance(scores, dict) and len(scores) > 0:
                    mean_importance = np.mean(list(scores.values()))
                    std_importance = np.std(list(scores.values())) if len(scores) > 1 else 0
                    metrics.update({
                        f"progressive_epoch_{epoch}_mean_importance": float(mean_importance),
                        f"progressive_epoch_{epoch}_std_importance": float(std_importance),
                    })

            mlflow.log_metrics(metrics, step=epoch)

            self.rank_allocation_history.append({
                'epoch': epoch,
                'total_rank': rank_allocation.get('total_rank', 0),
                'timestamp': datetime.now().isoformat()
            })

        except Exception as e:
            logger.debug(f"Failed to log progressive PEPT metrics: {e}")

    def log_batch_metrics(self, batch_idx: int, batch_metrics: Dict[str, float], phase: str = "train"):
        """Log batch-level metrics with adaptive sampling."""
        if not getattr(self.config, 'use_mlflow', True) or not self.run:
            return

        try:
            if phase == "train":
                log_frequency = max(10, 100 // (batch_idx // 1000 + 1))
            else:
                log_frequency = 10

            if batch_idx % log_frequency != 0:
                return

            for key, value in batch_metrics.items():
                if isinstance(value, (int, float)):
                    self.metrics_buffer[f"batch_{phase}_{key}"] = float(value)

            self.batch_metrics_history.append({
                'batch_idx': batch_idx,
                'phase': phase,
                'metrics': batch_metrics.copy(),
                'timestamp': time.time()
            })

            current_time = time.time()
            if current_time - self.last_flush_time > self.flush_interval and self.metrics_buffer:
                mlflow.log_metrics(self.metrics_buffer, step=batch_idx)
                self.metrics_buffer.clear()
                self.last_flush_time = current_time

        except Exception as e:
            logger.debug(f"Batch metrics logging skipped: {e}")

    def log_resource_usage(self, phase: str = "training"):
        """Log resource usage with minimal overhead."""
        if not getattr(self.config, 'use_mlflow', True) or not self.run:
            return

        try:
            metrics = {}

            metrics.update({
                f"{phase}_cpu_percent": psutil.cpu_percent(interval=0.1),
                f"{phase}_memory_percent": psutil.virtual_memory().percent,
                f"{phase}_memory_available_gb": psutil.virtual_memory().available / 1e9,
            })

            if torch.cuda.is_available():
                try:
                    current_memory = torch.cuda.memory_allocated() / 1e9
                    self.peak_gpu_memory_gb = max(self.peak_gpu_memory_gb, current_memory)

                    metrics.update({
                        f"{phase}_gpu_memory_current_gb": current_memory,
                        f"{phase}_gpu_memory_peak_gb": self.peak_gpu_memory_gb,
                        f"{phase}_gpu_utilization": torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0,
                    })

                    if hasattr(self.config, 'log_detailed_gpu_metrics') and self.config.log_detailed_gpu_metrics:
                        try:
                            import GPUtil
                            gpus = GPUtil.getGPUs()
                            for i, gpu in enumerate(gpus[:1]):
                                metrics.update({
                                    f"{phase}_gpu_{i}_load": gpu.load * 100,
                                    f"{phase}_gpu_{i}_memory_percent": gpu.memoryUtil * 100,
                                    f"{phase}_gpu_{i}_temperature": gpu.temperature,
                                })
                        except ImportError:
                            pass
                except Exception as gpu_error:
                    logger.debug(f"GPU metrics unavailable: {gpu_error}")

            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.metrics_buffer[key] = float(value)

        except Exception as e:
            logger.debug(f"Resource logging skipped: {e}")

    def flush_metrics(self):
        """Force flush buffered metrics to MLflow."""
        if not getattr(self.config, 'use_mlflow', True) or not self.run:
            return

        if self.metrics_buffer:
            try:
                mlflow.log_metrics(self.metrics_buffer)
                self.metrics_buffer.clear()
                self.last_flush_time = time.time()
                logger.debug(f"📊 Flushed {len(self.metrics_buffer)} metrics to MLflow")
            except Exception as e:
                logger.warning(f"Failed to flush metrics: {e}")

    def log_artifacts_organized(self, artifacts_dir: Path):
        """Enhanced artifact logging with v2.2 structure."""
        if not getattr(self.config, 'use_mlflow', True) or not self.run:
            return

        try:
            organized_dir = artifacts_dir / "mlflow_organized"
            organized_dir.mkdir(exist_ok=True, parents=True)

            artifact_categories = {
                "configs": ["*.yaml", "*.yml", "config.json", "*.config"],
                "checkpoints": ["*.pt", "*.pth", "*.bin", "*.safetensors"],
                "logs": ["*.log", "*.txt", "*.out"],
                "results": ["*results*.json", "*report*.json", "*summary*.json"],
                "metrics": ["*metrics*.csv", "*scores*.csv", "*analysis*.csv"],
                "visualizations": ["*.png", "*.pdf", "*.jpg", "*.svg"],
                "statistics": ["*stat*.json", "*power*.json", "*ci*.json"],
                "inference": ["*inference*.json", "*flops*.json", "*energy*.json"],
                "reports": ["*report*.md", "*summary*.md", "*readme*.md"],
            }

            files_copied = 0

            for category, patterns in artifact_categories.items():
                category_dir = organized_dir / category
                category_dir.mkdir(exist_ok=True)

                for pattern in patterns:
                    for file_path in artifacts_dir.rglob(pattern):
                        if file_path.is_file() and not any(ignore in str(file_path) for ignore in ['mlflow_organized', '__pycache__']):
                            try:
                                target_path = category_dir / file_path.name
                                if target_path.exists():
                                    target_path = category_dir / f"{file_path.stem}_{file_path.parent.name}{file_path.suffix}"
                                shutil.copy2(file_path, target_path)
                                files_copied += 1
                            except Exception as copy_error:
                                logger.debug(f"Failed to copy {file_path}: {copy_error}")

            if files_copied > 0:
                mlflow.log_artifacts(str(organized_dir), artifact_path="artifacts")
                logger.info(f"📦 Logged {files_copied} artifacts in organized structure")

            try:
                mlflow.log_artifacts(str(artifacts_dir), artifact_path="raw_artifacts")
            except Exception:
                pass

            try:
                shutil.rmtree(organized_dir)
            except:
                pass

        except Exception as e:
            logger.error(f"❌ Failed to log organized artifacts: {e}")
            try:
                mlflow.log_artifacts(str(artifacts_dir))
            except Exception as fallback_error:
                logger.error(f"Fallback artifact logging also failed: {fallback_error}")

    def log_model_checkpoint(self, model, experiment_dir: Path):
        """Log model checkpoint with v2.2 metadata."""
        if not getattr(self.config, 'use_mlflow', True) or not self.run:
            return

        try:
            checkpoint_dir = experiment_dir / "mlflow_model"
            checkpoint_dir.mkdir(exist_ok=True)

            model_path = checkpoint_dir / "model.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': self.config.model_dump(exclude_none=True) if hasattr(self.config, 'model_dump') else {},
                'metadata': {
                    'saved_at': datetime.now().isoformat(),
                    'bopet_version': '2.2',
                    'model_type': self._safe_str(getattr(self.config.peft, 'method', 'unknown')),
                    'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
                    'total_params': sum(p.numel() for p in model.parameters()),
                }
            }, model_path)

            config_path = checkpoint_dir / "config.yaml"
            with open(config_path, 'w') as f:
                if hasattr(self.config, 'model_dump'):
                    yaml.dump(self.config.model_dump(exclude_none=True), f, default_flow_style=False)
                else:
                    yaml.dump({}, f)

            mlflow.log_artifact(str(model_path), artifact_path="model")
            mlflow.log_artifact(str(config_path), artifact_path="model")

            if hasattr(self.config, 'register_model') and self.config.register_model:
                try:
                    model_name = f"bopet_{getattr(self.config.model, 'name', 'unknown').replace('/', '_')}"
                    mlflow.pytorch.log_model(
                        pytorch_model=model,
                        artifact_path="registered_model",
                        registered_model_name=model_name
                    )
                    logger.info(f"📋 Model registered as: {model_name}")
                except Exception as register_error:
                    logger.warning(f"Model registration failed: {register_error}")

            logger.info(f"📦 Model checkpoint logged to MLflow")

        except Exception as e:
            logger.error(f"❌ Model checkpoint logging failed: {e}")

    def end_run(self, status: str = "COMPLETED", error_message: Optional[str] = None):
        """End MLflow run with comprehensive summary."""
        if not getattr(self.config, 'use_mlflow', True) or not self.run:
            return

        try:
            self.flush_metrics()

            end_time = datetime.now()
            duration_seconds = (end_time - self.start_time).total_seconds()

            mlflow.log_metrics({
                "run_duration_seconds": duration_seconds,
                "peak_gpu_memory_gb": self.peak_gpu_memory_gb,
                "total_batches_logged": len(self.batch_metrics_history),
            })

            mlflow.log_params({
                "run_status": status,
                "run_end_time": end_time.isoformat(),
                "run_duration_hours": duration_seconds / 3600,
            })

            if error_message:
                mlflow.log_param("error_message", error_message[:500])

            if self.rank_allocation_history:
                summary = {
                    'total_epochs': len(self.rank_allocation_history),
                    'final_rank': self.rank_allocation_history[-1]['total_rank'] if self.rank_allocation_history else 0,
                    'rank_changes': sum(1 for entry in self.rank_allocation_history if entry.get('rank_change_count', 0) > 0),
                }
                mlflow.log_params({f"progressive_summary_{k}": v for k, v in summary.items()})

            mlflow.end_run()

            logger.info(f"📊 MLflow run {status}: {self.run_id}")
            logger.info(f"   Duration: {duration_seconds:.1f}s ({duration_seconds/3600:.2f}h)")
            logger.info(f"   Peak GPU Memory: {self.peak_gpu_memory_gb:.2f} GB")

            self.run = None
            self.run_id = None

        except Exception as e:
            logger.error(f"❌ Failed to end MLflow run: {e}")
            try:
                mlflow.end_run()
            except:
                pass

    def log_custom_metric(self, name: str, value: float, step: Optional[int] = None):
        """Helper to log custom metrics."""
        if not getattr(self.config, 'use_mlflow', True) or not self.run:
            return

        try:
            if step is not None:
                mlflow.log_metric(name, float(value), step=step)
            else:
                mlflow.log_metric(name, float(value))
        except Exception as e:
            logger.debug(f"Failed to log custom metric {name}: {e}")


# Factory function for backward compatibility
def create_mlflow_enhancer(config, run_name: Optional[str] = None) -> Optional[MLflowEnhancer]:
    """Factory function to create MLflow enhancer with error handling."""
    if not getattr(config, 'use_mlflow', True):
        return None

    try:
        return MLflowEnhancer(config, run_name)
    except Exception as e:
        logger.error(f"Failed to create MLflow enhancer: {e}")
        return None