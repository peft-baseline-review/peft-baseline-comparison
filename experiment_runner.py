#!/usr/bin/env python
"""
Scientific Experiment Runner with MLflow Integration - PRODUCTION READY v3.1
Complete integration with scientific hyperparameter calculator
Enhanced for SCI journal reproducibility standards
"""

import os
import sys
import json
import logging
import traceback
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import warnings

import mlflow
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader

# Ensure the repository root is importable regardless of the CWD.
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from config.experiment import ExperimentConfig
except ImportError as e:
    raise ImportError(
        f"Failed to import configuration modules: {e}. "
        f"Ensure the repository root is in PYTHONPATH and dependencies "
        f"are installed."
    ) from e

# Core modules (must exist)
from src.data.dataloader import TigrinyaDataLoader
from src.models.peft_factory import EnhancedPEFTModelFactory as PEFTModelFactory
from src.training.trainer import BOPETTrainer
from src.evaluation.evaluator import BOPETEvaluator

# Optional modules
try:
    from src.utils.hyperparameter_adapter import ScientificHyperparameterAdapter
    HYPERPARAM_ADAPTER_AVAILABLE = True
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"Hyperparameter adapter not available: {e}")
    HYPERPARAM_ADAPTER_AVAILABLE = False

try:
    from src.utils.mlflow_enhancer import MLflowEnhancer
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s',
    handlers=[
        logging.FileHandler('experiment_runner.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ScientificExperimentRunner:
    """
    Production-grade experiment runner with scientific hyperparameter optimization.
    
    Implements:
    1. Reproducible experimental setup (seeding, hardware detection)
    2. Scientifically calculated hyperparameters (based on dataset/model statistics)
    3. Comprehensive MLflow tracking for experiment provenance
    4. Graceful degradation with fallback strategies
    5. Detailed error reporting for debugging
    """
    
    def __init__(self, config_path: str):
        """
        Initialize experiment runner with configuration.
        
        Args:
            config_path: Path to experiment configuration YAML
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValidationError: If configuration is invalid
            ImportError: If required modules are missing
        """
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        self.experiment_config = self._load_config()
        self.experiment_dir = self._setup_experiment_directory()
        
        self.mlflow_enhancer = None
        
        # Set reproducibility FIRST
        self._set_seeds(self.experiment_config.seed)
        
        # Hardware detection and validation
        self._detect_and_validate_hardware()
        
        # Performance monitoring
        self.experiment_start = datetime.now()
        
        logger.info("=" * 80)
        logger.info("🧪 SCIENTIFIC EXPERIMENT RUNNER v3.1")
        logger.info("=" * 80)
        logger.info(f"Experiment: {self.experiment_config.experiment_name}")
        logger.info(f"Study ID: {self.experiment_config.study_id}")
        logger.info(f"Hypothesis: {self.experiment_config.hypothesis}")
        logger.info(f"PEFT Method: {self.experiment_config.peft.method.value}")
        logger.info(f"Model: {self.experiment_config.model.name}")
        logger.info(f"Output directory: {self.experiment_dir}")
        
    def _load_config(self) -> ExperimentConfig:
        """Load and validate experiment configuration."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            experiment_config = ExperimentConfig(**config_data)
            logger.info("✅ Configuration loaded and validated successfully")
            logger.info(f"   Task: {experiment_config.task.task_type.value}")
            logger.info(f"   Classes: {experiment_config.task.num_labels}")
            logger.info(f"   Model Architecture: {experiment_config.model.architecture.value}")
            return experiment_config
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            raise
    
    def _setup_experiment_directory(self) -> Path:
        """Create structured experiment directory."""
        if hasattr(self.experiment_config, 'get_experiment_path'):
            experiment_dir = self.experiment_config.get_experiment_path()
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            experiment_dir = Path("experiments") / self.experiment_config.study_id / \
                           f"{self.experiment_config.experiment_name}_{timestamp}"
        
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Save original configuration
        config_save_path = experiment_dir / "original_config.yaml"
        with open(config_save_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.experiment_config.model_dump(exclude_none=True), f, 
                     default_flow_style=False, sort_keys=False)
        
        # Create subdirectories
        subdirs = ["checkpoints", "artifacts", "logs", "results", "metrics", "visualizations"]
        for subdir in subdirs:
            (experiment_dir / subdir).mkdir(exist_ok=True)
        
        self._save_environment_info(experiment_dir)
        
        # Apply Windows-specific optimizations
        import platform
        if platform.system() == 'Windows':
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
            os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.enabled = True
            logger.info("Windows: Applied memory optimization settings")
        
        logger.info(f"📁 Experiment directory created: {experiment_dir}")
        return experiment_dir
    
    def _save_environment_info(self, experiment_dir: Path):
        """Save environment information for reproducibility."""
        env_info = {
            "timestamp": datetime.now().isoformat(),
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "command": " ".join(sys.argv),
            "working_directory": str(Path.cwd()),
            "git_hash": self._get_git_hash() if self._is_git_repo() else "N/A"
        }
        env_path = experiment_dir / "environment.json"
        with open(env_path, 'w', encoding='utf-8') as f:
            json.dump(env_info, f, indent=2)
    
    def _get_git_hash(self) -> str:
        try:
            import subprocess
            result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
            return result.stdout.strip()
        except:
            return "N/A"
    
    def _is_git_repo(self) -> bool:
        try:
            import subprocess
            result = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], capture_output=True, text=True)
            return result.returncode == 0
        except:
            return False
    
    def _set_seeds(self, seed: int):
        """Set all random seeds for reproducibility."""
        import random
        logger.info(f"🔧 Setting random seed: {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except:
                logger.warning("Deterministic algorithms not fully supported")
        os.environ['PYTHONHASHSEED'] = str(seed)
    
    def _detect_and_validate_hardware(self):
        """Detect hardware and adjust configuration if needed."""
        import psutil
        logger.info("🔍 Detecting hardware...")
        cpu_cores = psutil.cpu_count(logical=False)
        cpu_threads = psutil.cpu_count(logical=True)
        cpu_memory = psutil.virtual_memory().total / 1e9
        logger.info(f"💻 CPU: {cpu_cores} cores, {cpu_threads} threads, {cpu_memory:.1f} GB RAM")
        
        requested_device = self.experiment_config.device
        actual_device = requested_device
        if requested_device == "cuda":
            if not torch.cuda.is_available():
                logger.warning("⚠️ CUDA requested but not available. Switching to CPU.")
                actual_device = "cpu"
                self.experiment_config.device = "cpu"
            else:
                gpu_count = torch.cuda.device_count()
                gpu_info = []
                for i in range(gpu_count):
                    props = torch.cuda.get_device_properties(i)
                    gpu_info.append({
                        "name": props.name,
                        "memory_gb": props.total_memory / 1e9,
                        "compute": f"{props.major}.{props.minor}"
                    })
                logger.info(f"🎮 GPU: {gpu_count} device(s) available")
                for i, info in enumerate(gpu_info):
                    logger.info(f"     GPU {i}: {info['name']} ({info['memory_gb']:.1f} GB)")
                total_gpu_memory = sum(g['memory_gb'] for g in gpu_info)
                if total_gpu_memory < 8:
                    logger.info("🔧 Small GPU memory detected, enabling gradient accumulation")
                    if self.experiment_config.training.gradient_accumulation_steps < 2:
                        self.experiment_config.training.gradient_accumulation_steps = 2
        
        if actual_device == "cpu" and self.experiment_config.task.batch_size > 32:
            logger.warning(f"⚠️ Large batch size ({self.experiment_config.task.batch_size}) for CPU training")
    
    def _setup_mlflow(self):
        """Initialize enhanced MLflow tracking with fallback."""
        if not self.experiment_config.use_mlflow:
            logger.info("📊 MLflow tracking disabled by configuration")
            return
        if not MLFLOW_AVAILABLE:
            logger.warning("⚠️ MLflow not available, disabling tracking")
            self.experiment_config.use_mlflow = False
            return
        
        try:
            # End any active run
            if mlflow.active_run():
                mlflow.end_run()
            
            self.mlflow_enhancer = MLflowEnhancer(
                config=self.experiment_config,
                run_name=f"{self.experiment_config.experiment_name}_{datetime.now().strftime('%H%M%S')}"
            )
            mlflow.log_artifact(str(self.config_path), artifact_path="configs")
            logger.info(f"📊 MLflow run started: {self.mlflow_enhancer.run_id}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize MLflow: {e}")
            logger.info("Continuing without MLflow tracking")
            self.experiment_config.use_mlflow = False
            self.mlflow_enhancer = None
    
    def _load_data(self) -> tuple:
        """Load and prepare dataset with comprehensive statistics."""
        logger.info("📊 Loading dataset...")
        try:
            data_loader = TigrinyaDataLoader(
                config=self.experiment_config,
                use_cache=True,
                cache_manager=None
            )
            self._validate_labels(data_loader)
            
            train_loader, val_loader, test_loader = data_loader.create_dataloaders()
            data_stats = data_loader.get_data_stats()
            
            logger.info("📈 Dataset Statistics:")
            logger.info(f"  Train samples: {data_stats.get('train_samples', 0):,}")
            logger.info(f"  Validation samples: {len(val_loader.dataset) if val_loader else 0:,}")
            logger.info(f"  Test samples: {len(test_loader.dataset):,}")
            logger.info(f"  Classes: {data_stats.get('num_classes', 0)}")
            
            return train_loader, val_loader, test_loader, data_loader, data_stats
        except Exception as e:
            logger.error(f"❌ Failed to load data: {e}")
            raise

    def _validate_labels(self, dataloader):
        """
        Verify label consistency across dataset splits and model configuration.
        More flexible: does not require contiguous labels starting at 0,
        but checks that all labels are within the expected set and that
        model.num_labels matches the number of unique labels.
        """
        train_ds, val_ds, test_ds = dataloader.load_datasets()
        
        def extract_labels(ds):
            if hasattr(ds, "labels"):
                return ds.labels.tolist() if isinstance(ds.labels, torch.Tensor) else list(ds.labels)
            return []
        
        train_labels = extract_labels(train_ds)
        val_labels = extract_labels(val_ds)
        test_labels = extract_labels(test_ds)
        
        unique_labels = sorted(set(train_labels + val_labels + test_labels))
        expected_labels = list(range(len(dataloader.task_config.label_names)))
        
        # Check that all unique labels are within the expected range
        for label in unique_labels:
            if label not in expected_labels:
                logger.warning(f"Label {label} not in expected set {expected_labels}")
                # Not raising, just warning - dataset may have missing classes
        
        # Check that model.num_labels matches the number of label_names
        if hasattr(dataloader.config.model, "num_labels"):
            model_labels = getattr(dataloader.config.model, "num_labels")
            if model_labels != len(dataloader.task_config.label_names):
                raise AssertionError(
                    f"Model num_labels={model_labels} does not match len(label_names)={len(dataloader.task_config.label_names)}"
                )
            logger.info(f"✔ Model num_labels matches label_names ({model_labels})")
        
        logger.info("✅ Label validation passed (warnings may indicate missing classes).")
        return True

    def _create_model(self) -> tuple:
        """Create model with PEFT configuration and parameter analysis."""
        logger.info("🏗️ Creating model...")
        try:
            model = PEFTModelFactory.create_model(
                model_config=self.experiment_config.model,
                peft_config=self.experiment_config.peft,
                num_labels=self.experiment_config.task.num_labels,
                device=torch.device(self.experiment_config.device)
            )
            param_counts = PEFTModelFactory.count_parameters_with_allocation(model)
            logger.info("📊 Model Parameters:")
            logger.info(f"  Total parameters: {param_counts.get('total', 0):,}")
            logger.info(f"  Trainable parameters: {param_counts.get('trainable', 0):,}")
            logger.info(f"  Trainable percentage: {param_counts.get('trainable_percentage', 0):.2f}%")
            return model, param_counts
        except Exception as e:
            logger.error(f"❌ Model creation failed: {e}")
            raise

    def _calculate_optimal_hyperparameters(self, data_stats: Dict, model_param_count: int) -> Dict:
        """
        Calculate scientifically optimal hyperparameters and apply them to config.
        """
        if not HYPERPARAM_ADAPTER_AVAILABLE:
            logger.warning("⚠️ Hyperparameter adapter not available, using configuration defaults")
            return {}
        
        try:
            logger.info("🔬 Calculating optimal hyperparameters...")
            adapter = ScientificHyperparameterAdapter()
            seq_stats = data_stats.get('text_length_stats', {})
            optimal_params = adapter.calculate_optimal_params(
                experiment_config=self.experiment_config,
                dataset_size=data_stats.get('train_samples', 0),
                model_param_count=model_param_count,
                sequence_length_stats=seq_stats
            )
            # Apply optimized parameters to config
            if 'batch_size' in optimal_params:
                self.experiment_config.task.batch_size = optimal_params['batch_size']
            if 'learning_rate' in optimal_params:
                self.experiment_config.training.learning_rate = optimal_params['learning_rate']
            if 'effective_sequence_length' in optimal_params:
                self.experiment_config.task.max_length = optimal_params['effective_sequence_length']
            
            logger.info("✅ Applied optimized hyperparameters:")
            if 'batch_size' in optimal_params:
                logger.info(f"  Batch size: {optimal_params['batch_size']}")
            if 'learning_rate' in optimal_params:
                logger.info(f"  Learning rate: {optimal_params['learning_rate']:.2e}")
            
            return optimal_params
        except Exception as e:
            logger.warning(f"⚠️ Hyperparameter optimization failed: {e}")
            return {}

    def _create_trainer(self, model, train_loader, val_loader, data_loader):
        """Create trainer with risk mitigation and scheduler."""
        logger.info("👨‍🏫 Setting up trainer...")
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.experiment_config.training.learning_rate,
            weight_decay=self.experiment_config.training.weight_decay,
            betas=self.experiment_config.training.betas,
            eps=self.experiment_config.training.eps
        )
        
        from transformers import get_scheduler
        
        scheduler_name = self.experiment_config.training.lr_scheduler
        # Map to proper scheduler names
        scheduler_mapping = {
                'linear_warmup_cosine_decay': 'cosine',
                'cosine_warmup': 'cosine',
                'linear_warmup': 'linear',
                'cosine_decay': 'cosine',
            }
 
        scheduler_name = scheduler_mapping.get(scheduler_name, scheduler_name)
        
        # Calculate steps
        batches_per_epoch = len(train_loader)
        grad_acc = self.experiment_config.training.gradient_accumulation_steps
        optimizer_steps_per_epoch = max(1, batches_per_epoch // grad_acc + (1 if batches_per_epoch % grad_acc != 0 else 0))
        total_optimizer_steps = optimizer_steps_per_epoch * self.experiment_config.training.num_epochs
        warmup_steps = int(total_optimizer_steps * self.experiment_config.training.warmup_ratio)
        
        logger.info(f"  Total optimizer steps: {total_optimizer_steps:,}, Warmup steps: {warmup_steps:,}")
        
        try:
            scheduler = get_scheduler(
                name=scheduler_name,
                optimizer=optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_optimizer_steps
            )
        except ValueError as e:
            logger.warning(f"Failed to create scheduler '{scheduler_name}': {e}. Falling back to cosine.")
            scheduler = get_scheduler(
                name='cosine',
                optimizer=optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_optimizer_steps
            )
        
        # Class weights for imbalanced datasets
        class_weights = None
        if self.experiment_config.task.use_class_weights:
            try:
                class_weights = data_loader.get_class_weights()
                logger.info("  Using class weights")
            except Exception as e:
                logger.warning(f"Could not compute class weights: {e}")
        
        # Risk mitigation
        risk_instances = None
        try:
            from src.utils.risk_mitigation import create_risk_mitigation_instances
            risk_instances = create_risk_mitigation_instances(self.experiment_config)
            logger.info("  Risk mitigation enabled")
        except ImportError:
            logger.info("  Risk mitigation not available")
        
        trainer = BOPETTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=torch.device(self.experiment_config.device),
            config=self.experiment_config,
            class_weights=class_weights,
            risk_mitigation_instances=risk_instances,
            mlflow_enhancer=self.mlflow_enhancer
        )
        return trainer

    def run(self) -> Dict[str, Any]:
        """Execute complete experiment with scientific rigor."""
        logger.info("🚀 Starting experiment execution...")
        
        results = {
            "experiment_metadata": {
                "experiment_id": self.experiment_config.experiment_id,
                "name": self.experiment_config.experiment_name,
                "study_id": self.experiment_config.study_id,
                "hypothesis": self.experiment_config.hypothesis,
                "start_time": self.experiment_start.isoformat(),
                "config_path": str(self.config_path),
                "output_dir": str(self.experiment_dir),
                "reproducibility_info": {
                    "seed": self.experiment_config.seed,
                    "deterministic": torch.backends.cudnn.deterministic,
                    "git_hash": self._get_git_hash()
                }
            },
            "hardware_info": self._get_hardware_info(),
            "results": {},
            "errors": []
        }
        
        model = None
        trainer = None
        
        try:
            # Phase 1: MLflow
            if self.experiment_config.use_mlflow:
                self._setup_mlflow()

            # Reset CUDA peak memory stats for this experiment
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                logger.debug("Reset CUDA peak memory stats.")
            
            # Phase 2: Data loading
            train_loader, val_loader, test_loader, data_loader, data_stats = self._load_data()
            results["data_statistics"] = data_stats
            
            # Get a sample batch for inference stats (only if atomic inference will be used)
            sample_batch = None
            try:
                test_iter = iter(test_loader)
                sample_batch = next(test_iter)
            except StopIteration:
                logger.warning("Test loader is empty, cannot get sample batch.")
            
            if self.mlflow_enhancer:
                self.mlflow_enhancer.log_experiment_params(
                    additional_params={
                        'experiment_id': self.experiment_config.experiment_id,
                        'study_id': self.experiment_config.study_id,
                    },
                    data_stats=data_stats
                )
            
            # Phase 3: Model creation
            model, param_counts = self._create_model()
            if hasattr(model, "get_progressive_summary"):
                logger.info(model.get_progressive_summary())
            results["model_statistics"] = param_counts
            
            # Phase 4: Hyperparameter optimization (only if enabled)
            if self.experiment_config.peft.optimize_hyperparams:
                optimal_params = self._calculate_optimal_hyperparameters(
                    data_stats, param_counts.get('total', 0)
                )
            else:
                optimal_params = {}
                logger.info("🔧 Hyperparameter optimization disabled (using config defaults)")
            if optimal_params:
                results["optimal_hyperparameters"] = optimal_params
            
            # Phase 5: Trainer setup
            trainer = self._create_trainer(model, train_loader, val_loader, data_loader)
            
            # Phase 6: Training
            training_history = self._run_training(trainer)
            results["training_history"] = training_history
          
            # Phase 7: Evaluation
            test_results, model_metrics, efficiency_metrics, inference_metrics, report = self._run_evaluation(
                trainer, test_loader, model, sample_batch
            )
            
            # Build final results
            results["results"] = {
                "model_metrics": model_metrics,
                "efficiency_metrics": efficiency_metrics,
                "inference_metrics": inference_metrics,
                "param_counts": param_counts,
                "optimal_hyperparameters": optimal_params,
                "training_history": training_history,
                "experiment_report": report,
                "predictions": {
                    "labels": test_results.get('labels', []),
                    "predictions": test_results.get('predictions', [])
                }
            }
            
            # Phase 8: MLflow logging
            if self.mlflow_enhancer:
                self._log_to_mlflow(results)
            
            # Phase 9: Save results
            self._save_results(results)
            
            experiment_end = datetime.now()
            duration = experiment_end - self.experiment_start
            results["experiment_metadata"]["end_time"] = experiment_end.isoformat()
            results["experiment_metadata"]["duration_seconds"] = duration.total_seconds()
            
            if self.mlflow_enhancer:
                self.mlflow_enhancer.end_run(status="COMPLETED")
            
            self._log_experiment_summary(results, duration)
            return results
            
        except Exception as e:
            error_info = {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "timestamp": datetime.now().isoformat()
            }
            results["errors"].append(error_info)
            logger.error(f"❌ Experiment failed: {e}")
            logger.error(traceback.format_exc())
            
            # Save partial results
            try:
                self._save_results(results)
            except Exception as save_error:
                logger.error(f"Failed to save partial results: {save_error}")
            
            if self.mlflow_enhancer:
                self.mlflow_enhancer.end_run(status="FAILED", error_message=str(e))
            
            raise
        
        finally:
            # Cleanup
            del trainer
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def _run_training(self, trainer):
        """Run training phase."""
        logger.info("🚀 Starting training phase...")
        return trainer.train(
            save_path=str(self.experiment_dir / "checkpoints" / "best_model.pt")
        )
    
    def _run_evaluation(self, trainer, test_loader, model, sample_batch):
        logger.info("📈 Starting evaluation phase...")
        with torch.no_grad():
            test_results = trainer.evaluate(
                test_loader,
                return_predictions=True,
                collect_inference_stats=True
            )
        
        evaluator = BOPETEvaluator(self.experiment_config)
        
        # Build inference metrics
        inference_metrics = {}
        atomic_info = test_results.get("__atomic_inference__") or {}
        if atomic_info.get("inference_stats") and sample_batch is not None:
            inference_metrics = evaluator.build_inference_metrics_from_trainer(
                inference_stats=atomic_info.get("inference_stats"),
                num_samples=atomic_info.get("num_samples"),
                evaluation_time_seconds=atomic_info.get("evaluation_time_seconds"),
                model=model,
                sample_batch=sample_batch
            )
        else:
            logger.warning("Atomic inference stats not available; inference metrics will be empty.")
        
        # Classification metrics
        model_metrics = evaluator.evaluate_classification(
            y_true=test_results["labels"],
            y_pred=test_results["predictions"],
            y_prob=test_results.get("probabilities"),
            label_names=self.experiment_config.task.label_names,
            task_name=self.experiment_config.experiment_name,
            inference_metrics=inference_metrics
        )
        
        # Efficiency metrics
        efficiency_metrics = evaluator.compute_efficiency_metrics(
            model=model,
            model_metrics=model_metrics,
            inference_metrics=inference_metrics
        )
        
        # Generate report
        additional_info = {
            'training_history': trainer.history if hasattr(trainer, 'history') else {},
            'rank_history': trainer.rank_history   # <-- ADD THIS
        }
        report = evaluator.generate_scientific_report(
            model_metrics=model_metrics,
            efficiency_metrics=efficiency_metrics,
            experiment_name=self.experiment_config.experiment_name,
            config=self.experiment_config.model_dump(),
            additional_info=additional_info
        )
        report['inference_metrics'] = inference_metrics
        
        # Save visualizations and export report
        figures_dir = self.experiment_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        evaluator.generate_scientific_visualizations(report, save_dir=figures_dir)
        
        exported_paths = evaluator.export_report(report, self.experiment_dir / "reports")
        report["exported_files"] = {k: str(v) for k, v in exported_paths.items()}
        
        # Now return the report as the fifth value
        return test_results, model_metrics, efficiency_metrics, inference_metrics, report
    
    def _get_hardware_info(self) -> Dict[str, Any]:
        """Get comprehensive hardware information."""
        import platform
        import psutil
        
        info = {
            "system": {
                "platform": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "processor": platform.processor(),
                "python_version": platform.python_version(),
            },
            "cpu": {
                "cores_physical": psutil.cpu_count(logical=False),
                "cores_logical": psutil.cpu_count(logical=True),
                "memory_gb": psutil.virtual_memory().total / 1e9,
                "memory_available_gb": psutil.virtual_memory().available / 1e9,
            }
        }
        
        if torch.cuda.is_available():
            info["gpu"] = {
                "count": torch.cuda.device_count(),
                "devices": []
            }
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                info["gpu"]["devices"].append({
                    "index": i,
                    "name": props.name,
                    "memory_gb": props.total_memory / 1e9,
                    "compute_capability": f"{props.major}.{props.minor}",
                    "multi_processors": props.multi_processor_count
                })
        return info
    
    def _log_to_mlflow(self, results: Dict[str, Any]):
        """Comprehensive MLflow logging using mlflow_enhancer methods."""
        try:
            # Log model metrics
            model_metrics = results["results"].get("model_metrics", {})
            efficiency_metrics = results["results"].get("efficiency_metrics", {})
            
            if hasattr(self.mlflow_enhancer, 'log_model_metrics'):
                self.mlflow_enhancer.log_model_metrics(
                    model_metrics=model_metrics,
                    efficiency_metrics=efficiency_metrics,
                    include_ci=self.experiment_config.evaluation.compute_ci
                )
            else:
                # Fallback: log each metric directly
                for k, v in model_metrics.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(f"model/{k}", v)
                for k, v in efficiency_metrics.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(f"efficiency/{k}", v)
            
            # Log training history
            training_history = results["results"].get("training_history", {})
            if training_history and hasattr(self.mlflow_enhancer, 'log_training_metrics'):
                final_epoch = len(training_history.get('train_loss', []))
                self.mlflow_enhancer.log_training_metrics(training_history, final_epoch)
            
            # Log artifacts
            if hasattr(self.mlflow_enhancer, 'log_artifacts_organized'):
                self.mlflow_enhancer.log_artifacts_organized(self.experiment_dir)
            else:
                mlflow.log_artifacts(str(self.experiment_dir))
            
            logger.info("📊 Results logged to MLflow")
        except Exception as e:
            logger.warning(f"Failed to log to MLflow: {e}")
    
    def _save_results(self, results: Dict[str, Any]):
        """Save experiment results in multiple formats."""
        # Full JSON
        results_path = self.experiment_dir / "experiment_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str, ensure_ascii=False)
        
        # Markdown summary
        summary_path = self.experiment_dir / "experiment_summary.md"
        summary = self._generate_markdown_summary(results)
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        # Metrics CSV
        metrics_df = pd.DataFrame([results["results"].get("model_metrics", {})])
        csv_path = self.experiment_dir / "metrics" / "model_metrics.csv"
        metrics_df.to_csv(csv_path, index=False)
        
        eff_df = pd.DataFrame([results["results"].get("efficiency_metrics", {})])
        eff_path = self.experiment_dir / "metrics" / "efficiency_metrics.csv"
        eff_df.to_csv(eff_path, index=False)
        
        logger.info(f"💾 Results saved to {self.experiment_dir}")
    
    def _generate_markdown_summary(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive markdown summary."""
        metadata = results["experiment_metadata"]
        model_metrics = results["results"].get("model_metrics", {})
        efficiency = results["results"].get("efficiency_metrics", {})
        optimal_params = results["results"].get("optimal_hyperparameters", {})
        training_history = results["results"].get("training_history", {})
        
        train_losses = training_history.get('train_loss', [])
        val_losses = training_history.get('val_loss', [])
        
        summary = f"""# Scientific Experiment Report

## Experiment Information
- **Experiment ID**: `{metadata.get('experiment_id', 'N/A')}`
- **Name**: {metadata.get('name', 'N/A')}
- **Study ID**: {metadata.get('study_id', 'N/A')}
- **Hypothesis**: {metadata.get('hypothesis', 'N/A')}
- **Start Time**: {metadata.get('start_time', 'N/A')}
- **End Time**: {metadata.get('end_time', 'N/A')}
- **Duration**: {metadata.get('duration_seconds', 0):.1f} seconds
- **Configuration**: `{metadata.get('config_path', 'N/A')}`
- **Output Directory**: `{metadata.get('output_dir', 'N/A')}`

## Reproducibility Information
- **Random Seed**: {metadata.get('reproducibility_info', {}).get('seed', 'N/A')}
- **Git Commit**: `{metadata.get('reproducibility_info', {}).get('git_hash', 'N/A')}`
- **Deterministic Operations**: {metadata.get('reproducibility_info', {}).get('deterministic', 'N/A')}

## Model Configuration
- **Base Model**: {self.experiment_config.model.name}
- **Architecture**: {self.experiment_config.model.architecture.value}
- **PEFT Method**: {self.experiment_config.peft.method.value}
- **Rank**: {self.experiment_config.peft.rank}
- **Alpha**: {self.experiment_config.peft.alpha}
- **Dropout**: {self.experiment_config.peft.dropout}

## Dataset Information
- **Task Type**: {self.experiment_config.task.task_type.value}
- **Number of Classes**: {self.experiment_config.task.num_labels}
- **Train Samples**: {results.get('data_statistics', {}).get('train_samples', 0):,}
- **Test Samples**: {results.get('data_statistics', {}).get('test_samples', 0):,}

## Scientifically Optimized Hyperparameters
"""
        if optimal_params:
            summary += f"""
| Parameter | Value |
|-----------|-------|
| Batch Size | {optimal_params.get('batch_size', 'Default')} |
| Learning Rate | {optimal_params.get('learning_rate', 0):.2e} |
| Sequence Length | {optimal_params.get('effective_sequence_length', 'Default')} |
| Estimated Training Time | {optimal_params.get('estimated_training_time_hours', 0):.1f} hours |
"""
        else:
            summary += "\n*Using configuration defaults*\n"
        
        summary += f"""
## Performance Metrics

### Classification Metrics
| Metric | Value |
|--------|-------|
| Accuracy | {model_metrics.get('accuracy', 0):.4f} |
| F1 Weighted | {model_metrics.get('f1_weighted', 0):.4f} |
| F1 Macro | {model_metrics.get('f1_macro', 0):.4f} |
| Precision Macro | {model_metrics.get('precision_macro', 0):.4f} |
| Recall Macro | {model_metrics.get('recall_macro', 0):.4f} |

### Efficiency Metrics
| Metric | Value |
|--------|-------|
| Trainable Parameters | {efficiency.get('trainable_params', 0):,} |
| Total Parameters | {efficiency.get('total_params', 0):,} |
| Trainable Percentage | {efficiency.get('trainable_percentage', 0):.2f}% |
| Model Size | {efficiency.get('model_size_mb', 0):.2f} MB |
| Compression Ratio | {efficiency.get('compression_ratio', 0):.2f} |
| Composite Score | {results['results'].get('experiment_report', {}).get('overall_scores', {}).get('composite_score', 0):.4f} |

## Training History
- **Final Training Loss**: {train_losses[-1] if train_losses else 'N/A'}
- **Final Validation Loss**: {val_losses[-1] if val_losses else 'N/A'}
- **Training Epochs**: {len(train_losses)}

## Hardware Information
{json.dumps(results.get('hardware_info', {}), indent=2)}

---

*Generated by Scientific Experiment Runner v3.1 on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        return summary
    
    def _log_experiment_summary(self, results: Dict[str, Any], duration):
        """Log final experiment summary."""
        model_metrics = results["results"].get("model_metrics", {})
        efficiency = results["results"].get("efficiency_metrics", {})
        
        logger.info("=" * 80)
        logger.info("✅ EXPERIMENT COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info(f"Duration: {str(duration)}")
        logger.info(f"Accuracy: {model_metrics.get('accuracy', 0):.4f}")
        logger.info(f"F1 Weighted: {model_metrics.get('f1_weighted', 0):.4f}")
        logger.info(f"Trainable Parameters: {efficiency.get('trainable_params', 0):,}")
        logger.info(f"Trainable Percentage: {efficiency.get('trainable_percentage', 0):.2f}%")
        logger.info(f"Composite Score: {results['results'].get('experiment_report', {}).get('overall_scores', {}).get('composite_score', 0):.4f}")
        logger.info(f"Results saved to: {str(self.experiment_dir)}")
        if self.mlflow_enhancer:
            logger.info(f"📊 MLflow Run ID: {self.mlflow_enhancer.run_id}")


# Main execution guard
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run scientific experiment")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to experiment configuration YAML")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        runner = ScientificExperimentRunner(args.config)
        results = runner.run()
        print(f"\n✅ Experiment completed. Results saved to: {runner.experiment_dir}")
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        sys.exit(1)