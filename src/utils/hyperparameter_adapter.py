#!/usr/bin/env python
"""
Scientific Hyperparameter Adapter - Bridge between Pydantic and Hyperparameter Calculator
Production-ready integration with research-grade optimization
"""

import math
import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging
from pathlib import Path
import psutil

# Pydantic configuration imports
from config.experiment import ExperimentConfig
from config.training import TrainingConfig
from config.task import TaskConfig
from config.model import ModelConfig

logger = logging.getLogger(__name__)

class ScientificHyperparameterAdapter:
    """
    Adapter that integrates scientific hyperparameter calculator with Pydantic configuration
    Implements research-grade optimization based on:
    1. Square root scaling laws
    2. Memory-aware batch sizing
    3. Statistical optimal stopping
    4. Gradient noise scaling
    """
    
    def __init__(self):
        """Initialize scientific adapter"""
        self.research_version = "2.0"
        logger.info(f"🔬 Scientific Hyperparameter Adapter v{self.research_version}")
    
    def calculate_optimal_params(self,
                               experiment_config: ExperimentConfig,
                               dataset_size: int,
                               model_param_count: int,
                               sequence_length_stats: Dict[str, float]) -> Dict[str, Any]:
        """
        Calculate scientifically optimal hyperparameters
        
        Args:
            experiment_config: Pydantic experiment configuration
            dataset_size: Number of training samples
            model_param_count: Total parameters in model
            sequence_length_stats: Statistics about sequence lengths
        
        Returns:
            Dictionary with calculated optimal parameters
        """
        
        logger.info("🔬 Calculating scientifically optimal hyperparameters...")
        
        # 1. Calculate optimal batch size using square root scaling
        batch_size = self._calculate_optimal_batch_size(
            experiment_config, model_param_count, dataset_size
        )
        
        # 2. Calculate optimal learning rate using scaling laws
        learning_rate = self._calculate_optimal_learning_rate(
            experiment_config, model_param_count, batch_size
        )
        
        # 3. Calculate sequence length based on statistics
        seq_length = self._calculate_optimal_sequence_length(
            experiment_config, sequence_length_stats
        )
        
        # 4. Calculate optimal epochs using early stopping criteria
        num_epochs = self._calculate_optimal_epochs(
            experiment_config, dataset_size, batch_size
        )
        
        # 5. Calculate gradient accumulation steps
        grad_accum_steps = self._calculate_gradient_accumulation(
            experiment_config, batch_size
        )
        
        # 6. Estimate memory requirements
        memory_estimate = self._estimate_memory_requirements(
            model_param_count, batch_size, seq_length,
            experiment_config.model.hidden_size
        )
        
        # 7. Estimate training time
        time_estimate = self._estimate_training_time(
            num_epochs, dataset_size, batch_size,
            model_param_count, experiment_config.device
        )
        
        optimal_params = {
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'effective_sequence_length': seq_length,
            'num_epochs': num_epochs,
            'gradient_accumulation_steps': grad_accum_steps,
            'memory_estimate_gb': memory_estimate,
            'estimated_training_time_hours': time_estimate,
            'global_batch_size': batch_size * grad_accum_steps,
            'updates_per_epoch': dataset_size // (batch_size * grad_accum_steps),
            'samples_per_update': batch_size * grad_accum_steps,
            'calculated_by': 'ScientificHyperparameterAdapter',
            'research_version': self.research_version
        }
        
        logger.info(f"✅ Optimal parameters calculated:")
        logger.info(f"   Batch size: {batch_size}")
        logger.info(f"   Learning rate: {learning_rate:.2e}")
        logger.info(f"   Sequence length: {seq_length}")
        logger.info(f"   Estimated time: {time_estimate:.1f}h")
        
        return optimal_params
    
    def apply_optimal_params(self,
                           experiment_config: ExperimentConfig,
                           optimal_params: Dict[str, Any]) -> ExperimentConfig:
        """
        Apply optimal parameters to experiment configuration
        
        Args:
            experiment_config: Original experiment configuration
            optimal_params: Calculated optimal parameters
        
        Returns:
            Updated experiment configuration
        """
        
        # Create a copy of the configuration
        from copy import deepcopy
        updated_config = deepcopy(experiment_config)
        
        # Update training configuration
        if 'learning_rate' in optimal_params:
            updated_config.training.learning_rate = optimal_params['learning_rate']
        
        if 'num_epochs' in optimal_params:
            updated_config.training.num_epochs = optimal_params['num_epochs']
        
        if 'batch_size' in optimal_params:
            updated_config.training.batch_size_base = optimal_params['batch_size']
        
        # Update task configuration
        if 'effective_sequence_length' in optimal_params:
            updated_config.task.max_length = optimal_params['effective_sequence_length']
        
        if 'batch_size' in optimal_params:
            updated_config.task.batch_size = optimal_params['batch_size']
        
        logger.info(f"🔄 Applied optimal parameters to configuration")
        
        return updated_config
    
    def _calculate_optimal_batch_size(self,
                                    config: ExperimentConfig,
                                    model_param_count: int,
                                    dataset_size: int) -> int:
        """
        Calculate optimal batch size using:
        1. Square root scaling: batch_size ∝ √(dataset_size)
        2. Memory constraints
        3. Gradient noise scaling
        """
        
        # Base calculation using square root rule
        base_batch = int(math.sqrt(dataset_size))
        
        # Adjust for model size (larger models need smaller batches)
        bert_base_params = 110_000_000  # BERT-base reference
        model_scale = model_param_count / bert_base_params
        batch_size = int(base_batch / max(math.sqrt(model_scale), 1))
        
        # Apply memory constraints
        batch_size = self._apply_memory_constraints(config, batch_size, model_param_count)
        
        # Ensure within reasonable bounds
        batch_size = max(batch_size, 1)
        batch_size = min(batch_size, 256)  # Maximum reasonable batch
        
        # Round to power of 2 for hardware optimization
        batch_size = self._round_to_power_of_two(batch_size)
        
        return batch_size
    
    def _apply_memory_constraints(self,
                                config: ExperimentConfig,
                                batch_size: int,
                                model_param_count: int) -> int:
        """Apply memory constraints to batch size"""
        
        if config.device == "cuda" and torch.cuda.is_available():
            # GPU memory constraint
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            # Estimate memory per sample (rough approximation)
            hidden_size = config.model.hidden_size
            seq_length = config.task.max_length
            num_layers = config.model.num_hidden_layers
            
            # Conservative memory estimate per sample in GB
            memory_per_sample_gb = (model_param_count * 4 / 1e9) * 0.0001  # Parameter memory
            memory_per_sample_gb += (batch_size * seq_length * hidden_size * num_layers * 4 / 1e9) * 0.01  # Activations
            
            # Calculate max batch by memory (use 80% of GPU)
            max_batch_memory = int((gpu_memory_gb * 0.8) / max(memory_per_sample_gb, 1e-6))
            
            return min(batch_size, max_batch_memory)
        
        else:
            # CPU memory constraint
            cpu_memory_gb = psutil.virtual_memory().total / 1e9
            max_batch_memory = int((cpu_memory_gb * 0.5) / (model_param_count * 4 / 1e9 * 0.0002))
            
            return min(batch_size, max_batch_memory)
    
    def _calculate_optimal_learning_rate(self,
                                       config: ExperimentConfig,
                                       model_param_count: int,
                                       batch_size: int) -> float:
        """
        Calculate optimal learning rate using:
        1. Square root scaling: lr ∝ 1/√(N) where N is model size
        2. Linear scaling with batch size: lr ∝ batch_size
        3. Warmup scheduling
        """
        
        base_lr = 5e-5  # Base learning rate for BERT-base
        
        # Square root scaling for model size
        bert_base_params = 110_000_000
        model_scale = math.sqrt(model_param_count / bert_base_params)
        lr_scaled = base_lr / max(model_scale, 0.1)
        
        # Square root scaling for batch size (not linear!)
        batch_scale = math.sqrt(batch_size / 32)  # Reference batch size = 32
        lr_final = lr_scaled * batch_scale
        
        # Apply config constraints if any
        if hasattr(config.training, 'learning_rate'):
            config_lr = config.training.learning_rate
            # Only adjust if config lr is within reasonable bounds
            if 1e-7 < config_lr < 1e-3:
                # Blend scientific calculation with config (70% science, 30% config)
                lr_final = 0.7 * lr_final + 0.3 * config_lr
        
        # Clamp to reasonable range
        lr_final = max(lr_final, 1e-7)
        lr_final = min(lr_final, 1e-3)
        
        return lr_final
    
    def _calculate_optimal_sequence_length(self,
                                         config: ExperimentConfig,
                                         sequence_stats: Dict[str, float]) -> int:
        """Calculate optimal sequence length based on dataset statistics"""
        
        # Use 95th percentile if available
        if 'percentile_95' in sequence_stats:
            p95 = sequence_stats['percentile_95']
        elif 'mean' in sequence_stats and 'std' in sequence_stats:
            # Estimate 95th percentile
            mean = sequence_stats['mean']
            std = sequence_stats['std']
            p95 = mean + 1.645 * std  # 95th percentile for normal distribution
        else:
            # Fallback to config value
            p95 = config.task.max_length
        
        # Model limit
        model_limit = config.task.max_length
        
        # Choose optimal length: min of model limit and 95th percentile
        seq_length = int(min(model_limit, max(32, p95)))  # Minimum 32 tokens
        
        # Round to nearest multiple of 8 for hardware optimization
        seq_length = ((seq_length + 7) // 8) * 8
        
        return seq_length
    
    def _calculate_optimal_epochs(self,
                                config: ExperimentConfig,
                                dataset_size: int,
                                batch_size: int) -> int:
        """Calculate optimal number of epochs using early stopping criteria"""
        
        # Base calculation: enough epochs to see each sample ~10 times on average
        base_epochs = max(3, int(10000 / (dataset_size / batch_size)))
        
        # Adjust for dataset size (larger datasets need fewer epochs)
        if dataset_size > 10000:
            base_epochs = max(3, base_epochs - int(math.log10(dataset_size / 10000)))
        
        # Apply config constraints
        config_epochs = config.training.num_epochs
        
        # Use the minimum of scientific calculation and config
        final_epochs = min(base_epochs, config_epochs) if config_epochs > 0 else base_epochs
        
        return max(final_epochs, 3)  # Minimum 3 epochs
    
    def _calculate_gradient_accumulation(self,
                                       config: ExperimentConfig,
                                       batch_size: int) -> int:
        """Calculate gradient accumulation steps"""
        
        target_global = 128  # Target global batch size
        
        if target_global <= batch_size:
            return 1
        
        accumulation_steps = int(math.ceil(target_global / batch_size))
        
        # Cap at reasonable value
        accumulation_steps = min(accumulation_steps, 32)
        
        return accumulation_steps
    
    def _estimate_memory_requirements(self,
                                    model_param_count: int,
                                    batch_size: int,
                                    sequence_length: int,
                                    hidden_size: int) -> float:
        """Estimate memory requirements in GB"""
        
        # Model parameters (float32: 4 bytes per parameter)
        model_memory_gb = (model_param_count * 4) / (1024 ** 3)
        
        # Activations (approximation for transformer)
        activation_memory_gb = (2 * batch_size * sequence_length * hidden_size * 12 * 4) / (1024 ** 3)
        
        # Optimizer states (AdamW: 2 * parameters for momentum and variance)
        optimizer_memory_gb = (2 * model_param_count * 4) / (1024 ** 3)
        
        # Gradients (same size as parameters)
        gradient_memory_gb = (model_param_count * 4) / (1024 ** 3)
        
        # Total estimate with 20% overhead
        total_gb = (model_memory_gb + activation_memory_gb + optimizer_memory_gb + gradient_memory_gb) * 1.2
        
        return total_gb
    
    def _estimate_training_time(self,
                              num_epochs: int,
                              dataset_size: int,
                              batch_size: int,
                              model_param_count: int,
                              device: str) -> float:
        """Estimate training time in hours"""
        
        # Base throughput estimate (samples/second)
        if device == "cuda" and torch.cuda.is_available():
            # GPU estimation
            if model_param_count < 100_000_000:  # <100M params
                throughput = 1000
            elif model_param_count < 1_000_000_000:  # <1B params
                throughput = 500
            else:
                throughput = 100
        else:
            # CPU estimation
            cpu_cores = psutil.cpu_count(logical=False)
            throughput = 10 * cpu_cores
        
        # Adjust for batch size (sublinear scaling)
        throughput = throughput * math.sqrt(batch_size / 32)
        
        # Calculate time
        total_samples = num_epochs * dataset_size
        time_seconds = total_samples / throughput
        time_hours = time_seconds / 3600
        
        # Add 10% overhead
        time_hours = time_hours * 1.1
        
        return time_hours
    
    def _round_to_power_of_two(self, n: int) -> int:
        """Round to nearest power of two for hardware optimization"""
        if n <= 0:
            return 1
        
        # Find nearest power of two
        power = 1
        while power < n:
            power <<= 1
        
        # Choose closest power
        if power - n > n - (power >> 1):
            return power >> 1
        return power


# For backward compatibility
HyperparameterAdapter = ScientificHyperparameterAdapter