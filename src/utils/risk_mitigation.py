#!/usr/bin/env python
"""
Comprehensive Risk Mitigation for BOPET Experiments - UPDATED FOR PYDANTIC
"""

import os
import torch
import gc
import psutil
import signal
import sys
import time
from typing import Dict, Any, Optional, Callable, Tuple
import logging
import traceback
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

class ResourceMonitor:
    """Monitor system resources with Pydantic config support"""
    
    def __init__(self, config: Optional[Any] = None):
        """
        Initialize with Pydantic configuration
        
        Args:
            config: Can be Pydantic model, dictionary, or None
        """
        # Convert config to dictionary if it's a Pydantic model
        self.config = self._convert_to_dict(config)
        
        # Extract risk mitigation config from nested structure
        risk_config = self.config.get('risk_mitigation', {})
        if isinstance(risk_config, dict):
            # Extract from dict
            self.memory_threshold = risk_config.get('memory_threshold', 0.9)
            self.adaptive_batch_enabled = risk_config.get('batch_size_adaptation', True)
            self.memory_monitoring = risk_config.get('memory_monitoring', True)
            self.enabled = risk_config.get('enabled', True)
        elif hasattr(risk_config, '__dict__'):
            # Extract from object
            self.memory_threshold = getattr(risk_config, 'memory_threshold', 0.9)
            self.adaptive_batch_enabled = getattr(risk_config, 'batch_size_adaptation', True)
            self.memory_monitoring = getattr(risk_config, 'memory_monitoring', True)
            self.enabled = getattr(risk_config, 'enabled', True)
        else:
            # Default values
            self.memory_threshold = 0.9
            self.adaptive_batch_enabled = True
            self.memory_monitoring = True
            self.enabled = True
        
        # Initialize tracking
        self.memory_history = []
        self.batch_size_history = []
        
        logger.info(f"ResourceMonitor initialized: memory_monitoring={self.memory_monitoring}, "
                   f"batch_adaptation={self.adaptive_batch_enabled}")
    
    def _convert_to_dict(self, config: Any) -> Dict:
        """Convert config to dictionary, handling Pydantic models"""
        if config is None:
            return {}
        elif hasattr(config, 'model_dump'):
            # Pydantic v2
            return config.model_dump()
        elif hasattr(config, 'dict'):
            # Pydantic v1
            return config.dict()
        elif isinstance(config, dict):
            return config
        else:
            # Try to convert to dict via attributes
            try:
                return {k: v for k, v in config.__dict__.items() if not k.startswith('_')}
            except:
                return {}
    
    def check_memory(self, device_type: str = 'cuda') -> Dict[str, float]:
        """Check memory usage with config-driven thresholds"""
        if not self.enabled or not self.memory_monitoring:
            return {}
        
        stats = {}
        
        if device_type == 'cuda' and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            cached = torch.cuda.memory_reserved() / 1e9
            available = torch.cuda.get_device_properties(0).total_memory / 1e9
            usage_ratio = allocated / available if available > 0 else 0
            
            stats = {
                'allocated_gb': allocated,
                'cached_gb': cached,
                'available_gb': available,
                'usage_ratio': usage_ratio,
                'threshold': self.memory_threshold,
                'device': 'cuda'
            }
        else:
            # CPU memory
            memory = psutil.virtual_memory()
            stats = {
                'used_gb': memory.used / 1e9,
                'available_gb': memory.available / 1e9,
                'percent': memory.percent / 100,
                'threshold': self.memory_threshold,
                'device': 'cpu'
            }
        
        # Track history
        self.memory_history.append({
            'timestamp': time.time(),
            **stats
        })
        
        # Keep only last 100 entries
        if len(self.memory_history) > 100:
            self.memory_history = self.memory_history[-100:]
        
        return stats
    
    def should_reduce_batch_size(self) -> bool:
        """Check if batch size should be reduced based on config"""
        if not self.enabled or not self.adaptive_batch_enabled:
            return False
        
        stats = self.check_memory()
        
        if not stats:
            return False
        
        if 'usage_ratio' in stats:  # GPU
            return stats['usage_ratio'] > self.memory_threshold
        elif 'percent' in stats:  # CPU
            return stats['percent'] > self.memory_threshold
        
        return False
    
    def adaptive_batch_size(self, current_batch: int, min_batch: int = 1) -> int:
        """Calculate adaptive batch size based on memory usage"""
        if not self.enabled or not self.adaptive_batch_enabled:
            return current_batch
        
        stats = self.check_memory()
        
        if not stats:
            return current_batch
        
        # Determine reduction factor based on memory pressure
        if 'usage_ratio' in stats:  # GPU
            usage = stats['usage_ratio']
            if usage > 0.9:
                reduction = 0.5
            elif usage > 0.8:
                reduction = 0.75
            elif usage > 0.7:
                reduction = 0.9
            else:
                return current_batch
        else:  # CPU
            usage = stats['percent']
            if usage > 0.9:
                reduction = 0.5
            elif usage > 0.8:
                reduction = 0.75
            else:
                return current_batch
        
        new_batch = max(min_batch, int(current_batch * reduction))
        
        if new_batch != current_batch:
            logger.warning(f"Batch size adjusted from {current_batch} to {new_batch} "
                          f"(memory usage: {usage:.1%})")
            self.batch_size_history.append({
                'timestamp': time.time(),
                'old_batch': current_batch,
                'new_batch': new_batch,
                'memory_usage': usage
            })
        
        return new_batch
    
    def clear_memory(self):
        """Clear unused memory"""
        if not self.enabled:
            return
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        gc.collect()
        logger.debug("Memory cleared")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get resource monitoring summary"""
        return {
            'enabled': self.enabled,
            'memory_monitoring': self.memory_monitoring,
            'adaptive_batch': self.adaptive_batch_enabled,
            'threshold': self.memory_threshold,
            'memory_history_samples': len(self.memory_history),
            'batch_adjustments': len(self.batch_size_history),
            'config': self.config.get('risk_mitigation', {})
        }


class ExperimentRecovery:
    """Handle experiment interruption and recovery with Pydantic config"""
    
    def __init__(self, config: Optional[Any] = None):
        # Convert config to dictionary if it's a Pydantic model
        self.config = self._convert_to_dict(config)
        
        # Extract recovery config
        recovery_config = self.config.get('reproducibility', {})
        risk_config = self.config.get('risk_mitigation', {})
        
        # Set parameters
        self.checkpoint_dir = Path(risk_config.get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_freq = risk_config.get('checkpoint_frequency', 10)
        self.save_environment = recovery_config.get('save_environment_snapshot', True)
        self.save_code = recovery_config.get('save_code_snapshot', True)
        
        # Experiment info from config
        self.experiment_name = self.config.get('experiment_name', 'unknown_experiment')
        self.experiment_dir = Path(self.config.get('experiment_dir', 'results'))
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup signal handling
        self.setup_signal_handlers()
        
        self.last_checkpoint = None
        
        logger.info(f"ExperimentRecovery initialized: checkpoint_dir={self.checkpoint_dir}, "
                   f"checkpoint_freq={self.checkpoint_freq}")
    
    def _convert_to_dict(self, config: Any) -> Dict:
        """Convert config to dictionary, handling Pydantic models"""
        if config is None:
            return {}
        elif hasattr(config, 'model_dump'):
            # Pydantic v2
            return config.model_dump()
        elif hasattr(config, 'dict'):
            # Pydantic v1
            return config.dict()
        elif isinstance(config, dict):
            return config
        else:
            # Try to convert to dict via attributes
            try:
                return {k: v for k, v in config.__dict__.items() if not k.startswith('_')}
            except:
                return {}
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(sig, frame):
            logger.warning(f"Received signal {sig}, attempting graceful shutdown...")
            self.save_recovery_state()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def save_checkpoint(self, state: Dict[str, Any], 
                       filename: Optional[str] = None) -> str:
        """Save experiment checkpoint with config-driven naming"""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"checkpoint_{self.experiment_name}_{timestamp}.pt"
        
        checkpoint_path = self.checkpoint_dir / filename
        
        # Add config to state
        state['config'] = self.config
        state['experiment_info'] = {
            'name': self.experiment_name,
            'dir': str(self.experiment_dir),
            'timestamp': time.time()
        }
        
        try:
            torch.save(state, checkpoint_path)
            self.last_checkpoint = checkpoint_path
            logger.info(f"Checkpoint saved to {checkpoint_path}")
            
            # Cleanup old checkpoints
            self.cleanup_old_checkpoints(keep_last=3)
            
            return str(checkpoint_path)
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise
    
    def load_checkpoint(self, filename: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Load experiment checkpoint"""
        if filename is None and self.last_checkpoint:
            checkpoint_path = self.last_checkpoint
        elif filename:
            checkpoint_path = self.checkpoint_dir / filename
        else:
            # Find latest checkpoint
            checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.pt"))
            if not checkpoints:
                return None
            checkpoint_path = max(checkpoints, key=os.path.getctime)
        
        if checkpoint_path.exists():
            try:
                state = torch.load(checkpoint_path, map_location='cpu')
                logger.info(f"Loaded checkpoint from {checkpoint_path}")
                
                # Update config if checkpoint has newer config
                if 'config' in state:
                    logger.info("Checkpoint contains config, merging if needed")
                
                return state
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")
        
        return None
    
    def save_recovery_state(self):
        """Save recovery state on interruption"""
        recovery_state = {
            'timestamp': time.time(),
            'checkpoint': str(self.last_checkpoint) if self.last_checkpoint else None,
            'experiment_name': self.experiment_name,
            'experiment_dir': str(self.experiment_dir),
            'python_version': sys.version,
            'torch_version': torch.__version__,
            'config': self.config,
            'command': ' '.join(sys.argv)
        }
        
        recovery_path = self.checkpoint_dir / "recovery_state.json"
        import json
        with open(recovery_path, 'w', encoding='utf-8') as f:
            json.dump(recovery_state, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Recovery state saved to {recovery_path}")
        
        # Save environment snapshot if configured
        if self.save_environment:
            self.save_environment_snapshot()
        
        # Save code snapshot if configured
        if self.save_code:
            self.save_code_snapshot()
    
    def save_environment_snapshot(self):
        """Save environment snapshot"""
        try:
            import subprocess
            env_path = self.checkpoint_dir / "environment.txt"
            with open(env_path, 'w') as f:
                subprocess.run(["pip", "freeze"], stdout=f, check=True)
            logger.info(f"Environment snapshot saved to {env_path}")
        except Exception as e:
            logger.warning(f"Could not save environment snapshot: {e}")
    
    def save_code_snapshot(self):
        """Save code snapshot"""
        try:
            import shutil
            code_snapshot_dir = self.checkpoint_dir / "code_snapshot"
            code_snapshot_dir.mkdir(exist_ok=True)
            
            # Copy key directories
            for dir_name in ['src', 'configs', 'scripts']:
                if Path(dir_name).exists():
                    shutil.copytree(dir_name, code_snapshot_dir / dir_name, 
                                  dirs_exist_ok=True)
            
            # Copy main files
            for file_name in ['main_hydra.py', 'requirements.txt', 'setup.py']:
                if Path(file_name).exists():
                    shutil.copy2(file_name, code_snapshot_dir / file_name)
            
            logger.info(f"Code snapshot saved to {code_snapshot_dir}")
        except Exception as e:
            logger.warning(f"Could not save code snapshot: {e}")
    
    def cleanup_old_checkpoints(self, keep_last: int = 3):
        """Clean up old checkpoints"""
        try:
            checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.pt"),
                               key=os.path.getctime)
            if len(checkpoints) > keep_last:
                for old_checkpoint in checkpoints[:-keep_last]:
                    old_checkpoint.unlink()
                    logger.debug(f"Removed old checkpoint: {old_checkpoint}")
        except Exception as e:
            logger.warning(f"Failed to clean up old checkpoints: {e}")


class RetryManager:
    """Manage retry logic for failed trials with Pydantic config"""
    
    def __init__(self, config: Optional[Any] = None):
        # Convert config to dictionary if it's a Pydantic model
        self.config = self._convert_to_dict(config)
        
        # Extract retry config
        risk_config = self.config.get('risk_mitigation', {})
        optimization_config = self.config.get('optimization', {})
        
        # Set parameters
        self.max_retries = risk_config.get('max_retries', 3)
        self.backoff_factor = risk_config.get('backoff_factor', 2.0)
        self.retryable_errors = (RuntimeError, MemoryError, torch.cuda.OutOfMemoryError)
        
        # Trial-specific retry config
        self.enable_trial_retry = optimization_config.get('enable_retry', True)
        
        logger.info(f"RetryManager initialized: max_retries={self.max_retries}, "
                   f"backoff_factor={self.backoff_factor}")
    
    def _convert_to_dict(self, config: Any) -> Dict:
        """Convert config to dictionary, handling Pydantic models"""
        if config is None:
            return {}
        elif hasattr(config, 'model_dump'):
            # Pydantic v2
            return config.model_dump()
        elif hasattr(config, 'dict'):
            # Pydantic v1
            return config.dict()
        elif isinstance(config, dict):
            return config
        else:
            # Try to convert to dict via attributes
            try:
                return {k: v for k, v in config.__dict__.items() if not k.startswith('_')}
            except:
                return {}
    
    @contextmanager
    def retry_context(self, trial_params: Dict[str, Any], 
                     error_handler: Optional[Callable] = None):
        """Context manager for retry logic"""
        if not self.enable_trial_retry:
            yield trial_params
            return
        
        retry_count = 0
        
        while retry_count <= self.max_retries:
            try:
                yield trial_params
                break  # Success, exit loop
                
            except self.retryable_errors as e:
                retry_count += 1
                
                if retry_count > self.max_retries:
                    logger.error(f"Trial failed after {self.max_retries} retries")
                    raise
                
                # Calculate backoff delay
                delay = self.backoff_factor ** retry_count
                logger.warning(f"Trial failed (attempt {retry_count}/{self.max_retries}): {e}")
                logger.info(f"Retrying in {delay:.1f} seconds...")
                time.sleep(delay)
                
                # Adjust parameters for retry
                if error_handler:
                    trial_params = error_handler(e, trial_params, retry_count)
                else:
                    trial_params = self.adjust_parameters_for_retry(e, trial_params, retry_count)
                
                # Clear memory before retry
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    def adjust_parameters_for_retry(self, error: Exception, 
                                   params: Dict[str, Any], 
                                   retry_count: int) -> Dict[str, Any]:
        """Adjust parameters based on error type and config"""
        adjusted_params = params.copy()
        
        # Handle out of memory errors
        if any(oom_term in str(error).lower() for oom_term in 
               ["out of memory", "cuda out of memory"]):
            
            # Get reduction factor from config or use default
            risk_config = self.config.get('risk_mitigation', {})
            batch_reduction = risk_config.get('batch_reduction_factor', 0.5)
            rank_reduction = risk_config.get('rank_reduction_factor', 0.5)
            
            # Reduce batch size
            if 'batch_size' in adjusted_params:
                current_batch = adjusted_params['batch_size']
                adjusted_params['batch_size'] = max(1, int(current_batch * batch_reduction))
            
            # Reduce model rank if applicable
            if 'rank' in adjusted_params:
                current_rank = adjusted_params['rank']
                adjusted_params['rank'] = max(2, int(current_rank * rank_reduction))
                if 'alpha' in adjusted_params:
                    adjusted_params['alpha'] = adjusted_params['rank'] * 2
            
            logger.info(f"Adjusted parameters for OOM (retry {retry_count}): "
                       f"batch_size={adjusted_params.get('batch_size', 'N/A')}, "
                       f"rank={adjusted_params.get('rank', 'N/A')}")
        
        # Handle convergence issues
        elif any(term in str(error).lower() for term in 
                 ["convergence", "nan", "inf"]):
            
            # Adjust learning rate
            if 'learning_rate' in adjusted_params:
                current_lr = adjusted_params['learning_rate']
                adjusted_params['learning_rate'] = current_lr * 0.5
            
            # Add gradient clipping if not present
            if 'gradient_clip' not in adjusted_params:
                adjusted_params['gradient_clip'] = 1.0
            
            logger.info(f"Adjusted parameters for convergence (retry {retry_count}): "
                       f"lr={adjusted_params.get('learning_rate', 'N/A')}")
        
        return adjusted_params
    
    def get_config(self) -> Dict[str, Any]:
        """Get retry manager configuration"""
        return {
            'max_retries': self.max_retries,
            'backoff_factor': self.backoff_factor,
            'enable_trial_retry': self.enable_trial_retry,
            'config_source': self.config
        }


class ConvergenceMonitor:
    """Monitor and adjust for convergence issues with Pydantic config"""
    
    def __init__(self, config: Optional[Any] = None):
        # Convert config to dictionary if it's a Pydantic model
        self.config = self._convert_to_dict(config)
        
        # Extract convergence config
        training_config = self.config.get('training', {})
        risk_config = self.config.get('risk_mitigation', {})
        
        # Set parameters
        self.patience = training_config.get('patience', 5)
        self.improvement_threshold = risk_config.get('improvement_threshold', 0.001)
        self.min_lr = risk_config.get('min_learning_rate', 1e-7)
        self.enabled = risk_config.get('convergence_monitoring', True)
        
        # Tracking state
        self.best_metric = None
        self.stagnant_epochs = 0
        self.metric_history = []
        
        logger.info(f"ConvergenceMonitor initialized: patience={self.patience}, "
                   f"threshold={self.improvement_threshold}, min_lr={self.min_lr}")
    
    def _convert_to_dict(self, config: Any) -> Dict:
        """Convert config to dictionary, handling Pydantic models"""
        if config is None:
            return {}
        elif hasattr(config, 'model_dump'):
            # Pydantic v2
            return config.model_dump()
        elif hasattr(config, 'dict'):
            # Pydantic v1
            return config.dict()
        elif isinstance(config, dict):
            return config
        else:
            # Try to convert to dict via attributes
            try:
                return {k: v for k, v in config.__dict__.items() if not k.startswith('_')}
            except:
                return {}
    
    def check_convergence(self, current_metric: float, 
                         learning_rate: float) -> Dict[str, Any]:
        """Check convergence and suggest adjustments based on config"""
        if not self.enabled:
            return {'action': 'continue'}
        
        self.metric_history.append({
            'epoch': len(self.metric_history),
            'metric': current_metric,
            'lr': learning_rate
        })
        
        suggestions = {
            'action': 'continue',
            'current_metric': current_metric,
            'best_metric': self.best_metric,
            'stagnant_epochs': self.stagnant_epochs,
            'improvement_threshold': self.improvement_threshold
        }
        
        if self.best_metric is None:
            self.best_metric = current_metric
            return suggestions
        
        improvement = current_metric - self.best_metric
        
        if improvement > self.improvement_threshold:
            # Improvement detected
            self.best_metric = current_metric
            self.stagnant_epochs = 0
            suggestions['action'] = 'continue'
            suggestions['improvement'] = improvement
            
        else:
            # Stagnation detected
            self.stagnant_epochs += 1
            
            if self.stagnant_epochs >= self.patience:
                suggestions['action'] = 'adjust_learning_rate'
                suggestions['new_lr'] = max(self.min_lr, learning_rate * 0.5)
                suggestions['reason'] = f'Stagnant for {self.stagnant_epochs} epochs'
                suggestions['current_lr'] = learning_rate
                
                self.stagnant_epochs = 0  # Reset after adjustment
                
            else:
                suggestions['action'] = 'monitor'
                suggestions['warning'] = f'Stagnant for {self.stagnant_epochs} epochs'
        
        return suggestions
    
    def get_summary(self) -> Dict[str, Any]:
        """Get convergence monitoring summary"""
        return {
            'enabled': self.enabled,
            'best_metric': self.best_metric,
            'stagnant_epochs': self.stagnant_epochs,
            'patience': self.patience,
            'improvement_threshold': self.improvement_threshold,
            'min_lr': self.min_lr,
            'metric_history_length': len(self.metric_history),
            'config': self.config
        }


def create_risk_mitigation_instances(config: Any) -> Dict[str, Any]:
    """
    Factory function to create all risk mitigation instances with Pydantic support
    
    Args:
        config: Can be Pydantic model, dictionary, or DictConfig
        
    Returns:
        Dictionary with risk mitigation instances
    """
    return {
        'resource_monitor': ResourceMonitor(config),
        'recovery_manager': ExperimentRecovery(config),
        'retry_manager': RetryManager(config),
        'convergence_monitor': ConvergenceMonitor(config)
    }


# For backward compatibility
resource_monitor = ResourceMonitor()
recovery_manager = ExperimentRecovery()
retry_manager = RetryManager()
convergence_monitor = ConvergenceMonitor()