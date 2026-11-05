"""
BOPET Trainer for PEFT Models - Optimized for Scientific Research
Enhanced for reproducibility, efficiency
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import time
import gc

# Configuration imports
try:
    from config.experiment import ExperimentConfig
    from config.training import TrainingConfig
except ImportError:
    # Fallback for testing
    from dataclasses import dataclass
    
    @dataclass
    class TrainingConfig:
        gradient_accumulation_steps: int = 1
        max_grad_norm: float = 1.0
        num_epochs: int = 10
        early_stopping: bool = False
        patience: int = 5
        use_amp: bool = True
        learning_rate: float = 2e-5
        weight_decay: float = 0.01
    
    @dataclass
    class ExperimentConfig:
        training: TrainingConfig
        device: str = 'cuda'

logger = logging.getLogger(__name__)
logging.getLogger("progressive_peft").setLevel(logging.INFO)


import torch
import warnings
from peft.tuners.adalora import RankAllocator

# Mute the non-deterministic warning caused by kthvalue parallel sorting on CUDA
warnings.filterwarnings(
    "ignore", 
    message=".*kthvalue CUDA does not have a deterministic implementation.*"
)

# --- FIX 1: Resolve the Parameter Prefix Mismatch ---
def patched_element_score(self, n):
    import torch as _t
    if n in self.exp_avg_ipt and n in self.exp_avg_unc:
        score = self.exp_avg_ipt[n] * self.exp_avg_unc[n]
        return _t.nan_to_num(score, nan=0.0, posinf=1e30, neginf=0.0)
    for key in self.exp_avg_ipt.keys():
        if n.endswith(key) or key.endswith(n):
            score = self.exp_avg_ipt[key] * self.exp_avg_unc[key]
            return _t.nan_to_num(score, nan=0.0, posinf=1e30, neginf=0.0)
    # Fallback: return zeros matching the largest known score shape
    if self.exp_avg_ipt:
        template = next(iter(self.exp_avg_ipt.values()))
        return _t.zeros_like(template)
    return _t.tensor(0.0, dtype=_t.float32)

# --- FIX 2: Correct the Aggressive End-of-Training Pruning Loop ---
def patched_budget_schedule(self, step: int):
    """
    Enforces deltaT intervals during final fine-tuning to prevent
    costly masking calculations on every single training forward pass.
    """
    tinit = self.peft_config.tinit
    tfinal = self.peft_config.tfinal
    total_step = self.peft_config.total_step
    is_interval_step = (step % self.peft_config.deltaT == 0)

    if step <= tinit:
        budget = self.init_bgt
        mask_ind = False
    elif step > total_step - tfinal:
        budget = self.target_bgt
        mask_ind = is_interval_step  # FIX: Only prune on scheduled intervals
    else:
        mul_coeff = 1 - (step - tinit) / (total_step - tfinal - tinit)
        budget = int((self.init_bgt - self.target_bgt) * (mul_coeff**3) + self.target_bgt)
        mask_ind = is_interval_step
        
    return budget, mask_ind

# --- FIX 3: Variadic Keyword Argument Absorption ---
def robust_update_and_allocate(self, model, global_step, *args, **kwargs):
    """
    Safely absorbs extra initialization keywords (like force_mask=True)
    passed down by the framework during state changes.
    """
    return self.orig_update_and_allocate(model, global_step, *args, **kwargs)

# Bind fixes into PEFT's class structure globally
RankAllocator._element_score = patched_element_score
RankAllocator.budget_schedule = patched_budget_schedule

if not hasattr(RankAllocator, "orig_update_and_allocate"):
    RankAllocator.orig_update_and_allocate = RankAllocator.update_and_allocate
    RankAllocator.update_and_allocate = robust_update_and_allocate



class BOPETTrainer:
    """
    Scientific Trainer for PEFT Models with:
    - Mixed precision training
    - Gradient accumulation
    - Early stopping
    - Comprehensive logging
    - MLflow integration
    - Progressive PEPT support
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader],
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        config: ExperimentConfig,
        scheduler: Optional[Any] = None,
        class_weights: Optional[torch.Tensor] = None,
        mlflow_enhancer: Optional[Any] = None,
        **kwargs  # Accept additional arguments for backward compatibility
    ):
        """Initialize trainer with scientific standards"""
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.training_config = config.training
        
        # Handle backward compatibility for removed parameters
        if 'risk_mitigation_instances' in kwargs:
            logger.warning("⚠️ 'risk_mitigation_instances' parameter is deprecated and will be ignored")
        
        # Move model to device
        self.model.to(self.device)

        # At the end of __init__, after self.model.to(self.device)
        if hasattr(self.training_config, 'use_torch_compile') and self.training_config.use_torch_compile:
            try:
                self.model = torch.compile(self.model)
                logger.info("Model compiled with torch.compile")
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}. Proceeding without compilation.")
        
        # Training parameters
        self.grad_accumulation_steps = self.training_config.gradient_accumulation_steps
        self.max_grad_norm = self.training_config.max_grad_norm
        self.num_epochs = self.training_config.num_epochs
        self.early_stopping = self.training_config.early_stopping
        self.patience = self.training_config.patience
        
        # Class weights for imbalanced data
        self.class_weights = class_weights
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(self.device)
        
        # Mixed precision configuration
        self.use_amp = (
            self.training_config.use_amp
            and self.device.type == "cuda"
        )

        # Resolve the configured dtype before creating the scaler.
        configured_dtype = getattr(
            self.training_config, "amp_dtype", "float16"
        ).lower()

        if configured_dtype == "float16":
            self.amp_dtype = torch.float16
        elif configured_dtype == "bfloat16":
            self.amp_dtype = torch.bfloat16
        else:
            raise ValueError(
                f"Unsupported amp_dtype={configured_dtype!r}. "
                "Use 'float16' or 'bfloat16'."
            )

        # GradScaler is required for CUDA FP16, but not for BF16.
        if self.use_amp and self.amp_dtype == torch.float16:
            try:
                self.scaler = torch.amp.GradScaler(
                    "cuda",
                    enabled=True
                )
            except (AttributeError, TypeError):
                # Compatibility with older PyTorch releases.
                self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        else:
            self.scaler = None

        logger.info(
            "[AMP] enabled=%s | dtype=%s | GradScaler=%s",
            self.use_amp,
            str(self.amp_dtype),
            self.scaler is not None,
        )
        
        # MLflow tracking
        self.mlflow_enhancer = mlflow_enhancer
        self.mlflow_log_frequency = 100
        
        # Training state
        self.current_epoch = 0
        self.best_metric = 0
        self.patience_counter = 0

        # Monitoring metric for early stopping
        self.monitor_metric = getattr(self.training_config, 'monitor_metric', 'val_f1')
        self.monitor_mode = getattr(self.training_config, 'monitor_mode', 'max')
        self.best_metric = -float('inf') if self.monitor_mode == 'max' else float('inf')
                
        # Metrics history
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'train_f1': [], 'val_f1': [],
            'lr': [],  # Learning rate history
        }
        # Progressive rank tracking (for scientific visualization)
        self.rank_history = []


        try:
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass
        
        # Progressive PEPT setup
        self._setup_progressive_pept()

        # AdaLoRA-specific global step counter. AdaLoRA's pruning
        # schedule is measured in optimizer steps from the start of
        # training, not per-epoch batch indices. This counter is
        # incremented once per optimizer step, only when the model
        # exposes the PEFT AdaLoRA interface.
        self._adalora_global_step = 0
        self._nonfinite_events = 0
        self._successful_steps  = 0
        self._buffers_verified = False
        
        _base = getattr(self.model, "base_model", None)
        self._is_adalora_model = (
            _base is not None
            and hasattr(_base, "update_and_allocate")
            and not self.is_progressive_model
        )
        logger.info(f"[AdaLoRA] detection: _is_adalora_model={self._is_adalora_model}, "
                    f"base_class={type(_base).__name__ if _base is not None else None}")

       

        
        
        if self._is_adalora_model:
             
            logger.info(
                "[AdaLoRA validation] AMP=%s | dtype=%s | scaler=%s",
                self.use_amp,
                self.amp_dtype,
                self.scaler is not None,
            )
            # Defensive patch for huggingface/peft#1721: guard update_ipt
            # against parameters that have "lora_" in their name but no
            # gradient. Applies to PEFT 0.18.0 and is a no-op on versions
            # where the guard is already present.
            try:
                from peft.tuners.adalora.model import RankAllocator

                if not getattr(RankAllocator.update_ipt, "_is_patched", False):
                    _orig_update_ipt = RankAllocator.update_ipt

                    def _safe_update_ipt(self, model):
                        """
                        FP32 AdaLoRA importance estimation.

                        Assumption: gradients have already been unscaled by GradScaler
                        before update_and_allocate() is called.
                        """
                        import torch as _t

                        with _t.no_grad():
                            for n, p in model.named_parameters():

                                if "lora_" not in n or p.grad is None:
                                    continue

                                # Importance estimation must not inherit FP16 arithmetic.
                                p32 = p.detach().to(dtype=_t.float32)
                                g32 = p.grad.detach().to(dtype=_t.float32)

                                ipt_new = (p32 * g32).abs()

                                g_max = g32.abs().max().item()
                                g_nonzero = (g32 != 0).float().mean().item()
                                #logger.info(f"[AdaLoRA-FP32] {n}: g_max={g_max:.3e} nonzero={g_nonzero:.4f}")

                                if not _t.isfinite(ipt_new).all():
                                    logger.warning(
                                        "[AdaLoRA-FP32] Non-finite importance; "
                                        "skipping parameter: %s",
                                        n,
                                    )
                                     
                                    continue

                                 

                                # Create or migrate all importance buffers to FP32.
                                if n not in self.ipt:
                                    self.ipt[n] = _t.zeros_like(p32)
                                else:
                                    self.ipt[n] = self.ipt[n].to(
                                        device=p.device,
                                        dtype=_t.float32,
                                    )

                                if n not in self.exp_avg_ipt:
                                    self.exp_avg_ipt[n] = _t.zeros_like(p32)
                                else:
                                    self.exp_avg_ipt[n] = self.exp_avg_ipt[n].to(
                                        device=p.device,
                                        dtype=_t.float32,
                                    )

                                if n not in self.exp_avg_unc:
                                    self.exp_avg_unc[n] = _t.zeros_like(p32)
                                else:
                                    self.exp_avg_unc[n] = self.exp_avg_unc[n].to(
                                        device=p.device,
                                        dtype=_t.float32,
                                    )

                                # Preserve the AdaLoRA moving-average update equations.
                                self.ipt[n] = ipt_new

                                self.exp_avg_ipt[n] = (
                                    self.beta1 * self.exp_avg_ipt[n]
                                    + (1.0 - self.beta1) * ipt_new
                                )

                                self.exp_avg_unc[n] = (
                                    self.beta2 * self.exp_avg_unc[n]
                                    + (1.0 - self.beta2)
                                    * (ipt_new - self.exp_avg_ipt[n]).abs()
                                )

                    _safe_update_ipt._is_patched = True
                    RankAllocator.update_ipt = _safe_update_ipt
                    logger.info("Applied RankAllocator.update_ipt patch (peft#1721)")
            except Exception as _e:
                logger.debug(f"update_ipt patch not applied: {_e}")    
                
                        
        logger.info(f"Trainer initialized on {self.device}")
        logger.info(f"Gradient accumulation steps: {self.grad_accumulation_steps}")
        logger.info(f"Mixed precision: {self.use_amp}")
    
    def _setup_progressive_pept(self):
        """Setup progressive PEPT if model supports it"""
        self.is_progressive_model = False
        self.progressive_state = None
        
        # Check if model has progressive PEPT attributes
        model_class_name = self.model.__class__.__name__
        if hasattr(self.model, "step_post_backward"):
            self.is_progressive_model = True
            logger.info("Progressive PEPT model detected")
            
            self.progressive_state = {
                'allocation_updates': 0,
                'gradient_updates': 0,
                'last_allocation_step': 0,
            }
    
    def train(self, save_path: Optional[str] = None) -> Dict:
        """Main training loop with early stopping"""
        logger.info(f"Starting training for {self.num_epochs} epochs")
        logger.info(f"Monitoring metric: {self.monitor_metric} (mode: {self.monitor_mode})")
        
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch+1}/{self.num_epochs}")
            logger.info(f"{'='*60}")
            
            # Train for one epoch
            train_loss, train_acc, train_f1 = self.train_epoch(epoch)
            
            # Validate if validation loader exists
            if self.val_loader:
                val_result = self.evaluate(self.val_loader)
                val_loss, val_acc, val_f1 = val_result['loss'], val_result['accuracy'], val_result['f1_weighted']

                # Get the monitored metric value
                current_metric = val_result.get(self.monitor_metric, val_f1)  # fallback to val_f1 if key missing
                
                # Update history
                self.history['train_loss'].append(train_loss)
                self.history['train_acc'].append(train_acc)
                self.history['train_f1'].append(train_f1)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                self.history['val_f1'].append(val_f1)
                self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
                
                # Check improvement based on mode
                improved = False
                if self.monitor_mode == 'max':
                    if current_metric > self.best_metric:
                        improved = True
                else:  # 'min'
                    if current_metric < self.best_metric:
                        improved = True
                
                if improved:
                    self.best_metric = current_metric
                    self.patience_counter = 0
                    
                    # Save best model
                    if save_path:
                        self._save_best_model(epoch, val_f1, val_acc, train_loss, train_acc, save_path)
                        logger.info(f"Saved best model ({self.monitor_metric}: {current_metric:.4f})")
                else:
                    self.patience_counter += 1
                
                # Log metrics
                logger.info(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f}")
                logger.info(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f} | Val F1:   {val_f1:.4f}")
                
                # ============================================================
                # 🔬 Capture Progressive Rank Snapshot (per epoch)
                # ============================================================
                if self.is_progressive_model and hasattr(self.model, "current_allocation"):
                    if self.model.current_allocation:
                        self.rank_history.append({
                            "epoch": epoch,
                            "allocation": dict(self.model.current_allocation)
                        })

                # ============================================================
                # 🔥 TRUE Progressive PEPT Scientific Logging (Q1-grade)
                # ============================================================
                if self.is_progressive_model and self.mlflow_enhancer:

                    try:
                        if hasattr(self.model, "current_allocation") and self.model.current_allocation:

                            import numpy as np

                            ranks = np.array(
                                list(self.model.current_allocation.values()),
                                dtype=np.float32
                            )

                            # ---------- Rank Entropy ----------
                            p = ranks / (ranks.sum() + 1e-8)
                            entropy = float(-(p * np.log(p + 1e-8)).sum())

                            # ---------- Rank Variance ----------
                            rank_std = float(np.std(ranks))

                            # ---------- Gini Coefficient (reviewers LOVE this) ----------
                            sorted_ranks = np.sort(ranks)
                            n = len(sorted_ranks)
                            gini = float(
                                (np.sum((2*np.arange(1,n+1)-n-1)*sorted_ranks)) /
                                (n*np.sum(sorted_ranks)+1e-8)
                            )

                            self.mlflow_enhancer.log_custom_metric(
                                "progressive_rank_entropy",
                                entropy,
                                step=epoch
                            )

                            self.mlflow_enhancer.log_custom_metric(
                                "progressive_rank_std",
                                rank_std,
                                step=epoch
                            )

                            self.mlflow_enhancer.log_custom_metric(
                                "progressive_rank_gini",
                                gini,
                                step=epoch
                            )

                    except Exception as e:
                        logger.debug(f"Entropy logging skipped: {e}")

                #  NEW — Progressive PEPT epoch logging
                if self.is_progressive_model:
                    self._log_progressive_pept_epoch_summary(epoch)

                
                # Early stopping
                if self.early_stopping and self.patience_counter >= self.patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
            else:
                # No validation set
                logger.info(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f}")
        
        # Load best model if available
        if save_path and Path(save_path).exists():
            success = self._load_best_model(save_path)
            if success:
                logger.info("Loaded best model for final evaluation")
            else:
                logger.info("Using final model for evaluation (best checkpoint shape mismatch)")
                
        if self._is_adalora_model:
            total = self._successful_steps + self._nonfinite_events
            rate  = 100.0 * self._nonfinite_events / max(total, 1)
            logger.info(
                "[AMP] Non-finite gradient events: %d / %d (%.2f%%) | "
                "successful optimizer steps: %d | AdaLoRA step: %d",
                self._nonfinite_events, total, rate,
                self._successful_steps, self._adalora_global_step,
            )

        if self.is_progressive_model:
            logger.info(
                f"[TIMING_SUMMARY] method={self.config.peft.method} "
                f"total_steps={self.current_epoch * len(self.train_loader)} "
            )
        
        return self.history
    
    def _log_progressive_pept_epoch_summary(self, epoch: int):
        """
        Logs Progressive PEPT rank dynamics to MLflow.
        Safe for non-progressive models.
        """
        if not self.mlflow_enhancer:
            return

        try:
            if hasattr(self.model, "allocator"):
                current_rank = getattr(self.model.allocator, "current_rank", None)

                if current_rank is not None:
                    self.mlflow_enhancer.log_custom_metric(
                        "progressive_avg_rank",
                        float(current_rank),
                        step=epoch
                    )

            # Optional richer logging (research-grade)
            if hasattr(self.model, "lora_modules"):
                self.mlflow_enhancer.log_custom_metric(
                    "progressive_num_modules",
                    float(len(self.model.lora_modules)),
                    step=epoch
                )

        except Exception as e:
            logger.debug(f"PEPT MLflow logging skipped: {e}")

    
    def train_epoch(self, epoch: int) -> Tuple[float, float, float]:
        """Train for one epoch with enhanced MLflow integration"""
 
        self.model.train()
        
        total_loss_tensor = torch.zeros(1, dtype=torch.float64, device=self.device)

        pred_tensor_list = []
        label_tensor_list = []
        
        # Reset gradient accumulation
        self.optimizer.zero_grad()
        
        # Track effective optimizer steps
        effective_steps = 0
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")

        prev_time = time.time()
        data_times = []  # to collect data loading times 
        
        for step, batch in enumerate(pbar):

            current_time = time.time()
            if step > 0:  # skip the first iteration as there’s no previous
                data_time = current_time - prev_time
                data_times.append(data_time)

            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass with mixed precision
            if self.use_amp:
                loss, preds = self._mixed_precision_step(batch, step)
            else:
                loss, preds = self._standard_step(batch, step)
            
            # # Update progressive PEPT
            # if self.is_progressive_model and hasattr(self.model, 'update_gradient_buffer'):
            #     self._update_progressive_pept(step)
            
            # Track metrics
            batch_size = batch['labels'].size(0)

            total_loss_tensor += loss.detach() * batch_size * self.grad_accumulation_steps

            pred_tensor_list.append(preds.detach())
            label_tensor_list.append(batch['labels'].detach())

            
            # Check if optimizer step was taken
            if (step + 1) % self.grad_accumulation_steps == 0:
                # if hasattr(self.model, "current_allocation"):
                #     logger.info(f"Current Allocation: {self.model.current_allocation}")

                effective_steps += 1
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update progress bar
            loss_scalar = float(loss.detach().cpu())
            pbar.set_postfix({
                'loss': f"{loss_scalar * self.grad_accumulation_steps:.3f}",
                'lr': f"{current_lr:.2e}",
                'eff_steps': f"{effective_steps}"
            })
            
            # Enhanced MLflow logging
            global_step = epoch * len(self.train_loader) + step
            
            if self.mlflow_enhancer:
                # Log batch metrics more frequently at start, then reduce frequency
                log_frequency = self._calculate_adaptive_log_frequency(step)
                
                if step % log_frequency == 0:
                    # Calculate gradient norm if we just took an optimizer step
                    grad_norm = 0
                    if (step + 1) % self.grad_accumulation_steps == 0:
                        grad_norm = self._compute_gradient_norm()
                    
                    # Log batch metrics
                    batch_metrics = {
                        'loss': loss.item() * self.grad_accumulation_steps,
                        'learning_rate': current_lr,
                        'gradient_norm': grad_norm,
                    }
                    
                    # Add progressive PEPT metrics if available
                    if self.is_progressive_model and self.progressive_state:
                        batch_metrics.update({
                            'progressive_allocation_updates': self.progressive_state['allocation_updates'],
                            'progressive_gradient_updates': self.progressive_state['gradient_updates'],
                        })
                    
                    self.mlflow_enhancer.log_batch_metrics(
                        batch_idx=global_step,
                        batch_metrics=batch_metrics,
                        phase='train'
                    )
                
                # Log resource usage every 50 steps (adjustable)
                if step % 50 == 0:
                    self.mlflow_enhancer.log_resource_usage(phase="training")
                
                # Flush metrics periodically
                if step % 200 == 0:
                    self.mlflow_enhancer.flush_metrics()
            
            # Clear memory periodically
            if step % 200 == 0:
                self._clear_memory()
            
            # after all processing for this batch (e.g., after updating progress bar)
            prev_time = time.time()
        
        if data_times:
            avg_data_time = sum(data_times) / len(data_times)
            logger.info(f"Epoch {epoch+1}: Average data loading time per batch = {avg_data_time*1000:.2f} ms")
        
        
        # Log epoch summary to MLflow
        if self.mlflow_enhancer:
            epoch_metrics = {
                'epoch': epoch,
                'effective_steps': effective_steps,
                'final_lr': self.optimizer.param_groups[0]['lr'],
            }
            self.mlflow_enhancer.log_custom_metric('training_epoch', epoch, step=epoch)
            self.mlflow_enhancer.log_custom_metric('training_effective_steps', effective_steps, step=epoch)
        
        # Compute epoch metrics
        y_pred = torch.cat(pred_tensor_list).cpu().numpy()
        y_true = torch.cat(label_tensor_list).cpu().numpy()

        num_samples = len(y_true)

        epoch_loss = (total_loss_tensor / num_samples).item()

        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')

        
        return epoch_loss, accuracy, f1

    def _calculate_adaptive_log_frequency(self, step: int) -> int:
        """Calculate adaptive log frequency to avoid overwhelming MLflow"""
        # More frequent logging at the beginning, then reduce
        if step < 100:
            return 10  # Every 10 steps for first 100 steps
        elif step < 500:
            return 25  # Every 25 steps for next 400 steps
        elif step < 1000:
            return 50  # Every 50 steps
        else:
            return 100  # Every 100 steps thereafter

    def _compute_gradient_norm(self) -> float:
        """Compute gradient norm for logging"""
        try:
            total_norm = 0
            for p in self.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
            return total_norm ** 0.5
        except:
            return 0.0

    def _update_progressive_pept(self, step: int):
        """
        Correct Progressive PEPT v2.x integration
        """
        try:
            if hasattr(self.model, "step_post_backward"):
                self.model.step_post_backward()

                if self.progressive_state:
                    if hasattr(self.model, "allocator"):
                        self.progressive_state["gradient_updates"] = self.model.allocator.step

                # Optimizer reset handshake (CRITICAL)
                if getattr(self.model, "optimizer_reset_required", False):
                    self._reset_optimizer_for_progressive_pept()
                    self.model.clear_optimizer_flag()

                    if self.progressive_state:
                        self.progressive_state["allocation_updates"] += 1

        except Exception as e:
            logger.warning(f"Progressive PEPT update failed: {e}")

    def _reset_optimizer_for_progressive_pept(self):
        """
        Safe optimizer reset after rank reallocation.
        Preserves scheduler continuity by redirecting it to the new optimizer.
        """
        import time
        t_start = time.perf_counter()

        logger.info("🔄 Resetting optimizer due to Progressive PEPT rank change")

        opt_class = self.optimizer.__class__
        new_optimizer = opt_class(
            self.model.parameters(),
            lr=self.optimizer.param_groups[0]["lr"],
            weight_decay=self.training_config.weight_decay
        )
        self.optimizer = new_optimizer

        if self.scheduler:
            # Redirect scheduler to the new optimizer – do NOT reset last_epoch
            self.scheduler.optimizer = self.optimizer
        t_end = time.perf_counter()
        logger.info(f"[TIMING] optimizer_reset took {(t_end-t_start)*1000:.1f}ms")
    
    
    def _check_adalora_fp32_buffers(self):
        """Verify AdaLoRA importance buffers remain FP32."""

        target = getattr(self.model, "base_model", self.model)
        allocator = getattr(target, "rankallocator", None)

        if allocator is None:
            raise RuntimeError(
                "AdaLoRA rankallocator was not found. "
                "Check the installed PEFT model structure."
            )

        for name, tensor in allocator.exp_avg_ipt.items():
            assert tensor.dtype == torch.float32, (
                f"{name}: exp_avg_ipt is {tensor.dtype}, expected torch.float32"
            )

        for name, tensor in allocator.exp_avg_unc.items():
            assert tensor.dtype == torch.float32, (
                f"{name}: exp_avg_unc is {tensor.dtype}, expected torch.float32"
            )

        logger.info(
            "[AdaLoRA validation] FP32 importance buffers verified."
        )



    def _mixed_precision_step(self, batch: Dict,
        step: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """FP16 AMP training step with safe AdaLoRA allocation."""

        with torch.autocast(
            device_type="cuda",
            dtype=self.amp_dtype,
            enabled=self.use_amp,
        ):
            outputs = self.model(**batch)
            loss = self._compute_loss(outputs, batch)
            loss = loss / self.grad_accumulation_steps

        # Backward pass.
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Perform an optimizer update only at accumulation boundaries.
        if (step + 1) % self.grad_accumulation_steps == 0:

            if self.is_progressive_model:
                self._update_progressive_pept(step)

            # Unscale before clipping and before AdaLoRA reads gradients.
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)

            # Check gradients belonging to optimizer parameters.
            grads_are_finite = True

            for group in self.optimizer.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue

                    grad = p.grad

                    if grad.is_sparse:
                        grad_values = grad.coalesce().values()
                    else:
                        grad_values = grad

                    if not torch.isfinite(grad_values).all():
                        grads_are_finite = False
                        break

                if not grads_are_finite:
                    break

            if grads_are_finite:

                # Clip unscaled gradients.
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm,
                )

                # Apply the optimizer update.
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                # Update AdaLoRA only after a successful optimizer step.
                if self._is_adalora_model:
                    self._adalora_global_step += 1
                    self._successful_steps     += 1 

                    target = getattr(
                        self.model,
                        "base_model",
                        self.model,
                    )

                    try:
                        target.update_and_allocate(self._adalora_global_step)
                    except Exception as e:
                        logger.error("AdaLoRA update_and_allocate failed at step %d: %s",
                                    self._adalora_global_step, e, exc_info=True)
                        raise

                    
                    # Verify ONCE, not every step.
                    if not self._buffers_verified:
                        self._check_adalora_fp32_buffers()
                        self._buffers_verified = True

                # Scheduler advances only after a successful update.
                if self.scheduler is not None:
                    self.scheduler.step()

            else:
                self._nonfinite_events += 1                # ← HERE
                logger.warning(
                    "[AMP] Non-finite gradients detected at epoch batch step %d. "
                    "Optimizer and AdaLoRA allocation skipped.", step,
                )
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

            # Clear gradients after the optimizer and allocator operations.
            self.optimizer.zero_grad(set_to_none=True)

        return loss, torch.argmax(outputs.logits, dim=-1)
        
    def _standard_step(self, batch: Dict, step: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Training step without mixed precision"""
        outputs = self.model(**batch)
        loss = self._compute_loss(outputs, batch)
        loss = loss / self.grad_accumulation_steps

        # Backward pass
        loss.backward()

        # Gradient accumulation
        if (step + 1) % self.grad_accumulation_steps == 0:
            if self.is_progressive_model:
                self._update_progressive_pept(step)

            # NEW: finiteness check
            grads_are_finite = all(
                torch.isfinite(p.grad).all().item()
                for group in self.optimizer.param_groups
                for p in group["params"]
                if p.grad is not None
            )
            if not grads_are_finite:
                logger.warning(f"[FP32] Non-finite gradients at step {step}; skipping step")
                self.optimizer.zero_grad(set_to_none=True)
                return loss, torch.argmax(outputs.logits, dim=-1)
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            # AdaLoRA pruning schedule (no-op for non-AdaLoRA models)
            if self._is_adalora_model:
                self._adalora_global_step += 1
                target = getattr(self.model, "base_model", self.model)
                try:
                    target.update_and_allocate(self._adalora_global_step)
                except Exception as e:
                    logger.error(
                        f"AdaLoRA update_and_allocate failed at step "
                        f"{self._adalora_global_step}: {e}",
                        exc_info=True,
                    )
                    raise

            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.scheduler:
                self.scheduler.step()
                if step % 100 == 0:
                    logger.debug(
                        f"Step {step}: Scheduler stepped, "
                        f"LR = {self.optimizer.param_groups[0]['lr']:.2e}"
                    )

        return loss, torch.argmax(outputs.logits, dim=-1)
    
    def _compute_loss(self, outputs, batch) -> torch.Tensor:
        """Compute loss - FIXED for BERT models"""
        # Method 1: Use BERT's built-in loss if available
        if hasattr(outputs, 'loss') and outputs.loss is not None:
            loss = outputs.loss
        
        # Method 2: Custom loss with proper dimension handling
        else:
            logits = outputs.logits
            labels = batch['labels']
            
            # Ensure proper dimensions for CrossEntropyLoss
            if len(logits.shape) == 3:  # Sequence classification
                # For token classification: (batch_size, sequence_length, num_labels)
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            else:
                # For sentence classification: (batch_size, num_labels)
                if self.class_weights is not None:
                    loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
                else:
                    loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits, labels)
        
        return loss
     
    
    def _log_batch_metrics(self, step: int, loss: float, lr: float):
        """Log batch metrics to MLflow"""
        if not self.mlflow_enhancer:
            return
        
        batch_metrics = {
            'batch_loss': loss * self.grad_accumulation_steps,
            'batch_lr': lr
        }
        
        global_step = self.current_epoch * len(self.train_loader) + step
        self.mlflow_enhancer.log_batch_metrics(global_step, batch_metrics)
    
    def _clear_memory(self):
        """Clear unused memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
    
    def evaluate(
        self,
        loader: torch.utils.data.DataLoader,
        return_predictions: bool = False,
        baseline_predictions=None,
        collect_inference_stats: bool = False   # NEW (safe default)
        ) -> Dict:
        """Evaluate model with MLflow logging"""
        self.model.eval()
        try:
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass 
        
        # Track evaluation time
        eval_start = time.time()
        
        pred_tensor_list = []
        prob_tensor_list = []

        label_tensor_list = []
        inference_stats = [] if collect_inference_stats else None

        total_samples = 0
        total_loss_tensor = torch.zeros(1, dtype=torch.float64, device=self.device)

        with torch.inference_mode():  # 🔥 faster than no_grad
            for batch in tqdm(loader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                batch_size = batch['labels'].size(0)
                total_samples += batch_size

                start_t = time.perf_counter() if collect_inference_stats else None

                outputs = self.model(**batch)

                end_t = time.perf_counter() if collect_inference_stats else None

                loss = self._compute_loss(outputs, batch)
                total_loss_tensor += loss.detach() * batch_size

                preds = torch.argmax(outputs.logits, dim=-1)

                probs = torch.softmax(outputs.logits, dim=-1)
                prob_tensor_list.append(probs.detach())

                pred_tensor_list.append(preds.detach())
                label_tensor_list.append(batch['labels'].detach())

                if collect_inference_stats:
                    inference_stats.append({
                        "latency": end_t - start_t,
                        "batch_size": batch_size
                    })

        
        # Compute evaluation time
        eval_time = time.time() - eval_start
        
        # Compute metrics
        y_pred = torch.cat(pred_tensor_list).cpu().numpy()
        y_true = torch.cat(label_tensor_list).cpu().numpy()
        y_prob = torch.cat(prob_tensor_list).cpu().numpy()

        avg_loss = (total_loss_tensor / total_samples).item()

        accuracy = accuracy_score(y_true, y_pred)
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
                
        num_samples = len(y_true)

        result = {
            "loss": float(avg_loss),
            "accuracy": float(accuracy),
            "f1_weighted": float(f1_weighted),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "evaluation_time_seconds": float(eval_time),
            "samples_per_second": float(num_samples / eval_time) if eval_time > 0 else 0.0,
            
        }
        
        # Log evaluation metrics to MLflow
        if self.mlflow_enhancer:
            eval_metrics = {
                'eval_loss': avg_loss,
                'eval_accuracy': accuracy,
                'eval_f1_weighted': f1_weighted,
                'eval_time_seconds': eval_time,
                'eval_samples_per_second': float(num_samples / eval_time) if eval_time > 0 else 0.0,
            }
            
            # Log evaluation phase
            self.mlflow_enhancer.log_resource_usage(phase="evaluation")
            
            # Log evaluation metrics
            for metric_name, metric_value in eval_metrics.items():
                if isinstance(metric_value, (int, float)):
                    self.mlflow_enhancer.log_custom_metric(metric_name, metric_value)
        
        if return_predictions:
            result.update({
                "predictions": y_pred.tolist(),
                "labels": y_true.tolist(),
                "probabilities": y_prob.tolist(),   # 🔥 REQUIRED FOR CALIBRATION
            })


        # --- NEW: expose atomic inference data to evaluator ---
        result["__atomic_inference__"] = {
            "inference_stats": inference_stats,
            "num_samples": int(num_samples),
            "evaluation_time_seconds": float(eval_time),
        }

        
        # Statistical significance testing
        if baseline_predictions is not None:
            from scipy import stats

            baseline_predictions = np.asarray(baseline_predictions)

            model_correct = (y_pred == y_true)
            baseline_correct = (baseline_predictions == y_true)

            only_model = np.sum(model_correct & ~baseline_correct)
            only_baseline = np.sum(~model_correct & baseline_correct)

            if only_model + only_baseline > 0:
                chi2 = (abs(only_model - only_baseline) - 1) ** 2 / (only_model + only_baseline)
                p_value = 1 - stats.chi2.cdf(chi2, df=1)

                result['mcnemar_test'] = {
                    'chi2': float(chi2),
                    'p_value': float(p_value),
                    'significant': bool(p_value < 0.05)
                }


        
        # Add confidence intervals
        if num_samples >= 30:
            from statsmodels.stats.proportion import proportion_confint

            n_correct = int(np.sum(y_pred == y_true))
            ci_lower, ci_upper = proportion_confint(
                n_correct,
                num_samples,
                alpha=0.05,
                method='beta'
            )

            result['accuracy_ci_95'] = [float(ci_lower), float(ci_upper)]


        
        return result
    
    def _save_best_model(self, epoch: int, val_f1: float, val_acc: float, 
                        train_loss: float, train_acc: float, save_path: str):
        """Save best model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'val_f1': val_f1,
            'val_acc': val_acc,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'history': self.history,
        }
        
        # Ensure directory exists
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save with retry logic
        for attempt in range(3):
            try:
                torch.save(checkpoint, save_path)
                logger.debug(f"Model saved to {save_path}")
                return True
            except Exception as e:
                logger.warning(f"Save attempt {attempt + 1} failed: {e}")
                time.sleep(1)
        
        logger.error(f"Failed to save model after 3 attempts")
        return False
    
    def _load_best_model(self, path: str) -> bool:
        """Load best model from checkpoint. Returns True if successful, False otherwise."""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            is_qlora = self.config.peft.method == "qlora" or getattr(self.config.peft, 'quantization_enabled', False)
            
            if is_qlora:
                logger.info("Loading QLoRA model with quantization-aware loading")
                filtered_state_dict = {}
                for k, v in checkpoint.get('model_state_dict', {}).items():
                    if 'absmax' not in k and 'quant_map' not in k and 'quant_state' not in k:
                        filtered_state_dict[k] = v
                state_dict_to_load = filtered_state_dict
            else:
                state_dict_to_load = checkpoint['model_state_dict']
            
            missing, unexpected = self.model.load_state_dict(state_dict_to_load, strict=False)
            if missing:
                logger.warning(f"Missing keys when loading checkpoint: {missing[:10]}...")
            if unexpected:
                logger.warning(f"Unexpected keys when loading checkpoint: {unexpected[:10]}...")
            
            logger.info(f"Loaded model from checkpoint: {path} (non-strict)")
            return True
            
        except RuntimeError as e:
            if "size mismatch" in str(e):
                # Expected for progressive models – log as debug
                logger.debug(f"Shape mismatch when loading checkpoint (expected for progressive models): {e}")
            else:
                logger.error(f"Failed to load model from {path}: {e}")
            logger.info("Continuing with current model for evaluation.")
            return False
            
        except Exception as e:
            logger.error(f"Failed to load model from {path}: {e}")
            logger.info("Continuing with current model for evaluation.")
            return False
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary"""
        summary = {
            'history': self.history,
            'final_metrics': {
                'train_loss': self.history['train_loss'][-1] if self.history['train_loss'] else None,
                'train_acc': self.history['train_acc'][-1] if self.history['train_acc'] else None,
                'val_loss': self.history['val_loss'][-1] if self.history['val_loss'] else None,
                'val_acc': self.history['val_acc'][-1] if self.history['val_acc'] else None,
            },
            'best_metric': self.best_metric,
            'epochs_completed': self.current_epoch + 1,
            'early_stopping_triggered': self.patience_counter >= self.patience if self.early_stopping else False,
            "rank_history": self.rank_history,
        }
        
        # Add progressive PEPT stats if available
        if self.is_progressive_model and self.progressive_state:
            summary['progressive_pept'] = self.progressive_state
        
        return summary