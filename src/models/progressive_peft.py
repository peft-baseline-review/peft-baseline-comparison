import logging
import math
from typing import Any
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ============================================================
# Gradient Importance Estimator (Research-grade)
# ============================================================

@dataclass
class ProgressivePEFTConfig:
    # ===== Rank Control =====
    initial_rank: int = 4
    final_rank: int = 16
    min_rank: int = 2
    max_rank: int = 32

    temperature: float = 1.0                # softmax temperature
    importance_smoothing: float = 0.99      # EMA factor for module importance

    # ===== Scheduling =====
    rank_schedule: str = "adaptive"
    schedule_steps: int = 500

    # ===== Allocation Strategy =====
    allocation_strategy: str = "budget"     # "budget" or "structure"
    allocation_update_frequency: int = 200

    # ===== Importance Estimation =====
    ema_beta: float = 0.9
    warmup_steps: int = 50

    # ===== Stability Controls =====
    rank_change_threshold: int = 1


class GradientImportanceEstimator:
    def __init__(self, beta: float = 0.9, warmup_steps: int = 50, ema_beta: float = 0.99):
        self.beta = beta
        self.warmup_steps = warmup_steps
        self.step_count = 0
        self.ema_buffer = {}               # per-parameter EMA
        self.module_ema = {}                # per-module EMA (smoothed importance)
        self.ema_beta = ema_beta            # smoothing factor for module importance

    def update(self, model: nn.Module):
        self.step_count += 1
        module_scores = defaultdict(list)

        for name, param in model.named_parameters():
            if "lora_" not in name or param.grad is None:
                continue

            grad_norm = param.grad.data.norm(p=2).item()
            param_norm = param.data.norm(p=2).item()
            importance = grad_norm / (param_norm + 1e-8)

            if math.isnan(importance) or math.isinf(importance):
                continue

            if name not in self.ema_buffer:
                self.ema_buffer[name] = importance
            else:
                self.ema_buffer[name] = (
                    self.beta * self.ema_buffer[name]
                    + (1 - self.beta) * importance
                )

            if ".lora_A" in name:
                module_key = name.split(".lora_A")[0]
            elif ".lora_B" in name:
                module_key = name.split(".lora_B")[0]
            else:
                continue
            module_scores[module_key].append(importance)

        for module, scores in module_scores.items():
            if not scores:
                continue
            median_imp = float(np.median(scores))
            if module not in self.module_ema:
                self.module_ema[module] = median_imp
            else:
                self.module_ema[module] = (
                    self.ema_beta * self.module_ema[module]
                    + (1 - self.ema_beta) * median_imp
                )

    def get_importance(self):
        if self.step_count < self.warmup_steps:
            return {}, {}
        return self.ema_buffer.copy(), self.module_ema.copy()


# ============================================================
# Progressive Rank Allocator
# ============================================================

class ProgressiveRankAllocator:
    """
    Budget-constrained adaptive rank allocator.
    """

    def __init__(self, config):
        self.config = config
        self.current_rank = config.initial_rank
        self.step = 0
        self.last_reset_step = 0
        self.reset_cooldown = 50

        logger.info(f"ProgressiveRankAllocator: init_rank={config.initial_rank}, final_rank={config.final_rank}, schedule={config.rank_schedule}, steps={config.schedule_steps}")

    def update_schedule(self, importance_estimator):
        if self.config.rank_schedule == "linear":
            progress = min(1.0, self.step / self.config.schedule_steps)
            self.current_rank = int(
                self.config.initial_rank +
                progress * (self.config.final_rank - self.config.initial_rank)
            )

        elif self.config.rank_schedule == "exponential":
            growth = 1 - math.exp(-self.step / self.config.schedule_steps)
            self.current_rank = int(
                self.config.initial_rank +
                growth * (self.config.final_rank - self.config.initial_rank)
            )

        elif self.config.rank_schedule == "adaptive":
            importance_vals = list(importance_estimator.ema_buffer.values())
            if importance_vals:
                avg_imp = float(np.mean(importance_vals))
                growth = 1 - math.exp(-avg_imp)
            else:
                growth = 0.0

            self.current_rank = int(
                self.config.initial_rank +
                growth * (self.config.final_rank - self.config.initial_rank)
            )

            self.current_rank = max(
                self.config.min_rank,
                min(self.config.max_rank, self.current_rank)
            )

        elif self.config.rank_schedule == "fixed":
            pass

    def allocate(self, importance_dict, lora_modules):
        if self.config.allocation_strategy == "budget":
            return self._allocate_budget_based(importance_dict, lora_modules)
        elif self.config.allocation_strategy == "structure":
            return self._allocate_structure_adaptive(importance_dict, lora_modules)
        else:
            raise ValueError(f"Unknown allocation strategy: {self.config.allocation_strategy}")

    def _allocate_budget_based(self, importance_dict, lora_modules):
        if not importance_dict:
            return {name: self.current_rank for name in lora_modules}

        scores = np.array([importance_dict.get(name, 0.0) for name in lora_modules], dtype=np.float64)

        if np.any(np.isnan(scores)) or np.any(np.isinf(scores)):
            logger.warning("Scores contain NaN/Inf. Replacing with 1.0.")
            scores = np.where(np.isfinite(scores), scores, 1.0)

        scores = scores + 1e-8

        total = scores.sum()
        if total < 1e-12:
            logger.debug("Sum of scores near zero. Using uniform allocation.")
            scores = np.ones_like(scores) / len(scores)
        else:
            scores = scores / total

        total_budget = self.current_rank * len(lora_modules)

        allocation = {
            name: int(max(self.config.min_rank,
                        min(self.config.max_rank,
                            total_budget * score)))
            for name, score in zip(lora_modules, scores)
        }

        current_sum = sum(allocation.values())
        if current_sum > 0:
            scale = total_budget / current_sum
            for k in allocation:
                allocation[k] = int(
                    max(self.config.min_rank,
                        min(self.config.max_rank,
                            allocation[k] * scale))
                )

        return allocation

    def _allocate_structure_adaptive(self, importance_dict, lora_modules):
        if not importance_dict:
            return {name: self.current_rank for name in lora_modules}

        scores = np.array([importance_dict.get(name, 0.0) for name in lora_modules], dtype=np.float64)
        scores = np.maximum(scores, 1e-12)
        scores = scores / self.config.temperature
        exp_scores = np.exp(scores - scores.max())
        probs = exp_scores / exp_scores.sum()

        total_budget = self.current_rank * len(lora_modules)
        allocation = {}
        for i, name in enumerate(lora_modules):
            raw_rank = probs[i] * total_budget
            rank = int(np.clip(raw_rank, self.config.min_rank, self.config.max_rank))
            allocation[name] = rank
        return allocation


# ============================================================
# Progressive PEFT Wrapper (Research Version)
# ============================================================

class ProgressivePEFTWrapper(nn.Module):

    def __init__(self, model, config):
        super().__init__()

        self.model = model
        self.config = config
        self.last_reset_step = 0
        self.reset_cooldown = 50

        self.importance_estimator = GradientImportanceEstimator(
            beta=config.ema_beta,
            warmup_steps=config.warmup_steps
        )

        self.allocator = ProgressiveRankAllocator(config)

        self.lora_modules = self._find_lora_modules()

        self.current_allocation = {}

        self.optimizer_reset_required = False

        logger.info(f"ProgressivePEFTWrapper initialized with threshold={config.rank_change_threshold}")

    def _find_lora_modules(self):
        modules = []
        for name, module in self.model.named_modules():
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                modules.append(name)
        return modules

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def step_post_backward(self):
        self.allocator.step += 1
        self.importance_estimator.update(self.model)
        per_param_imp, module_imp = self.importance_estimator.get_importance()

        if self.allocator.step % self.config.allocation_update_frequency != 0:
            return

        self.allocator.update_schedule(self.importance_estimator)
        logger.info(f"[PROGRESSIVE] Step {self.allocator.step}: current_rank = {self.allocator.current_rank}")

        allocation = self.allocator.allocate(module_imp, self.lora_modules)

        if allocation:
            max_rank = max(allocation.values())
            logger.info(f"[PROGRESSIVE] Max allocated rank at step {self.allocator.step}: {max_rank}")

        sample_modules = list(allocation.keys())[:3]
        for mod in sample_modules:
            logger.info(f"[PROGRESSIVE]   allocated {mod} = {allocation[mod]}")

        self.current_allocation = allocation.copy()
        changed = self._apply_allocation(allocation)

        if changed:
            logger.info(f"[PROGRESSIVE] Rank changes applied at step {self.allocator.step}")
        else:
            logger.info(f"[PROGRESSIVE] No rank changes at step {self.allocator.step}")

    def _apply_allocation(self, allocation):
        changed_any = False
        for name, module in self.model.named_modules():
            if name not in allocation:
                continue
            target_rank = allocation[name]
            A, B = self._get_lora_weights(module)
            if A is None:
                continue
            current_rank = A.shape[0]
            diff = abs(target_rank - current_rank)
            if diff < self.config.rank_change_threshold:
                continue
            self._resize_lora_weights(module, target_rank)
            changed_any = True
        return changed_any

    def _resize_lora_weights(self, module, new_rank):
        """
        Resize LoRA weights to new_rank while preserving existing weights.
        Handles both ModuleDict (multiple adapters) and direct module cases.
        Updates the scaling factor correctly for each adapter.
        """
        try:
            if isinstance(module.lora_A, nn.ModuleDict):
                # Multiple adapters: resize each adapter individually
                keys = list(module.lora_A.keys())
                if not keys:
                    logger.warning("No adapters found in ModuleDict, skipping resize.")
                    return
                for key in keys:
                    old_A = module.lora_A[key].weight.data
                    old_B = module.lora_B[key].weight.data
                    target_A = module.lora_A[key]
                    target_B = module.lora_B[key]

                    device = old_A.device
                    dtype = old_A.dtype
                    in_features = old_A.shape[1]
                    out_features = old_B.shape[0]

                    new_A = torch.zeros((new_rank, in_features), device=device, dtype=dtype)
                    new_B = torch.zeros((out_features, new_rank), device=device, dtype=dtype)

                    copy_rank = min(new_rank, old_A.shape[0])
                    new_A[:copy_rank] = old_A[:copy_rank]
                    new_B[:, :copy_rank] = old_B[:, :copy_rank]

                    if new_A.shape == target_A.weight.shape:
                        target_A.weight.data.copy_(new_A)
                    else:
                        target_A.weight = nn.Parameter(new_A)
                        self.optimizer_reset_required = True

                    if new_B.shape == target_B.weight.shape:
                        target_B.weight.data.copy_(new_B)
                    else:
                        target_B.weight = nn.Parameter(new_B)
                        self.optimizer_reset_required = True

                    if hasattr(module, "scaling"):
                        alpha = self._get_alpha(module, key)
                        new_scaling = alpha / new_rank
                        if isinstance(module.scaling, dict):
                            module.scaling[key] = new_scaling
                        else:
                            # Fallback: convert to dict (should not happen with multiple adapters)
                            module.scaling = {key: new_scaling}
            else:
                # Single adapter
                key = "default"
                old_A = module.lora_A.weight.data
                old_B = module.lora_B.weight.data
                target_A = module.lora_A
                target_B = module.lora_B

                device = old_A.device
                dtype = old_A.dtype
                in_features = old_A.shape[1]
                out_features = old_B.shape[0]

                new_A = torch.zeros((new_rank, in_features), device=device, dtype=dtype)
                new_B = torch.zeros((out_features, new_rank), device=device, dtype=dtype)

                copy_rank = min(new_rank, old_A.shape[0])
                new_A[:copy_rank] = old_A[:copy_rank]
                new_B[:, :copy_rank] = old_B[:, :copy_rank]

                if new_A.shape == target_A.weight.shape:
                    target_A.weight.data.copy_(new_A)
                else:
                    target_A.weight = nn.Parameter(new_A)
                    self.optimizer_reset_required = True

                if new_B.shape == target_B.weight.shape:
                    target_B.weight.data.copy_(new_B)
                else:
                    target_B.weight = nn.Parameter(new_B)
                    self.optimizer_reset_required = True

                if hasattr(module, "scaling"):
                    alpha = self._get_alpha(module, key)
                    new_scaling = alpha / new_rank
                    if isinstance(module.scaling, dict):
                        module.scaling[key] = new_scaling
                    else:
                        module.scaling = {key: new_scaling}

        except Exception as e:
            logger.error(f"Error in _resize_lora_weights for module {getattr(module, 'name', 'unknown')}: {e}")
            raise

    def _get_alpha(self, module, key="default"):
        """Retrieve alpha value for the given adapter key."""
        if hasattr(module, "lora_alpha"):
            alpha_dict = module.lora_alpha
            if isinstance(alpha_dict, dict):
                return alpha_dict.get(key, 16)
            else:
                return alpha_dict
        logger.warning(f"No lora_alpha found for module {getattr(module, 'name', 'unknown')}, using default 16.")
        return 16

    def _get_lora_weights(self, module):
        """Safely retrieve lora_A and lora_B weights across PEFT versions."""
        if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
            return None, None

        lora_A = module.lora_A
        lora_B = module.lora_B

        if isinstance(lora_A, nn.ModuleDict):
            if not lora_A:
                return None, None
            key = list(lora_A.keys())[0]
            return lora_A[key].weight, lora_B[key].weight

        if hasattr(lora_A, "weight"):
            return lora_A.weight, lora_B.weight

        return None, None

    # ----------------------------------------------------------
    # 🔬 SCIENTIFIC VERIFICATION TOOL
    # ----------------------------------------------------------

    def get_progressive_summary(self):
        """
        Research-grade verification output.
        Shows which modules are actually controlled by Progressive PEFT.
        """
        summary = {
            "total_lora_modules": len(self.lora_modules),
            "modules": []
        }

        for name, module in self.model.named_modules():
            if not hasattr(module, "lora_A"):
                continue
            A, B = self._get_lora_weights(module)
            if A is None:
                continue
            rank = A.shape[0]
            summary["modules"].append({
                "name": name,
                "rank": rank,
            })

        return summary

    def clear_optimizer_flag(self):
        self.optimizer_reset_required = False


def create_progressive_peft_config_from_pydantic(cfg):
    if hasattr(cfg, "peft"):
        peft = cfg.peft
    else:
        peft = cfg

    return ProgressivePEFTConfig(
        initial_rank=getattr(peft, "initial_rank", 4),
        final_rank=getattr(peft, "final_rank", 16),
        min_rank=getattr(peft, "min_rank", 2),
        max_rank=getattr(peft, "max_rank", 32),
        rank_schedule=getattr(peft, "rank_schedule", "adaptive"),
        schedule_steps=getattr(peft, "schedule_steps", 500),
        allocation_strategy=getattr(peft, "allocation_strategy", "budget"),
        allocation_update_frequency=getattr(peft, "allocation_update_frequency", 200),
        ema_beta=getattr(peft, "ema_beta", 0.9),
        warmup_steps=getattr(peft, "warmup_steps", 50),
        rank_change_threshold=getattr(peft, "rank_change_threshold", 1),
        temperature=getattr(peft, "temperature", 1.0),
        importance_smoothing=getattr(peft, "importance_smoothing", 0.99),
    )


def wrap_model_with_progressive_peft(model: nn.Module, config: Any) -> ProgressivePEFTWrapper:
    progressive_config = create_progressive_peft_config_from_pydantic(config)
    return ProgressivePEFTWrapper(model, progressive_config)