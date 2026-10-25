"""
Unified PEFT Factory - Optimized for Scientific Research
Production-ready with comprehensive PEFT method support
"""

import copy

import torch
import torch.nn as nn
import logging
from typing import Dict, List, Optional, Any, Union
import warnings

from transformers import AutoTokenizer

# Import handling with fallbacks
try:
    from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig, AutoConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("transformers library not available")

try:
    from src.models.dy_lora import DyLoRAWrapper
    from src.models.sa_lora import SaLoRAWrapper
except ImportError:
    # Fallback if files not found
    DyLoRAWrapper = None
    SaLoRAWrapper = None

try:
    from peft import (
        get_peft_model, TaskType,
        LoraConfig, AdaLoraConfig, IA3Config,
        PrefixTuningConfig, PromptTuningConfig,
        prepare_model_for_kbit_training
    )
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logging.warning("peft library not available")

# Configuration imports with fallbacks
try:
    from config.model import ModelConfig
    from config.peft import PEFTConfig, PEFTMethod
except ImportError:
    # Fallback configuration classes
    from dataclasses import dataclass
    from enum import Enum

    class PEFTMethod(str, Enum):
        LORA = "lora"
        DORA = "dora"
        ADALORA = "adalora"
        IA3 = "ia3"
        PREFIX = "prefix"
        PROMPT = "prompt"
        PROGRESSIVE = "progressive"
        QLORA = "qlora"
        FULL = "full"
        VANILLA = "vanilla"

    @dataclass
    class PEFTConfig:
        method: PEFTMethod
        rank: int = 8
        alpha: int = 16
        dropout: float = 0.1
        target_modules: Optional[List[str]] = None
        use_dora: bool = False
        quantization_enabled: bool = False
        compute_dtype_str: str = "float16"
        quant_type: str = "nf4"
        double_quant: bool = True
        total_steps: int = 10000                # for AdaLoRA
        modules_to_not_convert: Optional[List[str]] = None
        optimize_hyperparams: bool = True

    @dataclass
    class ModelConfig:
        name: str
        architecture: str
        torch_dtype: str = "float32"
        freeze_base: bool = False
        target_modules: Optional[Union[List[str], Dict[str, List[str]]]] = None

logger = logging.getLogger(__name__)


class TargetModuleResolver:
    """Resolves target modules for different architectures and PEFT methods"""

    ARCHITECTURE_DEFAULTS = {
        'bert': {
            'lora': ['query', 'value'],
            'dora': ['query', 'value'],
            'adalora': ['query', 'value', 'key'],
            'ia3': ['key', 'value', 'output.dense', 'intermediate.dense'],
            'prefix': ['key', 'value'],
            'prompt': ['key', 'value'],
            'progressive': ['query', 'value', 'key', 'output.dense'],
            'qlora': ['query', 'value', 'key'],
        },
        'roberta': {
            'lora': ['query', 'value'],
            'dora': ['query', 'value'],
            'adalora': ['query', 'value', 'key'],
            'ia3': ['key', 'value', 'output.dense'],
            'prefix': ['key', 'value'],
            'prompt': ['key', 'value'],
            'progressive': ['query', 'value', 'key', 'output.dense'],
            'qlora': ['query', 'value', 'key'],
        },
        'xlm-roberta': {
            'lora': ['query', 'value'],
            'dora': ['query', 'value'],
            'adalora': ['query', 'value', 'key'],
            'ia3': ['key', 'value', 'output.dense'],
            'prefix': ['key', 'value'],
            'prompt': ['key', 'value'],
            'progressive': ['query', 'value', 'key', 'output.dense'],
            'qlora': ['query', 'value', 'key'],
        },
        'gpt2': {
            'lora': ['c_attn', 'c_proj', 'c_fc'],
            'dora': ['c_attn', 'c_proj', 'c_fc'],
            'adalora': ['c_attn', 'c_proj', 'c_fc'],
            'ia3': ['c_attn', 'c_proj', 'c_fc'],
            'prefix': ['c_attn'],
            'prompt': ['wte', 'wpe'],
            'progressive': ['c_attn', 'c_proj', 'c_fc'],
            'qlora': ['c_attn', 'c_proj', 'c_fc'],
        },
    }

    @staticmethod
    def resolve(model_config: ModelConfig, peft_config: PEFTConfig) -> List[str]:
        """Resolve target modules with priority order - FIXED"""
        # Handle prefix/prompt specially - they don't use target_modules
        method = peft_config.method.value if hasattr(peft_config.method, 'value') else peft_config.method
        method = method.lower()

        if method in ['prefix', 'prompt']:
            logger.info(f"Skipping target_modules for {method} tuning")
            return []  # Return empty list, not None

        # Priority 1: Direct specification in peft_config
        if peft_config.target_modules and peft_config.target_modules != ["auto"]:
            modules = TargetModuleResolver._ensure_list(peft_config.target_modules)
            logger.info(f"Using peft_config.target_modules: {modules}")
            return modules

        # Priority 2: Model config dictionary
        if model_config.target_modules:
            if isinstance(model_config.target_modules, dict):
                if method in model_config.target_modules:
                    modules = TargetModuleResolver._ensure_list(model_config.target_modules[method])
                    logger.info(f"Using model_config.target_modules['{method}']: {modules}")
                    return modules   # No normalization – rely on substring matching
            elif isinstance(model_config.target_modules, list):
                modules = TargetModuleResolver._ensure_list(model_config.target_modules)
                logger.info(f"Using model_config.target_modules (list): {modules}")
                return modules

        # Priority 3: Architecture defaults
        arch = model_config.architecture.lower()
        arch_map = {
            'xlm-roberta': 'xlm-roberta',
            'xlm_roberta': 'xlm-roberta',
            'xlmroberta': 'xlm-roberta',
            'roberta': 'roberta',
            'bert': 'bert',
            'tiroberta': 'roberta',
            'gpt2': 'gpt2',
            'gpt-2': 'gpt2',
            'llama': 'llama',
        }
        resolved_arch = arch_map.get(arch, 'bert')
        defaults = TargetModuleResolver.ARCHITECTURE_DEFAULTS.get(
            resolved_arch,
            TargetModuleResolver.ARCHITECTURE_DEFAULTS['bert']
        )
        modules = defaults.get(method, ['query', 'value'])
        logger.info(f"Using architecture defaults ({resolved_arch}.{method}): {modules}")
        return modules

    @staticmethod
    def _ensure_list(modules) -> List[str]:
        """Ensure input is a list of strings"""
        if modules is None:
            return ['query', 'value']
        elif isinstance(modules, str):
            return [modules]
        elif isinstance(modules, list):
            return modules
        else:
            logger.warning(f"Unexpected module type: {type(modules)}")
            return ['query', 'value']


class AutoDetectTargetModuleResolver:
    """
    Scientifically robust automatic target-module detector.

    Scans the loaded model and selects attention + MLP projection layers
    compatible with PEFT methods.

    Works across:
        - BERT
        - RoBERTa
        - XLM-R
        - GPT-2
        - Most HF transformer architectures
    """

    # Keywords that usually indicate attention projections
    ATTENTION_KEYWORDS = [
        "query", "key", "value",
        "q_proj", "k_proj", "v_proj",
        "c_attn",   # GPT-2 fused attention
    ]

    # Output projection names
    OUTPUT_KEYWORDS = [
        "out_proj",
        "o_proj",
        "c_proj",
        "output.dense",
    ]

    # Feed-forward layers
    FFN_KEYWORDS = [
        "fc",
        "c_fc",
        "intermediate.dense",
        "mlp",
    ]

    @staticmethod
    def resolve(model, method: str) -> List[str]:
        """
        Auto-detect target modules from real model structure.

        Args:
            model: loaded HF model
            method: peft method name (lora, adalora, ia3, etc.)
        """
        method = method.lower()

        # Prefix/Prompt DO NOT use target modules
        if method in ["prefix", "prompt"]:
            return []

        detected = set()

        for name, module in model.named_modules():
            # Only linear-like layers
            if not AutoDetectTargetModuleResolver._is_linear_like(module):
                continue

            lname = name.lower()

            # --- Attention projections ---
            if any(k in lname for k in AutoDetectTargetModuleResolver.ATTENTION_KEYWORDS):
                detected.add(AutoDetectTargetModuleResolver._extract_leaf(name))
                continue

            # --- Output projections ---
            if any(k in lname for k in AutoDetectTargetModuleResolver.OUTPUT_KEYWORDS):
                detected.add(AutoDetectTargetModuleResolver._extract_leaf(name))
                continue

            # --- AdaLoRA / Progressive allow FFN ---
            if method in ["adalora", "progressive", "qlora", "ia3"]:
                if any(k in lname for k in AutoDetectTargetModuleResolver.FFN_KEYWORDS):
                    detected.add(AutoDetectTargetModuleResolver._extract_leaf(name))

        detected = sorted(list(detected))

        if len(detected) == 0:
            raise RuntimeError(
                "Auto-detect failed: No valid PEFT target modules found. "
                "Check model architecture."
            )

        return detected

    @staticmethod
    def _extract_leaf(module_name: str) -> str:
        """
        Converts:
            encoder.layer.0.attention.self.query
        ->  query

        This keeps PEFT substring matching stable.
        """
        return module_name.split(".")[-1]

    @staticmethod
    def _is_linear_like(module) -> bool:
        """
        Detect HF projection layers safely.
        """
        classname = module.__class__.__name__.lower()

        if isinstance(module, nn.Linear):
            return True

        # GPT-2 Conv1D projection layers
        if "conv1d" in classname:
            return True

        return False


# ============================================================
# Adaptive Scientific Target Policy v1.0
# ============================================================

class AdaptiveScientificTargetPolicy:
    """
    Enforces scientifically fair adapter placement depending on PEFT method.

    Why this matters:
        - LoRA ≠ AdaLoRA ≠ IA3 in literature.
        - Reviewers expect controlled adapter scope.
        - Prevents silent parameter inflation.

    Policy:

        LoRA / DoRA  -> Attention projections only
        AdaLoRA      -> Attention + FFN
        IA3          -> FFN-heavy + projections
        Progressive  -> Full projection set (dynamic research method)
        QLoRA        -> Same as AdaLoRA (quantized adaptive)

    Note: Sets can be extended with regex patterns if needed.
    """

    ATTENTION_SET = {
        "query", "key", "value",
        "q_proj", "k_proj", "v_proj",
        "c_attn",
    }

    OUTPUT_SET = {
        "c_proj", "out_proj", "o_proj", "output.dense"
    }

    FFN_SET = {
        "c_fc", "intermediate.dense", "mlp"
    }

    @staticmethod
    def filter(method: str, detected_modules):
        method = method.lower()
        detected = set(detected_modules)

        # LoRA / DoRA  → Attention only
        if method in ["lora", "dora"]:
            return sorted(list(
                detected.intersection(
                    AdaptiveScientificTargetPolicy.ATTENTION_SET.union(
                        AdaptiveScientificTargetPolicy.OUTPUT_SET
                    )
                )
            ))

        # AdaLoRA / QLoRA → Attention + FFN
        if method in ["adalora", "qlora"]:
            return sorted(list(
                detected.intersection(
                    AdaptiveScientificTargetPolicy.ATTENTION_SET
                    | AdaptiveScientificTargetPolicy.OUTPUT_SET
                    | AdaptiveScientificTargetPolicy.FFN_SET
                )
            ))

        # IA3 → FFN-heavy scientific setup
        if method == "ia3":
            return sorted(list(
                detected.intersection(
                    AdaptiveScientificTargetPolicy.FFN_SET
                    | AdaptiveScientificTargetPolicy.ATTENTION_SET
                )
            ))

        # Progressive → Full scientific search space
        if method == "progressive":
            return sorted(list(detected))

        # fallback
        return sorted(list(detected))


# ============================================================
# Adaptive Rank Budget Controller v1.0
# ============================================================

class AdaptiveRankBudgetController:
    """
    Dynamically rescales LoRA/AdaLoRA rank so that TOTAL adapter
    capacity remains comparable across architectures.

    Scientific Motivation:
        Prevents deeper models from receiving unfair parameter advantage.
    """

    DEFAULT_TOTAL_RANK_BUDGET = 192  # reviewer-safe default

    @staticmethod
    def compute_rank(
        base_rank: int,
        detected_modules,
        architecture: str = "bert",
        total_budget: int = None,
        min_rank: int = 2,
        max_rank: int = 64
    ) -> int:

        if total_budget is None:
            total_budget = AdaptiveRankBudgetController.DEFAULT_TOTAL_RANK_BUDGET

        # Architecture scaling for fairness
        ARCH_MULTIPLIER = {
            "gpt2": 1.4,
            "roberta": 1.0,
            "xlm-roberta": 1.0,
            "bert": 1.0,
            "tiroberta": 1.0,
        }

        mult = ARCH_MULTIPLIER.get(architecture.lower(), 1.0)

        n = max(len(detected_modules), 1)

        scaled_rank = int((total_budget * mult) / n)
        final_rank = min(base_rank, scaled_rank)
        final_rank = max(min_rank, min(final_rank, max_rank))

        logger.info(
            f"[RankBudget] arch={architecture} modules={n} "
            f"| budget={total_budget} → rank={final_rank}"
        )

        return final_rank


# ============================================================
# Progressive Rank Thermostat v2.0
# ============================================================

class ProgressiveRankThermostat:
    """
    Scientifically grounded dynamic rank scheduler.

    Converts adaptive_rank into progressive rank bounds.

    Motivation:
        Keeps Progressive PEPT comparable across architectures.
    """

    INIT_SCALE = 0.5
    FINAL_SCALE = 2.0

    MIN_RANK = 2
    MAX_RANK = 128

    @staticmethod
    def compute_bounds(adaptive_rank: int):

        init_rank = int(adaptive_rank * ProgressiveRankThermostat.INIT_SCALE)
        final_rank = int(adaptive_rank * ProgressiveRankThermostat.FINAL_SCALE)

        init_rank = max(ProgressiveRankThermostat.MIN_RANK, init_rank)
        final_rank = min(ProgressiveRankThermostat.MAX_RANK, final_rank)

        logger.info(
            f"[RankThermostat] adaptive={adaptive_rank} "
            f"→ init={init_rank}, final={final_rank}"
        )

        return init_rank, final_rank


# ============================================================
# Adaptive Architecture Prior v2
# ============================================================

class AdaptiveArchitecturePriorV2:
    """
    Language- and architecture-aware adapter biasing.

    This module modifies detected PEFT target modules BEFORE
    scientific filtering.

    Goal:
        Encourage fair but linguistically informed placement.

    Design Principles:
        - Monolingual models (TiRoBERTa) benefit from FFN expansion.
        - Multilingual encoders stay attention-focused.
        - GPT-style decoders favor projection layers.
    """

    @staticmethod
    def apply(model_config, detected_modules):

        arch = model_config.architecture.lower()
        name = model_config.name.lower()

        detected = set(detected_modules)

        logger.info(f"[ArchPrior] Applying adaptive prior for {arch}")

        # --------------------------------------------------
        # TIRoBERTa / RoBERTa-style monolingual encoder
        # --------------------------------------------------
        if "tiroberta" in name or arch == "tiroberta":
            logger.info("[ArchPrior] TiRoBERTa detected → FFN enrichment")
            detected.update({
                "intermediate.dense",
                "output.dense",
            })

        # --------------------------------------------------
        # XLM-R multilingual encoder
        # --------------------------------------------------
        elif arch in ["xlm-roberta", "xlm_roberta"]:
            logger.info("[ArchPrior] XLM-R detected → balanced attention")
            detected.update({
                "query",
                "key",
                "value",
            })

        # --------------------------------------------------
        # mBERT-like encoders
        # --------------------------------------------------
        elif arch == "bert":
            logger.info("[ArchPrior] BERT prior → attention focused")
            detected.update({
                "query",
                "value",
            })

        # --------------------------------------------------
        # GPT-2 decoder prior
        # --------------------------------------------------
        elif arch == "gpt2":
            logger.info("[ArchPrior] GPT-2 prior → projection heavy")
            detected.update({
                "c_proj",
                "c_fc",
            })

        return sorted(list(detected))


# ============================================================
# Main PEFT Model Factory
# ============================================================

class PEFTModelFactory:
    """Unified factory for creating PEFT models with scientific rigor"""

    @staticmethod
    def create_model(
        model_config: ModelConfig,
        peft_config: PEFTConfig,
        num_labels: int,
        device: Optional[torch.device] = None
    ) -> nn.Module:
        """Main entry point - create model from configuration"""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library is required")

        method = peft_config.method.value if hasattr(peft_config.method, 'value') else peft_config.method
        method = method.lower()

        logger.info(f"Creating {method.upper()} model: {model_config.name}")

        # Handle special cases
        if method in ["full", "finetune"]:
            return PEFTModelFactory._create_full_finetuning(model_config, num_labels, device)

        if method == "vanilla":
            return PEFTModelFactory._create_vanilla_model(model_config, num_labels, device)

        if method == "qlora" or peft_config.quantization_enabled:
            return PEFTModelFactory._create_qlora_model(model_config, peft_config, num_labels)
        
        if method == "dy_lora":
            return PEFTModelFactory._create_dy_lora_model(model_config, peft_config, num_labels, device)
        
        if method == "sa_lora":
            return PEFTModelFactory._create_sa_lora_model(model_config, peft_config, num_labels, device)

        if method == "progressive":
            return PEFTModelFactory._create_progressive_model(model_config, peft_config, num_labels, device)

        # Standard PEFT methods
        return PEFTModelFactory._create_standard_model(model_config, peft_config, num_labels, device)

    @staticmethod
    def _create_full_finetuning(
        model_config: ModelConfig,
        num_labels: int,
        device: Optional[torch.device] = None
    ) -> nn.Module:
        """Create model for full fine-tuning"""
        logger.info("Creating full fine-tuning model")

        try:
            torch_dtype = getattr(torch, model_config.torch_dtype)
        except AttributeError:
            torch_dtype = torch.float32

        # GPT-2 needs different model class for sequence classification
        if "gpt2" in model_config.name.lower():
            tokenizer = AutoTokenizer.from_pretrained(model_config.name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                print(f"Set pad_token = eos_token: {tokenizer.pad_token}")

            model = AutoModelForSequenceClassification.from_pretrained(
                model_config.name,
                num_labels=num_labels,
                torch_dtype=torch_dtype,
                pad_token_id=tokenizer.pad_token_id  # Important for GPT-2
            )
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_config.name,
                num_labels=num_labels,
                torch_dtype=torch_dtype,
                ignore_mismatched_sizes=True
            )

        # Configure parameter training
        if not model_config.freeze_base:
            logger.info("Full fine-tuning - training all parameters")
        else:
            logger.info("Full fine-tuning with frozen base")
            for name, param in model.named_parameters():
                if "classifier" not in name:
                    param.requires_grad = False

        PEFTModelFactory._log_parameters(model)

        if device:
            model = model.to(device)

        return model
 
    @staticmethod
    def _create_vanilla_model(
        model_config: ModelConfig,
        num_labels: int,
        device: Optional[torch.device] = None
    ) -> nn.Module:
        """Create model with only classifier trainable."""
        logger.info("Creating vanilla model (classifier only)")

        try:
            torch_dtype = getattr(torch, model_config.torch_dtype)
        except AttributeError:
            torch_dtype = torch.float32

        # Load base model (handle GPT‑2 specially)
        if "gpt2" in model_config.name.lower():
            tokenizer = AutoTokenizer.from_pretrained(model_config.name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                print(f"Set pad_token = eos_token: {tokenizer.pad_token}")

            model = AutoModelForSequenceClassification.from_pretrained(
                model_config.name,
                num_labels=num_labels,
                torch_dtype=torch_dtype,
                pad_token_id=tokenizer.pad_token_id
            )
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_config.name,
                num_labels=num_labels,
                torch_dtype=torch_dtype,
                ignore_mismatched_sizes=True
            )

        # Freeze all except classifier (handling different naming conventions)
        classifier_keywords = ["classifier", "score", "out_proj"]  # common names
        for name, param in model.named_parameters():
            if any(keyword in name for keyword in classifier_keywords):
                param.requires_grad = True
                logger.debug(f"Trainable: {name}")
            else:
                param.requires_grad = False

        # Verify at least one trainable parameter exists
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if trainable_params == 0:
            raise RuntimeError(
                "Vanilla model has zero trainable parameters. "
                "Check classifier layer naming for this architecture."
            )

        PEFTModelFactory._log_parameters(model)

        if device:
            model = model.to(device)

        return model
    @staticmethod
    def _create_dy_lora_model(model_config, peft_config, num_labels, device):
        """Create DyLoRA model with proper parameter freezing."""
        if DyLoRAWrapper is None:
            raise ImportError("DyLoRAWrapper not available. Check src/models/dy_lora.py")

        # Load the raw base model (no PEFT)
        try:
            torch_dtype = getattr(torch, model_config.torch_dtype)
        except AttributeError:
            torch_dtype = torch.float32

        if "gpt2" in model_config.name.lower():
            tokenizer = AutoTokenizer.from_pretrained(model_config.name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForSequenceClassification.from_pretrained(
                model_config.name,
                num_labels=num_labels,
                torch_dtype=torch_dtype,
                pad_token_id=tokenizer.pad_token_id,
            )
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_config.name,
                num_labels=num_labels,
                torch_dtype=torch_dtype,
                ignore_mismatched_sizes=True,
            )

        # Freeze the entire base model (all parameters)
        for param in model.parameters():
            param.requires_grad = False

        # Determine target modules (same logic as for LoRA)
        target_modules = peft_config.target_modules
        if target_modules is None or target_modules == ["auto"]:
            # Auto‑detect on the raw model
            raw_modules = AutoDetectTargetModuleResolver.resolve(model, "lora")
            raw_modules = AdaptiveArchitecturePriorV2.apply(model_config, raw_modules)
            target_modules = AdaptiveScientificTargetPolicy.filter("lora", raw_modules)
            logger.info(f"[DyLoRA] Auto-detected target modules: {target_modules}")

        # Wrap with DyLoRAWrapper (replaces selected linear layers with DyLoRALinear)
        max_rank = getattr(peft_config, 'max_rank', 16)
        min_rank = getattr(peft_config, 'min_rank', 2)
        alpha = getattr(peft_config, 'alpha', 16)
        dropout = getattr(peft_config, 'dropout', 0.1)
        model = DyLoRAWrapper(model, target_modules, max_rank, min_rank, alpha, dropout)

        # After wrapping, freeze ALL parameters
        for param in model.parameters():
            param.requires_grad = False

        # Then unfreeze only the lora parameters and the classifier head
        for name, param in model.named_parameters():
            if 'lora' in name or any(keyword in name for keyword in ['classifier', 'score', 'out_proj']):
                param.requires_grad = True

        # Log parameters (should now show correct total and trainable)
        PEFTModelFactory._log_parameters(model)

        if device:
            model = model.to(device)
        return model

    @staticmethod
    def _create_sa_lora_model(model_config, peft_config, num_labels, device):
        """Create SaLoRA model."""
        if SaLoRAWrapper is None:
            raise ImportError("SaLoRAWrapper not available. Check src/models/sa_lora.py")
        # Create standard LoRA base model
        base_config = copy.deepcopy(peft_config)
        base_config.method = "lora"
        base_model = PEFTModelFactory._create_standard_model(model_config, base_config, num_labels, device)
        initial_rank = getattr(peft_config, 'initial_rank', 16)
        final_rank = getattr(peft_config, 'final_rank', 4)
        prune_frequency = getattr(peft_config, 'prune_frequency', 500)
        prune_ratio = getattr(peft_config, 'prune_ratio', 0.3)
        model = SaLoRAWrapper(base_model, initial_rank, final_rank, prune_frequency, prune_ratio)
        model.prune_criterion = getattr(peft_config, 'prune_criterion', 'magnitude')
        return model

    @staticmethod
    def _create_standard_model(
        model_config: ModelConfig,
        peft_config: PEFTConfig,
        num_labels: int,
        device: Optional[torch.device] = None
    ) -> nn.Module:
        if not PEFT_AVAILABLE:
            raise ImportError("peft library is required for PEFT models")

        method = peft_config.method.value if hasattr(peft_config.method, 'value') else peft_config.method
        method = method.lower()

        try:
            torch_dtype = getattr(torch, model_config.torch_dtype)
        except AttributeError:
            torch_dtype = torch.float32

        # Load base model (handle GPT-2 specially but with unified path for QLoRA? No, QLoRA handled separately)
        if "gpt2" in model_config.name.lower():
            tokenizer = AutoTokenizer.from_pretrained(model_config.name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForSequenceClassification.from_pretrained(
                model_config.name,
                num_labels=num_labels,
                torch_dtype=torch_dtype,
                pad_token_id=tokenizer.pad_token_id,
            )
            model.config.problem_type = "single_label_classification"
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_config.name,
                num_labels=num_labels,
                torch_dtype=torch_dtype,
                ignore_mismatched_sizes=True
            )

        # Handle prefix and prompt tuning specially
        if method in ['prefix', 'prompt']:
            logger.info(f"Creating {method} tuning model (no target modules)")
            if method == 'prefix':
                config = PrefixTuningConfig(
                    task_type=TaskType.SEQ_CLS,
                    num_virtual_tokens=peft_config.num_virtual_tokens if hasattr(peft_config, 'num_virtual_tokens') else 20,
                    encoder_hidden_size=getattr(model.config, 'hidden_size', 768),
                    prefix_projection=False,
                )
            else:  # method == 'prompt'
                config = PromptTuningConfig(
                    task_type=TaskType.SEQ_CLS,
                    num_virtual_tokens=peft_config.num_virtual_tokens if hasattr(peft_config, 'num_virtual_tokens') else 20,
                    tokenizer_name_or_path=model_config.name,
                )
        else:
            if peft_config.target_modules is not None and peft_config.target_modules != ["auto"]:
                # Use explicitly provided target modules
                target_modules = peft_config.target_modules
                logger.info(f"Using explicit target modules from config: {target_modules}")
            else:
                # Standard PEFT methods with target modules
                raw_modules = AutoDetectTargetModuleResolver.resolve(model, method)
                raw_modules = AdaptiveArchitecturePriorV2.apply(model_config, raw_modules)
                target_modules = AdaptiveScientificTargetPolicy.filter(method, raw_modules)
                logger.info(f"[ScientificPolicy] {method} → {target_modules}")

            # --- Adaptive Rank Budget (conditional) ---
            base_rank = peft_config.rank if peft_config.rank is not None else 8
            if getattr(peft_config, 'use_rank_budget', False):
                adaptive_rank = AdaptiveRankBudgetController.compute_rank(
                    base_rank,
                    target_modules,
                    model_config.architecture,
                    total_budget=AdaptiveRankBudgetController.DEFAULT_TOTAL_RANK_BUDGET
                )
            else:
                adaptive_rank = base_rank

            logger.info(f"Architecture detected: {model_config.architecture}")
            logger.info(f"Rank used: {adaptive_rank} (budget scaling: {getattr(peft_config, 'use_rank_budget', False)})")

            if method in ["lora", "dora"]:
                config = LoraConfig(
                    task_type=TaskType.SEQ_CLS,
                    r=adaptive_rank,
                    lora_alpha=peft_config.alpha if peft_config.alpha is not None else 16,
                    lora_dropout=peft_config.dropout if peft_config.dropout is not None else 0.1,
                    target_modules=target_modules,
                    bias=getattr(peft_config, 'bias', "none"),
                    use_dora=(method == "dora" or getattr(peft_config, 'use_dora', False))
                )
            elif method == "adalora":
                # Use total_steps from config with fallback
                total_steps = getattr(peft_config, 'total_steps', 10000)

                # New optional fields enable the native dynamic pruning
                # schedule. If unset, fall back to fixed-rank behavior
                # (identical to the previous implementation).
                _init_r = getattr(peft_config, 'adalora_init_rank', None)
                _target_r = getattr(peft_config, 'adalora_target_rank', None)
                _init_r = adaptive_rank if _init_r is None else int(_init_r)
                _target_r = adaptive_rank if _target_r is None else int(_target_r)
                if _init_r < _target_r:
                    raise ValueError(
                        f"AdaLoRA: adalora_init_rank ({_init_r}) < "
                        f"adalora_target_rank ({_target_r}); "
                        f"init_r must be >= target_r."
                    )
                logger.info(
                    f"[AdaLoRA] init_r={_init_r}, target_r={_target_r}, "
                    f"total_step={total_steps}"
                )
                config = AdaLoraConfig(
                    task_type=TaskType.SEQ_CLS,
                    target_r=_target_r,
                    init_r=_init_r,
                    lora_alpha=peft_config.alpha if peft_config.alpha is not None else 16,
                    lora_dropout=peft_config.dropout if peft_config.dropout is not None else 0.1,
                    target_modules=target_modules,
                    bias=getattr(peft_config, 'bias', "none"),
                    total_step=total_steps,
                    tinit=100,
                    tfinal=1000,
                    deltaT=10,
                    beta1=0.85,
                    beta2=0.85,
                )
            elif method == "ia3":
                ff_modules = [m for m in target_modules if m in AdaptiveScientificTargetPolicy.FFN_SET]
                config = IA3Config(
                    task_type=TaskType.SEQ_CLS,
                    target_modules=target_modules,
                    feedforward_modules=ff_modules,
                )
            else:
                logger.warning(f"Unknown method {method}, falling back to LoRA")
                config = LoraConfig(
                    task_type=TaskType.SEQ_CLS,
                    r=adaptive_rank,
                    lora_alpha=peft_config.alpha,
                    lora_dropout=peft_config.dropout,
                    target_modules=target_modules,
                    bias="none"
                )

        model = get_peft_model(model, config)
        PEFTModelFactory._log_parameters(model)
        if hasattr(model, 'peft_config'):
            logger.info(f"Applied PEFT config: {model.peft_config}")
            cfg = list(model.peft_config.values())[0]
            if hasattr(cfg, "target_modules"):
                logger.info(f"Target modules used: {cfg.target_modules}")

        if device:
            model = model.to(device)

        return model

    @staticmethod
    def _create_qlora_model(
        model_config: ModelConfig,
        peft_config: PEFTConfig,
        num_labels: int
    ) -> nn.Module:
        if not PEFT_AVAILABLE:
            raise ImportError("peft library is required for QLoRA")

        compute_dtype = getattr(torch, peft_config.compute_dtype_str)

        # Determine modules to exclude from quantization
        if peft_config.modules_to_not_convert is not None:
            modules_to_not_convert = peft_config.modules_to_not_convert
        else:
            # Architecture‑specific defaults
            if "roberta" in model_config.name.lower() or model_config.architecture.lower() == "roberta":
                modules_to_not_convert = ["classifier"]
            elif "gpt2" in model_config.name.lower():
                modules_to_not_convert = ["lm_head", "score"]
            else:
                modules_to_not_convert = ["classifier", "score"]  # fallback

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=peft_config.quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=peft_config.double_quant,
            modules_to_not_convert=modules_to_not_convert,
        )

        # Unified loading for all models (including GPT-2)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_config.name,
            num_labels=num_labels,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=compute_dtype,
        )

        # Special handling for GPT-2 tokenizer (if needed)
        if "gpt2" in model_config.name.lower():
            tokenizer = AutoTokenizer.from_pretrained(model_config.name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.pad_token_id
            model.config.problem_type = "single_label_classification"

        model = prepare_model_for_kbit_training(model)

        method = peft_config.method.value if hasattr(peft_config.method, "value") else peft_config.method
        method = method.lower()

        if peft_config.target_modules is not None and peft_config.target_modules != ["auto"]:
            target_modules = peft_config.target_modules
            logger.info(f"Using explicit target modules from config: {target_modules}")
        else:
            raw_modules = AutoDetectTargetModuleResolver.resolve(model, method)
            raw_modules = AdaptiveArchitecturePriorV2.apply(model_config, raw_modules)
            target_modules = AdaptiveScientificTargetPolicy.filter(method, raw_modules)
            logger.info(f"[ScientificPolicy] {method} → {target_modules}")

        # --- Adaptive Rank Budget (conditional) ---
        base_rank = peft_config.rank if peft_config.rank is not None else 8
        if getattr(peft_config, 'use_rank_budget', False):
            adaptive_rank = AdaptiveRankBudgetController.compute_rank(
                base_rank,
                target_modules,
                model_config.architecture,
                total_budget=AdaptiveRankBudgetController.DEFAULT_TOTAL_RANK_BUDGET
            )
        else:
            adaptive_rank = base_rank

        logger.info(f"Rank used: {adaptive_rank} (budget scaling: {getattr(peft_config, 'use_rank_budget', False)})")

        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=adaptive_rank,
            lora_alpha=peft_config.alpha,
            lora_dropout=peft_config.dropout,
            target_modules=target_modules,
            bias="none",
        )

        model = get_peft_model(model, lora_config)
        PEFTModelFactory._log_parameters(model)
        return model

    @staticmethod
    def _create_progressive_model(
        model_config: ModelConfig,
        peft_config: PEFTConfig,
        num_labels: int,
        device: Optional[torch.device] = None
    ) -> nn.Module:
        """
        Create Progressive PEFT model with support for explicit target modules.
        If peft_config.target_modules is provided (and not "auto"), it is used directly.
        Otherwise, auto-detection and scientific policies are applied.
        """
        logger.info("Creating progressive PEFT model")

        # Import progressive wrapper (fail gracefully if not available)
        try:
            from src.models.progressive_peft import (
                ProgressivePEFTWrapper,
                create_progressive_peft_config_from_pydantic,
                wrap_model_with_progressive_peft
            )
        except ImportError:
            logger.warning("Progressive PEPT not available, falling back to LoRA")
            peft_config.method = PEFTMethod.LORA if hasattr(PEFTMethod, "LORA") else "lora"
            return PEFTModelFactory._create_standard_model(
                model_config, peft_config, num_labels, device
            )

        # ------------------------------------------------------------------
        # Step 1: Determine target modules (explicit or auto-detected)
        # ------------------------------------------------------------------
        temp_model = None
        try:
            if peft_config.target_modules is not None and peft_config.target_modules != ["auto"]:
                # Use explicitly provided modules
                target_modules = peft_config.target_modules
                logger.info(f"Using explicit target modules for progressive: {target_modules}")
                # We still need the module count for rank budget, which we have.
            else:
                # Load temporary base model for auto-detection
                logger.info("Loading temporary base model for scientific detection")
                try:
                    torch_dtype = getattr(torch, model_config.torch_dtype)
                except AttributeError:
                    torch_dtype = torch.float32

                if "gpt2" in model_config.name.lower():
                    tokenizer = AutoTokenizer.from_pretrained(model_config.name)
                    if tokenizer.pad_token is None:
                        tokenizer.pad_token = tokenizer.eos_token
                    temp_model = AutoModelForSequenceClassification.from_pretrained(
                        model_config.name,
                        num_labels=num_labels,
                        torch_dtype=torch_dtype,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                else:
                    temp_model = AutoModelForSequenceClassification.from_pretrained(
                        model_config.name,
                        num_labels=num_labels,
                        torch_dtype=torch_dtype,
                        ignore_mismatched_sizes=True,
                    )

                # Auto-detect raw modules
                raw_modules = AutoDetectTargetModuleResolver.resolve(temp_model, "progressive")
                # Apply architecture prior
                raw_modules = AdaptiveArchitecturePriorV2.apply(model_config, raw_modules)
                # Apply scientific policy (for progressive, this returns all detected modules)
                target_modules = AdaptiveScientificTargetPolicy.filter("progressive", raw_modules)
                logger.info(f"[ScientificPolicy] progressive → {target_modules}")

        finally:
            if temp_model is not None:
                del temp_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # ------------------------------------------------------------------
        # Step 2: Compute adaptive rank (if budget scaling enabled)
        # ------------------------------------------------------------------
        base_rank = peft_config.rank if peft_config.rank is not None else 8
        if getattr(peft_config, 'use_rank_budget', False):
            adaptive_rank = AdaptiveRankBudgetController.compute_rank(
                base_rank,
                target_modules,
                model_config.architecture,
                total_budget=AdaptiveRankBudgetController.DEFAULT_TOTAL_RANK_BUDGET
            )
        else:
            adaptive_rank = base_rank

        # ------------------------------------------------------------------
        # Step 3: Apply progressive rank thermostat
        # ------------------------------------------------------------------
        # After computing adaptive_rank
        if peft_config.allocation_strategy == "structure" or peft_config.rank_schedule == "fixed":
            # No thermostat scaling – use the adaptive rank as the constant global budget
            init_rank = adaptive_rank
            final_rank = adaptive_rank
            logger.info(f"[ProgressiveSetup] Fixed budget mode: rank={adaptive_rank}")
        else:
            # Apply thermostat for growth schedules
            init_rank, final_rank = ProgressiveRankThermostat.compute_bounds(adaptive_rank)
            logger.info(f"[ProgressiveSetup] adaptive_rank={adaptive_rank} | init={init_rank} | final={final_rank}")
        logger.info(f"Rank budget scaling: {getattr(peft_config, 'use_rank_budget', False)}")

        # Update config with computed values
        peft_config.initial_rank = init_rank
        peft_config.final_rank = final_rank
        peft_config.rank = adaptive_rank
        peft_config.target_modules = target_modules

        # ------------------------------------------------------------------
        # Step 4: Create base LoRA model with the same target modules
        # ------------------------------------------------------------------
        logger.info("Creating LoRA base model for Progressive wrapping")
        base_config = PEFTConfig(
            method=PEFTMethod.LORA if hasattr(PEFTMethod, "LORA") else "lora",
            rank=adaptive_rank,
            alpha=peft_config.alpha,
            dropout=peft_config.dropout,
            target_modules=target_modules,   # use same modules
        )

        base_model = PEFTModelFactory._create_standard_model(
            model_config,
            base_config,
            num_labels,
            device,
        )

        # Safety check: ensure LoRA adapters were actually added
        if not any("lora_A" in n for n, _ in base_model.named_modules()):
            raise RuntimeError(
                "No LoRA adapters detected before Progressive wrapping. "
                "Check target_modules resolution."
            )

        # ------------------------------------------------------------------
        # Step 5: Wrap with ProgressivePEFTWrapper
        # ------------------------------------------------------------------
        model = wrap_model_with_progressive_peft(base_model, peft_config)
        logger.info("Created progressive PEFT model successfully")

        return model

    @staticmethod
    def _log_parameters(model: nn.Module):
        """Log parameter counts"""
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        logger.info(f"Parameters: {trainable:,} trainable / {total:,} total ({trainable/total*100:.2f}%)")
        if hasattr(model, "peft_config"):
            logger.info(f"Active adapters: {list(model.peft_config.keys())}")

    @staticmethod
    def count_parameters_with_allocation(model: nn.Module) -> Dict[str, Any]:
        """Count parameters including allocation details"""
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        counts = {
            "total": total,
            "trainable": trainable,
            "trainable_percentage": (trainable / total * 100) if total > 0 else 0,
        }

        # Detect progressive PEPT models
        model_class_name = model.__class__.__name__
        if "Progressive" in model_class_name:
            logger.info(f"Detected progressive model: {model_class_name}")
            counts["progressive"] = True

            if hasattr(model, 'get_allocation_summary'):
                try:
                    counts["allocation_summary"] = model.get_allocation_summary()
                except Exception as e:
                    logger.warning(f"Failed to get allocation summary: {e}")

        return counts


# Alias for backward compatibility
EnhancedPEFTModelFactory = PEFTModelFactory