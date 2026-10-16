# src/models/__init__.py - FIXED VERSION
"""
Models module exports - Fixed for proper imports
"""

from .peft_factory import (
    EnhancedPEFTModelFactory,
    PEFTModelFactory,
    TargetModuleResolver  # ✅ Changed from PydanticTargetModuleResolver
)

# For backward compatibility
PydanticTargetModuleResolver = TargetModuleResolver  # ✅ Add alias

from .progressive_peft import (
    ProgressivePEFTWrapper,
    create_progressive_peft_config_from_pydantic,
    wrap_model_with_progressive_peft
)

__all__ = [
    "EnhancedPEFTModelFactory",
    "PEFTModelFactory",
    "TargetModuleResolver",
    "PydanticTargetModuleResolver",  # ✅ Keep for backward compatibility
    "ProgressivePEFTWrapper",
    "create_progressive_peft_config_from_pydantic",
    "wrap_model_with_progressive_peft",
]