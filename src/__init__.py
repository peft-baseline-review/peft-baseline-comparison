"""
BOPET Research Framework
Bayesian Optimization for Parameter-Efficient Fine-Tuning
"""

import sys
from pathlib import Path

# Add the src directory to Python path
src_dir = Path(__file__).parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Version
__version__ = "1.0.0"