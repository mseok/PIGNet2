"""
PIGNet: Deep learning framework for protein-ligand interaction prediction.

This package provides tools for predicting binding affinity between proteins and ligands
using graph neural networks with physics-based features.

Main Features:
- Protein-ligand binding affinity prediction
- Support for ensemble models
- Physics-based interaction features
- Batch processing capabilities
- Detailed interaction analysis
"""

from pathlib import Path

# Version information
__version__ = "2.0.0"

# Import models
from .models import PIGNet, PIGNetMorse

# Import data processing
from .data import ComplexDataModule, complex_to_data

# Import core functionality (will be created)
try:
    from .core import PIGNetCore

    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False

# Import utilities (will be created)
try:
    from .utils import (
        batch_predict_from_dataframe,
        calculate_enrichment_factor,
        compare_models,
        quick_predict,
        screen_compound_library,
    )

    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False

# Import analysis tools (will be created)
try:
    from .analysis import (
        generate_detailed_explanation,
        analyze_interactions,
        extract_pocket_residues,
        fragment_ligand,
    )

    ANALYSIS_AVAILABLE = True
except ImportError:
    ANALYSIS_AVAILABLE = False

# Re-export for convenience
__all__ = [
    # Models
    "PIGNet",
    "PIGNetMorse",
    # Data
    "ComplexDataModule",
    "complex_to_data",
    # Core (if available)
    "PIGNetCore",
    # Utilities (if available)
    "batch_predict_from_dataframe",
    "screen_compound_library",
    "calculate_enrichment_factor",
    "compare_models",
    "quick_predict",
    # Analysis (if available)
    "generate_detailed_explanation",
    "analyze_interactions",
    "extract_pocket_residues",
    "fragment_ligand",
    # Version
    "__version__",
]
