# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

PIGNet2 is a deep learning framework for protein-ligand binding affinity prediction using graph neural networks with physics-based features. The repository is currently undergoing migration from a legacy `src/` directory structure to a modern modularized `pignet2/` package structure.

## Key Architecture

### Dual Codebase Structure
- **Legacy (`src/`)**: Original research codebase with script-based execution, Hydra configuration, flat module structure
- **New (`pignet2/`)**: Modern Python package with CLI interface, LLM agent integration, async support, and enhanced analysis capabilities

### Core Components
- **Models**: Graph neural networks (GatedGAT, InteractionNet) with physics-based features
- **Data Processing**: RDKit and MDAnalysis for molecular structure handling and protonation
- **Configuration**: Hydra YAML configs (legacy) + Click CLI (new)
- **Checkpoints**: Pre-trained models in `src/ckpt/` (pda_0.pt to pda_3.pt)

## Development Commands

### Installation and Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Install pre-commit hooks
pre-commit install
```

### Code Quality
```bash
# Format code with black and isort (automatically via pre-commit)
black pignet2/ src/
isort pignet2/ src/

# Run pre-commit manually
pre-commit run --all-files
```

### Training
```bash
# Legacy approach (still functional)
cd experiments
bash training_scripts/baseline.sh

# New CLI approach (preferred)
pignet2 train --config baseline --epochs 300
```

### Prediction
```bash
# Legacy approach
python src/exe/predict.py ./src/ckpt/pda_0.pt -p protein.pdb -l ligand.sdf

# New CLI approach (preferred)
pignet2 predict model.pt -p protein.pdb -l ligand.sdf
pignet2 quick -p protein.pdb -l ligand.sdf  # Uses ensemble of all 4 models
```

### Benchmarking
```bash
# Run CASF2016 benchmark
bash experiments/benchmark_scripts/casf2016_scoring.sh
python src/benchmark/scoring_power.py -f results.txt

# New CLI approach
pignet2 benchmark scoring --model model.pt --dataset casf2016
```

### Testing Single Components
```bash
# Test prediction on example files
python -m pignet2.predict src/ckpt/pda_0.pt -p examples/1a0q/1a0q_protein.pdb -l examples/1a0q/1a0q_ligand.sdf

# Test data preprocessing
python -m pignet2.data.preprocess -p protein.pdb -l ligand.sdf
```

## Migration Guidelines

### When Working with Legacy Code (`src/`)
- Uses direct imports without package structure
- Requires manual path manipulation in scripts
- Configuration via Hydra YAML files in `src/config/`
- Execution through standalone scripts in `src/exe/`

### When Working with New Code (`pignet2/`)
- Uses proper Python package structure with relative imports
- Configuration via Click CLI decorators
- Two model loading approaches:
  - `PIGNet2Core`: Direct in-memory loading (fast)
  - `PIGNet2Wrapper`: Subprocess-based (compatibility)
- Enhanced features: LLM agents, async support, detailed interaction explanations

### Key Migration Tasks
1. **Models**: Core architectures migrated, maintain compatibility with legacy checkpoints
2. **Data Processing**: Enhanced with MDAnalysis, backward compatible with RDKit-only approach
3. **CLI**: New Click-based interface wraps legacy functionality
4. **Configuration**: Still uses Hydra configs from `src/config/`, gradual migration to CLI args

## Important Implementation Details

### Model Loading Strategy
```python
# Direct loading (new, fast)
from pignet2.core import PIGNet2Core
model = PIGNet2Core(checkpoint_path)

# Subprocess wrapper (compatibility)
from pignet2.pignet_wrapper import PIGNet2Wrapper
model = PIGNet2Wrapper(checkpoint_path)
```

### Physics-Based Features
- Van der Waals interactions
- Hydrogen bonding
- Hydrophobic contacts
- Salt bridges
- π-π stacking
- Cation-π interactions
- Calculated in `models/physics.py` (both versions)

### Data Processing Pipeline
1. Protonation (OpenBabel or MDAnalysis)
2. Feature extraction (atomic, residue-level)
3. Graph construction (PyTorch Geometric)
4. Physics feature calculation
5. Batch processing with DataLoader

### Configuration Management
- Hydra configs in `src/config/` and `pignet2/config/`
- Structured as: `data/`, `model/`, `run/` subdirectories
- Override with CLI: `--config-name baseline`

## Common Development Patterns

### Adding New Features to pignet2
1. Implement in appropriate module under `pignet2/`
2. Add CLI command in `pignet2/cli.py` if user-facing
3. Maintain backward compatibility with legacy code
4. Use type hints and follow existing code style

### Modifying Model Architecture
1. Changes go in `pignet2/models/` (new) or `src/models/` (legacy)
2. Ensure checkpoint compatibility if modifying existing models
3. Update both `PIGNet2Core` and `PIGNet2Wrapper` if needed

### Working with Configurations
1. Hydra configs define model architecture, training params, data settings
2. New features should support both Hydra config and CLI arguments
3. Default configs: `baseline.yaml`, `pignet.yaml`, `pignet_covalent.yaml`

## Critical Files and Their Purposes

### Core Model Files
- `pignet2/models/pignet.py`: Main PIGNet model implementation
- `pignet2/models/layers/`: Custom GNN layers (GatedGAT, InteractionNet)
- `pignet2/models/physics.py`: Physics-based feature calculations

### Entry Points
- `pignet2/cli.py`: Main CLI interface (new)
- `src/exe/train.py`: Training script (legacy)
- `src/exe/predict.py`: Prediction script (legacy)

### Data Processing
- `pignet2/data/preprocess.py`: Enhanced preprocessing with MDAnalysis
- `src/data/data.py`: Legacy data loading and processing

### Integration Points
- `pignet2/agents.py`: LLM agent tools (LangChain, LlamaIndex)
- `pignet2/async_wrapper.py`: Asynchronous prediction interface
- `pignet2/analysis/`: Enhanced interaction analysis and explanation

## Dependencies and Environment

### Core Dependencies
- PyTorch 1.9.1 with CUDA 11.1
- PyTorch Geometric 2.0.3
- RDKit for molecular processing
- MDAnalysis for enhanced structure analysis
- Hydra for configuration management
- Click for CLI interface

### Optional Dependencies
- LangChain/LlamaIndex for agent integration
- asyncio for async predictions
- tqdm for progress bars during testing