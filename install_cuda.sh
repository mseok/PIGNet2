#!/bin/bash

# Installation script for PIGNet2 with CUDA 12 support
# This script installs PyTorch and PyTorch Geometric with proper CUDA versions

echo "PIGNet2 Installation with CUDA 12 Support"
echo "========================================="

# Detect CUDA version
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1-2)
    echo "Detected CUDA version: $CUDA_VERSION"
else
    echo "Warning: nvidia-smi not found. Assuming CUDA 12.4"
    CUDA_VERSION="12.4"
fi

# Parse command line arguments
CUDA_VERSION_ARG=""
CPU_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --cuda)
            CUDA_VERSION_ARG="$2"
            shift 2
            ;;
        --cpu)
            CPU_ONLY=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--cuda VERSION] [--cpu]"
            echo "  --cuda VERSION: Specify CUDA version (e.g., 11.8, 12.4)"
            echo "  --cpu: Install CPU-only version"
            exit 1
            ;;
    esac
done

# Override CUDA version if specified
if [ ! -z "$CUDA_VERSION_ARG" ]; then
    CUDA_VERSION="$CUDA_VERSION_ARG"
    echo "Using specified CUDA version: $CUDA_VERSION"
fi

# Set PyTorch index URL based on CUDA version
if [ "$CPU_ONLY" = true ]; then
    echo "Installing CPU-only version..."
    TORCH_INDEX="https://download.pytorch.org/whl/cpu"
    CUDA_TAG="cpu"
elif [[ "$CUDA_VERSION" == "12"* ]]; then
    echo "Installing for CUDA 12.x..."
    TORCH_INDEX="https://download.pytorch.org/whl/cu124"
    CUDA_TAG="cu124"
elif [[ "$CUDA_VERSION" == "11.8" ]]; then
    echo "Installing for CUDA 11.8..."
    TORCH_INDEX="https://download.pytorch.org/whl/cu118"
    CUDA_TAG="cu118"
else
    echo "Warning: Unsupported CUDA version $CUDA_VERSION"
    echo "Defaulting to CUDA 12.4"
    TORCH_INDEX="https://download.pytorch.org/whl/cu124"
    CUDA_TAG="cu124"
fi

echo ""
echo "Step 1: Installing PyTorch with CUDA support..."
echo "------------------------------------------------"
pip install torch==2.6.0 --index-url $TORCH_INDEX

# Verify PyTorch installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')" || exit 1

echo ""
echo "Step 2: Installing PyTorch Geometric and related packages..."
echo "------------------------------------------------------------"

# Get PyTorch version for PyG installation
TORCH_VERSION=$(python -c "import torch; print(torch.__version__.split('+')[0])")
echo "PyTorch version: $TORCH_VERSION"

# Install PyTorch Geometric packages
if [ "$CPU_ONLY" = true ]; then
    pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+cpu.html --force-reinstall
else
    pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_TAG}.html --force-reinstall
fi

# Install PyTorch Geometric
pip install torch-geometric

echo ""
echo "Step 3: Installing PIGNet2 and remaining dependencies..."
echo "--------------------------------------------------------"

# Install PIGNet2 in editable mode
pip install -e .

echo ""
echo "Step 4: Verifying installation..."
echo "----------------------------------"

# Verify all imports work
python -c "
import torch
import torch_geometric
import torch_scatter
import torch_sparse
print('✓ PyTorch:', torch.__version__)
print('✓ PyTorch Geometric:', torch_geometric.__version__)
print('✓ CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('✓ CUDA version:', torch.version.cuda)
    print('✓ GPU:', torch.cuda.get_device_name(0))

# Test PIGNet2 imports
from src.models import PIGNet
from src.core import PIGNetCore
print('✓ PIGNet2 modules imported successfully')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================"
    echo "✓ Installation completed successfully!"
    echo "============================================"
    echo ""
    echo "You can now use PIGNet2 with:"
    echo "  - Python API: from src.models import PIGNet"
    echo "  - CLI: pignet --help"
    echo ""
    echo "For optional dependencies:"
    echo "  pip install -e .[analysis]  # For analysis features"
    echo "  pip install -e .[agents]    # For LLM integration"
    echo "  pip install -e .[dev]       # For development tools"
else
    echo ""
    echo "❌ Installation verification failed!"
    echo "Please check the error messages above."
    exit 1
fi
