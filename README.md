# PIGNet2: A versatile deep learning-based protein-ligand interaction prediction model for accurate binding affinity scoring and virtual screening
This repository is the official implementation of [PIGNet2: A versatile deep learning-based protein-ligand interaction prediction model for accurate binding affinity scoring and virtual screening](https://arxiv.org/abs/2307.01066).

## Installation

### Quick Install with CUDA Support (Recommended)
Clone the repository and use the installation script for automatic CUDA configuration:
```bash
git clone https://github.com/mseok/PIGNet2.git
cd PIGNet2

# For CUDA 12.x (auto-detects CUDA version)
./install_cuda.sh

# For specific CUDA version
./install_cuda.sh --cuda 12.1

# For CPU-only installation
./install_cuda.sh --cpu
```

The script will:
1. Detect your CUDA version (or use the specified one)
2. Install PyTorch with appropriate CUDA support
3. Install PyTorch Geometric packages with matching CUDA versions
4. Install PIGNet2 and all other dependencies
5. Verify the installation

### Manual Installation with CUDA 12
If you prefer manual installation with CUDA 12 support:
```bash
# Install PyTorch with CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install PyTorch Geometric packages
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cu121.html
pip install torch-geometric

# Install PIGNet2
pip install -e .
```

### Alternative: Manual Installation
If you prefer to set up the environment manually:

#### Environment Setup
You can use `conda` or `venv` for environment setting.
For the case of using `conda`, create the environment named `pignet2` as following:
```console
conda create -n pignet2 python=3.9
conda activate pignet2
conda install rdkit=2022.03.4 openbabel pymol-open-source -c conda-forge
```

#### Install Dependencies
```console
pip install -r requirements.txt
```

### Optional Dependencies
For additional features, install extra packages:
```bash
# For advanced analysis features
pip install -e ".[analysis]"

# For LLM agent integration
pip install -e ".[agents]"

# For development tools
pip install -e ".[dev]"

# Install everything
pip install -e ".[analysis,agents,dev]"
```

### Verify Installation
After installation, verify that PIGNet2 is properly installed:
```bash
# Check if the CLI is available
pignet --help

# Test Python import
python -c "from src.models import PIGNet; print('PIGNet2 installed successfully!')"
```

## Data

Donwload our source data into `dataset` directory in this repository.
By executing `dataset/download.sh`, you can download all the following datasets.
> training dataset
- PDBbind v2020 scoring
- PDBbind v2020 docking
- PDBbind v2020 cross
- PDBbind v2020 random
> benchmark dataset
- CASF-2016 socring
- CASF-2016 docking
- CASF-2016 screening
- DUD-E screening
- derivative benchmark

Then, you can extract the downloaded files by executing `dataset/untar.sh`.

## Training
Training scripts can be found in `experiments/training_scripts` directory.
We provide 4 scripts for training.
- `baseline.sh`: training without any data augmentation
- `only_nda.sh`: training only with negative data augmentation
- `only_pda.sh`: training only with positive data augmentation
- `pda_nda.sh`: training with both positive and negative data augmentation

If you execute the script, the result files will be generated in your **current working directory**.
By default, we recommend you to execute training scripts at `experiemnts` directory.
All the result files are placed in `outputs/${EXPERIMENT_NAME}` directory.

## Benchmark
Benchmark scripts can be found in `experiments/benchmark_scripts` directory.
We provide 5 scripts for benchmark.
- `casf2016_scoring.sh`: benchmark on CASF-2016 scoring benchmark
- `casf2016_docking.sh`: benchmark on CASF-2016 docking benchmark
- `casf2016_screening.sh`: benchmark on CASF-2016 screening benchmark
- `dude.sh`: benchmark on DUD-E benchmark
- `derivative.sh`: benchmark on derivative benchmark (2015)

After training, you have to set the `${BENCHMARK_DIR}` in each benchmark scripts, which is set as `experiments/outputs/${EXPERIMENT_NAME}` as default.
Since `experiments/outputs` is set as a root directory of each experiment, it is highly recommended to place the `outputs` directory inside `experiments` directory.
For using our pre-trained model for benchmark, please refer to the [next section](#pre-trained-models).

After that, you will get the benchmark result files in `experiments/outputs/${EXPERIMENT_NAME}/benchmark`.
To benchmark each result files, you can execute `src/benchmark/*.py`.
For example, you can perform DUD-E benchmark by the following command.
```console
src/benchmark/dude_screening_power.py -f experiments/outputs/${EXPERIMENT_NAME}/benchmark/result_dude_${EPOCH}.txt -v
```

## Pre-trained Models
You can find the pre-trained models in `src/ckpt`.
We provide PIGNet2 models trained with both positive and negative data augemntation, which is the best model.
You can execute the `experiments/benchmark/pretrained_*.sh` scripts to get the benchmark results of pre-trained models.
The scripts will generate result files in `experiments/pretrained`.

# Using PIGNet2

## New Modular Python API (Recommended)
PIGNet2 now provides a clean, modular API for easy integration into your projects:

```python
from src.models import PIGNet, PIGNetMorse
from src.core import PIGNetCore
from src.utils import quick_predict, batch_predict_from_dataframe
from src.analysis import generate_detailed_explanation

# Quick prediction with ensemble
affinity = quick_predict("protein.pdb", "ligand.sdf")
print(f"Predicted affinity: {affinity:.2f} kcal/mol")

# Advanced usage with PIGNetCore
core = PIGNetCore(
    model_paths=["src/ckpt/pda_0.pt"],
    device="cuda",
    ensemble=False
)
result = core.predict_complex(
    "protein.pdb", 
    "ligand.sdf",
    track_residues=True,
    detailed_interactions=True
)

# Batch processing from DataFrame
import pandas as pd
df = pd.DataFrame({
    'protein_path': ['prot1.pdb', 'prot2.pdb'],
    'ligand_path': ['lig1.sdf', 'lig2.sdf']
})
results = batch_predict_from_dataframe(df, "src/ckpt/pda_0.pt")

# Generate detailed analysis
explanation = generate_detailed_explanation(
    core, "protein.pdb", "ligand.sdf",
    energy_breakdown=True,
    fragment_analysis=True
)
```

## Command Line Interface (CLI)
PIGNet2 now includes a user-friendly CLI powered by Click:

```bash
# Quick prediction with ensemble of all 4 models
pignet quick -p protein.pdb -l ligand.sdf

# Single model prediction
pignet predict src/ckpt/pda_0.pt -p protein.pdb -l ligand.sdf

# Batch prediction from CSV
pignet batch compounds.csv model.pt -o results.csv

# Detailed explanation with analysis
pignet explain model.pt -p protein.pdb -l ligand.sdf --energy-breakdown --fragment-analysis

# Training a new model
pignet train --config-name baseline --epochs 300

# Testing a model
pignet test model.pt --data-dir test_data/
```

## Legacy Command Line Usage
> [!NOTE]  
> We highly recommend to use SMINA-optimized ligand conformations and doing 4-model ensemble to get accurate results.

Prepare protein pdb file and ligand sdf.
Execute the following command to generate the result in `$OUTPUT` path (the output path is `predict.txt` by default):
```console
python src/exe/predict.py ./src/ckpt/pda_0.pt -p $PROTEIN -l $LIGAND -o $OUTPUT
```
By default, each element of result are named as `$(basename $PROTEIN .pdb)_$(basename $LIGAND .sdf)_${idx}`, where `${idx}` is an index of ligand conformation.

## Case 1: a single pdb and a single sdf with one conformation
```console
python src/exe/predict.py ./src/ckpt/pda_0.pt -p examples/protein.pdb -l examples/ligand_single_conformation.sdf -o examples/case1.txt
```

## Case 2: a single pdb and a single sdf with multiple conformations
`src/exe/predict.py` automatically enumerates all conformations in ligand sdf.
```console
python src/exe/predict.py ./src/ckpt/pda_0.pt -p examples/protein.pdb -l examples/ligand1.sdf -o examples/case2.txt
```

## Case 3: a single pdb and multiple sdfs with multiple conformations
`src/exe/predict.py` automatically make protein-ligand pairs for a single pdb and all ligand sdfs.
```console
python src/exe/predict.py ./src/ckpt/pda_0.pt -p examples/protein.pdb -l examples/ligand1.sdf examples/ligand2.sdf -o examples/case3.txt
```

## Case 4: multiple pdbs and multiple sdfs with multiple conformations
In this case, you should match the order of ligand and protein files and all of them sequentially.
For example, if you have `protein1-ligand1`, `protein1-ligand2`, `protein2-ligand3`, you should do like following:
```console
python src/exe/predict.py ./src/ckpt/pda_0.pt -p protein1.pdb protein1.pdb protein2.pdb -l ligand1.sdf ligand2.sdf ligand3.sdf
```

# Explanation about the results

The result file is a tab-separated file with the following columns:

```
protein_ligand_single_conformation_0    0.000   -3.990  -2.074  -1.021  0.000   -0.894  0.000
```

Each of the numeric columns corresponds to:

- True label (which is just set to 0.000 in inference)
- Total predicted binding affinity (= sum of the right-hand values)
- van der Waals energy
- hydrogen bond energy
- metal-ligand coordination energy
- hydrophobic energy
- dummy variable (please ignore this column)
