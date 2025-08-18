# PIGNet2: A versatile deep learning-based protein-ligand interaction prediction model for accurate binding affinity scoring and virtual screening

This repository is the official implementation of [PIGNet2: A versatile deep learning-based protein-ligand interaction prediction model for accurate binding affinity scoring and virtual screening](https://arxiv.org/abs/2307.01066).

## Package Structure

PIGNet2 has been restructured into a modern Python package with the following architecture:

- **`pignet2/`** - Main package directory
  - **`models/`** - Neural network architectures (PIGNet, PIGNetMorse)
  - **`data/`** - Data processing and loading utilities
  - **`exe/`** - Execution scripts for training, testing, and prediction
  - **`analysis/`** - Interaction analysis and explanation tools
  - **`ckpt/`** - Pre-trained model checkpoints
  - **`config/`** - Configuration files for models and datasets
  - **`cli.py`** - Command-line interface
  - **`core.py`** - Direct model loading and prediction interface
  - **`utils.py`** - Utility functions

## Installation

### Default
```
pip install pignet2
```


### GPU-supported Installation
```
pip install pignet2 --extra-index-url https://download.pytorch.org/whl/cu124 --find-links https://data.pyg.org/whl/torch-2.6.0+cu124.html
```
```
```

### Alternative: Manual Installation
If you prefer to set up the environment manually:

#### Environment Setup
You can use `conda` or `venv` for environment setting.
For the case of using `conda`, create the environment named `pignet2` as following:
```console
conda create -n pignet2 python=3.10
conda activate pignet2
conda install rdkit openbabel pymol-open-source -c conda-forge
```

#### Install PIGNet2
```console
# Install in development mode (recommended for development)
pip install -e .

# Or install normally
pip install .
```
```
```

### Verify Installation
After installation, verify that PIGNet2 is properly installed:
```bash
# Check if the CLI is available
pignet --help

# Test Python import
python -c "from pignet2.models import PIGNet; print('PIGNet2 installed successfully!')"
```

## Data

Download our source data into `dataset` directory in this repository.
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
You can find the pre-trained models in `pignet2/ckpt/`:

- `pda_0.pt` - Model trained with positive data augmentation (fold 0)
- `pda_1.pt` - Model trained with positive data augmentation (fold 1)
- `pda_2.pt` - Model trained with positive data augmentation (fold 2)
- `pda_3.pt` - Model trained with positive data augmentation (fold 3)

We provide PIGNet2 models trained with both positive and negative data augmentation, which represent the best performing models.
You can execute the `experiments/benchmark/pretrained_*.sh` scripts to get the benchmark results of pre-trained models.
The scripts will generate result files in `experiments/pretrained`.

### Using Pre-trained Models

```python
from pignet2.core import PIGNetCore

# Single model
core = PIGNetCore("pignet2/ckpt/pda_0.pt")

# Ensemble of all 4 models (recommended)
core = PIGNetCore([
    "pignet2/ckpt/pda_0.pt",
    "pignet2/ckpt/pda_1.pt", 
    "pignet2/ckpt/pda_2.pt",
    "pignet2/ckpt/pda_3.pt"
], ensemble=True)
```

# Using PIGNet2

## New Modular Python API (Recommended)
PIGNet2 now provides a clean, modular API for easy integration into your projects:

```python
from pignet2.models import PIGNet, PIGNetMorse
from pignet2.core import PIGNetCore
from pignet2.analysis import generate_detailed_explanation

# Advanced usage with PIGNetCore
core = PIGNetCore(
    model_paths=["pignet2/ckpt/pda_0.pt"],
    device="cuda",
    ensemble=False
)
result = core.predict_complex(
    "protein.pdb", 
    "ligand.sdf",
    track_residues=True,
    detailed_interactions=True
)

# Ensemble prediction with all 4 models
core = PIGNetCore(
    model_paths=[
        "pignet2/ckpt/pda_0.pt",
        "pignet2/ckpt/pda_1.pt", 
        "pignet2/ckpt/pda_2.pt",
        "pignet2/ckpt/pda_3.pt"
    ],
    device="cuda",
    ensemble=True
)
result = core.predict_complex("protein.pdb", "ligand.sdf")
print(f"Predicted affinity: {result['affinity']:.2f} {result['affinity_unit']}")

# Batch processing
protein_ligand_pairs = [
    ('prot1.pdb', 'lig1.sdf'),
    ('prot2.pdb', 'lig2.sdf')
]
results = core.predict_batch(protein_ligand_pairs, track_residues=True)

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
pignet predict pignet2/ckpt/pda_0.pt -p protein.pdb -l ligand.sdf

# Batch prediction from CSV
pignet batch compounds.csv pignet2/ckpt/pda_0.pt -o results.csv

# Detailed explanation with analysis
pignet explain -c pignet2/ckpt/pda_0.pt -p protein.pdb -l ligand.sdf --energy-breakdown --fragment-analysis

# Training a new model
pignet train --config-name baseline --epochs 300

# Testing a model
pignet test pignet2/ckpt/pda_0.pt --data-dir test_data/
```

## Legacy Command Line Usage
> [!NOTE]
> We highly recommend to use SMINA-optimized ligand conformations and doing 4-model ensemble to get accurate results.

Prepare protein pdb file and ligand sdf.
Execute the following command to generate the result in `$OUTPUT` path (the output path is `predict.txt` by default):
```console
python pignet2/exe/predict.py ./pignet2/ckpt/pda_0.pt -p $PROTEIN -l $LIGAND -o $OUTPUT
```
By default, each element of result are named as `$(basename $PROTEIN .pdb)_$(basename $LIGAND .sdf)_${idx}`, where `${idx}` is an index of ligand conformation.

## Case 1: a single pdb and a single sdf with one conformation
```console
python pignet2/exe/predict.py ./pignet2/ckpt/pda_0.pt -p examples/protein.pdb -l examples/ligand_single_conformation.sdf -o examples/case1.txt
```

## Case 2: a single pdb and a single sdf with multiple conformations
`pignet2/exe/predict.py` automatically enumerates all conformations in ligand sdf.
```console
python pignet2/exe/predict.py ./pignet2/ckpt/pda_0.pt -p examples/protein.pdb -l examples/ligand1.sdf -o examples/case2.txt
```

## Case 3: a single pdb and multiple sdfs with multiple conformations
`pignet2/exe/predict.py` automatically make protein-ligand pairs for a single pdb and all ligand sdfs.
```console
python pignet2/exe/predict.py ./pignet2/ckpt/pda_0.pt -p examples/protein.pdb -l examples/ligand1.sdf examples/ligand2.sdf -o examples/case3.txt
```

## Case 4: multiple pdbs and multiple sdfs with multiple conformations
In this case, you should match the order of ligand and protein files and all of them sequentially.
For example, if you have `protein1-ligand1`, `protein1-ligand2`, `protein2-ligand3`, you should do like following:
```console
python pignet2/exe/predict.py ./pignet2/ckpt/pda_0.pt -p protein1.pdb protein1.pdb protein2.pdb -l ligand1.sdf ligand2.sdf ligand3.sdf
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
