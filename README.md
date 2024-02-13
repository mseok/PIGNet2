# PIGNet2: A versatile deep learning-based protein-ligand interaction prediction model for accurate binding affinity scoring and virtual screening
This repository is the official implementation of [PIGNet2: A versatile deep learning-based protein-ligand interaction prediction model for accurate binding affinity scoring and virtual screening](https://arxiv.org/abs/2307.01066).

## Installation
You can download this repository by `git clone https://github.com/mseok/PIGNet2.git`. Then, you can proceed with the following steps.

## Requirements
### Environment Setup
You can use `conda` or `venv` for environment setting.
For the case of using `conda`, create the environment named `pignet2` as following.
```console
conda create -n pignet2 python=3.9
conda activate pignet2
```

### Install Dependencies
```console
pip install -r requirements.txt
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
