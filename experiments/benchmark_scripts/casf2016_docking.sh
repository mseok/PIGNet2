#!/bin/bash

EXP=only_nda  # Experiment name to benchmark
SEED=0  # Select a seed of the experiment
EPOCH=3  # Select an epoch of the experiment

ROOT_DIR=$(git rev-parse --show-toplevel)
DATA_DIR=${ROOT_DIR}/dataset/Benchmark/docking
EXE_DIR=${ROOT_DIR}/src
BENCHMARK_DIR=${ROOT_DIR}/experiments/outputs/${EXP}/${SEED}

export CUDA_VISIBLE_DEVICES=$((0+${SEED}))

date
python -u ${EXE_DIR}/exe/test.py \
  hydra.run.dir=${BENCHMARK_DIR} \
  run.ngpu=1 \
  run.batch_size=400 \
  run.checkpoint_file=ckpt/save_${EPOCH}.pt \
  run.log_file=${BENCHMARK_DIR}/benchmark/output_docking_${EPOCH}.log \
  data=messi/casf2016_docking \
  data.casf2016_docking.root_data_dir=${DATA_DIR} \
  data.casf2016_docking.key_dir=${EXE_DIR}/keys/casf2016/docking \
  data.casf2016_docking.test_result_path=${BENCHMARK_DIR}/benchmark/result_docking_${EPOCH}.txt \
  run.num_workers=4
date
