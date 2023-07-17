#!/bin/bash

EXP=  # Experiment name to benchmark
SEED=  # Select a seed of the experiment
EPOCH=  # Select an epoch of the experiment

ROOT_DIR=$(git rev-parse --show-toplevel)
DATA_DIR=${ROOT_DIR}/dataset/Benchmark/DUD-E
EXE_DIR=${ROOT_DIR}/src
BENCHMARK_DIR=${ROOT_DIR}/experiments/outputs/${EXP}/${SEED}

export CUDA_VISIBLE_DEVICES=$((0+${SEED}))

date
python -u ${EXE_DIR}/exe/test.py \
  hydra.run.dir=${BENCHMARK_DIR} \
  run.ngpu=1 \
  run.batch_size=300 \
  run.checkpoint_file=ckpt/save_${EPOCH}.pt \
  run.log_file=${BENCHMARK_DIR}/benchmark/output_dude_${EPOCH}.log \
  data=messi/dude \
  data.dude.root_data_dir=${DATA_DIR} \
  data.dude.key_dir=${EXE_DIR}/keys/DUD-E \
  data.dude.test_result_path=${BENCHMARK_DIR}/benchmark/result_dude_${EPOCH}.txt \
  run.num_workers=4
date
