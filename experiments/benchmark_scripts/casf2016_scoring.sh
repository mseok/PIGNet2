#!/bin/bash

EXP=  # Experiment name to benchmark
SEED=  # Select a seed of the experiment
EPOCH=  # Select an epoch of the experiment

ROOT_DIR=$(git rev-parse --show-toplevel)
DATA_DIR=${ROOT_DIR}/dataset/Benchmark/scoring
EXE_DIR=${ROOT_DIR}/src
BENCHMARK_DIR=${ROOT_DIR}/experiments/outputs/${EXP}/${SEED}

date
python -u ${EXE_DIR}/exe/test.py \
  hydra.run.dir=${BENCHMARK_DIR} \
  run.ngpu=0 \
  run.batch_size=400 \
  run.checkpoint_file=ckpt/save_${EPOCH}.pt \
  run.log_file=${BENCHMARK_DIR}/benchmark/output_scoring_${EPOCH}.log \
  data=messi/casf2016_scoring \
  data.casf2016_scoring.root_data_dir=${DATA_DIR} \
  data.casf2016_scoring.key_dir=${EXE_DIR}/keys/casf2016/scoring \
  data.casf2016_scoring.test_result_path=${BENCHMARK_DIR}/benchmark/result_scoring_${EPOCH}.txt \
  run.num_workers=4
date
