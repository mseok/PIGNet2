#!/bin/bash

ROOT_DIR=$(git rev-parse --show-toplevel)
DATA_DIR=${ROOT_DIR}/dataset/Benchmark/derivative
EXE_DIR=${ROOT_DIR}/src

# Using pre-trained model
BENCHMARK_DIR=${ROOT_DIR}/experiments/pretrained
for SEED in {0..3};
do
  python -u ${EXE_DIR}/exe/test.py \
    hydra.run.dir=${BENCHMARK_DIR} \
    run.ngpu=0 \
    run.batch_size=400 \
    run.checkpoint_file=${EXE_DIR}/ckpt/pda_${SEED}.pt \
    run.log_file=${BENCHMARK_DIR}/benchmark/output_der_${SEED}.log \
    data=messi/derivative \
    data.derivative.root_data_dir=${DATA_DIR} \
    data.derivative.key_dir=${EXE_DIR}/keys/derivative \
    data.derivative.test_result_path=${BENCHMARK_DIR}/benchmark/result_der_${SEED}.txt \
    run.num_workers=4
done
