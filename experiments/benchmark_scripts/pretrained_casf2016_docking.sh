#!/bin/bash

ROOT_DIR=$(git rev-parse --show-toplevel)
DATA_DIR=${ROOT_DIR}/dataset/Benchmark/docking
EXE_DIR=${ROOT_DIR}/src

SEED=0
export CUDA_VISIBLE_DEVICES=$((0+${SEED}))

# Using pre-trained model
BENCHMARK_DIR=${ROOT_DIR}/experiments/pretrained
for SEED in {0..3};
do
  python -u ${EXE_DIR}/exe/test.py \
    hydra.run.dir=${BENCHMARK_DIR} \
    run.ngpu=1 \
    run.batch_size=400 \
    run.checkpoint_file=${EXE_DIR}/ckpt/pda_${SEED}.pt \
    run.log_file=${BENCHMARK_DIR}/benchmark/output_docking_${SEED}.log \
    data=messi/casf2016_docking \
    data.casf2016_docking.root_data_dir=${DATA_DIR} \
    data.casf2016_docking.key_dir=${EXE_DIR}/keys/casf2016/docking \
    data.casf2016_docking.test_result_path=${BENCHMARK_DIR}/benchmark/result_docking_${SEED}.txt \
    run.num_workers=4
done
