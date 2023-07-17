#!/bin/bash

untar() {
  file=${1}
  dir=$(basename ${file} .tar.xz | rev | cut -d_ -f1 | rev)

  if [[ "$file" == *"PDBbind-v2020"* ]]; then
    tar_dir=${PWD}/PDBbind-v2020/${dir}
  else
    tar_dir=${PWD}/Benchmark/${dir}
  fi

  mkdir -p ${tar_dir}
  tar -xf ${file} -C ${tar_dir}

  if [ ! -d ${tar_dir}/data ]; then
    if [ -f ${tar_dir}/data_5_sdf ]; then
      ln -s ${tar_dir}/data_5_sdf ${tar_dir}/data
    else
      ln -s ${tar_dir}/data_5 ${tar_dir}/data
    fi
  fi
}

mkdir -p PDBbind-v2020 Benchmark

untar tarfiles/PDBbind-v2020_scoring.tar.xz
untar tarfiles/PDBbind-v2020_docking.tar.xz
untar tarfiles/PDBbind-v2020_cross.tar.xz
untar tarfiles/PDBbind-v2020_random.tar.xz
untar tarfiles/PDBbind-v2020_pda.tar.xz

untar tarfiles/CASF-2016_scoring.tar.xz
untar tarfiles/CASF-2016_docking.tar.xz
untar tarfiles/CASF-2016_screening.tar.xz
untar tarfiles/DUD-E.tar.xz
untar tarfiles/derivative.tar.xz
