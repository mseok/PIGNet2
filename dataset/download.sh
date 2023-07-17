#!/bin/bash

download() {
  file=${1}
  url="https://zenodo.org/record/8091220/files/${file}?download=1"
  if [ ! -f tarfiles/${file} ]; then
    wget ${url} -O tarfiles/${file}
  fi
}

mkdir -p tarfiles

# Training
download PDBbind-v2020_scoring.tar.xz
download PDBbind-v2020_docking.tar.xz
download PDBbind-v2020_cross.tar.xz
download PDBbind-v2020_random.tar.xz
download PDBbind-v2020_pda.tar.xz

# Benchmark
download CASF-2016_scoring.tar.xz
download CASF-2016_docking.tar.xz
download CASF-2016_screening.tar.xz
download DUD-E.tar.xz
download derivative.tar.xz
