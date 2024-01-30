#!/bin/bash
PDBBIND_V2020=$1

smina_local_opt() {
  pdb=$(basename $(dirname $1))
  target=${PDBBIND_V2020}/${pdb}/${pdb}_protein.pdb
  ligand=${PDBBIND_V2020}/${pdb}/${pdb}_ligand.sdf
  new_file=$(dirname $1)/local_opt.sdf
  if [ -f ${new_file} ]; then
    exit
  fi
  smina \
    -r $target \
    -l $1 \
    -o $new_file \
    --local_only \
    --minimize \
    -q
}

export -f smina_local_opt

smina_local_opt $2
