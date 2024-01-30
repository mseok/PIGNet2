#!/bin/bash
PDBBIND_V2020=$1

get_rmsd() {
  dir=$(dirname $1)
  pdb=$(basename $dir)
  filename=${dir}/${pdb}_.mol2
  filename_all=${dir}/${pdb}_*.mol2
  obabel $1 -O $filename -m -xu 2>/dev/null
  for file in $(/bin/ls -v $filename_all);
  do
    echo -en "$file\t"
    DockRMSD $file ${PDBBIND_V2020}/${pdb}/${pdb}_ligand.mol2 \
      | grep "Calculated Docking RMSD" \
      | awk '{print $4}'
done > ${dir}/rmsd.txt
  rm $filename_all
}


export -f get_rmsd

get_rmsd $2
