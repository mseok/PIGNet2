# Positive data augmentation processing codes

All codes in this directory works with a **single** PDB or file.
You need to install [Smina](https://sourceforge.net/projects/smina/) and [DockRMSD](https://zhanggroup.org/DockRMSD/).
Note that you should add PATH for both `Smina` and `DockRMSD`, to execute them directly.
If you want to do multiprocessing, you should install [GNU parallel](https://www.gnu.org/software/parallel/). (You can install with conda by `conda install -c conda-forge parallel`)

Here, we assume that the PDBbind data are downloaded at `$PDBBIND_DIR` directory.
So, ligand and protein files for certain `$PDB` are `$PDBBIND_DIR/$PDB/${PDB}_ligand.sdf` and `$PDBBIND_DIR/$PDB/${PDB}_protein.pdb`, respectively.

## Generating various conformations for a ligand.
Default conformer generation option is ETKDG (v3), but you can additionally generate UFF- and MMFF- optimized conformations via `--uff` and `--mmff` flags.
The files are generated with names like `{etkdg,uff,mmff}/${PDB}/${PDB}.sdf`.

```console
# generate ETKDG, UFF, MMFF conformations
./generate_conformers.py $PDBBIND_DIR $PDB --uff --mmff
```

To do multiprocessing, you can do the following:

```console
# get only PDBID, without README or INDEX
/bin/ls -d $PDBBIND_DIR/???? > PDBs.txt
parallel -j${NCPU} -a PDBs.txt "./generate_conformers.py $PDBBIND_DIR {} --uff --mmff"
```

## Minimizing generated conformations.
For all generated conformations, we also minimize them using Smina.
Here, an input is generated conformation file (.sdf). (e.g. `etkdg/1a30/1a30.sdf`)
The corresponding output is generated with names like `{etkdg,uff,mmff}/${PDB}/local_opt.sdf`.

```console
./local_opt.sh $PDBBIND_DIR $FILE
```

To do multiprocessing, you can do the following:

```console
# get generated files
echo {etkdg,uff,mmff}/????/????.sdf | xargs /bin/ls > generated_files.txt
parallel -j${NCPU} -a generated_files.txt "./local_opt.sh $PDBBIND_DIR {}"
```

## Calculate RMSD of generated conformations with crystal ligand structure.
For all generated conformations, we calculate RMSD with crystal ligand structure.
Here, an input is generated and minimized conformation file (.sdf). (e.g. `etkdg/1a30/local_opt.sdf`)
The corresponding output is generated with names like `{etkdg,uff,mmff}/${PDB}/rmsd.txt`.
You need `DockRMSD` in this process.

```console
./run_dockrmsd.sh $PDBBIND_DIR $FILE
```

To do multiprocessing, you can do the following:

```console
# get generated files
echo {etkdg,uff,mmff}/????/local_opt.sdf | xargs /bin/ls > minimized_files.txt
parallel -j${NCPU} -a minimized_files.txt "./run_dockrmsd.sh $PDBBIND_DIR {}"
```

## Filter with geometric and energetic criteria.
Filter effective conformations from all of generated ones.
The following code execute with default settings (min and max RMSD between poses: 0.25 and 2.0, energy criteria 1kcal/mol).
`score-pdbbind.txt` file is pre-computed smina scores for all crystal protein-ligand complexes in PDBBind v2020 refined set.

```console
./filter_valid.py $PDB
```

To do multiprocessing, you can do the following:

```console
parallel -j${NCPU} -a PDBs.txt "./filter_valid.py ${PDB}"
```
