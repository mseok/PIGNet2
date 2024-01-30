# Preprocessing codes

All codes in this directory works with a **single** PDB or file.
You need to install [pymol-open-source](https://github.com/schrodinger/pymol-open-source), [openbabel](https://open-babel.readthedocs.io/en/latest/Installation/install.html) for basic preprocessing.
Also, for protonation of ligand and protein, you should additionally install [dimorphite_dl](https://github.com/durrantlab/dimorphite_dl/) and [reduce](https://github.com/rlabduke/reduce), respectively.
The codes are largely contributed by [Sang-Yeon Hwang](https://scholar.google.co.kr/citations?user=HizmmwYAAAAJ&hl=ko).

`pymol-open-source`, `openbabel`, and `dimorphite_dl` can be installed through conda as following, but you should build `reduce` from source.

```console
conda install -c conda-forge pymol-open-source
conda install -c conda-forge openbabel
pip install dimorphite_dl
```

## Preprocessing
To preprocess data, you can execute `./generate_data.py` with protein and ligand files.
By default, this code will protonate both protein and ligand with `reduce` and `dimorphite_dl`, respectively.
You can turn off this features with `--no-prot-pdb` and `--no-prot-sdf` flags.

```console
./generate_data.py -p $protein_pdb_file -l $ligand_sdf_file
```
