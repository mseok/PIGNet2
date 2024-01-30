#!/usr/bin/env python
import os
import subprocess
import sys
from pathlib import Path
from tempfile import mkstemp
from typing import Dict, List, Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem.rdMolAlign import CalcRMS


def get_idx2rmsd(rmsd_file: Path) -> Dict[int, float]:
    idx2rmsd = {}
    with rmsd_file.open("r") as f:
        for idx, line in enumerate(f):
            rmsd = line.split()[-1]
            idx2rmsd[idx] = float(rmsd)
    return idx2rmsd


def get_crystal_score(pdb: str, file: Path):
    with file.open("r") as f:
        for line in f:
            if line.startswith(pdb):
                break
    return float(line.split()[-1])


def preproecss_mol2(mol: Chem.Mol, pdb: str, idx: int) -> Optional[Path]:
    fd, path = mkstemp(suffix=".sdf", prefix=f"{pdb}_{idx}")
    os.close(fd)
    w = Chem.SDWriter(path)
    w.write(mol)
    w.close()

    try:
        fd, temp_path = mkstemp(suffix=".mol2", prefix=f"{pdb}_{idx}_")
        os.close(fd)
        subprocess.run(["obabel", path, "-O", temp_path, "-xu"])
        return Path(temp_path)
    except:
        os.unlink(path)
        return


def get_dockrmsd_rmsd(query_mol2: Path, template_mol2: Path) -> float:
    result = subprocess.run(
        ["DockRMSD", query_mol2, template_mol2, "-s"], capture_output=True
    )
    rmsd = float(result.stdout)
    return rmsd


def main():
    crystal_score = get_crystal_score(args.pdb, args.crystal_score_file)

    mols = []
    for conf_type in args.conf_types:
        mol_path = args.input_dir / conf_type / args.pdb / "local_opt.sdf"
        rmsd_path = args.input_dir / conf_type / args.pdb / "rmsd.txt"
        if not (mol_path.exists() and rmsd_path.exists()):
            continue
        idx2rmsd = get_idx2rmsd(rmsd_path)

        for idx, mol in enumerate(Chem.SDMolSupplier(str(mol_path))):
            rmsd = idx2rmsd[idx]
            if rmsd > 2:
                continue
            if abs(float(mol.GetProp("minimizedAffinity")) - crystal_score) > 1:
                continue
            mol.SetProp("crystal_rmsd", str(rmsd))
            mol.SetProp("conf_type", conf_type)
            mols.append(mol)

    if not mols:
        return

    sorted_mols = sorted(mols, key=lambda mol: float(mol.GetProp("crystal_rmsd")))
    mol2s = []
    selected_indice = [0]
    for idx, mol in enumerate(sorted_mols[1:]):
        if args.use_dockrmsd:
            query_mol2 = preproecss_mol2(mol, args.pdb, idx)
            if query_mol2 is None:
                continue
            mol2s.append(query_mol2)

            rmsds = []
            for selected_idx in selected_indice:
                pattern = f"{args.pdb}_{selected_idx}*"
                template_mol2 = [mol2 for mol2 in mol2s if mol2.match(pattern)][0]
                rmsd = get_dockrmsd_rmsd(query_mol2, template_mol2)
                rmsds.append(rmsd)
        else:
            rmsds = []
            for selected_idx in selected_indice:
                try:
                    rmsd = CalcRMS(mol, sorted_mols[selected_idx], maxMatches=1000)
                except:
                    rmsd = 1000
                rmsds.append(rmsd)

        if min(rmsds) > args.min_threshold:
            if args.max_threshold is None or max(rmsds) < args.max_threshold:
                selected_indice.append(idx)

        args.output_dir.mkdir(exist_ok=True, parents=True)
    out_path = args.output_dir / f"{args.pdb}.sdf"
    w = Chem.SDWriter(str(out_path))
    for idx, mol in enumerate(sorted_mols):
        if idx in selected_indice:
            w.write(mol)
    w.close()
    return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("pdb")
    parser.add_argument("--input_dir", type=Path, default=".")
    parser.add_argument("--output_dir", type=Path, default="FINAL")
    parser.add_argument(
        "--conf_types",
        choices=["etkdg", "uff", "mmff"],
        nargs="+",
        default=["etkdg", "uff", "mmff"],
    )
    parser.add_argument(
        "--crystal_score_file",
        type=Path,
        default="./score-pdbbind.txt",
    )
    parser.add_argument("--min_threshold", type=float, default=0.25)
    parser.add_argument("--max_threshold", type=float, default=2.0)
    parser.add_argument("--use_dockrmsd", action="store_true")
    args = parser.parse_args()

    main()
