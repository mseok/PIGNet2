#!/usr/bin/env python
import os
from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem, Mol, rdDistGeom, rdMolAlign


def get_etkdg_parameters(
    ps: AllChem.EmbedParameters,
    seed: int = 2022,
    use_random_coords: bool = True,
    num_threads: int = 1,
    prune_rms_thresh: float = 1,
    box_size_mult: int = 2,
    clear_confs: bool = False,
    enforce_chirality: bool = True,
) -> AllChem.EmbedParameters:
    ps.randomSeed = seed
    ps.useRandomCoords = use_random_coords
    ps.numThreads = num_threads
    ps.pruneRmsThresh = prune_rms_thresh
    ps.boxSizeMult = box_size_mult
    ps.clearConfs = clear_confs
    ps.enforceChirality = enforce_chirality
    return ps


def generate_conformers(mol: Mol, ps: AllChem.EmbedParameters, n_confs: int = 1000):
    confids = AllChem.EmbedMultipleConfs(mol, n_confs, ps)
    return mol, confids


def opt_conformers(mol: Mol, ff_type: str):
    if ff_type.lower() == "uff":
        results = AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=2000)
    elif ff_type.lower() == "mmff":
        results = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=2000)

    results = sorted(results, key=lambda result: result[1])
    confids = [idx for idx, result in enumerate(results) if not result[0]]
    return mol, confids


def write_mols(mol: Mol, confids: List[int], file: Path):
    if len(confids) > 1:
        file.parent.mkdir(parents=True, exist_ok=True)
        rdMolAlign.AlignMolConformers(mol)
        writer = Chem.SDWriter(str(file))
        for confid in confids:
            writer.write(mol, confId=confid)
        writer.close()
    return


def main():
    ligand = Chem.SDMolSupplier(
        str(args.path / args.key / f"{args.key}_ligand.sdf"), removeHs=False
    )[0]
    ps = AllChem.srETKDGv3()
    ps = get_etkdg_parameters(
        ps,
        clear_confs=False,
        num_threads=2,
        prune_rms_thresh=0.25,
    )
    ligand, confids = generate_conformers(ligand, ps)

    if args.uff:
        uff_ligand = deepcopy(ligand)
        path = Path("uff") / args.key / f"{args.key}.sdf"
        uff_ligand, uff_confids = opt_conformers(uff_ligand, "uff")
        write_mols(uff_ligand, uff_confids[1:], path)

    if args.mmff:
        mmff_ligand = deepcopy(ligand)
        path = Path("mmff") / args.key / f"{args.key}.sdf"
        mmff_ligand, mmff_confids = opt_conformers(mmff_ligand, "mmff")
        write_mols(mmff_ligand, mmff_confids[1:], path)

    path = Path("etkdg") / args.key / f"{args.key}.sdf"
    write_mols(ligand, confids, path)
    return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, help="pdbbind data path")
    parser.add_argument("key", type=str, help="pdb id")
    parser.add_argument("--uff", action="store_true")
    parser.add_argument("--mmff", action="store_true")
    args, _ = parser.parse_known_args()

    main()
