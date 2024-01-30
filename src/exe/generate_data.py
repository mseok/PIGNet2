import argparse
import os
import sys
import inspect
import pickle
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from rdkit import Chem
from pymol import cmd
from rdkit.Chem import AllChem

PathLike = Union[str, os.PathLike]

MAX_NUM_RETRIALS = 20


def extract(
    receptor_pdb: Path,
    ligand_sdf: Path,
    ligand_idx: int = 1,
    distance: float = 5.0,
    tmp_ext: str = ".sdf",
    correct_oxygens: bool = False,
    unbond_metals: bool = True,
    remove_h_in_tmp_save: bool = True,
    remove_h_in_final_save: bool = True,
    ligand_sdf_is_large: bool = False,
    retry_with_obabel: bool = True,
    retry_with_cutoff: bool = True,
    connect_cutoff: Optional[float] = None,
    tmp_dir: Optional[Union[str, Path]] = None,
    silent: bool = False,
) -> Optional[AllChem.Mol]:
    cmd.reinitialize()
    cmd.set("max_threads", 1)
    if connect_cutoff is not None:
        cmd.set("connect_cutoff", connect_cutoff)

    # Load and clean the protein.
    cmd.load(receptor_pdb, "prot")
    if unbond_metals:
        cmd.unbond("metals", "*")
    if remove_h_in_tmp_save:
        cmd.remove("h.")

    # Load the ligand as name 'lig'.
    if ligand_sdf_is_large:
        tmp_sdf = extract_ith_mol(ligand_sdf, ligand_idx)
        cmd.load(tmp_sdf, "lig")
        os.remove(tmp_sdf)
    else:
        cmd.load(ligand_sdf, "lig_all")
        cmd.create("lig", f"%lig_all and state {ligand_idx}")
        cmd.delete("%lig_all")

    # Extract the pocket.
    cmd.create("pocket", f"br. (%prot and not h.) w. {distance} of (%lig and not h.)")
    cmd.delete("%lig")
    if not cmd.count_atoms("%pocket"):
        return

    # Clean the pocket.
    if correct_oxygens:
        _neutralize_pi_oxygens()
        if cmd.count_atoms("het"):
            _rebond_monovalent_oxygens()

    # Transfer the pocket into `AllChem.Mol`.
    tmp_ext = "." + tmp_ext.lstrip(".")
    fd, tmp_path = tempfile.mkstemp(suffix=tmp_ext, dir=tmp_dir)
    os.close(fd)
    cmd.save(tmp_path, "%pocket")
    (pocket,) = read_mols(tmp_path, removeHs=remove_h_in_final_save)

    # If failed, retry after obabel re-save.
    if pocket is None and retry_with_obabel:
        if not silent:
            print(
                f"{receptor_pdb}, {ligand_sdf}:",
                "Retrying with obabel",
                file=sys.stderr,
            )
        from openbabel import pybel

        fd, tmp_path2 = tempfile.mkstemp(suffix=tmp_ext, dir=tmp_dir)
        os.close(fd)
        pybel_mol = next(pybel.readfile(tmp_ext.lstrip("."), tmp_path))
        pybel_mol.write(tmp_ext.lstrip("."), tmp_path2, overwrite=True)

        (pocket,) = read_mols(tmp_path2, removeHs=remove_h_in_final_save)
        os.remove(tmp_path2)

    os.remove(tmp_path)

    # If failed, retry by reducing `connect_cutoff`.
    if pocket is None and retry_with_cutoff:
        frame = inspect.currentframe()
        _args, _varargs, _keywords, _locals = inspect.getargvalues(frame)
        args_to_pass = {arg: _locals[arg] for arg in _args}
        args_to_pass["retry_with_cutoff"] = False

        cutoffs = [0.35 - 0.01 * i for i in range(1, MAX_NUM_RETRIALS + 1)]
        for cutoff in cutoffs:
            if not silent:
                print(
                    f"{receptor_pdb}, {ligand_sdf}:",
                    f"Retrying with cutoff {cutoff}",
                    file=sys.stderr,
                )
            args_to_pass["connect_cutoff"] = cutoff
            pocket = extract(**args_to_pass)
            if pocket:
                break

    return pocket


def _rebond_monovalent_oxygens(selection: str = "het"):
    model: chempy.models.Indexed = cmd.get_model(selection)
    monovalent_oxygens: List[List[chempy.Atom]] = []

    # Get monovalent oxygens and their bonded atoms as pairs.
    for i, atom in enumerate(model.atom):
        if atom.symbol == "O":
            bonds: List[chempy.Bond] = [bond for bond in model.bond if i in bond.index]
            if len(bonds) == 1:
                a, b = bonds[0].index
                begin = a if i == a else b
                end = b if begin == a else a
                assert begin == i
                monovalent_oxygens.append([model.atom[begin], model.atom[end]])

    # Rebond oxygen bonds.
    for oxygen, other in monovalent_oxygens:
        cmd.unbond(f"id {oxygen.id}", f"id {other.id}")
        # [O]=X
        if oxygen.formal_charge == 0:
            bond_order = 2
        # [O-]-X
        elif oxygen.formal_charge == -1:
            bond_order = 1
        cmd.bond(f"id {oxygen.id}", f"id {other.id}", bond_order)


def _neutralize_pi_oxygens(selection: str = "polymer"):
    """Neutralize '=[O-]', which somtimes appears in ASP and GLU."""
    model: chempy.models.Indexed = cmd.get_model(selection)
    # Scan "=O" oxygens.
    for i, atom in enumerate(model.atom):
        if atom.symbol == "O":
            bonds: List[chempy.Bond] = [bond for bond in model.bond if i in bond.index]
            if len(bonds) == 1 and bonds[0].order == 2:
                # If not neutral.
                if atom.formal_charge:
                    cmd.alter(f"index {atom.index}", "formal_charge=0")


def extract_ith_mol(
    sdf: Path,
    idx: int,
    tmp_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Extract i-th record from an SDF file (i >= 1)."""
    fd, tmp_path = tempfile.mkstemp(suffix=sdf.suffix, dir=tmp_dir)
    os.close(fd)
    mol = AllChem.SDMolSupplier(str(sdf), removeHs=False, sanitize=False)[idx - 1]
    # Old RDKit versions don't support context manager for `SDWriter`.
    f = AllChem.SDWriter(tmp_path)
    f.write(mol)
    f.close()
    return Path(tmp_path)


def read_mols(
    file_path: Union[str, Path],
    sanitize: bool = True,
    removeHs: bool = True,
    rebond: bool = True,
) -> List[Optional[Chem.Mol]]:
    kwargs = {"sanitize": sanitize, "removeHs": removeHs}
    path = Path(file_path)

    if path.suffix in (".sdf", ".mol"):
        mols = Chem.SDMolSupplier(str(path), **kwargs)
    elif path.suffix == ".mol2":
        mols = mols_from_mol2_file(path, **kwargs)
    elif path.suffix == ".pdb":
        mols = mols_from_pdb_file(path, rebond=rebond, **kwargs)
    elif path.suffix == ".smi":
        mols = mols_from_smi_file(path, sanitize=sanitize)
    else:
        raise NotImplementedError

    return mols


def mols_from_mol2_file(
    file_path: Union[str, Path],
    sanitize: bool = True,
    removeHs: bool = True,
) -> List[Chem.Mol]:
    """Read molecules from a Mol2 file.

    A multi-mol version of `rdkit.Chem.MolFromMol2File`."""
    # For the .mol2 case, the delimiter line should be included as the beginning
    # of a block. So we read line-by-line unlike the .pdb case.
    delimiter = "@<TRIPOS>MOLECULE"
    blocks = []
    with Path(file_path).open() as f:
        for line in f:
            if line.startswith(delimiter):
                blocks.append(line)
            # Not meeting the first molecule yet.
            elif not blocks:
                continue
            else:
                blocks[-1] += line

    mols = [
        Chem.MolFromMol2Block(block, sanitize=sanitize, removeHs=removeHs)
        for block in blocks
        if block.strip()
    ]
    return mols


def mols_from_pdb_file(
    file_path: Union[str, Path],
    sanitize: bool = True,
    removeHs: bool = True,
    rebond: bool = True,
) -> List[Chem.Mol]:
    """Read molecules from a PDB file.

    A multi-model version of `rdkit.Chem.MolFromPDBFile`."""
    delimiter = "ENDMDL"
    with Path(file_path).open() as f:
        blocks = f.read().split(delimiter)

    mols = [
        Chem.MolFromPDBBlock(
            block, sanitize=sanitize, removeHs=removeHs, proximityBonding=rebond
        )
        for block in blocks
        if block.strip() and block.strip() != "END"
    ]
    return mols


def mols_from_smi_file(
    file_path: Union[str, Path],
    sanitize: bool = True,
) -> List[Chem.Mol]:
    """Read molecules from a SMILES file.

    Substitute for `rdkit.Chem.SmilesMolSupplier`."""
    mols = []
    with Path(file_path).open() as f:
        for line in f:
            try:
                # `maxsplit=1` allows names with whitespaces.
                smiles, name = line.split(maxsplit=1)
                name = name.strip()
            except ValueError:
                smiles = line.strip()
                name = None

            mol = Chem.MolFromSmiles(smiles, sanitize=sanitize)
            if name is not None:
                mol.SetProp("_Name", name)
            mols.append(mol)

    return mols


def extract_binding_pocket(
    ligand_mol: Chem.Mol,
    pdb_path: PathLike,
    distance: float = 5.0,
) -> Optional[Chem.Mol]:
    with tempfile.NamedTemporaryFile(suffix=".sdf") as ligand_file:
        writer = Chem.SDWriter(ligand_file.name)
        try:
            writer.write(ligand_mol)
        finally:
            writer.close()
        return extract(pdb_path, ligand_file.name, distance=distance)


def main(args: argparse.Namespace):
    mol_ligand = read_mols(args.ligand_file)[0]
    mol_target = extract_binding_pocket(mol_ligand, args.protein_file)

    with open(args.save_file_path / args.ligand_file.stem, "wb") as f:
        pickle.dump((mol_ligand, mol_target), f)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-p", "--protein_file", type=Path, help="protein .pdb file")
    parser.add_argument(
        "-l",
        "--ligand_file",
        type=Path,
        help="ligand files (.pdb | .mol2 | .sdf)",
    )
    parser.add_argument(
        "-s",
        "--save_file_path",
        type=Path,
        help="save files (.pt) directory",
        default="."
    )
    args = parser.parse_args()

    main(args)
