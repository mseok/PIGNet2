import os
import shutil
import subprocess
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List, Optional

import dimorphite_dl
from pymol import cmd
from rdkit import Chem
from rdkit.Chem import AllChem

REDUCE_COMMAND = "reduce"


def protonate_mol(
    mol: Chem.Mol,
    min_ph: float = 6.4,
    max_ph: float = 8.4,
    pka_precision: float = 1.0,
    max_variants: int = 128,
) -> List[Chem.Mol]:
    """Protonate a molecule.

    Args:
        mol: An input `rdkit.Chem.Mol` object to protonate.
        min_ph: Dimorphite-DL parameter.
        max_ph: Dimorphite-DL parameter.
        pka_precision: Dimorphite-DL parameter.
        max_variants: Dimorphite-DL parameter.

    Returns:
        A list of protonated `rdkit.Chem.Mol` objects.
    """
    params = {
        "min_ph": min_ph,
        "max_ph": max_ph,
        "pka_precision": pka_precision,
        "max_variants": max_variants,
    }
    engine = dimorphite_dl.DimorphiteDL(**params)

    smi = Chem.MolToSmiles(mol)
    protonated_smi = engine.protonate(smi)[0]
    protonated_mol = AllChem.AssignBondOrdersFromTemplate(Chem.MolFromSmiles(protonated_smi), mol)

    return protonated_mol


def get_temp_path(path: Optional[Path] = None, **kwargs) -> Path:
    """Return a temporary path if `path` is None, or else return as-is.

    Args:
        kwargs: kwargs for `tempfile.NamedTemporaryFile`.
    """
    # Arg `delete` should not be overwritten.
    kwargs.pop("delete", None)
    if path is None:
        with NamedTemporaryFile(delete=False, **kwargs) as f:
            path = Path(f.name)
    return path


def protonate_ligand(
    mol: AllChem.Mol,
    min_ph: float = 7.4,
    max_ph: float = 7.4,
    pka_precision: float = 0.0,
    **kwargs
) -> Optional[AllChem.Mol]:
    """Protonate an rdkit Mol object using dimorphite-dl.

    Args:
        kwargs: kwargs for `dimorphite_dl.protonate_mol`.
    """
    protonated_mol = protonate_mol(
        mol, min_ph=min_ph, max_ph=max_ph, pka_precision=pka_precision, **kwargs
    )
    return protonated_mol


def protonate_pdb(
    pdb_input: Path,
    pdb_output: Optional[Path] = None,
    correct_hetatm: bool = True,
    resave_by_pymol: bool = True,
    silent: bool = True,
) -> Path:
    stderr = open(os.devnull, "w") if silent else None

    if correct_hetatm:
        pdb_tmp = get_temp_path(suffix="_HETATM.pdb")
        _correct_hetatm(pdb_input, pdb_tmp)
    else:
        pdb_tmp = get_temp_path(suffix=".pdb")
        shutil.copy(pdb_input, pdb_tmp)

    pdb_no_h = get_temp_path(suffix="_noH.pdb")
    with pdb_no_h.open("w") as f:
        subprocess.run(
            [REDUCE_COMMAND, "-Trim", pdb_tmp], stdout=f, text=True, stderr=stderr
        )

    pdb_output = get_temp_path(pdb_output, suffix="_reduce.pdb")
    with pdb_output.open("w") as f:
        subprocess.run(
            [REDUCE_COMMAND, "-BUILD", "-NUC", "-NOFLIP", pdb_no_h],
            stdout=f,
            text=True,
            stderr=stderr,
        )

    if resave_by_pymol:
        _resave_pdb(pdb_output)

    if silent:
        stderr.close()
    pdb_tmp.unlink()
    pdb_no_h.unlink()

    return pdb_output


def _correct_hetatm(pdb_input: Path, pdb_output: Optional[Path] = None) -> None:
    """Alter non-polymer atoms as HETATM and resave using PyMOL.

    Args:
        pdb_output: If not given, overwrite `pdb_input`.
    """
    if pdb_output is None:
        pdb_output = pdb_input
    cmd.reinitialize()
    cmd.set("max_threads", 1)
    cmd.set("retain_order", 1)
    cmd.load(pdb_input)
    cmd.alter("not polymer", "type='HETATM'")
    cmd.save(pdb_output)


def _resave_pdb(pdb_input: Path, pdb_output: Optional[Path] = None) -> None:
    """Resave a .pdb file using PyMOL. HETATMs are rebonded using distance.

    Args:
        pdb_output: If not given, overwrite `pdb_input`.
    """
    if pdb_output is None:
        pdb_output = pdb_input
    cmd.reinitialize()
    cmd.set("max_threads", 1)
    cmd.set("retain_order", 1)
    cmd.set("connect_mode", 3)
    cmd.load(pdb_input)
    cmd.save(pdb_output)



