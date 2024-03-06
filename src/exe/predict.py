# CUDA devices should be recognized first.
# isort: off
import set_cuda

# isort: on
import argparse
import time
from pathlib import Path
from typing import List, Tuple, Union
from warnings import warn

import pymol
import torch
import torch_geometric.data as pyg_data
from omegaconf import open_dict
from rdkit import Chem

# isort: off
import path
import utils
from generate_data import extract_binding_pocket, read_mols
from protonate import protonate_ligand, protonate_pdb
from data.data import complex_to_data
from utils import PathLike

PathOrMol = Union[PathLike, Chem.Mol]


def run(
    model: torch.nn.Module,
    data: pyg_data.Data,
    device: torch.device,
):
    model.eval()
    data = data.to(device)

    with torch.no_grad():
        model.predict_step(data)


def read_data(
    protein: Path,
    ligand: Path,
    conv_range: Tuple[float, float],
    protonate_sdf: bool = True,
    protonate_protein: bool = True,
):
    name_root = f"{protein.stem}_{ligand.stem}"

    # Prepare ligand Mol.
    mol_ligand = read_mols(ligand)
    if protonate_sdf:
        mol_ligand = [protonate_ligand(ligand) for ligand in mol_ligand]

    # Prepare protein Mol.
    if protonate_protein:
        protein = protonate_pdb(protein)

    # Prepare pocket
    data = []
    for idx, ligand in enumerate(mol_ligand):
        mol_target = extract_binding_pocket(ligand, protein)
        name = f"{name_root}_{idx}"
        # Return None if error occurred above.
        if ligand is None or mol_target is None:
            datum = None
        else:
            datum = complex_to_data(ligand, mol_target, key=name, conv_range=conv_range)

        data.append((datum, name))

    if protonate_protein:
        # print("Unlinking protonated protein: ", protein)
        protein.unlink()

    return data


def main(args: argparse.Namespace):
    # Check inputs.
    assert len(args.protein_files) == 1 or len(args.protein_files) == len(
        args.ligand_files
    )

    # Reuse the protein file if only one is given.
    if len(args.protein_files) == 1:
        args.protein_files *= len(args.ligand_files)

    print("Load from:", args.checkpoint_file.resolve())
    print("Writing results to:", args.output_path.resolve())

    # Set GPUs
    # os.environ['CUDA_VISIBLE_DEVICES'] = utils.cuda_visible_devices(config.ngpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", repr(device))

    # Load the model.
    checkpoint = torch.load(args.checkpoint_file, map_location=device)
    config = utils.merge_configs(checkpoint["config"], {})
    assert "data" not in config
    with open_dict(config):
        config.data = {"test": {"test_result_path": str(args.output_path)}}
    model = utils.initialize_state(device, checkpoint, config)[0]

    # Prepare the data.
    data_list = [
        read_data(
            protein_file,
            ligand_file,
            config.model.conv_range,
            not args.no_prot_sdf,
            not args.no_prot_pdb,
        )
        for protein_file, ligand_file in zip(args.protein_files, args.ligand_files)
    ]

    # Filter erroneous data.
    data_list, names = zip(*((datum for data in data_list for datum in data if datum[0] is not None)))
    data = pyg_data.Batch.from_data_list(data_list)

    # Prediction starts.
    start_time = time.time()
    model.reset_log()
    run(model, data, device)
    print("Time:", time.time() - start_time)

    utils.write_predictions(model, config, False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("checkpoint_file", type=Path, help="checkpoint file to load")
    parser.add_argument(
        "-p", "--protein_files", nargs="+", type=Path, help="protein .pdb files"
    )
    parser.add_argument(
        "-l",
        "--ligand_files",
        nargs="+",
        type=Path,
        help="ligand files (.pdb | .mol2 | .sdf)",
    )
    parser.add_argument(
        "--no-prot-pdb", action="store_true", help="don't protonate the input PDB"
    )
    parser.add_argument(
        "--no-prot-sdf", action="store_true", help="don't protonate the input SDF"
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=Path,
        default="predict.txt",
        help="text file to write results",
    )
    parser.add_argument("-x", "--leave-protein", action="store_true")
    args = parser.parse_args()

    main(args)
