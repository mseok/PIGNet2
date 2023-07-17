#!/usr/bin/env python
import argparse
import glob
import os
from functools import partial

import numpy as np
from scipy.stats import kendalltau, linregress
from sklearn.metrics import r2_score

HERE = os.path.dirname(os.path.realpath(__file__))


def run(protein_id, lines):
    lines = [l for l in lines if protein_id in l[0]]
    id_to_pred = {l[0]: float(l[2]) for l in lines}

    # read experimental data
    exp_filename = f"{HERE}/derivative_data/{protein_id}/data.txt"
    with open(exp_filename) as f:
        lines = f.readlines()
        lines = [l.split() for l in lines]
        id_to_exp = {l[0]: float(l[1]) for l in lines}

    ligand_ids = sorted(id_to_exp.keys())
    pignet_predictions, experiments, pignet_mols, rmsds = [], [], [], []
    for ligand_id in ligand_ids:
        pignet_predictions.append(id_to_pred[f"{protein_id}_{ligand_id}"])
        experiments.append(id_to_exp[ligand_id])
    return (pignet_predictions, experiments, ligand_ids)


def calculate_tau(true, pred):
    assert len(true) == len(pred)
    sorted_true, sorted_pred = zip(*sorted(zip(true, pred)))
    idx = [i + 1 for i in range(len(true))]
    _, pred_idx = zip(*sorted(zip(sorted_pred, idx)))
    tau, _ = kendalltau(idx, pred_idx)
    return tau


def main(args: argparse.Namespace) -> None:
    files = glob.glob(args.file_prefix + "*")
    try:
        if "txt" in files[0]:
            files = sorted(
                files, key=lambda file: int(file.split("_")[-1].split(".")[0])
            )
        else:
            files = sorted(files, key=lambda file: int(file.split("_")[-1]))
    except Exception:
        pass

    print(
        "File",
        "R (mean)",
        "R (median)",
        "tau (mean)",
        "tau (median)",
        "MAE (mean)",
        "MAE (median)",
        sep="\t",
    )

    for file in files:
        with open(file, "r") as f:
            lines = f.readlines()
        lines = [line.split() for line in lines]
        protein_ids = sorted(list(set([l[0].split("_")[0] for l in lines])))
        _run = partial(run, lines=lines)
        # post processing
        result = dict()
        for idx, protein_id in enumerate(protein_ids):
            pignet, experiments_kcal, ligand_ids = _run(protein_id)
            pignet_ic50 = [e / -1.36 for e in pignet]
            r2 = r2_score(experiments_kcal, pignet)
            _, _, r, _, _ = linregress(pignet, experiments_kcal)
            mae = np.abs(np.array(experiments_kcal) - np.array(pignet)).mean()
            tau = calculate_tau(experiments_kcal, pignet)
            result[protein_id] = (r2, r, tau, mae, pignet, experiments_kcal)

        sorted_protein_ids = sorted(
            list(result.keys()), key=lambda x: result[x][1], reverse=True
        )

        rs = [result[pdb_id][1] for pdb_id in sorted_protein_ids]
        taus = [result[pdb_id][2] for pdb_id in sorted_protein_ids]
        maes = [result[pdb_id][3] for pdb_id in sorted_protein_ids]
        num_derivatives = [len(result[pdb_id][4]) for pdb_id in sorted_protein_ids]

        print(
            f"{file}",
            f"{np.average(rs):.2f}",
            f"{np.median(rs):.2f}",
            f"{np.average(taus):.2f}",
            f"{np.median(taus):.2f}",
            f"{np.average(maes):.2f}",
            f"{np.median(maes):.2f}",
            sep="\t",
        )
        if args.full_result:
            for protein_id in sorted_protein_ids:
                print(
                    f"\t{protein_id+',':<15}",  # pdb id
                    f"{result[protein_id][1]:.2f},",  # R
                    f"{result[protein_id][2]:.2f},",  # Kendaltau
                    f"{result[protein_id][3]:.1f},",  # MAE
                    f"{len(result[protein_id][4])} derivatives,",  # number of derivatives
                    f"{(max(result[protein_id][5])-min(result[protein_id][5])):.1f} kcal/mol",  # energy diff
                    sep="\t",
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-f",
        "--file_prefix",
        type=str,
        default="result_derivative_benchmark2_",
        help=" ",
    )
    parser.add_argument("-v", "--full_result", action="store_true")

    args = parser.parse_args()

    main(args)
