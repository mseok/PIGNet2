#!/usr/bin/env python
import argparse
import glob
import os
import tarfile
from statistics import mean
from typing import List

import numpy as np
from scipy import stats

# Archive of *_rmsd.dat files.
RMSD_DATA = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "decoys_docking.tar"
)


def bootstrap_confidence(
    values: List[float], n: int = 10000, confidence: float = 0.9
) -> np.ndarray:
    metrics = []
    for _ in range(n):
        indice = np.random.randint(0, len(values), len(values))
        sampled = [values[i] for i in indice]
        metrics.append(sum(sampled) / len(sampled))
    metrics = np.array(metrics)
    return stats.t.interval(
        confidence, len(metrics) - 1, loc=np.mean(metrics), scale=np.std(metrics)
    )


def main(args: argparse.Namespace) -> None:
    id_to_rmsd = dict()
    with tarfile.open(RMSD_DATA) as tar:
        for member in tar.getmembers():
            with tar.extractfile(member) as f:
                lines = f.readlines()[1:]
                # Extracting tar strangely gives bytes, so `decode` is needed.
                lines = [line.decode().split() for line in lines]
                lines = [[line[0], float(line[1])] for line in lines]
                dic = dict(lines)
                id_to_rmsd.update(dic)

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
    if args.full_result:
        print("File", "Top1", "Top2", "Top3", "CI_l", "CI_u", sep="\t")
    for file in files:
        with open(file, "r") as f:
            lines = f.readlines()
            lines = [line.split() for line in lines]
            id_to_pred = {line[0]: float(line[2]) for line in lines}

        pdbs = sorted(list(set(key.split()[0].split("_")[0] for key in id_to_pred)))
        topn_successed_pdbs = []
        for pdb in pdbs:
            selected_keys = [key for key in id_to_pred if pdb in key]
            pred = [id_to_pred[key] for key in selected_keys]
            pred, sorted_keys = zip(*sorted(zip(pred, selected_keys)))
            rmsd = [id_to_rmsd[key] for key in sorted_keys]
            topn_successed = []
            for topn in [1, 2, 3]:
                if min(rmsd[:topn]) < 2.0:
                    topn_successed.append(1)
                else:
                    topn_successed.append(0)
            topn_successed_pdbs.append(topn_successed)

        if args.full_result:
            print(file, end="\t")
            for topn in [1, 2, 3]:
                successed = [success[topn - 1] for success in topn_successed_pdbs]
                print(round(mean(successed), 3), end="\t")
            top1_success = [success[0] for success in topn_successed_pdbs]
            confidence_interval = bootstrap_confidence(top1_success, args.n_bootstrap)
            print(round(confidence_interval[0], 3), end="\t")
            print(round(confidence_interval[1], 3))
        else:
            for topn in [1]:
                successed = [success[topn - 1] for success in topn_successed_pdbs]
                print(round(mean(successed), 3))
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-v", "--full_result", action="store_true")
    parser.add_argument(
        "-f", "--file_prefix", type=str, default="result_docking_", help=" "
    )
    parser.add_argument("-n", "--n_bootstrap", type=int, default=100, help=" ")
    args = parser.parse_args()

    main(args)
