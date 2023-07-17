#!/usr/bin/env python
import argparse
import glob
import os
from statistics import mean
from typing import Dict, List

import numpy as np
from scipy import stats

TARGET_FILE = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "TargetInfo.dat"
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


def choose_best_pose(id_to_pred: Dict[str, float]) -> Dict[str, float]:
    pairs = ["_".join(k.split("_")[:-1]) for k in id_to_pred.keys()]
    pairs = sorted(list(set(pairs)))
    retval = {p: [] for p in pairs}
    for key in id_to_pred.keys():
        pair = "_".join(key.split("_")[:-1])
        retval[pair].append(id_to_pred[key])
    for key in retval.keys():
        retval[key] = min(retval[key])
    return retval


def main(args: argparse.Namespace) -> None:
    true_binder_dict: Dict[str, List[str]] = dict()
    with open(TARGET_FILE, "r") as f:
        for line in f:
            if line.lstrip().startswith("#"):
                continue
            target, *binders = line.split()
            true_binder_dict[target] = binders[:5]
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
        print("File", "EF:Top1%", "5%", "10%", "CI_l", "CI_u", sep="\t", end="\t")
        print("SR:Top1%", "Top5%", "Top10%", "CI_l", "CI_u", sep="\t")
    for file in files:
        ntb_top = []
        ntb_total = []
        high_affinity_success = []
        with open(file, "r") as f:
            lines = f.readlines()
            lines = [line.split() for line in lines]
        id_to_pred = {line[0]: float(line[2]) for line in lines}
        id_to_pred = choose_best_pose(id_to_pred)

        pdbs = sorted(list(set([key.split("_")[0] for key in id_to_pred.keys()])))
        for pdb in pdbs:
            selected_keys = [
                key for key in id_to_pred.keys() if key.split("_")[0] == pdb
            ]
            preds = [id_to_pred[key] for key in selected_keys]
            preds, selected_keys = zip(*sorted(zip(preds, selected_keys)))
            true_binders = [
                key
                for key in selected_keys
                if key.split("_")[1] in true_binder_dict[pdb]
            ]
            ntb_top_pdb, ntb_total_pdb, high_affinity_success_pdb = [], [], []
            for topn in [0.01, 0.05, 0.1]:
                n = int(topn * len(selected_keys))
                top_keys = selected_keys[:n]
                n_top_true_binder = len(list(set(top_keys) & set(true_binders)))
                ntb_top_pdb.append(n_top_true_binder)
                ntb_total_pdb.append(len(true_binders) * topn)
                # Check if the best binder was found.
                if f"{pdb}_{true_binder_dict[pdb][0]}" in top_keys:
                    high_affinity_success_pdb.append(1)
                else:
                    high_affinity_success_pdb.append(0)
            ntb_top.append(ntb_top_pdb)
            ntb_total.append(ntb_total_pdb)
            high_affinity_success.append(high_affinity_success_pdb)

        if args.full_result:
            print(file, end="\t")
            for i in range(3):
                ef = []
                for j in range(len(ntb_total)):
                    if ntb_total[j][i] == 0:
                        continue
                    ef.append(ntb_top[j][i] / ntb_total[j][i])
                if i == 0:
                    confidence_interval = bootstrap_confidence(ef, args.n_bootstrap)
                print(round(mean(ef), 3), end="\t")
            print(round(confidence_interval[0], 3), end="\t")
            print(round(confidence_interval[1], 3), end="\t")
            for i in range(3):
                success = []
                for j in range(len(ntb_total)):
                    if high_affinity_success[j][i] > 0:
                        success.append(1)
                    else:
                        success.append(0)
                if i == 0:
                    confidence_interval = bootstrap_confidence(
                        success, args.n_bootstrap
                    )
                print(round(mean(success), 3), end="\t")
            print(round(confidence_interval[0], 3), end="\t")
            print(round(confidence_interval[1], 3))
        else:
            for i in range(1):
                ef = []
                for j in range(len(ntb_total)):
                    if ntb_total[j][i] == 0:
                        continue
                    ef.append(ntb_top[j][i] / ntb_total[j][i])
                print(round(mean(ef), 3))
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-v", "--full_result", action="store_true")
    parser.add_argument(
        "-f", "--file_prefix", type=str, default="result_screening_", help=" "
    )
    parser.add_argument("-n", "--n_bootstrap", type=int, default=100, help=" ")
    args = parser.parse_args()

    main(args)
