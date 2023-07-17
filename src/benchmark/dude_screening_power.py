#!/usr/bin/env python
import argparse
import glob
from collections import defaultdict
from statistics import mean
from typing import List

import numpy as np
from scipy import stats


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
        print("File", "Top0.5%", "Top1%", "Top5%", "Top10%", "CI_l", "CI_u", sep="\t")
    for file in files:
        ntb_top = []
        ntb_total = []
        with open(file, "r") as f:
            lines = f.readlines()
            lines = [line.split() for line in lines]
        id_to_pred = defaultdict(dict)
        for line in lines:
            key = line[0]
            gene = key.split("_")[0]
            id_to_pred[gene][key] = float(line[2])

        genes = sorted(list(id_to_pred.keys()))
        for gene in genes:
            selected_keys = sorted(list(id_to_pred[gene].keys()))
            preds = [id_to_pred[gene][key] for key in selected_keys]
            preds, selected_keys = zip(*sorted(zip(preds, selected_keys)))
            if gene in ("ampc",):
                # In case of AMPC target, the active compounds just have numeric names
                # rather than "CHEMBL*".
                true_binders = [k for k in selected_keys if "_C" not in k]
            else:
                true_binders = [k for k in selected_keys if "CHEMBL" in k]
            ntb_top_pdb, ntb_total_pdb = [], []
            for topn in [0.005, 0.01, 0.05, 0.1]:
                n = int(topn * len(selected_keys))
                top_keys = selected_keys[:n]
                n_top_true_binder = len(list(set(top_keys) & set(true_binders)))
                ntb_top_pdb.append(n_top_true_binder)
                ntb_total_pdb.append(len(true_binders) * topn)
            ntb_top.append(ntb_top_pdb)
            ntb_total.append(ntb_total_pdb)

        if args.full_result:
            print(file, end="\t")
            for i in range(len(ntb_top[0])):
                ef = []
                for j in range(len(ntb_total)):
                    if ntb_total[j][i] == 0:
                        continue
                    ef.append(ntb_top[j][i] / ntb_total[j][i])
                if i == 0:
                    confidence_interval = bootstrap_confidence(ef, args.n_bootstrap)
                print(round(mean(ef), 3), end="\t")
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
        "-f", "--file_prefix", type=str, default="result_dude_", help=" "
    )
    parser.add_argument("-n", "--n_bootstrap", type=int, default=100, help=" ")
    args = parser.parse_args()
    main(args)
