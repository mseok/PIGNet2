from collections import defaultdict
from math import log

import numpy as np
from scipy.special import rel_entr
from scipy.stats import gaussian_kde


def read_dude(file: str):
    aff_dic = {}
    with open(file) as f:
        for line in f.readlines():
            key, _, pred, *_ = line.split()
            aff_dic[key] = float(pred)
    return aff_dic


def get_active_decoy(aff_dic: dict, threshold: float = -6.8):
    active = []
    decoy = []

    for key, value in aff_dic.items():
        target, ligand, idx = key.split("_")
        if "CHEMBL" in ligand or ligand.isdigit():
            active.append(value)
        else:
            decoy.append(value)

    return active, decoy


def get_active_decoy_per_target(aff_dic: dict, threshold: float = -6.8):
    actives = defaultdict(list)
    decoys = defaultdict(list)

    for key, value in aff_dic.items():
        target, ligand, idx = key.split("_")
        if "CHEMBL" in ligand or ligand.isdigit():
            actives[target].append(value)
        else:
            decoys[target].append(value)

    return actives, decoys


def get_min_max(*dics):
    max = -15.0
    min = 0.0
    for dic in dics:
        for key in dic.keys():
            if dic[key] > max:
                max = dic[key]
            if dic[key] < min:
                min = dic[key]
    return min, max


def kl_divergence(p, q):
    return sum(p[i] * log(p[i] / q[i]) for i in range(len(p)))


def get_kldiv(active, decoy, min, max, bins=100):
    active = np.array(active)
    decoy = np.array(decoy)

    x = np.linspace(min, max, bins)

    kde_active = gaussian_kde(active)
    kde_decoy = gaussian_kde(decoy)

    y_active = kde_active(x)
    y_decoy = kde_decoy(x)

    # nat -> ln
    kl = rel_entr(y_active, y_decoy)
    kl = np.sum(kl)
    kl = kl * (x[1] - x[0])

    return y_active, y_decoy, kl


def plot(x, y_actives, y_decoys):
    import matplotlib.pyplot as plt

    for key in y_actives:
        active_plot = plt.plot(x, y_actives[key], label=f"active-{key}")
        color = active_plot[0].get_color()
        plt.plot(x, y_decoys[key], label=f"decoy-{key}", linestyle="--", color=color)

    plt.legend()
    plt.xlabel("Regression Values")
    plt.ylabel("Density")
    # plt.show()
    plt.savefig("./figures/KL.pdf", dpi=300, bbox_inches="tight")
    return


def main():
    base = read_dude("../result-HP/baseline-2.1/result_dude_2100.txt")
    tnda = read_dude("../result-HP/tnda-2.1/result_dude_2100.txt")
    tpda = read_dude("../result-HP/dockrmsd-tpda-2.1/result_dude_44.txt")

    # get max and min of each dict
    MIN, MAX = get_min_max(base, tnda, tpda)
    MIN = max(MIN, -15.0)
    MAX = min(MAX, 0.0)

    active_base, decoy_base = get_active_decoy(base)
    active_tnda, decoy_tnda = get_active_decoy(tnda)
    active_tpda, decoy_tpda = get_active_decoy(tpda)

    base_active, base_decoy, kl_base = get_kldiv(active_base, decoy_base, MIN, MAX, 300)
    tnda_active, tnda_decoy, kl_tnda = get_kldiv(active_tnda, decoy_tnda, MIN, MAX, 300)
    tpda_active, tpda_decoy, kl_tpda = get_kldiv(active_tpda, decoy_tpda, MIN, MAX, 300)
    print(
        f"KL-Divergence: baseline {kl_base:.3f}, tnda {kl_tnda:.3f}, tpda {kl_tpda:.3f}"
    )
    return


if __name__ == "__main__":
    main()
