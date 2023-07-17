#!/usr/bin/env python
import argparse
import os
import subprocess
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Dict, List

import numpy as np
from plot_benchmark_result import EXE


def get_benchmark_results(exp: Path) -> List[float]:
    prev_dir = os.getcwd()
    os.chdir(exp / "benchmark")

    exe = EXE[BENCHMARK]
    cmd = [exe, "-v"]
    if args.best:
        cmd.append("-f")
        cmd.append("best_result_screening_")
    results = subprocess.run(cmd, capture_output=True, text=True)
    results = results.stdout.split("\n")[:-1]
    results = [result.split()[:2] for result in results]
    results = [float(result[1]) for result in results]

    os.chdir(prev_dir)
    return results


def mp(
    inputs: List[str],
    nprocs: int = 4,
    func: Callable[[str], Any] = get_benchmark_results,
) -> Dict[str, Any]:
    pool = Pool(nprocs)
    results = pool.map_async(func, inputs)
    results.wait()
    pool.close()
    pool.join()
    data = results.get()
    return data


def process_files(files: List[Path]) -> Dict[str, List[float]]:
    *_, exp_type, exp_name, _, result_file = str(files[0]).split("/")
    result_file = (
        RESULT_PATH / exp_type / ".".join(exp_name.split(".")[:-1]) / result_file
    )
    if result_file.exists():
        return

    dic = defaultdict(list)
    for file in files:
        if not file.exists():
            print(file, "not exists")
            return
        with file.open("r") as f:
            for line in f:
                line = line.split()
                dic[line[0]].append(list(map(float, line[1:])))

    for key in dic.keys():
        avg = np.stack(dic[key])
        avg = np.sum(avg, 0) / avg.shape[0]
        dic[key] = [round(elem, 3) for elem in avg.tolist()]

    with result_file.open("w") as w:
        for key, values in dic.items():
            values = "\t".join(list(map(str, values)))
            w.write(f"{key}\t{values}\n")
    print(result_file, "done")
    return dic


def average_predictions(args: argparse.Namespace):
    prefix = "result" if not args.best else "best_result"
    if args.no_benchmark:
        files = [
            [exp / f"{prefix}_{BENCHMARK}_{epoch}.txt" for exp in args.exps]
            for epoch in args.epochs
        ]
    else:
        files = [
            [
                exp / "benchmark" / f"{prefix}_{BENCHMARK}_{epoch}.txt"
                for exp in args.exps
            ]
            for epoch in args.epochs
        ]
    if args.nprocs is not None and args.nprocs > 1:
        data = mp(files, args.nprocs, process_files)
    else:
        for file_list in files:
            process_files(file_list)
    return


def average_benchmarks(args: argparse.Namespace):
    if args.nprocs is not None and args.nprocs > 1:
        data = mp(args.exps, args.nprocs, get_benchmark_results)
        results = {}
        for exp, result in zip(args.exps, data):
            results[exp] = result
    else:
        results = {}
        for exp in args.exps:
            results[exp] = get_benchmark_results(exp, args.benchmark)
    print(results)
    print(list(set([len(values) for values in results.values()])))
    assert (
        len(list(set([len(values) for values in results.values()]))) == 1
    ), "Number of the benchmark results are not matched"
    results_per_epoch = list(zip(*results.values()))
    average_per_epoch = list(map(mean, results_per_epoch))
    average_per_epoch = [round(avg, 3) for avg in average_per_epoch]
    for avg in average_per_epoch:
        print(avg)
    return


def main(args: argparse.Namespace):
    (
        *_,
        exp_type,
        exp_name,
    ) = str(
        args.exps[0]
    ).split("/")
    result_path = RESULT_PATH / exp_type / ".".join(exp_name.split(".")[:-1])
    result_path.mkdir(parents=True, exist_ok=True)
    if args.pred:
        average_predictions(args)
    else:
        average_benchmarks(args)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "exps",
        nargs="+",
        type=Path,
        metavar="PATH",
        help="experiment dir or log file paths",
    )
    parser.add_argument("-b", "--benchmark", type=str, metavar="benchmark types")
    parser.add_argument("--best", action="store_true", help="screening best (fast)")
    parser.add_argument("-n", "--nprocs", type=int, help="multiprocessing cores (fast)")
    parser.add_argument("--no_benchmark", action="store_true")
    parser.add_argument(
        "-e",
        "--epochs",
        nargs="+",
        type=int,
        metavar="EPOCH",
        required=True,
        help="list of epochs",
    )
    parser.add_argument("-r", "--result_path", type=Path, default="results")
    parser.add_argument("-p", "--pred", action="store_true")
    args, unnamed = parser.parse_known_args()

    BENCHMARK = "scoring" if args.benchmark == "ranking" else args.benchmark
    RESULT_PATH = args.result_path
    RESULT_PATH.mkdir(parents=True, exist_ok=True)

    main(args)
