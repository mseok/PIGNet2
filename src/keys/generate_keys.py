#!/usr/bin/env python
import argparse
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scoring_key_dir", type=Path)
    parser.add_argument("--key_dir", type=Path)
    parser.add_argument("--data_list", type=Path)
    args = parser.parse_args()

    with (args.scoring_key_dir / "test_keys.txt").open("r") as f:
        test_keys = [line.strip() for line in f]

    with args.data_list.open("r") as f:
        data = [line.strip() for line in f]

    train_file = args.key_dir / "train_keys.txt"
    test_file = args.key_dir / "test_keys.txt"
    with train_file.open("w") as train_f, test_file.open("w") as test_f:
        for d in data:
            if d.split("_")[0] in test_keys:
                test_f.write(d + "\n")
            else:
                train_f.write(d + "\n")
