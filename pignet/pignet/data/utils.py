import os
import pickle
from typing import Dict, List, Optional, Tuple

import pandas as pd


def _check_file(base: str, exts: Tuple[str, ...] = (".pkl", ".txt")) -> Optional[str]:
    """Check if \'<base>.pkl\' or \'<base>.txt\' exists.
    Return `None` if neither exists.
    Used in `read_keys`.
    """
    for ext in exts:
        path = base + ext
        if os.path.exists(path):
            return path


def _read_lines(path: str) -> List[str]:
    """Get stripped lines from a .pkl or .txt file.
    Used in `read_keys`.
    """
    # Pickle file
    try:
        with open(path, "rb") as f:
            out = pickle.load(f)
    # Text file
    except pickle.UnpicklingError:
        with open(path) as f:
            out = [line.strip() for line in f]
    return out


def read_keys(key_dir: str) -> Tuple[List[str], List[str]]:
    """Read training keys and test keys from a dir.

    Returns:
        train_keys: List[str] | []
            Empty list if no 'train_keys.{pkl,txt}' exists under `key_dir`.
        train_keys: List[str] | []
            Empty list if no 'test_keys.{pkl,txt}' exists under `key_dir`.
    """
    # Check the key file paths.
    train_key_path = _check_file(os.path.join(key_dir, "train_keys"))
    if train_key_path is None:
        train_keys = []
    else:
        train_keys = _read_lines(train_key_path)

    test_key_path = _check_file(os.path.join(key_dir, "test_keys"))
    if test_key_path is None:
        test_keys = []
    else:
        test_keys = _read_lines(test_key_path)

    return train_keys, test_keys


def read_labels(path) -> Dict[str, float]:
    """Read a key-to-label dict."""
    with open(path) as f:
        lines = (line.strip().split() for line in f)
        id_to_y = {row[0]: float(row[1]) for row in lines}
    return id_to_y


def read_metadata(path) -> pd.DataFrame:
    return pd.read_csv(path, index_col="id")
