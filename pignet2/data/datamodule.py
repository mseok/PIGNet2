import random
from collections import defaultdict
from typing import Dict, List

import pandas as pd
import torch
from omegaconf import DictConfig
from torch_geometric import seed_everything
from torch_geometric.loader import DataLoader

from .data import ComplexDataset
from .utils import read_keys, read_labels, read_metadata


class ComplexDataModule:
    def __init__(self, config: DictConfig):
        self.config = config
        self.batch_size = config.run.batch_size
        self.num_workers = config.run.num_workers
        self.conv_range = config.model.conv_range
        self.pin_memory = config.run.pin_memory

        # Reproducibility
        self.seed = None
        if getattr(config.run, "seed") is not None:
            self.seed = config.run.seed
            self.generator = torch.Generator()
            self.generator.manual_seed(self.seed)

        # Should be removed when Lightning is used.
        self.prepare_data()
        # Should be removed when Lightning is used.
        self.setup()

    # Lightning method
    def prepare_data(self) -> None:
        # Prepare training & test keys.
        # Lightning intends to split data in `setup`,
        # but we do here because our splits are fixed.
        self.train_keys = defaultdict(list)
        self.test_keys = defaultdict(list)
        self.id_to_y = defaultdict(dict)
        self.selected_keys = defaultdict(list)
        self.metadata: Dict[str, pd.DataFrame] = dict()

        for task in self.tasks:
            train_keys, test_keys = read_keys(self.config.data[task].key_dir)
            self.train_keys[task] = train_keys
            self.test_keys[task] = test_keys

            if self.config.data[task].processed_data_dir is None:
                self.id_to_y[task] = read_labels(self.config.data[task].label_file)

            if metadata_file := self.config.data[task].get("metadata_file"):
                self.metadata[task] = read_metadata(metadata_file)

        self.filter_keys()

    def filter_keys(self) -> None:
        """Filter `train_keys` and `test_keys` in-place.

        Filtering list:
            1. RMSD
                filter out keys s.t. RMSD < `{train,test}_min_rmsd`.
        """
        for task in self.tasks:
            data_config = self.config.data[task]
            metadata = self.metadata.get(task)

            # RMSD filter
            if (min_rmsd := data_config.get("train_min_rmsd")) is not None:
                index = metadata.query(f"`rmsd` >= {min_rmsd}").index
                self.train_keys[task] = index.intersection(
                    self.train_keys[task]
                ).tolist()
            if (min_rmsd := data_config.get("test_min_rmsd")) is not None:
                index = metadata.query(f"`rmsd` >= {min_rmsd}").index
                self.test_keys[task] = index.intersection(self.test_keys[task]).tolist()

    # Lightning method
    def setup(self) -> None:
        self.train_datasets = dict()
        self.test_datasets = dict()

        for task in self.tasks:
            # Setting 'processed_data_dir' takes priority than 'data_dir'.
            if self.config.data[task].processed_data_dir is not None:
                train_dataset = ComplexDataset(
                    keys=self.train_keys[task],
                    processed_data_dir=self.config.data[task].processed_data_dir,
                )
                test_dataset = ComplexDataset(
                    keys=self.test_keys[task],
                    processed_data_dir=self.config.data[task].processed_data_dir,
                )

            elif self.config.data[task].data_dir is not None:
                train_dataset = ComplexDataset(
                    keys=self.train_keys[task],
                    data_dir=self.config.data[task].data_dir,
                    id_to_y=self.id_to_y[task],
                    conv_range=self.conv_range,
                    pos_noise_std=getattr(self.config.data[task], "pos_noise_std", 0.0),
                    pos_noise_max=getattr(self.config.data[task], "pos_noise_max", 0.0),
                )
                test_dataset = ComplexDataset(
                    keys=self.test_keys[task],
                    data_dir=self.config.data[task].data_dir,
                    id_to_y=self.id_to_y[task],
                    conv_range=self.conv_range,
                )

            self.train_datasets[task] = train_dataset
            self.test_datasets[task] = test_dataset

    @property
    def tasks(self) -> List[str]:
        return list(self.config.data)

    @property
    def num_features(self) -> int:
        try:
            sample = next(iter(self.train_datasets.values()))[0]
        except (IndexError, StopIteration):
            sample = next(iter(self.test_datasets.values()))[0]
        return sample.x.size(-1)

    @property
    def size(self) -> Dict[str, List[int]]:
        size_dict = dict()
        for task in self.tasks:
            len_train = len(self.train_datasets[task])
            len_test = len(self.test_datasets[task])
            size_dict[task] = [len_train, len_test]
        return size_dict

    # Lightning method
    def train_dataloader(self) -> Dict[str, DataLoader]:
        return {
            task: DataLoader(
                self.train_datasets[task],
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=True,
                pin_memory=self.pin_memory,
                worker_init_fn=seed_everything if self.seed is not None else None,
                generator=self.generator if self.seed is not None else None,
            )
            for task in self.tasks
        }

    # Lightning method
    def val_dataloader(self) -> Dict[str, DataLoader]:
        return {
            task: DataLoader(
                self.test_datasets[task],
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=True,
                pin_memory=self.pin_memory,
                worker_init_fn=seed_everything if self.seed is not None else None,
                generator=self.generator if self.seed is not None else None,
            )
            for task in self.tasks
        }

    # Lightning method
    def test_dataloader(self) -> DataLoader:
        """Used for test and prediction.
        Assume only one dataset is used when this method is called.
        """
        try:
            (dataset,) = self.test_datasets.values()
        except ValueError:
            raise RuntimeError("Test run allows only one dataset!")
        else:
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=True,
                pin_memory=self.pin_memory,
                worker_init_fn=seed_everything if self.seed is not None else None,
                generator=self.generator if self.seed is not None else None,
            )

    def sample_keys(self) -> None:
        for task in self.tasks:
            n_samples = getattr(self.config.data[task], "n_samples", 0)
            if n_samples:
                pdb_to_files = defaultdict(list)
                for key in self.train_keys[task]:
                    if "_" in key:
                        pdb = key.split("_")[0]
                        pdb_to_files[pdb].append(key)

                # keys = list(pdb_to_files.keys())
                keys = []
                for key, files in pdb_to_files.items():
                    remained_files = set(files) - set(self.selected_keys[key])
                    if len(remained_files) > n_samples:
                        sampled_keys = random.sample(remained_files, n_samples)
                    else:
                        sampled_keys = remained_files
                        self.remove_selected_keys(key)
                    keys += sampled_keys
                    self.selected_keys[key] += sampled_keys

                self.train_datasets[task].keys = keys

                # test
                pdb_to_files = defaultdict(list)
                for key in self.test_keys[task]:
                    if "_" in key:
                        pdb = key.split("_")[0]
                        pdb_to_files[pdb].append(key)

                # keys = list(pdb_to_files.keys())
                keys = []
                for key, files in pdb_to_files.items():
                    remained_files = set(files) - set(self.selected_keys[key])
                    if len(remained_files) > n_samples:
                        sampled_keys = random.sample(remained_files, n_samples)
                    else:
                        sampled_keys = remained_files
                        self.remove_selected_keys(key)
                    keys += sampled_keys
                    self.selected_keys[key] += sampled_keys

                self.test_datasets[task].keys = keys
        return

    def remove_selected_keys(self, key: str) -> None:
        self.selected_keys[key] = []
        return

    def approximate_size(self, task: str) -> List[int]:
        train_counts = defaultdict(int)
        for key in self.train_datasets[task].keys:
            train_counts[key.split("_")[0]] += 1
        test_counts = defaultdict(int)
        for key in self.test_datasets[task].keys:
            test_counts[key.split("_")[0]] += 1
        return sum(train_counts.values()), sum(test_counts.values())
