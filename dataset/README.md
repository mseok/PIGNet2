# Data

Donwload our source data into this directory in this repository.
There are two files in this directory: `download.sh` and `untar.sh`.

- `download.sh` is a script for downloading all the datasets.
- `untar.sh` is a script for extracting all the downloaded files.

By executing `download.sh`, you can download all the following datasets in `tarfiles` directory.
> training dataset
- PDBbind v2020 scoring: preprocessed pdbbind v2020 refined set
- PDBbind v2020 docking: preprocessed re-docking augmented dataset
- PDBbind v2020 cross: preprocessed cross-docking augmented dataset
- PDBbind v2020 random: preprocessed random-docking augmented dataset
> benchmark dataset
- CASF-2016 socring: preprocessed CASF-2016 scoring benchmark dataset
- CASF-2016 docking: preprocessed CASF-2016 docking benchmark dataset
- CASF-2016 screening: preprocessed CASF-2016 screening benchmark dataset
- DUD-E screening: preprocessed DUD-E screening benchmark dataset (docked with smina, single conformation)
- derivative benchmark: preprocessed derivative benchmark 2015 dataset

After that, you can extract the downloaded files by executing `untar.sh`.
This will make two new dataset, PDBbind-v2020 and Benchmark, in this directory.

Also, you can generate PDA data with codes in `generate_PDA`
