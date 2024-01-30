#!/bin/bash

ls data > test_keys.txt
sed "s/$/\t0.0/g" test_keys.txt > pdb2affinity.txt
tar -cf data.tar data test_keys.txt pdb2affinity.txt
