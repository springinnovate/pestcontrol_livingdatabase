#!/bin/bash

lower_bound=$1
upper_bound=$2
step_size=$3

input_csv1="./base_data/BioControlDatabaseMetadata.csv"
input_csv2="./base_data/abundance_template_rbound.csv"

for ((i=lower_bound; i<upper_bound; i+=step_size)); do
    next_bound=$((i + step_size))
    echo python3 ingest_table.py "$input_csv1" "$input_csv2" --n_dataset_rows "$i" "$next_bound"
done
