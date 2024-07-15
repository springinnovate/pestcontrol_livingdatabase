#!/bin/bash

lower_bound=0
upper_bound=750007
step_size=10000

input_csv1="./base_data/BioControlDatabaseMetadata.csv"
input_csv2="./base_data/abundance_template_rbound.csv"

current_start=$lower_bound

while [ $current_start -lt $upper_bound ]; do
    current_end=$((current_start + step_size))
    if [ $current_end -gt $upper_bound ]; then
        current_end=$upper_bound
    fi

    echo python3 ingest_table.py "$input_csv1" "$input_csv2" --n_dataset_rows "$current_start" "$current_end"

    current_start=$((current_end + 1))
done
