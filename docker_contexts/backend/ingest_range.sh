#!/bin/bash

lower_bound=0
upper_bound=506186
step_size=10000



input_csv1="./base_data/filled_template_metadata_RAIF.csv"
input_csv2="./base_data/RAIF_dataset.csv"

for ((i=lower_bound; i<upper_bound; i+=step_size)); do
    next_bound=$((i + step_size))
    python3 ingest_table.py "$input_csv1" "$input_csv2" --n_dataset_rows "$i" "$next_bound"
done
python3 ingest_table.py "$input_csv1" "$input_csv2" --n_dataset_rows "$i" $((upper_bound+1))
