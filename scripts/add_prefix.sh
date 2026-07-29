#!/bin/bash

# Uso: ./add_prefix.sh input.txt output.txt

if [ $# -ne 2 ]; then
    echo "Uso: $0 input.txt output.txt"
    exit 1
fi

input_file="$1"
output_file="$2"
prefix="root://fndca1.fnal.gov:1094"

# Evitar sobrescribir sin querer
> "$output_file"

while IFS= read -r line; do
    # Evitar líneas vacías opcionalmente (quita este if si no quieres)
    if [ -n "$line" ]; then
        echo "${prefix}${line}" >> "$output_file"
    else
        echo "" >> "$output_file"
    fi
done < "$input_file"

echo "Listo. Output guardado en $output_file"