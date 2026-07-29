#!/bin/bash

set -e

WORK_DIR="$(pwd)"

DEFNAME="aurora_SBND2026A_gen2_BNBLight_DevSample_prodgenie_corsika_proton_rockbox0p1_sbnd_CV_v10_14_02_03_flatcaf_sbnd"

OUTPUT_DIR="$WORK_DIR"
OUTPUT_FILE="${OUTPUT_DIR}/flatcafs.txt"

mkdir -p "$OUTPUT_DIR"

echo "Fetching file locations for definition:"
echo "  $DEFNAME"

samweb -e sbnd list-file-locations \
    --defname="$DEFNAME" \
    | awk -F'\t' '{
        dir=$1
        file=$2

        gsub("dcache:", "", dir)
        gsub("enstore:", "", dir)
        gsub(/ +$/, "", dir)
        gsub(/^ +/, "", file)

        print dir "/" file
    }' \
    > "$OUTPUT_FILE"

echo "Done: $(wc -l < "$OUTPUT_FILE") files written to:"
echo "  $OUTPUT_FILE"
