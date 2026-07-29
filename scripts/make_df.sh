#!/bin/bash
set -e
trap 'echo "Error en la línea $LINENO"' ERR
INPUT_LIST="$WORK_DIR/scripts/ccqeLambdaCafFiles.txt"
OUTPUT="$WORK_DIR/data/df/ccqeLambda.df"
CONFIG="lambda_pre_config.py"
N_FILES="2"
N_JOBS="0"
CURR_DIR="$(pwd)"

if [ "$N_JOBS" -gt 0 ]; then
    OUTPUT="$(basename "$OUTPUT")"
else
    mkdir -p "$(dirname "$OUTPUT")"
fi

echo "Running df maker..."
cd "${CAFPYANA_WD}"
python "${CAFPYANA_DIR}/run_df_maker.py" \
    -c "${CAFPYANA_DIR}/configs/${CONFIG}" \
    -l "$INPUT_LIST" \
    -o "$OUTPUT" \
    -nfile "$N_FILES" \
    -ngrid "$N_JOBS"
echo "Done."
cd "${CURR_DIR}"