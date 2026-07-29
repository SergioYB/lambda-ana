#!/bin/bash

export WORK_DIR="$(pwd)"
CAFPYANA_DIR="/home/syebes/cafpyana"

echo "Working directoy exported: ${WORK_DIR}"
echo "Cafpyana directory: ${CAFPYANA_DIR}"

echo "Sourcing cafpyana setup"
cd "$CAFPYANA_DIR" || exit 1

echo "Currently at: ${"$(pwd)"}"

if [ "$PWD" != "$WORK_DIR" ]; then
    source setup.sh
fi



cd "$WORK_DIR" || exit 1

echo ""
echo "cafpyana correctly setup"