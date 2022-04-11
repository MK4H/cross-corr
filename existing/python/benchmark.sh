#!/bin/bash

set -eu -o pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

if [[ $# -lt 7 || $# -gt 8 ]]
then
    echo "Wrong number of arguments!" >&2
    echo "Usage: $0 <algorithm> <data_type> <iterations> <adaptive_limit> <input1_path> <input2_path> <timings_path> [output_path]" >&2
    exit 1
fi

ALG="$1"
DATA_TYPE="$2"
ITERATIONS="$3"
ADAPTIVE_LIMIT="$4"
IN1="$(realpath -e "$5")"
IN2="$(realpath -e "$6")"
TIMES="$(realpath "$7")"

if [[ $# -eq 8 ]]
then
    OUT_PATH="$(realpath "$8")"
    OUT_OPTION="-o \"${OUT_PATH}\""
else
    OUT_OPTION=""
fi

WORK_DIR="${PWD}"

cd "${DIR}"

poetry run bash -c "cd \"${WORK_DIR}\" && python3 \"${DIR}/crosscorr.py\" ${OUT_OPTION} -i \"${ITERATIONS}\" -d \"${DATA_TYPE}\" -t \"${TIMES}\" -l \"${ADAPTIVE_LIMIT}\" \"${ALG}\" \"${IN1}\" \"${IN2}\""
