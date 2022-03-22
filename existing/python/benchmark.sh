#!/bin/bash

set -eu -o pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

if [[ $# -lt 6 || $# -gt 7 ]]
then
    echo "Wrong number of arguments!" >&2
    echo "Usage: $0 <algorithm> <data_type> <iterations> <input1_path> <input2_path> <timings_path> [output_path]" >&2
    exit 1
fi

ALG="$1"
DATA_TYPE="$2"
ITERATIONS="$3"
IN1="$(realpath -e "$4")"
IN2="$(realpath -e "$5")"
TIMES="$(realpath "$6")"

if [[ $# -eq 7 ]]
then
    OUT_PATH="$(realpath "$7")"
    OUT_OPTION="-o \"${OUT_PATH}\""
else
    OUT_OPTION=""
fi

WORK_DIR="${PWD}"

cd "${DIR}"

poetry run bash -c "cd \"${WORK_DIR}\" && python3 \"${DIR}/crosscorr.py\" ${OUT_OPTION} -i \"${ITERATIONS}\" -d \"${DATA_TYPE}\" -t \"${TIMES}\" \"${ALG}\" \"${IN1}\" \"${IN2}\""
