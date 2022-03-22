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
else
    OUT_PATH=""
fi

matlab -nodisplay -nosplash -nodesktop -batch "alg = '${ALG}'; data_type = '${DATA_TYPE}'; iterations = ${ITERATIONS}; in1_path = '${IN1}'; in2_path = '${IN2}'; out_path = '${OUT_PATH}'; timings_path = '${TIMES}'; run('${DIR}/cross_corr.m'); exit;"
