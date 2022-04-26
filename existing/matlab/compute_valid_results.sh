#!/bin/bash

set -eu -o pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

if [[ $# -ne 5 ]]
then
    echo "Wrong number of arguments!" >&2
    echo "Usage: $0 <algorithm> <data_type> <input1_path> <input2_path> <output_path>" >&2
    exit 1
fi

ALG="$1"
DATA_TYPE="$2"
IN1="$(realpath -e "$3")"
IN2="$(realpath -e "$4")"
OUT="$(realpath "$5")"

matlab -nodisplay -nosplash -nodesktop -batch "alg = '${ALG}'; data_type = '${DATA_TYPE}'; iterations = 1; adaptive_limit = 0; in1_path = '${IN1}'; in2_path = '${IN2}'; out_path = '${OUT}'; run('${DIR}/compute_valid_results.m'); exit;"
