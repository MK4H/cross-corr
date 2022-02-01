#!/bin/bash

set -eu -o pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

if [[ $# -ne 7 ]]
then
    echo "Wrong number of arguments!" >&2
    echo "Usage: $0 <algorithm> <data_type> <iterations> <input1_path> <input2_path> <output_path> <timings_path>" >&2
    exit 1
fi

ITERATIONS="$3"

IN1="$(realpath -e "$4")"
IN2="$(realpath -e "$5")"
OUT="$(realpath "$6")"
TIMES="$(realpath "$7")"

matlab -nodisplay -nosplash -nodesktop -batch "alg = '$1'; data_type = '$2'; iterations = $ITERATIONS; in1_path = '${IN1}'; in2_path = '${IN2}'; out_dir = '${OUT}'; timings_path = '${TIMES}'; run('${DIR}/benchmark_cached.m'); exit;"
