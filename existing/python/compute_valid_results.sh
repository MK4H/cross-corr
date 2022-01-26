#!/bin/bash

set -eu -o pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

if [[ $# -ne 5 ]]
then
    echo "Wrong number of arguments!" >&2
    echo "Usage: $0 <algorithm> <data_type> <input1_path> <input2_path> <output_path>" >&2
    exit 1
fi


WORK_DIR="${PWD}"

cd "${DIR}"

poetry run bash -c "cd ${WORK_DIR} && python3 \"${DIR}/crosscorr.py\" -o \"$5\" -d \"$2\" \"$1\" \"$3\" \"$4\""
