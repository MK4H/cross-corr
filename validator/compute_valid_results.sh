#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

if [[ $# -ne 3 ]]
then
    echo "Wrong number of arguments!" >&2
    echo "Usage: $0 <input1_path> <input2_path> <output_path>" >&2
    exit 1
fi

matlab -nodisplay -nosplash -nodesktop -r "in1_path = '$1'; in2_path = '$2'; out_path = '$3'; run('${DIR}/compute_valid.m'); exit;"
