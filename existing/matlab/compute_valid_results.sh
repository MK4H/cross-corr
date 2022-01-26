#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

if [[ $# -ne 5 ]]
then
    echo "Wrong number of arguments!" >&2
    echo "Usage: $0 <algorithm> <data_type> <input1_path> <input2_path> <output_path>" >&2
    exit 1
fi


matlab -nodisplay -nosplash -nodesktop -r "alg = '$1'; data_type = '$2'; in1_path = '$3'; in2_path = '$4'; out_path = '$5'; run('${DIR}/cross_corr.m'); exit;"
