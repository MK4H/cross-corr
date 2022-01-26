#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

if [[ $# -ne 4 && $# -ne 6 ]]
then
    echo "Wrong number of arguments!" >&2
    echo "Usage: $0 [-d|--data_type single|double] <technology> <algorithm> <input1_path> <input2_path> <output_path>" >&2
    exit 1
fi

if [[ $1 == "-d" || $1 == "--data_type" ]]
then
    DATA_TYPE="$2"
    shift 2

    if [[ "${DATA_TYPE}" != "single" && "${DATA_TYPE}" != "double" ]]
    then
        echo "Unkwnown data type ${DATA_TYPE}" >&2
        exit 1
    fi
else
    DATA_TYPE="single"
fi

TECH="$1"

exec "${DIR}/${TECH}/compute_valid_results.sh" "$2" "${DATA_TYPE}" "$3" "$4" "$5"
