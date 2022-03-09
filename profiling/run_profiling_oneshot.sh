#!/bin/bash

set -eu -o pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

if [[ $# -ne 4 ]]
then
	echo "Usage: $0 <algorithm> <args_path> <left_input> <right_input>" >& 2
	exit 1
fi


cd "${DIR}/../build"

ALG="$1"
ARGS="$2"
LEFT="$3"
RIGHT="$4"

OUT="${DIR}/${ALG}"

/opt/nvidia/nsight-compute/2021.2.1/ncu --set full --export "${OUT}" ./cross run --args_path "${ARGS}" "${ALG}" "${LEFT}" "${RIGHT}"

echo "Results: ${OUT}"
