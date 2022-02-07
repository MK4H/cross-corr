#!/bin/bash

set -eu -o pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

if [[ $# -ne 1 ]]
then
	echo "Usage: $0 <algorithm>" >& 2
	exit 1
fi


cd "${DIR}/../build"

ALG="$1"

for SIZE in 64 128 256 512
do
    /opt/nvidia/nsight-compute/2021.2.1/ncu --set full --export "${DIR}/${ALG}_${SIZE}" ./cross run "${ALG}_one_to_one" "../data/ina_${SIZE}_${SIZE}_1_1.csv" "../data/ina_${SIZE}_${SIZE}_1_2.csv"
done
