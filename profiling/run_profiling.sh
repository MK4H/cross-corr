#!/bin/bash

set -eu -o pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cd "${DIR}/../build"


for ALG in nai_orig nai_warp_shuffle nai_warp_shuffle_work_distribution nai_multirow_shuffle nai_multileft_shuffle nai_shift_per_warp nai_shift_per_warp_shared_mem_rows
do
	for SIZE in 64 128 256 512
	do
		/opt/nvidia/nsight-compute/2021.2.1/ncu --set full --export "${DIR}/${ALG}_${SIZE}" ./cross run "${ALG}_one_to_one" "../data/ina_${SIZE}_${SIZE}_1_1.csv" "../data/ina_${SIZE}_${SIZE}_1_2.csv"
	done
done
