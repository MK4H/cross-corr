#!/bin/bash

set -eu -o pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cd "${DIR}/../build"

/opt/nvidia/nsight-compute/2021.2.1/ncu --set full --export ./profile ./cross run nai_orig_n_to_mn ../data/ina_512_512_2_1.csv ../data/ina_512_512_4_1.csv