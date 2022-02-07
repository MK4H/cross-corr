#!/bin/bash

set -eu -o pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cd "${DIR}/../build"

/opt/nvidia/nsight-compute/2021.2.1/ncu --set full --export "${DIR}/nai_shuffle" ./cross run nai_warp_shuffle_one_to_one ../data/ina_64_64_1_1.csv ../data/ina_64_64_1_2.csv
