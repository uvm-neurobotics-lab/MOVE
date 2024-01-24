#!/bin/bash
# Make sure this is run from the root of the project, not the scripts folder
# Usage: scripts/generate_all_rmnk.sh
#

if [ ! -f "scripts/generate_all_rmnk.sh" ]; then
    echo "Please run this script from the root of the project, not the scripts folder"
    exit 1
fi

                            #       R       M   N       K    SEED   FILENAME
Rscript scripts/generate_rmnk.r     0.0     10  100     0    0      "data/rmnk_0_10_100_0_0.dat"
Rscript scripts/generate_rmnk.r     0.0     10  100     1    0      "data/rmnk_0_10_100_1_0.dat"
Rscript scripts/generate_rmnk.r     0.0     10  100     2    0      "data/rmnk_0_10_100_2_0.dat"

# RHO experiments
Rscript scripts/generate_rmnk.r     0.0     10  100     5    0      "data/rmnk_0_10_100_5_0.dat"
Rscript scripts/generate_rmnk.r     "0.25"   10  100     5    0      "data/rmnk_0.25_10_100_5_0.dat"
Rscript scripts/generate_rmnk.r     "0.7"   10  100     5    0      "data/rmnk_0.7_10_100_5_0.dat"
Rscript scripts/generate_rmnk.r     "0.99"   10  100     5    0      "data/rmnk_0.99_10_100_5_0.dat"
Rscript scripts/generate_rmnk.r     "-0.25"  10  100     5    0      "data/rmnk_-0.25_10_100_5_0.dat"
Rscript scripts/generate_rmnk.r     "-0.7"  10  100     5    0      "data/rmnk_-0.7_10_100_5_0.dat"

# M experiments
Rscript scripts/generate_rmnk.r     0       2  100     5    0      "data/rmnk_0_2_100_5_0.dat"
Rscript scripts/generate_rmnk.r     0       5  100     5    0      "data/rmnk_0_5_100_5_0.dat"
Rscript scripts/generate_rmnk.r     0.25    2  100     5    0      "data/rmnk_0.25_2_100_5_0.dat"
Rscript scripts/generate_rmnk.r     0.25    5  100     5    0      "data/rmnk_0.25_5_100_5_0.dat"
Rscript scripts/generate_rmnk.r     0.25    10  100    5    0      "data/rmnk_0.25_10_100_5_0.dat"
Rscript scripts/generate_rmnk.r     0.25    20  100    5    0      "data/rmnk_0.25_20_100_5_0.dat"