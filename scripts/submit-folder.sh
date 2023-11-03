#!/bin/bash
if [ $# -eq 0 ]
  then
    echo "Path to folder not specified"
    exit 1
fi

folder=$1

out_dir="../results/$(basename $folder)"

for filename in ./$folder/*; do
    echo "Submitting $filename to out_dir $out_dir"
    sbatch scripts/submit-move-experiment.sh $filename $out_dir
done
