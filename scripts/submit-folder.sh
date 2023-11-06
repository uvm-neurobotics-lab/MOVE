#!/bin/bash
if [ $# -eq 0 ]
  then
    echo "Path to folder not specified"
    exit 1
fi

folder=$1
shift # past argument or value

# named arguments from commandline
while [[ $# -gt 1 ]]
do
key="$1"

case $key in
    -r|--repeats)
    repeats="$2"
    shift # past argument
    ;;
    *)
    echo "Unknown option $key"
    exit 1
    ;;
esac
shift # past argument or value
done

repeats=${repeats:-1}
echo "Running $repeats times"

out_dir="../results/$(basename $folder)"

mkdir -p ../results/jobs

for i in $(seq 1 $repeats); do
  for filename in ./$folder/*; do
      if [ -d "$filename" ]; then
        echo "Skipping directory $filename"
          continue
      fi
      echo "Submitting $filename to out_dir $out_dir"
      sbatch scripts/submit-move-experiment.sh $filename $out_dir
  done
done