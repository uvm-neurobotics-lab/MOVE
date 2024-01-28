#!/bin/bash
if [ $# -eq 0 ]
  then
    echo "Path to folder not specified"
    exit 1
fi

folder=$1
shift # past argument or value

# named arguments from commandline
while [[ $# -gt 0 ]]
do
  key="$1"
  case $key in
  # -d | --dry) is a flag (no value after it)
      -d|--dry)
      dry_run=1
      # shift # past argument
      ;;
      -r|--repeats)
      repeats="$2"
      shift # past argument
      ;;
      -t|--target)
      target="$2"
      shift # past argument
      ;;
      -c|--condition)
      condition="$2"
      shift # past argument
      ;;
      -hc|--hillclimber)
      hillclimber=1
      # shift # past argument
      ;;
      *)
      echo "Unknown option $key"
      exit 1
      ;;
  esac
  shift # past argument or value
done

repeats=${repeats:-1}
dry_run=${dry_run:-0}
target=${target:-"."}
condition=${condition:-"."}
echo "Running $repeats times"

out_dir="../results/$(basename $folder)"

mkdir -p ../results/jobs
echo "Dry run: $dry_run"
for i in $(seq 1 $repeats); do
  for filename in ./$folder/*; do
      if [ -d "$filename" ]; then
        echo "Skipping directory $filename"
          continue
      fi
      if [[ "$filename" != *"$target"* ]];then
        printf 'skip %s\n' "$filename"
        continue
      fi
      if [[ "$filename" != *"$condition"* ]];then
        printf 'skip %s\n' "$filename"
        continue
      fi

     echo "Submitting $filename to out_dir $out_dir"

    script_name=""
    if [ $hillclimber -eq 1 ]; then
      script_name="submit-hillclimber-experiment.sh"
    else
      script_name="submit-move-experiment.sh"
    fi

    if [ $dry_run -eq 0 ]; then
      sbatch scripts/$script_name $filename $out_dir
    else
      echo "would do: sbatch scripts/$script_name $filename $out_dir"
    fi


  done
done