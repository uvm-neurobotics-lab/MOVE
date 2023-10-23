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
    -g|--gens)
    gens="$2"
    shift # past argument
    ;;
    -r|--repeats)
    repeats="$2"
    shift # past argument
    ;;
    -w|--workers)
    workers="$2"
    shift # past argument
    ;;
    *)
    echo "Unknown option $key"
    exit 1
    ;;
esac
shift # past argument or value
done

gens=${gens:-1000}
repeats=${repeats:-1}
workers=${workers:-1}

echo "Using $gens gens"
echo "Running $repeats times"

echo "Using a maximum of "$workers" workers"

re='^[0-9]+$'
if ! [[ $workers =~ $re ]] ; then
   echo "error: workers [$workers] is not a number" >&2; exit 1
fi

d=$(dirname $0)

folder_basename=$(basename $folder)
logdir=../logs/$folder_basename
mkdir -p -- $logdir

for i in $(seq 1 $repeats); do
  echo "Running all scripts in $folder with $gens gens"
  run_id=$(date +%Y%m%d%H%M%S)
  running=0
  for filename in ./$folder/*; do
      echo "Running: $filename"
      
      # trap ctrl-c and call ctrl_c()
      trap ctrl_c INT
      function ctrl_c() {
        echo "Trapped CTRL-C"
        # pass sigint to all child processes
        pkill -P $$
        exit 1
      }

      # python3 -O move.py -c $filename -g $gens -pr & #  >$logdir/$(basename $filename)_$run_id.log &
      python3 move.py -c $filename -g $gens -pr & #  >$logdir/$(basename $filename)_$run_id.log &
      
      running=$((running+1))
      if [ $running -eq $workers ]; then
        echo "Waiting for $running processes to finish"
        wait
        running=0
      fi
  done

  # wait for all processes to finish
  wait
done
