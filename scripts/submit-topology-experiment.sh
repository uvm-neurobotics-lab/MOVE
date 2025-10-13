#!/bin/bash
# specify a partition
#SBATCH --partition=nvgpu
# Request nodes
#SBATCH --nodes=1
# Request some processor cores
#SBATCH --ntasks=1
# Maximum runtime of 24 hours
#SBATCH --time=48:00:00
# GPUS
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
# Name of this job
#SBATCH --job-name=MOVE
# Output of this job, stderr and stdout are joined by default
# %x=job-name %j=jobid
#SBATCH --output=../results/jobs/%j.out
# stop 60 seconds early to save output
#SBATCH --signal=B:SIGINT@60

# Allow for the use of conda activate
source ~/.bashrc


# Move to submission directory
cd ${SLURM_SUBMIT_DIR}

# your job execution follows:
echo "starting job"

CONDA_ENV="move"
OUTDIR="../results/default"
JOBS=1


if [ ! -z "$1" ]
  then
    echo "Using $1 as conda env"
    CONDA_ENV=$1
fi


conda activate "$CONDA_ENV"

nvidia-smi


CMDS=()
if [ $JOBS -gt 1 ]
then
  echo "Running $JOBS jobs"
  for i in $(seq 2 $JOBS)
  do
    CMDS+=("time python analysis/topology_mutations.py &")
  done

fi

CMDS+=("time python analysis/topology_mutations.py")

mkdir -p ${OUTDIR}
NOW="$(date +"%D %T")"
echo "$SLURM_JOB_ID : $NOW : run $EXPERIMENT_FILE : $CMDS" >> ../results/ongoing.out
echo "START $SLURM_JOB_ID : $NOW : run $EXPERIMENT_FILE : $CMDS" >> "../results/$OUTDIR/job.out"
for ((i = 0; i < ${#CMDS[@]}; i++)) 
do
    echo "CMD $i: ${CMDS[$i]}"
    eval "${CMDS[$i]}"
done
NOW="$(date +"%D %T")"
echo "END   $SLURM_JOB_ID : $NOW : run $EXPERIMENT_FILE : $CMDS" >> ../results/record.out
echo "END   $SLURM_JOB_ID : $NOW : run $EXPERIMENT_FILE : $CMDS" >> "../results/$OUTDIR/job.out"
sed -i "/$SLURM_JOB_ID/d" ../results/ongoing.out
