#!/bin/bash
##SBATCH --job-name=Test                                #Setting a job name
#SBATCH --nodes=1                                       # Run on a single node
#SBATCH --ntasks-per-node=2
#SBATCH --partition=ai                                  # Run in ai queue
#SBATCH --qos=MAX_MEM
#SBATCH --account=ai
##SBATCH --gres=gpu:tesla_v100:1
#SBATCH --mem=200G
#SBATCH --time=7-0:0:0                                  # Time limit days-hours:minutes:seconds
#SBATCH --output=test-%j.out                            # Standard output and error log
#SBATCH --mail-type=ALL                                 # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=dkukul19@ku.edu.tr                  # Where to send mail


echo "Setting stack size to unlimited..."
ulimit -s unlimited
ulimit -l unlimited
ulimit -a
echo


nvidia-smi
source activate llava

python load_model_checkpoints.py

