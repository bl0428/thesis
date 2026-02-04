#!/bin/bash
#SBATCH --job-name=train         # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=12       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=1G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1             # gpus per task
#SBATCH --time=24:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=bl0428@princeton.edu

module purge
module load anaconda3/2024.6
source /scratch/gpfs/GRIFFITHS/bl0428/thesis/mujoco-env/bin/activate

cd /scratch/gpfs/GRIFFITHS/bl0428/thesis/model/
python train_duel.py
