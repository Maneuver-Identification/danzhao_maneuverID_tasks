#!/bin/bash
#SBATCH -n 40
#SBATCH -N 2
#SBATCH -o run.sh.log

source /etc/profile
module load anaconda/2021a
# source activate /home/gridsan/DA30449/.conda/envs/tse-vision-env

# srun python main.py \
# 	--model_size=${MODEL} \
# 	--dataset_size=${DATA} \
# 	--num_nodes=${SLURM_NNODES} \
# 	--num_epochs=${NUM_EPOCHS} \
#     --batch_size=${BATCH_SIZE} \
# 	--gpus=2 \
# 	2>${ERR_LOG} 1>${LOG_FILE}
    
python multivar_sim.py    

echo "all done"