#!/bin/bash
#SBATCH --job-name=EffNet_Train                # Job name
#SBATCH --output=EffNet_%j.log                 # Output log file; %j is replaced with job ID
#SBATCH --error=EffNet_%j.err                  # Error log file
#SBATCH --partition=agpu06                     # Use partition "agpu06"
#SBATCH --nodes=1                            # Number of nodes
#SBATCH --ntasks-per-node=1                  # 1 task per node
#SBATCH --qos=gpu                            # Quality of Service (if required)
#SBATCH --time=6:00:00                      # Time limit (HH:MM:SS)
#SBATCH --gres=gpu:1                         # Request 1 GPU per node

#####################
# Environment Setup
#####################
cd /home/dgnystro/walmart_project/training

# Activate the virtual environment
source /home/dgnystro/walmart_project/myenv/bin/activate

echo "Starting EfficientNetB2 training job on $SLURM_JOB_NODELIST ..."
echo "Job started at: $(date)"

#####################
# Run the Training Command
#####################
python train.py \
--dataset_path /home/dgnystro/walmart_project/training/datasets/produce_dataset \
--train_batch 256 \
--test_batch 64 \
--num_epochs 30 \
--save_image_dir saved_images \
--output_model produce-dataset-efficientnet_b2.zip \
--model_name efficientnet_b2

echo "EfficientNet training job finished at: $(date)"