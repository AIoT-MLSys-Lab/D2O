#!/bin/bash

#SBATCH --job-name=gsm8k_merge_0.4       # 作业名称
#SBATCH --account=PAS2473		    # Project ID
#SBATCH --output=/users/PAS2473/brucewan666/xjwu/LLM_merge/output_logs_0503/gsm8k_merge_0.4.log        # 输出日志文件
#SBATCH --error=/users/PAS2473/brucewan666/xjwu/LLM_merge/output_logs_0503/gsm8k_merge_0.4_error.log         # 错误日志文件
#SBATCH --nodes=1                   # 节点数
#SBATCH --ntasks-per-node=1         # 每个节点的任务数
#SBATCH --cpus-per-task=4           # 每个任务使用的 CPU 核心数
#SBATCH --gpus-per-node=1           # GPU per node
#SBATCH --mem=50G                   # 内存限制
#SBATCH --time=04:00:00             # 作业运行时间限制

# 运行命令或脚本 wget https://repo.anaconda.com/archive/Anaconda3-2023.07-2-Linux-x86_64.sh
source $HOME/anaconda3/bin/activate /users/PAS2473/brucewan666/anaconda3/envs/kivi
# module load cuda 
export CUDA_V171704ISIBLE_DEVICES=0


# python /users/PAS2473/brucewan666/xjwu/LLM_merge/run_lm_eval_harness_generation.py --model_name_or_path meta-llama/Llama-2-7b-hf \
#         --tasks gsm8k \
#         --cache_dir /fs/scratch/PAS2473/zhongwei_models \
#         --use_real_drop True \
#         --hh_ratio 0.1 \
#         --recent_ratio 0.1


python /users/PAS2473/brucewan666/xjwu/LLM_merge/run_lm_eval_harness_generation.py --model_name_or_path meta-llama/Llama-2-7b-hf \
        --tasks gsm8k \
        --cache_dir /fs/scratch/PAS2473/zhongwei_models \
        --use_real_merge True \
        --hh_ratio 0.2 \
        --recent_ratio 0.2

