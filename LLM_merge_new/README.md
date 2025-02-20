# LLM_merge
LLM kv cache merge codes


###################################### PiQA #######################################################
  

# Step 1: Prepare inference text
task=piqa
shots=5
python -u generate_task_data.py \
  --output-file /home/wan.512/ECG_LLMs/KV_cache_opt/LLM_merge/data/piqa-5.jsonl \
  --task-name piqa \
  --num-fewshot 5


CUDA_VISIBLE_DEVICES=0 python run_lm_eval_harness.py \
  --input-path /home/wan.512/ECG_LLMs/KV_cache_opt/LLM_merge/data/piqa-5.jsonl \
  --output-path /home/wan.512/ECG_LLMs/KV_cache_opt/LLM_merge/results/piqa-5-llama-full.jsonl \
  --model-name huggyllama/llama-7b \
  --model-type llama


CUDA_VISIBLE_DEVICES=1 python run_lm_eval_harness.py \
  --input-path /home/wan.512/ECG_LLMs/KV_cache_opt/LLM_merge/data/piqa-5.jsonl \
  --output-path /home/wan.512/ECG_LLMs/KV_cache_opt/LLM_merge/results/piqa-5-llama-local.jsonl \
  --model-name huggyllama/llama-7b \
  --model-type llama \
  --enable_small_cache \
  --heavy_ratio 0 \
  --recent_ratio 0.2

  # Step 2: Generate the output from LLaMA-7b with H2O
model=huggyllama/llama-7b
model_arch=llama
 CUDA_VISIBLE_DEVICES=0 python run_lm_eval_harness.py \
  --input-path /home/wan.512/ECG_LLMs/KV_cache_opt/LLM_merge/data/piqa-5.jsonl \
  --output-path /home/wan.512/ECG_LLMs/KV_cache_opt/LLM_merge/results/piqa-5-llama-h2o-undrop.jsonl \
  --model-name huggyllama/llama-7b \
  --model-type llama \
  --enable_small_cache \
  --heavy_ratio 0.1 \
  --recent_ratio 0.1


##### Real drop of tokens

CUDA_VISIBLE_DEVICES=0 python run_lm_eval_harness.py \
  --input-path /home/wan.512/ECG_LLMs/KV_cache_opt/LLM_merge/data/piqa-5.jsonl \
  --output-path /home/wan.512/ECG_LLMs/KV_cache_opt/LLM_merge/results/piqa-5-llama-real_drop.jsonl \
  --model-name huggyllama/llama-7b \
  --model-type llama \
  --heavy_ratio 0.1 \
  --recent_ratio 0.1 \
  --use_real_drop



######### Real merge of tokens

CUDA_VISIBLE_DEVICES=1 python run_lm_eval_harness.py \
  --input-path /home/wan.512/ECG_LLMs/KV_cache_opt/LLM_merge/data/piqa-5.jsonl \
  --output-path /home/wan.512/ECG_LLMs/KV_cache_opt/LLM_merge/results/piqa-5-llama-real_merge.jsonl \
  --model-name huggyllama/llama-7b \
  --model-type llama \
  --heavy_ratio 0.1 \
  --recent_ratio 0.1 \
  --use_real_merge

################ eval ####################

  CUDA_VISIBLE_DEVICES=0 python evaluate_task_result.py \
  --result-file /home/wan.512/ECG_LLMs/KV_cache_opt/LLM_merge/results/piqa-5-llama-full.jsonl \
  --task-name piqa \
  --num-fewshot 5 \
  --model-type llama

CUDA_VISIBLE_DEVICES=1 python evaluate_task_result.py \
  --result-file /home/wan.512/ECG_LLMs/KV_cache_opt/LLM_merge/results/piqa-5-llama-local.jsonl \
  --task-name piqa \
  --num-fewshot 5 \
  --model-type llama

CUDA_VISIBLE_DEVICES=2 python evaluate_task_result.py \
  --result-file /home/wan.512/ECG_LLMs/KV_cache_opt/LLM_merge/results/piqa-5-llama-h2o-undrop.jsonl \
  --task-name piqa \
  --num-fewshot 5 \
  --model-type llama

CUDA_VISIBLE_DEVICES=3 python evaluate_task_result.py \
  --result-file /home/wan.512/ECG_LLMs/KV_cache_opt/LLM_merge/results/piqa-5-llama-real_drop.jsonl \
  --task-name piqa \
  --num-fewshot 5 \
  --model-type llama

CUDA_VISIBLE_DEVICES=0 python evaluate_task_result.py \
  --result-file /home/wan.512/ECG_LLMs/KV_cache_opt/LLM_merge/results/piqa-5-llama-real_merge.jsonl \
  --task-name piqa \
  --num-fewshot 5 \
  --model-type llama
