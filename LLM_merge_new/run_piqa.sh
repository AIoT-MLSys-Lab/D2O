

CUDA_VISIBLE_DEVICES=0 python run_lm_eval_harness.py \
  --input-path /home/wan.512/ECG_LLMs/KV_cache_opt/LLM_merge/data/piqa-5.jsonl \
  --output-path /home/wan.512/ECG_LLMs/KV_cache_opt/LLM_merge/results/piqa-5-llama-local.jsonl \
  --model-name huggyllama/llama-7b \
  --model-type llama \
  --enable_small_cache \
  --heavy_ratio 0 \
  --recent_ratio 0.2

  # Step 2 (H2O): Generate the output from LLaMA-7b with H2O

 CUDA_VISIBLE_DEVICES=1 python run_lm_eval_harness.py \
  --input-path /home/wan.512/ECG_LLMs/KV_cache_opt/LLM_merge/data/piqa-5.jsonl \
  --output-path /home/wan.512/ECG_LLMs/KV_cache_opt/LLM_merge/results/piqa-5-llama-h2o-undrop.jsonl \
  --model-name huggyllama/llama-7b \
  --model-type llama \
  --enable_small_cache \
  --heavy_ratio 0.1 \
  --recent_ratio 0.1


##### Real drop of tokens

CUDA_VISIBLE_DEVICES=2 python run_lm_eval_harness.py \
  --input-path /home/wan.512/ECG_LLMs/KV_cache_opt/LLM_merge/data/piqa-5.jsonl \
  --output-path /home/wan.512/ECG_LLMs/KV_cache_opt/LLM_merge/results/piqa-5-llama-real_drop.jsonl \
  --model-name huggyllama/llama-7b \
  --model-type llama \
  --heavy_ratio 0.1 \
  --recent_ratio 0.1 \
  --use_real_drop


######### Real merge of tokens

CUDA_VISIBLE_DEVICES=3 python run_lm_eval_harness.py \
  --input-path /home/wan.512/ECG_LLMs/KV_cache_opt/LLM_merge/data/piqa-5.jsonl \
  --output-path /home/wan.512/ECG_LLMs/KV_cache_opt/LLM_merge/results/piqa-5-llama-real_merge.jsonl \
  --model-name huggyllama/llama-7b \
  --model-type llama \
  --heavy_ratio 0.1 \
  --recent_ratio 0.1 \
  --use_real_merge
