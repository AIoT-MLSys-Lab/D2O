import argparse
import json, tqdm
import torch
import copy


from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from not_real_drop_lm_eval.modify_llama import convert_kvcache_llama_heavy_recent, LlamaAttention_heavy_hitter
# from utils_lm_eval.modify_gptneox import convert_kvcache_gpt_neox_heavy_recent, GPTNeoXAttention_Mask
# from utils_lm_eval.modify_opt import convert_kvcache_opt_heavy_recent, OPTAttention_Mask
# from kv_token_merge.modify_llama import 
import time
from kv_token_merge.modify_llama import H2OLlamaForCausalLM_streaming
from kv_token_merge.modify_llama_merge import H2OLlamaForCausalLM_streaming_merge 


ENABLE_Heavy_Hitter_FUNCTIONS = {
    "llama": convert_kvcache_llama_heavy_recent,
    # "opt": convert_kvcache_opt_heavy_recent,
    # "gpt_neox": convert_kvcache_gpt_neox_heavy_recent,
}

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                        prog = 'ProgramName',
                        description = 'What the program does',
                        epilog = 'Text at the bottom of help')

    parser.add_argument('--input-path', type=str, default=None)
    parser.add_argument('--output-path', type=str, default=None)
    parser.add_argument('--enable_small_cache', action='store_true')
    parser.add_argument('--model-name', type=str, default='facebook/opt-350m')
    parser.add_argument('--model-type', type=str, default='opt')
    parser.add_argument("--cache-dir", type=str, default='/local/scratch0/data_and_checkpoint/models')
    parser.add_argument("--use_real_drop", action="store_true")
    parser.add_argument("--use_real_merge", action="store_true")

    parser.add_argument("--heavy_ratio", type=float, default=0.1)
    parser.add_argument("--recent_ratio", type=float, default=0.1)

    parser.add_argument("--heavy_hitter_size", type=int, default=16)
    parser.add_argument("--recent_size", type=int, default=240)
    
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    model_name = args.model_name

    config = AutoConfig.from_pretrained(model_name, cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=args.cache_dir)
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=args.cache_dir)

    if args.enable_small_cache:
          
        print('Enable Small Cache Size')
        config.heavy_ratio = args.heavy_ratio
        config.recent_ratio = args.recent_ratio
        checkpoint = copy.deepcopy(model.state_dict())
        model = ENABLE_Heavy_Hitter_FUNCTIONS[args.model_type](model, config)
        model.load_state_dict(checkpoint)

    if args.use_real_drop:

        print('Using real drop for LLMs ###################')

        config.hh_size = args.heavy_hitter_size
        config.recent_size = args.recent_size
        config.heavy_ratio = args.heavy_ratio
        config.recent_ratio = args.recent_ratio

        model = H2OLlamaForCausalLM_streaming.from_pretrained(
        args.model_name,
        device_map="auto",
        config=config,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        cache_dir = args.cache_dir
         )

    if args.use_real_merge:
    
        print('Using real merge for LLMs ###################')

        config.hh_size = args.heavy_hitter_size
        config.recent_size = args.recent_size
        config.heavy_ratio = args.heavy_ratio
        config.recent_ratio = args.recent_ratio

        model = H2OLlamaForCausalLM_streaming_merge.from_pretrained(
        args.model_name,
        device_map="auto",
        config=config,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        cache_dir = args.cache_dir
         )
      

    model.half().eval().cuda()

    requests = []
    with open(input_path, 'r') as f:
        for line in f:
            if line.strip() != '':
                requests.append(json.loads(line))

    results = []


    start_time = time.time()


    with torch.no_grad():
        for request in tqdm.tqdm(requests):
            result = {'request': request, 'result': {}}
            prompt = request['prompt']
            input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').input_ids.to(model.device)

            logits = model(input_ids).logits.log_softmax(dim=-1)

            values, indices = logits.squeeze(0).topk(dim=-1, k=1)
            tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))
            
            gold_indices = input_ids[:, 1:] # skip first
            logprobs = [None] + torch.gather(logits, -1, gold_indices.unsqueeze(-1)).squeeze(-1).squeeze(0).detach().cpu().tolist()
            top_logprobs = [None] + [{tokenizer.convert_ids_to_tokens(i.item()): v.item()} for v, i in zip(values.squeeze(-1), indices.squeeze(-1))]
            
            result['result'] = {
                "choices": [
                    {
                        "text": prompt, 
                        "logprobs": {
                            "tokens": tokens, 
                            "token_logprobs": logprobs, 
                            "top_logprobs": top_logprobs, 
                            "text_offset": []
                        }, 
                        "finish_reason": "length"
                    }
                ], 
                "request_time": {
                    "batch_time": 0, 
                    "batch_size": 1}
            }
            
            results.append(result)

    with open(output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

    print('Total inference time: ', time.time() - start_time)


"""

# Step 2 (Full Cache Baseline): Generate the output from LLaMA-7b with Full Cache
task=openbookqa
model=huggyllama/llama-7b
model_arch=llama
CUDA_VISIBLE_DEVICES=0 python run_lm_eval_harness.py \
  --input-path openbookqa-5.jsonl \
  --output-path /home/wan.512/ECG_LLMs/KV_cache_opt/LLM_merge/results/openbookqa-5-llama-full.jsonl \
  --model-name huggyllama/llama-7b \
  --model-type llama


CUDA_VISIBLE_DEVICES=0 python run_lm_eval_harness.py \
  --input-path openbookqa-5.jsonl \
  --output-path /home/wan.512/ECG_LLMs/KV_cache_opt/LLM_merge/results/openbookqa-5-llama-local.jsonl \
  --model-name huggyllama/llama-7b \
  --model-type llama \
  --enable_small_cache \
  --heavy_ratio 0 \
  --recent_ratio 0.2


# Step 2 (H2O): Generate the output from LLaMA-7b with H2O
model=huggyllama/llama-7b
model_arch=llama
 

##### Real drop of tokens

CUDA_VISIBLE_DEVICES=3 python run_lm_eval_harness.py \
  --input-path openbookqa-5.jsonl \
  --output-path /home/wan.512/ECG_LLMs/KV_cache_opt/LLM_merge/results/openbookqa-5-llama-real_drop.jsonl \
  --model-name huggyllama/llama-7b \
  --model-type llama \
  --heavy_ratio 0.1 \
  --recent_ratio 0.1 \
  --use_real_drop


######### Real merge of tokens

CUDA_VISIBLE_DEVICES=2 python run_lm_eval_harness.py \
  --input-path openbookqa-5.jsonl \
  --output-path /home/wan.512/ECG_LLMs/KV_cache_opt/LLM_merge/results/openbookqa-5-llama-real_merge.jsonl \
  --model-name huggyllama/llama-7b \
  --model-type llama \
  --heavy_ratio 0.05 \
  --recent_ratio 0.05 \
  --use_real_merge


# Step 3: Evaluate the performance of generated text

/home/wan.512/ECG_LLMs/KV_cache_opt/LLM_merge/results/openbookqa-5-llama-full.jsonl
/home/wan.512/ECG_LLMs/KV_cache_opt/LLM_merge/results/openbookqa-5-llama-local.jsonl
/home/wan.512/ECG_LLMs/KV_cache_opt/LLM_merge/results/openbookqa-5-llama-h2o-undrop.jsonl
/home/wan.512/ECG_LLMs/KV_cache_opt/LLM_merge/results/openbookqa-5-llama-real_drop.jsonl
/home/wan.512/ECG_LLMs/KV_cache_opt/LLM_merge/results/openbookqa-5-llama-real_merge.jsonl



CUDA_VISIBLE_DEVICES=0 python evaluate_task_result.py \
  --result-file /home/wan.512/ECG_LLMs/KV_cache_opt/LLM_merge/results/openbookqa-5-llama-full.jsonl \
  --task-name openbookqa \
  --num-fewshot 5 \
  --model-type llama

CUDA_VISIBLE_DEVICES=1 python evaluate_task_result.py \
  --result-file /home/wan.512/ECG_LLMs/KV_cache_opt/LLM_merge/results/openbookqa-5-llama-local.jsonl \
  --task-name openbookqa \
  --num-fewshot 5 \
  --model-type llama

CUDA_VISIBLE_DEVICES=2 python evaluate_task_result.py \
  --result-file /home/wan.512/ECG_LLMs/KV_cache_opt/LLM_merge/results/openbookqa-5-llama-h2o-undrop.jsonl \
  --task-name openbookqa \
  --num-fewshot 5 \
  --model-type llama

CUDA_VISIBLE_DEVICES=3 python evaluate_task_result.py \
  --result-file /home/wan.512/ECG_LLMs/KV_cache_opt/LLM_merge/results/openbookqa-5-llama-real_drop.jsonl \
  --task-name openbookqa \
  --num-fewshot 5 \
  --model-type llama

CUDA_VISIBLE_DEVICES=2 python evaluate_task_result.py \
  --result-file /home/wan.512/ECG_LLMs/KV_cache_opt/LLM_merge/results/openbookqa-5-llama-real_merge.jsonl \
  --task-name openbookqa \
  --num-fewshot 5 \
  --model-type llama


  

###################################### COPA #######################################################

# Step 1: Prepare inference text
task=copa
shots=5
python -u generate_task_data.py \
  --output-file /home/wan.512/ECG_LLMs/KV_cache_opt/LLM_merge/data/copa-5.jsonl \
  --task-name copa \
  --num-fewshot 5


# Step 2 (Full Cache Baseline): Generate the output from LLaMA-7b with Full Cache
task=copa
model=huggyllama/llama-7b
model_arch=llama


CUDA_VISIBLE_DEVICES=0 python run_lm_eval_harness.py \
  --input-path /home/wan.512/ECG_LLMs/KV_cache_opt/LLM_merge/data/copa-5.jsonl \
  --output-path /home/wan.512/ECG_LLMs/KV_cache_opt/LLM_merge/results/copa-5-llama-full.jsonl \
  --model-name huggyllama/llama-7b \
  --model-type llama


CUDA_VISIBLE_DEVICES=1 python run_lm_eval_harness.py \
  --input-path /home/wan.512/ECG_LLMs/KV_cache_opt/LLM_merge/data/copa-5.jsonl \
  --output-path /home/wan.512/ECG_LLMs/KV_cache_opt/LLM_merge/results/copa-5-llama-local.jsonl \
  --model-name huggyllama/llama-7b \
  --model-type llama \
  --enable_small_cache \
  --heavy_ratio 0 \
  --recent_ratio 0.2

  # Step 2 (H2O): Generate the output from LLaMA-7b with H2O
model=huggyllama/llama-7b
model_arch=llama
 CUDA_VISIBLE_DEVICES=0 python run_lm_eval_harness.py \
  --input-path /home/wan.512/ECG_LLMs/KV_cache_opt/LLM_merge/data/copa-5.jsonl \
  --output-path /home/wan.512/ECG_LLMs/KV_cache_opt/LLM_merge/results/copa-5-llama-h2o-undrop.jsonl \
  --model-name huggyllama/llama-7b \
  --model-type llama \
  --enable_small_cache \
  --heavy_ratio 0.1 \
  --recent_ratio 0.1


##### Real drop of tokens

CUDA_VISIBLE_DEVICES=0 python run_lm_eval_harness.py \
  --input-path /home/wan.512/ECG_LLMs/KV_cache_opt/LLM_merge/data/copa-5.jsonl \
  --output-path /home/wan.512/ECG_LLMs/KV_cache_opt/LLM_merge/results/copa-5-llama-real_drop.jsonl \
  --model-name huggyllama/llama-7b \
  --model-type llama \
  --heavy_ratio 0.1 \
  --recent_ratio 0.1 \
  --use_real_drop



######### Real merge of tokens

CUDA_VISIBLE_DEVICES=1 python run_lm_eval_harness.py \
  --input-path /home/wan.512/ECG_LLMs/KV_cache_opt/LLM_merge/data/copa-5.jsonl \
  --output-path /home/wan.512/ECG_LLMs/KV_cache_opt/LLM_merge/results/copa-5-llama-real_merge.jsonl \
  --model-name huggyllama/llama-7b \
  --model-type llama \
  --heavy_ratio 0.1 \
  --recent_ratio 0.1 \
  --use_real_merge


  ############# eval #####################


  
CUDA_VISIBLE_DEVICES=0 python evaluate_task_result.py \
  --result-file /home/wan.512/ECG_LLMs/KV_cache_opt/LLM_merge/results/copa-5-llama-full.jsonl \
  --task-name copa \
  --num-fewshot 5 \
  --model-type llama

CUDA_VISIBLE_DEVICES=1 python evaluate_task_result.py \
  --result-file /home/wan.512/ECG_LLMs/KV_cache_opt/LLM_merge/results/copa-5-llama-local.jsonl \
  --task-name copa \
  --num-fewshot 5 \
  --model-type llama

CUDA_VISIBLE_DEVICES=2 python evaluate_task_result.py \
  --result-file /home/wan.512/ECG_LLMs/KV_cache_opt/LLM_merge/results/copa-5-llama-h2o-undrop.jsonl \
  --task-name copa \
  --num-fewshot 5 \
  --model-type llama

CUDA_VISIBLE_DEVICES=3 python evaluate_task_result.py \
  --result-file /home/wan.512/ECG_LLMs/KV_cache_opt/LLM_merge/results/copa-5-llama-real_drop.jsonl \
  --task-name copa \
  --num-fewshot 5 \
  --model-type llama

CUDA_VISIBLE_DEVICES=0 python evaluate_task_result.py \
  --result-file /home/wan.512/ECG_LLMs/KV_cache_opt/LLM_merge/results/copa-5-llama-real_merge.jsonl \
  --task-name copa \
  --num-fewshot 5 \
  --model-type llama




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

  # Step 2 (H2O): Generate the output from LLaMA-7b with H2O
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

CUDA_VISIBLE_DEVICES=0 python run_lm_eval_harness.py \
  --input-path /home/wan.512/ECG_LLMs/KV_cache_opt/LLM_merge/data/piqa-5.jsonl \
  --output-path /home/wan.512/ECG_LLMs/KV_cache_opt/LLM_merge/results/piqa-5-llama-real_merge.jsonl \
  --model-name huggyllama/llama-7b \
  --model-type llama \
  --heavy_ratio 0.1 \
  --recent_ratio 0.1 \
  --use_real_merge

################ eval ####################

  CUDA_VISIBLE_DEVICES=1 python evaluate_task_result.py \
  --result-file /home/wan.512/ECG_LLMs/KV_cache_opt/LLM_merge/results/piqa-5-llama-full.jsonl \
  --task-name piqa \
  --num-fewshot 5 \
  --model-type llama

CUDA_VISIBLE_DEVICES=2 python evaluate_task_result.py \
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

"""