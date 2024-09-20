import argparse
import json
import os.path

import tqdm
import torch
import copy
from copy import deepcopy
import dataclasses
# from xopen import xopen
import math
import matplotlib.pyplot as plt 

from rouge import Rouge
import logging
import numpy as np

# from lost_in_the_middle.prompting import (
#     Document,
#     get_closedbook_qa_prompt,
#     get_qa_prompt,
#     get_qa_prompt_index
# )

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.models.llama.configuration_llama import LlamaConfig
from kv_token_merge.modify_llama_merge import H2OLlamaAttention_streaming 
from kv_token_merge.modify_llama import H2OLlamaForCausalLM, H2OLlamaAttention
from kv_token_merge.modify_llama_merge import H2OLlamaForCausalLM_streaming_merge

from not_real_drop_hh.modify_llama import convert_kvcache_llama_heavy_recent, LlamaAttention_heavy_hitter


from kv_token_merge.modify_llama_merge import H2OLlamaAttention_merge, H2OLlamaForCausalLM_merge



os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


ENABLE_Heavy_Hitter_FUNCTIONS = {
    "llama": None,
    "llama_h2o": H2OLlamaForCausalLM,
    "prune_merge": H2OLlamaForCausalLM_merge,
    "un_drop_h2o": convert_kvcache_llama_heavy_recent
}

TAGET_MODULE = {
    "llama": None,
    "llama_h2o": H2OLlamaAttention,
    "prune_merge": H2OLlamaAttention_merge,
    "un_drop_h2o": LlamaAttention_heavy_hitter
}

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--input_path", type=str, default="")
    parser.add_argument("--output_path", type=str, default="")

    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument("--cache_dir", type=str, default="/fs/scratch/PAS2473/zhongwei_models")

    parser.add_argument("--hh_size", type=int, default=200)
    parser.add_argument("--recent_size", type=int, default=200)

    parser.add_argument("--heavy_ratio", type=float, default=None)
    parser.add_argument("--recent_ratio", type=float, default=None)

    parser.add_argument('--enable_h2o_cache', action='store_true')

    parser.add_argument('--enable_prune_merge', action='store_true')

    parser.add_argument('--enable_undrop_h2o', action='store_true')

    parser.add_argument("--sample_num", type=int, default=100)
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    set_seed(args)

    model_name = args.model_name
    input_path = args.input_path
    output_path = args.output_path

    config = AutoConfig.from_pretrained(model_name, cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, cache_dir=args.cache_dir)

    if args.batch_size>1:
        tokenizer.pad_token = tokenizer.eos_token

    if args.enable_h2o_cache:
        print('Enabling H2O KV cache')
        config.hh_size = args.hh_size
        config.recent_size = args.recent_size
        config.hh_ratio = None
        config.recent_ratio = None
        model = ENABLE_Heavy_Hitter_FUNCTIONS['llama_h2o'].from_pretrained(model_name, config=config,
                                                                            cache_dir=args.cache_dir)

    elif args.enable_prune_merge:

        config.hh_size = args.hh_size
        config.recent_size = args.recent_size
        config.hh_ratio = None
        config.recent_ratio = None
        model = ENABLE_Heavy_Hitter_FUNCTIONS['prune_merge'].from_pretrained(model_name, config=config,
                                                                            cache_dir=args.cache_dir)

    elif args.enable_undrop_h2o:

        config.hh_size = args.hh_size
        config.recent_size = args.recent_size
        config.hh_ratio = args.heavy_ratio
        config.recent_ratio = args.recent_ratio
        model = ENABLE_Heavy_Hitter_FUNCTIONS['un_drop_h2o'].from_pretrained(model_name, config=config,
                                                                            cache_dir=args.cache_dir)

    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=args.cache_dir)

    model.half().eval().cuda()

    requests = []
    with open(input_path, 'r') as f:
        for line in f:
            if line.strip() != '':
                requests.append(json.loads(line))

    print(len(requests))
    if args.sample_num < len(requests):
        print('Sample {} Examples from {} samples'.format(args.sample_num, len(requests)))
    requests = requests[:args.sample_num]

    results = []
    rouge = Rouge()
    rouge1_score_list = []
    rouge2_score_list = []
    rougel_score_list = []

    with torch.no_grad():
        for request in tqdm.tqdm(requests):
            result = {'request': request, 'result': {}}
            prompt = request['article']
            label = request['summary_gt']
            temperature = request['temperature']
            stop = request['stop']

            input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').input_ids.to(model.device)

            model.config.hh_size = int((request['max_tokens'] + len(input_ids[0]))*0.1)
            model.config.recent_size = int((request['max_tokens'] + len(input_ids[0]))*0.1)

            # print(model.config.hh_size)
            # breakpoint()

            output_sequences = model.generate(
                input_ids=input_ids,
                max_length=request['max_tokens'] + len(input_ids[0]),
                temperature=temperature,
                top_k=args.k,
                top_p=request['top_p'],
                do_sample=True,
                num_return_sequences=request['n'],
                return_dict_in_generate=True, output_scores=True,
            )

            if args.enable_h2o_cache:
                for name, m in model.named_modules():
                    if isinstance(m, TAGET_MODULE['llama_h2o']):
                        m._clean_cache()
                        # breakpoint()

            if args.enable_prune_merge:
                for name, m in model.named_modules():
                    if isinstance(m, TAGET_MODULE['prune_merge']):
                        m._clean_cache()
                        # breakpoint()


            tokens = tokenizer.convert_ids_to_tokens(output_sequences['sequences'].squeeze(0))[len(input_ids[0]):]
            logprobs = [logits.log_softmax(dim=-1).max().item() for logits in output_sequences['scores']]
            top_logprobs = [{i: v for i, v in zip(tokens, logprobs)}]

            generate_text = tokenizer.decode(output_sequences['sequences'].squeeze(0)[len(input_ids[0]):])
            generate_text = generate_text[: generate_text.find(stop[0])]

            scores = rouge.get_scores(generate_text, label)[0]
            rouge1_score_list.append(scores['rouge-1']['f'])
            rouge2_score_list.append(scores['rouge-2']['f'])
            rougel_score_list.append(scores['rouge-l']['f'])

            result['result'] = {
                "choices": [
                    {
                        "text": generate_text,
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
            print('rouge-1: {:.6f}, rouge-2: {:.6f}, rouge-l: {:.6f}'.format(np.mean(rouge1_score_list), np.mean(rouge2_score_list), np.mean(rougel_score_list)))

    with open(output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')


"""
# Full baseline on XSUM
bash scripts/summarization/eval.sh xsum 5 full 2

# H2O KV Cache on XSUM
bash scripts/summarization/eval.sh xsum 5 h2o 0 50 50

rouge-1: 0.294890, rouge-2: 0.105549, rouge-l: 0.244708

# pruning_merge KV Cache on XSUM
bash scripts/summarization/eval.sh xsum 5 pruing_merge 1 50 50


Results: ######################## 

rouge-1: 0.325336, rouge-2: 0.126485, rouge-l: 0.274510  full

rouge-1: 0.323776, rouge-2: 0.123203, rouge-l: 0.266987  H2o

rouge-1: 0.322280, rouge-2: 0.124021, rouge-l: 0.270728 merge

task=$1
shots=$2
method=$3
GPU=$4
HH_SIZE=$5
RECENT_SIZE=$6

if [[ ${method} == 'h2o' ]]; then
    CUDA_VISIBLE_DEVICES=${GPU} python -u run_summarization.py \
        --input_path data/summarization_data/${task}_${shots}shot.jsonl \
        --output_path summary_results/${task}_${shots}shot_h2o_hh${1}_local${2}.jsonl \
        --model_name meta-llama/Llama-2-7b-hf \
        --heavy_ratio ${HH_SIZE} \
        --recent_ratio ${RECENT_SIZE} \
        --cache_dir /fs/scratch/PAS2473/zhongwei_models \
        --enable_h2o_cache
elif [[ ${method} == 'pruing_merge' ]]; then
    CUDA_VISIBLE_DEVICES=${GPU} python -u run_summarization.py \
        --input_path data/summarization_data/${task}_${shots}shot.jsonl \
        --output_path summary_results/${task}_${shots}shot_h2o_hh${1}_local${2}.jsonl \
        --model_name meta-llama/Llama-2-7b-hf \
        --heavy_ratio ${HH_SIZE} \
        --recent_ratio ${RECENT_SIZE} \
        --cache_dir /fs/scratch/PAS2473/zhongwei_models \
        --enable_prune_merge

elif [[ ${method} == 'un_drop_h2o' ]]; then
    CUDA_VISIBLE_DEVICES=${GPU} python -u run_summarization.py \
        --input_path data/summarization_data/${task}_${shots}shot.jsonl \
        --output_path summary_results/${task}_${shots}shot_h2o_hh${1}_local${2}.jsonl \
        --model_name meta-llama/Llama-2-7b-hf \
        --heavy_ratio ${HH_SIZE} \
        --recent_ratio ${RECENT_SIZE} \
        --cache_dir /fs/scratch/PAS2473/zhongwei_models 
elif [[ ${method} == 'full' ]]; then
    CUDA_VISIBLE_DEVICES=${GPU} python -u run_summarization.py \
        --input_path data/summarization_data/${task}_${shots}shot.jsonl \
        --output_path summary_results/${task}_${shots}shot_full.jsonl \
        --model_name meta-llama/Llama-2-7b-hf \
        --cache_dir /fs/scratch/PAS2473/zhongwei_models
else
    echo 'unknown argment for method'
fi


"""