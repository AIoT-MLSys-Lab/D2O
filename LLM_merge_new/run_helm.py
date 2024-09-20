#!/usr/bin/env python
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Conditional text generation with the auto-regressive models of the library (GPT/GPT-2/CTRL/Transformer-XL/XLNet)
"""


import argparse
import logging

import numpy as np
import torch
import json
import tqdm 
import copy 

from transformers import (
    CTRLLMHeadModel,
    CTRLTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    TransfoXLLMHeadModel,
    TransfoXLTokenizer,
    XLMTokenizer,
    XLMWithLMHeadModel,
    XLNetLMHeadModel,
    XLNetTokenizer,
)

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from not_real_drop_hh.modify_llama import convert_kvcache_llama_heavy_recent, LlamaAttention_heavy_hitter
import time
from kv_token_merge.modify_llama import H2OLlamaForCausalLM_streaming
from kv_token_merge.modify_llama_merge import H2OLlamaForCausalLM_streaming_merge 

# from utils_hh.modify_gptneox import convert_kvcache_gpt_neox_heavy_recent, GPTNeoXAttention_Mask
# from utils_hh.modify_opt import convert_kvcache_opt_heavy_recent, OPTAttention_Mask


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

MODEL_CLASSES = {
    "gpt2": (GPT2LMHeadModel, GPT2Tokenizer),
    "ctrl": (CTRLLMHeadModel, CTRLTokenizer),
    "openai-gpt": (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "xlnet": (XLNetLMHeadModel, XLNetTokenizer),
    "transfo-xl": (TransfoXLLMHeadModel, TransfoXLTokenizer),
    "xlm": (XLMWithLMHeadModel, XLMTokenizer),
}

# Padding text to help Transformer-XL and XLNet with short prompts as proposed by Aman Rusia
# in https://github.com/rusiaaman/XLNet-gen#methodology
# and https://medium.com/@amanrusia/xlnet-speaks-comparison-to-gpt-2-ea1a4e9ba39e
PREFIX = """In 1991, the remains of Russian Tsar Nicholas II and his family
(except for Alexei and Maria) are discovered.
The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
remainder of the story. 1883 Western Siberia,
a young Grigori Rasputin is asked by his father and a group of men to perform magic.
Rasputin has a vision and denounces one of the men as a horse thief. Although his
father initially slaps him for making such an accusation, Rasputin watches as the
man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

ENABLE_Heavy_Hitter_FUNCTIONS = {
    "llama": convert_kvcache_llama_heavy_recent,
    # "opt": convert_kvcache_opt_heavy_recent,
    # "gpt_neox": convert_kvcache_gpt_neox_heavy_recent,
}

TAGET_MODULE = {
    "llama": LlamaAttention_heavy_hitter,
    # "opt": OPTAttention_Mask,
    # "gpt_neox": GPTNeoXAttention_Mask,
}

def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_path", type=str, default="")
    parser.add_argument("--output_path", type=str, default="")

    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument('--model_arch', type=str, default='opt')
    parser.add_argument("--cache_dir", type=str, default="/local/scratch0/data_and_checkpoint/models")

    parser.add_argument("--heavy_ratio", type=float, default=0.1)
    parser.add_argument("--recent_ratio", type=float, default=0.1)

    parser.add_argument("--heavy_hitter_size", type=int, default=16)
    parser.add_argument("--recent_size", type=int, default=240)
    parser.add_argument('--enable_small_cache', action='store_true')
    parser.add_argument("--use_real_drop", action="store_true")
    parser.add_argument("--use_real_merge", action="store_true")


    parser.add_argument("--sample_num", type=int, default=1000)
    

    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

    logger.warning(f"device: {args.device}, n_gpu: {args.n_gpu}, 16-bits training: {args.fp16}")
    set_seed(args)

    model_name = args.model_name
    input_path = args.input_path
    output_path = args.output_path 

    config = AutoConfig.from_pretrained(model_name, cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, cache_dir=args.cache_dir)
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=args.cache_dir)

    if args.enable_small_cache:
        print('Enable Small Cache Size')
        config.heavy_ratio = args.heavy_ratio
        config.recent_ratio = args.recent_ratio
        checkpoint = copy.deepcopy(model.state_dict())
        model = ENABLE_Heavy_Hitter_FUNCTIONS[args.model_arch](model, config)
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
        # torch_dtype=torch.float16,
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
        # torch_dtype=torch.float16,
        trust_remote_code=True,
        cache_dir = args.cache_dir
         )
    

    model.half().eval().cuda()
    logger.info(args)

    requests = []
    with open(input_path, 'r') as f:
        for line in f:
            if line.strip() != '':
                requests.append(json.loads(line))

    print(len(requests))
    if args.sample_num < len(requests):
        print('Sample {} Examples'.format(args.sample_num))
    requests = requests[:args.sample_num]

    results = []

    
    start_time = time.time()
    with torch.no_grad():
        for request in tqdm.tqdm(requests):
            request = request['request']
            result = {'request': request, 'result': {}}
            prompt = request['prompt']
            temperature = request['temperature']
            stop = request['stop']

            input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').input_ids.to(model.device)

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

            for name, m in model.named_modules():
                if isinstance(m, TAGET_MODULE[args.model_arch]):
                    m._reset_masks()

            tokens = tokenizer.convert_ids_to_tokens(output_sequences['sequences'].squeeze(0))[len(input_ids[0]):]
            logprobs = [logits.log_softmax(dim=-1).max().item() for logits in output_sequences['scores']]
            top_logprobs = [{i: v for i, v in zip(tokens, logprobs)}]

            generate_text = tokenizer.decode(output_sequences['sequences'].squeeze(0)[len(input_ids[0]):])
            generate_text = generate_text[: generate_text.find(stop[0])]

############ from github 233 line 

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

    print('the total inference time: ', time.time() - start_time)

    with open(output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')


if __name__ == "__main__":
    main()


""""
# Step 1: prepare inference text
# Examples of converting inference data to jsonl format is provided in helm/command/get_data.sh
# And the data is provided in data/


model=huggyllama/llama-7b
model_arch=llama


#### full #################
CUDA_VISIBLE_DEVICES=0 python run_helm.py \
  --input_path /home/wan.512/ECG_LLMs/KV_cache_opt/LLM_merge/data/xsum.jsonl \
  --output_path /home/wan.512/ECG_LLMs/KV_cache_opt/LLM_merge/results/generate_xsum_llama7b_full.jsonl \
  --model_name huggyllama/llama-7b \
  --model_arch llama 

#### local ##############
CUDA_VISIBLE_DEVICES=0 python run_helm.py \
  --input_path /home/wan.512/ECG_LLMs/KV_cache_opt/LLM_merge/data/xsum.jsonl \
  --output_path /home/wan.512/ECG_LLMs/KV_cache_opt/LLM_merge/results/generate_xsum_llama7b_local.jsonl \
  --model_name huggyllama/llama-7b \
  --model_arch llama 
  --enable_small_cache \
  --heavy_ratio 0 \
  --recent_ratio 0.2

#### h2o un_drop ##############
CUDA_VISIBLE_DEVICES=1 python run_helm.py \
  --input_path /home/wan.512/ECG_LLMs/KV_cache_opt/LLM_merge/data/xsum.jsonl \
  --output_path /home/wan.512/ECG_LLMs/KV_cache_opt/LLM_merge/results/generate_xsum_llama7b_un_drop.jsonl \
  --model_name huggyllama/llama-7b \
  --model_arch llama \
  --enable_small_cache \
  --heavy_ratio 0.1 \
  --recent_ratio 0.1

  
#### h2o real_drop ##############
CUDA_VISIBLE_DEVICES=2 python run_helm.py \
  --input_path /home/wan.512/ECG_LLMs/KV_cache_opt/LLM_merge/data/xsum.jsonl \
  --output_path /home/wan.512/ECG_LLMs/KV_cache_opt/LLM_merge/results/generate_xsum_llama7b_real_drop.jsonl \
  --model_name huggyllama/llama-7b \
  --model_arch llama 
  --use_real_drop
  --heavy_ratio 0.1 \
  --recent_ratio 0.1

#### h2o real drop ##############
CUDA_VISIBLE_DEVICES=3 python run_helm.py \
  --input_path /home/wan.512/ECG_LLMs/KV_cache_opt/LLM_merge/data/xsum.jsonl \
  --output_path /home/wan.512/ECG_LLMs/KV_cache_opt/LLM_merge/results/generate_xsum_llama7b_real_drop_merge.jsonl \
  --model_name huggyllama/llama-7b \
  --model_arch llama 
  --heavy_ratio 0.1 \
  --recent_ratio 0.1
  --use_real_merge


 # Step 3: Evaluate the performance of generated text (refer helm/command/eval.sh)
cd helm
TASK=xsum
JSONL=generate_xsum_llama7b.jsonl
OUTPUT=xsum_llama7b_result
ARCH=llama

python scripts/offline_eval/import_results.py together /home/wan.512/ECG_LLMs/KV_cache_opt/LLM_merge/results/generate_xsum_llama7b_full.jsonl --cache-dir prod_env/cache
helm-run --conf src/helm/benchmark/presentation/xsum/run_specs_llama.conf --local --max-eval-instances 100 --num-train-trials=1 --suite xsum_llama7b_result -n 1
helm-summarize --suite xsum_llama7b_result
# The results are writted into a tex file that can be found in benchmark_output/runs/xsum_llama7b_result/groups/latex/ 
"""