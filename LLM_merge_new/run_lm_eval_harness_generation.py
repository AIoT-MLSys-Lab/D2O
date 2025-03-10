import os
os.environ["WANDB_DISABLED"] = "true"
os.environ["HUGGING_FACE_HUB_TOKEN"] = "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"
import argparse
import json, tqdm
import torch
import copy

import math
import time
from lm_eval import evaluator, utils
from lm_eval.tasks import initialize_tasks, include_path
from lm_eval.api.registry import ALL_TASKS

# from utils.process_args import process_args
from transformers import LlamaConfig, AutoTokenizer, FalconConfig, MistralConfig
from utils.data import set_seed
from datasets import load_dataset
import os
from dataclasses import dataclass, field
from typing import Optional

import transformers
from LMEval_kv_token_merge.modeling_llama import LMEvalLlamaForCausalLM
# from LMEval_kv_token_merge.modeling_llama_drop import LMEvalLlamaForCausalLM_drop, H2OLlamaAttention
from LMEval_kv_token_merge.modeling_llama3_7b_13b_drop import LMEvalLlamaForCausalLM_drop, H2OLlamaAttention
# from LMEval_kv_token_merge.modeling_llama3_70b_drop import LMEvalLlamaForCausalLM_drop, H2OLlamaAttention
from LMEval_kv_token_merge.modeling_llama_drop_merge import LMEvalLlamaForCausalLM_merge, H2OLlamaAttention_merge

from LLM_merge_new.LMEval_kv_token_merge.v436_modeling_falcon import LMEvalFalconForCausalLM, FalconConfig

from accelerate import Accelerator
accelerator = Accelerator()

######################### process args #############################################

TAGET_MODULE = {
    "llama": None,
    "real_drop": H2OLlamaAttention,
    "real_merge": H2OLlamaAttention_merge,
}

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default=None, metadata={"help": "Output model local path, do not set manually"}
    )

    output_model_filename: Optional[str] = field(
        default="test-output", metadata={"help": "Output model relative manifold path"}
    )
    lora: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use LoRA"},
    )
    lora_mode: Optional[str] = field(
        default="q",
        metadata={"help": "LoRA mode"},
    )
    lora_r: Optional[int] = field(
        default=1,
        metadata={"help": "LoRA r"},
    )
    lora_alpha: Optional[float] = field(
        default=1.,
        metadata={"help": "LoRA alpha"},
    )
    lora_dropout: Optional[float] = field(
        default=0.,
        metadata={"help": "LoRA dropout"},
    )

    hh_ratio: Optional[float] = field(
        default=0.1,
        metadata={"help": "important ratio"},
    )

    recent_ratio: Optional[float] = field(
        default=0.1,
        metadata={"help": "recent ratio"},
    )

    hh_size: Optional[int] = field(
        default=200,
        metadata={"help": "important windows size"},
    )

    recent_size: Optional[int] = field(
        default=200,
        metadata={"help": "recent window size"},
    )
    use_our_imp: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to use our KV cache quantization implementation."},
    )

    use_full: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use our KV cache quantization implementation."},
    )

    use_real_drop: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use our KV cache quantization implementation."},
    )

    use_real_merge: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use our KV cache quantization implementation."},
    )

    model_type: Optional[str] = field(
        default="llama",
        metadata={"help": "lmodel_typelama"},
    )



@dataclass
class DataArguments:
    dataset: Optional[str] = field(
        default='c4',
        metadata={"help": "The dataset used for fine-tuning the model."},
    )
    eval_tasks: Optional[str] = field(
        default='wikitext',
        metadata={"help": "The dataset used for evaluation."},
    )
    tasks: Optional[str] = field(
        default='wikitext',
        metadata={"help": "The dataset used for evaluation."},
    )
    batch_size: Optional[int] = field(
        default=1,
        metadata={"help": "The batch size."},
    )
    num_fewshot: Optional[int] = field(
        default=0,
        metadata={"help": "The number of fewshot examples."},
    )
    output_path: Optional[str] = field(
        default='/users/PAS2473/brucewan666/Efficient_LLMs/KV_cache_opt/LLM_merge/results',
        metadata={"help": "The output path."},
    )
    e: Optional[bool] = field(
        default=False,
        metadata={"help": "Evaluate on LongBench-E."},
    )



@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default='/fs/scratch/PAS2473/zhongwei_models')
    optim: Optional[str] = field(default="adamw_torch")
    output_dir: Optional[str] = field(default="/users/PAS2473/brucewan666/Efficient_LLMs/KV_cache_opt/LLM_merge/results")
    model_max_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated). 512 or 1024"
        },
    )
    num_train_epochs: Optional[int] = field(default=1)
    n_train_samples: Optional[int] = field(default=None)
    n_eval_samples: Optional[int] = field(default=None)
    qat: Optional[bool] = field(default=False)
    exp_name: Optional[str] = field(default="test")


def process_args():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    os.makedirs(training_args.output_dir, exist_ok=True)

    model_args.output_model_local_path = os.path.join(
        training_args.output_dir, "models", str(model_args.output_model_filename)
    )

    return model_args, data_args, training_args


if __name__ == '__main__':

    set_seed(42)

    model_args, data_args, training_args = process_args()


    dtype = torch.float16
    if 'llama' in model_args.model_name_or_path.lower():
        print('model name ', model_args.model_name_or_path)

        
        config = LlamaConfig.from_pretrained(model_args.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, 
                                            use_fast=False, 
                                            trust_remote_code=True, 
                                            # tokenizer_type='llama',
                                            model_max_length=training_args.model_max_length)

    elif model_args.model_type == 'falcon':

        config = FalconConfig.from_pretrained(model_args.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, 
                                            use_fast=False, 
                                            trust_remote_code=True, 
                                            # tokenizer_type='llama',
                                            model_max_length=training_args.model_max_length)

    
    else:
        raise NotImplementedError

    if torch.cuda.device_count() > 1:
        parallel = True
        low_cpu_mem_usage=True
    else:
        parallel = False
        low_cpu_mem_usage=True
    
    if model_args.use_full:
        use_real_drop = model_args.use_real_drop


        if model_args.model_type == 'llama':

            model = LMEvalLlamaForCausalLM(
                use_real_drop=use_real_drop,
                pretrained=model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                dtype=dtype,
                low_cpu_mem_usage=low_cpu_mem_usage,
            )


        elif model_args.model_type == 'falcon':

            model = LMEvalFalconForCausalLM(
                use_real_drop=use_real_drop,
                pretrained=model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                dtype=dtype,
                low_cpu_mem_usage=low_cpu_mem_usage,
            )


        
    elif model_args.use_real_drop:

        print('Enabling H2O KV cache')
        hh_size = model_args.hh_size
        recent_size = model_args.recent_size
        hh_ratio = model_args.hh_ratio
        recent_ratio = model_args.recent_ratio
        use_real_drop = model_args.use_real_drop

        # breakpoint()

        model = LMEvalLlamaForCausalLM_drop(
            TAGET_MODULE = TAGET_MODULE,
            use_real_drop=use_real_drop,
            hh_size=hh_size,
            recent_size=recent_size,
            hh_ratio=hh_ratio,
            recent_ratio=recent_ratio,
            pretrained=model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            dtype=dtype,
            low_cpu_mem_usage=low_cpu_mem_usage,
        )

        # model = ENABLE_Heavy_Hitter_FUNCTIONS['llama_h2o'].from_pretrained(model_name, config=config,
        #                                                                     cache_dir=args.cache_dir)

        print('use_real_drop ################ ')

    elif model_args.use_real_merge:

        print('Enabling H2O KV cache')
        hh_size = model_args.hh_size
        recent_size = model_args.recent_size
        hh_ratio = model_args.hh_ratio
        recent_ratio = model_args.recent_ratio
        use_real_drop = model_args.use_real_drop

        # breakpoint()

        model = LMEvalLlamaForCausalLM_merge(
            TAGET_MODULE = TAGET_MODULE,
            use_real_merge=model_args.use_real_merge,
            hh_size=hh_size,
            recent_size=recent_size,
            hh_ratio=hh_ratio,
            recent_ratio=recent_ratio,
            pretrained=model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            dtype=dtype,
            low_cpu_mem_usage=low_cpu_mem_usage,
        )

        print('use_real_drop_merge ################ ')

    else:
        raise NotImplementedError
    # model = model.eval().cuda()

    if data_args.tasks is not None:
        initialize_tasks()
        tasks_list = data_args.tasks.split(",")
        task_names = utils.pattern_match(tasks_list, ALL_TASKS)
        # breakpoint()
        for task in [task for task in tasks_list if task not in task_names]:
            if os.path.isfile(task):
                config = utils.load_yaml_config(task)
                task_names.append(config)
        task_missing = [
            task
            for task in tasks_list
            if task not in task_names and "*" not in task
        ]  # we don't want errors if a wildcard ("*") task name was used

        if task_missing:
            missing = ", ".join(task_missing)
            raise ValueError(
                f"Tasks {missing} were not found. Try `lm-eval --tasks list` for list of available tasks."
            )
        results = evaluator.simple_evaluate(
            model=model,
            # model_args='parallelize=True',
            tasks=task_names,
            log_samples=True
            # no_cache=True,
            # num_fewshot=data_args.num_fewshot,
        )
        print(evaluator.make_table(results))
        # samples = results["samples"]
        # filepath = f"./output_samples/{training_args.exp_name}.json"
        # with open(filepath, "w") as f:
        #     json.dump(samples, f)
        # if data_args.output_path is not None:
        #     os.makedirs(os.path.dirname(data_args.output_path), exist_ok=True)
        #     # otherwise cannot save
        #     results["config"]["model"] = model_args.model_name_or_path
        #     with open(data_args.output_path, "w") as f:
        #         json.dump(results, f, indent=2)

"""

############ for full ##################################
            
cd lm-evaluation-harness a
pip install -e .
cd ..

CUDA_VISIBLE_DEVICES=0 python run_lm_eval_harness_generation.py --model_name_or_path meta-llama/Meta-Llama-3-8B \
            --tasks gsm8k \
            --cache_dir /fs/scratch/PAS2473/zhongwei_models \
            --use_full True \
            --model_type falcon

CUDA_VISIBLE_DEVICES=1 python run_lm_eval_harness_generation.py --model_name_or_path meta-llama/Meta-Llama-3-8B \
            --tasks gsm8k \
            --cache_dir /fs/scratch/PAS2473/zhongwei_models \
            --use_real_drop True \
            --hh_size 200 \
            --recent_size 200


|Tasks|Version|  Filter  |n-shot|  Metric   |Value |   |Stderr|
|-----|-------|----------|-----:|-----------|-----:|---|-----:|
|gsm8k|Yaml   |get-answer|     5|exact_match|0.1342|±  |0.0094|

CUDA_VISIBLE_DEVICES=0 python run_lm_eval_harness_generation.py --model_name_or_path meta-llama/Llama-2-7b-hf \
            --tasks coqa \
            --cache_dir /fs/scratch/PAS2473/zhongwei_models \
            --use_full True

seq_length_avg: 698.38

|Tasks|Version|Filter|n-shot|Metric|Value |   |Stderr|                                                
|-----|-------|------|-----:|------|-----:|---|-----:|                                                
|coqa |Yaml   |none  |     0|em    |0.6388|±  |0.0183|                       
|     |       |none  |     0|f1    |0.7735|±  |0.0141|

CUDA_VISIBLE_DEVICES=0 python run_lm_eval_harness_generation.py --model_name_or_path meta-llama/Llama-2-7b-hf \
            --tasks truthfulqa_gen \
            --cache_dir /fs/scratch/PAS2473/zhongwei_models \
            --use_full True



|    Tasks     |Version|Filter|n-shot|  Metric   | Value |   |Stderr|                                 
|--------------|-------|------|-----:|-----------|------:|---|-----:|                                 
|truthfulqa_gen|Yaml   |none  |     0|bleu_max   |30.7416|±  |0.8283|                                 
|              |       |none  |     0|bleu_acc   | 0.3427|±  |0.0166|                                 
|              |       |none  |     0|bleu_diff  |-6.0858|±  |0.9610|                                 
|              |       |none  |     0|rouge1_max |56.4043|±  |0.8541|                                 
|              |       |none  |     0|rouge1_acc | 0.3207|±  |0.0163|                                 
|              |       |none  |     0|rouge1_diff|-7.1784|±  |1.0577|                                 
|              |       |none  |     0|rouge2_max |42.0232|±  |1.0192|                                 
|              |       |none  |     0|rouge2_acc | 0.3048|±  |0.0161|                                 
|              |       |none  |     0|rouge2_diff|-8.4101|±  |1.2498|                                 
|              |       |none  |     0|rougeL_max |53.5556|±  |0.8818|                                 
|              |       |none  |     0|rougeL_acc | 0.3231|±  |0.0164|                                 
|              |       |none  |     0|rougeL_diff|-7.3682|±  |1.0668|

############# for real_drop ###################################

0.2--242 0.4--484 0.6--726 0.8--968

CUDA_VISIBLE_DEVICES=0 python run_lm_eval_harness_generation.py --model_name_or_path meta-llama/Llama-2-7b-hf \
            --tasks gsm8k \
            --cache_dir /fs/scratch/PAS2473/zhongwei_models \
            --use_real_drop True \
            --hh_size 484 \
            --recent_size 484

|Tasks|Version|  Filter  |n-shot|  Metric   |Value |   |Stderr|                  
|-----|-------|----------|-----:|-----------|-----:|---|-----:|                           
|gsm8k|Yaml   |get-answer|     5|exact_match|0.1357|±  |0.0094|        

CUDA_VISIBLE_DEVICES=0 python run_lm_eval_harness_generation.py --model_name_or_path meta-llama/Llama-2-7b-hf \
            --tasks coqa \
            --cache_dir /fs/scratch/PAS2473/zhongwei_models \
            --use_real_drop True \
            --hh_size 40 \
            --recent_size 600

0.2--140 0.4--280 0.6--420 0.8--640
 
|Tasks|Version|Filter|n-shot|Metric|Value |   |Stderr|
|-----|-------|------|-----:|------|-----:|---|-----:|
|coqa |Yaml   |none  |     0|em    |0.5657|±  |0.0195|
|     |       |none  |     0|f1    |0.7256|±  |0.0152|

# (skip first 2 layer)
|Tasks|Version|Filter|n-shot|Metric|Value |   |Stderr|
|-----|-------|------|-----:|------|-----:|---|-----:|
|coqa |Yaml   |none  |     0|em    |0.5667|±  |0.0195|
|     |       |none  |     0|f1    |0.7268|±  |0.0151|

# 注意到truthfulqa_gen数据集下文本长度仅在200+这个范围
CUDA_VISIBLE_DEVICES=2 python run_lm_eval_harness_generation.py --model_name_or_path meta-llama/Llama-2-7b-hf \
            --tasks truthfulqa_gen \
            --cache_dir /fs/scratch/PAS2473/zhongwei_models \
            --use_real_drop True \
            --hh_size 10 \
            --recent_size 10  

|    Tasks     |Version|Filter|n-shot|  Metric   | Value  |   |Stderr|
|--------------|-------|------|-----:|-----------|-------:|---|-----:|
|truthfulqa_gen|Yaml   |none  |     0|bleu_max   | 17.7137|±  |0.6045|
|              |       |none  |     0|bleu_acc   |  0.2962|±  |0.0160|
|              |       |none  |     0|bleu_diff  | -6.4486|±  |0.6589|
|              |       |none  |     0|rouge1_max | 42.6721|±  |0.8069|
|              |       |none  |     0|rouge1_acc |  0.2766|±  |0.0157|
|              |       |none  |     0|rouge1_diff|-10.9684|±  |0.7620|
|              |       |none  |     0|rouge2_max | 24.9938|±  |0.8602|
|              |       |none  |     0|rouge2_acc |  0.2350|±  |0.0148|
|              |       |none  |     0|rouge2_diff|-11.6498|±  |0.9127|
|              |       |none  |     0|rougeL_max | 39.7447|±  |0.8010|
|              |       |none  |     0|rougeL_acc |  0.2791|±  |0.0157|
|              |       |none  |     0|rougeL_diff|-10.7935|±  |0.7638|

############# for real_merge ###################################

CUDA_VISIBLE_DEVICES=0 python run_lm_eval_harness_generation.py --model_name_or_path meta-llama/Llama-2-7b-hf \
        --tasks gsm8k \
        --cache_dir /fs/scratch/PAS2473/zhongwei_models \
        --use_real_merge True \
        --hh_size 484 \
        --recent_size 484 

|Tasks|Version|  Filter  |n-shot|  Metric   |Value |   |Stderr|                  
|-----|-------|----------|-----:|-----------|-----:|---|-----:|                           
|gsm8k|Yaml   |get-answer|     5|exact_match|0.1236|±  |0.0091|


|Tasks|Version|  Filter  |n-shot|  Metric   |Value |   |Stderr|
|-----|-------|----------|-----:|-----------|-----:|---|-----:|
|gsm8k|Yaml   |get-answer|     5|exact_match|0.1145|±  |0.0088|


|Tasks|Version|  Filter  |n-shot|  Metric   |Value |   |Stderr|
|-----|-------|----------|-----:|-----------|-----:|---|-----:|
|gsm8k|Yaml   |get-answer|     5|exact_match|0.1122|±  |0.0087|

|Tasks|Version|  Filter  |n-shot|  Metric   |Value |   |Stderr|
|-----|-------|----------|-----:|-----------|-----:|---|-----:|
|gsm8k|Yaml   |get-answer|     5|exact_match|0.1221|±  | 0.009|


CUDA_VISIBLE_DEVICES=0 python run_lm_eval_harness_generation.py --model_name_or_path meta-llama/Llama-2-7b-hf \
        --tasks coqa \
        --cache_dir /fs/scratch/PAS2473/zhongwei_models \
        --use_real_merge True \
        --hh_size 200 \
        --recent_size 200

pivot merge:
|Tasks|Version|Filter|n-shot|Metric|Value |   |Stderr|
|-----|-------|------|-----:|------|-----:|---|-----:|
|coqa |Yaml   |none  |     0|em    |0.5607|±  |0.0197|
|     |       |none  |     0|f1    |0.7160|±  |0.0155|

pivot merge pro (set include_self=True):
|Tasks|Version|Filter|n-shot|Metric|Value |   |Stderr|
|-----|-------|------|-----:|------|-----:|---|-----:|
|coqa |Yaml   |none  |     0|em    |0.5572|±  |0.0196|
|     |       |none  |     0|f1    |0.7171|±  |0.0154|

avg merge:
|Tasks|Version|Filter|n-shot|Metric|Value |   |Stderr|
|-----|-------|------|-----:|------|-----:|---|-----:|
|coqa |Yaml   |none  |     0|em    |0.5642|±  |0.0196|
|     |       |none  |     0|f1    |0.7165|±  |0.0155|

avg merge (skip first 2 layer)
|Tasks|Version|Filter|n-shot|Metric|Value |   |Stderr|
|-----|-------|------|-----:|------|-----:|---|-----:|
|coqa |Yaml   |none  |     0|em    |0.5652|±  |0.0196|
|     |       |none  |     0|f1    |0.7164|±  |0.0155|

weighted avg (skip first 2 layer)
|Tasks|Version|Filter|n-shot|Metric|Value |   |Stderr|
|-----|-------|------|-----:|------|-----:|---|-----:|
|coqa |Yaml   |none  |     0|em    |0.5602|±  |0.0196|
|     |       |none  |     0|f1    |0.7194|±  |0.0152|

filter & weighted avg (skip first 2 layer)
    |Tasks|Version|Filter|n-shot|Metric|Value |   |Stderr|
    |-----|-------|------|-----:|------|-----:|---|-----:|
    |coqa |Yaml   |none  |     0|em    |0.5662|±  |0.0195|
    |     |       |none  |     0|f1    |0.7288|±  |0.0150|

MA & filter & weighted avg (skip first 2 layer)
|Tasks|Version|Filter|n-shot|Metric|Value |   |Stderr|
|-----|-------|------|-----:|------|-----:|---|-----:|
|coqa |Yaml   |none  |     0|em    |0.5667|±  |0.0195|
|     |       |none  |     0|f1    |0.7271|±  |0.0151|    

CUDA_VISIBLE_DEVICES=0 python run_lm_eval_harness_generation.py --model_name_or_path meta-llama/Llama-2-7b-hf \
        --tasks truthfulqa_gen \
        --cache_dir /fs/scratch/PAS2473/zhongwei_models \
        --use_real_merge True \
        --hh_size 10 \
        --recent_size 10

|    Tasks     |Version|Filter|n-shot|  Metric   | Value  |   |Stderr|
|--------------|-------|------|-----:|-----------|-------:|---|-----:|
|truthfulqa_gen|Yaml   |none  |     0|bleu_max   | 17.7137|±  |0.6045|
|              |       |none  |     0|bleu_acc   |  0.2962|±  |0.0160|
|              |       |none  |     0|bleu_diff  | -6.4486|±  |0.6589|
|              |       |none  |     0|rouge1_max | 42.6721|±  |0.8069|
|              |       |none  |     0|rouge1_acc |  0.2766|±  |0.0157|
|              |       |none  |     0|rouge1_diff|-10.9684|±  |0.7620|
|              |       |none  |     0|rouge2_max | 24.9938|±  |0.8602|
|              |       |none  |     0|rouge2_acc |  0.2350|±  |0.0148|
|              |       |none  |     0|rouge2_diff|-11.6498|±  |0.9127|
|              |       |none  |     0|rougeL_max | 39.7447|±  |0.8010|
|              |       |none  |     0|rougeL_acc |  0.2791|±  |0.0157|
|              |       |none  |     0|rougeL_diff|-10.7935|±  |0.7638|

"""