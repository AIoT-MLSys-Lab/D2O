
import os
from datasets import load_dataset
import torch
import json
from tqdm import tqdm
import numpy as np
import random
import argparse
os.environ["WANDB_DISABLED"] = "true"
from typing import Optional
from utils.process_args import process_args
from transformers import LlamaConfig, MistralConfig, AutoTokenizer
import os
from dataclasses import dataclass, field
import transformers

from transformers import AutoConfig


# for llama 3
from LMEval_kv_token_merge.modeling_llama3_7b_13b_drop import Llama3ForCausalLM_drop, Llama3Attention_drop
from LMEval_kv_token_merge.modeling_llama3_7b_13b_d2o import Llama3ForCausalLM_D2O, LlamaAttention3_D2O  # 
from LMEval_kv_token_merge.modeling_llama3_full import Llama3ForCausalLM
from LMEval_kv_token_merge.modeling_llama3_streaming import Llama3Attention_streaming, Llama3ForCausalLM_streaming


TAGET_MODULE_FOR_LLAMA_3 = {
    "llama3": None,
    "real_drop": Llama3Attention_drop,
    "real_d2o": LlamaAttention3_D2O,
    "real_stream": Llama3Attention_streaming
}



@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default=None, metadata={"help": "Output model local path, do not set manually"}
    )

    
    output_model_filename: Optional[str] = field(
        default="test-output", metadata={"help": "Output model relative manifold path"}
    )
    load_quant: Optional[str] = field(
        default=None,
        metadata={"help": "The path to a quantized model"},
    )
    w_bit: Optional[int] = field(
        default=4,
        metadata={"help": "The model weight bit width."},
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
    
    use_full: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use our d2o method."},
    )

    use_real_drop: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use our d2o method."},
    )

    use_d2o: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use our d2o method."},
    )

    use_real_streaming: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use our d2o method."},
    )

    model_type: Optional[str] = field(
        default='llama',
        metadata={"help": "The path to a quantized model"},
    )


    action_name: str = field(
        default='full', metadata={"help": "Output model local path, do not set manually"}
    )

    alpha: Optional[float] = field(
        default=0.5,
        metadata={"help": "recent ratio"},
    )

    belta: Optional[float] = field(
        default=0.5,
        metadata={"help": "recent ratio"},
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
        default='./outputs',
        metadata={"help": "The output path."},
    )
    e: Optional[bool] = field(
        default=False,
        metadata={"help": "Evaluate on LongBench-E."},
    )
    use_our_imp: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to use our KV cache quantization implementation."},
    )



@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default='/fs/scratch/PAS2473/zhongwei_models')
    optim: Optional[str] = field(default="adamw_torch")
    output_dir: Optional[str] = field(default="./outputs")
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


# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    # For results in KIVI paper (Llama, Llama-Chat, Mistral-7B-v0.1), we do not apply any special treatment to the prompt.
    # For lmsys/longchat-7b-v1.5-32k and mistralai/Mistral-7B-Instruct-v0.2, we need to rewrite the prompt a little bit.
    if "longchat" in model_name.lower():
        from fastchat.model import get_conversation_template
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif "mistral-v0.2-instruct" in model_name.lower():
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt

def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response

def get_pred(model, tokenizer, data, max_length, max_gen, prompt_format, dataset, device, model_name):
    preds = []
    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        # if "chatglm3" in model:
        #     tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, model_name)
        input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input.input_ids.shape[-1]
        if dataset == "samsum": # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length+1,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
            )[0]
        else:
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
            )[0]

        ################## modified here #############################
        
        json_obj["length"]

        if model_args.model_type == 'llama3':

            if model_args.use_real_drop:
                for name, m in model.named_modules():
                    if isinstance(m, TAGET_MODULE_FOR_LLAMA_3['real_drop']):
                        m._clean_cache()

            elif model_args.use_d2o:
                for name, m in model.named_modules():
                    if isinstance(m, TAGET_MODULE_FOR_LLAMA_3['real_d2o']):
                        # breakpoint()
                        m._clean_cache()


                 ##### clean and reset the dynamic cache size ####################

                num_layers = len(model.model.layers)

                for layer_idx in range(num_layers):

                    model.model.layers[layer_idx].self_attn.layer_hh_ratio = 0
                    model.model.layers[layer_idx].self_attn.layer_recent_ratio = 0

                    model.model.layers[layer_idx].self_attn.multimodal_entropy = 0

                    model.model.layers[layer_idx].self_attn.prfill_size = 0

                    model.model.layers[layer_idx].self_attn.prfill_score = None

            elif model_args.use_real_streaming:
                for name, m in model.named_modules():
                    if isinstance(m, TAGET_MODULE_FOR_LLAMA_3['real_stream']):
                        # breakpoint()
                        m._clean_cache()

    
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        pred = post_process(pred, model_name)
        preds.append({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]})
        
    return preds

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    seed_everything(42)
    # args = parse_args()
    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model_name = args.model

    # define your model
    model_args, data_args, training_args = process_args()
    # print(model_args, data_args, training_args)
    model_name = model_args.model_name_or_path.split("/")[-1]
    # dtype = torch.bfloat16 if training_args.bf16 else torch.float
    dtype = torch.float16
    
    if 'llama' in model_args.model_name_or_path.lower() or 'longchat' in model_args.model_name_or_path.lower():

        if model_args.model_type == 'llama3':

            config = AutoConfig.from_pretrained(model_args.model_name_or_path)
            tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, 
                                                use_fast=False, 
                                                trust_remote_code=True, 
                                               )
        else:
        

            config = LlamaConfig.from_pretrained(model_args.model_name_or_path)
            tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, 
                                                use_fast=False, 
                                                trust_remote_code=True, 
                                                tokenizer_type='llama')
                                            # model_max_length=training_args.model_max_length)

    if model_args.model_type == 'llama3' :
           
        if model_args.use_full:


            config.hh_size = model_args.hh_size
            config.recent_size = model_args.recent_size
            config.hh_ratio = model_args.hh_ratio
            config.recent_ratio = model_args.recent_ratio

            model = Llama3ForCausalLM.from_pretrained(
                pretrained_model_name_or_path=model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                # use_flash_attention_2=True,
                device_map="auto",
            )
        
        elif model_args.use_real_drop:

            config.hh_size = model_args.hh_size
            config.recent_size = model_args.recent_size
            config.hh_ratio = model_args.hh_ratio
            config.recent_ratio = model_args.recent_ratio

            config.alpha = model_args.alpha
            config.belta = model_args.belta
            
            
            model = Llama3ForCausalLM_drop.from_pretrained(
                pretrained_model_name_or_path=model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                # use_flash_attention_2=True,
                device_map="auto",
            )

        elif model_args.use_d2o:

            config.hh_size = model_args.hh_size
            config.recent_size = model_args.recent_size
            config.hh_ratio = model_args.hh_ratio
            config.recent_ratio = model_args.recent_ratio
            config.alpha = model_args.alpha
            config.belta = model_args.belta


            model = Llama3ForCausalLM_D2O.from_pretrained(
                pretrained_model_name_or_path=model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                # use_flash_attention_2=True,
                device_map="auto",
            )

        elif model_args.use_real_streaming:

            config.hh_size = model_args.hh_size
            config.recent_size = model_args.recent_size
            config.hh_ratio = model_args.hh_ratio
            config.recent_ratio = model_args.recent_ratio

            model = Llama3ForCausalLM_streaming.from_pretrained(
                pretrained_model_name_or_path=model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                # use_flash_attention_2=True,
                device_map="auto",
            )


    model.eval()
    max_length = model2maxlen[model_name]
    if data_args.e:
        datasets = ["triviaqa"]
    else:
        datasets = ["triviaqa"]
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    # predict on each dataset
    if not os.path.exists("pred"):
        os.makedirs("pred")
    if not os.path.exists("pred_e"):
        os.makedirs("pred_e")
    for dataset in datasets:
        if data_args.e:

            if dataset in ["narrativeqa","qmsum", "musique"]:
                
                data = load_dataset('THUDM/LongBench', f"{dataset}", split='test')

            else:
                
                data = load_dataset('THUDM/LongBench', f"{dataset}_e", split='test')

            if not os.path.exists(f"pred_e/{model_name}_{model_args.action_name}"):
                os.makedirs(f"pred_e/{model_name}_{model_args.action_name}")
            out_path = f"pred_e/{model_name}_{model_args.action_name}/{dataset}.jsonl"
        else:
            data = load_dataset('THUDM/LongBench', dataset, split='test')
            if not os.path.exists(f"pred/{model_name}_{model_args.action_name}"):
                os.makedirs(f"pred/{model_name}_{model_args.action_name}")
            out_path = f"pred/{model_name}_{model_args.action_name}/{dataset}.jsonl"
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        preds = get_pred(model, tokenizer, data, max_length, max_gen, prompt_format, dataset, device, model_name)
        with open(out_path, "w", encoding="utf-8") as f:
            for pred in preds:
                json.dump(pred, f, ensure_ascii=False)
                f.write('\n')



"""
### running sample #######################

CUDA_VISIBLE_DEVICES=0 python run_pred_long_bench_sample.py --model_name_or_path meta-llama/Meta-Llama-3-8B \
    --cache_dir /fs/scratch/PAS2473/zhongwei_new_models \
    --use_d2o True \
    --model_type llama3 \
    --hh_ratio 0.1 \
    --recent_ratio 0.1 \
    --action_name d2o_0.2 \
    --e True 

"""

 
