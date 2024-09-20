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
from transformers import MistralForCausalLM
from transformers import LlamaForCausalLM, AutoConfig

# from kv_token_merge.modify_llama import H2OLlamaAttention_drop, H2OLlamaForCausalLM_drop
# from kv_token_merge.modify_llama_merge import H2OLlamaAttention_merge, H2OLlamaForCausalLM_merge

# for LLama 1 and 2 
# from LMEval_kv_token_merge.modeling_llama import LlamaForCausalLM # for full
from transformers import LlamaForCausalLM
from LMEval_kv_token_merge.modeling_llama_drop import H2OLlamaAttention, H2OLlamaForCausalLM # for drop
from LMEval_kv_token_merge.modeling_llama_streaming import streaming_LlamaAttention, streaming_LlamaForCausalLM # for LLM streaming
from LMEval_kv_token_merge.modeling_llama_drop_merge import H2OLlamaAttention_merge, H2OLlamaForCausalLM_merge # for merge 

# for llama 3
from LMEval_kv_token_merge.modeling_llama3_7b_13b_drop import Llama3ForCausalLM_drop, Llama3Attention_drop
from LMEval_kv_token_merge.modeling_llama3_7b_13b_merge import Llama3ForCausalLM_merge, LlamaAttention3_merge
from LMEval_kv_token_merge.modeling_llama3 import Llama3Attention, Llama3ForCausalLM
from LMEval_kv_token_merge.modeling_llama3_streaming import Llama3Attention_streaming, Llama3ForCausalLM_streaming


# from LMEval_kv_token_merge.v436_modeling_mistral import MistralAttention, MistralForCausalLM
from transformers import MistralForCausalLM
from LMEval_kv_token_merge.v436_modeling_mistral_drop import MistralAttention_drop, MistralForCausalLM_drop
from LMEval_kv_token_merge.v436_modeling_mistral_merge import MistralAttention_merge, MistralForCausalLM_merge
from LMEval_kv_token_merge.v436_modeling_mistral_streaming import MistralAttention_streaming, MistralForCausalLM_streaming

# for falcon 
from LMEval_kv_token_merge.v436_modeling_falcon import FalconAttention, FalconForCausalLM
from LMEval_kv_token_merge.v436_modeling_falcon_drop import FalconAttention_drop, FalconForCausalLM_drop
from LMEval_kv_token_merge.v436_modeling_falcon_merge import FalconAttention_merge, FalconForCausalLM_merge
from LMEval_kv_token_merge.v436_modeling_falcon_streaming import FalconAttention_streaming, FalconForCausalLM_streaming
from transformers import FalconConfig

TAGET_MODULE_FOR_LLAMA_12 = {
    "llama": None,
    "real_drop": H2OLlamaAttention,
    "real_merge": H2OLlamaAttention_merge,  
    "real_stream": streaming_LlamaAttention
}

TAGET_MODULE_FOR_LLAMA_3 = {
    "llama3": None,
    "real_drop": Llama3Attention_drop,
    "real_merge": LlamaAttention3_merge,
    "real_stream": Llama3Attention_streaming
}

TAGET_MODULE_FOR_MISTRAL = {
    "mistral": None,
    "real_drop": MistralAttention_drop,
    "real_merge": MistralAttention_merge,
    "real_stream": MistralAttention_streaming
}


TAGET_MODULE_FOR_FALCON = {
    "llama": None,
    "real_drop": FalconAttention_drop,
    "real_merge": FalconAttention_merge,
    "real_stream": FalconAttention_streaming,
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

    use_real_streaming: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use our KV cache quantization implementation."},
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

        if model_args.model_type == 'llama':


            if model_args.use_real_drop:
                for name, m in model.named_modules():
                    if isinstance(m, TAGET_MODULE_FOR_LLAMA_12['real_drop']):
                        m._clean_cache()

            elif model_args.use_real_merge:
                for name, m in model.named_modules():
                    if isinstance(m, TAGET_MODULE_FOR_LLAMA_12['real_merge']):
                        # breakpoint()
                        m._clean_cache()

            elif model_args.use_real_streaming:
                for name, m in model.named_modules():
                    if isinstance(m, TAGET_MODULE_FOR_LLAMA_12['real_stream']):
                        # breakpoint()
                        m._clean_cache()

        elif model_args.model_type == 'llama3':

            if model_args.use_real_drop:
                for name, m in model.named_modules():
                    if isinstance(m, TAGET_MODULE_FOR_LLAMA_3['real_drop']):
                        m._clean_cache()

            elif model_args.use_real_merge:
                for name, m in model.named_modules():
                    if isinstance(m, TAGET_MODULE_FOR_LLAMA_3['real_merge']):
                        # breakpoint()
                        m._clean_cache()

            elif model_args.use_real_streaming:
                for name, m in model.named_modules():
                    if isinstance(m, TAGET_MODULE_FOR_LLAMA_3['real_stream']):
                        # breakpoint()
                        m._clean_cache()

        
        elif model_args.model_type == 'mistral':
    
            if model_args.use_real_drop:
                for name, m in model.named_modules():
                    if isinstance(m, TAGET_MODULE_FOR_MISTRAL['real_drop']):
                        
                        m._clean_cache()

            elif model_args.use_real_merge:
                for name, m in model.named_modules():
                    if isinstance(m, TAGET_MODULE_FOR_MISTRAL['real_merge']):
                        # breakpoint()
                        m._clean_cache()

            elif model_args.use_real_streaming:
                # breakpoint()
                for name, m in model.named_modules():
                    if isinstance(m, TAGET_MODULE_FOR_MISTRAL['real_stream']):
                        # breakpoint()
                        m._clean_cache()


        elif model_args.model_type == 'falcon':
        
            if model_args.use_real_drop:
                for name, m in model.named_modules():
                    if isinstance(m, TAGET_MODULE_FOR_FALCON['real_drop']):
                        m._clean_cache()

            elif model_args.use_real_merge:
                for name, m in model.named_modules():
                    if isinstance(m, TAGET_MODULE_FOR_FALCON['real_merge']):
                        # breakpoint()
                        m._clean_cache()

            elif model_args.use_real_streaming:
                for name, m in model.named_modules():
                    if isinstance(m, TAGET_MODULE_FOR_FALCON['real_stream']):
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
    elif 'mistral' in model_args.model_name_or_path.lower():
        config = MistralConfig.from_pretrained(model_args.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, 
                                            use_fast=False, 
                                            trust_remote_code=True)

    elif 'falcon' in model_args.model_name_or_path.lower():
        config = FalconConfig.from_pretrained(model_args.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, 
                                            use_fast=False, 
                                            trust_remote_code=True)

    else:
        raise NotImplementedError


    # breakpoint()


    
    if model_args.model_type == 'llama' :
       
        if model_args.use_full:

            model = LlamaForCausalLM.from_pretrained(
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

            model = H2OLlamaForCausalLM.from_pretrained(
                pretrained_model_name_or_path=model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                # use_flash_attention_2=True,
                device_map="auto",
            )

        elif model_args.use_real_merge:

            config.hh_size = model_args.hh_size
            config.recent_size = model_args.recent_size
            config.hh_ratio = model_args.hh_ratio
            config.recent_ratio = model_args.recent_ratio

            config.alpha = model_args.alpha
            config.belta = model_args.belta

            

            model = H2OLlamaForCausalLM_merge.from_pretrained(
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

            model = streaming_LlamaForCausalLM.from_pretrained(
                pretrained_model_name_or_path=model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                # use_flash_attention_2=True,
                device_map="auto",
            )


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

        elif model_args.use_real_merge:

            config.hh_size = model_args.hh_size
            config.recent_size = model_args.recent_size
            config.hh_ratio = model_args.hh_ratio
            config.recent_ratio = model_args.recent_ratio
            config.alpha = model_args.alpha
            config.belta = model_args.belta


            model = Llama3ForCausalLM_merge.from_pretrained(
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


    if model_args.model_type == 'mistral' :

        ##### continue to develop ########################################3


        if model_args.use_full:
    
            model = MistralForCausalLM.from_pretrained(
                pretrained_model_name_or_path=model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                load_in_8bit=True,
                # use_flash_attention_2=True,
                device_map="auto",
            )
        
        elif model_args.use_real_drop:

            config.hh_size = model_args.hh_size
            config.recent_size = model_args.recent_size
            config.hh_ratio = model_args.hh_ratio
            config.recent_ratio = model_args.recent_ratio

            # breakpoint()

            model = MistralForCausalLM_drop.from_pretrained(
                pretrained_model_name_or_path=model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                load_in_8bit=True,
                # use_flash_attention_2=True,
                device_map="auto",
            )

        elif model_args.use_real_merge:

            config.hh_size = model_args.hh_size
            config.recent_size = model_args.recent_size
            config.hh_ratio = model_args.hh_ratio
            config.recent_ratio = model_args.recent_ratio

            model = MistralForCausalLM_merge.from_pretrained(
                pretrained_model_name_or_path=model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                load_in_8bit=True,
                # use_flash_attention_2=True,
                device_map="auto",
            )

        elif model_args.use_real_streaming:

            config.hh_size = model_args.hh_size
            config.recent_size = model_args.recent_size
            config.hh_ratio = model_args.hh_ratio
            config.recent_ratio = model_args.recent_ratio

            model = MistralForCausalLM_streaming.from_pretrained(
                pretrained_model_name_or_path=model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                load_in_8bit=True,
                # use_flash_attention_2=True,
                device_map="auto",
            )

    if model_args.model_type == 'falcon' :

        if model_args.use_full:
        
            model = FalconForCausalLM.from_pretrained(
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

            model = FalconForCausalLM_drop.from_pretrained(
                pretrained_model_name_or_path=model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                # use_flash_attention_2=True,
                device_map="auto",
            )

        elif model_args.use_real_merge:

            config.hh_size = model_args.hh_size
            config.recent_size = model_args.recent_size
            config.hh_ratio = model_args.hh_ratio
            config.recent_ratio = model_args.recent_ratio

            model = FalconForCausalLM_merge.from_pretrained(
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

            model = FalconForCausalLM_streaming.from_pretrained(
                pretrained_model_name_or_path=model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                # use_flash_attention_2=True,
                device_map="auto",
            )



    # else:
    #     raise NotImplementedError

    # Load model directly
    # tokenizer = AutoTokenizer.from_pretrained("togethercomputer/LLaMA-2-7B-32K")
    # model = AutoModelForCausalLM.from_pretrained("togethercomputer/LLaMA-2-7B-32K")

    model.eval()
    max_length = model2maxlen[model_name]
    if data_args.e:

        # datasets = [ "narrativeqa","qasper", "multifieldqa_en", "hotpotqa", "2wikimqa","musique", "gov_report", "qmsum","multi_news", 
        #             "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]

        datasets = ["narrativeqa", "qmsum", "musique"]
        # datasets = ["musique"]
        # datasets = [ "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", 
        #             "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    else:
        datasets = ["triviaqa", "qasper", "trec", "samsum", "lcc", "repobench-p", "qmsum", "multi_news"]
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
new for the benchmark ##################################


######################################### llama3 #####################################

CUDA_VISIBLE_DEVICES=0 python run_pred_long_bench.py --model_name_or_path meta-llama/Meta-Llama-3-8B \
    --cache_dir /fs/scratch/PAS2473/zhongwei_models \
    --model_type llama3 \
    --use_full True \
    --action_name full_llama3 \
    --e True 

CUDA_VISIBLE_DEVICES=0 python run_pred_long_bench.py --model_name_or_path meta-llama/Meta-Llama-3-8B \
    --cache_dir /fs/scratch/PAS2473/zhongwei_models \
    --use_real_drop True \
    --model_type llama3 \
    --action_name drop_llama3_02 \
    --hh_ratio 0.1 \
    --recent_ratio 0.1 \
    --e True 

CUDA_VISIBLE_DEVICES=0 python run_pred_long_bench.py --model_name_or_path meta-llama/Meta-Llama-3-8B \
    --cache_dir /fs/scratch/PAS2473/zhongwei_models \
    --use_real_drop True \
    --model_type llama3 \
    --hh_ratio 0.0 \
    --recent_ratio 0.2 \
    --action_name local_llama3_02 \
    --e True 

CUDA_VISIBLE_DEVICES=0 python run_pred_long_bench.py --model_name_or_path meta-llama/Meta-Llama-3-8B \
    --cache_dir /fs/scratch/PAS2473/zhongwei_models \
    --model_type llama3 \
    --use_real_streaming True \
    --hh_ratio 0.01 \
    --recent_ratio 0.2 \
    --action_name stream_llama3_02 \
    --e True 



######################################### mistral #####################################

CUDA_VISIBLE_DEVICES=0 python run_pred_long_bench.py --model_name_or_path mistralai/Mistral-7B-Instruct-v0.2 \
    --cache_dir /fs/scratch/PAS2473/zhongwei_models \
    --model_type mistral \
    --use_full True \
    --action_name full_mistral \
    --e True 

CUDA_VISIBLE_DEVICES=0 python run_pred_long_bench.py --model_name_or_path mistralai/Mistral-7B-Instruct-v0.2 \
    --cache_dir /fs/scratch/PAS2473/zhongwei_models \
    --use_real_drop True \
    --model_type mistral \
    --action_name drop_mistral_02 \
    --hh_ratio 0.1 \
    --recent_ratio 0.1 \
    --e True 

CUDA_VISIBLE_DEVICES=0 python run_pred_long_bench.py --model_name_or_path mistralai/Mistral-7B-Instruct-v0.2 \
    --cache_dir /fs/scratch/PAS2473/zhongwei_models \
    --use_real_drop True \
    --model_type mistral \
    --hh_ratio 0.0 \
    --recent_ratio 0.2 \
    --action_name local_mistral_02 \
    --e True 

CUDA_VISIBLE_DEVICES=0 python run_pred_long_bench.py --model_name_or_path mistralai/Mistral-7B-Instruct-v0.2 \
    --cache_dir /fs/scratch/PAS2473/zhongwei_models \
    --model_type mistral \
    --use_real_streaming True \
    --hh_ratio 0.01 \
    --recent_ratio 0.2 \
    --action_name stream_mistral_02 \
    --e True 


######################################### falcon #####################################
    

CUDA_VISIBLE_DEVICES=0 python run_pred_long_bench.py --model_name_or_path tiiuae/falcon-7b-instruct \
    --cache_dir /fs/scratch/PAS2473/zhongwei_models \
    --model_type falcon \
    --use_full True \
    --action_name full_falcon \
    --e True 

CUDA_VISIBLE_DEVICES=1 python run_pred_long_bench.py --model_name_or_path tiiuae/falcon-7b-instruct \
    --cache_dir /fs/scratch/PAS2473/zhongwei_models \
    --use_real_drop True \
    --model_type falcon \
    --action_name drop_falcon_02 \
    --hh_ratio 0.1 \
    --recent_ratio 0.1 \
    --e True 

CUDA_VISIBLE_DEVICES=2 python run_pred_long_bench.py --model_name_or_path tiiuae/falcon-7b-instruct \
    --cache_dir /fs/scratch/PAS2473/zhongwei_models \
    --use_real_drop True \
    --model_type falcon \
    --hh_ratio 0.0 \
    --recent_ratio 0.2 \
    --action_name local_falcon_02 \
    --e True 

CUDA_VISIBLE_DEVICES=3 python run_pred_long_bench.py --model_name_or_path tiiuae/falcon-7b-instruct \
    --cache_dir /fs/scratch/PAS2473/zhongwei_models \
    --model_type falcon \
    --use_real_streaming True \
    --hh_ratio 0.0 \
    --recent_ratio 0.2 \
    --action_name stream_falcon_02 \
    --e True 



"""

 

"""

################### for full ####################

CUDA_VISIBLE_DEVICES=0 python run_pred_long_bench.py --model_name_or_path meta-llama/Llama-2-7b-hf \
    --cache_dir /fs/scratch/PAS2473/zhongwei_models \
    --use_full True \
    --action_name full \
    --e True 

##################### for drop #################

CUDA_VISIBLE_DEVICES=3 python run_pred_long_bench.py --model_name_or_path meta-llama/Llama-2-7b-hf \
    --cache_dir /fs/scratch/PAS2473/zhongwei_models \
    --use_real_drop True \
    --hh_ratio 0.2 \
    --recent_ratio 0.2 \
    --action_name drop \
    --e True 
 
###################### for merge ##################

CUDA_VISIBLE_DEVICES=0 python run_pred_long_bench.py --model_name_or_path meta-llama/Llama-2-7b-hf \
    --cache_dir /fs/scratch/PAS2473/zhongwei_models \
    --use_real_merge True \
    --hh_ratio 0.2 \
    --recent_ratio 0.2 \
    --action_name merge_0502_MA \
    --e True 

############ eval ############################################# 

python eval_long_bench.py --model {MODEL} # MODEL is the dir name under pred/ Currently it support Llama family model and Mistral model.


python eval_long_bench.py --model Llama-2-7b-hf_merge_0.2 --e 


# MODEL is the dir name under pred/ Currently it support Llama family model and Mistral model.


######################## new for osc falcon ###########################################


CUDA_VISIBLE_DEVICES=0 python run_pred_long_bench.py --model_name_or_path tiiuae/falcon-7b-instruct \
    --cache_dir /fs/scratch/PAS2473/zhongwei_models \
    --model_type falcon \
    --use_real_merge True \
    --hh_ratio 0.05 \
    --recent_ratio 0.15 \
    --action_name falcon_merge_005_015 \
    --e True 


CUDA_VISIBLE_DEVICES=1 python run_pred_long_bench.py --model_name_or_path tiiuae/falcon-7b-instruct \
    --cache_dir /fs/scratch/PAS2473/zhongwei_models \
    --model_type falcon \
    --use_real_merge True \
    --hh_ratio 0.1 \
    --recent_ratio 0.1 \
    --action_name falcon_merge_01_01 \
    --e True 

    
CUDA_VISIBLE_DEVICES=2 python run_pred_long_bench.py --model_name_or_path tiiuae/falcon-7b-instruct \
    --cache_dir /fs/scratch/PAS2473/zhongwei_models \
    --model_type falcon \
    --use_real_merge True \
    --hh_ratio 0.15 \
    --recent_ratio 0.05 \
    --action_name falcon_merge_015_005 \
    --e True 



"""