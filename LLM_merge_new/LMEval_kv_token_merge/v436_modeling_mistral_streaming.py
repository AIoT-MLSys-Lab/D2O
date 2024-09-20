# coding=utf-8
# Copyright 2023 Mistral AI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
""" PyTorch Mistral model."""
import inspect
import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
# from transformers.cache_utils import Cache, DynamicCache
# from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers.models.mistral.configuration_mistral import MistralConfig

from LMEval_kv_token_merge.v433_modeling_mistral import _make_sliding_window_causal_mask

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

    _flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "MistralConfig"


# Copied from transformers.models.llama.modeling_llama._get_unpad_data
def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->Mistral
class MistralRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        MistralRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


# Copied from transformers.models.llama.modeling_llama.LlamaRotaryEmbedding with Llama->Mistral
class MistralRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MistralMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def apply_rotary_pos_emb_single(x, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed

def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

def _prepare_decoder_attention_mask(
        attention_mask, input_shape, inputs_embeds, past_key_values_length, sliding_window
    ):
    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    combined_attention_mask = None
    if input_shape[-1] > 1:
        combined_attention_mask = _make_sliding_window_causal_mask(
            input_shape,
            inputs_embeds.dtype,
            device=inputs_embeds.device,
            past_key_values_length=past_key_values_length,
            sliding_window=sliding_window,
        )

    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
            inputs_embeds.device
        )
        combined_attention_mask = (
            expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
        )

    return combined_attention_mask


class H2OKVCache_LayerWise:
    def __init__(
        self,
        hh_size=4,
        recent_size=512,
        k_seq_dim=2,
        v_seq_dim=2,
        hh_ratio=None,
        recent_ratio=None
    ):
        # print(f"H2OKVCache-LayerWise: {hh_size}, {recent_size}")
        self.hh_size = hh_size
        self.recent_size = recent_size
        self.cache_size = hh_size + recent_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.hh_ratio = hh_ratio
        self.recent_ratio = recent_ratio
        self.hh_score = None
        self.sink_key = None
        self.sink_value = None 
        self.threshold = None
        print(f"H2OKVCache-LayerWise: {hh_ratio}, {recent_ratio}")

    def __call__(self, past_key_values, attn_score_cache):
        
        # start_time = time.time()
        if attn_score_cache.shape[-2]>1:
            self.recent_size = int(attn_score_cache.shape[-1] * self.recent_ratio)
            self.sink_key = past_key_values[0][:, :, :4, :]
            self.sink_value = past_key_values[1][:, :, :4, :]
        # breakpoint()
        k_recent = past_key_values[0][:, :, -self.recent_size:, :]
        v_recent = past_key_values[1][:, :, -self.recent_size:, :]
        
        k_recent = torch.cat((self.sink_key, k_recent), dim=-2)
        v_recent = torch.cat((self.sink_value, v_recent), dim=-2)
        
        # end_time = time.time()
        # print(end_time - start_time)
        # print(k_recent.shape)
        
        return (k_recent, v_recent)
        
    def _update_hh_score(self, attn_score_cache):

        ############## stop here and find the bug ##############
        
        num_new_tokens = attn_score_cache.shape[2]

        if self.hh_score is None:
            self.hh_score = attn_score_cache.sum(0).sum(1)
        else:
            # breakpoint()
            attn_score_cache = attn_score_cache.sum(0).sum(1)   
            attn_score_cache[..., :-num_new_tokens] = attn_score_cache[..., :-num_new_tokens] + self.hh_score # 18
            self.hh_score = attn_score_cache

    def _clean_scores(self):
        self.hh_score = None


class MistralAttention_streaming(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: MistralConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = MistralRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

        self.kv_cache = H2OKVCache_LayerWise(
            hh_size=self.config.hh_size,
            recent_size=self.config.recent_size,
            k_seq_dim=2,
            v_seq_dim=2,
            hh_ratio=self.config.hh_ratio,
            recent_ratio=self.config.recent_ratio
        )

    def _clean_cache(self):
        self.kv_cache._clean_scores()

    

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)


        # breakpoint()

        attention_mask = None


        # past_key_values_length = None

        inputs_embeds = query_states

        if past_key_value is None:
                past_key_values_length = 0
        else:
            past_key_values_length = past_key_value[0].shape[-2]

        # breakpoint()


        attention_mask = _prepare_decoder_attention_mask(
            attention_mask,
            (query_states.shape[0], query_states.shape[-2]),
            inputs_embeds,
            past_key_values_length,
            sliding_window=4096
        )

        kv_seq_len = key_states.shape[-2]


        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]


        position_length = kv_seq_len
        if not position_ids.nelement() > 1:
            if position_length < position_ids.item()+1:
                position_length = position_ids.item()+1

        cos, sin = self.rotary_emb(value_states, seq_len=position_length)
        ### Shift Pos: query pos is min(cache_size, idx)
        # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        query_states = apply_rotary_pos_emb_single(query_states, cos, sin, position_ids)
        key_states = apply_rotary_pos_emb_single(key_states, cos, sin, position_ids)



        # if past_key_value is not None:
        #     # reuse k, v, self_attention
        #     key_states = torch.cat([past_key_value[0], key_states], dim=2)
        #     value_states = torch.cat([past_key_value[1], value_states], dim=2)

        # past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
        past_key_value = self.kv_cache(past_key_value, attn_weights.detach().clone())

        # print('hidden state', hidden_states.shape)

        # print('attn_weights', attn_weights.shape)

        # print('past_key_value', past_key_value[0].shape)
        
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value




from LMEval_kv_token_merge.v433_modeling_mistral import MistralForCausalLM


class MistralForCausalLM_streaming(MistralForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        num_layers = len(self.model.layers)
        # for layer_idx in range(2, num_layers):
        for layer_idx in range(num_layers):
            self.model.layers[layer_idx].self_attn = MistralAttention_streaming(config)

import os
import transformers
from lm_eval.utils import MultiTokenEOSCriteria, stop_sequences_criteria
from lm_eval.models.huggingface import HFLM, eval_logger, _get_accelerate_args
from lm_eval import utils
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES,
)
from accelerate import Accelerator, find_executable_batch_size, DistributedType



class LMEvalMistralForCausalLM_drop(HFLM):
    AUTO_MODEL_CLASS = None
    _DEFAULT_MAX_LENGTH = 2048
    def __init__(
        self,
        ##############################
        TAGET_MODULE = None,
        use_real_drop: Optional[bool] = True,
        use_real_merge: Optional[bool] = True,
        hh_size: Optional[int] = None,
        recent_size: Optional[int]=None,
        hh_ratio: Optional[float]=None,
        recent_ratio: Optional[float]=None,
        ###############################
        pretrained: Optional[str] = "gpt2",
        revision: Optional[str] = "main",
        subfolder: Optional[str] = None,
        tokenizer: Optional[str] = None,
        truncation: Optional[bool] = False,
        max_length: Optional[int] = None,
        device: Optional[str] = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        max_batch_size: Optional[int] = 64,
        low_cpu_mem_usage: Optional[bool] = True,
        trust_remote_code: Optional[bool] = False,
        use_fast_tokenizer: Optional[bool] = True,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        # arguments used for splitting a model across GPUs naively.
        # only used if `parallelize=True`.
        parallelize: Optional[bool] = False,
        device_map_option: Optional[str] = "auto",
        max_memory_per_gpu: Optional[Union[int, str]] = None,
        max_cpu_memory: Optional[Union[int, str]] = None,
        offload_folder: Optional[str] = "./offload",
        load_in_8bit: Optional[bool] = False,
        load_in_4bit: Optional[bool] = False,
        bnb_4bit_quant_type: Optional[str] = None,
        bnb_4bit_compute_dtype: Optional[Union[str, torch.dtype]] = None,
        gptq: Optional[Union[bool, str]] = False,
        gptq_use_triton: Optional[bool] = False,
    ) -> None:
        super().__init__()

        assert isinstance(device, str)
        assert isinstance(pretrained, str)
        assert isinstance(batch_size, (int, str))

        gpus = torch.cuda.device_count()
        accelerator = Accelerator()

        if not (parallelize or accelerator.num_processes > 1):
            # use user-passed device
            device_list = set(
                ["cuda", "cpu"]
                + [f"cuda:{i}" for i in range(torch.cuda.device_count())]
                + ["mps", "mps:0"]
            )
            if device:
                if device not in device_list:
                    device = int(device)
                self._device = torch.device(device)
                eval_logger.info(f"Using device '{device}'")
                if device in ("mps", "mps:0") and "dev" not in torch.__version__:
                    eval_logger.info(
                        "MPS: Setting dtype to float32. To use float16 with MPS, please install a nightly build of "
                        "PyTorch: pip3 install --pre torch torchvision torchaudio --index-url "
                        "https://download.pytorch.org/whl/nightly/cpu"
                    )
            else:
                eval_logger.info("Device not specified")
                eval_logger.info(f"Cuda Available? {torch.cuda.is_available()}")
                self._device = (
                    torch.device("cuda")
                    if torch.cuda.is_available()
                    else torch.device("cpu")
                )
        else:
            if device != "cuda":
                eval_logger.info(
                    f"Using `accelerate launch` or `parallelize=True`, device '{device}' will be overridden when placing model."
                )
            # TODO: include in warning that `load_in_8bit` etc. affect this too
            self._device = device

        model_kwargs = {}
        if parallelize:
            model_kwargs = _get_accelerate_args(
                device_map_option,
                max_memory_per_gpu,
                max_cpu_memory,
                offload_folder,
            )

        # TODO: update this to be less of a hack once subfolder is fixed in HF
        revision = revision + ("/" + subfolder if subfolder is not None else "")

        self._config = transformers.AutoConfig.from_pretrained(
            pretrained,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )
        # self._config.k_bits = k_bits
        # self._config.v_bits = v_bits
        # self._config.group_size = group_size

        #### add config here for the drop method ##################

        self._config.hh_size = hh_size
        self._config.recent_size = recent_size
        self._config.hh_ratio = hh_ratio
        self._config.recent_ratio = recent_ratio


        self.use_real_drop = use_real_drop ########## new add here #########
        self.TAGET_MODULE = TAGET_MODULE


        self._config.attention_dropout = 0.0
        if not gptq:
            if load_in_4bit:
                assert (
                    transformers.__version__ >= "4.30.0"
                ), "load_in_4bit requires transformers >= 4.30.0"
            if transformers.__version__ >= "4.30.0":
                model_kwargs["load_in_4bit"] = load_in_4bit
                if load_in_4bit:
                    if bnb_4bit_quant_type:
                        model_kwargs["bnb_4bit_quant_type"] = bnb_4bit_quant_type
                    if bnb_4bit_compute_dtype:
                        model_kwargs["bnb_4bit_compute_dtype"] = utils.get_dtype(
                            bnb_4bit_compute_dtype
                        )

            # breakpoint()
            self._model = MistralForCausalLM_drop.from_pretrained(
                pretrained,
                cache_dir=cache_dir,
                revision=revision,
                config=self._config,
                torch_dtype=utils.get_dtype(dtype),
                low_cpu_mem_usage=low_cpu_mem_usage,
                trust_remote_code=trust_remote_code,
                load_in_8bit=load_in_8bit,
                **model_kwargs,
            )
        else:
            raise NotImplementedError
        # forever after, access self._model through self.model property
        self.model.eval()
        self.model.tie_weights()
        if gpus <= 1 and not parallelize:
            # place model onto device, if not using HF Accelerate in any form
            try:
                self.model.to(self.device)
            except ValueError:
                eval_logger.info(
                    "Failed to place model onto specified device. This may be because the model is quantized via `bitsandbytes`. If the desired GPU is being used, this message is safe to ignore."
                )

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained if tokenizer is None else tokenizer,
            revision=revision,
            trust_remote_code=trust_remote_code,
            use_fast=use_fast_tokenizer,
        )

        self.truncation = truncation

        self.use_real_merge = use_real_merge

        self.vocab_size = self.tokenizer.vocab_size
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self._max_length = max_length

        self.batch_schedule = 1
        self.batch_sizes = {}
        self.max_batch_size = max_batch_size

        if str(batch_size).startswith("auto"):
            batch_size = batch_size.split(":")
            self.batch_size_per_gpu = batch_size[0]
            self.batch_schedule = float(batch_size[1]) if len(batch_size) > 1 else 1
        else:
            self.batch_size_per_gpu = int(batch_size)

        # multigpu data-parallel support when launched with accelerate
        if gpus > 1:
            if parallelize:
                if accelerator.num_processes > 1:
                    raise RuntimeError(
                        "Attempted to use both a HF Accelerate `device_map` and to launch via `accelerate launch`. If this is the case, please either remove `parallelize=True` from --model_args or launch outside of the Accelerate launcher."
                    )
                else:
                    pass
            elif gpus > accelerator.num_processes:
                # TODO: make sure there's still never an edge case where we unintentionally default to CPU
                eval_logger.warning(
                    "WARNING: The number of total system GPUs does not match the number of spawned processes. "
                    "If you would like to use data parallelism, please launch the script "
                    "with 'accelerate launch *script*'. "
                    f"Current run will proceed with {accelerator.num_processes} devices."
                )
                self._rank = accelerator.local_process_index
                self._world_size = accelerator.num_processes
                # manually set model to use gpu, for case where many GPUs available but
                # only seek to use one
                self._device = (
                    torch.device(f"cuda:{accelerator.local_process_index}")
                    if torch.cuda.is_available()
                    else torch.device("cpu")
                )
                try:
                    self.model.to(self.device)
                except ValueError:
                    eval_logger.info(
                        "Failed to place model onto specified device. This may be because the model is quantized via `bitsandbytes`. If the desired GPU is being used, this message is safe to ignore."
                    )
            else:
                assert accelerator.distributed_type in [
                    DistributedType.FSDP,
                    DistributedType.MULTI_GPU,
                ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
                if accelerator.distributed_type == DistributedType.FSDP:
                    self._model = accelerator.prepare(self.model)
                else:
                    self._model = accelerator.prepare_model(
                        self.model, evaluation_mode=True
                    )
                self._device = torch.device(f"cuda:{accelerator.local_process_index}")
                self.accelerator = accelerator

                if self.accelerator.is_local_main_process:
                    eval_logger.info(f"Using {gpus} devices with data parallelism")

                self._rank = self.accelerator.local_process_index
                self._world_size = self.accelerator.num_processes