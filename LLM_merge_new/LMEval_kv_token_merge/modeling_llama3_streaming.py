import math
from typing import Optional, Tuple

import pdb
import torch
from torch import nn
import torch.utils.checkpoint

import torch.nn.functional as F
from typing import List, Optional, Tuple, Union

# from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    rotate_half,
    apply_rotary_pos_emb,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
)

# from LMEval_kv_token_merge.modeling_llama import LlamaForCausalLM

from LMEval_kv_token_merge.v433_modeling_llama import LlamaForCausalLM

import types
from LMEval_kv_token_merge.v433_modeling_llama import LlamaLinearScalingRotaryEmbedding, LlamaDynamicNTKScalingRotaryEmbedding, LlamaRotaryEmbedding

__all__ = ["H2OLlamaForCausalLM", "H2OLlamaAttention",
            'H2OLlamaAttention_streaming', 'H2OLlamaForCausalLM_streaming']


from transformers.configuration_utils import PretrainedConfig
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





LLAMA_PRETRAINED_CONFIG_ARCHIVE_MAP = {}

class LlamaConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`LlamaModel`]. It is used to instantiate an LLaMA
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the LLaMA-7B.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the LLaMA model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`LlamaModel`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
            `num_attention_heads`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. Llama 1 supports up to 2048 tokens,
            Llama 2 up to 4096, CodeLlama up to 16384.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 1):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 2):
            End of stream token id.
        pretraining_tp (`int`, *optional*, defaults to 1):
            Experimental feature. Tensor parallelism rank used during pretraining. Please refer to [this
            document](https://huggingface.co/docs/transformers/parallelism) to understand more about it. This value is
            necessary to ensure exact reproducibility of the pretraining results. Please refer to [this
            issue](https://github.com/pytorch/pytorch/issues/76232).
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. Currently supports two scaling
            strategies: linear and dynamic. Their scaling factor must be a float greater than 1. The expected format is
            `{"type": strategy name, "factor": scaling factor}`. When using this flag, don't update
            `max_position_embeddings` to the expected new maximum. See the following thread for more information on how
            these scaling strategies behave:
            https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/. This is an
            experimental feature, subject to breaking API changes in future versions.
        attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.

    ```python
    >>> from transformers import LlamaModel, LlamaConfig

    >>> # Initializing a LLaMA llama-7b style configuration
    >>> configuration = LlamaConfig()

    >>> # Initializing a model from the llama-7b style configuration
    >>> model = LlamaModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "llama"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self._rope_scaling_validation()
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def _rope_scaling_validation(self):
        """
        Validate the `rope_scaling` configuration.
        """
        if self.rope_scaling is None:
            return

        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 2:
            raise ValueError(
                "`rope_scaling` must be a dictionary with with two fields, `type` and `factor`, "
                f"got {self.rope_scaling}"
            )
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_factor = self.rope_scaling.get("factor", None)
        if rope_scaling_type is None or rope_scaling_type not in ["linear", "dynamic"]:
            raise ValueError(
                f"`rope_scaling`'s type field must be one of ['linear', 'dynamic'], got {rope_scaling_type}"
            )
        if rope_scaling_factor is None or not isinstance(rope_scaling_factor, float) or rope_scaling_factor <= 1.0:
            raise ValueError(f"`rope_scaling`'s factor field must be a float > 1, got {rope_scaling_factor}")



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


# the same as the original code 
def _make_causal_mask(
    bsz: int, tgt_len: int, past_key_values_length: int, dtype: torch.dtype, device: torch.device):
    """
    Make causal mask used for bi-directional self-attention.
    """
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def apply_rotary_pos_emb_single(x, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed


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


class Llama3Attention_streaming(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig):
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
        self._init_rope()

        self.kv_cache = H2OKVCache_LayerWise(
            hh_size=config.hh_size,
            recent_size=config.recent_size,
            k_seq_dim=2,
            v_seq_dim=2,
            hh_ratio=config.hh_ratio,
            recent_ratio=config.recent_ratio
        )

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _clean_cache(self):
        self.kv_cache._clean_scores()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.size()

        # print('hidden_states', hidden_states.shape)

        if self.config.pretraining_tp > 1:
            key_value_slicing = (
                self.num_key_value_heads * self.head_dim
            ) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [
                F.linear(hidden_states, query_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [
                F.linear(hidden_states, key_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [
                F.linear(hidden_states, value_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        # remake causal mask


        # breakpoint()


        if past_key_value is None:
            past_key_values_length = 0
        else:
            past_key_values_length = past_key_value[0].shape[-2]

        attention_mask = _make_causal_mask(
            bsz=bsz,
            tgt_len=q_len,
            past_key_values_length=past_key_values_length,
            dtype=query_states.dtype,
            device=query_states.device,
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

                # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # # repeat k/v heads if n_kv_heads < n_heads
        # key_states = repeat_kv(key_states, self.num_key_value_groups)
        # value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
            self.head_dim
        )

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
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )

        past_key_value = self.kv_cache(past_key_value, attn_weights.detach().clone())

        # print('attn_weights', attn_weights.shape)

        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(
                self.hidden_size // self.config.pretraining_tp, dim=2
            )
            o_proj_slices = self.o_proj.weight.split(
                self.hidden_size // self.config.pretraining_tp, dim=1
            )
            attn_output = sum(
                [
                    F.linear(attn_output[i], o_proj_slices[i])
                    for i in range(self.config.pretraining_tp)
                ]
            )
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class Llama3ForCausalLM_streaming(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        num_layers = len(self.model.layers)
        # for layer_idx in range(2, num_layers):
        for layer_idx in range(num_layers):
            self.model.layers[layer_idx].self_attn = Llama3Attention_streaming(config)




class LMEvalLlamaForCausalLM_drop(HFLM):
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
            self._model = H2OLlamaForCausalLM.from_pretrained(
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



############################ for streaming, the position embedding is different #####################

## H2O KV Cache dropping with Position rolling
class H2OLlamaAttention_streaming(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig):
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
        self._init_rope()

        self.kv_cache = H2OKVCache_LayerWise(
            hh_size=self.config.hh_size,
            recent_size=self.config.recent_size,
            k_seq_dim=2,
            v_seq_dim=2,
            hh_ratio=self.config.heavy_ratio,
            recent_ratio=self.config.recent_ratio
        )

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _clean_cache(self):
        self.kv_cache._clean_scores()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (
                self.num_key_value_heads * self.head_dim
            ) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [
                F.linear(hidden_states, query_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [
                F.linear(hidden_states, key_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [
                F.linear(hidden_states, value_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        # remake causal mask
        attention_mask = _make_causal_mask(
            bsz=bsz,
            tgt_len=q_len,
            past_key_values_length=past_key_value[0].shape[-2] if past_key_value is not None else 0,
            dtype=query_states.dtype,
            device=query_states.device,
        )

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        if not position_ids.nelement() > 1:
            position_ids[0][0] = kv_seq_len - 1

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        ### Shift Pos: query pos is min(cache_size, idx)
        # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        query_states = apply_rotary_pos_emb_single(query_states, cos, sin, position_ids)
        ###

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        ### Shift Pos: key pos is the pos in cache (Rolling KV Cache and using relative pos emb)
        key_position_ids = torch.arange(kv_seq_len, device=position_ids.device).unsqueeze(0)
        key_states = apply_rotary_pos_emb_single(key_states, cos, sin, key_position_ids)
        ###

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
            self.head_dim
        )

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
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )
        # core code
        past_key_value = self.kv_cache(past_key_value, attn_weights.detach().clone()) 

        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(
                self.hidden_size // self.config.pretraining_tp, dim=2
            )
            o_proj_slices = self.o_proj.weight.split(
                self.hidden_size // self.config.pretraining_tp, dim=1
            )
            attn_output = sum(
                [
                    F.linear(attn_output[i], o_proj_slices[i])
                    for i in range(self.config.pretraining_tp)
                ]
            )
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class H2OLlamaForCausalLM_streaming(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        num_layers = len(self.model.layers)
        for layer_idx in range(num_layers):
            self.model.layers[layer_idx].self_attn = H2OLlamaAttention_streaming(config)


class LMEvalLlamaForCausalLM_drop_streaming(HFLM):
    AUTO_MODEL_CLASS = None
    _DEFAULT_MAX_LENGTH = 2048
    def __init__(
        self,
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
            self._model = H2OLlamaForCausalLM_streaming.from_pretrained(
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