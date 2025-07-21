# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch import nn
from torch.nn.parameter import Parameter
from dataclasses import dataclass
from typing import Optional

from vllm.attention.backends.abstract import AttentionMetadata
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size)
from vllm.forward_context import get_forward_context
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               MergedColumnParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn, causal_conv1d_update)
from vllm.model_executor.layers.mamba.ops.mamba_ssm import (
    selective_scan_fn, selective_state_update)
from vllm.model_executor.models.mamba_cache import MambaCacheParams
from vllm.model_executor.utils import set_weight_attrs
from vllm import envs

@dataclass
class PrefillDecodeInfo:
    has_prefill: bool
    has_decode: bool
    hidden_states_BC_p: Optional[torch.Tensor]
    hidden_states_BC_d: Optional[torch.Tensor]
    gate_p: Optional[torch.Tensor]
    gate_d: Optional[torch.Tensor]
    state_indices_tensor_p: Optional[torch.Tensor]
    state_indices_tensor_d: Optional[torch.Tensor]
    context_lens_tensor_p: Optional[torch.Tensor]
    context_lens_tensor_d: Optional[torch.Tensor]
    num_prefills: int
    num_decodes: int

# Adapted from transformers.models.mamba.modeling_mamba.MambaMixer
@CustomOp.register("mamba_mixer")
class MambaMixer(CustomOp):
    """
    Compute ∆, A, B, C, and D the state space parameters and compute
    the `contextualized_states`. A, D are input independent
    (see Mamba paper [1] Section 3.5.2 "Interpretation of A"
    for why A isn't selective) ∆, B, C are input-dependent
    (this is a key difference between Mamba and the linear time
    invariant S4, and is why Mamba is called
    **selective** state spaces)
    """

    def __init__(self,
                 hidden_size: int,
                 ssm_state_size: int,
                 conv_kernel_size: int,
                 intermediate_size: int,
                 time_step_rank: int,
                 use_conv_bias: bool,
                 use_bias: bool,
                 use_rms_norm: bool,
                 rms_norm_has_weight: bool = True,
                 rms_norm_eps: float = 1e-5,
                 activation="silu",
                 is_lora_enabled: bool = False):
        super().__init__()
        self.time_step_rank = time_step_rank
        self.ssm_state_size = ssm_state_size
        self.use_rms_norm = use_rms_norm
        self.activation = activation
        self.is_lora_enabled = is_lora_enabled

        self.conv1d = ColumnParallelLinear(
            input_size=conv_kernel_size,
            output_size=intermediate_size,
            bias=use_conv_bias,
        )
        # unsqueeze to fit conv1d weights shape into the linear weights shape.
        # Can't do this in `weight_loader` since it already exists in
        # `ColumnParallelLinear` and `set_weight_attrs`
        # doesn't allow to override it
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)

        self.in_proj = MergedColumnParallelLinear(hidden_size,
                                                  [intermediate_size] * 2,
                                                  bias=use_bias)

        # selective projection used to make dt, B and C input dependent
        self.x_proj = RowParallelLinear(
            intermediate_size,
            time_step_rank + ssm_state_size * 2,
            bias=False,
        )
        # time step projection (discretization) -
        # In the forward we need to apply dt_proj without the bias,
        # as the bias is added in the selective scan kernel.
        self.dt_proj = ColumnParallelLinear(time_step_rank,
                                            intermediate_size,
                                            bias=True,
                                            skip_bias_add=True)

        def weight_loader(param: Parameter, loaded_weight: torch.Tensor):
            tp_rank = get_tensor_model_parallel_rank()
            tp_size = get_tensor_model_parallel_world_size()
            param.data.copy_(
                loaded_weight.data.split(loaded_weight.shape[0] // tp_size,
                                         dim=0)[tp_rank])

        def A_weight_loader(param: Parameter, loaded_weight: torch.Tensor):
            weight_loader(param, -torch.exp(loaded_weight.float()))

        tp_size = get_tensor_model_parallel_world_size()
        self.A = nn.Parameter(
            torch.empty(
                intermediate_size // tp_size,
                ssm_state_size,
                dtype=torch.float32,
            ))
        self.D = nn.Parameter(torch.ones(intermediate_size // tp_size))

        set_weight_attrs(self.D, {"weight_loader": weight_loader})
        set_weight_attrs(self.A, {"weight_loader": A_weight_loader})

        self.out_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=use_bias,
            input_is_parallel=True,
        )

        self.dt_layernorm = RMSNorm(
            time_step_rank,
            eps=rms_norm_eps,
            has_weight=rms_norm_has_weight,
        ) if use_rms_norm else None

        self.b_layernorm = RMSNorm(
            ssm_state_size,
            eps=rms_norm_eps,
            has_weight=rms_norm_has_weight,
        ) if use_rms_norm else None

        self.c_layernorm = RMSNorm(
            ssm_state_size,
            eps=rms_norm_eps,
            has_weight=rms_norm_has_weight,
        ) if use_rms_norm else None

    def forward_native(self, hidden_states: torch.Tensor,
                       conv_state: torch.Tensor, ssm_state: torch.Tensor):
        pass


    def _get_prefill_decode_info(self, attn_metadata, mamba_cache_params, hidden_states_BC, gate) -> PrefillDecodeInfo:
        """
        Helper func to determine prefill/decode presence in the batch and extract relevant indices and tensors.
        Handles batch order: V1 = decode->prefill, V0 = prefill->decode.
        """
        num_prefill_tokens = attn_metadata.num_prefill_tokens
        num_decode_tokens = attn_metadata.num_decode_tokens
        num_prefills = attn_metadata.num_prefills
        num_decodes = attn_metadata.num_decode_tokens
        has_prefill = num_prefill_tokens > 0
        has_decode = num_decode_tokens > 0

        if has_prefill and has_decode:
            def split(tensor, sizes, dim):
                return torch.split(tensor, sizes, dim=dim) if tensor is not None else (None, None)
            if envs.VLLM_USE_V1:
                # V1: decode, prefill order
                hidden_states_BC_d, hidden_states_BC_p = split(hidden_states_BC, [num_decode_tokens, num_prefill_tokens], -1)
                gate_d, gate_p = split(gate, [num_decode_tokens, num_prefill_tokens], -1)
                state_indices_tensor_d, state_indices_tensor_p = split(mamba_cache_params.state_indices_tensor, [num_decodes, num_prefills], 0)
                context_lens_tensor_d, context_lens_tensor_p = split(attn_metadata.context_lens_tensor, [num_decodes, num_prefills], 0)
            else:
                # V0: prefill, decode order
                hidden_states_BC_p, hidden_states_BC_d = split(hidden_states_BC, [num_prefill_tokens, num_decode_tokens], -1)
                gate_p, gate_d = split(gate, [num_prefill_tokens, num_decode_tokens], -1)
                state_indices_tensor_p, state_indices_tensor_d = split(mamba_cache_params.state_indices_tensor, [num_prefills, num_decodes], 0)
                context_lens_tensor_p, context_lens_tensor_d = split(attn_metadata.context_lens_tensor, [num_prefills, num_decodes], 0)
        else:
            hidden_states_BC_p = hidden_states_BC if has_prefill else None
            hidden_states_BC_d = hidden_states_BC if has_decode else None
            gate_p = gate if has_prefill else None
            gate_d = gate if has_decode else None
            state_indices_tensor_p = mamba_cache_params.state_indices_tensor if has_prefill else None
            state_indices_tensor_d = mamba_cache_params.state_indices_tensor if has_decode else None
            context_lens_tensor_p = attn_metadata.context_lens_tensor if has_prefill else None
            context_lens_tensor_d = attn_metadata.context_lens_tensor if has_decode else None

        return PrefillDecodeInfo(
            has_prefill=has_prefill,
            has_decode=has_decode,
            hidden_states_BC_p=hidden_states_BC_p,
            hidden_states_BC_d=hidden_states_BC_d,
            gate_p=gate_p,
            gate_d=gate_d,
            state_indices_tensor_p=state_indices_tensor_p,
            state_indices_tensor_d=state_indices_tensor_d,
            context_lens_tensor_p=context_lens_tensor_p,
            context_lens_tensor_d=context_lens_tensor_d,
            num_prefills=num_prefills,
            num_decodes=num_decodes,
        )

    def _normalize_ssm_params(self, time_step, B, C):
        if self.use_rms_norm:
            time_step = self.dt_layernorm(time_step.contiguous())
            B = self.b_layernorm(B.contiguous())
            C = self.c_layernorm(C.contiguous())
        return time_step, B, C

    def _run_prefill_flow(self, *, conv_weights, mamba_cache_params, attn_metadata, hidden_states_BC_p, gate_p, state_indices_tensor_p, context_lens_tensor_p, num_prefills):
        conv_out_p = causal_conv1d_fn(
            hidden_states_BC_p,
            conv_weights,
            self.conv1d.bias,
            activation=self.activation,
            conv_states=mamba_cache_params.conv_state,
            has_initial_state=(context_lens_tensor_p > 0) if context_lens_tensor_p is not None else False,
            cache_indices=state_indices_tensor_p,
            query_start_loc=attn_metadata.query_start_loc[:num_prefills + 1] if attn_metadata.query_start_loc is not None else None
        )
        conv_out_p = conv_out_p[..., :hidden_states_BC_p.shape[-1]].contiguous()
        ssm_params_p = self.x_proj(conv_out_p.transpose(-2, -1).contiguous())[0]
        time_step_p, B_p, C_p = torch.split(ssm_params_p, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1)
        time_step_p, B_p, C_p = self._normalize_ssm_params(time_step_p, B_p, C_p)
        discrete_time_step_p = self.dt_proj(time_step_p)[0].transpose(-2, -1)
        time_proj_bias = self.dt_proj.bias.float() if hasattr(self.dt_proj, "bias") and self.dt_proj.bias is not None else None
        scan_out_p = selective_scan_fn(
            conv_out_p,
            mamba_cache_params.ssm_state,
            discrete_time_step_p,
            self.A,
            B_p.transpose(-2, -1),
            C_p.transpose(-2, -1),
            self.D.float(),
            gate_p,
            time_proj_bias,
            delta_softplus=True,
            cache_indices=state_indices_tensor_p,
            has_initial_state=(context_lens_tensor_p > 0) if context_lens_tensor_p is not None else False,
            query_start_loc=attn_metadata.query_start_loc[:num_prefills + 1] if attn_metadata.query_start_loc is not None else None
        )
        return scan_out_p

    def _run_decode_flow(self, *, conv_weights, mamba_cache_params, hidden_states_BC_d, gate_d, state_indices_tensor_d):
        conv_out_d = causal_conv1d_update(
            hidden_states_BC_d.transpose(0, 1),
            mamba_cache_params.conv_state,
            conv_weights,
            self.conv1d.bias,
            self.activation,
            conv_state_indices=state_indices_tensor_d
        )
        conv_out_d = conv_out_d.transpose(0, 1).contiguous()
        ssm_params_d = self.x_proj(conv_out_d.transpose(-2, -1).contiguous())[0]
        time_step_d, B_d, C_d = torch.split(ssm_params_d, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1)
        time_step_d, B_d, C_d = self._normalize_ssm_params(time_step_d, B_d, C_d)
        discrete_time_step_d = self.dt_proj(time_step_d)[0].transpose(-2, -1)
        time_proj_bias = self.dt_proj.bias.float() if hasattr(self.dt_proj, "bias") and self.dt_proj.bias is not None else None
        scan_out_d = selective_state_update(
            mamba_cache_params.ssm_state,
            conv_out_d.transpose(0, 1),
            discrete_time_step_d.transpose(0, 1),
            self.A,
            B_d,
            C_d,
            self.D,
            gate_d.transpose(0, 1),
            time_proj_bias,
            dt_softplus=True,
            state_batch_indices=state_indices_tensor_d
        )
        scan_out_d = scan_out_d.transpose(0, 1)
        return scan_out_d

    #Optimized for prefill vs decode kernels
    def forward_cuda(self, hidden_states: torch.Tensor,
                        mamba_cache_params: MambaCacheParams):
        attn_metadata = get_forward_context().attn_metadata

        # 1. Gated MLP initial linear projection
        projected_states = self.in_proj(hidden_states)[0]
        projected_states = projected_states.transpose(-2, -1)
        hidden_states_BC, gate = projected_states.chunk(2, dim=-2)

        # 2. Get prefill/decode info
        batch_info = self._get_prefill_decode_info(attn_metadata, mamba_cache_params, hidden_states_BC, gate)
        conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))
        outputs = []
        if batch_info.has_prefill:
            scan_out_p = self._run_prefill_flow(
                conv_weights=conv_weights,
                mamba_cache_params=mamba_cache_params,
                attn_metadata=attn_metadata,
                hidden_states_BC_p=batch_info.hidden_states_BC_p,
                gate_p=batch_info.gate_p,
                state_indices_tensor_p=batch_info.state_indices_tensor_p,
                context_lens_tensor_p=batch_info.context_lens_tensor_p,
                num_prefills=batch_info.num_prefills
            )
            outputs.append(scan_out_p)
        if batch_info.has_decode:
            scan_out_d = self._run_decode_flow(
                conv_weights=conv_weights,
                mamba_cache_params=mamba_cache_params,
                hidden_states_BC_d=batch_info.hidden_states_BC_d,
                gate_d=batch_info.gate_d,
                state_indices_tensor_d=batch_info.state_indices_tensor_d
            )
            outputs.append(scan_out_d)
        scan_outputs_combined = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=-1)
        if self.is_lora_enabled:
            scan_outputs_combined = scan_outputs_combined.transpose(-2, -1).contiguous()
            out = self.out_proj(scan_outputs_combined)[0]
        else:
            out = self.out_proj(scan_outputs_combined.transpose(-2, -1))[0]
        return out
