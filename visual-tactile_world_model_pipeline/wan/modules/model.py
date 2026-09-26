# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention import FeedForward
from diffusers.models.normalization import AdaLayerNormSingle
from diffusers.utils.torch_utils import maybe_allow_in_graph
from einops import rearrange
from diffusers.models.modeling_utils import ModelMixin

from .attention import attention

__all__ = ['WanModel']


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@torch.amp.autocast('cuda', enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@torch.amp.autocast('cuda', enabled=False)
def rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()


class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        x_float = x.float()
        if self.elementwise_affine:
            weight = self.weight.float()
            bias = self.bias.float()
        else:
            weight = None
            bias = None
        y = F.layer_norm(x_float, self.normalized_shape, weight, bias, eps=self.eps)
        return y.type_as(x)


class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        x = x.to(dtype=self.q.weight.dtype)

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        x = attention(
            q=rope_apply(q, grid_sizes, freqs),
            k=rope_apply(k, grid_sizes, freqs),
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size)

        # output
        x = x.flatten(2)
        x = x.to(dtype=self.o.weight.dtype)
        x = self.o(x)
        return x


class WanCrossAttention(WanSelfAttention):

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim
        x = x.to(dtype=self.q.weight.dtype)
        context = context.to(dtype=self.q.weight.dtype)

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        # compute attention
        x = attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = x.to(dtype=self.o.weight.dtype)
        x = self.o(x)
        return x


class WanJointAttention(nn.Module):

    def __init__(self, dim, num_heads, qk_norm=True, eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens=None):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            seq_lens(Tensor, optional): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim
        x = x.to(dtype=self.q.weight.dtype)

        # query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(x)).view(b, -1, n, d)
        v = self.v(x).view(b, -1, n, d)

        # attention
        x = attention(q, k, v, k_lens=seq_lens)

        # output
        x = x.flatten(2)
        x = x.to(dtype=self.o.weight.dtype)
        x = self.o(x)
        return x


class ActionRotaryPosEmbed(nn.Module):

    def __init__(self, dim: int, base_seq_length: int = 57, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.base_seq_length = base_seq_length
        self.theta = theta

    def forward(
        self,
        hidden_states: torch.Tensor,
        seq_length: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        grid = torch.arange(seq_length, dtype=torch.float32, device=hidden_states.device).unsqueeze(0)
        grid = grid / self.base_seq_length
        grid = grid.unsqueeze(-1)

        start = 1.0
        end = self.theta
        freqs = self.theta ** torch.linspace(
            math.log(start, self.theta),
            math.log(end, self.theta),
            self.dim // 2,
            device=hidden_states.device,
            dtype=torch.float32,
        )
        freqs = freqs * math.pi / 2.0
        freqs = freqs * (grid * 2 - 1)

        cos_freqs = freqs.cos().repeat_interleave(2, dim=-1)
        sin_freqs = freqs.sin().repeat_interleave(2, dim=-1)

        if self.dim % 2 != 0:
            cos_padding = torch.ones_like(cos_freqs[:, :, : self.dim % 2])
            sin_padding = torch.zeros_like(sin_freqs[:, :, : self.dim % 2])
            cos_freqs = torch.cat([cos_padding, cos_freqs], dim=-1)
            sin_freqs = torch.cat([sin_padding, sin_freqs], dim=-1)

        return cos_freqs, sin_freqs


def _apply_action_rotary_emb(x: torch.Tensor, freqs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    cos, sin = freqs
    batch_size = x.shape[0]
    if cos.shape[0] == 1 and batch_size > 1:
        cos = cos.repeat(batch_size, 1, 1)
    if sin.shape[0] == 1 and batch_size > 1:
        sin = sin.repeat(batch_size, 1, 1)

    x_real, x_imag = x.unflatten(-1, (-1, 2)).unbind(-1)
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(-2)
    return (x.float() * cos.unsqueeze(2) + x_rotated.float() * sin.unsqueeze(2)).to(x.dtype)


class ActionSelfAttention(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qk_norm: bool = True,
        eps: float = 1e-6,
        bias: bool = True,
        out_bias: bool = True,
    ):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim, bias=bias)
        self.k = nn.Linear(dim, dim, bias=bias)
        self.v = nn.Linear(dim, dim, bias=bias)
        self.o = nn.Linear(dim, dim, bias=out_bias)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        query = self.norm_q(self.q(hidden_states)).view(batch_size, sequence_length, self.num_heads, self.head_dim)
        key = self.norm_k(self.k(hidden_states)).view(batch_size, sequence_length, self.num_heads, self.head_dim)
        value = self.v(hidden_states).view(batch_size, sequence_length, self.num_heads, self.head_dim)

        if rotary_emb is not None:
            query = _apply_action_rotary_emb(query, rotary_emb)
            key = _apply_action_rotary_emb(key, rotary_emb)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask[:, None, None, :]

        hidden_states = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = self.o(hidden_states)
        return hidden_states


class ActionCrossAttention(nn.Module):

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: int,
        num_heads: int,
        qk_norm: bool = True,
        eps: float = 1e-6,
        bias: bool = True,
        out_bias: bool = True,
    ):
        assert query_dim % num_heads == 0
        super().__init__()
        self.query_dim = query_dim
        self.cross_attention_dim = cross_attention_dim
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.q = nn.Linear(query_dim, query_dim, bias=bias)
        self.k = nn.Linear(cross_attention_dim, query_dim, bias=bias)
        self.v = nn.Linear(cross_attention_dim, query_dim, bias=bias)
        self.o = nn.Linear(query_dim, query_dim, bias=out_bias)
        self.norm_q = WanRMSNorm(query_dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(query_dim, eps=eps) if qk_norm else nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = hidden_states.to(dtype=self.q.weight.dtype)
        encoder_hidden_states = encoder_hidden_states.to(dtype=self.k.weight.dtype)

        batch_size, sequence_length, _ = hidden_states.shape
        context_length = encoder_hidden_states.shape[1]

        query = self.norm_q(self.q(hidden_states)).view(batch_size, sequence_length, self.num_heads, self.head_dim)
        key = self.norm_k(self.k(encoder_hidden_states)).view(batch_size, context_length, self.num_heads, self.head_dim)
        value = self.v(encoder_hidden_states).view(batch_size, context_length, self.num_heads, self.head_dim)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask[:, None, None, :]

        hidden_states = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = self.o(hidden_states)
        return hidden_states


@maybe_allow_in_graph
class ActionTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        cross_attention_dim: int,
        qk_norm: str = "rms_norm_across_heads",
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = True,
        attention_out_bias: bool = True,
        eps: float = 1e-6,
        elementwise_affine: bool = False,
    ):
        super().__init__()

        self.norm1 = WanLayerNorm(dim, eps=eps, elementwise_affine=elementwise_affine)
        self.attn1 = ActionSelfAttention(
            dim=dim,
            num_heads=num_attention_heads,
            qk_norm=(qk_norm is not None),
            eps=eps,
            bias=attention_bias,
            out_bias=attention_out_bias,
        )

        self.norm2 = WanLayerNorm(dim, eps=eps, elementwise_affine=elementwise_affine)
        self.attn2 = ActionCrossAttention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim,
            num_heads=num_attention_heads,
            qk_norm=(qk_norm is not None),
            eps=eps,
            bias=attention_bias,
            out_bias=attention_out_bias,
        )

        self.ff = FeedForward(dim, activation_fn=activation_fn)
        self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = hidden_states.size(0)
        num_ada_params = self.scale_shift_table.shape[0]
        ada_values = self.scale_shift_table[None, None] + temb.reshape(batch_size, temb.size(1), num_ada_params, -1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = ada_values.unbind(dim=2)

        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa

        attn_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            rotary_emb=rotary_emb,
        )
        hidden_states = hidden_states + attn_hidden_states * gate_msa

        attn_hidden_states = self.attn2(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
        )
        hidden_states = hidden_states + attn_hidden_states

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        ff_output = self.ff(norm_hidden_states)
        hidden_states = hidden_states + ff_output * gate_mlp
        return hidden_states


def add_action_expert(
    self,
    num_layers: int = 28,
    inner_dim: int = 2048,
    activation_fn: str = "gelu-approximate",
    norm_eps: float = 1e-6,
    action_in_channels: int = 14,
    action_out_channels: int = None,
    action_num_attention_heads: int = 16,
    action_attention_head_dim: int = 32,
    action_rope_dim: int = None,
    action_final_embeddings: bool = True,
    learnable_action_state: bool = False,
    norm_elementwise_affine: bool = False,
    attention_bias: bool = True,
    attention_out_bias: bool = True,
    qk_norm: str = "rms_norm_across_heads",
    action_output_modalities: Optional[Dict[str, int]] = None,
    **kwargs,
):

    if action_out_channels is None:
        action_out_channels = action_in_channels

    self.action_in_channels = action_in_channels
    self.action_out_channels = action_out_channels
    self.action_inner_dim = action_num_attention_heads * action_attention_head_dim
    self.action_output_modalities = action_output_modalities or None
    self.action_output_slices = None
    if self.action_output_modalities is not None:
        self.action_output_slices = {}
        start = 0
        for name, modality_dim in self.action_output_modalities.items():
            self.action_output_slices[name] = slice(start, start + int(modality_dim))
            start += int(modality_dim)

    self.learnable_action_state = learnable_action_state
    if self.learnable_action_state:
        self.action_state = nn.Parameter(torch.randn(1, 1, action_in_channels))

    self.action_proj_in = nn.Linear(action_in_channels, self.action_inner_dim)
    self.action_scale_shift_table = nn.Parameter(torch.randn(2, self.action_inner_dim) / self.action_inner_dim**0.5)
    self.action_time_embed = AdaLayerNormSingle(self.action_inner_dim, use_additional_conditions=False)

    if action_rope_dim is None:
        action_rope_dim = self.action_inner_dim
    self.action_rope = ActionRotaryPosEmbed(
        dim=action_rope_dim,
        base_seq_length=57,
        theta=10000.0,
    )

    total_video_layers = max(1, int(getattr(self, "num_layers", num_layers)))
    action_num_layers = max(1, min(int(num_layers), total_video_layers))
    self.action_num_layers = action_num_layers
    if action_num_layers == 1:
        self.action_apply_indices = [total_video_layers - 1]
    else:
        self.action_apply_indices = [
            int(round(i * (total_video_layers - 1) / float(action_num_layers - 1)))
            for i in range(action_num_layers)
        ]

    self.action_blocks = nn.ModuleList(
        [
            ActionTransformerBlock(
                dim=self.action_inner_dim,
                num_attention_heads=action_num_attention_heads,
                attention_head_dim=action_attention_head_dim,
                cross_attention_dim=inner_dim,
                qk_norm=qk_norm,
                activation_fn=activation_fn,
                attention_bias=attention_bias,
                attention_out_bias=attention_out_bias,
                eps=norm_eps,
                elementwise_affine=norm_elementwise_affine,
            )
            for _ in range(action_num_layers)
        ]
    )

    self.action_proj_out = nn.Linear(self.action_inner_dim, action_out_channels)
    self.action_final_embeddings = action_final_embeddings
    if not self.action_final_embeddings:
        self.action_proj_extra = nn.Linear(self.action_inner_dim, self.action_inner_dim)

    self.action_norm_out = nn.LayerNorm(self.action_inner_dim, eps=1e-6, elementwise_affine=False)


def preprocessing_action_states(
    self,
    action_states: torch.Tensor = None,
    action_timestep: torch.LongTensor = None,
):

    assert getattr(self, "action_expert", False) is True
    assert action_states is not None and action_timestep is not None

    action_dtype = self.action_proj_in.weight.dtype
    action_states = action_states.to(dtype=action_dtype)

    batch_size = action_states.shape[0]
    action_seq_length = action_states.shape[1]

    if getattr(self, "learnable_action_state", False):
        action_states = self.action_state.repeat(batch_size, action_seq_length, 1).to(
            dtype=action_states.dtype,
            device=action_states.device,
        )

    action_rotary_emb = self.action_rope(action_states, action_seq_length)
    action_hidden_states = self.action_proj_in(action_states)

    action_temb, action_embedded_timestep = self.action_time_embed(
        action_timestep.flatten(),
        batch_size=batch_size,
        hidden_dtype=action_hidden_states.dtype,
    )

    action_temb = action_temb.view(batch_size, -1, action_temb.size(-1))
    action_embedded_timestep = action_embedded_timestep.view(batch_size, -1, action_embedded_timestep.size(-1))

    return action_temb, action_embedded_timestep, action_rotary_emb, action_hidden_states


def _choose_reduced_attention_spec(base_dim, base_heads, ratio, require_even_head_dim=False):
    reduced_dim = max(1, int(base_dim * ratio))
    reduced_heads = max(1, int(base_heads * ratio))

    while reduced_heads > 1 and reduced_dim % reduced_heads != 0:
        reduced_heads -= 1

    if require_even_head_dim:
        while reduced_heads > 1 and ((reduced_dim // reduced_heads) % 2 != 0):
            reduced_heads -= 1

    if require_even_head_dim and (reduced_dim // reduced_heads) % 2 != 0:
        reduced_dim += reduced_dim % 2

    return reduced_dim, reduced_heads


class WanAttentionBlock(nn.Module):

    def __init__(self,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 enable_multiview_attn=False,
                 enable_tactile_intra_attn=True,
                 eps=1e-6,
                 # ratios to reduce parameter counts for tactile/joint attention
                 tactile_dim_ratio=0.25,
                 joint_dim_ratio=0.5):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.enable_multiview_attn = enable_multiview_attn
        self.enable_tactile_intra_attn = enable_tactile_intra_attn
        self.eps = eps

        # layers
        # main joint attention (may be reduced internally to save params)
        if enable_multiview_attn:
            self.norm_joint = WanLayerNorm(dim, eps)
            joint_inner_dim, joint_heads = _choose_reduced_attention_spec(
                dim, num_heads, joint_dim_ratio, require_even_head_dim=False)
            if joint_inner_dim == dim and joint_heads == num_heads:
                self.joint_down = nn.Identity()
                self.joint_up = nn.Identity()
                self.joint_attn = WanJointAttention(dim, num_heads, qk_norm, eps)
            else:
                self.joint_down = nn.Linear(dim, joint_inner_dim)
                self.joint_up = nn.Linear(joint_inner_dim, dim)
                self.joint_attn = WanJointAttention(joint_inner_dim, joint_heads, qk_norm, eps)
            self.joint_inner_dim = joint_inner_dim
            self.joint_heads = joint_heads
        else:
            self.norm_joint = nn.Identity()
            self.joint_attn = None
            self.joint_down = nn.Identity()
            self.joint_up = nn.Identity()
            self.joint_inner_dim = dim
            self.joint_heads = num_heads

        # standard normalization and main self-attention (visual)
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps)

        # tactile-specific intra-view attention: use reduced internal dim to save params
        if enable_multiview_attn and enable_tactile_intra_attn:
            self.norm1_tactile = WanLayerNorm(dim, eps)
            tactile_inner_dim, tactile_heads = _choose_reduced_attention_spec(
                dim, num_heads, tactile_dim_ratio, require_even_head_dim=True)
            if tactile_inner_dim == dim and tactile_heads == num_heads:
                self.tactile_down = nn.Identity()
                self.tactile_up = nn.Identity()
                self.self_attn_tactile = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps)
            else:
                self.tactile_down = nn.Linear(dim, tactile_inner_dim)
                self.tactile_up = nn.Linear(tactile_inner_dim, dim)
                self.self_attn_tactile = WanSelfAttention(tactile_inner_dim, tactile_heads, window_size, qk_norm, eps)
            self.tactile_inner_dim = tactile_inner_dim
            self.tactile_heads = tactile_heads
        else:
            self.norm1_tactile = self.norm1
            self.self_attn_tactile = None
            self.tactile_down = nn.Identity()
            self.tactile_up = nn.Identity()
            self.tactile_inner_dim = dim
            self.tactile_heads = num_heads
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WanCrossAttention(dim, num_heads, (-1, -1), qk_norm,
                                            eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def _multiview_joint_attn(self, x, seq_lens, view_batch_sizes, attn_module,
                              norm_module):
        if (not self.enable_multiview_attn) or view_batch_sizes is None:
            return x

        if torch.is_tensor(view_batch_sizes):
            view_batch_sizes = view_batch_sizes.tolist()
        view_batch_sizes = [int(v) for v in view_batch_sizes]

        assert sum(view_batch_sizes) == x.size(0), (
            f'sum(view_batch_sizes)={sum(view_batch_sizes)} must equal batch size {x.size(0)}'
        )

        x_norm = norm_module(x)
        outputs = []
        start = 0
        for num_views in view_batch_sizes:
            end = start + num_views
            scene_x = x_norm[start:end]  # [V, L, C]

            if num_views <= 1:
                outputs.append(scene_x.new_zeros(scene_x.shape))
                start = end
                continue

            if seq_lens is None:
                # [1, V * L, C]
                joint_x = scene_x.reshape(1, -1, scene_x.size(-1))
                # project to smaller joint dim if needed
                if hasattr(self.joint_down, 'weight'):
                    joint_x = joint_x.to(dtype=self.joint_down.weight.dtype)
                joint_x_small = self.joint_down(joint_x)
                joint_y_small = attn_module(joint_x_small, seq_lens=None)
                if hasattr(self.joint_up, 'weight'):
                    joint_y_small = joint_y_small.to(dtype=self.joint_up.weight.dtype)
                joint_y = self.joint_up(joint_y_small).reshape(
                    num_views, scene_x.size(1), scene_x.size(2))
            else:
                scene_lens = seq_lens[start:end].tolist()
                packed_x = []
                for idx, cur_len in enumerate(scene_lens):
                    packed_x.append(scene_x[idx, :cur_len])
                packed_x = torch.cat(packed_x, dim=0)  # [sum_len, C]
                packed_x = packed_x.unsqueeze(0)  # [1, sum_len, C]
                # project, run small joint attn, then up-project
                if hasattr(self.joint_down, 'weight'):
                    packed_x = packed_x.to(dtype=self.joint_down.weight.dtype)
                packed_x_small = self.joint_down(packed_x)
                packed_lens = seq_lens.new_tensor([packed_x_small.size(1)])
                packed_y_small = attn_module(packed_x_small, seq_lens=packed_lens).squeeze(0)
                if hasattr(self.joint_up, 'weight'):
                    packed_y_small = packed_y_small.to(dtype=self.joint_up.weight.dtype)
                packed_y = self.joint_up(packed_y_small)

                joint_y = scene_x.new_zeros(scene_x.shape)
                offset = 0
                for idx, cur_len in enumerate(scene_lens):
                    joint_y[idx, :cur_len] = packed_y[offset:offset + cur_len]
                    offset += cur_len

            outputs.append(joint_y)
            start = end

        return x + torch.cat(outputs, dim=0)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        view_batch_sizes=None,
        tactile_view_mask=None,
        apply_multiview_joint_attn=True,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, L1, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        e = (self.modulation.unsqueeze(0).to(dtype=e.dtype) + e).chunk(6, dim=2)

        if tactile_view_mask is not None:
            if torch.is_tensor(tactile_view_mask):
                tactile_view_mask = tactile_view_mask.to(x.device).bool().flatten()
            else:
                tactile_view_mask = torch.tensor(
                    tactile_view_mask, dtype=torch.bool, device=x.device)
            assert tactile_view_mask.numel() == x.size(0), (
                f'tactile_view_mask length {tactile_view_mask.numel()} must equal batch size {x.size(0)}'
            )

        x_visual = self.norm1(x).float() * (1 + e[1].squeeze(2)) + e[0].squeeze(2)

        # self-attention
        if (self.self_attn_tactile is not None and tactile_view_mask is not None
                and tactile_view_mask.any()):
            if tactile_view_mask.all():
                x_tactile = self.norm1_tactile(x).float() * (
                    1 + e[1].squeeze(2)) + e[0].squeeze(2)
                y = self.self_attn_tactile(x_tactile, seq_lens, grid_sizes,
                                           freqs)
            else:
                y = x_visual.new_zeros(x.shape)

                visual_ids = torch.nonzero(~tactile_view_mask,
                                           as_tuple=False).squeeze(1)
                tactile_ids = torch.nonzero(tactile_view_mask,
                                            as_tuple=False).squeeze(1)

                x_visual_only = x_visual.index_select(0, visual_ids)
                y_visual = self.self_attn(
                    x_visual_only,
                    seq_lens.index_select(0, visual_ids),
                    grid_sizes.index_select(0, visual_ids),
                    freqs,
                )
                y_visual = y_visual.to(dtype=y.dtype)
                y.index_copy_(0, visual_ids, y_visual)

                x_tactile = self.norm1_tactile(x).float() * (
                    1 + e[1].squeeze(2)) + e[0].squeeze(2)
                x_tactile_only = x_tactile.index_select(0, tactile_ids)
                # project to reduced tactile dim, run small tactile attn, then up-project
                if hasattr(self.tactile_down, 'weight'):
                    x_tactile_only = x_tactile_only.to(dtype=self.tactile_down.weight.dtype)
                x_tactile_small = self.tactile_down(x_tactile_only)
                # slice freqs to match smaller head dim
                tactile_head_dim = self.tactile_inner_dim // max(1, self.tactile_heads)
                assert tactile_head_dim % 2 == 0, (
                    f'tactile head dim must be even, got {tactile_head_dim}')
                freqs_small = freqs[:, :tactile_head_dim // 2]
                y_tactile_small = self.self_attn_tactile(
                    x_tactile_small,
                    seq_lens.index_select(0, tactile_ids),
                    grid_sizes.index_select(0, tactile_ids),
                    freqs_small,
                )
                y_tactile = self.tactile_up(y_tactile_small)
                y_tactile = y_tactile.to(dtype=y.dtype)
                y.index_copy_(0, tactile_ids, y_tactile)
        else:
            y = self.self_attn(x_visual, seq_lens, grid_sizes, freqs)

        x = x + y * e[2].squeeze(2)

        # optional multiview joint-attention (global across views)
        if apply_multiview_joint_attn:
            x = self._multiview_joint_attn(x, seq_lens, view_batch_sizes,
                                           self.joint_attn,
                                           self.norm_joint)

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            x = x + self.cross_attn(self.norm3(x), context, context_lens)
            ffn_input = self.norm2(x).float() * (1 + e[4].squeeze(2)) + e[3].squeeze(2)
            if hasattr(self.ffn[0], 'weight'):
                ffn_input = ffn_input.to(dtype=self.ffn[0].weight.dtype)
            y = self.ffn(ffn_input)
            x = x + y * e[5].squeeze(2)
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        return x


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, L1, C]
        """
        e = (self.modulation.unsqueeze(0).to(dtype=e.dtype) + e.unsqueeze(2)).chunk(2, dim=2)
        x = self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2)
        x = self.head(x.to(dtype=self.head.weight.dtype))
        return x


class WanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlock']

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 enable_multiview_attn=False,
                 enable_tactile_intra_attn=True,
                 max_num_views=8,
                 use_view_pos_emb=True,
                 activation_fn='gelu-approximate',
                 tactile_dim_ratio=0.25,
                 joint_dim_ratio=0.5,
                 eps=1e-6,
                 attention_bias=True,
                 attention_out_bias=True,
                 init_weights=True,
                 use_activation_checkpoint=False,
                 action_expert=False,
                 action_in_channels=None,
                 action_out_channels=None,
                 action_num_attention_heads=None,
                 action_attention_head_dim=None,
                 action_rope_dim=None,
                 action_num_layers=None,
                 action_final_embeddings=True,
                 learnable_action_state=False,
                 action_norm_elementwise_affine=False,
                 action_output_modalities=None):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            enable_multiview_attn (`bool`, *optional*, defaults to False):
                Whether to enable multi-view triple-attention inside each Wan block.
            enable_tactile_intra_attn (`bool`, *optional*, defaults to True):
                Whether to use a dedicated tactile-only intra-view self-attention
                branch (randomly initialized), while visual views keep using
                pretrained self-attention weights.
            max_num_views (`int`, *optional*, defaults to 8):
                Max number of views supported by view positional embeddings.
            use_view_pos_emb (`bool`, *optional*, defaults to True):
                Whether to add learnable view-id embeddings to token features.
            tactile_dim_ratio (`float`, *optional*, defaults to 0.25):
                Internal width multiplier for tactile self-attention blocks.
            joint_dim_ratio (`float`, *optional*, defaults to 0.5):
                Internal width multiplier for multiview joint-attention blocks.
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
            init_weights (`bool`, *optional*, defaults to True):
                Whether to run explicit Xavier initialization after module creation.
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v', 'ti2v', 's2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.activation_fn = activation_fn
        self.attention_bias = attention_bias
        self.attention_out_bias = attention_out_bias
        self.enable_multiview_attn = enable_multiview_attn
        self.enable_tactile_intra_attn = enable_tactile_intra_attn
        self.max_num_views = max_num_views
        self.use_view_pos_emb = use_view_pos_emb
        self.tactile_dim_ratio = tactile_dim_ratio
        self.joint_dim_ratio = joint_dim_ratio
        self.eps = eps
        self.use_activation_checkpoint = use_activation_checkpoint
        self.action_expert = action_expert

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.view_embedding = nn.Embedding(
            max_num_views,
            dim) if (enable_multiview_attn and use_view_pos_emb) else None

        # blocks
        self.blocks = nn.ModuleList([
            WanAttentionBlock(dim, ffn_dim, num_heads, window_size, qk_norm,
                              cross_attn_norm, enable_multiview_attn,
                              enable_tactile_intra_attn, eps,
                              tactile_dim_ratio=tactile_dim_ratio,
                              joint_dim_ratio=joint_dim_ratio) for _ in range(num_layers)
        ])

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        if self.action_expert:
            if action_in_channels is None:
                action_in_channels = self.dim
            if action_out_channels is None:
                action_out_channels = action_in_channels
            if action_num_attention_heads is None:
                action_num_attention_heads = num_heads
            if action_attention_head_dim is None:
                action_attention_head_dim = max(1, self.dim // action_num_attention_heads)
            if action_num_layers is None:
                action_num_layers = num_layers

            add_action_expert(
                self,
                num_layers=action_num_layers,
                inner_dim=self.dim,
                activation_fn=activation_fn,
                norm_eps=eps,
                action_in_channels=action_in_channels,
                action_out_channels=action_out_channels,
                action_num_attention_heads=action_num_attention_heads,
                action_attention_head_dim=action_attention_head_dim,
                action_rope_dim=action_rope_dim,
                action_final_embeddings=action_final_embeddings,
                learnable_action_state=learnable_action_state,
                norm_elementwise_affine=action_norm_elementwise_affine,
                attention_bias=attention_bias,
                attention_out_bias=attention_out_bias,
                qk_norm=qk_norm,
                action_output_modalities=action_output_modalities,
            )

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
                               dim=1)

        # initialize weights
        if init_weights:
            self.init_weights()

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        y=None,
        view_batch_sizes=None,
        tactile_view_mask=None,
        return_video=True,
        action_states=None,
        action_timestep=None,
        return_action=False,
        history_action_state=None,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x
            view_batch_sizes (List[int] or Tensor, *optional*):
                Multi-view grouping definition. Each element is the number of
                views for one scene in the flattened batch. Example: [4, 4]
                means batch contains 2 scenes and each scene has 4 views.
            tactile_view_mask (List[bool] or Tensor, *optional*):
                Boolean mask over the flattened view batch. `True` means this
                sample is a tactile view and should use tactile-specific
                intra-view self-attention parameters.

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == 'i2v':
            assert y is not None
        if return_action:
            assert self.action_expert, "return_action=True requires action_expert=True in WanModel."
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        input_dtype = self.patch_embedding.weight.dtype
        x = [u.to(device=device, dtype=input_dtype) for u in x]
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long, device=device) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long, device=device)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        if self.enable_multiview_attn and self.use_view_pos_emb and view_batch_sizes is not None:
            if torch.is_tensor(view_batch_sizes):
                view_batch_sizes = view_batch_sizes.tolist()
            view_batch_sizes = [int(v) for v in view_batch_sizes]
            assert sum(view_batch_sizes) == x.size(0), (
                f'sum(view_batch_sizes)={sum(view_batch_sizes)} must equal batch size {x.size(0)}'
            )
            max_views = max(view_batch_sizes)
            assert max_views <= self.max_num_views, (
                f'max view count {max_views} exceeds max_num_views={self.max_num_views}'
            )
            view_ids = []
            for num_views in view_batch_sizes:
                view_ids.extend(range(num_views))
            view_ids = torch.tensor(view_ids,
                                    dtype=torch.long,
                                    device=x.device)
            x = x + self.view_embedding(view_ids).unsqueeze(1)

        # time embeddings
        if t.dim() == 1:
            t = t.expand(t.size(0), seq_len)
        time_dtype = next(self.time_embedding.parameters()).dtype
        with torch.amp.autocast('cuda', enabled=False):
            bt = t.size(0)
            t = t.flatten()
            t_embed = sinusoidal_embedding_1d(self.freq_dim, t).unflatten(0, (bt, seq_len)).to(dtype=time_dtype)
            e = self.time_embedding(t_embed)
            e0 = self.time_projection(e).unflatten(2, (6, self.dim))
            assert e.dtype == time_dtype and e0.dtype == time_dtype

        # context
        context_lens = None
        context_dtype = next(self.text_embedding.parameters()).dtype
        context = torch.stack([
            torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
            for u in context
        ]).to(dtype=context_dtype)
        context = self.text_embedding(context)

        if return_action:
            assert action_states is not None and action_timestep is not None
            if history_action_state is not None:
                history_len = history_action_state.size(1)
                action_states = torch.cat((history_action_state, action_states), dim=1)
                action_timestep = torch.cat(
                    (torch.zeros_like(action_timestep[:, :history_len]), action_timestep),
                    dim=1,
                )
            action_temb, action_embedded_timestep, action_rotary_emb, action_hidden_states = preprocessing_action_states(
                self,
                action_states,
                action_timestep,
            )
            action_batch_size = action_hidden_states.size(0)
            action_block_cursor = 0
            action_apply_indices = getattr(self, "action_apply_indices", list(range(len(self.action_blocks))))

        # arguments
        base_kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            view_batch_sizes=view_batch_sizes,
            tactile_view_mask=tactile_view_mask)

        use_multiview = self.enable_multiview_attn and (view_batch_sizes is not None)
        # ensure view_batch_sizes and tactile_view_mask are tensors on correct device for checkpoint
        if view_batch_sizes is not None and (not torch.is_tensor(view_batch_sizes)):
            try:
                view_batch_sizes = torch.tensor(view_batch_sizes, dtype=torch.long, device=device)
            except Exception:
                view_batch_sizes = torch.tensor(list(view_batch_sizes), dtype=torch.long, device=device)
        if tactile_view_mask is not None and (not torch.is_tensor(tactile_view_mask)):
            tactile_view_mask = torch.tensor(tactile_view_mask, dtype=torch.bool, device=device)

        for block_idx, block in enumerate(self.blocks):
            apply_multiview_joint_attn = use_multiview and ((block_idx + 1) % 3 == 0)
            if getattr(self, 'use_activation_checkpoint', False):
                def _run_block(x, e, seq_lens, grid_sizes, freqs, context, context_lens, view_batch_sizes, tactile_view_mask, block=block, apply_multiview_joint_attn=apply_multiview_joint_attn):
                    return block(
                        x,
                        e=e,
                        seq_lens=seq_lens,
                        grid_sizes=grid_sizes,
                        freqs=freqs,
                        context=context,
                        context_lens=context_lens,
                        view_batch_sizes=view_batch_sizes,
                        tactile_view_mask=tactile_view_mask,
                        apply_multiview_joint_attn=apply_multiview_joint_attn,
                    )

                x = torch.utils.checkpoint.checkpoint(
                    _run_block,
                    x,
                    base_kwargs['e'],
                    base_kwargs['seq_lens'],
                    base_kwargs['grid_sizes'],
                    base_kwargs['freqs'],
                    base_kwargs['context'],
                    base_kwargs['context_lens'],
                    view_batch_sizes,
                    tactile_view_mask,
                    use_reentrant=False,
                )
            else:
                x = block(
                    x,
                    apply_multiview_joint_attn=apply_multiview_joint_attn,
                    **base_kwargs,
                )

            if return_action and action_block_cursor < len(self.action_blocks):
                if block_idx == action_apply_indices[action_block_cursor]:
                    if x.size(0) % action_batch_size == 0:
                        action_n_view = x.size(0) // action_batch_size
                        final_hidden_states = rearrange(x, '(b v) l c -> b (v l) c', v=action_n_view)
                    else:
                        final_hidden_states = x

                    action_hidden_states = self.action_blocks[action_block_cursor](
                        hidden_states=action_hidden_states,
                        encoder_hidden_states=final_hidden_states,
                        temb=action_temb,
                        rotary_emb=action_rotary_emb,
                    )
                    action_block_cursor += 1

        # head
        if return_video:
            x = self.head(x, e)

        # unpatchify
        if return_video:
            x = self.unpatchify(x, grid_sizes)
            video_output = [u.float() for u in x]
        else:
            video_output = None

        if return_action:
            if self.action_final_embeddings:
                action_scale_shift_values = self.action_scale_shift_table[None, None] + action_embedded_timestep[:, :, None]
                action_shift, action_scale = action_scale_shift_values[:, :, 0], action_scale_shift_values[:, :, 1]
                action_hidden_states = self.action_norm_out(action_hidden_states)
                action_hidden_states = action_hidden_states * (1 + action_scale) + action_shift
            else:
                action_hidden_states = self.action_norm_out(action_hidden_states)
                action_hidden_states = self.action_proj_extra(action_hidden_states)

            if history_action_state is not None:
                action_hidden_states = action_hidden_states[:, history_len:]

            action_output = self.action_proj_out(action_hidden_states)
            if self.action_output_slices is not None:
                action_splits = {
                    name: action_output[..., slc]
                    for name, slc in self.action_output_slices.items()
                }
            else:
                action_splits = None

        if return_action or return_video:
            return_data = []
            if return_video:
                return_data.append(video_output)
            if return_action:
                return_data.append(action_output)
                if action_splits is not None:
                    return_data.append(action_splits)
            return tuple(return_data) if len(return_data) > 1 else return_data[0]

        return [u.float() for u in self.unpatchify(self.head(x, e), grid_sizes)]

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        if self.view_embedding is not None:
            nn.init.normal_(self.view_embedding.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
