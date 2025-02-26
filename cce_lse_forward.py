# Copyright (C) 2024 Apple Inc. All Rights Reserved.
import torch
import triton
import triton.language as tl

from cut_cross_entropy.tl_autotune import cce_forward_autotune
from cut_cross_entropy.tl_utils import b_bin_fn, tl_softcapping


@triton.jit
def _mm_forward(
    accum,
    a_ptrs,
    partial_mask_a,
    b_ptrs,
    partial_mask_b,
    stride_ad,
    stride_bd,
    D,
    BLOCK_D: tl.constexpr,
    EVEN_D: tl.constexpr,
):
    d_inds = tl.arange(0, BLOCK_D)

    b_ptrs = b_ptrs + d_inds * stride_bd
    a_ptrs = a_ptrs + d_inds * stride_ad

    for d in range(0, tl.cdiv(D, BLOCK_D)):
        if EVEN_D:
            mask = partial_mask_b
        else:
            mask = partial_mask_b & (d_inds < (D - d * BLOCK_D))

        b = tl.load(b_ptrs, mask=mask, other=0.0)

        if EVEN_D:
            mask = partial_mask_a
        else:
            mask = partial_mask_a & (d_inds < (D - d * BLOCK_D))

        a = tl.load(a_ptrs, mask=mask, other=0.0)

        accum += tl.sum(a * b)

        b_ptrs += BLOCK_D * stride_bd
        a_ptrs += BLOCK_D * stride_ad

    return accum


def _cce_lse_forward_kernel(
    E,
    C,
    Bias,
    LSE,
    Valids,
    VocabOrdering,
    softcap,
    B,
    D,
    V,
    BMax,
    stride_eb,
    stride_ed,
    stride_cv,
    stride_cd,
    stride_biasv,
    stride_vb,
    B_BIN,
    BLOCK_B: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_D: tl.constexpr,
    GROUP_B: tl.constexpr,
    EVEN_D: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_VALIDS: tl.constexpr,
    HAS_VOCAB_ORDERING: tl.constexpr,
    HAS_SOFTCAP: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_b_chunks = tl.cdiv(B, BLOCK_B)
    num_v_chunks = tl.cdiv(V, BLOCK_V)
    num_v_in_group = GROUP_B * num_v_chunks
    group_id = pid // num_v_in_group
    first_pid_b = group_id * GROUP_B
    group_size_b = min(num_b_chunks - first_pid_b, GROUP_B)
    pid_b = first_pid_b + ((pid % num_v_in_group) % group_size_b)
    pid_v = (pid % num_v_in_group) // group_size_b

    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    if HAS_VALIDS:
        offs_b = tl.load(Valids + stride_vb * offs_b, mask=offs_b < B, other=BMax)

    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    if HAS_VOCAB_ORDERING:
        offs_v = tl.load(VocabOrdering + offs_v, mask=offs_v < V, other=V)

    offs_d = tl.arange(0, BLOCK_D)
    e_ptrs = E + (offs_b[:, None] * stride_eb + offs_d[None, :] * stride_ed)
    c_ptrs = C + (offs_v[None, :] * stride_cv + offs_d[:, None] * stride_cd)

    accum = tl.zeros((BLOCK_B, BLOCK_V), dtype=tl.float32)
    for d in range(0, tl.cdiv(D, BLOCK_D)):
        e_mask = offs_b[:, None] < BMax
        if not EVEN_D:
            e_mask = e_mask & (offs_d[None, :] < (D - d * BLOCK_D))

        e = tl.load(e_ptrs, mask=e_mask, other=0.0)

        c_mask = offs_v[None, :] < V
        if not EVEN_D:
            c_mask = c_mask & (offs_d[:, None] < (D - d * BLOCK_D))

        c = tl.load(c_ptrs, mask=c_mask, other=0.0)

        accum = tl.dot(e, c, accum)

        e_ptrs += BLOCK_D * stride_ed
        c_ptrs += BLOCK_D * stride_cd

    if HAS_BIAS:
        bias = tl.load(Bias + offs_v * stride_biasv, mask=offs_v < V, other=0.0)
        bias = bias.to(dtype=accum.dtype)
        accum += bias[None, :]

    if HAS_SOFTCAP:
        accum = tl_softcapping(accum, softcap)

    accum = tl.where(offs_v[None, :] < V, accum, float("-inf"))

    row_max = tl.max(accum, 1)
    row_sum = tl.sum(tl.exp(accum - row_max[:, None]), 1)
    row_lse = row_max + tl.log(row_sum)

    if HAS_VALIDS:
        direct_offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
        tl.atomic_min(LSE + direct_offs_b, row_lse, mask=direct_offs_b < B)
    else:
        tl.atomic_min(LSE + offs_b, row_lse, mask=offs_b < B)


_cce_lse_forward_kernel = triton.jit(_cce_lse_forward_kernel)
_cce_lse_forward_kernel = triton.heuristics(  # type: ignore
    {
        "EVEN_D": lambda args: (args["D"] % args["BLOCK_D"]) == 0,
        "HAS_VALIDS": lambda args: args["Valids"] is not None,
        "HAS_BIAS": lambda args: args["Bias"] is not None,
        "HAS_VOCAB_ORDERING": lambda args: args["VocabOrdering"] is not None,
        "HAS_SOFTCAP": lambda args: args["softcap"] is not None,
        "GROUP_B": lambda args: 8,
    }
)(_cce_lse_forward_kernel)
_cce_lse_forward_kernel = cce_forward_autotune()(_cce_lse_forward_kernel)  # type: ignore


def cce_lse_forward_kernel(
    e: torch.Tensor,
    c: torch.Tensor,
    bias: torch.Tensor | None,
    valids: torch.Tensor | None,
    softcap: float | None = None,
    vocab_ordering: torch.Tensor | None = None,
) -> torch.Tensor:
    assert c.size(1) == e.size(1)
    assert e.dtype in (
        torch.float16,
        torch.bfloat16,
    ), "Forward requires embeddings to be bf16 or fp16"
    assert c.dtype in (
        torch.float16,
        torch.bfloat16,
    ), "Forward requires classifier to be bf16 or fp16"

    if valids is not None:
        assert valids.ndim == 1
        B = valids.size(0)
    else:
        B = e.size(0)

    lse = torch.full((B,), float("inf"), dtype=torch.float32, device=e.device)

    def grid(META):
        return (triton.cdiv(B, META["BLOCK_B"]) * triton.cdiv(c.size(0), META["BLOCK_V"]),)

    if vocab_ordering is not None:
        assert vocab_ordering.ndim == 1
        assert vocab_ordering.numel() == c.size(0)
        assert vocab_ordering.stride(0) == 1

    _cce_lse_forward_kernel[grid](
        e,
        c,
        bias,
        lse,
        valids,
        vocab_ordering,
        softcap,
        B,
        e.size(1),
        c.size(0),
        e.size(0),
        e.stride(0),
        e.stride(1),
        c.stride(0),
        c.stride(1),
        1 if bias is None else bias.stride(0),
        1 if valids is None else valids.stride(0),
        B_BIN=b_bin_fn(B),
    )

    return lse
