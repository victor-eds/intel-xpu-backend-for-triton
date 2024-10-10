import sys, os, random

import torch
import triton
import triton.language as tl
import triton.language.core as core
from triton.language.standard import _log2, sum, zeros_like
import torch.utils.benchmark as benchmark


@triton.jit
def _compare_and_swap(x, ids, flip, i: core.constexpr, n_dims: core.constexpr):
    n_outer: core.constexpr = x.numel >> n_dims
    shape: core.constexpr = [n_outer * 2**i, 2, 2**(n_dims - i - 1)]
    y = core.reshape(x, shape)
    # slice left/right with 'stride' 2**(n_dims - i - 1)
    mask = core.arange(0, 2)[None, :, None]
    left = core.broadcast_to(sum(y * (1 - mask), 1)[:, None, :], shape)
    right = core.broadcast_to(sum(y * mask, 1)[:, None, :], shape)
    left = core.reshape(left, x.shape)
    right = core.reshape(right, x.shape)

    # idx
    y_idx = core.reshape(ids, shape)
    left_idx = core.broadcast_to(sum(y_idx * (1 - mask), 1)[:, None, :], shape)
    right_idx = core.broadcast_to(sum(y_idx * mask, 1)[:, None, :], shape)
    left_idx = core.reshape(left_idx, x.shape)
    right_idx = core.reshape(right_idx, x.shape)

    # actual compare-and-swap
    idtype = core.get_int_dtype(bitwidth=x.dtype.primitive_bitwidth,
                                signed=True)
    ileft = left.to(idtype, bitcast=True)
    iright = right.to(idtype, bitcast=True)
    ix = x.to(idtype, bitcast=True)

    cond = (left > right) ^ flip

    ret = ix ^ core.where(cond, ileft ^ iright, zeros_like(ix))

    new_ids = ids ^ core.where(cond, left_idx ^ right_idx, zeros_like(ids))

    return ret.to(x.dtype, bitcast=True), new_ids


@triton.jit
def _bitonic_merge(x, ids, stage: core.constexpr, order: core.constexpr,
                   n_dims: core.constexpr):
    '''
    order_type 0 == ascending
    order_type 1 == descending
    order_type 2 == alternating
    '''
    n_outer: core.constexpr = x.numel >> n_dims
    core.static_assert(stage <= n_dims)
    # flip denotes whether to re-arrange sub-sequences of elements in ascending or
    # descending order.
    # if flip = 00000000... then all elements will be re-arranged ascendingly at this stage
    # if flip = 00110011... then all the elements will be re-arranged alternatingly (with
    # a stride of 2) at this stage
    if order == 2:
        shape: core.constexpr = [
            n_outer * 2**(n_dims - 1 - stage), 2, 2**stage
        ]
        flip = core.reshape(
            core.broadcast_to(core.arange(0, 2)[None, :, None], shape),
            x.shape)
    else:
        flip = order
    # perform `stage` rounds of `compare-and-swap`
    for i in core.static_range(stage):
        x, ids = _compare_and_swap(x, ids, flip, i + (n_dims - stage), n_dims)
    return x, ids


@triton.jit
def argsort(x,
            ids,
            dim: core.constexpr = None,
            descending: core.constexpr = core.CONSTEXPR_0):
    # handle default dimension or check that it is the most minor dim
    _dim: core.constexpr = len(x.shape) - 1 if dim is None else dim
    core.static_assert(_dim == len(x.shape) - 1,
                       "only minor dimension is currently supported")
    # iteratively run bitonic merge-sort steps
    n_dims: core.constexpr = _log2(x.shape[_dim])

    for i in core.static_range(1, n_dims + 1):
        x, ids = _bitonic_merge(x, ids, i, 2 if i < n_dims else descending,
                                n_dims)
    return x, ids


@triton.jit
def topk_kernel(
    # Pointers to matrices
    x_ptr,
    o_ptr,
    id_ptr,
    stride_m,
    the_k,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    m_offset = pid_m * stride_m * BLOCK_M
    out_offset = pid_m * the_k * BLOCK_M
    k_off = tl.arange(0, BLOCK_N)

    x_ptrs = x_ptr + m_offset + (tl.arange(0, BLOCK_M)[:, None] * stride_m +
                                 k_off[None, :])

    # shape: [BLOCK_M, BLOCK_N]
    x = tl.load(x_ptrs)
    ids = tl.broadcast_to(tl.arange(0, BLOCK_N)[None, :], (BLOCK_M, BLOCK_N))

    # o, ids = argsort(x, ids, 1, True)
    o, ids = argsort(x, ids, 1, True)

    o_ptrs = o_ptr + out_offset + (tl.arange(0, BLOCK_M)[:, None] * the_k +
                                   k_off[None, :])
    id_ptrs = id_ptr + out_offset + (tl.arange(0, BLOCK_M)[:, None] * the_k +
                                     k_off[None, :])
    mask_k = k_off[None, :] < the_k
    tl.store(o_ptrs, o, mask=mask_k)
    tl.store(id_ptrs, ids, mask=mask_k)


def run_triton(x, k):
    o = torch.empty((x.size(0), k), dtype=x.dtype, device='cuda')
    ids = torch.empty((x.size(0), k), dtype=torch.int64, device='cuda')

    BLOCK_M = 4
    BLOCK_N = x.size(1)

    grid = (triton.cdiv(x.size(0), BLOCK_M), triton.cdiv(x.size(1), BLOCK_N))
    topk_kernel[grid](x, o, ids, x.stride(0), k, BLOCK_M, BLOCK_N)
    return o, ids


def run_pytorch(x, k):
    return torch.topk(x, k)


def log2(x):
    if x == 1:
        return 0
    return 1 + log2(x // 2)

results = []

label = 'TopK'

for n in [2**i for i in range(5, 12)]:
    m = n
    x = torch.randn(n, m, device='cuda', dtype=torch.float16)
    for k in [2**i for i in range(1, log2(n))]:
        sub_label = f'[{n}x{m}, {k}]'
        results.append(benchmark.Timer(
            stmt='run_triton(x, k)',
            setup='from __main__ import run_triton',
            globals={'x': x, 'k': k},
            label=label,
            sub_label=sub_label,
            description='Triton').adaptive_autorange())
        results.append(benchmark.Timer(
            stmt='run_pytorch(x, k)',
            setup='from __main__ import run_pytorch',
            globals={'x': x, 'k': k},
            label=label,
            sub_label=sub_label,
            description='PyTorch').adaptive_autorange())


compare = benchmark.Compare(results)
compare.colorize()
compare.print()
