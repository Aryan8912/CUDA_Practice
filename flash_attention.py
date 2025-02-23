import torch
import triton
import triton.language as tl
import math

@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale,
    L, O,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    
    # Initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_DMODEL)
    
    # Initialize pointers
    q_ptrs = Q + off_hz * stride_qh + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    k_ptrs = K + off_hz * stride_kh + offs_n[None, :] * stride_kn + offs_k[:, None] * stride_kk
    v_ptrs = V + off_hz * stride_vh + offs_n[:, None] * stride_vk + offs_k[None, :] * stride_vn
    
    # Load Q, K, V
    q = tl.load(q_ptrs)
    k = tl.load(k_ptrs)
    v = tl.load(v_ptrs)
    
    # Compute attention scores
    s = tl.dot(q, k) * sm_scale
    
    # Compute softmax
    s = s - tl.max(s, 1)[:, None]
    s = tl.exp(s)
    s = s / tl.sum(s, 1)[:, None]
    
    # Compute output
    o = tl.dot(s, v)
    
    # Write output
    o_ptrs = O + off_hz * stride_oh + offs_m[:, None] * stride_om + offs_k[None, :] * stride_on
    tl.store(o_ptrs, o)

class FlashAttention(torch.nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = hidden_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        
        # Initialize projection matrices
        self.q_proj = torch.nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = torch.nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = torch.nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = torch.nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, q, k, v, key_padding_mask=None):
        batch_size, seq_len, _ = q.shape
        
        # Project Q, K, V
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Prepare for Triton kernel
        q = q.permute(0, 2, 1, 3)  # [B, H, S, D]
        k = k.permute(0, 2, 1, 3)  # [B, H, S, D]
        v = v.permute(0, 2, 1, 3)  # [B, H, S, D]
        
        # Initialize output tensor
        o = torch.empty_like(q)
        
        # Launch kernel
        grid = (triton.cdiv(seq_len, 128), batch_size * self.num_heads)
        _fwd_kernel[grid](
            q, k, v, self.scaling,
            None, o,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            batch_size, self.num_heads, seq_len,
            BLOCK_M=128, BLOCK_N=128, BLOCK_DMODEL=self.head_dim,
        )
        
        # Reshape output
        o = o.permute(0, 2, 1, 3).contiguous()
        o = o.view(batch_size, seq_len, self.hidden_dim)
        
        # Final projection
        o = self.out_proj(o)
        return o

# Example usage
if __name__ == "__main__":
    # Create sample inputs
    batch_size = 2
    seq_len = 1024
    hidden_dim = 512
    num_heads = 8
    
    # Initialize model
    flash_attn = FlashAttention(hidden_dim=hidden_dim, num_heads=num_heads)
    
    # Create random input tensors
    q = torch.randn(batch_size, seq_len, hidden_dim, device='cuda')
    k = torch.randn(batch_size, seq_len, hidden_dim, device='cuda')
    v = torch.randn(batch_size, seq_len, hidden_dim, device='cuda')
    
    # Forward pass
    output = flash_attn(q, k, v)
    print(f"Output shape: {output.shape}")
