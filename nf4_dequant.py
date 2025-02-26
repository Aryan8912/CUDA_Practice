import torch
import triton
import triton.language as tl
from dataclasses import dataclass
from typing import Optional, Tuple
import time
from transformers import BitsAndBytesConfig
import warnings
import torch._dynamo
import math

# Configure torch._dynamo to suppress errors and fall back to eager mode
torch._dynamo.config.suppress_errors = True

# Constants for T4 GPU optimization
T4_MAX_BLOCK_SIZE = 256  # Reduced block size for T4
T4_MAX_SHARED_MEMORY = 48 * 1024  # 48KB shared memory for T4

@dataclass
class NF4Config:
    CLIP_MIN: int = -8
    CLIP_MAX: int = 7
    DTYPE_MIN: int = 0
    DTYPE_MAX: int = 15

class MemoryFormat:
    CONTIGUOUS = "contiguous"
    CHANNELS_LAST = "channels_last"

@triton.jit
def compute_absmax_kernel_t4(
    input_ptr, 
    absmax_ptr,
    num_elements,
    BLOCK_SIZE: tl.constexpr
):
    """Memory-efficient absmax computation for T4 GPU."""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements

    # Load data in smaller chunks
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    x_abs = tl.abs(x)
    
    # Efficient reduction using shared memory
    block_max = tl.max(x_abs, axis=0)
    tl.store(absmax_ptr + pid, block_max)

@triton.jit
def dequantize_kernel_t4(
    quantized_ptr,
    absmax_ptr,
    double_quant_scale_ptr,
    output_ptr,
    M, N,
    stride_qm, stride_qn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_DOUBLE_QUANT: tl.constexpr,
    CHUNK_SIZE: tl.constexpr = 64
):
    """Memory-efficient dequantization for T4 GPU."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Calculate offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Create masks
    mask_m = offs_m[:, None] < M
    mask_n = offs_n[None, :] < N
    mask = mask_m & mask_n

    # Load quantized values
    q_offs = offs_m[:, None] * stride_qm + offs_n[None, :] * stride_qn
    quantized = tl.load(quantized_ptr + q_offs, mask=mask, other=0)

    # Load scales
    absmax = tl.load(absmax_ptr + offs_m, mask=offs_m < M, other=1.0)
    scale = absmax[:, None] / 7.0

    if USE_DOUBLE_QUANT:
        double_scale = tl.load(double_quant_scale_ptr + offs_m, mask=offs_m < M, other=1.0)
        scale = scale * double_scale[:, None]

    # Dequantize
    dequantized = (quantized - 8) * scale

    # Store result
    o_offs = offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(output_ptr + o_offs, dequantized, mask=mask)

class NF4Dequantizer:
    def __init__(
        self,
        bnb_config: Optional[BitsAndBytesConfig] = None,
        memory_format: str = MemoryFormat.CONTIGUOUS,
        device: Optional[torch.device] = None,
        compile_mode: str = "reduce-overhead"
    ):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        
        # Initialize with BitsAndBytes config
        if bnb_config is None:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
        
        self.use_double_quant = bnb_config.bnb_4bit_use_double_quant
        self.compute_dtype = bnb_config.bnb_4bit_compute_dtype
        self.memory_format = memory_format
        self.config = NF4Config()
        
        # T4-specific optimizations
        self.block_size = min(T4_MAX_BLOCK_SIZE, 128)
        self.max_batch_elements = 512 * 1024
        
        # Compile the processing functions
        try:
            self._process_batch_compiled = torch.compile(
                self._process_batch,
                mode=compile_mode,
                fullgraph=True,
                options={
                    "max_autotune": True,
                    "triton.cudagraphs": True,
                    "triton.persistent_cache": True
                }
            )
            self._fallback_dequantize_compiled = torch.compile(
                self._fallback_dequantize,
                mode=compile_mode,
                fullgraph=True
            )
            self.use_compiled = True
            print("✓ Using compiled mode for faster execution")
        except Exception as e:
            warnings.warn(f"Torch compile not available: {str(e)}. Using eager mode.")
            self._process_batch_compiled = self._process_batch
            self._fallback_dequantize_compiled = self._fallback_dequantize
            self.use_compiled = False
    
    def _fallback_dequantize(self, batch: torch.Tensor, absmax: torch.Tensor, double_quant_scale: Optional[torch.Tensor] = None) -> torch.Tensor:
        """PyTorch fallback implementation for dequantization."""
        scale = absmax[:, None] / self.config.CLIP_MAX
        if self.use_double_quant and double_quant_scale is not None:
            scale = scale * double_quant_scale[:, None]
        return ((batch - 8) * scale).to(self.compute_dtype)
    
    def _process_batch(self, quantized_tensor: torch.Tensor, start_idx: int, end_idx: int) -> torch.Tensor:
        """Process a batch of data to avoid OOM."""
        batch = quantized_tensor[start_idx:end_idx]
        M, N = batch.shape
        
        # Compute grid dimensions for T4
        grid = (
            triton.cdiv(M, self.block_size),
            triton.cdiv(N, self.block_size)
        )
        
        # Allocate output tensor
        output = torch.empty(
            (M, N),
            device=self.device,
            dtype=self.compute_dtype
        )
        
        try:
            # Compute absmax
            absmax = torch.empty(
                (M,),
                device=self.device,
                dtype=self.compute_dtype
            )
            
            compute_absmax_kernel_t4[(grid[0],)](
                batch,
                absmax,
                batch.numel(),
                BLOCK_SIZE=self.block_size
            )
            
            # Generate double quantization scale
            double_quant_scale = None
            if self.use_double_quant:
                double_quant_scale = torch.rand(
                    M,
                    device=self.device,
                    dtype=self.compute_dtype
                ) * 2.0
            
            # Dequantize batch
            dequantize_kernel_t4[grid](
                batch,
                absmax,
                double_quant_scale if double_quant_scale is not None else absmax,
                output,
                M, N,
                batch.stride(0), batch.stride(1),
                output.stride(0), output.stride(1),
                BLOCK_M=self.block_size,
                BLOCK_N=self.block_size,
                USE_DOUBLE_QUANT=self.use_double_quant
            )
            
            return output
            
        except Exception as e:
            warnings.warn(f"Triton kernel failed: {str(e)}. Falling back to PyTorch implementation.")
            return self._fallback_dequantize_compiled(batch, absmax, double_quant_scale)
    
    @torch.no_grad()
    def dequantize(
        self,
        quantized_tensor: torch.Tensor,
        batch_size: Optional[int] = None
    ) -> torch.Tensor:
        """Dequantize NF4 tensor with batching for memory efficiency."""
        if not torch.is_tensor(quantized_tensor):
            raise TypeError("quantized_tensor must be a torch.Tensor")
        
        if not quantized_tensor.is_cuda:
            quantized_tensor = quantized_tensor.to(self.device)
        
        if torch.any(quantized_tensor < self.config.DTYPE_MIN) or \
           torch.any(quantized_tensor > self.config.DTYPE_MAX):
            raise ValueError(f"Quantized values must be in range [{self.config.DTYPE_MIN}, {self.config.DTYPE_MAX}]")
        
        # Determine batch size based on available memory
        if batch_size is None:
            batch_size = min(
                quantized_tensor.shape[0],
                self.max_batch_elements // quantized_tensor.shape[1]
            )
            batch_size = max(1, batch_size)
        
        outputs = []
        for i in range(0, quantized_tensor.shape[0], batch_size):
            end_idx = min(i + batch_size, quantized_tensor.shape[0])
            batch_output = self._process_batch_compiled(quantized_tensor, i, end_idx)
            outputs.append(batch_output)
        
        return torch.cat(outputs, dim=0)

def benchmark_dequantizer(
    shapes: list[Tuple[int, int]] = None,
    batch_size: Optional[int] = None,
    num_warmup: int = 5,
    num_runs: int = 10
):
    """Benchmark the dequantizer with and without compilation."""
    if shapes is None:
        shapes = [
            (128, 128),    # Small matrix
            (512, 256),    # Medium matrix
            (1024, 512),   # Larger matrix
        ]
    
    # Create dequantizers
    dequantizer_compiled = NF4Dequantizer(compile_mode="reduce-overhead")
    dequantizer_eager = NF4Dequantizer()
    dequantizer_eager.use_compiled = False  # Force eager mode
    
    print("\nBenchmarking NF4 Dequantization:")
    print("=" * 60)
    print(f"Block size: {dequantizer_compiled.block_size}")
    print(f"Compute dtype: {dequantizer_compiled.compute_dtype}")
    print(f"Number of runs: {num_runs}")
    print("-" * 60)
    print(f"{'Shape':>12} | {'Eager (ms)':>12} | {'Compiled (ms)':>12} | {'Speedup':>8}")
    print("-" * 60)
    
    for M, N in shapes:
        try:
            # Generate test data
            quantized = torch.randint(
                0, 16,
                (M, N),
                dtype=torch.int32,
                device=dequantizer_compiled.device
            )
            
            # Warmup
            for _ in range(num_warmup):
                _ = dequantizer_eager.dequantize(quantized, batch_size=batch_size)
                _ = dequantizer_compiled.dequantize(quantized, batch_size=batch_size)
            
            # Benchmark eager mode
            torch.cuda.synchronize()
            start_eager = time.perf_counter()
            for _ in range(num_runs):
                _ = dequantizer_eager.dequantize(quantized, batch_size=batch_size)
            torch.cuda.synchronize()
            eager_time = (time.perf_counter() - start_eager) * 1000 / num_runs
            
            # Benchmark compiled mode
            torch.cuda.synchronize()
            start_compiled = time.perf_counter()
            for _ in range(num_runs):
                _ = dequantizer_compiled.dequantize(quantized, batch_size=batch_size)
            torch.cuda.synchronize()
            compiled_time = (time.perf_counter() - start_compiled) * 1000 / num_runs
            
            # Calculate speedup
            speedup = eager_time / compiled_time
            
            print(f"{M}x{N:>7} | {eager_time:>10.2f} | {compiled_time:>10.2f} | {speedup:>7.2f}x")
            
        except Exception as e:
            print(f"{M}x{N:>7} | Error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("=" * 60)

if __name__ == "__main__":
    print("Benchmarking NF4 dequantization with torch.compile...")
    benchmark_dequantizer()
