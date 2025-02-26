import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Tuple, Any
import math

class MemoryEfficientLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X: torch.Tensor, linear: nn.Linear, 
               labels: torch.Tensor, transform_fn: Callable,
               batch_size: int = 2) -> torch.Tensor:
        """
        Forward pass that processes data in batches to save memory.
        
        Args:
            X: Input tensor
            linear: Linear layer
            labels: Target labels
            transform_fn: Function to transform outputs (e.g., cross entropy)
            batch_size: Size of mini-batches for processing
        """
        # Save tensors needed for backward pass
        ctx.save_for_backward(X, labels)
        ctx.linear = linear
        ctx.transform_fn = transform_fn
        ctx.batch_size = batch_size
        
        # Get dimensions
        B, seq_len, hidden_dim = X.shape
        vocab_size = linear.weight.shape[0]
        
        # Process in batches to save memory
        total_loss = 0
        num_batches = (B + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, B)
            
            # Get current batch
            batch_X = X[start_idx:end_idx]
            batch_labels = labels[start_idx:end_idx]
            
            # Forward pass for current batch
            # Compute logits efficiently without materializing full tensor
            batch_output = linear(batch_X)  # [batch_size, seq_len, vocab_size]
            
            # Apply transformation function (e.g., cross entropy)
            loss = transform_fn(batch_output, batch_labels)
            total_loss += loss * (end_idx - start_idx)
            
            # Free memory
            del batch_output
            torch.cuda.empty_cache()
        
        return total_loss / B

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None, None, None]:
        """Backward pass that computes gradients in a memory-efficient way."""
        X, labels = ctx.saved_tensors
        linear = ctx.linear
        transform_fn = ctx.transform_fn
        batch_size = ctx.batch_size
        
        # Initialize gradient accumulators
        grad_X = torch.zeros_like(X)
        grad_weight = torch.zeros_like(linear.weight)
        if linear.bias is not None:
            grad_bias = torch.zeros_like(linear.bias)
        
        # Process in batches
        num_batches = (X.size(0) + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, X.size(0))
            
            # Get current batch
            batch_X = X[start_idx:end_idx].detach().requires_grad_()
            batch_labels = labels[start_idx:end_idx]
            
            # Forward pass with grad tracking
            with torch.enable_grad():
                batch_output = linear(batch_X)
                loss = transform_fn(batch_output, batch_labels)
                
            # Backward pass for this batch
            batch_grads = torch.autograd.grad(
                loss,
                [batch_X, linear.weight, linear.bias] if linear.bias is not None else [batch_X, linear.weight],
                grad_output * (end_idx - start_idx) / X.size(0)
            )
            
            # Accumulate gradients
            grad_X[start_idx:end_idx] = batch_grads[0]
            grad_weight += batch_grads[1]
            if linear.bias is not None:
                grad_bias += batch_grads[2]
            
            # Free memory
            del batch_output, batch_grads
            torch.cuda.empty_cache()
        
        return grad_X, None, None, None, None

# Example transformation functions
def cross_entropy_transform(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross entropy loss transformation."""
    return F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), reduction='mean')

def mse_transform(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error transformation."""
    return F.mse_loss(output, target, reduction='mean')

def custom_transform(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Custom transformation example (L1 loss with scaling)."""
    return F.l1_loss(output, target, reduction='mean') * 0.5

# Test the implementation
def test_memory_efficient_linear():
    # Parameters
    batch_size = 4
    seq_len = 512
    hidden_dim = 1024
    vocab_size = 32000  # Smaller for testing
    
    # Create test data
    X = torch.randn(batch_size, seq_len, hidden_dim, device='cuda')
    labels = torch.randint(0, vocab_size, (batch_size, seq_len), device='cuda')
    
    # Create linear layer
    linear = nn.Linear(hidden_dim, vocab_size, bias=True).cuda()
    
    # Test with different transformation functions
    transforms = {
        'cross_entropy': cross_entropy_transform,
        'mse': mse_transform,
        'custom': custom_transform
    }
    
    for name, transform_fn in transforms.items():
        print(f"\nTesting {name} transformation:")
        
        # Memory efficient forward pass
        output = MemoryEfficientLinear.apply(X, linear, labels, transform_fn, 2)
        
        # Compute gradients
        output.backward()
        
        print(f"Output shape: {output.shape}")
        print(f"Output value: {output.item():.4f}")

if __name__ == "__main__":
    test_memory_efficient_linear()
