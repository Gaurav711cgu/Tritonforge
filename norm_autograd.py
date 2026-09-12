"""
FusedRMSNormAutograd — torch.autograd.Function implementation with custom forward and backward passes.
Supports full PyTorch backpropagation and torch.autograd.gradcheck validation.
"""

import torch
from tritonforge.kernels.norm import fused_rmsnorm, pytorch_rmsnorm


class FusedRMSNormAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        # Guarantee inputs are contiguous for kernel dispatch
        x_contig = x.contiguous()
        weight_contig = weight.contiguous()

        # Fused forward pass via Triton kernel (with hardware/CPU fallback routing)
        out = fused_rmsnorm(x_contig, weight_contig, eps)

        # Calculate RMS factor for backward pass derivation
        # RMS = sqrt(mean(x^2) + eps)
        var = x_contig.pow(2).mean(-1, keepdim=True)
        rrms = torch.rsqrt(var + eps)

        # Save for backward pass
        ctx.save_for_backward(x_contig, weight_contig, rrms)
        ctx.eps = eps
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, weight, rrms = ctx.saved_tensors
        N = x.shape[-1]

        # Derivatives:
        # y = x * rrms * weight
        # dL/dx = rrms * weight * grad_output - (rrms^3 / N) * x * sum(grad_output * weight * x, dim=-1)
        # dL/dweight = sum(grad_output * x * rrms, across batch dims)

        grad_w_term = grad_output * weight
        dot_product = (grad_w_term * x).sum(dim=-1, keepdim=True)

        dx = rrms * (grad_w_term - (rrms.pow(2) / N) * x * dot_product)

        # Reduce grad_weight across all leading batch/seq dimensions
        if x.ndim > 1:
            reduce_dims = list(range(x.ndim - 1))
            dweight = (grad_output * x * rrms).sum(dim=reduce_dims)
        else:
            dweight = grad_output * x * rrms

        return dx, dweight, None


def fused_rmsnorm_autograd(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Wrapper function invoking FusedRMSNormAutograd.apply."""
    return FusedRMSNormAutograd.apply(x, weight, eps)
