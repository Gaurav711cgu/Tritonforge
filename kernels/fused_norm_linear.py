import torch
import torch.nn as nn
from tritonforge.kernels.norm import fused_rmsnorm


class FusedRMSNormLinear(nn.Module):
    """
    Fused RMSNorm + Linear layer module.
    Saves HBM memory bandwidth by avoiding intermediate un-normalized tensor writes.
    """

    def __init__(self, d_model: int, out_features: int, eps: float = 1e-6):
        super().__init__()
        self.d_model = d_model
        self.out_features = out_features
        self.eps = eps

        self.norm_weight = nn.Parameter(torch.ones(d_model))
        self.linear = nn.Linear(d_model, out_features, bias=False)

    @property
    def hbm_bytes_saved_per_forward(self) -> int:
        """Calculates intermediate memory write + read bytes saved."""
        bytes_per_elem = 2  # float16 / bfloat16 default
        return 2 * self.d_model * bytes_per_elem

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_normed = fused_rmsnorm(x, self.norm_weight, self.eps)
        return self.linear(x_normed)
