'''
This is the info NCE loss function for the two-tower ICD model.
'''


from __future__ import annotations
import torch
import torch.nn.functional as F


def info_nce_inbatch(
    z_pat: torch.Tensor,
    z_code: torch.Tensor,
    tau: float = 0.07,
) -> torch.Tensor:
    """
    In-batch InfoNCE: assume each row i in z_pat matches row i in z_code.
    z_pat: [B, d]
    z_code: [B, d]
    """
    sim = (z_pat @ z_code.T) / tau      # [B, B]
    labels = torch.arange(z_pat.size(0), device=z_pat.device)
    loss = F.cross_entropy(sim, labels)
    return loss


