from __future__ import annotations
from typing import Dict, Iterable, List

import numpy as np
import torch
from tqdm import tqdm


@torch.no_grad()
def evaluate_recall_at_k(
    z_codes: torch.Tensor,
    loader,
    model: torch.nn.Module,
    device: torch.device,
    k_list: Iterable[int] = (50, 100, 200),
) -> Dict[int, float]:
    """
    Compute Recall@K for a trained patient tower and fixed ICD code embeddings.

    Args:
        z_codes: [N_codes, d] tensor of ICD embeddings (already normalized).
        loader:  DataLoader for PatientDataset (val or test).
                 Each batch must contain 'txt_inputs', 'ehr', and 'pos_codes'.
        model:   PatientTower instance.
        device:  torch.device ('cuda', 'mps', or 'cpu').
        k_list:  iterable of K values, e.g. (50, 100, 200).

    Returns:
        Dict mapping K -> Recall@K (float in [0,1]).
    """
    model.eval()
    k_list = list(k_list)
    max_k = max(k_list)

    recalls = {k: [] for k in k_list}
    z_codes = z_codes.to(device)  # ensure same device

    for batch in tqdm(loader, desc="Eval"):
        txt_inputs = {k: v.to(device) for k, v in batch["txt_inputs"].items()}
        ehr = batch["ehr"].to(device)
        pos_lists: List[List[int]] = batch["pos_codes"]

        # Encode patients -> [B, d]
        z_pat = model(txt_inputs, ehr)              # [B, d]

        # Similarity against all ICD codes: [B, N_codes]
        scores = z_pat @ z_codes.T                  # dot product (cosine, since normalized)

        # Get top max_k indices for each patient
        topk_idx = torch.topk(scores, k=max_k, dim=-1).indices.cpu().numpy()

        # Compute hit or miss per K
        for i, pos in enumerate(pos_lists):
            if not pos:
                # No ground-truth codes for this encounter (shouldn't happen with our filter)
                continue
            pos_set = set(pos)
            for k in k_list:
                hit = any(idx in pos_set for idx in topk_idx[i, :k])
                recalls[k].append(1.0 if hit else 0.0)

    # Aggregate mean recall per K
    out = {}
    for k in k_list:
        if len(recalls[k]) == 0:
            out[k] = 0.0
        else:
            out[k] = float(np.mean(recalls[k]))
    return out
