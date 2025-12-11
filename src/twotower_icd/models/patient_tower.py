'''
Here, the constructor loads clinicalBert and builds the ehr mlp and fusion layers. 
The forward pass encodes textsand lab vlaues and fuses them using concatenation'''


from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from ..data.constants import LAB_KEYS


class PatientTower(nn.Module):
    """
    ClinicalBERT text encoder + EHR MLP + gated fusion.
    Output: patient embedding [B, d], L2-normalized.
    """

    def __init__(
        self,
        txt_model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        ehr_dim: int = len(LAB_KEYS) + 2,
        d: int = 768,
    ):
        super().__init__()
        self.txt_encoder = AutoModel.from_pretrained(txt_model_name)
        hidden_txt = self.txt_encoder.config.hidden_size

        self.txt_proj = nn.Linear(hidden_txt, d)

        self.ehr_proj = nn.Sequential(
            nn.Linear(ehr_dim, d),
            nn.ReLU(),
            nn.Linear(d, d),
        )

        self.gate = nn.Linear(2 * d, d)

    def forward(self, txt_inputs: dict, ehr_vec: torch.Tensor) -> torch.Tensor:
        # txt_inputs: dict of tensors (input_ids, attention_mask, etc.)
        txt_outputs = self.txt_encoder(**txt_inputs)
        cls = txt_outputs.last_hidden_state[:, 0]  # [B, hidden_txt]

        zt = F.normalize(self.txt_proj(cls), dim=-1)
        ze = F.normalize(self.ehr_proj(ehr_vec), dim=-1)

        concat = torch.cat([zt, ze], dim=-1)       # [B, 2d]
        z = torch.tanh(self.gate(concat))          # [B, d]
        z = F.normalize(z, dim=-1)
        return z
