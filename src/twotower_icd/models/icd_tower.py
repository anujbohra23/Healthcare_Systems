'''
This loads BioClinicalBERT and builds the ICD encoder, forward pass encodes ICD text
using ClinicalBERT and projects to d-dim embedding, L2-normalized. 
'''



from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel 

class ICDTower(nn.Module):
    def __init__(self, code_model_name: str = "emilyalsentzer/Bio_ClinicalBERT", d: int = 768):
        super().__init__()
        self.enc = AutoModel.from_pretrained(code_model_name)
        hidden = self.enc.config.hidden_size
        self.proj = nn.Linear(hidden, d)

    def forward(self, code_inputs: dict) -> torch.Tensor:
        outputs = self.enc(**code_inputs)
        cls = outputs.last_hidden_state[:, 0]      # [B, hidden]
        z = self.proj(cls)                         # [B, d]
        z = F.normalize(z, dim=-1)
        return z
    
