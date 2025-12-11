# this is a central place for all hyperparams and file paths
from dataclasses import dataclass
from typing import Optional, List


'''
this contains all the file paths required for data, if wewant to pass data then
we use DataPaths instead of diff strs for each one of them
'''
@dataclass
class DataPaths:
    icd_csv: str
    patients_csv: str
    labels_csv: str
    splits_csv: str



'''
this contains all training params, we can modify if req here
'''
@dataclass
class TrainingConfig:
    txt_backbone: str = "emilyalsentzer/Bio_ClinicalBERT"
    code_backbone: str = "emilyalsentzer/Bio_ClinicalBERT"
    batch_size: int = 32
    max_seq_len: int = 512
    lr: float = 2e-5
    epochs: int = 1
    device: str = "cuda"
    out_dir: str = "./artifacts"
    num_workers: int = 2
    tau: float = 0.07           # temperature for contrastive loss
    seed: int = 42
    eval_k_list: Optional[List[int]] = None

    def __post_init__(self):
        if self.eval_k_list is None:
            self.eval_k_list = [50, 100, 200]
