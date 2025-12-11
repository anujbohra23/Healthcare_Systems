# src/twotower_icd/training/eval_retriever.py
from __future__ import annotations
import os
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from ..config import DataPaths, TrainingConfig
from ..data.scalers import EHRScaler
from ..data.icd_catalog import ICDCatalog
from ..data.labels_index import PatientLabelIndex
from ..data.datasets import PatientDataset, PatientCollator
from ..models.patient_tower import PatientTower
from ..utils.metrics import evaluate_recall_at_k


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def eval_retriever(
    data_paths: DataPaths,
    cfg: TrainingConfig,
    split: str = "val",
) -> Dict[str, float]:
    """
    Evaluate a trained patient tower + precomputed ICD embeddings on a given split.
    """

    assert split in {"val", "test"}, "split must be 'val' or 'test'"
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    # 1. Load data
    icd_raw = pd.read_csv(data_paths.icd_csv)
    patients = pd.read_csv(data_paths.patients_csv)
    labels = pd.read_csv(data_paths.labels_csv)
    splits = pd.read_csv(data_paths.splits_csv)

    # 2. Rebuild ICD catalog + label index to match training mapping
    icd_catalog = ICDCatalog.from_df(icd_raw)
    plindex = PatientLabelIndex.from_df(labels, icd_catalog.code2idx)

    # 3. Load scaler
    scaler_path = os.path.join(cfg.out_dir, "ehr_scaler.json")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found at {scaler_path}")
    scaler = EHRScaler.load(scaler_path)

    # 4. Tokenizer for patient notes
    tok_txt = AutoTokenizer.from_pretrained(cfg.txt_backbone, use_fast=True)

    # 5. Build dataset/loader for requested split
    ds = PatientDataset(patients, splits, split, scaler, plindex)
    collator = PatientCollator(txt_tokenizer=tok_txt, max_seq_len=cfg.max_seq_len)
    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collator,
    )

    # 6. Load patient model weights
    pat_model = PatientTower(
        txt_model_name=cfg.txt_backbone,
        d=768,
    ).to(device)

    # Default: use final checkpoint "patient_tower.pt"
    ckpt_path = os.path.join(cfg.out_dir, "patient_tower.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Patient tower checkpoint not found at {ckpt_path}")
    pat_model.load_state_dict(torch.load(ckpt_path, map_location=device))

    # 7. Load ICD embeddings (pick the latest code_embeds_epoch*.pt)
    files = [f for f in os.listdir(cfg.out_dir) if f.startswith("code_embeds_epoch")]
    if not files:
        raise FileNotFoundError(f"No code_embeds_epoch*.pt found in {cfg.out_dir}")
    files_sorted = sorted(files)
    best = files_sorted[-1]  # last one (highest epoch)
    code_ckpt_path = os.path.join(cfg.out_dir, best)
    ckpt = torch.load(code_ckpt_path, map_location="cpu")
    Z = ckpt["embeddings"]  # [N_codes, d], already normalized

    # 8. Compute Recall@K
    metrics = evaluate_recall_at_k(
        z_codes=Z,
        loader=dl,
        model=pat_model,
        device=device,
        k_list=cfg.eval_k_list,
    )

    # Wrap with nicer keys
    out = {f"Recall@{k}": v for k, v in metrics.items()}
    return out
