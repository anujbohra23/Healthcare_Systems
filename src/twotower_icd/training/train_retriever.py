'''
Set seed, device, create output directory. Load all 4 CSVs → icd_raw, patients, labels, splits.

Build:
ICDCatalog from icd_raw
PatientLabelIndex from (labels, code2idx)
Fit EHRScaler on train split only, and save it as ehr_scaler.json.
Instantiate tokenizers for text + ICD.
Build PatientDataset for train and val and their DataLoaders.

Instantiate:
PatientTower
ICDTower
AdamW optimizer on both.

For each epoch:
Loop over train batches:
Encode patients → z_pat
For each patient, pick one positive ICD index, get its text → decode through ICD tower → z_code
Compute InfoNCE loss: info_nce_inbatch(z_pat, z_code)
Backprop + optimizer step.

After epoch:
Pre-embed all ICD codes and save embeddings + mappings.
Save model weights.
So this file is your training script brain, wiring together all the other pieces.

'''






from __future__ import annotations
import os
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

from ..config import DataPaths, TrainingConfig
from ..data.scalers import EHRScaler
from ..data.icd_catalog import ICDCatalog
from ..data.labels_index import PatientLabelIndex
from ..data.datasets import PatientDataset, PatientCollator
from ..models.patient_tower import PatientTower
from ..models.icd_tower import ICDTower
from ..models.loss import info_nce_inbatch


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def embed_all_codes(
    icd_catalog: ICDCatalog,
    tokenizer,
    model: ICDTower,
    device: torch.device,
    max_len_code: int = 64,
    batch_size: int = 128,
) -> torch.Tensor:
    texts = icd_catalog.code_texts()
    all_embeds = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Embed ICD codes"):
            batch_texts = texts[i : i + batch_size]
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_len_code,
                return_tensors="pt",
            ).to(device)
            z = model(inputs)          # [b, d]
            all_embeds.append(z.cpu())
    Z = torch.cat(all_embeds, dim=0)   # [N_codes, d]
    return Z


def train_retriever(
    data_paths: DataPaths,
    cfg: TrainingConfig,
) -> Dict[str, float]:
    os.makedirs(cfg.out_dir, exist_ok=True)
    set_seed(cfg.seed)

    device = torch.device(cfg.device)

    # Load CSVs
    icd_raw = pd.read_csv(data_paths.icd_csv)
    patients = pd.read_csv(data_paths.patients_csv)
    labels = pd.read_csv(data_paths.labels_csv)
    splits = pd.read_csv(data_paths.splits_csv)

    # ICD Catalog & label index
    icd_catalog = ICDCatalog.from_df(icd_raw)
    plindex = PatientLabelIndex.from_df(labels, icd_catalog.code2idx)

    # Fit scaler on TRAIN split only
    train_df = patients.merge(splits, on="encounter_id", how="left")
    train_df = train_df[train_df["split"] == "train"].reset_index(drop=True)
    scaler = EHRScaler.fit_from_df(train_df)
    scaler.save(os.path.join(cfg.out_dir, "ehr_scaler.json"))

    # Tokenizers
    tok_txt = AutoTokenizer.from_pretrained(cfg.txt_backbone, use_fast=True)
    tok_code = AutoTokenizer.from_pretrained(cfg.code_backbone, use_fast=True)

    # Datasets / loaders
    collator = PatientCollator(txt_tokenizer=tok_txt, max_seq_len=cfg.max_seq_len)

    d_train = PatientDataset(patients, splits, "train", scaler, plindex)
    d_val = PatientDataset(patients, splits, "val", scaler, plindex)

    train_loader = DataLoader(
        d_train,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        d_val,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collator,
    )

    # Models
    pat_model = PatientTower(
        txt_model_name=cfg.txt_backbone,
        d=768,
    ).to(device)
    code_model = ICDTower(
        code_model_name=cfg.code_backbone,
        d=768,
    ).to(device)

    optim = torch.optim.AdamW(
        list(pat_model.parameters()) + list(code_model.parameters()),
        lr=cfg.lr,
    )

    for epoch in range(1, cfg.epochs + 1):
        pat_model.train()
        code_model.train()
        losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            txt_inputs = {k: v.to(device) for k, v in batch["txt_inputs"].items()}
            ehr = batch["ehr"].to(device)
            pos_lists = batch["pos_codes"]

            # For now: sample exactly ONE positive code per patient
            code_texts = icd_catalog.code_texts()
            pos_choice = []
            for lst in pos_lists:
                if not lst:
                    pos_choice.append(0)  # fallback
                else:
                    pos_choice.append(np.random.choice(lst))

            pos_texts = [code_texts[i] for i in pos_choice]
            code_inputs = tok_code(
                pos_texts,
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt",
            ).to(device)

            z_pat = pat_model(txt_inputs, ehr)
            z_code = code_model(code_inputs)

            loss = info_nce_inbatch(z_pat, z_code, tau=cfg.tau)
            optim.zero_grad()
            loss.backward()
            optim.step()

            losses.append(loss.item())
            if len(losses) % 50 == 0:
                pbar.set_postfix({"loss": np.mean(losses[-50:])})

        mean_loss = float(np.mean(losses))
        print(f"Epoch {epoch} mean loss: {mean_loss:.4f}")

        # Build full ICD embeddings and save
        Z = embed_all_codes(icd_catalog, tok_code, code_model, device)
        torch.save(
            {
                "embeddings": Z,
                "code2idx": icd_catalog.code2idx,
                "idx2code": icd_catalog.idx2code,
            },
            os.path.join(cfg.out_dir, f"code_embeds_epoch{epoch}.pt"),
        )

        # Save model weights
        torch.save(
            pat_model.state_dict(),
            os.path.join(cfg.out_dir, f"patient_tower_epoch{epoch}.pt"),
        )
        torch.save(
            code_model.state_dict(),
            os.path.join(cfg.out_dir, f"icd_tower_epoch{epoch}.pt"),
        )

    # final save (last epoch aliases)
    torch.save(pat_model.state_dict(), os.path.join(cfg.out_dir, "patient_tower.pt"))
    torch.save(code_model.state_dict(), os.path.join(cfg.out_dir, "icd_tower.pt"))

    return {"mean_train_loss": mean_loss}
