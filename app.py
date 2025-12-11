#!/usr/bin/env python
"""
TwoTower-ICD Starter (full)

End-to-end minimal pipeline for ICD retrieve:
  1) Load CSVs (ICD catalog, patients, labels, splits)
  2) Preprocess (scale EHR, tokenize text)
  3) Train Two-Tower (contrastive, in-batch negatives)
  4) Build ICD embeddings, evaluate Recall@K, (optionally) build FAISS index

Usage (example):
  python twotower_icd_starter.py \
    --icd_csv /path/icd_codes_8k.csv \
    --patients_csv /path/patients_scaled.csv \
    --labels_csv /path/labels_scaled.csv \
    --splits_csv /path/splits_scaled.csv \
    --out_dir ./artifacts \
    --epochs 1 --batch_size 32 --max_seq_len 512 \
    --device mps     # (on Apple silicon); omit on CUDA machines

Expected columns:
  icd_csv:      code, title, description, synonyms, level (category/leaf)
  patients_csv: encounter_id, note_text, age, sex, lab_* (see LAB_KEYS below)
  labels_csv:   encounter_id, code
  splits_csv:   encounter_id, split (train/val/test)
"""
import os
import json
import math
import random
import argparse
from typing import List, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel

# -------------------------
# Repro & small utils
# -------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

# -------------------------
# Data loading & preprocessing
# -------------------------
LAB_KEYS = [
    "a1c", "glucose", "creatinine", "egfr", "ldl", "hdl", "triglycerides",
    "wbc", "hgb", "platelets", "crp", "troponin", "bnp", "alt", "ast"
]

class EHRScaler:
    """Z-score scaler for lab_* with robust NaN handling + age/sex features."""
    def __init__(self):
        self.means = None
        self.stds  = None

    def fit(self, df: pd.DataFrame):
        X = df[[f"lab_{k}" for k in LAB_KEYS]].apply(pd.to_numeric, errors="coerce").values
        self.means = np.nanmean(X, axis=0)
        self.stds  = np.nanstd(X, axis=0) + 1e-6

    def transform_row(self, row: pd.Series) -> np.ndarray:
        labs = np.array([
            pd.to_numeric(row.get(f"lab_{k}"), errors="coerce") for k in LAB_KEYS
        ], dtype=np.float32)
        # replace NaNs with train means
        labs = np.where(np.isnan(labs), self.means, labs)
        labs = (labs - self.means) / self.stds

        age = float(pd.to_numeric(row.get("age", 0.0), errors="coerce") or 0.0) / 100.0
        sex_raw = str(row.get("sex", "M")).upper()
        sex = 1.0 if sex_raw.startswith("M") else 0.0

        return np.concatenate([labs, [age, sex]]).astype(np.float32)

class ICDCatalog:
    """Builds retrievable ICD items (prefer leaves)."""
    def __init__(self, df: pd.DataFrame):
        leaves = df[df.get("level", "leaf") == "leaf"].copy()
        if leaves.empty:
            leaves = df.copy()
        leaves["text"] = (
            leaves.get("title", "").astype(str).fillna("") + " \n" +
            leaves.get("description", "").astype(str).fillna("") + " \n" +
            leaves.get("synonyms", "").astype(str).fillna("")
        )
        self.items = leaves[["code", "text"]].reset_index(drop=True)
        self.code2idx = {c: i for i, c in enumerate(self.items["code"].tolist())}
        self.idx2code = {i: c for c, i in self.code2idx.items()}

class PatientLabelIndex:
    """Map encounter_id -> list[int(code_idx)] for positives."""
    def __init__(self, labels_df: pd.DataFrame, code2idx: Dict[str, int]):
        enc2codes: Dict[int, List[int]] = {}
        for enc, g in labels_df.groupby("encounter_id"):
            idxs = [code2idx[c] for c in g["code"].tolist() if c in code2idx]
            if idxs:
                enc2codes[int(enc)] = sorted(set(idxs))
        self.enc2codes = enc2codes

    def positives(self, enc_id: int) -> List[int]:
        return self.enc2codes.get(int(enc_id), [])

class PatientDataset(Dataset):
    """Yields fused inputs (text + EHR vector) and positive code indices."""
    def __init__(self, patients_df, splits_df, split, scaler: EHRScaler, plindex: PatientLabelIndex):
        df = patients_df.merge(splits_df, on="encounter_id", how="left")
        df = df[df["split"] == split].reset_index(drop=True)
        df = df[df["encounter_id"].isin(plindex.enc2codes.keys())].reset_index(drop=True)
        self.df = df
        self.scaler = scaler
        self.plindex = plindex

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        enc_id = int(row["encounter_id"])
        ehr_vec = self.scaler.transform_row(row)
        note_text = str(row.get("note_text", ""))[:20000]
        pos_codes = self.plindex.positives(enc_id)
        return {
            "encounter_id": enc_id,
            "text": note_text,
            "ehr": ehr_vec,
            "pos_codes": pos_codes,
        }

# -------------------------
# Models
# -------------------------
class PatientTower(nn.Module):
    """Text encoder (HF) + EHR MLP → gated fusion → L2-normalized vector."""
    def __init__(self, txt_model: str = "distilbert-base-uncased",
                 ehr_dim: int = len(LAB_KEYS)+2, d: int = 768):
        super().__init__()
        self.txt = AutoModel.from_pretrained(txt_model)
        h = self.txt.config.hidden_size
        self.txt_proj = nn.Linear(h, d)
        self.ehr_proj = nn.Sequential(
            nn.Linear(ehr_dim, d), nn.ReLU(), nn.Linear(d, d)
        )
        self.gate = nn.Linear(2*d, d)

    def forward(self, txt_inputs, ehr_vec):
        h_cls = self.txt(**txt_inputs).last_hidden_state[:, 0]     # [B,h]
        zt = F.normalize(self.txt_proj(h_cls), dim=-1)             # [B,d]
        ze = F.normalize(self.ehr_proj(ehr_vec), dim=-1)           # [B,d]
        z  = torch.tanh(self.gate(torch.cat([zt, ze], dim=-1)))    # [B,d]
        return F.normalize(z, dim=-1)

class CodeTower(nn.Module):
    """ICD text encoder (HF) → projection → L2-normalized vector."""
    def __init__(self, code_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 d: int = 768):
        super().__init__()
        self.enc = AutoModel.from_pretrained(code_model)
        h = self.enc.config.hidden_size
        self.proj = nn.Linear(h, d)

    def forward(self, code_inputs):
        h_cls = self.enc(**code_inputs).last_hidden_state[:, 0]
        return F.normalize(self.proj(h_cls), dim=-1)

# -------------------------
# Batching, loss, embedding, eval
# -------------------------
class Collator:
    def __init__(self, tok_txt, tok_code, max_len_txt=512, max_len_code=64):
        self.tok_txt = tok_txt
        self.tok_code = tok_code
        self.max_len_txt = max_len_txt
        self.max_len_code = max_len_code

    def __call__(self, batch):
        texts = [b["text"] for b in batch]
        ehr = torch.tensor(np.stack([b["ehr"] for b in batch]), dtype=torch.float32)
        pos_codes = [b["pos_codes"] for b in batch]
        txt_inputs = self.tok_txt(
            texts, padding=True, truncation=True, max_length=self.max_len_txt, return_tensors="pt"
        )
        return {"txt_inputs": txt_inputs, "ehr": ehr, "pos_codes": pos_codes}

def info_nce_inbatch(zp: torch.Tensor, zc: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
    """
    In-batch negatives. Assumes positives are row-aligned: sim[i, i] is the target.
    """
    sim = (zp @ zc.T) / tau            # [B, B]
    labels = torch.arange(zp.size(0), device=zp.device)
    return F.cross_entropy(sim, labels)

def embed_all_codes(icd_items: pd.DataFrame, tok_code, code_model: CodeTower,
                    device, max_len_code=64, batch_size=256) -> torch.Tensor:
    texts = icd_items["text"].tolist()
    outs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embed ICD codes"):
        chunk = texts[i:i+batch_size]
        inp = tok_code(chunk, padding=True, truncation=True, max_length=max_len_code, return_tensors="pt").to(device)
        with torch.no_grad():
            z = code_model(inp)
        outs.append(z.cpu())
    Z = torch.cat(outs, dim=0)
    return F.normalize(Z, dim=-1)

def evaluate_recall_at_k(z_codes: torch.Tensor,
                         loader: DataLoader,
                         patient_model: PatientTower,
                         device,
                         k_list=(50, 100, 200)) -> Dict[int, float]:
    patient_model.eval()
    recalls = {k: [] for k in k_list}
    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval Recall@K"):
            txt_inputs = {k: v.to(device) for k, v in batch["txt_inputs"].items()}
            ehr = batch["ehr"].to(device)
            pos_lists = batch["pos_codes"]
            zp = patient_model(txt_inputs, ehr)       # [B,d]
            scores = zp @ z_codes.T                   # [B,N]
            topk = torch.topk(scores, k=max(k_list), dim=-1).indices.cpu().numpy()
            for i, pos in enumerate(pos_lists):
                if not pos:
                    continue
                pos_set = set(pos)
                for k in k_list:
                    hit = any((idx in pos_set) for idx in topk[i, :k])
                    recalls[k].append(1.0 if hit else 0.0)
    return {k: (float(np.mean(recalls[k])) if recalls[k] else 0.0) for k in k_list}

# -------------------------
# Main
# -------------------------
def main():
    default_device = (
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    ap = argparse.ArgumentParser()
    ap.add_argument("--icd_csv", required=True)
    ap.add_argument("--patients_csv", required=True)
    ap.add_argument("--labels_csv", required=True)
    ap.add_argument("--splits_csv", required=True)
    ap.add_argument("--out_dir", default="./artifacts")
    ap.add_argument("--txt_backbone", default="distilbert-base-uncased")
    ap.add_argument("--code_backbone", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_seq_len", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--device", default=default_device)
    ap.add_argument("--num_workers", type=int, default=0)  # 0 is safest on Mac
    ap.add_argument("--build_faiss", action="store_true", help="Build a FAISS index for ICD vectors (requires faiss-cpu).")
    args = ap.parse_args()

    set_seed(SEED)
    ensure_dir(args.out_dir)
    device = torch.device(args.device)

    # Load CSVs
    icd_raw   = pd.read_csv(args.icd_csv)
    patients  = pd.read_csv(args.patients_csv)
    labels    = pd.read_csv(args.labels_csv)
    splits    = pd.read_csv(args.splits_csv)

    # Catalog & labels
    icd = ICDCatalog(icd_raw)
    plindex = PatientLabelIndex(labels, icd.code2idx)

    # Scaler (train-only fit)
    train_df = patients.merge(splits, on="encounter_id", how="left")
    train_df = train_df[train_df["split"] == "train"].reset_index(drop=True)
    scaler = EHRScaler()
    scaler.fit(train_df)

    # Tokenizers
    tok_txt  = AutoTokenizer.from_pretrained(args.txt_backbone, use_fast=True)
    tok_code = AutoTokenizer.from_pretrained(args.code_backbone, use_fast=True)

    # Datasets & loaders
    collator = Collator(tok_txt, tok_code, max_len_txt=args.max_seq_len, max_len_code=64)
    d_train  = PatientDataset(patients, splits, "train", scaler, plindex)
    d_val    = PatientDataset(patients, splits, "val",   scaler, plindex)

    L_train = DataLoader(d_train, batch_size=args.batch_size, shuffle=True,
                         num_workers=args.num_workers, collate_fn=collator)
    L_val   = DataLoader(d_val,   batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, collate_fn=collator)

    # Models & optimizer
    pat = PatientTower(args.txt_backbone, ehr_dim=len(LAB_KEYS)+2, d=768).to(device)
    cod = CodeTower(args.code_backbone, d=768).to(device)
    opt = torch.optim.AdamW(list(pat.parameters()) + list(cod.parameters()), lr=args.lr)

    # Cache ICD texts once
    icd_texts = icd.items["text"].tolist()

    # Train
    for epoch in range(1, args.epochs + 1):
        pat.train(); cod.train()
        losses = []
        pbar = tqdm(L_train, desc=f"Epoch {epoch}")
        for batch in pbar:
            txt_inputs = {k: v.to(device) for k, v in batch["txt_inputs"].items()}
            ehr = batch["ehr"].to(device)
            pos_lists = batch["pos_codes"]

            # choose one positive per sample (fallback to random if empty)
            chosen = []
            for lst in pos_lists:
                if lst:
                    chosen.append(random.choice(lst))
                else:
                    chosen.append(random.randrange(len(icd_texts)))
            pos_texts = [icd_texts[i] for i in chosen]
            code_inputs = tok_code(pos_texts, padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)

            zp = pat(txt_inputs, ehr)           # [B,d]
            zc = cod(code_inputs)               # [B,d] (aligned positives)
            loss = info_nce_inbatch(zp, zc, tau=0.07)

            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
            if len(losses) % 50 == 0:
                pbar.set_postfix({"loss": sum(losses[-50:]) / 50.0})

        print(f"Epoch {epoch} mean loss: {np.mean(losses):.4f}")

        # Embed all ICD codes
        cod.eval()
        Z = embed_all_codes(icd.items, tok_code, cod, device, batch_size=256)
        Z = Z.to(device)
        torch.save({
            "embeddings": Z,
            "code2idx": icd.code2idx,
            "idx2code": icd.idx2code
        }, os.path.join(args.out_dir, f"code_embeds_epoch{epoch}.pt"))

        # Evaluate Recall@K on VAL
        rks = evaluate_recall_at_k(Z, L_val, pat, device, k_list=(50, 100, 200))
        print({f"Recall@{k}": v for k, v in rks.items()})

    # Save models + scaler + config
    torch.save(pat.state_dict(), os.path.join(args.out_dir, "patient_tower.pt"))
    torch.save(cod.state_dict(), os.path.join(args.out_dir, "code_tower.pt"))
    with open(os.path.join(args.out_dir, "ehr_scaler.json"), "w") as f:
        json.dump({"means": list(map(float, scaler.means)), "stds": list(map(float, scaler.stds))}, f, indent=2)
    with open(os.path.join(args.out_dir, "run_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Optional: FAISS index (inner product)
    if args.build_faiss:
        try:
            import faiss  # pip install faiss-cpu
            Z_np = Z.cpu().numpy().astype("float32")
            faiss.normalize_L2(Z_np)
            index = faiss.IndexFlatIP(Z_np.shape[1])
            index.add(Z_np)
            faiss.write_index(index, os.path.join(args.out_dir, "faiss_ip.index"))
            print("FAISS index written.")
        except Exception as e:
            print("FAISS not built:", e)

    print("Done. Artifacts in:", args.out_dir)

if __name__ == "__main__":
    main()
