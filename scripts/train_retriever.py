#the train retriever is used to train the model. This is the one where all the things are gonna get combined together. 

import argparse
from twotower_icd.config import DataPaths, TrainingConfig
from twotower_icd.training.train_retriever import train_retriever


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--icd_csv", required=True)
    ap.add_argument("--patients_csv", required=True)
    ap.add_argument("--labels_csv", required=True)
    ap.add_argument("--splits_csv", required=True)
    ap.add_argument("--out_dir", default="./artifacts")
    ap.add_argument("--txt_backbone", default="emilyalsentzer/Bio_ClinicalBERT")
    ap.add_argument("--code_backbone", default="emilyalsentzer/Bio_ClinicalBERT")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_seq_len", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tau", type=float, default=0.07)
    args = ap.parse_args()

    data_paths = DataPaths(
        icd_csv=args.icd_csv,
        patients_csv=args.patients_csv,
        labels_csv=args.labels_csv,
        splits_csv=args.splits_csv,
    )

    cfg = TrainingConfig(
        txt_backbone=args.txt_backbone,
        code_backbone=args.code_backbone,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        lr=args.lr,
        epochs=args.epochs,
        device=args.device,
        out_dir=args.out_dir,
        tau=args.tau,
    )

    metrics = train_retriever(data_paths, cfg)
    print("Training finished:", metrics)


if __name__ == "__main__":
    main()
