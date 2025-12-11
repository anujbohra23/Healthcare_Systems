# scripts/eval_retriever.py
import argparse

from twotower_icd.config import DataPaths, TrainingConfig
from twotower_icd.training.eval_retriever import eval_retriever


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--icd_csv", required=True)
    ap.add_argument("--patients_csv", required=True)
    ap.add_argument("--labels_csv", required=True)
    ap.add_argument("--splits_csv", required=True)
    ap.add_argument("--out_dir", default="./artifacts")
    ap.add_argument("--txt_backbone", default="emilyalsentzer/Bio_ClinicalBERT")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_seq_len", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--split", choices=["val", "test"], default="val")
    args = ap.parse_args()

    data_paths = DataPaths(
        icd_csv=args.icd_csv,
        patients_csv=args.patients_csv,
        labels_csv=args.labels_csv,
        splits_csv=args.splits_csv,
    )

    cfg = TrainingConfig(
        txt_backbone=args.txt_backbone,
        code_backbone=args.txt_backbone,  # not needed here, but keep consistent
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        device=args.device,
        out_dir=args.out_dir,
        num_workers=args.num_workers,
    )

    metrics = eval_retriever(data_paths, cfg, split=args.split)
    print(f"Evaluation on split='{args.split}':")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
