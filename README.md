---

# Two-Tower ICD Code Retrieval System

A scalable retrieval–re-ranking architecture for automated ICD-10 coding using EHR structured data, clinical text, and transformer-based embeddings.

This repository implements a **production-ready** version of the model described in the research work. It includes

* A **Patient Tower** combining BERT-based clinical text embeddings + MLP-based structured EHR embeddings
* An **ICD Tower** for encoding 8K+ ICD-10 disease descriptions
* A **contrastive retrieval objective** (multi-positive InfoNCE) for learning a shared embedding space
* **ANN retrieval pipeline** (via vector search using precomputed ICD embeddings)
* A modular, extensible codebase designed for deployment in hospitals or cloud systems

---

# Overview of the Architecture

The system follows a **retrieval → re-ranking pipeline**, inspired by large-scale recommender systems and adapted for healthcare coding.

## 1. Patient Tower (User Tower)

Encodes an individual patient encounter into a **single dense embedding (size 768)** by combining:

### **a. Clinical Notes Encoder**

* Pretrained language model (e.g., **BioClinicalBERT**, DistilBERT)
* Input: patient’s discharge summary / clinical notes
* Output: CLS embedding → projected to 768-dim → L2-normalized

### **b. Structured EHR Encoder**

* Lab values (A1C, creatinine, LDL, CRP, troponin, etc.)
* Demographics (age, sex)
* Passed through an MLP → projected to 768-dim → L2-normalized

### **c. Fusion (Gated Combination)**

The two embeddings are fused via a gating mechanism:

```
z = tanh( W [z_text ; z_ehr] )
```

Result → normalized 768-d patient embedding.

---

## 2. ICD Tower (Item Tower)

Encodes each ICD-10 disease description into a **dense 768-dim embedding**.

* Uses a text encoder (MiniLM / ClinicalBERT / DistilBERT)
* Input: ICD title + long description + synonyms
* Precomputed once and stored as:

  ```
  code_embeds_epochX.pt
  ```
* Enables fast **ANN vector search** for retrieval.

---

## 3. Contrastive Retrieval Objective (Multi-Positive InfoNCE)

During training, each patient may have **multiple ICD codes** (multi-label classification).

Instead of picking a single positive, this training code uses **multi-positive contrastive learning**:

* All correct ICD codes = *positives*
* All other ICD codes in the batch = *in-batch negatives*
* Loss pushes patient embedding toward its true codes and away from incorrect ones

This improves retrieval significantly compared to single-positive InfoNCE.

---

## 4. ANN Retrieval (Nearest Neighbor Search)

At inference:

1. A patient is encoded → `z_patient`
2. ICD embedding matrix `Z_icd` is loaded (N × 768)
3. Compute similarity:

   ```
   scores = z_patient · Z_icd^T
   ```
4. Retrieve top-K codes (e.g., K = 100)

These top-K codes are later passed to a **re-ranking model (LLM)** in the full paper pipeline.

---

# Project Directory Structure

```
deployable_twotower/
│
├── src/twotower_icd/
│   ├── data/
│   │   ├── datasets.py         # PatientDataset + Collator
│   │   ├── scalers.py          # EHRScaler for structured data
│   │   ├── icd_catalog.py      # ICDCatalog: codes, text, indices
│   │   ├── labels_index.py     # Patient → ICD label mapping
│   │
│   ├── models/
│   │   ├── patient_tower.py    # ClinicalBERT + EHR fusion encoder
│   │   ├── icd_tower.py        # ICD text encoder
│   │
│   ├── training/
│   │   ├── train_retriever.py  # Training loop (multi-positive InfoNCE)
│   │   ├── eval_retriever.py   # Evaluate Recall@K
│   │
│   ├── utils/
│   │   ├── metrics.py          # recall@K implementation
│   │
│   ├── config.py               # DataPaths + TrainingConfig dataclasses
│
├── scripts/
│   ├── train_retriever.py      # CLI wrapper for training
│   ├── eval_retriever.py       # CLI wrapper for evaluation
│
├── artifacts/                  # Saved checkpoints & ICD embeddings
│   ├── ehr_scaler.json
│   ├── patient_tower.pt
│   ├── code_embeds_epoch1.pt
│
├── requirements.txt
└── README.md  
```

---

# Training the Retriever

### **1. Prepare your virtual environment**

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### **2. Train the Two-Tower Model**

Example:

```bash
python scripts/train_retriever.py \
  --icd_csv Data/icd_codes_8k.csv \
  --patients_csv Data/patients_scaled.csv \
  --labels_csv Data/labels_scaled.csv \
  --splits_csv Data/splits_scaled.csv \
  --out_dir ./artifacts_clinical \
  --txt_backbone emilyalsentzer/Bio_ClinicalBERT \
  --code_backbone distilbert-base-uncased \
  --batch_size 4 \
  --max_seq_len 256 \
  --epochs 3 \
  --device mps
```

### During training, the system:

1. Loads structured + unstructured patient data
2. Builds ICD catalog
3. Trains **Patient Tower + ICD Tower jointly**
4. Uses **multi-positive InfoNCE** for retrieval
5. After each epoch:

   * Computes ICD embeddings
   * Saves code embedding file
   * Evaluates **Recall@50/100/200**
   * Saves model checkpoints

---

# Evaluation (Recall@K)

After training:

```bash
python scripts/eval_retriever.py \
  --icd_csv Data/icd_codes_8k.csv \
  --patients_csv Data/patients_scaled.csv \
  --labels_csv Data/labels_scaled.csv \
  --splits_csv Data/splits_scaled.csv \
  --out_dir ./artifacts_clinical \
  --device mps \
  --split test \
  --txt_backbone emilyalsentzer/Bio_ClinicalBERT
```

Output:

```
Evaluation on split='test':
  Recall@50: 0.1190
  Recall@100: 0.2200
  Recall@200: 0.3843
```

This means:

* In top-50 retrieved codes → ~12% contain a correct ICD
* In top-200 → ~38% accuracy
* These numbers improve dramatically with:

  * more epochs
  * larger batch sizes
  * multi-positive InfoNCE
  * hard-negative sampling
  * clinicalBERT backbones

---

# How Retrieval Works Internally

### **1. Build patient embedding**

```
ClinicalBERT(text) → z_text (768)
MLP(labs+demographics) → z_ehr (768)
z_patient = fuse(z_text, z_ehr) → normalized 768-d vector
```

### **2. Build ICD embedding**

```
ICD text → transformer encoder → 768-d vector
```

### **3. Contrastive Training**

For each patient:

* Positives = all ICD codes assigned to that encounter
* Negatives = all ICDs from other samples in the batch
* Loss pulls positives closer, pushes negatives away

### **4. Retrieval**

Compute:

```
scores = dot(z_patient, Z_codes)
```

Sort scores → take top-K.

---

