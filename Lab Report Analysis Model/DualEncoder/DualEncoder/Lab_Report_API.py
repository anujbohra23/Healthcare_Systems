# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
import pdfplumber
import re
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np


# --- Model Architecture ---
class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(out_features, out_features)
        self.dropout = nn.Dropout(dropout)
        self.shortcut = (
            nn.Identity()
            if in_features == out_features
            else nn.Linear(in_features, out_features)
        )

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return self.relu(out + self.shortcut(x))


class DualEncoderModel(nn.Module):
    def __init__(self, lab_cont_dim=18, conv_cat_dims=[49, 17, 17], num_classes=18):
        super().__init__()
        self.lab_cont_encoder = nn.Sequential(
            ResidualBlock(lab_cont_dim, 64), ResidualBlock(64, 64)
        )
        self.conv_cat_embeddings = nn.ModuleList(
            [nn.Embedding(dim + 1, 16) for dim in conv_cat_dims]
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 + 16 * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, lab_cont, conv_cat):
        lab_emb = self.lab_cont_encoder(lab_cont)
        conv_embs = [
            emb(conv_cat[:, i]) for i, emb in enumerate(self.conv_cat_embeddings)
        ]
        fused = torch.cat([lab_emb] + conv_embs, dim=1)
        return self.classifier(fused)


# --- Helper Functions ---
def extract_text_from_pdf(pdf_path: str) -> str:
    with pdfplumber.open(pdf_path) as pdf:
        return "\n".join(
            page.extract_text() for page in pdf.pages if page.extract_text()
        )


def extract_lab_values(text: str) -> dict:
    pattern = r"([A-Za-z\s/()-]+):\s*([\d.]+)\s*([A-Za-z/%]+)?"
    matches = re.findall(pattern, text)
    return {test.strip(): float(value) for test, value, _ in matches}


def prepare_input(lab_data: dict) -> tuple:
    lab_features = [
        "ALT (SGPT)",
        "AST (SGOT)",
        "Bilirubin",
        "Albumin",
        "Platelet Count",
        "Total Cholesterol",
        "BP Systolic",
        "BP Diastolic",
        "Troponin",
        "Ejection Fraction",
        "HbA1c",
        "Fasting Glucose",
        "Postprandial Glucose",
        "Triglycerides",
        "Insulin Level",
        "WBC Count",
        "Fever",
        "Hematocrit",
    ]
    lab_tensor = torch.tensor(
        [[lab_data.get(f, -1) for f in lab_features]], dtype=torch.float32
    )
    conv_tensor = torch.zeros((1, 3), dtype=torch.long)  # Dummy conversation categories
    return lab_tensor, conv_tensor


# --- FastAPI Setup ---
app = FastAPI(title="Medical Severity Predictor")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model
model = DualEncoderModel()
model_path = os.getenv("MODEL_PATH", "dual_encoder_model.pth")
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()


# --- API Endpoints ---
class PredictionResult(BaseModel):
    diagnoses: list
    confidence: float


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResult)
async def predict_from_pdf(file: UploadFile = File(...)):
    try:
        # Save uploaded file
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        # Process PDF
        text = extract_text_from_pdf(temp_path)
        lab_data = extract_lab_values(text)

        # Prepare model input
        lab_tensor, conv_tensor = prepare_input(lab_data)

        # Make prediction
        with torch.no_grad():
            logits = model(lab_tensor, conv_tensor)
            probs = torch.softmax(logits, dim=1)
            top_probs, top_indices = torch.topk(probs, 3)

        # Format results
        diagnoses = [f"ICD-{idx.item()}" for idx in top_indices[0]]

        # Cleanup
        os.remove(temp_path)

        return {"diagnoses": diagnoses, "confidence": top_probs[0][0].item()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
