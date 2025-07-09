from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from transformers import BertTokenizer, BertModel
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder

app = FastAPI(
    title="Medical Triage API",
    description="API for emergency triage classification",
    version="1.0.0",
)


# ====== Models ======
class TriageRequest(BaseModel):
    text: str
    age: int
    gender: str


class ESIRequest(BaseModel):
    text: str
    age: int
    gender: str


class HealthClassRequest(BaseModel):
    text: str


class TriageResponse(BaseModel):
    esi_level: int
    confidence: float
    health_type: str
    triage_output: str
    selected_method: str


class ESIResponse(BaseModel):
    esi_level: int
    confidence: float


class HealthClassResponse(BaseModel):
    health_type: str
    explanation: str
    final_decision: str


class ESIClassifier(nn.Module):
    def __init__(self, hidden_dim=128, num_classes=5):
        super(ESIClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.gender_embed = nn.Embedding(3, 4)
        self.fc = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size + 4 + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, input_ids, attention_mask, age, gender):
        bert_out = self.bert(
            input_ids=input_ids, attention_mask=attention_mask
        ).pooler_output
        gender_embed = self.gender_embed(gender)
        combined = torch.cat((bert_out, gender_embed, age.unsqueeze(1)), dim=1)
        logits = self.fc(combined)
        return logits


# ====== Model Loading Functions ======
def load_esi_model(model_path: str):
    """Load the ESI classification model with better error handling"""
    print(f"Attempting to load model from: {model_path}")

    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}")
        raise FileNotFoundError(f"Model file {model_path} not found")

    # Define the model architecture
    model = ESIClassifier()

    # Load the model weights
    try:
        state_dict = torch.load(model_path, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"ERROR loading model weights: {str(e)}")
        raise

    model.eval()
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    return model, tokenizer


# ====== Inference Functions ======
def predict_esi(
    model_path: str, test_text: str, test_age: int, test_gender: str
) -> Tuple[int, float]:
    """Predict ESI level using the neural network model"""
    try:
        # Create a proper absolute path for the model file
        model_path = os.path.abspath(model_path)
        print(f"Using model path: {model_path}")

        # Load the model and tokenizer
        model, tokenizer = load_esi_model(model_path)
        device = torch.device("cpu")  # Force CPU for stability
        model = model.to(device)

        # Format input like the training data
        formatted_text = (
            f"Patient Description: {test_text} Age: {test_age} Gender: {test_gender}"
        )
        print(f"Input text: {formatted_text}")

        # Tokenize input
        encoding = tokenizer(
            formatted_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128,
        )

        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        # Prepare age and gender inputs
        age_tensor = torch.tensor([test_age], dtype=torch.float).to(device)

        gender_encoder = LabelEncoder()
        gender_encoder.fit(["Male", "Female", "Other"])
        gender_tensor = torch.tensor(
            [gender_encoder.transform([test_gender])[0]], dtype=torch.long
        ).to(device)

        # Make prediction
        with torch.no_grad():
            logits = model(input_ids, attention_mask, age_tensor, gender_tensor)
            probs = torch.softmax(logits, dim=1).squeeze()
            predicted_class = torch.argmax(probs).item()
            confidence = probs[predicted_class].item()

        # Convert to ESI level (adjusting for zero-indexing)
        esi_label = predicted_class + 1
        confidence_score = float(confidence)  # Ensure it's a Python float, not tensor

        print(
            f"ESI prediction successful: level={esi_label}, confidence={confidence_score:.4f}"
        )

        return esi_label, confidence_score
    except Exception as e:
        print(f"Error in ESI prediction: {str(e)}")
        import traceback

        traceback.print_exc()
        return 3, 0.0


def classify_health_type(text: str) -> Dict[str, Any]:
    """Classify whether input describes mental health or physical health issue"""
    # This is a simplified version since we don't have access to the LLM API in this example
    # In a real implementation, this would call your LLM service

    # Placeholder logic - would be replaced with actual LLM call
    mental_health_keywords = [
        "anxiety",
        "depression",
        "stress",
        "mental",
        "psychiatric",
        "panic",
        "mood",
        "thoughts",
        "suicide",
        "behavior",
    ]

    has_mental_indicators = any(
        keyword in text.lower() for keyword in mental_health_keywords
    )

    if has_mental_indicators:
        health_type = "Mental Health"
        explanation = "Text contains keywords related to mental health conditions."
    else:
        health_type = "Physical Health"
        explanation = "Text primarily describes physical symptoms or conditions."

    return {
        "health_type": health_type,
        "explanation": explanation,
        "final_decision": health_type,
    }


def full_triage_classification(text: str, age: int, gender: str) -> Dict[str, Any]:
    """Perform full triage classification combining ESI and health type"""
    # Get ESI classification
    try:
        # Update the model path to an absolute path or confirm it exists
        model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "esi_classifier.pt"
        )
        print(f"Looking for model at: {model_path}")

        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}, trying default path...")
            model_path = "esi_classifier.pt"

        esi_label, confidence = predict_esi(model_path, text, age, gender)
        print(f"ESI prediction successful: level={esi_label}, confidence={confidence}")
    except Exception as e:
        print(f"Error in ESI classifier: {str(e)}")
        import traceback

        traceback.print_exc()
        esi_label, confidence = 3, 0.0  # Default to mid-level ESI

    # Get health type classification
    try:
        health_result = classify_health_type(text)
        health_type = health_result["final_decision"]
        health_explanation = health_result["explanation"]
    except Exception as e:
        print(f"Error in health type classifier: {str(e)}")
        health_type = "Unknown"
        health_explanation = f"Error: {str(e)}"

    # Create output text summary
    triage_output = (
        f"Patient ({age}, {gender}) presenting with: {text}\n"
        f"ESI Level: {esi_label} (Confidence: {confidence:.2f})\n"
        f"Health Type: {health_type}\n"
        f"Explanation: {health_explanation}"
    )

    # Return structured response with proper values assigned to each field
    return {
        "esi_level": esi_label,
        "confidence": confidence,
        "health_type": health_type,
        "triage_output": triage_output,
        "selected_method": "Neural Network Classification",
    }


@app.post("/triage", response_model=TriageResponse)
def triage(request: TriageRequest):
    """Perform complete triage classification"""
    try:
        # Debug the input
        print(f"Processing triage request: {request.dict()}")

        result = full_triage_classification(request.text, request.age, request.gender)

        # Debug the output
        print(f"Triage result: {result}")

        return result
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Triage classification failed: {str(e)}"
        )


# ====== API Endpoints ======
@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Triage Classification API"}


@app.post("/triage", response_model=TriageResponse)
def triage(request: TriageRequest):
    """Perform complete triage classification"""
    try:
        result = full_triage_classification(request.text, request.age, request.gender)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Triage classification failed: {str(e)}"
        )


@app.post("/classify/esi", response_model=ESIResponse)
def classify_esi(request: ESIRequest):
    """Predict ESI level only"""
    try:
        esi_level, confidence = predict_esi(
            "esi_classifier.pt", request.text, request.age, request.gender
        )
        return {"esi_level": esi_level, "confidence": confidence}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"ESI classification failed: {str(e)}"
        )


@app.post("/classify/health-type", response_model=HealthClassResponse)
def classify_health(request: HealthClassRequest):
    """Classify health issue as mental or physical"""
    try:
        result = classify_health_type(request.text)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Health type classification failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    print("Starting Medical Triage API...")
    uvicorn.run("emergency_api:app", host="0.0.0.0", port=8000, reload=True)
