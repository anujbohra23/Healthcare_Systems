import torch
import torch.nn as nn
from transformers import (
    BertModel,
    BertTokenizer,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from sklearn.preprocessing import LabelEncoder
import re
import pandas as pd
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
import torch.nn.functional as F

hf_token = ""
groq_token = ""


# 1) Emotion-Fusion Setup
emotion_df = pd.read_csv(
    r"C:\Users\Anuj Bohra\Desktop\Healthcare_Systems\Severity Analysis\Data Files\emotion_lexicon.csv"
)
emotion_df["word"] = emotion_df["word"].str.lower().str.strip()
emotion_df.set_index("word", inplace=True)
emotion_dict = emotion_df.to_dict(orient="index")


def compute_emotion_scores(text, emotion_dict):
    text = text.lower()
    tokens = re.findall(r"\b\w+\b", text)
    categories = list(next(iter(emotion_dict.values())).keys())
    scores = dict.fromkeys(categories, 0)
    for t in tokens:
        if t in emotion_dict:
            for emo, val in emotion_dict[t].items():
                scores[emo] += val
    return scores


# === 2) Load MentalBERT ===
MBERT_CHECKPOINT = "mental/mental-bert-base-uncased"
tokenizer_mbert = AutoTokenizer.from_pretrained(MBERT_CHECKPOINT, token=hf_token)
model_mbert = AutoModelForSequenceClassification.from_pretrained(
    MBERT_CHECKPOINT, num_labels=2, token=hf_token
)
model_mbert.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_mbert.to(device)
label_map = {0: "Physical Health", 1: "Mental Health"}


def classify_with_emotion_fusion(text: str):
    inputs = tokenizer_mbert(
        text, return_tensors="pt", truncation=True, padding=True, max_length=512
    ).to(device)
    with torch.no_grad():
        outputs = model_mbert(**inputs)
        probs = F.softmax(outputs.logits, dim=1).squeeze()
        phys_conf, ment_conf = float(probs[0]), float(probs[1])
        cls_idx = int(torch.argmax(probs).item())

    emo_scores = compute_emotion_scores(text, emotion_dict)
    negative_emotions = {"negative", "fear", "sadness", "anger", "disgust"}
    dominant = [
        e for e, v in emo_scores.items() if v == max(emo_scores.values()) and v > 0
    ]
    neg_dominant = any(e in negative_emotions for e in dominant)

    if ment_conf > 0.7 and neg_dominant:
        final = "Mental Health"
    elif phys_conf > 0.7 and neg_dominant:
        final = "Physical Health"
    elif ment_conf < 0.6 and phys_conf < 0.6:
        final = "Uncertain"
    else:
        final = label_map[cls_idx]

    return {
        "physical_conf": phys_conf,
        "mental_conf": ment_conf,
        "emotion_scores": emo_scores,
        "dominant_emotions": dominant,
        "final_decision": final,
    }


from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate


# === 3) LLM Triage Setup ===
llm = ChatGroq(
    api_key=groq_token, model_name="llama3-8b-8192", temperature=0.2, max_tokens=200
)

try:
    ChatGroq.model_rebuild()
except AttributeError:
    print(
        "Warning: ChatGroq.model_rebuild() call failed. This might be okay in newer versions."
    )

triage_prompt = PromptTemplate.from_template(
    """
    You are a highly reliable medical triage assistant.
    Based on the patient's input, classification (emotional or physical health), lab results, diagnosis, and treatment plan, assign an Emergency Severity Index (ESI) level from 1 to 5.

    The levels are:
    - ESI-1: Immediate life-saving intervention required
    - ESI-2: High-risk situation, confused/lethargic/disoriented, or severe pain/distress
    - ESI-3: Multiple resources needed and vital signs are stable
    - ESI-4: One resource needed
    - ESI-5: No resources needed

    Provide:
    1. ESI Level: [1–5]
    2. Reason: [Short explanation]
    3. Confidence: [float between 0 and 1]

    Classification: {classification}
    Symptoms: {symptoms}
    Lab Results:
    {lab_results}
    Diagnosis: {diagnosis}
    Treatment Plan: {treatment_plan}
"""
)


def llm_triage(
    classification: str,
    symptoms: str,
    lab_results: str,
    diagnosis: str,
    treatment_plan: str,
) -> str:
    chain = triage_prompt | llm
    out = chain.invoke(
        {
            "classification": classification,
            "symptoms": symptoms,
            "lab_results": lab_results,
            "diagnosis": diagnosis,
            "treatment_plan": treatment_plan,
        }
    )
    return out.content.strip()


# === Neural Net Model Definition and Prediction ===
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


def predict_esi(
    model_path,
    test_text="Patient is unconscious and not breathing.",
    test_age=82,
    test_gender="Male",
    max_len=128,
):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ESIClassifier().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    encoding = tokenizer(
        test_text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_len,
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    age_tensor = torch.tensor([test_age], dtype=torch.float).to(device)

    gender_encoder = LabelEncoder()
    gender_encoder.fit(["Male", "Female", "Other"])
    gender_tensor = torch.tensor(
        [gender_encoder.transform([test_gender])[0]], dtype=torch.long
    ).to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask, age_tensor, gender_tensor)
        probs = F.softmax(logits, dim=1).squeeze()
        predicted_class = torch.argmax(probs).item()
        confidence = probs[predicted_class].item()

    esi_label = predicted_class + 1
    return esi_label, confidence


# === Final ESI Decision Function with Dynamic Input ===
def final_esi_decision_from_json(
    json_data,
    nn_model_path=r"C:\Users\Anuj Bohra\Desktop\Healthcare_Systems\Severity Analysis\esi_classifier.pt",
):
    """
    Takes a JSON dictionary as input and returns the final ESI decision.

    Args:
        json_data (dict): A dictionary containing the input fields.
                           Expected keys: "symptoms", "age", "gender",
                           "lab_results" (as a dictionary), "diagnosis",
                           "treatment_plan".
        nn_model_path (str): Path to the saved neural network model.

    Returns:
        dict: A dictionary containing the final ESI decision and related information.
    """
    symptoms = json_data.get("symptoms")
    age = json_data.get("age")
    gender = json_data.get("gender")
    lab_results = json_data.get("lab_results", {})
    diagnosis = json_data.get("diagnosis", "")
    treatment_plan = json_data.get("treatment_plan", "")

    if not all([symptoms, age, gender]):
        return {"error": "Missing required fields: symptoms, age, or gender"}

    # --- Format Lab Results for LLM ---
    lab_str = "\n".join(f"{k}: {v}" for k, v in lab_results.items())

    # --- Emotion Fusion Classification ---
    fused = classify_with_emotion_fusion(f"{symptoms}\nLab Results:\n{lab_str}")
    classification = fused["final_decision"]

    # --- LLM Prediction ---
    llm_output = llm_triage(
        classification=classification,
        symptoms=symptoms,
        lab_results=lab_str,
        diagnosis=diagnosis,
        treatment_plan=treatment_plan,
    )

    # Parse LLM confidence and ESI level
    esi_llm = None
    confidence_llm = None
    esi_match = re.search(r"ESI Level:\s*(\d)", llm_output)
    confidence_match = re.search(r"Confidence:\s*([0-9]*\.?[0-9]+)", llm_output)

    if esi_match and confidence_match:
        try:
            esi_llm = int(esi_match.group(1))
            confidence_llm = float(confidence_match.group(1))
        except ValueError:
            print(f"Error parsing LLM output: {llm_output}")
            esi_llm = None
            confidence_llm = None
    else:
        print(f"Could not parse ESI Level or Confidence from LLM output: {llm_output}")

    # --- Neural Net Prediction ---
    esi_nn, confidence_nn = predict_esi(
        model_path=nn_model_path, test_text=symptoms, test_age=age, test_gender=gender
    )

    # --- Decision Based on Confidence ---
    final_source = None
    final_esi = None
    final_conf = None

    if confidence_llm is not None and confidence_nn is not None:
        if confidence_nn >= confidence_llm:
            final_source = "Neural Network"
            final_esi = esi_nn
            final_conf = confidence_nn
        else:
            final_source = "LLM"
            final_esi = esi_llm
            final_conf = confidence_llm
    elif confidence_nn is not None:
        final_source = "Neural Network"
        final_esi = esi_nn
        final_conf = confidence_nn
    elif confidence_llm is not None:
        final_source = "LLM"
        final_esi = esi_llm
        final_conf = confidence_llm
    else:
        return {"error": "Could not get confidence scores from either model"}

    # --- Output ---
    result = {
        "Final ESI Level": final_esi,
        "Confidence": round(final_conf, 4) if final_conf is not None else None,
        "Chosen Model": final_source,
        "LLM Confidence": (
            round(confidence_llm, 4) if confidence_llm is not None else None
        ),
        "Neural Net Confidence": (
            round(confidence_nn, 4) if confidence_nn is not None else None
        ),
        "LLM Output": llm_output,
        "Emotion Classification": classification,
        "Dominant Emotions": fused["dominant_emotions"],
    }
    return result


if __name__ == "__main__":
    # Example usage (you would replace this with actual API calls in a real application)
    sample_json_data = {
        "symptoms": "Patient is experiencing severe shortness of breath and chest tightness.",
        "age": 70,
        "gender": "Female",
        "lab_results": {"Oxygen Saturation": 88},
        "diagnosis": "Possible pulmonary embolism",
        "treatment_plan": "Administer oxygen, start anticoagulation.",
    }

    output = final_esi_decision_from_json(
        sample_json_data, nn_model_path="esi_classifier.pt"
    )
    print("\nFinal ESI Decision:\n", output)

    sample_json_mental = {
        "symptoms": "Patient reports feeling anxious and having panic attacks.",
        "age": 28,
        "gender": "Male",
        "lab_results": {},
        "diagnosis": "Panic disorder",
        "treatment_plan": "Initiate therapy, consider SSRIs.",
    }
    output_mental = final_esi_decision_from_json(
        sample_json_mental, nn_model_path="esi_classifier.pt"
    )
    print("\nFinal ESI Decision (Mental Health):\n", output_mental)
