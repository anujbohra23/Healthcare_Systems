from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict

from triage_model import final_esi_decision_from_json

app = FastAPI()


class PatientData(BaseModel):
    symptoms: str
    age: int
    gender: str
    lab_results: Dict = {}
    diagnosis: str = ""
    treatment_plan: str = ""


@app.get("/")
async def root():
    return {"message": "Welcome to the Triage API!"}


@app.post("/triage/")
async def predict_triage(patient_data: PatientData):
    try:
        result = final_esi_decision_from_json(patient_data.model_dump())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
