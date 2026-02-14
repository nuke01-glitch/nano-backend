from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from catboost import CatBoostRegressor

app = FastAPI()

# IMPORTANT: Allows your Vercel frontend to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the "Gold Standard" model
model = CatBoostRegressor().load_model("nanotech_model_final.cbm")

class NanoInput(BaseModel):
    formula: str
    crystal_structure: str
    material_class: str
    size_nm: float
    shape: str

@app.get("/")
def home():
    return {"status": "Nanotech API Online", "version": "1.0"}

@app.post("/predict")
def predict(data: NanoInput):
    # Prepare data for model
    input_df = pd.DataFrame([data.dict()])
    
    # Get predictions
    preds = model.predict(input_df)
    
    # Return formatted results
    return {
        "bandgap": f"{preds[0][0]:.2f} eV",
        "density": f"{preds[0][1]:.2f} g/cm³",
        "formation_energy": f"{preds[0][2]:.2f} eV",
        "specific_heat": f"{preds[0][3]:.4f} J/gK",
        "binding_energy": "193.39 eV (Baseline)" 
    }