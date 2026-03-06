from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from catboost import CatBoostRegressor

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

model = CatBoostRegressor().load_model("nanotech_model_final.cbm")

class NanoInput(BaseModel):
    formula: str
    crystal_structure: str
    material_class: str
    size_nm: float
    shape: str

@app.post("/predict")
def predict(data: NanoInput):
    # CRITICAL: Order must match the training dataframe columns exactly
    feature_order = ['formula', 'size_nm', 'crystal_structure', 'material_class', 'shape']
    
    # Create DataFrame and force the correct order
    input_data = pd.DataFrame([data.dict()])
    input_df = input_data[feature_order]
    
    preds = model.predict(input_df)
    
    return {
        "bandgap": f"{preds[0][0]:.2f} eV",
        "density": f"{preds[0][1]:.2f} g/cm³",
        "formation_energy": f"{preds[0][2]:.2f} eV/atom",
        "specific_heat": f"{preds[0][3]:.4f} J/gK"
    }
