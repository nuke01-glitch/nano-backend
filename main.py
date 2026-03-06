import os
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

# Load model using absolute path for robustness on Render
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "nanotech_model_final.cbm")
model = CatBoostRegressor()
model.load_model(MODEL_PATH)

class NanoInput(BaseModel):
    formula: str
    crystal_structure: str
    material_class: str
    size_nm: float
    shape: str

@app.post("/predict")
def predict(data: NanoInput):
    # 1. Define feature order used in training
    feature_order = ['formula', 'size_nm', 'crystal_structure', 'material_class', 'shape']
    
    # 2. Convert input to DataFrame
    input_df = pd.DataFrame([data.dict()])[feature_order]
    
    # 3. CRITICAL FIX: Ensure categorical features are strings, 
    # and numerical features remain numeric.
    # Note: Adjust 'cat_cols' list to match exactly what you used for cat_features in training
    cat_cols = ['formula', 'crystal_structure', 'material_class', 'shape']
    
    for col in cat_cols:
        input_df[col] = input_df[col].astype(str)
        
    # Ensure numerical features are numeric
    input_df['size_nm'] = pd.to_numeric(input_df['size_nm'])
    
    # 4. Get prediction
    preds = model.predict(input_df)
    
    # 5. Return formatted results
    return {
        "bandgap": f"{preds[0][0]:.2f} eV",
        "density": f"{preds[0][1]:.2f} g/cm³",
        "formation_energy": f"{preds[0][2]:.2f} eV/atom",
        "specific_heat": f"{preds[0][3]:.4f} J/gK"
    }
