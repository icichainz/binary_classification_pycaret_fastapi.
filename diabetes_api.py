# -*- coding: utf-8 -*-

import pandas as pd
from pycaret.classification import load_model, predict_model
from fastapi import FastAPI
import uvicorn
from pydantic import create_model

# Create the app
app = FastAPI()

# Load trained Pipeline
model = load_model("diabetes_api")

# Create input/output pydantic models
input_model = create_model(
    "diabetes_api_input",
    **{
        'Number of times pregnant': (float, 13.0),
        'Plasma glucose concentration a 2 hours in an oral glucose tolerance test': (float, 152.0),
        'Diastolic blood pressure (mm Hg)': (float, 90.0),
        'Triceps skin fold thickness (mm)': (float, 33.0),
        '2-Hour serum insulin (mu U/ml)': (float, 29.0),
        'Body mass index (weight in kg/(height in m)^2)': (float, 26.799999237060547),
        'Diabetes pedigree function': (float, 0.7310000061988831),
        'Age (years)': (float, 43.0)
    }
)

output_model = create_model("diabetes_api_output", prediction=(int,1))


# Define predict function
@app.post("/predict", response_model=output_model)
def predict(data: input_model): # type: ignore
    data = pd.DataFrame([data.dict()])
    predictions = predict_model(model, data=data)
    return {"prediction": predictions["prediction_label"].iloc[0]}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
