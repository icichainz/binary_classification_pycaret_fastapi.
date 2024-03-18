# The fast api api that serve the model.
from fastapi import FastAPI
from pydantic import BaseModel
from pycaret.regression import load_model

# Load the saved PyCaret model
model = load_model('path_to_saved_model')

# Create a FastAPI application
app = FastAPI()

# Define a request body model
class InputData(BaseModel):
    feature1: float
    feature2: float

# Define an API endpoint to make predictions
@app.post("/predict")
def predict(data: InputData):
    inputs = [[data.feature1, data.feature2]]  # Preprocess input data if required
    predictions = model.predict(inputs)  # Make predictions
    return {"predictions": predictions}

# Run the FastAPI application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)