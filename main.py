from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
import tensorflow as tf
import pickle

app = FastAPI(title="Plant Health Monitoring API")

# Load the saved artifacts
model = tf.keras.models.load_model('models/plant_health_model.h5', compile=False)
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('models/label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)


# Define the data structure for one sensor reading
class SensorReading(BaseModel):
    temperature_C: float
    humidity_percent: float
    soil_moisture_percent: float
    light_intensity_percent: float
    nitrogen_percent: float
    phosphorus_percent: float
    potassium_percent: float


# Define the input for the API (list of last 5 readings)
class PredictionInput(BaseModel):
    readings: List[SensorReading]


@app.post("/predict")
async def predict_health(data: PredictionInput):
    if len(data.readings) != 5:
        raise HTTPException(status_code=400, detail="Exactly 5 historical readings are required for LSTM.")

    # 1. Convert input to numpy array
    input_list = [
        [r.temperature_C, r.humidity_percent, r.soil_moisture_percent,
         r.light_intensity_percent, r.nitrogen_percent, r.phosphorus_percent, r.potassium_percent]
        for r in data.readings
    ]

    # 2. Scale the data
    scaled_input = scaler.transform(input_list).reshape(1, 5, 7)

    # 3. Run prediction
    health_pred, sensor_pred = model.predict(scaled_input)

    # 4. Process outputs
    current_health_idx = np.argmax(health_pred)
    current_health_label = le.inverse_transform([current_health_idx])[0]

    # Inverse scale the predicted next sensor values
    future_sensors = scaler.inverse_transform(sensor_pred)[0]

    return {
        "predicted_current_health": current_health_label,
        "health_confidence": float(np.max(health_pred)),
        "predicted_next_values": {
            "temperature_C": float(future_sensors[0]),
            "humidity_%": float(future_sensors[1]),
            "soil_moisture_%": float(future_sensors[2]),
            "light_intensity_%": float(future_sensors[3]),
            "nitrogen_%": float(future_sensors[4]),
            "phosphorus_%": float(future_sensors[5]),
            "potassium_%": float(future_sensors[6])
        }
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)