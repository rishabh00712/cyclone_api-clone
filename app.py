from flask import Flask, request, jsonify
import requests
import pickle
import torch
import numpy as np
import os

from sklearn.preprocessing import StandardScaler
import joblib
import torch
import torch.nn as nn  # This is where the nn module comes from
from dotenv import load_dotenv

load_dotenv()
port = int(os.getenv('PORT', 5000))

# Initialize Flask app
app = Flask(__name__)
import torch

# Define the model architecture again (same as during training)
class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.gelu1 = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.gelu2 = nn.GELU()
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu1(x)
        x = self.fc2(x)
        x = self.gelu2(x)
        x = self.fc3(x)
        return x

# Define the model with the same input, hidden, and output dimensions
input_dim = 2  # 'Pressure' and 'Wind_Speed'
hidden_dim = 8  # Same number of neurons in hidden layers
output_dim = 1  # Predicting Wind_Speed

# Initialize the model
model = MLPModel(input_dim, hidden_dim, output_dim)

# Load the model weights
model.load_state_dict(torch.load('cyclone_detection_model.pth'))
model.eval()  # Set the model to evaluation mode (important for dropout/batch norm layers, if any)

print("Model loaded successfully!")


# Now, the model is ready to make predictions

# Load the scaler (if you have it stored separately)

scaler = joblib.load('scaler.pkl')  # Update with your scaler path

# Function to fetch timezone
def get_timezone(lat, lng):
    timezone_api_url = "http://api.timezonedb.com/v2.1/get-time-zone"
    timezone_params = {
        "key": "0C55SAC1WN7L",
        "format": "json",
        "by": "position",
        "lat": lat,
        "lng": lng
    }
    response = requests.get(timezone_api_url, params=timezone_params)
    response.raise_for_status()
    return response.json().get("zoneName", "UTC")




@app.route('/weather', methods=['GET'])
def weather_data():
    try:
        latitude = request.args.get('latitude', type=float)
        longitude = request.args.get('longitude', type=float)

        if latitude is None or longitude is None:
            return jsonify({"error": "Latitude and longitude are required"}), 400

        # Get timezone
        timezone = get_timezone(latitude, longitude)

# Fetch weather data
        weather_api_url = "https://api.open-meteo.com/v1/forecast"

        weather_params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": "surface_pressure,wind_speed_10m",
            "timezone": timezone,
        }

        response = requests.get(weather_api_url, params=weather_params)

        response.raise_for_status()

        weather_data = response.json()

        def calculate_average(arr):
            return sum(arr) / len(arr) if arr else 0


        # Extract hourly data for the past 7 days
        sur_pressure = weather_data.get("hourly", {}).get("surface_pressure", [])
        wind_speed_10m = weather_data.get("hourly", {}).get("wind_speed_10m", [])

        # To calculate the average of each parameter over the past 7 days (assuming hourly data)
        # Ensure that you have 7 days of data (168 hours)

        def average_over_days(data, days=7):
            # Each day has 24 hours
            day_hours = 24
            num_hours = days * day_hours

            # Check if we have enough data (at least 7 days)
            if len(data) < num_hours:
                raise ValueError("Insufficient data for 7 days.")

            # Split the data into 7 days, calculate daily averages, and then calculate the overall average
            daily_averages = [calculate_average(data[i:i+day_hours]) for i in range(0, num_hours, day_hours)]
            return calculate_average(daily_averages)

        # Calculate averages for each required parameter
        avg_surface_pressure = average_over_days(sur_pressure)
        avg_wind_speed_10m = average_over_days(wind_speed_10m)


# Prepare input for the model (convert averages into tensor)
       
        scaled_input_data = scaler.transform([[avg_surface_pressure, avg_wind_speed_10m]])
        input_data = torch.tensor(scaled_input_data, dtype=torch.float32)

       # Make the prediction
        with torch.no_grad():  # Disable gradient tracking for inference
            prediction = model(input_data)
            
        # Check and modify the prediction if needed
        if prediction.item() > 100:
         prediction = torch.tensor([95.0]) 

        # Print the predicted output (e.g., wind speed or some other metric)
        print(f"Predicted value: {prediction.item()}")

        # Return the response
        return jsonify({
        "latitude": latitude,
        "longitude": longitude,
        "avg_surface_pressure":avg_surface_pressure,
        "avg_wind_speed_10m":avg_wind_speed_10m,
        "cyclone_probability %": round(prediction.item(), 2)
})


    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"API request failed: {e}"}), 500
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {e}"}), 500


if __name__ == '__main__':
    app.run(debug=True, port = port)
