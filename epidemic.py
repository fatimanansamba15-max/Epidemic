import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestClassifier
from geopy.geocoders import Nominatim
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ==========================================
# 1. PAGE CONFIGURATION SETUP
# ==========================================
st.set_page_config(page_title="Malaria Outbreak Intelligence Engine", layout="wide", page_icon="🦟")
st.title("🦟 Malaria Early-Warning Outbreak Intelligence Engine")
st.caption("Vector niche predictive analytics mapping current climate signals to Anopheles transmission risks.")

# Initialize session storage elements
if "malaria_results" not in st.session_state:
    st.session_state.malaria_results = None
if "audit_history" not in st.session_state:
    st.session_state.audit_history = []

# ==========================================
# 2. MALARIA VECTOR BIOLOGY TRAINING ENGINE
# ==========================================
@st.cache_resource
def train_malaria_model():
    np.random.seed(42)
    n_samples = 2500

    temp = np.random.uniform(40, 105, n_samples)
    rain = np.random.uniform(0, 15, n_samples)
    humidity = np.random.uniform(30, 100, n_samples)
    elevation = np.random.uniform(0, 3000, n_samples)

    temp_risk = np.where((temp >= 72) & (temp <= 95), 35, -15)
    rain_risk = rain * 6.0
    humidity_risk = np.where(humidity > 65, 25, -20)
    elevation_drain = elevation * 0.03

    total_ecological_score = temp_risk + rain_risk + humidity_risk - elevation_drain
    malaria_target = (total_ecological_score > 20).astype(int)

    # Maintain matching column syntax to satisfy feature-name alignment checks
    X = pd.DataFrame({'Temp': temp, 'Rain': rain, 'Humidity': humidity, 'Elevation': elevation})
    clf = RandomForestClassifier(n_estimators=120, random_state=42)
    clf.fit(X, malaria_target)
    return clf

model = train_malaria_model()

# ==========================================
# 3. GLOBAL GEOGRAPHY & LIVE API ENGINE
# ==========================================
def get_district_coordinates(location_string):
    geolocator = Nominatim(user_agent="malaria_outbreak_tracker_2026")
    try:
        location = geolocator.geocode(location_string, timeout=7)
        if location:
            return location.latitude, location.longitude, location.address
        return None, None, None
    except Exception:
        return None, None, None

def get_live_weather_and_elevation(lat, lon):
    elev_url = f"https://api.open-meteo.com/v1/elevation?latitude={lat}&longitude={lon}"
    weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,precipitation&temperature_unit=fahrenheit&precipitation_unit=inch"
    headers = {'User-Agent': 'MalariaOutbreakResearch/4.0'}

    try:
        elev_res = requests.get(elev_url, headers=headers, timeout=8).json()
        weather_res = requests.get(weather_url, headers=headers, timeout=8).json()
        
        # Pull correct parameter slice out of array payload response ecosystem
        elevation_list = elev_res.get('elevation', [150.0])
        elevation = elevation_list[0] if isinstance(elevation_list, list) else elevation_list
        current_data = weather_res.get('current', {})

        if 'temperature_2m' in current_data:
            return {
                'temp': float(current_data['temperature_2m']),
                'humidity': float(current_data['relative_humidity_2m']),
                'rain': float(current_data.get('precipitation', 0.0)),
                'elevation': float(elevation) if elevation is not None else 150.0
            }
        else:
            raise ValueError()
    except Exception:
        # Algorithmic backup generation fallback simulation parameters if request times out
        equator_proximity = max(0, 1 - (abs(lat) / 90.0))
        calculated_temp = 68.0 + (equator_proximity * 32.0) + (np.sin(lon) * 2.5)
        calculated_humidity = 50.0 + (equator_proximity * 38.0) + (np.cos(lat) * 4.0)
        calculated_rain = max(0.1, (np.sin(lat * lon) * 3.5) + 1.5)
        calculated_elevation = max(40.0, 550.0 - (abs(lat) * 6.0) + (abs(lon) % 10) * 12)

        return {
            'temp': round(calculated_temp, 1),
            'humidity': round(min(100.0, calculated_humidity), 1),
            'rain': round(calculated_rain, 2),
            'elevation': round(calculated_elevation, 1)
        }

def log_to_history(name, address, lat, lon, metrics, prob, prediction):
    record = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Queried Name": name,
        "Resolved Address": address,
        "Latitude": lat,
        "Longitude": lon,
        "Temp (°F)": metrics['temp'],
        "Humidity (%)": metrics['humidity'],
        "Rainfall (in)": metrics['rain'],
        "Elevation (m)": metrics['elevation'],
        "Vector Match Prob (%)": round(prob, 2),
        "Risk Assessment": "CRITICAL RISK" if prediction == 1 else "STABLE ECOSYSTEM"
    }
    if not any(h['Resolved Address'] == address and h['Timestamp'].split()[0] == record['Timestamp'].split()[0] for h in st.session_state.audit_history):
        st.session_state.audit_history.append(record)

def run_risk_calculation(metrics):
    # Form input features as DataFrame with structured named columns to remove sklearn warnings
    query_features = pd.DataFrame([{
        'Temp': metrics['temp'], 
        'Rain': metrics['rain'], 
        'Humidity': metrics['humidity'], 
        'Elevation': metrics['elevation']
    }])

    # Compute baseline calculation using random forest matrix model weights
    probability_score = float(model.predict_proba(query_features)[0][1] * 100)

    # 1. Biological High Risk Rule Guardrail
    if metrics['humidity'] > 70.0 and metrics['elevation'] < 1200.0 and metrics['temp'] > 75.0 and metrics['rain'] > 0.2:
        probability_score = max(probability_score, 75.0)
        prediction = 1
    # 2. Biological Low Risk Shield Inhibitor Guardrail
    elif metrics['elevation'] >= 1500.0 or metrics['temp'] < 60.0 or metrics['humidity'] < 50.0 or metrics['rain'] == 0:
        probability_score = min(probability_score, 35.0)
        prediction = 0
    # 3. Model Autonomy Hand-off Balanced Threshold Output Choice
    else:
        prediction = 1 if probability_score >= 50.0 else 0
        
    return probability_score, prediction

# ==========================================
# 4. INTERFACE LAYOUT & ANALYSIS EXECUTION
# ==========================================
st.sidebar.header("📍 Vector Sentinel Hub")
st.sidebar.write("Type your target country, state, or specific district below:")
