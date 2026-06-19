import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestClassifier
from geopy.geocoders import Nominatim
import folium
from streamlit_folium import st_folium

# 1. Page Configuration Setup
st.set_page_config(page_title="Malaria Outbreak Intelligence Engine", layout="wide")
st.title("🦟 Malaria Early-Warning Outbreak Intelligence Engine")
st.caption("Vector niche predictive analytics mapping current climate signals to Anopheles transmission risks.")

# ==========================================
# 2. MALARIA VECTOR BIOLOGY TRAINING ENGINE
# ==========================================
@st.cache_data
def train_malaria_model():
    """Trains a classifier based on the physiological constraints of Anopheles mosquitoes."""
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
    elev_url = f"https://open-meteo.com{lat}&longitude={lon}"
    weather_url = f"https://open-meteo.com{lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,precipitation&temperature_unit=fahrenheit&precipitation_unit=inch"
    headers = {'User-Agent': 'MalariaOutbreakResearch/4.0'}

    try:
        elev_res = requests.get(elev_url, headers=headers, timeout=8).json()
        weather_res = requests.get(weather_url, headers=headers, timeout=8).json()
        elevation = elev_res.get('elevation', 150.0)
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
        is_la_p
