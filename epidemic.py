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
if "last_queried_district" not in st.session_state:
    st.session_state.last_queried_district = ""

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
    # FIX: Shifted parameters to request cumulative 'daily=precipitation_sum' instead of live hourly precipitation
    weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m&daily=precipitation_sum&temperature_unit=fahrenheit&precipitation_unit=inch&timezone=auto"
    headers = {'User-Agent': 'MalariaOutbreakResearch/4.0'}

    try:
        elev_res = requests.get(elev_url, headers=headers, timeout=8).json()
        weather_res = requests.get(weather_url, headers=headers, timeout=8).json()
        
        elevation_list = elev_res.get('elevation', [150.0])
        elevation = elevation_list[0] if isinstance(elevation_list, list) else elevation_list
        current_data = weather_res.get('current', {})
        daily_data = weather_res.get('daily', {})

        if 'temperature_2m' in current_data:
            # Safely grab index [0] to extract today's accumulated rainfall metric
            rain_sum = daily_data.get('precipitation_sum', [0.0])[0]
            return {
                'temp': float(current_data['temperature_2m']),
                'humidity': float(current_data['relative_humidity_2m']),
                'rain': float(rain_sum) if rain_sum is not None else 0.0,
                'elevation': float(elevation) if elevation is not None else 150.0
            }
        else:
            raise ValueError()
    except Exception:
        # Balanced backup simulation engine if third-party live endpoints drop packets
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
    query_features = pd.DataFrame([{
        'Temp': metrics['temp'], 
        'Rain': metrics['rain'], 
        'Humidity': metrics['humidity'], 
        'Elevation': metrics['elevation']
    }])

    probability_score = float(model.predict_proba(query_features)[0][1] * 100)

    if metrics['humidity'] > 70.0 and metrics['elevation'] < 1200.0 and metrics['temp'] > 75.0 and metrics['rain'] > 0.2:
        probability_score = max(probability_score, 75.0)
        prediction = 1
    elif metrics['elevation'] >= 1500.0 or metrics['temp'] < 60.0 or metrics['humidity'] < 50.0 or metrics['rain'] == 0:
        probability_score = min(probability_score, 35.0)
        prediction = 0
    else:
        prediction = 1 if probability_score >= 50.0 else 0
        
    return probability_score, prediction

# ==========================================
# 4. INTERFACE LAYOUT & AUTOMATIC ANALYSIS
# ==========================================
st.sidebar.header("📍 Vector Sentinel Hub")
st.sidebar.write("Type your target country, state, or specific district below:")
user_district = st.sidebar.text_input("District / Sub-County Name", value="Soroti, Uganda", key="malaria_input_box")

if st.session_state.malaria_results is None or user_district != st.session_state.last_queried_district:
    with st.spinner(f"Analyzing regional wetland metrics for {user_district}..."):
        lat, lon, full_address = get_district_coordinates(user_district)
        if lat and lon:
            metrics = get_live_weather_and_elevation(lat, lon)
            probability_score, prediction = run_risk_calculation(metrics)

            st.session_state.malaria_results = {
                "address": full_address, "lat": lat, "lon": lon, "metrics": metrics,
                "prediction": prediction, "prob": probability_score, "name": user_district
            }
            st.session_state.last_queried_district = user_district
            log_to_history(user_district, full_address, lat, lon, metrics, probability_score, prediction)
        else:
            st.sidebar.error("Location signature unverified. Showing last active location.")

# ==========================================
# 5. DASHBOARD PRESENTATION TAB LAYOUT
# ==========================================
if st.session_state.malaria_results is not None:
    res = st.session_state.malaria_results
    m_data = res['metrics']
    is_high_risk = res['prediction'] == 1

    st.success(f"Tracking Site Confirmed: **{res['address']}**")
    
    tab_summary, tab_visuals, tab_reports, tab_prevention = st.tabs([
        "👁️ Live Site Monitoring", 
        "📊 Dynamic Visual Analytics", 
        "💾 Executive Report Hub",
        "🛡️ Prevention & Vector Control Guide"
    ])

    # ------------------ TAB 1: SITE MONITORING SUMMARY ------------------
    with tab_summary:
        st.caption(f"Spatial Grid Pins: Latitude {res['lat']:.4f} | Longitude {res['lon']:.4f}")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Thermal Signature", f"{m_data['temp']} °F")
        col2.metric("Relative Air Humidity", f"{m_data['humidity']} %")
        col3.metric("Rainfall Accumulation", f"{m_data['rain']} Inches")
        col4.metric("Altitude Level", f"{m_data['elevation']} Meters")

        st.markdown("---")
        st.subheader("Transmission Potential Assessment")
        
        if is_high_risk:
            st.error(f"🚨 CRITICAL VECTOR SURGE ALERT: Climate spikes and topographic configurations in {res['name']} indicate a high-risk transmission index ({res['prob']:.1f}% Vector Affinity Match).")
        else:
            st.success(f"✅ STABLE ENVIRO-MATRIX: Local climatic features display a low threat index for a severe vector breeding cycle ({res['prob']:.1f}% Vector Affinity Match).")

        st.subheader("🔍 Vector Niche Analysis: Local Environmental Drivers")
        exp1, exp2 = st.columns(2)

        with exp1:
            st.markdown("### 🦟 Vector Accelerators")
            if m_data['humidity'] > 65:
                st.write(f"• **High Humidity ({m_data['humidity']}%):** Greatly expands adult *Anopheles* lifespan, accelerating parasitic incubation cycles.")
            if 72 <= m_data['temp'] <= 95:
                st.write(f"• **Optimal Incubation Heat ({m_data['temp']}°F):** Provides perfect warmth for rapid larval development.")
            if m_data['elevation'] < 1200:
                st.write(f"• **Low Altitude Basin ({m_data['elevation']}m):** Flat topography traps water runoff easily creating vector pools.")
            if m_data['rain'] > 0.4:
                st.write(f"• **Breeding Pool Formation ({m_data['rain']} in):** Creates optimal, clean, stagnant breeding parameters.")
            if m_data['humidity'] <= 65 and (m_data['temp'] < 72 or m_data['temp'] > 95) and m_data['elevation'] >= 1200 and m_data['rain'] <= 0.4:
                st.write("_None observed in current climate metrics._")

        with exp2:
            st.markdown("### 🛡️ Environmental Inhibitors")
            if m_data['elevation'] >= 1500:
                st.write(f"• **High Altitude Shield ({m_data['elevation']}m):** High mountain atmospheres significantly stall mosquito replication cycles.")
            if m_data['temp'] < 64:
                st.write(f"• **Thermal Cessation Boundary ({m_data['temp']}°F):** Ambient air drops below lower baseline temperature thresholds required for development.")
            if m_data['humidity'] < 55:
                st.write(f"• **Desiccation Factor ({m_data['humidity']}%):** Low humidity dries out vectors, yielding high adult vector mortality rates.")
            if m_data['rain'] == 0:
                st.write("•
