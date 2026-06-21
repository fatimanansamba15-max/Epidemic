import streamlit as st
import pandas as pd
import numpy as np
import requests
from geopy.geocoders import Nominatim
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# ==========================================
# 1. PAGE CONFIGURATION SETUP
# ==========================================
st.set_page_config(page_title="Malaria Outbreak Intelligence Engine", layout="wide", page_icon="🦟")
st.title("🦟 Malaria Early-Warning Outbreak Intelligence Engine")
st.caption("Predictive sub-seasonal modeling mapping forward climate forcing vectors to lagged epidemiological surge windows.")

# Initialize session storage elements
if "malaria_results" not in st.session_state:
    st.session_state.malaria_results = None
if "audit_history" not in st.session_state:
    st.session_state.audit_history = []
if "last_queried_district" not in st.session_state:
    st.session_state.last_queried_district = ""

# ==========================================
# 2. FUTURE & HISTORICAL CLIMATE DATA PIPELINE
# ==========================================
def get_district_coordinates(location_string):
    geolocator = Nominatim(user_agent="malaria_forecaster_engine_2026")
    try:
        location = geolocator.geocode(location_string, timeout=7)
        if location:
            return location.latitude, location.longitude, location.address
        return None, None, None
    except Exception:
        return None, None, None

def get_live_and_forecast_weather(lat, lon):
    """
    Pulls a combined weather matrix:
    - Past 14 days of actual rainfall (to compute active larval pool foundations)
    - Future 14 days of projected temperature/humidity/precipitation (to model future parasite trends)
    """
    today = datetime.now().date()
    past_start = today - timedelta(days=14)
    future_end = today + timedelta(days=14)
    
    # Combined URL request to capture the historical footprint and the forward forecast envelope
    weather_url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
        f"&current=temperature_2m,relative_humidity_2m"
        f"&daily=temperature_2m_mean,relative_humidity_2m_mean,precipitation_sum"
        f"&start_date={past_start}&end_date={future_end}"
        f"&temperature_unit=fahrenheit&precipitation_unit=inch&timezone=auto"
    )
    
    elev_url = f"https://api.open-meteo.com/v1/elevation?latitude={lat}&longitude={lon}"
    headers = {'User-Agent': 'MalariaOutbreakForecaster/6.0'}

    try:
        elev_res = requests.get(elev_url, headers=headers, timeout=8).json()
        weather_res = requests.get(weather_url, headers=headers, timeout=8).json()
        
        elevation_list = elev_res.get('elevation', [150.0])
        elevation = elevation_list[0] if isinstance(elevation_list, list) else elevation_list
        
        daily_data = weather_res.get('daily', {})
        time_series = daily_data.get('time', [])
        
        # Parse timeseries array into a structured Pandas DataFrame
        df_climate = pd.DataFrame({
            'date': pd.to_datetime(time_series),
            'temp_mean': daily_data.get('temperature_2m_mean', [75.0]*len(time_series)),
            'humidity_mean': daily_data.get('relative_humidity_2m_mean', [60.0]*len(time_series)),
            'precipitation': daily_data.get('precipitation_sum', [0.0]*len(time_series))
        })
        
        # Clean up any potential missing or null observations from the API fields
        df_climate = df_climate.fillna(method='ffill').fillna(method='bfill')
        
        return df_climate, float(elevation)
        
    except Exception:
        # Structured historical fallback simulation if remote weather APIs time out
        total_days = (future_end - past_start).days + 1
        date_range = [past_start + timedelta(days=i) for i in range(total_days)]
        
        equator_proximity = max(0, 1 - (abs(lat) / 90.0))
        base_temp = 68.0 + (equator_proximity * 24.0)
        base_hum = 55.0 + (equator_proximity * 30.0)
        
        df_climate = pd.DataFrame({
            'date': pd.to_datetime(date_range),
            'temp_mean': [base_temp + np.sin(i)*2 for i in range(total_days)],
            'humidity_mean': [min(100.0, base_hum + np.cos(i)*5) for i in range(total_days)],
            'precipitation': [max(0.0, np.random.normal(0.15, 0.2)) if i % 4 == 0 else 0.0 for i in range(total_days)]
        })
        calculated_elevation = max(100.0, 1200.0 - (abs(lat) * 15))
        return df_climate, float(calculated_elevation)

# ==========================================
# 3. ADVANCED TIME-LAGGED EPIDEMIOLOGICAL MODEL
# ==========================================
def run_predictive_horizon_engine(df_climate, elevation):
    """
    Computes moving 14-day future risk projections using biological lag logic:
    - Larval Breeding Niche: Linked to cumulative rainfall footprints over prior 14 days
    - Parasite Incubation (EIP): Non-linear speed scaling directly with forecasted daily heat profiles
    """
    timeline_records = []
    today_dt = pd.to_datetime(datetime.now().date())
    
    # Filter the dataset to process individual days inside our active look-forward window
    df_future = df_climate[df_climate['date'] >= today_dt].copy()
    
    for _, row in df_future.iterrows():
        target_date = row['date']
        
        # 1. Biological Lag-Phase Anchor: Gather preceding 14-day cumulative rainfall pool index
        past_window = df_climate[(df_climate['date'] <= target_date) & (df_climate['date'] >= target_date - timedelta(days=14))]
        cumulative_rain = past_window['precipitation'].sum()
        
        # Current environmental parameters forecasted for this specific horizon mark
        T = row['temp_mean']
        H = row['humidity_mean']
        
        # 2. Extrinsic Incubation Period (EIP Degree-Day Model) for Parasite Rate Maturation
        if 64.4 <= T <= 104.0:
            # Mathematical degree-day formula (D / (T_mean - T_min)) mapped into an exponential efficiency score
            parasite_incubation_speed = 111.0 / (T - 64.4)
            t_score = 1.0 - min(1.0, (parasite_incubation_speed / 30.0))  # Faster incubation = higher biological threat index
        else:
            t_score = 0.0
            
        # 3. Adult Vector Longevity Score (Sigmoidal Humidity Envelope)
        h_score = 1.0 / (1.0 + np.exp(-0.15 * (H - 60.0)))
        
        # 4. Larval Hydrological Habitat Suitability Score
        if cumulative_rain == 0:
            r_score = 0.0
        elif cumulative_rain > 8.5:
            r_score = 0.20  # Suppressed scaling due to flash flood runoff washing out aquatic habitats
        else:
            r_score = 1.0 - (((cumulative_rain - 3.5) / 5.0) ** 2)
            r_score = max(0.0, min(1.0, r_score))
            
        # 5. Topographic Elevation Attenuation Layer
        if elevation >= 1600.0:
            e_factor = 0.05
        elif elevation <= 600.0:
            e_factor = 1.0
        else:
            e_factor = 1.0 - ((elevation - 600.0) / 1000.0)
            
        # Compile total weighted forward risk affinity metric
        raw_affinity = (t_score * 0.35) + (h_score * 0.25) + (r_score * 0.25) + (e_factor * 0.15)
        risk_percentage = float(raw_affinity * 100.0)
        
        # Enforce severe climate-forcing environmental inhibitor caps
        if elevation >= 1800.0 or T < 61.0 or H < 45.0:
            risk_percentage = min(risk_percentage, 10.0)
        elif cumulative_rain == 0.0 and H < 52.0:
            risk_percentage = min(risk_percentage, 12.0)
            
        timeline_records.append({
            'Date': target_date.strftime('%Y-%m-%d'),
            'Temperature': round(T, 1),
            'Humidity': round(H, 1),
            'Accumulated Rain': round(cumulative_rain, 2),
            'Outbreak Risk %': round(risk_percentage, 1),
            'Verdict': "CRITICAL SURGE" if risk_percentage >= 48.0 else "CONTROLLED VECTOR MATRIX"
        })
        
    df_horizon = pd.DataFrame(timeline_records)
    return df_horizon

# ==========================================
# 4. INTERFACE RUNTIME CONTROLLER
# ==========================================
st.sidebar.header("📍 Vector Sentinel Hub")
st.sidebar.write("Type your target country, state, or specific district below:")
user_district = st.sidebar.text_input("District / Sub-County Name", value="Soroti, Uganda", key="malaria_input_box")

# PROTECTIVE RESET: Instantly wipe the view if the user is transitioning from an obsolete schema version
if st.session_state.malaria_results is not None:
    if "horizon_dataframe" not in st.session_state.malaria_results:
        st.session_state.malaria_results = None
        st.session_state.last_queried_district = ""

if st.session_state.malaria_results is None or user_district != st.session_state.last_queried_district:
    with st.spinner(f"Running sub-seasonal time-lagged epidemiological analytics for {user_district}..."):
        lat, lon, full_address = get_district_coordinates(user_district)
        if lat and lon:
            df_climate, elevation = get_live_and_forecast_weather(lat, lon)
            df_horizon = run_predictive_horizon_engine(df_climate, elevation)
            
            # Extract current index metrics for the metrics display panels
            current_idx = df_horizon.iloc[0]
            
            st.session_state.malaria_results = {
                "address": full_address, "lat": lat, "lon": lon, "elevation": elevation,
                "current_metrics": current_idx, "horizon_dataframe": df_horizon, "name": user_district
            }
            st.session_state.last_queried_district = user_district
            
            # Save tracking snapshot to audit log
            record = {
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Target District": user_district,
                "Current Risk %": current_idx['Outbreak Risk %'],
                "14-Day Max Risk %": df_horizon['Outbreak Risk %'].max(),
                "Elevation (m)": elevation
            }
            st.session_state.audit_history.append(record)
        else:
            st.sidebar.error("Location signature unverified. Adjust spelling and retry.")

# ==========================================
# 5. DASHBOARD PRESENTATION TAB LAYOUT
# ==========================================
if st.session_state.malaria_results is not None:
    res = st.session_state.malaria_results
    curr = res['current_metrics']
    df_hz = res['horizon_dataframe']
    
    st.success(f"Tracking Site Confirmed: **{res['address']}**")
    
    tab_summary, tab_visuals, tab_reports, tab_prevention = st.tabs([
        "👁️ Live Site Monitoring", 
        "🔮 14-Day Outbreak Horizon Forecast", 
        "💾 Executive Report Hub",
        "🛡️ Prevention & Vector Control Guide"
    ])

    # ------------------ TAB 1: SITE MONITORING SUMMARY ------------------
    with tab_summary:
        st.caption(f"Spatial Grid Pins: Latitude {res['lat']:.4f} | Longitude {res['lon']:.4f}")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Thermal Signature (Today)", f"{curr['Temperature']} °F")
        col2.metric("Relative Air Humidity", f"{curr['Humidity']} %")
        col3.metric("14-Day Accumulated Rain", f"{curr['Accumulated Rain']:.2f} In")
        col4.metric("Altitude Level", f"{res['elevation']} Meters")

        st.markdown("---")
        st.subheader("Current Transmission Potential Assessment")
        
        if curr['Outbreak Risk %'] >= 48.0:
            st.error(f"🚨 ACTIVE VECTOR SURGE ALERT: Real-time climate parameters indicate an elevated outbreak threat index right now ({curr['Outbreak Risk %']}% Vector Affinity Match).")
        else:
            st.success(f"✅ STABLE ENVIRO-MATRIX: Current climatic criteria indicate a well-contained risk matrix profile ({curr['Outbreak Risk %']}% Vector Affinity Match).")

        # Peak future risk alert notification layer
        max_future_risk = df_hz['Outbreak Risk %'].max()
        max_risk_row = df_hz[df_hz['Outbreak Risk %'] == max_future_risk].iloc[0]
        
        st.subheader("Forward Intelligence Forecast Notification")
        if max_future_risk >= 48.0:
            st.warning(f"⚠️ PREDICTIVE TREND ALERT: Sub-seasonal climate accumulation curves indicate transmission potential is trending upward and will peak at a critical **{max_future_risk}%** on **{max_risk_row['Date']}** due to rolling hydrological delay mechanics.")
        else:
            st.info(f"✨ PREDICTIVE TREND LOOK-AHEAD: Environmental models project that vector transmission parameters will remain safe and stable throughout the upcoming 14-day observation corridor.")

    # ------------------ TAB 2: 14-DAY OUTBREAK HORIZON FORECAST ------------------
    with tab_visuals:
        st.subheader("🔮 Predictive Outbreak Horizon Graph")
        st.write("This time-series chart maps the dynamic, rolling development wave of the *Plasmodium* incubation clock over the coming two weeks.")
        
        # Plotly Time-Series Horizon Graph mapping the future transmission surge
        fig_horizon = go.Figure()
        
        fig_horizon.add_trace(go.Scatter(
            x=df_hz['Date'], y=df_hz['Outbreak Risk %'],
            mode='lines+markers', name='Projected Outbreak Index %',
            line=dict(color='crimson' if max_future_risk >= 48.0 else 'darkgreen', width=3),
            marker=dict(size=7, symbol='diamond')
        ))
        
        # Add static reference trigger baseline indicators
        fig_horizon.add_hline(y=48.0, line_dash="dash", line_color="orange", annotation_text="Outbreak Trigger Threshold (48%)")
        
        fig_horizon.update_layout(
            xaxis_title="Upcoming Calendar Horizon",
            yaxis_title="Outbreak Index Probability (%)",
            yaxis=dict(range=[0, 105]),
            height=340,
            margin=dict(l=20, r=20, t=20, b=20),
            hovermode="x unified"
        )
        st.plotly_chart(fig_horizon, use_container_width=True)
        
        # Data table matrix view
        st.markdown("**Daily Predictive Analytical Ledger Matrix**")
        st.dataframe(df_hz, use_container_width=True, height=220)

    # ------------------ TAB 3: REPORTS & DOWNLOAD HUB ------------------
    with tab_reports:
        st.subheader("Data Export Center")
        st.write("Generate and download forward-looking compliance records, environmental diagnostics datasets, and analytics ledgers.")

        rep_col1, rep_col2 = st.columns(2)

        with rep_col1:
            st.markdown("### 📄 Forward Predictive Executive Summary")
            st.write("Generates an individual summary text report containing future outbreak trend vectors.")
            
            report_txt = f"""MALARIA EARLY-WARNING OUTBREAK FORECAST REPORT
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
----------------------------------------------------------------------
TARGET GEOGRAPHIC SITE: {res['address']}
Topographic Elevation: {res['elevation']} Meters

CURRENT STATUS QUO BOUNDS:
- Current Transmission Index: {curr['Outbreak Risk %']}%
- Current Threat Level: {curr['Verdict']}

14-DAY PREDICTIVE FORECAST HORIZON OUTLOOK:
- Peak Outbreak Index Value: {max_future_risk}%
- Expected Surge Peak Date Calendar: {max_risk_row['Date']}
- Outbreak Threshold Condition: {"🚨 CRITICAL VECTOR INTERVENTION REQUIRED" if max_future_risk >= 48.0 else "✅ MAINTENANCE PROTOCOLS SUFFICIENT"}
----------------------------------------------------------------------
Notice: Field intelligence modeling derived from non-linear biological lag systems equations.
"""
            st.download_button(
                label="📥 Download Predictive Intelligence Summary (.txt)",
                data=report_txt,
                file_name=f"Malaria_Future_Forecast_Report_{res['name'].replace(' ', '_')}.txt",
                mime="text/plain",
                use_container_width=True
            )

        with rep_col2:
            st.markdown("### 🗃️ Session Search Audit Record History Ledger")
            st.write("Download an aggregated tabular audit record tracking all target geographic queries.")
            
            if st.session_state.audit_history:
                history_df = pd.DataFrame(st.session_state.audit_history)
                st.dataframe(history_df, use_container_width=True, height=150)
                
                csv_buffer = history_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Session Audit History (.csv)",
                    data=csv_buffer,
                    file_name="Malaria_Outbreak_Session_Audit_Ledger.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.info("No query sessions recorded in the buffer ledger yet.")

    # ------------------ TAB 4: MALARIA PREVENTION & CONTROL ------------------
    with tab_prevention:
        st.subheader("🛡️ Vector Control & Malaria Prevention Protocols")
        st.write("Deploying tactical environmental workflows and individual barriers based on WHO-aligned standards.")
        
        if max_future_risk >= 48.0:
            st.warning(f"⚠️ **Pre-Emptive Risk Action Required for {res['name']}:** A biological surge is projected to peak around **{max_risk_row['Date']}**. Mosquito controls, distribution audits for Long-Lasting Insecticidal Nets (LLINs), and proactive larviciding in known standing pooling sectors should be scheduled immediately *before* the peak date arrives.")
        else:
            st.info(f"💡 **Active Risk Guidance for {res['name']}:** No future outbreak cycles are indicated over this sub-seasonal corridor. Maintain baseline vector monitoring networks and normal source reductions.")
            
        st.markdown("---")
        prev_col1, prev_col2 = st.columns(2)
        
        with prev_col1:
            st.markdown("### 🏠 Personal & Household Protections")
            st.markdown("""
            * **Long-Lasting Insecticidal Nets (LLINs):** Sleep under factory-treated insecticidal mosquito bednets every night.
            * **Indoor Residual Spraying (IRS):** Apply recommended long-lasting chemical insecticides to inside walls and ceilings.
            * **Topical Repellents:** Apply spatial skin repellents containing active ingredients like DEET during peak vector biting hours.
            * **Structural Screening:** Install tight wire mesh screens on house windows and doors.
            """)
            
        with prev_col2:
            st.markdown("### 🚜 Environmental & Community Management")
            st.markdown("""
            * **Source Reduction & Drainage:** Eliminate stagnant fresh-water pools, clear blocked roadside ditches, and drain puddles.
            * **Biological Larviciding:** Apply regular targeted biological larvicides (such as Bti) into permanent wetland habitats.
            * **Ecosystem Clearing:** Clear high weeds and thick bush cover away from residential boundaries.
            * **Chemoprevention & Vaccination:** Coordinate execution of seasonal malaria chemoprevention protocols for vulnerable groups.
            """)
