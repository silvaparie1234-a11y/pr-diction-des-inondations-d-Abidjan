import streamlit as st
import pandas as pd
import joblib
import folium
from streamlit_folium import st_folium
import plotly.express as px
import datetime
import numpy as np
from fpdf import FPDF
import requests

# Configuration de la page
st.set_page_config(page_title="Abidjan Flood Sentinel Pro", layout="wide", page_icon="🌊")

# --- CHARGEMENT DU MODÈLE ---
@st.cache_resource
def load_model():
    return joblib.load('flood_xgboost.pkl')

model = load_model()

# --- FONCTION MÉTÉO EN DIRECT ---
def get_live_weather():
    API_KEY = "0fd3d4ce78a76525f5a9cf1af7ce6dee"
    CITY = "Abidjan"
    url = f"http://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={API_KEY}&units=metric"
    try:
        response = requests.get(url).json()
        # OpenWeather renvoie la pluie en mm pour la dernière heure (si disponible)
        rain = response.get('rain', {}).get('1h', 0)
        temp = response.get('main', {}).get('temp', 25)
        return rain, temp
    except:
        return 0, 25

# --- FONCTION EXPORT PDF ---
def create_pdf(commune, risk, rain, level, impact):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, "RAPPORT OFFICIEL - SENTINELLE DES CRUES ABIDJAN", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, f"Commune : {commune}", ln=True)
    pdf.cell(200, 10, f"Date et Heure : {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}", ln=True)
    pdf.cell(200, 10, f"Indice de Risque : {risk*100:.1f}%", ln=True)
    pdf.cell(200, 10, f"Pluviometrie detectee : {rain} mm/h", ln=True)
    pdf.cell(200, 10, f"Niveau des eaux : {level} m", ln=True)
    pdf.cell(200, 10, f"Population exposee estimee : {impact:,} personnes", ln=True)
    pdf.ln(10)
    
    status = "ALERTE ROUGE : Plan d'urgence recommande." if risk > 0.7 else "VIGILANCE : Surveillance accrue."
    pdf.set_text_color(255, 0, 0) if risk > 0.7 else pdf.set_text_color(0, 0, 0)
    pdf.multi_cell(0, 10, f"CONCLUSION : {status}")
    return pdf.output(dest='S').encode('latin-1')

# --- DONNÉES DES COMMUNES ---
communes = {
    "Abobo": {"coords": [5.416, -4.018], "alt": 85, "drain": 0.4, "pop": 1100000},
    "Adjamé": {"coords": [5.358, -4.022], "alt": 40, "drain": 0.5, "pop": 370000},
    "Anyama": {"coords": [5.494, -4.051], "alt": 90, "drain": 0.6, "pop": 150000},
    "Attécoubé": {"coords": [5.337, -4.041], "alt": 15, "drain": 0.3, "pop": 260000},
    "Bingerville": {"coords": [5.355, -3.885], "alt": 45, "drain": 0.7, "pop": 70000},
    "Cocody": {"coords": [5.348, -3.988], "alt": 50, "drain": 0.8, "pop": 450000},
    "Koumassi": {"coords": [5.298, -3.948], "alt": 4, "drain": 0.3, "pop": 430000},
    "Marcory": {"coords": [5.302, -3.985], "alt": 5, "drain": 0.4, "pop": 250000},
    "Plateau": {"coords": [5.326, -4.019], "alt": 25, "drain": 0.9, "pop": 10000},
    "Port-Bouët": {"coords": [5.258, -3.938], "alt": 2, "drain": 0.5, "pop": 420000},
    "Songon": {"coords": [5.322, -4.266], "alt": 30, "drain": 0.6, "pop": 60000},
    "Treichville": {"coords": [5.300, -4.010], "alt": 6, "drain": 0.5, "pop": 100000},
    "Yopougon": {"coords": [5.347, -4.081], "alt": 45, "drain": 0.4, "pop": 1200000},
}

# --- SIDEBAR ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/f/fe/Flag_of_C%C3%B4te_d%27Ivoire.svg", width=100)
st.sidebar.title("🛡️ Contrôle Sentinelle")
selected_commune = st.sidebar.selectbox("Commune ciblée", list(communes.keys()))

st.sidebar.subheader("📡 Source des Données")
mode = st.sidebar.radio("Mode de fonctionnement", ["Direct Météo (API)", "Simulation Manuelle"])

if mode == "Direct Météo (API)":
    live_rain, live_temp = get_live_weather()
    rainfall = st.sidebar.number_input("Pluie actuelle (mm/h)", value=float(live_rain), disabled=True)
    st.sidebar.success(f"Connecté : {live_temp}°C à Abidjan")
else:
    rainfall = st.sidebar.slider("Pluie simulée (mm/h)", 0, 150, 40)

river_level = st.sidebar.slider("Niveau Lagune (m)", 0.0, 8.0, 2.5)
soil_moisture = st.sidebar.slider("Saturation Sol (%)", 0, 100, 50)

# Calcul du risque avec le modèle
c_data = communes[selected_commune]
input_df = pd.DataFrame([[rainfall, river_level, soil_moisture, c_data["alt"], c_data["drain"]]], 
                        columns=['rainfall_mm', 'river_level_m', 'soil_moisture_index', 'elevation_m', 'drainage_capacity'])
proba = model.predict_proba(input_df)[0][1]

# --- DASHBOARD ---
st.title(f"📍 État d'alerte : {selected_commune}")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Risque d'Inondation", f"{proba*100:.1f}%")
with col2:
    st.metric("Population à risque", f"{int(c_data['pop'] * proba):,}")
with col3:
    color = "🔴 CRITIQUE" if proba > 0.7 else ("🟠 VIGILANCE" if proba > 0.4 else "🟢 NORMAL")
    st.metric("Niveau d'Alerte", color)
with col4:
    st.write("Générer Document Officiel")
    pdf_file = create_pdf(selected_commune, proba, rainfall, river_level, int(c_data['pop'] * proba))
    st.download_button("📥 Export PDF", data=pdf_file, file_name=f"rapport_{selected_commune}.pdf", mime="application/pdf")

# ONGLETS
tab1, tab2, tab3 = st.tabs(["🗺️ Cartographie", "📈 Historique 24h", "📊 Facteurs Techniques"])

with tab1:
    m = folium.Map(location=[5.34, -4.00], zoom_start=11, tiles="CartoDB positron")
    folium.CircleMarker(
        location=c_data["coords"],
        radius=20,
        color='red' if proba > 0.5 else 'green',
        fill=True,
        popup=f"Alerte {selected_commune}"
    ).add_to(m)
    st_folium(m, width="100%", height=450)

with tab2:
    st.subheader("📊 Tendance du risque sur les dernières 24h")
    # Création d'un historique simulé basé sur le risque actuel
    times = [datetime.datetime.now() - datetime.timedelta(hours=x) for x in range(24, 0, -1)]
    history = [max(0, min(100, proba*100 + np.random.normal(0, 7))) for _ in range(24)]
    fig_hist = px.area(x=times, y=history, labels={'x': 'Temps', 'y': 'Risque %'}, color_discrete_sequence=['#e74c3c'])
    st.plotly_chart(fig_hist, use_container_width=True)

with tab3:
    st.subheader("🧬 Analyse Radar des Variables")
    features = ['Pluie', 'Niveau Lagune', 'Humidité Sol', 'Défaut Drainage']
    vals = [rainfall/1.5, river_level*12, soil_moisture, (1-c_data['drain'])*100]
    df_radar = pd.DataFrame(dict(r=vals, theta=features))
    fig_radar = px.line_polar(df_radar, r='r', theta='theta', line_close=True)
    fig_radar.update_traces(fill='toself')
    st.plotly_chart(fig_radar, use_container_width=True)
