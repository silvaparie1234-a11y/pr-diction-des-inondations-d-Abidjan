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
import sqlite3  # Pour la base de données

# --- CONFIGURATION ---
st.set_page_config(page_title="Abidjan Flood Sentinel Pro", layout="wide", page_icon="🌊")

# --- INITIALISATION BASE DE DONNÉES ---
def init_db():
    conn = sqlite3.connect('flood_history.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS history 
                 (timestamp DATETIME, commune TEXT, risk REAL, rain REAL)''')
    conn.commit()
    conn.close()

def save_prediction(commune, risk, rain):
    conn = sqlite3.connect('flood_history.db')
    c = conn.cursor()
    c.execute("INSERT INTO history VALUES (?, ?, ?, ?)", 
              (datetime.datetime.now(), commune, risk*100, rain))
    conn.commit()
    conn.close()

def get_history():
    conn = sqlite3.connect('flood_history.db')
    df = pd.read_sql_query("SELECT * FROM history ORDER BY timestamp DESC LIMIT 50", conn)
    conn.close()
    return df

init_db()

# --- CHARGEMENT DU MODÈLE ---
@st.cache_resource
def load_model():
    return joblib.load('flood_xgboost.pkl')

model = load_model()

# --- MÉTÉO ---
def get_live_weather():
    API_KEY = "0fd3d4ce78a76525f5a9cf1af7ce6dee"
    url = f"http://api.openweathermap.org/data/2.5/weather?q=Abidjan&appid={API_KEY}&units=metric"
    try:
        r = requests.get(url).json()
        return r.get('rain', {}).get('1h', 0), r.get('main', {}).get('temp', 25)
    except: return 0, 25

# --- SIDEBAR & LOGO ---
# Remplace l'URL par le lien de ton image sur GitHub si tu en as une
LOGO_URL = "https://cdn-icons-png.flaticon.com/512/4005/4005817.png" 
st.sidebar.image(LOGO_URL, width=100)
st.sidebar.title("Flood Sentinel Pro")

# --- FORMULAIRE ---
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

selected_commune = st.sidebar.selectbox("Zone", list(communes.keys()))
mode = st.sidebar.radio("Données", ["Direct Météo", "Manuel"])

if mode == "Direct Météo":
    rain, temp = get_live_weather()
    st.sidebar.success(f"Direct : {temp}°C")
else:
    rain = st.sidebar.slider("Pluie (mm/h)", 0, 150, 40)

level = st.sidebar.slider("Lagune (m)", 0.0, 8.0, 2.0)
soil = st.sidebar.slider("Sol (%)", 0, 100, 50)

# Calcul
c = communes[selected_commune]
df_in = pd.DataFrame([[rain, level, soil, c['alt'], c['drain']]], columns=['rainfall_mm', 'river_level_m', 'soil_moisture_index', 'elevation_m', 'drainage_capacity'])
proba = model.predict_proba(df_in)[0][1]

# Sauvegarde automatique dans la base de données
save_prediction(selected_commune, proba, rain)

# --- DASHBOARD ---
st.title(f"📍 Surveillance : {selected_commune}")
cols = st.columns(4)
cols[0].metric("Risque", f"{proba*100:.1f}%")
cols[1].metric("Impact Pop.", f"{int(c['pop']*proba):,}")
cols[2].metric("Statut", "🔴 ALERTE" if proba > 0.7 else "🟢 OK")

# Onglets
t1, t2 = st.tabs(["🗺️ Carte & Radar", "📜 Historique Réel (Database)"])

with t1:
    m = folium.Map(location=[5.34, -4.00], zoom_start=11)
    folium.CircleMarker(c['coords'], radius=20, color='red', fill=True).add_to(m)
    st_folium(m, width="100%", height=400)

with t2:
    st.subheader("Données enregistrées en base de données")
    hist_df = get_history()
    if not hist_df.empty:
        st.dataframe(hist_df, use_container_width=True)
        fig = px.line(hist_df, x='timestamp', y='risk', color='commune', title="Évolution des risques consultés")
        st.plotly_chart(fig)
    else:
        st.info("Aucun historique pour le moment.")
