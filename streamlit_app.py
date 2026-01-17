import streamlit as st
import pandas as pd
import joblib
import folium
from streamlit_folium import st_folium
import plotly.express as px
import os

st.set_page_config(page_title="Abidjan Flood Sentinel Pro", layout="wide")

# --- CHARGEMENT DU MODÈLE ---
@st.cache_resource
def load_model():
    return joblib.load('models/flood_xgboost.pkl')

model = load_model()

# --- DONNÉES DES COMMUNES D'ABIDJAN ---
# Données approximatives : [Lat, Lon, Altitude(m), Capacité_Drainage(0-1), Population_Estimée]
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

st.sidebar.title("🛠️ Paramètres Métro")
selected_commune = st.sidebar.selectbox("Sélectionner la Commune", list(communes.keys()))

# --- SIMULATION CAPTEURS ---
st.sidebar.subheader("🌡️ Données Temps Réel")
rainfall = st.sidebar.slider("Intensité Pluie (mm/h)", 0, 150, 50)
river_level = st.sidebar.slider("Niveau Lagune/Canaux (m)", 0.0, 8.0, 2.5)
soil_moisture = st.sidebar.slider("Saturation du sol (%)", 0, 100, 60)

# Récupération des données de la commune choisie
c_data = communes[selected_commune]
elevation = c_data["alt"]
drainage = c_data["drain"]

# --- CALCUL DU RISQUE ---
input_data = pd.DataFrame([[rainfall, river_level, soil_moisture, elevation, drainage]], 
                          columns=['rainfall_mm', 'river_level_m', 'soil_moisture_index', 'elevation_m', 'drainage_capacity'])

proba = model.predict_proba(input_data)[0][1]

# --- INTERFACE ---
st.header(f"📍 Surveillance : {selected_commune}")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Risque d'Inondation", f"{proba*100:.1f}%")
with col2:
    st.metric("Altitude", f"{elevation} m")
with col3:
    impact = int(c_data["pop"] * proba) if proba > 0.4 else 0
    st.metric("Pop. Exposée", f"{impact:,}")
with col4:
    color = "🔴" if proba > 0.7 else ("🟠" if proba > 0.4 else "🟢")
    st.metric("Statut Alerte", color)

# --- CARTE & ANALYSE ---
tab1, tab2 = st.tabs(["🗺️ Carte de Vigilance", "📈 Analyse des Facteurs"])

with tab1:
    m = folium.Map(location=[5.34, -4.00], zoom_start=11, tiles="CartoDB positron")
    
    # Dessiner toutes les communes pour voir la situation globale
    for name, info in communes.items():
        # Pour le point sélectionné, on calcule le risque réel, pour les autres on met un bleu neutre
        is_selected = (name == selected_commune)
        circle_color = 'red' if (is_selected and proba > 0.7) else ('blue' if not is_selected else 'orange')
        
        folium.CircleMarker(
            location=info["coords"],
            radius=10 if not is_selected else 20,
            color=circle_color,
            fill=True,
            popup=f"{name} (Alt: {info['alt']}m)"
        ).add_to(m)
    
    st_folium(m, width="100%", height=500)

with tab2:
    # Graphique de comparaison des risques
    features = ['Pluie', 'Niveau Eau', 'Sol', 'Défaut Drainage']
    # On inverse l'altitude car plus c'est bas, plus c'est risqué
    vals = [rainfall/1.5, river_level*10, soil_moisture/2, (1-drainage)*100]
    fig = px.line_polar(r=vals, theta=features, line_close=True, range_r=[0,100])
    fig.update_traces(fill='toself')
    st.plotly_chart(fig)

if proba > 0.7:
    st.error(f"⚠️ URGENCE : La commune de {selected_commune} présente un risque critique d'inondation. Déclenchement du plan ORSEC suggéré.")