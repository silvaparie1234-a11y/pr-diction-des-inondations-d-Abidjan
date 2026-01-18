import streamlit as st
import pandas as pd
import joblib
import folium
from streamlit_folium import st_folium
import plotly.express as px
import os

# Configuration de la page
st.set_page_config(page_title="Abidjan Flood Sentinel Pro", layout="wide")

# --- CHARGEMENT DU MODÈLE ---
@st.cache_resource
def load_model():
    # Correction : Le fichier est à la racine sur GitHub, pas dans un dossier /models
    model_path = 'flood_xgboost.pkl'
    return joblib.load(model_path)

# Chargement du modèle
model = load_model()

# --- DONNÉES DES 13 COMMUNES D'ABIDJAN ---
# [Latitude, Longitude, Altitude(m), Capacité_Drainage(0-1), Population_Estimée]
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

# --- SIDEBAR (PARAMÈTRES) ---
st.sidebar.title("🛠️ Paramètres Métro")
selected_commune = st.sidebar.selectbox("Sélectionner la Commune", list(communes.keys()))

st.sidebar.subheader("🌡️ Données Temps Réel")
rainfall = st.sidebar.slider("Intensité Pluie (mm/h)", 0, 150, 50)
river_level = st.sidebar.slider("Niveau Lagune/Canaux (m)", 0.0, 8.0, 2.5)
soil_moisture = st.sidebar.slider("Saturation du sol (%)", 0, 100, 60)

# Extraction des données de la commune
c_data = communes[selected_commune]
elevation = c_data["alt"]
drainage = c_data["drain"]

# --- CALCUL DU RISQUE ---
# Création du DataFrame pour le modèle (doit correspondre exactement aux colonnes d'entraînement)
input_data = pd.DataFrame([[rainfall, river_level, soil_moisture, elevation, drainage]], 
                          columns=['rainfall_mm', 'river_level_m', 'soil_moisture_index', 'elevation_m', 'drainage_capacity'])

proba = model.predict_proba(input_data)[0][1]

# --- INTERFACE PRINCIPALE ---
st.header(f"📍 Surveillance : {selected_commune}")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Risque d'Inondation", f"{proba*100:.1f}%")
with col2:
    st.metric("Altitude", f"{elevation} m")
with col3:
    # Calcul de la population impactée
    impact = int(c_data["pop"] * proba) if proba > 0.4 else 0
    st.metric("Pop. Exposée", f"{impact:,}")
with col4:
    color_label = "🔴 CRITIQUE" if proba > 0.7 else ("🟠 VIGILANCE" if proba > 0.4 else "🟢 NORMAL")
    st.metric("Statut Alerte", color_label)

# --- CARTE & ANALYSE ---
tab1, tab2 = st.tabs(["🗺️ Carte de Vigilance", "📈 Analyse des Facteurs"])

with tab1:
    # Création de la carte centrée sur Abidjan
    m = folium.Map(location=[5.34, -4.00], zoom_start=11, tiles="CartoDB positron")
    
    for name, info in communes.items():
        is_selected = (name == selected_commune)
        # Détermination de la couleur du point
        if is_selected:
            dot_color = 'red' if proba > 0.7 else ('orange' if proba > 0.4 else 'green')
            size = 20
        else:
            dot_color = 'blue'
            size = 8
            
        folium.CircleMarker(
            location=info["coords"],
            radius=size,
            color=dot_color,
            fill=True,
            fill_opacity=0.7,
            popup=f"{name} (Alt: {info['alt']}m)"
        ).add_to(m)
    
    st_folium(m, width="100%", height=500)

with tab2:
    st.subheader("📊 Facteurs d'Influence (Radar)")
    # Graphique radar pour visualiser les causes du risque
    features = ['Pluie', 'Niveau Eau', 'Saturation Sol', 'Défaut Drainage']
    # Normalisation des valeurs pour le graphique
    vals = [rainfall/1.5, river_level*10, soil_moisture, (1-drainage)*100]
    
    df_radar = pd.DataFrame(dict(r=vals, theta=features))
    fig = px.line_polar(df_radar, r='r', theta='theta', line_close=True, range_r=[0,100])
    fig.update_traces(fill='toself', line_color='red' if proba > 0.5 else 'blue')
    st.plotly_chart(fig, use_container_width=True)

# Message d'alerte dynamique
if proba > 0.7:
    st.error(f"⚠️ URGENCE : Risque critique d'inondation à {selected_commune}. Les autorités recommandent la mise en sécurité immédiate.")
elif proba > 0.4:
    st.warning(f"⚠️ VIGILANCE : Risque modéré à {selected_commune}. Surveillez la montée des eaux.")
