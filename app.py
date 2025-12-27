import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from sklearn.cluster import KMeans
from folium.plugins import HeatMap
import numpy as np

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Crime Hotspot Detection",
    layout="wide"
)

st.title("🚨 Crime Hotspot Detection – Chicago")

# ---------------- Load Data ----------------
@st.cache_data
def load_data():
    return pd.read_csv("cleaned_chicago_crime.csv")

df = load_data()

# ---------------- Prepare coordinates ----------------
coords = df[["LATITUDE", "LONGITUDE"]].dropna().copy()

# ---------------- Elbow Method ----------------
inertias = []
K_RANGE = range(2, 11)

for k_val in K_RANGE:
    km = KMeans(n_clusters=k_val, random_state=42)
    km.fit(coords)
    inertias.append(km.inertia_)

# ---------------- Sidebar Controls ----------------
st.sidebar.header("⚙️ Controls")
st.sidebar.subheader("📉 Optimal Hotspot Count (Elbow Method)")

elbow_df = pd.DataFrame({
    "Number of Clusters (k)": list(K_RANGE),
    "Inertia": inertias
})

st.sidebar.line_chart(elbow_df.set_index("Number of Clusters (k)"))

k = st.sidebar.slider("Number of hotspots", 2, 10, 5)
show_heatmap = st.sidebar.checkbox("Show crime density heatmap")

# ---------------- KMeans Clustering ----------------
kmeans = KMeans(n_clusters=k, random_state=42)
coords["cluster"] = kmeans.fit_predict(coords)

hotspots = kmeans.cluster_centers_

# ---------------- Map ----------------
m = folium.Map(
    location=[coords["LATITUDE"].mean(), coords["LONGITUDE"].mean()],
    zoom_start=10
)

for _, row in coords.iterrows():
    folium.CircleMarker(
        location=[row["LATITUDE"], row["LONGITUDE"]],
        radius=2,
        color="red",
        fill=True,
        fill_opacity=0.5,
    ).add_to(m)

if show_heatmap:
    HeatMap(coords[["LATITUDE", "LONGITUDE"]].values.tolist()).add_to(m)

for i, (lat, lon) in enumerate(hotspots):
    folium.CircleMarker(
        location=[lat, lon],
        radius=12,
        color="blue",
        fill=True,
        fill_opacity=0.9,
        popup=f"Hotspot {i + 1}",
    ).add_to(m)

st_folium(m, width=1000, height=550)
