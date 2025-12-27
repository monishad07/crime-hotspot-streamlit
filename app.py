import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from sklearn.cluster import KMeans
from folium.plugins import HeatMap

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Crime Hotspot Detection",
    layout="wide"
)

st.title("🚨 Crime Hotspot Detection – Chicago")

st.markdown("""
This application identifies **crime hotspots** in Chicago using **K-Means clustering**  
and visualizes crime density on an interactive map.
""")

# ---------------- Load Data ----------------
@st.cache_data
def load_data():
    return pd.read_csv("cleaned_chicago_crime.csv")

df = load_data()
coords = df[["LATITUDE", "LONGITUDE"]].dropna()

# ---------------- Sidebar Controls ----------------
st.sidebar.header("⚙️ Controls")

k = st.sidebar.slider(
    "Number of hotspots",
    min_value=2,
    max_value=10,
    value=5
)

show_heatmap = st.sidebar.checkbox("Show crime density heatmap")

# ---------------- KMeans Clustering ----------------
kmeans = KMeans(n_clusters=k, random_state=42)
coords["cluster"] = kmeans.fit_predict(coords)

hotspots = kmeans.cluster_centers_

# ---------------- Create Map ----------------
m = folium.Map(
    location=[coords["LATITUDE"].mean(), coords["LONGITUDE"].mean()],
    zoom_start=10
)

# Plot crime points (light gray)
for _, row in coords.iterrows():
    folium.CircleMarker(
        location=[row["LATITUDE"], row["LONGITUDE"]],
        radius=1,
        color="gray",
        fill=True,
        fill_opacity=0.3,
    ).add_to(m)

# Optional heatmap
if show_heatmap:
    HeatMap(coords[["LATITUDE", "LONGITUDE"]].values.tolist()).add_to(m)

# Plot hotspot centers
for i, (lat, lon) in enumerate(hotspots):
    folium.CircleMarker(
        location=[lat, lon],
        radius=14,
        color="blue",
        fill=True,
        fill_opacity=0.8,
        popup=f"Hotspot {i + 1}",
    ).add_to(m)

# ---------------- Show Map ----------------
st.subheader("📍 Crime Hotspot Map")
st_folium(m, width=1000, height=550)

# ---------------- Hotspot Summary ----------------
st.subheader("📊 Hotspot Summary")

summary = coords.groupby("cluster").size().reset_index(name="Crime Count")

def risk_level(count):
    if count > 500:
        return "High"
    elif count > 200:
        return "Medium"
    else:
        return "Low"

summary["Risk Level"] = summary["Crime Count"].apply(risk_level)
summary["Hotspot"] = summary["cluster"] + 1

summary = summary[["Hotspot", "Crime Count", "Risk Level"]]

st.dataframe(summary, use_container_width=True)

# ---------------- Footer ----------------
st.markdown("---")
st.caption("📌 Built using Streamlit, Folium & Machine Learning (K-Means)")
