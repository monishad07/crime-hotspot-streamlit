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
st.markdown(
    "This application identifies **crime hotspots** in Chicago using K-Means clustering "
    "and visualizes crime density on an interactive map."
)

# ---------------- Load Data ----------------
@st.cache_data
def load_data():
    return pd.read_csv("cleaned_chicago_crime.csv")

df = load_data()

# ---------------- Sidebar Controls ----------------
st.sidebar.header("⚙️ Controls")
k = st.sidebar.slider("Number of hotspots", 2, 10, 5)
show_heatmap = st.sidebar.checkbox("Show crime density heatmap")

# ---------------- Prepare coordinates ----------------
# Use a copy so we can safely assign new columns
coords = df[["LATITUDE", "LONGITUDE"]].dropna().copy()

# ---------------- KMeans Clustering ----------------
kmeans = KMeans(n_clusters=k, random_state=42)
# fit_predict returns numeric labels; convert explicitly to int dtype
coords["cluster"] = kmeans.fit_predict(coords).astype(int)

hotspots = kmeans.cluster_centers_

# ---------------- Color palette for clusters ----------------
CLUSTER_COLORS = [
    "red", "green", "purple", "orange", "darkred",
    "cadetblue", "darkgreen", "darkpurple", "pink", "black"
]

# ---------------- Create map ----------------
m = folium.Map(
    location=[coords["LATITUDE"].mean(), coords["LONGITUDE"].mean()],
    zoom_start=10
)

# Plot crime points colored by cluster
for _, row in coords.iterrows():
    # ensure cluster index is an int (defensive)
    cluster_idx = int(row["cluster"])
    color = CLUSTER_COLORS[cluster_idx % len(CLUSTER_COLORS)]
    folium.CircleMarker(
        location=[row["LATITUDE"], row["LONGITUDE"]],
        radius=2,
        color=color,
        fill=True,
        fill_opacity=0.5,
    ).add_to(m)

# Optional heatmap overlay
if show_heatmap:
    HeatMap(coords[["LATITUDE", "LONGITUDE"]].values.tolist()).add_to(m)

# Plot hotspot centers (blue)
for i, (lat, lon) in enumerate(hotspots):
    folium.CircleMarker(
        location=[lat, lon],
        radius=14,
        color="blue",
        fill=True,
        fill_opacity=0.9,
        popup=f"Hotspot {i + 1}",
    ).add_to(m)

# ---------------- Display map ----------------
st.subheader("📍 Crime Hotspot Map")
st_folium(m, width=1000, height=550)

# ---------------- Hotspot summary (KPI cards) ----------------
st.subheader("📊 Crime Hotspot Insights")
summary = coords.groupby("cluster").size().reset_index(name="Crime Count")

# KPIs
total_crimes = len(coords)
active_hotspots = k
most_dense_hotspot = summary["Crime Count"].max()

col1, col2, col3 = st.columns(3)
col1.metric("Total Crime Points", total_crimes)
col2.metric("Active Hotspots", active_hotspots)
col3.metric("Most Dense Hotspot", most_dense_hotspot)

st.markdown("---")
st.caption("📌 Built using Streamlit, Folium & K-Means clustering")
