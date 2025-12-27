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
coords = df[["LATITUDE", "LONGITUDE"]].dropna().copy()

# ---------------- KMeans Clustering ----------------
kmeans = KMeans(n_clusters=k, random_state=42)
coords["cluster"] = kmeans.fit_predict(coords).astype(int)

hotspots = kmeans.cluster_centers_

# ---------------- Color palette ----------------
CLUSTER_COLORS = [
    "red", "green", "purple", "orange", "darkred",
    "cadetblue", "darkgreen", "darkpurple", "pink", "black"
]

# ---------------- Create Map ----------------
m = folium.Map(
    location=[coords["LATITUDE"].mean(), coords["LONGITUDE"].mean()],
    zoom_start=10
)

# Plot crime points
for _, row in coords.iterrows():
    cluster_idx = int(row["cluster"])
    color = CLUSTER_COLORS[cluster_idx % len(CLUSTER_COLORS)]

    folium.CircleMarker(
        location=[row["LATITUDE"], row["LONGITUDE"]],
        radius=2,
        color=color,
        fill=True,
        fill_opacity=0.5,
    ).add_to(m)

# Optional Heatmap
if show_heatmap:
    HeatMap(coords[["LATITUDE", "LONGITUDE"]].values.tolist()).add_to(m)

# Plot hotspot centers
for i, (lat, lon) in enumerate(hotspots):
    folium.CircleMarker(
        location=[lat, lon],
        radius=14,
        color="blue",
        fill=True,
        fill_opacity=0.9,
        popup=f"Hotspot {i + 1}",
    ).add_to(m)

# ---------------- Display Map ----------------
st.subheader("📍 Crime Hotspot Map")
st_folium(m, width=1000, height=550)

# ---------------- KPI SECTION ----------------
st.subheader("📊 Crime Hotspot Insights")

summary = coords.groupby("cluster").size().reset_index(name="Crime Count")

total_crimes = len(coords)
active_hotspots = k
most_dense_hotspot = summary["Crime Count"].max()

col1, col2, col3 = st.columns(3)
col1.metric("Total Crime Points", total_crimes)
col2.metric("Active Hotspots", active_hotspots)
col3.metric("Most Dense Hotspot", most_dense_hotspot)

# ---------------- Optional Table ----------------
st.markdown("### 📋 Detailed Hotspot Breakdown")
show_table = st.checkbox("Show detailed table")

if show_table:
    summary["Hotspot"] = summary["cluster"] + 1

    def risk_level(count):
        if count > 500:
            return "High"
        elif count > 200:
            return "Medium"
        else:
            return "Low"

    summary["Risk Level"] = summary["Crime Count"].apply(risk_level)
    summary = summary[["Hotspot", "Crime Count", "Risk Level"]]

    st.dataframe(summary, use_container_width=True)

st.markdown("---")
st.caption("📌 Built using Streamlit, Folium & K-Means clustering")
