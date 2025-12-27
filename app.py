import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from sklearn.cluster import KMeans
from folium.plugins import HeatMap

# Page config
st.set_page_config(page_title="Crime Hotspot Detection", layout="wide")
st.title("🚨 Crime Hotspot Detection - Chicago")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("cleaned_chicago_crime.csv")

df = load_data()

st.write("### Dataset Preview")
st.dataframe(df.head())

# Slider for number of hotspots
k = st.slider("Select number of hotspots", 2, 10, 5)

# Checkbox for heatmap
show_heatmap = st.checkbox("Show crime density heatmap")

# Remove missing values
coords = df[["LATITUDE", "LONGITUDE"]].dropna().copy()

kmeans = KMeans(n_clusters=k, random_state=42)
coords["cluster"] = kmeans.fit_predict(coords)

# Get hotspot centers
hotspots = kmeans.cluster_centers_

# Create map
m = folium.Map(
    location=[coords["LATITUDE"].mean(), coords["LONGITUDE"].mean()],
    zoom_start=10
)

# Colors for clusters
colors = [
    "red", "green", "purple", "orange", "darkred",
    "lightred", "beige", "darkblue", "darkgreen", "cadetblue"
]

# Plot crime points (cluster-colored)
for _, row in coords.iterrows():
    folium.CircleMarker(
        location=[row["LATITUDE"], row["LONGITUDE"]],
        radius=2,
        color=colors[row["cluster"] % len(colors)],
        fill=True,
        fill_opacity=0.5,
    ).add_to(m)

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

# Optional heatmap
if show_heatmap:
    HeatMap(coords[["LATITUDE", "LONGITUDE"]].values.tolist()).add_to(m)

# Show map
st.write("### Crime Hotspot Map")
st_folium(m, width=900, height=500)
