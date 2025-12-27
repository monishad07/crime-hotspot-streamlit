import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from sklearn.cluster import KMeans

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

# Remove missing values
coords = df[["LATITUDE", "LONGITUDE"]].dropna()

# Apply KMeans using slider value
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(coords)

# Get hotspot locations
hotspots = kmeans.cluster_centers_

# Create map
m = folium.Map(
    location=[coords["LATITUDE"].mean(), coords["LONGITUDE"].mean()],
    zoom_start=10
)

# Optional: plot crime points (light)
for _, row in coords.iterrows():
    folium.CircleMarker(
        location=[row["LATITUDE"], row["LONGITUDE"]],
        radius=1,
        color="gray",
        fill=True,
        fill_opacity=0.3,
    ).add_to(m)

# Plot hotspots (IMPORTANT PART)
for i, (lat, lon) in enumerate(hotspots):
    folium.CircleMarker(
        location=[lat, lon],
        radius=12,
        color="blue",
        fill=True,
        fill_opacity=0.8,
        popup=f"Hotspot {i+1}",
    ).add_to(m)

# Show map
st.write("### Crime Hotspot Map")
st_folium(m, width=900, height=500)
