import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from sklearn.cluster import KMeans

# Page title
st.set_page_config(page_title="Crime Hotspot Detection", layout="wide")
st.title("🚨 Crime Hotspot Detection - Chicago")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("cleaned_chicago_crime.csv")

df = load_data()

st.write("### Dataset Preview")
st.dataframe(df.head())

# Slider for clusters
k = st.slider("Select number of hotspots", 2, 10, 5)

# Clustering
kmeans = KMeans(n_clusters=k, random_state=42)
df["cluster"] = kmeans.fit_predict(df[["LATITUDE", "LONGITUDE"]])

# Create map
m = folium.Map(location=[df["LATITUDE"].mean(), df["LONGITUDE"].mean()], zoom_start=10)

# Plot points
for _, row in df.iterrows():
    folium.CircleMarker(
        location=[row["LATITUDE"], row["LONGITUDE"]],
        radius=2,
        color="red",
        fill=True,
        fill_opacity=0.6,
    ).add_to(m)

# Show map
st.write("### Crime Hotspot Map")
st_folium(m, width=900, height=500)

