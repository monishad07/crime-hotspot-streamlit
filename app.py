import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
from sklearn.cluster import KMeans
from folium.plugins import HeatMap
import streamlit.components.v1 as components

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Crime Hotspot Analysis – Chicago",
    layout="wide"
)

# ---------------- Custom CSS ----------------
st.markdown("""
<style>
.main-title {
    font-size: 36px;
    font-weight: 700;
    color: #1f2937;
}
.subtitle {
    font-size: 16px;
    color: #4b5563;
    margin-bottom: 25px;
}
.section-title {
    font-size: 24px;
    font-weight: 600;
    color: #111827;
    margin-top: 30px;
}
.metric-box {
    background-color: #f9fafb;
    padding: 20px;
    border-radius: 10px;
    border: 1px solid #e5e7eb;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Title ----------------
st.markdown('<div class="main-title">Crime Hotspot Analysis – Chicago</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">'
    'This application performs spatial crime analysis for the city of Chicago using '
    'K-Means clustering to identify high-density crime zones and visualize spatial patterns.'
    '</div>',
    unsafe_allow_html=True
)

# ---------------- Load Data ----------------
@st.cache_data
def load_data():
    return pd.read_csv("cleaned_chicago_crime.csv")

df = load_data()

# ---------------- Sidebar ----------------
st.sidebar.markdown("### Analysis Configuration")
st.sidebar.markdown("Adjust clustering and visualization parameters")

k = st.sidebar.slider(
    "Number of hotspot regions",
    min_value=2,
    max_value=10,
    value=5
)

show_heatmap = st.sidebar.checkbox(
    "Enable crime density heatmap",
    value=False
)

# ---------------- Prepare coordinates ----------------
coords = df[["LATITUDE", "LONGITUDE"]].dropna().copy()

# ---------------- KMeans Clustering ----------------
kmeans = KMeans(n_clusters=k, random_state=42)
coords["cluster"] = kmeans.fit_predict(coords).astype(int)
hotspots = kmeans.cluster_centers_

# ---------------- Color palette ----------------
CLUSTER_COLORS = [
    "#dc2626", "#16a34a", "#7c3aed", "#ea580c", "#7f1d1d",
    "#0891b2", "#166534", "#6b21a8", "#be185d", "#111827"
]

# ---------------- Create Map ----------------
m = folium.Map(
    location=[coords["LATITUDE"].mean(), coords["LONGITUDE"].mean()],
    zoom_start=10,
    tiles="CartoDB positron"
)

# Plot crime points
for _, row in coords.iterrows():
    folium.CircleMarker(
        location=[row["LATITUDE"], row["LONGITUDE"]],
        radius=2,
        color=CLUSTER_COLORS[int(row["cluster"]) % len(CLUSTER_COLORS)],
        fill=True,
        fill_opacity=0.5,
        weight=0
    ).add_to(m)

# Heatmap
if show_heatmap:
    HeatMap(
        coords[["LATITUDE", "LONGITUDE"]].values.tolist(),
        radius=15,
        blur=10
    ).add_to(m)

# Hotspot centers
for i, (lat, lon) in enumerate(hotspots):
    folium.CircleMarker(
        location=[lat, lon],
        radius=14,
        color="#1d4ed8",
        fill=True,
        fill_opacity=0.9,
        popup=f"Hotspot Region {i + 1}",
    ).add_to(m)

# ---------------- Map + Legend ----------------
st.markdown('<div class="section-title">Spatial Crime Hotspot Map</div>', unsafe_allow_html=True)

col_map, col_legend = st.columns([4.2, 1])

with col_map:
    folium_static(m, width=900, height=560)

with col_legend:
    components.html(
        """
        <div style="
            background-color:#ffffff;
            padding:18px;
            border-radius:10px;
            border:1px solid #e5e7eb;
            font-family:Arial;
        ">
            <h4 style="margin-top:0;color:#111827;">Map Legend</h4>

            <b>Crime Clusters</b><br>
            <span style="color:#dc2626;">●</span> Cluster 1<br>
            <span style="color:#16a34a;">●</span> Cluster 2<br>
            <span style="color:#7c3aed;">●</span> Cluster 3<br>
            <span style="color:#ea580c;">●</span> Cluster 4<br>
            <span style="color:#7f1d1d;">●</span> Cluster 5<br><br>

            <b>Hotspot Center</b><br>
            Blue circle represents centroid of a hotspot region
        </div>
        """,
        height=300
    )

# ---------------- KPI SECTION ----------------
st.markdown('<div class="section-title">Crime Hotspot Summary</div>', unsafe_allow_html=True)

summary = coords.groupby("cluster").size().reset_index(name="Crime Count")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
    st.metric("Total Crime Records", len(coords))
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
    st.metric("Identified Hotspot Regions", k)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
    st.metric("Highest Cluster Density", summary["Crime Count"].max())
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Optional Table ----------------
st.markdown('<div class="section-title">Hotspot Distribution Details</div>', unsafe_allow_html=True)

show_table = st.checkbox("Display detailed hotspot statistics")

if show_table:
    summary["Hotspot Region"] = summary["cluster"] + 1

    def risk_level(count):
        if count > 500:
            return "High"
        elif count > 200:
            return "Moderate"
        else:
            return "Low"

    summary["Risk Classification"] = summary["Crime Count"].apply(risk_level)
    summary = summary[["Hotspot Region", "Crime Count", "Risk Classification"]]

    st.dataframe(summary, use_container_width=True)

# ---------------- Footer ----------------
st.markdown("---")
st.caption(
    "Spatial crime analysis dashboard developed using Streamlit, Folium, and K-Means clustering."
)
