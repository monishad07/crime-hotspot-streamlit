import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
from sklearn.cluster import KMeans
from folium.plugins import HeatMap
import streamlit.components.v1 as components

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Chicago Crime Hotspot Analysis",
    layout="wide",
)

# ---------------- Custom CSS for Styling ----------------
st.markdown(
    """
    <style>
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 12px;
    }
    /* Title styling */
    .stApp h1 {
        color: #1f2937;
        font-size: 2.2rem;
        font-weight: 700;
    }
    .stApp h2 {
        color: #111827;
    }
    .metric-label, .stMetric {
        font-size: 1.1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- Page Title ----------------
st.title("Chicago Crime Hotspot Analysis")
st.markdown(
    "Visualize and explore **crime hotspots** in Chicago with interactive mapping and clustering insights.",
)

# ---------------- Load Data ----------------
@st.cache_data
def load_data():
    return pd.read_csv("cleaned_chicago_crime.csv")

df = load_data()

# ---------------- Sidebar ----------------
st.sidebar.header("Controls")
st.sidebar.markdown("Adjust clustering and visualization options:")

k = st.sidebar.slider("Number of Hotspots", 2, 10, 5)
show_heatmap = st.sidebar.checkbox("Show Crime Density Heatmap")

# ---------------- Prepare coordinates ----------------
coords = df[["LATITUDE", "LONGITUDE"]].dropna().copy()

# ---------------- KMeans Clustering ----------------
kmeans = KMeans(n_clusters=k, random_state=42)
coords["cluster"] = kmeans.fit_predict(coords).astype(int)
hotspots = kmeans.cluster_centers_

# ---------------- Color palette ----------------
CLUSTER_COLORS = [
    "#EF4444", "#10B981", "#8B5CF6", "#F59E0B", "#B91C1C",
    "#3B82F6", "#059669", "#7C3AED", "#EC4899", "#000000"
]

# ---------------- Create Map ----------------
m = folium.Map(
    location=[coords["LATITUDE"].mean(), coords["LONGITUDE"].mean()],
    zoom_start=10,
    tiles="CartoDB positron",
)

# Crime points
for _, row in coords.iterrows():
    color = CLUSTER_COLORS[int(row["cluster"]) % len(CLUSTER_COLORS)]
    folium.CircleMarker(
        location=[row["LATITUDE"], row["LONGITUDE"]],
        radius=3,
        color=color,
        fill=True,
        fill_opacity=0.6,
    ).add_to(m)

# Heatmap overlay
if show_heatmap:
    HeatMap(coords[["LATITUDE", "LONGITUDE"]].values.tolist(), radius=15).add_to(m)

# Hotspot centers
for i, (lat, lon) in enumerate(hotspots):
    folium.CircleMarker(
        location=[lat, lon],
        radius=12,
        color="#2563EB",
        fill=True,
        fill_opacity=0.9,
        popup=f"Hotspot {i+1}",
    ).add_to(m)

# ---------------- Map + Legend Layout ----------------
st.subheader("Crime Hotspot Map")
col_map, col_legend = st.columns([4, 1])

with col_map:
    folium_static(m, width=900, height=550)

with col_legend:
    components.html(
        """
        <div style="
            background-color: #ffffff;
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #d1d5db;
            box-shadow: 1px 1px 8px rgba(0,0,0,0.1);
            font-family: Arial;
            font-size: 0.95rem;
        ">
            <h4 style="margin-bottom:10px; color:#111827;">Map Legend</h4>
            <b>Crime Clusters</b><br>
            <span style="color:#EF4444;">●</span> Cluster 1<br>
            <span style="color:#10B981;">●</span> Cluster 2<br>
            <span style="color:#8B5CF6;">●</span> Cluster 3<br>
            <span style="color:#F59E0B;">●</span> Cluster 4<br>
            <span style="color:#B91C1C;">●</span> Cluster 5<br><br>
            <b style="color:#2563EB;">● Blue Circle</b> Hotspot Center<br>
            <b>🔥 Heatmap</b> Crime Density (Darker = Higher)
        </div>
        """,
        height=280,
    )

# ---------------- KPI Section ----------------
st.subheader("Crime Hotspot Insights")
summary = coords.groupby("cluster").size().reset_index(name="Crime Count")

total_crimes = len(coords)
active_hotspots = k
most_dense_hotspot = summary["Crime Count"].max()

col1, col2, col3 = st.columns(3)
col1.metric("Total Crime Points", total_crimes)
col2.metric("Active Hotspots", active_hotspots)
col3.metric("Most Dense Hotspot", most_dense_hotspot)

# ---------------- Optional Detailed Table ----------------
st.markdown("### Detailed Hotspot Breakdown")
show_table = st.checkbox("Show Table")
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
st.caption("Built with Streamlit, Folium, and K-Means clustering")
