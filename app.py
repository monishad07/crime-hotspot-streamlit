# import streamlit as st
# import pandas as pd
# import folium
# from streamlit_folium import folium_static
# from sklearn.cluster import KMeans
# from folium.plugins import HeatMap
# import streamlit.components.v1 as components


# # ---------------- Page Config ----------------
# st.set_page_config(
#     page_title="Crime Hotspot Detection",
#     layout="wide"
# )

# st.title("🚨 Crime Hotspot Detection – Chicago")
# st.markdown(
#     "This application identifies **crime hotspots** in Chicago using K-Means clustering "
#     "and visualizes crime density on an interactive map."
# )

# # ---------------- Load Data ----------------
# @st.cache_data
# def load_data():
#     return pd.read_csv("cleaned_chicago_crime.csv")

# df = load_data()

# # ---------------- Sidebar Controls ----------------
# st.sidebar.header("⚙️ Controls")
# k = st.sidebar.slider("Number of hotspots", 2, 10, 5)
# show_heatmap = st.sidebar.checkbox("Show crime density heatmap")

# # ---------------- Prepare coordinates ----------------
# coords = df[["LATITUDE", "LONGITUDE"]].dropna().copy()

# # ---------------- KMeans Clustering ----------------
# kmeans = KMeans(n_clusters=k, random_state=42)
# coords["cluster"] = kmeans.fit_predict(coords).astype(int)
# hotspots = kmeans.cluster_centers_

# # ---------------- Color palette ----------------
# CLUSTER_COLORS = [
#     "red", "green", "purple", "orange", "darkred",
#     "cadetblue", "darkgreen", "darkpurple", "pink", "black"
# ]

# # ---------------- Create Map ----------------
# m = folium.Map(
#     location=[coords["LATITUDE"].mean(), coords["LONGITUDE"].mean()],
#     zoom_start=10
# )

# # Plot crime points
# for _, row in coords.iterrows():
#     color = CLUSTER_COLORS[int(row["cluster"]) % len(CLUSTER_COLORS)]
#     folium.CircleMarker(
#         location=[row["LATITUDE"], row["LONGITUDE"]],
#         radius=2,
#         color=color,
#         fill=True,
#         fill_opacity=0.5,
#     ).add_to(m)

# # Heatmap
# if show_heatmap:
#     HeatMap(coords[["LATITUDE", "LONGITUDE"]].values.tolist()).add_to(m)

# # Hotspot centers
# for i, (lat, lon) in enumerate(hotspots):
#     folium.CircleMarker(
#         location=[lat, lon],
#         radius=14,
#         color="blue",
#         fill=True,
#         fill_opacity=0.9,
#         popup=f"Hotspot {i + 1}",
#     ).add_to(m)

# # ---------------- MAP + LEGEND LAYOUT ----------------
# st.subheader("📍 Crime Hotspot Map")

# col_map, col_legend = st.columns([4, 1])

# with col_map:
#     folium_static(m, width=900, height=550)

# with col_legend:
#     components.html(
#         """
#         <div style="
#             background-color: white;
#             padding: 15px;
#             border-radius: 10px;
#             border: 2px solid #ccc;
#             box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
#             font-family: Arial;
#             width: 100%;
#         ">
#             <h4>🗺️ Map Legend</h4>

#             <b>Crime Clusters</b><br>
#             <span style="color:red;">●</span> Cluster 1<br>
#             <span style="color:green;">●</span> Cluster 2<br>
#             <span style="color:purple;">●</span> Cluster 3<br>
#             <span style="color:orange;">●</span> Cluster 4<br>
#             <span style="color:darkred;">●</span> Cluster 5<br><br>

#             <b>🔵 Blue Circle</b> – Hotspot Center<br>
#         </div>
#         """,
#         height=280,
#     )

# # ---------------- KPI SECTION ----------------
# st.subheader("📊 Crime Hotspot Insights")

# summary = coords.groupby("cluster").size().reset_index(name="Crime Count")

# total_crimes = len(coords)
# active_hotspots = k
# most_dense_hotspot = summary["Crime Count"].max()

# col1, col2, col3 = st.columns(3)
# col1.metric("Total Crime Points", total_crimes)
# col2.metric("Active Hotspots", active_hotspots)
# col3.metric("Most Dense Hotspot", most_dense_hotspot)

# # ---------------- Optional Table ----------------
# st.markdown("### 📋 Detailed Hotspot Breakdown")
# show_table = st.checkbox("Show detailed table")

# if show_table:
#     summary["Hotspot"] = summary["cluster"] + 1

#     def risk_level(count):
#         if count > 500:
#             return "High"
#         elif count > 200:
#             return "Medium"
#         else:
#             return "Low"

#     summary["Risk Level"] = summary["Crime Count"].apply(risk_level)
#     summary = summary[["Hotspot", "Crime Count", "Risk Level"]]

#     st.dataframe(summary, use_container_width=True)

# st.markdown("---")
# st.caption("📌 Built using Streamlit, Folium & K-Means clustering")


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

# ---------------- GLOBAL UI STYLING ----------------
st.markdown("""
<style>
/* App background */
.stApp {
    background-color: #f5f7fa;
}

/* Main title */
.main-title {
    font-size: 36px;
    font-weight: 700;
    color: #1f2937;
}

/* Subtitle */
.sub-title {
    font-size: 16px;
    color: #4b5563;
    margin-bottom: 25px;
}

/* Section headers */
.section-header {
    font-size: 24px;
    font-weight: 600;
    color: #111827;
    margin-top: 30px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e5e7eb;
}

/* KPI cards */
.kpi-card {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 12px;
    border-left: 6px solid #2563eb;
    box-shadow: 0 4px 8px rgba(0,0,0,0.04);
}

/* Table container */
.dataframe {
    background-color: white;
    border-radius: 10px;
}

/* Checkbox & slider spacing */
.stCheckbox, .stSlider {
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown('<div class="main-title">Crime Hotspot Analysis – Chicago</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">'
    'This application performs spatial crime analysis for the city of Chicago using '
    'K-Means clustering to identify high-density crime regions and visualize geographic patterns.'
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
    2, 10, 5
)

show_heatmap = st.sidebar.checkbox(
    "Enable density heatmap"
)

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
    color = CLUSTER_COLORS[int(row["cluster"]) % len(CLUSTER_COLORS)]
    folium.CircleMarker(
        location=[row["LATITUDE"], row["LONGITUDE"]],
        radius=2,
        color=color,
        fill=True,
        fill_opacity=0.5,
    ).add_to(m)

# Heatmap
if show_heatmap:
    HeatMap(coords[["LATITUDE", "LONGITUDE"]].values.tolist()).add_to(m)

# Hotspot centers
for i, (lat, lon) in enumerate(hotspots):
    folium.CircleMarker(
        location=[lat, lon],
        radius=14,
        color="#2563eb",
        fill=True,
        fill_opacity=0.9,
        popup=f"Hotspot Region {i + 1}",
    ).add_to(m)

# ---------------- MAP + LEGEND ----------------
st.markdown('<div class="section-header">Spatial Hotspot Map</div>', unsafe_allow_html=True)

col_map, col_legend = st.columns([4, 1])

with col_map:
    folium_static(m, width=900, height=550)

with col_legend:
    components.html(
        """
        <div style="
            background-color:#ffffff;
            padding:18px;
            border-radius:12px;
            border:1px solid #e5e7eb;
            font-family:Arial;
        ">
            <h4 style="margin-top:0;color:#111827;">Map Legend</h4>

            <b>Cluster Distribution</b><br>
            <span style="color:red;">●</span> Cluster 1<br>
            <span style="color:green;">●</span> Cluster 2<br>
            <span style="color:purple;">●</span> Cluster 3<br>
            <span style="color:orange;">●</span> Cluster 4<br>
            <span style="color:darkred;">●</span> Cluster 5<br><br>

            <b>Hotspot Center</b><br>
            Blue marker indicates cluster centroid
        </div>
        """,
        height=300
    )

# ---------------- KPI SECTION ----------------
st.markdown('<div class="section-header">Crime Hotspot Summary</div>', unsafe_allow_html=True)

summary = coords.groupby("cluster").size().reset_index(name="Crime Count")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
    st.metric("Total Crime Records", len(coords))
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
    st.metric("Identified Hotspot Regions", k)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
    st.metric("Highest Cluster Density", summary["Crime Count"].max())
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- TABLE ----------------
st.markdown('<div class="section-header">Hotspot Distribution Details</div>', unsafe_allow_html=True)

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
