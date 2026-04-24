import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from math import radians, sin, cos, sqrt, atan2
import pydeck as pdk
import matplotlib.cm as cm

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(layout="wide")
st.title("Delivery Optimization Dashboard")

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("roorkee_orders_jan12_peak_bad_weather_NO_riders.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("⏱ Controls")

window = st.sidebar.slider("Window (min)", 3, 10, 6)

start_time = st.sidebar.slider(
    "Start Time",
    min_value=df["timestamp"].min().to_pydatetime(),
    max_value=df["timestamp"].max().to_pydatetime(),
    value=df["timestamp"].min().to_pydatetime(),
    step=pd.Timedelta(minutes=1),
    format="HH:mm"
)

# -----------------------------
# FILTER DATA
# -----------------------------
end_time = start_time + pd.Timedelta(minutes=window)
batch_df = df[(df["timestamp"] >= start_time) & (df["timestamp"] < end_time)].copy()

orders = len(batch_df)

# -----------------------------
# RIDER MODEL
# -----------------------------
st.sidebar.header("🛵 Riders")

R_base = st.sidebar.slider("Base Riders", 40, 120, 60)
f_peak = st.sidebar.slider("Peak", 0.8, 1.3, 1.1)
f_weather = st.sidebar.slider("Weather", 0.5, 1.0, 0.7)
s_platform = st.sidebar.slider("Platform", 0.2, 0.6, 0.4)
u1 = st.sidebar.slider("Idle", 0.2, 0.5, 0.3)

base_riders = R_base * f_peak * f_weather * s_platform * u1
target_ratio = st.sidebar.slider("Supply Ratio", 0.5, 1.5, 0.7)

riders_available = int(0.5 * base_riders + 0.5 * (target_ratio * orders)) if orders > 0 else int(base_riders)

# -----------------------------
# STORE LOCATION
# -----------------------------
STORE_LAT = 29.8543
STORE_LON = 77.8880

# -----------------------------
# DISTANCE FUNCTION
# -----------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return 2 * R * atan2(sqrt(a), sqrt(1-a))

# -----------------------------
# CLUSTERING
# -----------------------------
if orders > 0:
    coords = batch_df[["latitude","longitude"]].values
    db = DBSCAN(eps=0.5/6371, min_samples=2, metric='haversine').fit(np.radians(coords))
    batch_df["cluster"] = db.labels_

batch_df = batch_df.dropna()

# -----------------------------
# BATCHING PARAMETERS
# -----------------------------
T_pickup = 2
T_drop = 2
speed = 20

def create_batches(cluster_df, sla):
    cluster_df = cluster_df.copy()
    cluster_df["dist"] = cluster_df.apply(
        lambda row: haversine(STORE_LAT, STORE_LON, row["latitude"], row["longitude"]), axis=1
    )
    cluster_df = cluster_df.sort_values("dist")

    batches = []
    i = 0

    while i < len(cluster_df):
        k = 0
        while i+k < len(cluster_df):
            subset = cluster_df.iloc[i:i+k+1]
            avg_dist = subset["dist"].mean()
            t = T_pickup + (avg_dist/speed)*60 + (k+1)*T_drop

            if t <= sla:
                k += 1
            else:
                break

        batch = cluster_df.iloc[i:i+max(1,k)]
        batches.append(batch)
        i += max(1,k)

    return batches

# -----------------------------
# SLA BASE
# -----------------------------
sla_batches = []
for cid in batch_df["cluster"].unique():
    cluster_df = batch_df[batch_df["cluster"] == cid]

    if cid == -1:
        for idx in cluster_df.index:
            sla_batches.append(cluster_df.loc[[idx]])
    else:
        sla_batches.extend(create_batches(cluster_df, 10))

sla_riders = len(sla_batches)

# -----------------------------
# RELAXED SLA
# -----------------------------
pressure = sla_riders / riders_available if riders_available > 0 else 10

if pressure < 0.8:
    strategy, sla = "Strict", 10
elif pressure < 1.2:
    strategy, sla = "Balanced", 11
else:
    strategy, sla = "Aggressive", 12

relaxed_batches = []
for cid in batch_df["cluster"].unique():
    cluster_df = batch_df[batch_df["cluster"] == cid]

    if cid == -1:
        for idx in cluster_df.index:
            relaxed_batches.append(cluster_df.loc[[idx]])
    else:
        relaxed_batches.extend(create_batches(cluster_df, sla))

relaxed_riders = len(relaxed_batches)

# -----------------------------
# ASSIGN BATCH COLORS (FIX)
# -----------------------------
batch_df["batch_id"] = -1

for i, batch in enumerate(relaxed_batches):
    batch_df.loc[batch.index, "batch_id"] = i

cmap = cm.get_cmap('tab20', len(relaxed_batches))

batch_df["r"] = batch_df["batch_id"].apply(lambda x: int(cmap(x % 20)[0]*255))
batch_df["g"] = batch_df["batch_id"].apply(lambda x: int(cmap(x % 20)[1]*255))
batch_df["b"] = batch_df["batch_id"].apply(lambda x: int(cmap(x % 20)[2]*255))

# -----------------------------
# KPI
# -----------------------------
c1,c2,c3,c4 = st.columns(4)

c1.metric("Orders", orders)
c2.metric("Available", riders_available)
c3.metric("SLA Riders", sla_riders)
c4.metric("Relaxed SLA Riders", relaxed_riders)

# -----------------------------
# MAP (FINAL FIXED)
# -----------------------------
st.subheader("Rider Routes")

orders_layer = pdk.Layer(
    "ScatterplotLayer",
    data=batch_df,
    get_position='[longitude, latitude]',
    get_color='[r, g, b]',
    get_radius=60,
)

route_paths = []
for i, batch in enumerate(relaxed_batches):
    path = [[STORE_LON, STORE_LAT]]
    for _, row in batch.iterrows():
        path.append([row["longitude"], row["latitude"]])

    color = cmap(i % 20)

    route_paths.append({
        "path": path,
        "color": [int(color[0]*255), int(color[1]*255), int(color[2]*255)]
    })

routes = pdk.Layer(
    "PathLayer",
    data=route_paths,
    get_path="path",
    get_color="color",
    width_scale=3,
)

store_layer = pdk.Layer(
    "ScatterplotLayer",
    data=pd.DataFrame({"lat":[STORE_LAT],"lon":[STORE_LON]}),
    get_position='[lon,lat]',
    get_color='[255,255,0]',
    get_radius=100,
)

st.pydeck_chart(pdk.Deck(
    layers=[routes, orders_layer, store_layer],
    initial_view_state=pdk.ViewState(
        latitude=batch_df["latitude"].mean() if orders>0 else STORE_LAT,
        longitude=batch_df["longitude"].mean() if orders>0 else STORE_LON,
        zoom=13
    )
))

# -----------------------------
# STATUS
# -----------------------------
st.subheader("Status")

st.write("Strategy:", strategy)
st.write("Relaxed SLA:", sla)

if relaxed_riders > riders_available:
    st.error("Rider Shortage")
else:
    st.success("System Balanced")

# -----------------------------
# CLEAN GRAPH
# -----------------------------
st.subheader("Comparison")

fig, ax = plt.subplots(figsize=(6,3))

labels = ["Available","SLA","Relaxed"]
values = [riders_available, sla_riders, relaxed_riders]

bars = ax.bar(labels, values)

for bar in bars:
    ax.text(bar.get_x()+bar.get_width()/2,
            bar.get_height(),
            int(bar.get_height()),
            ha='center')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

st.pyplot(fig)