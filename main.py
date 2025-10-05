import osmnx as ox
import networkx as nx
import geopandas as gpd
import folium
from shapely.geometry import Point
import polyline
from shapely.ops import unary_union
from math import radians, cos, sin, sqrt, atan2, log
import pandas as pd
import requests
import fiona
import pickle
import numpy as np
# --- Load site data ---
sites = gpd.read_file("map.geojson").to_crs(epsg=4326)

print ("start graph loading")

with open("ON_MB_Road_Data.pkl", "rb") as f:
    G = pickle.load(f) 

print ("Graph loaded")

nodes, data = zip(*G.nodes(data=True))
nodes_gdf = gpd.GeoDataFrame(
    data,
    index=nodes,
    geometry=[Point(d["x"], d["y"]) for d in data],
    crs="EPSG:4326"
)

# Spatial index is built automatically on demand
nodes_gdf.sindex

print ("Nodes GeoDataFrame created")

edge_data = []
for u, v, key, data in G.edges(keys=True, data=True):
    x_mid = (G.nodes[u]['x'] + G.nodes[v]['x']) / 2
    y_mid = (G.nodes[u]['y'] + G.nodes[v]['y']) / 2
    edge_data.append({
        "u": u,
        "v": v,
        "key": key,
        "length": data["length"],  # original length
        "geometry": Point(x_mid, y_mid)
    })

# Convert to GeoDataFrame
edges_gdf = gpd.GeoDataFrame(edge_data, geometry="geometry", crs="EPSG:4326")

ON511_urls = {
    "events": "https://511on.ca/api/v2/get/event",
    "construction": "https://511on.ca/api/v2/get/constructionprojects",
    "road_conditions": "https://511on.ca/api/v2/get/roadconditions",
    "rest_areas": "https://511on.ca/api/v2/get/allrestareas"
}

params = {"format": "json", "lang": "en"}
responses = {}
datas = {}

# --- Fetch data from API ---
for key, url in ON511_urls.items():
    try:
        print(f"🔹 Fetching {key} ...")
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()  # raise an error if status code != 200
        if r.text.strip():  # Only parse JSON if response is not empty
            datas[key] = r.json()
        else:
            print(f"⚠️ Empty response for {key}")
            datas[key] = []
        responses[key] = r
        print(f"✅ Fetched {key} successfully")
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed for {key}: {e}")
        datas[key] = []
    except ValueError:
        print(f"⚠️ Failed to parse JSON for {key}, got:\n{r.text[:200]}...")
        datas[key] = []

# --- Convert data to DataFrames ---
dfs = {}
for key, data in datas.items():
    if isinstance(data, list):
        dfs[key] = pd.DataFrame(data)
    elif isinstance(data, dict):
        list_key = next((k for k, v in data.items() if isinstance(v, list)), None)
        if list_key:
            dfs[key] = pd.DataFrame(data[list_key])
        else:
            print(f"⚠️ No list found in {key} response")
            dfs[key] = pd.DataFrame()
    else:
        dfs[key] = pd.DataFrame()

print ("✅ DataFrames created")

# --- Filter events ---
if "events" in dfs and not dfs["events"].empty:
    cols = ["Latitude", "Longitude", "EventType", "Severity"]
    dfs["events"] = dfs["events"][[c for c in cols if c in dfs["events"].columns]]

roadwork_df = dfs["events"][
    dfs["events"]["EventType"].str.contains("Construction|Roadwork|Closure", case=False, na=False)
].copy()

severity_df = dfs["events"][
    dfs["events"]["Severity"].str.contains("Unknown|Low|Medium|High|Critical", case=False, na=False)
].copy()

lanes_closed_df = dfs["construction"][
    dfs["construction"]["LanesAffected"].str.contains(
        "No Data|1 Right Lane(s)|1 Left Lane(s)|2 Alternating Lane(s)", case=False, na=False, regex=False
    )
].copy()

road_cond_df = dfs["road_conditions"][
    dfs["road_conditions"]["Condition"].str.contains(
        "No Report|Construction|Roadwork|Closure|Ice|Snow|Flood|Accident", case=False, na=False, regex=False
    )
].copy()

visibility_df = dfs["road_conditions"][
    dfs["road_conditions"]["Visibility"].str.contains("Good|Poor|Very Poor", case=False, na=False, regex=False)
].copy()

drifting_df = dfs["road_conditions"][
    dfs["road_conditions"]["Drifting"].str.contains("Yes|No", case=False, na=False, regex=False)
].copy()

facilities_df = dfs["rest_areas"][
    dfs["rest_areas"]["FoodServices"].str.contains("Available|Limited|Unavailable", case=False, na=False, regex=False)
].copy()

accessibility_df = dfs["rest_areas"][
    dfs["rest_areas"]["Accessible"].str.contains("Yes|No", case=False, na=False)
].copy()

# --- Event weights ---
event_type_weights = {
    "construction": 3,
    "roadwork": 2,
    "closure": 1000
}

severity_weights = {
    "unknown": 1,
    "medium": 3,
    "high": 5,
    "critical": 50,
}

const_lanes_affected_weights = {
    "2 alternating lane(s)": 6,
    "1 right lane(s)": 5,
    "1 left lane(s)": 4,
    "no data": 1,
}

road_cond_weights = {
    "no report": 1,
    "construction": 2,
    "roadwork": 4,
    "closure": 1000,
    "ice": 2,
    "snow": 2,
    "flood": 30,
    "accident": 5,
}

visibility_weights = {"good": 1, "poor": 2, "very poor": 3}
drifting_weights = {"no": 1, "yes": 10}

facilities_weights = {"available": -2, "limited": -1, "unavailable": 0}
accessibility_weight = {"yes": -1, "no": 0}

# --- Haversine function ---
def haversine(n1, n2, G):
    lon1, lat1 = G.nodes[n1]['x'], G.nodes[n1]['y']
    lon2, lat2 = G.nodes[n2]['x'], G.nodes[n2]['y']
    R = 6371  # Earth radius in km
    dlon, dlat = radians(lon2 - lon1), radians(lat2 - lat1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))

# --- Define key nodes ---
bruce = ox.distance.nearest_nodes(
    G,
    sites[sites["Name"] == "Bruce Power"].geometry.iloc[0].x,
    sites[sites["Name"] == "Bruce Power"].geometry.iloc[0].y
)
pickering = ox.distance.nearest_nodes(
    G,
    sites[sites["Name"] == "Pickering"].geometry.iloc[0].x,
    sites[sites["Name"] == "Pickering"].geometry.iloc[0].y
)
darlington = ox.distance.nearest_nodes(
    G,
    sites[sites["Name"] == "Darlington"].geometry.iloc[0].x,
    sites[sites["Name"] == "Darlington"].geometry.iloc[0].y
)
chalk = ox.distance.nearest_nodes(
    G,
    sites[sites["Name"] == "Chalk River"].geometry.iloc[0].x,
    sites[sites["Name"] == "Chalk River"].geometry.iloc[0].y
)
pinawa = ox.distance.nearest_nodes(
    G,
    sites[sites["Name"] == "CNL Pinawa"].geometry.iloc[0].x,
    sites[sites["Name"] == "CNL Pinawa"].geometry.iloc[0].y
)
ignace = ox.distance.nearest_nodes(
    G,
    sites[sites["Name"] == "Ignace"].geometry.iloc[0].x,
    sites[sites["Name"] == "Ignace"].geometry.iloc[0].y
)

start = bruce
end = ignace

# --- Initialize map ---
m = folium.Map(location=[sites.geometry.y.mean(), sites.geometry.x.mean()], zoom_start=5)

def add_path(color, tooltip):
    route = nx.astar_path(
        G, start, end,
        heuristic=lambda a, b: haversine(a, b, G),
        weight="length"
    )
    route_coords = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in route]
    folium.PolyLine(route_coords, color=color, weight=5, tooltip=tooltip).add_to(m)
    print(f"✅ ✅ ✅ ✅ ✅ ✅ ✅ Added new Graph")

add_path("yellow", "Pre-Adjustment Path")

# --- Load hazards data ---
hazards_df = pd.read_csv("Hazards.csv")

# Define hazard impact: (range in degrees, length multiplier, color)
hazard_info = {
    "fire": [0.1, 10000000, "red", "fire"],
    "flood": [0.05, 10000000, "blue", "water"]
}

projected_crs = "EPSG:32617"

def map_to_nearest_nodes(
    df,
    nodes_gdf,
    lon_col="Longitude",
    lat_col="Latitude",
    polyline_col="EncodedPolyline",
    value_cols=None,          # columns from df you want to append
    agg="max"                 # how to aggregate multiple values per node
):
    """
    Maps features to nearest nodes and appends aggregated values to nodes_gdf.
    """

    # --- 1️⃣ Create geometry for df ---
    if lon_col in df.columns and lat_col in df.columns:
        df_geom = gpd.GeoDataFrame(
            df.copy(),
            geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
            crs=projected_crs
        )
    elif polyline_col in df.columns:
        def get_midpoint(encoded_poly):
            points = polyline.decode(encoded_poly)
            if not points:
                return None
            mid_idx = len(points) // 2
            lat, lon = points[mid_idx]
            return Point(lon, lat)

        df_geom = df.copy()
        df_geom['geometry'] = df_geom[polyline_col].apply(get_midpoint)
        df_geom = gpd.GeoDataFrame(df_geom, geometry='geometry', crs=projected_crs)
    else:
        raise ValueError(f"No valid geometry found. Provide {lon_col}/{lat_col} or {polyline_col}.")

    # --- 2️⃣ Spatial join (nearest node) ---
    nearest = gpd.sjoin_nearest(df_geom, nodes_gdf, how="left", distance_col="dist_to_node")

    # --- 3️⃣ Aggregate values by node ---
    if value_cols is not None:
        grouped = nearest.groupby('index_right')[value_cols].agg(agg)
        grouped.index.name = nodes_gdf.index.name  # align indices if needed

        # --- 4️⃣ Join back to nodes_gdf ---
        nodes_gdf = nodes_gdf.join(grouped, how="left")

    return nodes_gdf

population_df = pd.read_csv("Populations.csv")
gas_stations_df = pd.read_csv("Refuelling_Truck_Stops.csv")
healthcare_gdf = gpd.read_file("HealthCare.geojson")

# --- Map Data to Graph Nodes ---

nodes_gdf_proj = nodes_gdf.to_crs(projected_crs)
nodes_gdf_full = nodes_gdf_proj.copy()

healthcare_gdf = healthcare_gdf.to_crs(projected_crs)
healthcare_gdf.sindex

# Map and append each dataset with unique column names
nodes_gdf_full = nodes_gdf_full.join(
    map_to_nearest_nodes(roadwork_df, nodes_gdf_proj, value_cols=["EventType"], agg="first")
    .rename(columns={"EventType": "roadwork_EventType"})[["roadwork_EventType"]]
)

nodes_gdf_full = nodes_gdf_full.join(
    map_to_nearest_nodes(severity_df, nodes_gdf_proj, value_cols=["Severity"], agg="first")
    .rename(columns={"Severity": "roadwork_Severity"})[["roadwork_Severity"]]
)

nodes_gdf_full = nodes_gdf_full.join(
    map_to_nearest_nodes(lanes_closed_df, nodes_gdf_proj, value_cols=["LanesAffected"], agg="first")
    .rename(columns={"LanesAffected": "lanes_LanesAffected"})[["lanes_LanesAffected"]]
)

nodes_gdf_full = nodes_gdf_full.join(
    map_to_nearest_nodes(road_cond_df, nodes_gdf_proj, value_cols=["Condition"], agg="first")
    .rename(columns={"Condition": "road_Condition"})[["road_Condition"]]
)

nodes_gdf_full = nodes_gdf_full.join(
    map_to_nearest_nodes(visibility_df, nodes_gdf_proj, value_cols=["Visibility"], agg="first")
    .rename(columns={"Visibility": "road_Visibility"})[["road_Visibility"]]
)

nodes_gdf_full = nodes_gdf_full.join(
    map_to_nearest_nodes(drifting_df, nodes_gdf_proj, value_cols=["Drifting"], agg="first")
    .rename(columns={"Drifting": "road_Drifting"})[["road_Drifting"]]
)

nodes_gdf_full = nodes_gdf_full.join(
    map_to_nearest_nodes(facilities_df, nodes_gdf_proj, value_cols=["FoodServices"], agg="first")
    .rename(columns={"FoodServices": "facilities_FoodServices"})[["facilities_FoodServices"]]
)

nodes_gdf_full = nodes_gdf_full.join(
    map_to_nearest_nodes(accessibility_df, nodes_gdf_proj, value_cols=["Accessible"], agg="first")
    .rename(columns={"Accessible": "accessibility_Accessible"})[["accessibility_Accessible"]]
)

nodes_gdf_full = nodes_gdf_full.join(
    map_to_nearest_nodes(gas_stations_df, nodes_gdf_proj, value_cols=["Truck stop"], agg="first")
    [["Truck stop"]]
)

print(nodes_gdf_full.columns)


print("✅ Mapped data to nearest graph nodes")
# --- 1️⃣ Prepare hazards GeoDataFrame ---
hazards_gdf = gpd.GeoDataFrame(
    hazards_df.copy(),
    geometry=gpd.points_from_xy(hazards_df.Longitude, hazards_df.Latitude),
    crs="EPSG:4326"
).to_crs(projected_crs)

# Apply buffer in meters (hazard influence radius)

hazards_gdf["geometry"] = hazards_gdf.geometry.buffer(hazards_gdf["Radius"])

# --- 2️⃣ Reproject edges to projected CRS ---
edges_gdf_proj = edges_gdf.to_crs(projected_crs)
edges_gdf_proj.sindex

# --- 3️⃣ Spatial join edges within hazard buffers ---
edges_hazards_join = gpd.sjoin(edges_gdf_proj, hazards_gdf, how="left", predicate="intersects")

print("✅ Spatial join of edges and hazards done")

# Assign hazard multiplier per edge
edges_gdf_proj["hazard_multiplier"] = edges_hazards_join.groupby(edges_hazards_join.index)["Hazards_Type"] \
    .apply(lambda x: np.prod([hazard_info[t.lower()][1] for t in x if pd.notna(t)]))
edges_gdf_proj["hazard_multiplier"] = edges_gdf_proj["hazard_multiplier"].fillna(1.0)

# --- 4️⃣ Prepare population GeoDataFrame ---
pop_gdf = gpd.GeoDataFrame(
    population_df.copy(),
    geometry=gpd.points_from_xy(population_df.Longitude, population_df.Latitude),
    crs="EPSG:4326"
).to_crs(projected_crs)

# Create buffer (population influence radius)
buffer_m = 10000
pop_gdf["geometry"] = pop_gdf.geometry.buffer(buffer_m)

# --- 5️⃣ Spatial join edges within population buffers ---
edges_pop_join = gpd.sjoin(edges_gdf_proj, pop_gdf, how="left", predicate="intersects")

print("✅ Spatial join of edges and population done")

edges_health_join = gpd.sjoin(edges_gdf_proj, healthcare_gdf, how="left", predicate="intersects")

print("✅ Spatial join of edges and healthcare done")

# Vectorized computation of population multiplier 
def compute_pop_multiplier_joined(edges_pop_join, buffer_m):
    epsilon = 1e-9
    safe_buffer = max(buffer_m, epsilon)

    # Parse population safely
    edges_pop_join["pop_value"] = (
        edges_pop_join["Population"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .astype(float)
        .fillna(0)
    )

    # Distance from edge centroid to population center (precomputed by sjoin)
    if "dist" not in edges_pop_join.columns:
        # If sjoin didn't compute distances, approximate by centroid distance
        edges_pop_join["dist"] = edges_pop_join.geometry.distance(
            edges_pop_join["geometry_right"].centroid
            if "geometry_right" in edges_pop_join
            else edges_pop_join["geometry"].centroid
        )

    # Influence decay based on distance
    edges_pop_join["influence"] = np.exp(-edges_pop_join["dist"] / safe_buffer)

    # Weighted influence by populatio
    edges_pop_join["pop_effect"] = edges_pop_join["pop_value"]* edges_pop_join["influence"]

    # Combine all influences per edge
    pop_mult = (
        1
        + edges_pop_join.groupby(edges_pop_join.index)["pop_effect"].sum()
    )

    return pop_mult.reindex(edges_pop_join.index.unique(), fill_value=1.0)

def compute_healthcare_multiplier_joined(edges_health_join, buffer_m):
    epsilon = 1e-9
    safe_buffer = max(buffer_m, epsilon)

    # Distance from edge to healthcare
    if "dist" not in edges_health_join.columns:
        edges_health_join["dist"] = edges_health_join.geometry.distance(
            edges_health_join["geometry_right"].centroid
            if "geometry_right" in edges_health_join
            else edges_health_join["geometry"].centroid
        )

    # Influence: closer = higher effect
    edges_health_join["influence"] = np.exp(-edges_health_join["dist"] / safe_buffer)

 
    edges_health_join["effect"] = 1 - (edges_health_join["influence"] * 0.1)

    # Take minimum multiplier per edge (closest healthcare dominates)
    health_mult = edges_health_join.groupby(edges_health_join.index)["effect"].min()

    # Ensure within bounds
    health_mult = np.clip(health_mult, 0.5, 1.0)

    return health_mult.reindex(edges_health_join.index.unique(), fill_value=1.0)

edges_gdf_proj["population_multiplier"] = compute_pop_multiplier_joined(edges_pop_join, buffer_m)
print("✅ Population multipliers computed (fast vectorized)")

edges_gdf_proj["healthcare_multiplier"] = compute_healthcare_multiplier_joined(edges_health_join, buffer_m)
print("✅ Healthcare multipliers computed (fast vectorized)")


# --- 6️⃣ Combined multiplier ---
edges_gdf_proj["total_multiplier"] = edges_gdf_proj["hazard_multiplier"] * edges_gdf_proj["population_multiplier"]
#edges_gdf_proj["total_multiplier"] = edges_gdf_proj["population_multiplier"]

# --- Ensure edges_gdf_proj index matches G edges ---
# edges_gdf_proj should have columns: u, v, key, total_multiplier
for idx, row in edges_gdf_proj.iterrows():
    u, v, key = row["u"], row["v"], row["key"]
    if G.has_edge(u, v, key):
        # Update edge length by the combined multiplier
        G[u][v][key]["length"] *= row["total_multiplier"]

add_path("green", "Population & Hazards Path")

print("✅ Hazards and population multipliers applied to edges")

# --- 1️⃣ Compute multipliers on nodes ---
nodes_gdf_full["event_multiplier"] = (
    nodes_gdf_full["roadwork_EventType"].str.lower().map(event_type_weights).fillna(1)
)
nodes_gdf_full["severity_multiplier"] = (
    nodes_gdf_full["roadwork_Severity"].str.lower().map(severity_weights).fillna(1)
)
nodes_gdf_full["roadwork_total_multiplier"] = (
    nodes_gdf_full["event_multiplier"] * nodes_gdf_full["severity_multiplier"]
)

nodes_gdf_full["lane_multiplier"] = (
    nodes_gdf_full["lanes_LanesAffected"].str.lower().map(const_lanes_affected_weights).fillna(1)
)

nodes_gdf_full["road_cond_multiplier"] = (
    nodes_gdf_full["road_Condition"].str.lower().map(road_cond_weights).fillna(1)
)
nodes_gdf_full["visibility_multiplier"] = (
    nodes_gdf_full["road_Visibility"].str.lower().map(visibility_weights).fillna(1)
)
nodes_gdf_full["drifting_multiplier"] = (
    nodes_gdf_full["road_Drifting"].str.lower().map(drifting_weights).fillna(1)
)
nodes_gdf_full["road_total_multiplier"] = (
    nodes_gdf_full["road_cond_multiplier"] *
    nodes_gdf_full["visibility_multiplier"] *
    nodes_gdf_full["drifting_multiplier"]
)

# --- 2️⃣ Combine all relevant multipliers for each node ---
# Here you can multiply event, lane, and road multipliers together
nodes_gdf_full["node_multiplier"] = (
    nodes_gdf_full["roadwork_total_multiplier"] *
    nodes_gdf_full["lane_multiplier"] *
    nodes_gdf_full["road_total_multiplier"]
)



# --- 3️⃣ Apply multipliers to graph edges ---
for u, v, key, data in G.edges(keys=True, data=True):
    # Get node multiplier for each endpoint safely (default=1 if missing)
    u_mult = nodes_gdf_full.loc[u, "node_multiplier"] if u in nodes_gdf_full.index else 1
    v_mult = nodes_gdf_full.loc[v, "node_multiplier"] if v in nodes_gdf_full.index else 1

    # Decide final multiplier for edge (e.g., max of endpoints)
    edge_multiplier = max(u_mult, v_mult)

    # Update edge length
    data["length"] *= edge_multiplier


print("✅ Adjusted edge lengths for hazards and events")

max_distance_m = 1_200_000  # 1200 km
reachable_stations = []

print("starting route calculation")

# --- Calculate routes ---
add_path("blue", "ON 511 Adjusted Path")

print("Route calculated")

def assign_nearest_node(df, nodes_gdf, lat_col="Latitude", lon_col="Longitude", node_col="graph_node"):
    """
    Assign nearest node from nodes_gdf to each row in df using spatial indexing.
    Args:
        df: pandas DataFrame with lat/lon columns
        nodes_gdf: GeoDataFrame of nodes with geometry column
        lat_col, lon_col: column names for coordinates
        node_col: output column for nearest node
        projected_crs: CRS in meters for accurate distances
    Returns:
        df_copy with nearest node index in node_col
    """
    df_copy = df.copy()
    
    # Convert df to GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df_copy,
        geometry=gpd.points_from_xy(df_copy[lon_col], df_copy[lat_col]),
        crs="EPSG:4326"
    )
    
    # Reproject both to projected CRS
    gdf = gdf.to_crs(projected_crs)
    nodes_proj = nodes_gdf.to_crs(projected_crs)
    
    # Ensure spatial index exists
    _ = nodes_proj.sindex
    
    # Nearest-neighbor join
    nearest = gpd.sjoin_nearest(gdf, nodes_proj, how="left", distance_col="dist_to_node")
    
    # Save nearest node index back to original df
    df_copy[node_col] = nearest["index_right"].values
    
    return df_copy

gas_stations_df = assign_nearest_node(gas_stations_df, nodes_gdf, lat_col="Latitude", lon_col="Longitude")

# --- Determine reachable gas stations ---
if nx.shortest_path_length(G, start, end, weight="length") >= max_distance_m:
    for _, row in gas_stations_df.iterrows():
        station_node = row["graph_node"]
        try:
            dist_to_station = nx.shortest_path_length(G, start, station_node, weight="length")
            if dist_to_station <= max_distance_m:
                reachable_stations.append((row["Truck stop"], station_node, dist_to_station))
        except nx.NetworkXNoPath:
            continue
else:
    print("Starting Point - Destination within 1200 km, no refuelling necessary.")

add_path("red", "Full Path with Refuelling")

# --- Add population markers ---
for _, row in population_df.iterrows():
    pop = int(row["Population"].replace(",", ""))

    # Assign color based on population thresholds
    if pop < 50_000:
        color = "white"      
        icon_color = "black" 
    elif pop < 200_000:
        color = "pink"
        icon_color = "white"
    elif pop < 500_000:
        color = "purple"
        icon_color = "white"
    else:
        color = "darkpurple"       
        icon_color = "white"

    folium.Marker(
        location=[row["Latitude"], row["Longitude"]],
        popup=f"{row['Name']}: {pop:,} people",
        icon=folium.Icon(color=color, icon="person", prefix="fa", icon_color=icon_color)
    ).add_to(m)

# --- Add hazard markers ---
for _, row in hazards_df.iterrows():
    folium.Marker(
        location=[row["Latitude"], row["Longitude"]], 
        popup=f"Hazard {row['Hazards_Type']}",
        icon=folium.Icon(
            color=hazard_info[row["Hazards_Type"].lower()][2],
            icon=hazard_info[row["Hazards_Type"].lower()][3],
            prefix="fa"
        )
    ).add_to(m)
 
    folium.Circle(
        location=[row["Latitude"], row["Longitude"]],
        radius=row["Radius"],  # radius in meters
        color=hazard_info[row["Hazards_Type"].lower()][2],
        fill=True,
        fill_opacity=0.3,
    ).add_to(m)

for _, row in gas_stations_df.iterrows():
    folium.Marker(
        location=[row["Latitude"], row["Longitude"]],
        popup=f"⛽ Chosen Gas Station: {row['Truck stop']}",
        icon=folium.Icon(color="green", icon="gas-pump", prefix="fa")
    ).add_to(m)

    healthcare_gdf = healthcare_gdf.to_crs(epsg=4326)  # Reproject back to WGS84 for folium
"""
for _, row in healthcare_gdf.iterrows():
    folium.Marker(
        location=[row.geometry.y, row.geometry.x],
        popup=f"🏥 Healthcare Facility: {row.get('Name')}",
        icon=folium.Icon(color="red", icon="hospital", prefix="fa")
    ).add_to(m)
"""
    
# --- Add site markers ---
for name in sites["Name"]:
    row = sites[sites["Name"] == name]
    folium.Marker(
        location=[row.geometry.y.values[0], row.geometry.x.values[0]],
        popup=name,
        icon=folium.Icon(color="purple" if name == "Ignace" else "blue")
    ).add_to(m)

print("Map generation starting.")

# --- Save map ---
m.save("Optimized_Nuclear_Route.html")
print("Map saved as Optimized_Nuclear_Route.html")

print("Map generation finished.")

