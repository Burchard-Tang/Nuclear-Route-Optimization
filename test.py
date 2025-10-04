"""import requests
import pandas as pd

ON511_urls = {
    "events": "https://511on.ca/api/v2/get/event",
    "construction": "https://511on.ca/api/v2/get/constructionprojects",
    "road_conditions": "https://511on.ca/api/v2/get/roadconditions",
    "rest_areas": "https://511on.ca/api/v2/get/allrestareas"
}

params = {"format": "json", "lang": "en"}

responses = {}
datas = {}

for key, url in ON511_urls.items():
    try:
        print(f"🔹 Fetching {key} ...")
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()  # raise an error if status code != 200

        # Only parse JSON if response is not empty
        if r.text.strip():
            datas[key] = r.json()
        else:
            print(f"⚠️ Empty response for {key}")
            datas[key] = []

        responses[key] = r
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed for {key}: {e}")
        datas[key] = []
    except ValueError:
        print(f"⚠️ Failed to parse JSON for {key}, got:\n{r.text[:200]}...")
        datas[key] = []

# Convert valid lists to DataFrames
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

# Optional: simplify the events DataFrame
if "events" in dfs and not dfs["events"].empty:
    cols = [
        "ID", "Organization", "RoadwayName", "DirectionOfTravel",
        "Description", "Latitude", "Longitude", "EventType", "Severity"
    ]
    dfs["events"] = dfs["events"][[c for c in cols if c in dfs["events"].columns]]

# Show summary
for name, df in dfs.items():
    print(f"✅ {name}: {len(df)} rows, {len(df.columns)} columns")

"""

import networkx as nx
import geopandas as gpd
from shapely.geometry import Point, LineString

# Load GraphML
G = nx.read_graphml("ON_MB_Road_Data.graphml")

# Extract node coordinates
nodes = []
for node, data in G.nodes(data=True):
    if "x" in data and "y" in data:
        nodes.append({
            "id": node,
            "geometry": Point(float(data["x"]), float(data["y"]))
        })

nodes_gdf = gpd.GeoDataFrame(nodes, geometry="geometry", crs="EPSG:4326")

# Extract edges
edges = []
for u, v, data in G.edges(data=True):
    if all(k in G.nodes[u] for k in ("x", "y")) and all(k in G.nodes[v] for k in ("x", "y")):
        p1 = (float(G.nodes[u]["x"]), float(G.nodes[u]["y"]))
        p2 = (float(G.nodes[v]["x"]), float(G.nodes[v]["y"]))
        edges.append({
            "source": u,
            "target": v,
            "geometry": LineString([p1, p2])
        })

edges_gdf = gpd.GeoDataFrame(edges, geometry="geometry", crs="EPSG:4326")

# Save for QGIS
nodes_gdf.to_file("nodes.geojson", driver="GeoJSON")
edges_gdf.to_file("edges.geojson", driver="GeoJSON")