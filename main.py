import osmnx as ox      
import networkx as nx 
import geopandas as gpd
import folium       
from shapely.geometry import Point      
from math import radians, cos, sin, sqrt, atan2     
from shapely.ops import unary_union     
import pandas as pd    
import requests 

sites = gpd.read_file("map.geojson").to_crs(epsg=4326)

G = ox.load_graphml("ON_MB_Road_Data.graphml")

# API endpoint and parameters
trafev_url = "https://511on.ca/api/v2/get/event"
trafev_params = {
    "format": "json",
    "lang": "en"
}

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

#access by dfs["events"], dfs["construction"], etc.
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

if "events" in dfs and not dfs["events"].empty:
    cols = [
        "Latitude", "Longitude", "EventType", "Severity"
    ]
    dfs["events"] = dfs["events"][[c for c in cols if c in dfs["events"].columns]]
    

# Filter for active roadwork or lane closures
roadwork_df = dfs["events"][dfs["events"]["EventType"].str.contains("Construction|Roadwork|Closure", case=False, na=False)].copy()
severity_df = dfs["events"][dfs["events"]["Severity"].str.contains("Unknown|Low|Medium|High|Critical", case=False, na=False)].copy()
lanes_closed_df = dfs["construction"][dfs["construction"]["LanesAffected"].str.contains("No Data|1 Right Lane(s)|1 Left Lane(s)|2 Alternating Lane(s)", case=False, na=False)].copy()
road_cond_df = dfs["road_conditions"][dfs["road_conditions"]["Condition"].str.contains("No Report|Construction|Roadwork|Closure|Ice|Snow|Flood|Accident", case=False, na=False)].copy()
visibility_df = dfs["road_conditions"][dfs["road_conditions"]["Visibility"].str.contains("Good|Poor|Very Poor", case=False, na=False)].copy()
drifting_df = dfs["road_conditions"][dfs["road_conditions"]["Drifting"].str.contains("Yes|No", case=False, na=False)].copy()
facilities_df = dfs["rest_areas"][dfs["rest_areas"]["Facilities"].str.contains("Available|Limited|Unavailable", case=False, na=False)].copy()
accessibility_df = dfs["rest_areas"][dfs["rest_areas"]["Accessibility"].str.contains("Yes|No", case=False, na=False)].copy()

roadwork_nodes = [
    ox.distance.nearest_nodes(G, row["Longitude"], row["Latitude"])
    for _, row in roadwork_df.iterrows()
]
roadwork_df.loc[:, "graph_node"] = roadwork_nodes


#events param:
event_type_weights = {
    "construction": 3,   # 5× longer = avoided if possible
    "roadwork": 2,
    "closure": 1000, # effectively impassable
}

severity_weights = {
    "unknown": 1,   
    "medium": 3,
    "high": 5,
    "critical": 50, 
}

#construction param:
const_lanes_affected_weights = {
    "2 alternating lane(s)": 6, 
    "1 right lane(s)": 5,
    "1 left lane(s)": 4,
    "no data": 0,
}

#road conditions param:
road_cond_weights = {
    "no report": 0, 
    "construction": 1,
    "roadwork": 3,
    "closure": 1000,
    "ice": 1,
    "snow": 1,
    "flood": 30,
    "accident": 5,
}

visibility_weights = {
    "good": 0, 
    "poor": 1,
    "very poor": 3,
}

drifting_weights = {
    "no": 1, 
    "yes": 10,
}

#rest stops weight
facilities_weights = {
    "available": -2,
    "limited": -1,
    "unavailable": 0,
}

accessibility_weight = {
    "yes": -1,
    "no": 0,
}


def haversine(n1, n2, G):
    lon1, lat1 = G.nodes[n1]['x'], G.nodes[n1]['y']
    lon2, lat2 = G.nodes[n2]['x'], G.nodes[n2]['y']
    R = 6371  # Earth radius in km
    dlon, dlat = radians(lon2 - lon1), radians(lat2 - lat1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))


bruce = ox.distance.nearest_nodes(G, sites[sites["Name"] == "Bruce Power"].geometry.iloc[0].x,
                                     sites[sites["Name"] == "Bruce Power"].geometry.iloc[0].y)
pickering = ox.distance.nearest_nodes(G, sites[sites["Name"] == "Pickering"].geometry.iloc[0].x,
                                     sites[sites["Name"] == "Pickering"].geometry.iloc[0].y)
darlington = ox.distance.nearest_nodes(G, sites[sites["Name"] == "Darlington"].geometry.iloc[0].x,
                                     sites[sites["Name"] == "Darlington"].geometry.iloc[0].y)
chalk = ox.distance.nearest_nodes(G, sites[sites["Name"] == "Chalk River"].geometry.iloc[0].x,
                                     sites[sites["Name"] == "Chalk River"].geometry.iloc[0].y)
pinawa = ox.distance.nearest_nodes(G, sites[sites["Name"] == "CNL Pinawa"].geometry.iloc[0].x,
                                     sites[sites["Name"] == "CNL Pinawa"].geometry.iloc[0].y)
ignace = ox.distance.nearest_nodes(G, sites[sites["Name"] == "Ignace"].geometry.iloc[0].x,
                                     sites[sites["Name"] == "Ignace"].geometry.iloc[0].y)

m = folium.Map(location=[sites.geometry.y.mean(), sites.geometry.x.mean()], zoom_start=5)

population_df = pd.read_csv("Populations.csv")

population_nodes = [
    ox.distance.nearest_nodes(G, row["Longitude"], row["Latitude"])
    for _, row in population_df.iterrows()
]

population_df["graph_node"] = population_nodes

for _, row in population_df.iterrows():
    station_node = row["graph_node"]
    folium.Marker(
        location=[row["Latitude"], row["Longitude"]],
        popup=f"People",
        icon=folium.Icon(color="brown", icon="person", prefix="fa")
    ).add_to(m)

hazards_df = pd.read_csv("Hazards.csv")

# Define hazard impact: (range in degrees, length multiplier)
hazard_info = {
    "fire": [0.1,100000,"red"],
    "flood": [0.05,100000,"blue"]
}

hazard_nodes = [
    ox.distance.nearest_nodes(G, row["Longitude"], row["Latitude"])
    for _, row in hazards_df.iterrows()
]

hazards_df["graph_node"] = hazard_nodes

#goes through each node and
for u, v, key, data in G.edges(keys=True, data=True):
    x = (G.nodes[u]['x'] + G.nodes[v]['x']) / 2
    y = (G.nodes[u]['y'] + G.nodes[v]['y']) / 2
    midpoint = Point(x, y)
    for _, row in hazards_df.iterrows():
        hazard_loc = Point(float(row["Longitude"]), float(row["Latitude"]))
        hazard_range = hazard_info[row["Hazards_Type"].lower()][0]
        if hazard_loc.buffer(hazard_range).contains(midpoint):
            data["length"] *= hazard_info[row["Hazards_Type"].lower()][1]

    if data["length"] < 1000:
        for _, row in roadwork_df.iterrows():
            if row["graph_node"] in (u, v):
                for condition, multiplier in event_type_weights.items():
                    if condition.lower() in row["EventType"].lower():
                        data["length"] *= multiplier
                        data["length"] = data["length"]*multiplier


    #add another for events/etc...
for u, v, key, data in G.edges(keys=True, data=True):
    x = (G.nodes[u]['x'] + G.nodes[v]['x']) / 2
    y = (G.nodes[u]['y'] + G.nodes[v]['y']) / 2
    midpoint = Point(x, y)
    for _, row in hazards_df.iterrows():
        hazard_loc = Point(float(row["Longitude"]), float(row["Latitude"]))
        hazard_range = hazard_info[row["EventTypes"].lower()][0]
        if hazard_loc.buffer(hazard_range).contains(midpoint):
            data["length"] *= hazard_info[row["EventTypes"].lower()][1]

    if data["length"] < 1000:
        for _, row in roadwork_df.iterrows():
            if row["graph_node"] in (u, v):
                data["length"] += event_type_weights[row["EventTypes"]]+event_type_weights[row["Severity"]]

    #end of events node calcs

for _, row in hazards_df.iterrows():
    hazard_node = row["graph_node"]
    folium.Marker(
        location=[row["Latitude"], row["Longitude"]],
        popup=f"Hazard {row['Hazards_Type']}",
        icon=folium.Icon(color=hazard_info[row["Hazards_Type"].lower()][2], icon="fire", prefix="fa")
    ).add_to(m)
    


gas_stations_df = pd.read_csv("Refuelling_Truck_Stops.csv")

gas_station_nodes = [
    ox.distance.nearest_nodes(G, row["Longitude"], row["Latitude"])
    for _, row in gas_stations_df.iterrows()
]

gas_stations_df["graph_node"] = gas_station_nodes

max_distance_m = 1_200_000  # 1200 km
reachable_stations = []

for _, row in gas_stations_df.iterrows():
    station_node = row["graph_node"]
    folium.Marker(
        location=[row["Latitude"], row["Longitude"]],
        popup=f"⛽ Chosen Gas Station: {row["Truck stop"]}",
        icon=folium.Icon(color="green", icon="gas-pump", prefix="fa")
    ).add_to(m)

route = nx.astar_path(G, bruce, ignace, heuristic=lambda a, b: haversine(a,b,G), weight="length")
route_coords = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in route]
folium.PolyLine(route_coords, color="blue", weight=5, tooltip=f"Bruce to Ignace").add_to(m)

"""
route = nx.astar_path(G, pickering, ignace, heuristic=lambda a, b: haversine(a,b,G), weight="length")
route_coords = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in route]
folium.PolyLine(route_coords, color="blue", weight=5, tooltip=f"Pickering to Ignace").add_to(m)

route = nx.astar_path(G, darlington, ignace, heuristic=lambda a, b: haversine(a,b,G), weight="length")
route_coords = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in route]
folium.PolyLine(route_coords, color="blue", weight=5, tooltip=f"Darlington to Ignace").add_to(m)

route = nx.astar_path(G, chalk, ignace, heuristic=lambda a, b: haversine(a,b,G), weight="length")
route_coords = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in route]
folium.PolyLine(route_coords, color="blue", weight=5, tooltip=f"Chalk River to Ignace").add_to(m)

route = nx.astar_path(G, pinawa, ignace, heuristic=lambda a, b: haversine(a,b,G), weight="length")
route_coords = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in route]
folium.PolyLine(route_coords, color="blue", weight=5, tooltip=f"CNL Pinawa to Ignace").add_to(m)
"""

    # Groups of shortest path caoculations for each starting point#
if nx.shortest_path_length(G, bruce, ignace, weight="length") >= max_distance_m:   
    for _, row in gas_stations_df.iterrows():
        station_node = row["graph_node"]
        try:
            dist_to_station = nx.shortest_path_length(G, bruce, station_node, weight="length")
            if dist_to_station <= max_distance_m:
                reachable_stations.append((row["Truck stop"], station_node, dist_to_station))
        except nx.NetworkXNoPath:
            continue    # Skip if there's no valid path
"""
if nx.shortest_path_length(G, pickering, ignace, weight="length") >= max_distance_m:   
    for _, row in gas_stations_df.iterrows():
        station_node = row["graph_node"]
        try:
            dist_to_station = nx.shortest_path_length(G, pickering, station_node, weight="length")
            if dist_to_station <= max_distance_m:
                reachable_stations.append((row["Truck stop"], station_node, dist_to_station))
        except nx.NetworkXNoPath:
            continue    # Skip if there's no valid path

if nx.shortest_path_length(G, darlington, ignace, weight="length") >= max_distance_m:   
    for _, row in gas_stations_df.iterrows():
        station_node = row["graph_node"]
        try:
            dist_to_station = nx.shortest_path_length(G, darlington, station_node, weight="length")
            if dist_to_station <= max_distance_m:
                reachable_stations.append((row["Truck stop"], station_node, dist_to_station))
        except nx.NetworkXNoPath:
            continue    # Skip if there's no valid path

if nx.shortest_path_length(G, chalk, ignace, weight="length") >= max_distance_m:   
    for _, row in gas_stations_df.iterrows():
        station_node = row["graph_node"]
        try:
            dist_to_station = nx.shortest_path_length(G, chalk, station_node, weight="length")
            if dist_to_station <= max_distance_m:
                reachable_stations.append((row["Truck stop"], station_node, dist_to_station))
        except nx.NetworkXNoPath:
            continue    # Skip if there's no valid path

if nx.shortest_path_length(G, pinawa, ignace, weight="length") >= max_distance_m:   
    for _, row in gas_stations_df.iterrows():
        station_node = row["graph_node"]
        try:
            dist_to_station = nx.shortest_path_length(G, pinawa, station_node, weight="length")
            if dist_to_station <= max_distance_m:
                reachable_stations.append((row["Truck stop"], station_node, dist_to_station))
        except nx.NetworkXNoPath:
            continue    # Skip if there's no valid path

else:
    print(f"Starting Point - Destination within 1200 km, no refuelling necessary.")
"""

route_coords = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in route]
folium.PolyLine(route_coords, color="green", weight=5).add_to(m)

for name in sites["Name"]:
    row = sites[sites["Name"] == name]
    folium.Marker(
        location=[row.geometry.y.values[0], row.geometry.x.values[0]],
        popup=name,
        icon=folium.Icon(color="purple" if name == "Ignace" else "blue")
    ).add_to(m)

    m.save("Optimized_Nuclear_Route.html")
print("Map saved as Optimized_Nuclear_Route.html")