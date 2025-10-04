import osmnx as ox      
import networkx as nx 
import geopandas as gpd
import folium       
from shapely.geometry import Point      
from math import radians, cos, sin, sqrt, atan2     
from shapely.ops import unary_union     
import pandas as pd     

sites = gpd.read_file("nuclear_sites.geojson").to_crs(epsg=4326)

G = ox.load_graphml("ON_MB_Road_Data.graphml")