import geopandas as gpd

# Load the GeoJSON file
healthcare_gdf = gpd.read_file("HealthCare.geojson")

# Check the first few rows
print(healthcare_gdf.head())

# Inspect the CRS (Coordinate Reference System)
print(healthcare_gdf.crs)
