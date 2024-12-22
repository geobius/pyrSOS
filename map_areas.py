#!/usr/bin/python3

import folium

# Create a map centered at a specific location (e.g., latitude 0, longitude 0)
map = folium.Map(location=[38.7590150318984, 22.264349385021337], zoom_start=9)

# Add four markers at different locations
folium.Marker([38.435060171263565, 22.430173580824246], popup="Delphoi").add_to(map)
folium.Marker([39.208763998564685, 22.3983051125636], popup="Domokos").add_to(map)
folium.Marker([38.25154372303237, 22.84543182051415], popup="Prodromos").add_to(map)
folium.Marker([38.40329145041182, 23.212477490004826], popup="Yliki").add_to(map)

# Save the map to an HTML file
map.save("map_with_markers.html")
