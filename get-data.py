import datetime
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import keyring
import traceback
from sodapy import Socrata

import pdb; pdb.set_trace()

# TODO: After everything looks good, add typing hints to the code.

class DumperData:
    def __init__(self, api_token=""):
        self.oak311_api_token = ""
        self.open()

    def open(self):
        self.oak311_client = Socrata("data.oaklandca.gov", self.oak311_api_token)

    def close(self):
        self.oak311_client.close()
        self.oak311_client = None
    
    def run_query(self, offset=0, limit=10, where="", order=""):
        try:
            results = self.oak311_client.get("quth-gb8e", offset=offset, limit=limit, where=where, order=order)
            query_dataframe = pd.DataFrame.from_records(results)
            query_dataframe.insert(0, "show_on_map", [True]*len(query_dataframe))
        except Exception as e:
            print("Unable to obtain information from the client. See traceback below.")
            print(traceback.format_exc())
            return None

        query_dataframe.insert(0, "lat", [0.0]*len(query_dataframe))
        query_dataframe.insert(0, "lon", [0.0]*len(query_dataframe))
        row = 0

        for i, (x, y) in enumerate(zip(query_dataframe['srx'], query_dataframe['sry'])):
            if x and y:
                req_coord = web_mercator_to_wgs84(float(x), float(y))
                query_dataframe.loc[i, "lat"] = req_coord[0]
                query_dataframe.loc[i, "lon"] = req_coord[1]

        return query_dataframe

# Cannot use the typical 2D distance equation due to the curvature of the Earth.

def dist_between_latlon(coord1=(), coord2=()):
    if len(coord1) != len(coord2):
        raise Exception("Coord1 and Coord2 are not the same size.")

    if len(coord1) != 2:
        raise Exception("Coord1 does not have ONLY 2 elements.")

    if len(coord2) != 2:
        raise Exception("Coord2 does not have ONLY 2 elements.")

    earth_radius = 6378137  # Earth's radius in meters

    # Convert degrees to radians
    lat1_rad = math.radians(coord1[0])
    lon1_rad = math.radians(coord1[1])
    lat2_rad = math.radians(coord2[0])
    lon2_rad = math.radians(coord2[1])

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = (math.sin(dlat / 2) ** 2 +
         math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2)
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1 - a))

    dist = earth_radius*c/1000
    return dist

def web_mercator_to_wgs84(x=0, y=0):
    # Converts EPSG:3857 to EPSG:4326
    earth_radius = 6378137  # Earth's radius in meters
    lon_rad = x/earth_radius
    lat_rad = 2*math.atan(math.exp(y/earth_radius)) - math.pi/2

    return (math.degrees(lat_rad), math.degrees(lon_rad))

def get_requests_within_latlon_radius(query_dataframe=pd.DataFrame(), center_coord=(), center_radius=0):
    if len(center_coord) != 2:
        raise Exception("Centercoord does not have ONLY 2 elements.")
    
    for i, row in query_dataframe.iterrows():
        req_coord = (row['lat'], row['lon'])
        
        if row['show_on_map'] and \
           dist_between_latlon(req_coord, center_coord) > center_radius:
            query_dataframe.loc[i, "show_on_map"] = False

    return

def run_map(query_dataframe=pd.DataFrame()):
    lat = []
    lon = []

    for i, row in query_dataframe.iterrows():
        if row['show_on_map']:
            lat.append(row['lat'])
            lon.append(row['lon'])

    fig = go.Figure(go.Scattermap(
    lat=lat,
    lon=lon,
    mode='markers',
    marker=go.scattermap.Marker(size=5)))

    fig.update_layout(
        autosize=True,
        hovermode='closest',
        map=dict(
        bearing=0,
        center=dict(
            lat=37.804747, 
            lon=-122.272
        ),
        pitch=0, zoom=10
        ),
    )

    fig.show()

# Use REGEX and add more arguments.

def run_monthly_plot(query_dataframe=pd.DataFrame()):
    months  = np.array([1, 2, 3, 4, 5, 6, 7])
    req_num = np.array([0]*7)

    for i, row in query_dataframe.iterrows():
        if row['show_on_map']:
            month = int(row['datetimeinit'][5:7])
            print(row['datetimeinit'])
            req_num[month - 1] = req_num[month - 1] + 1

    plt.xlabel('Month') 
    plt.ylabel('Number of requests') 
    plt.title('2025 Illegal Dumping Requests')
    plt.plot(months, req_num)
    plt.show()

def main():
    api_token = keyring.get_password("oak311", "api_token")
    dump = DumperData(api_token)
    query_dataframe = dump.run_query(offset=0, limit=100000, 
                                     where="REQCATEGORY='ILLDUMP' AND date_extract_y(DATETIMEINIT)=2025", 
                                     order="DATETIMEINIT DESC")                              
    #get_requests_within_latlon_radius(query_dataframe,(37.8248742, -122.2783469), 50)
    run_map(query_dataframe)
    run_monthly_plot(query_dataframe)
    
    dump.close()

if __name__ == "__main__":
    main()