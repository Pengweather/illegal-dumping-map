import math
import pandas as pd
import plotly.graph_objects as go
import keyring
import traceback
from sodapy import Socrata

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
        except Exception as e:
            print("Unable to obtain information from the client. See traceback below.")
            print(traceback.format_exc())
            query_dataframe = None

        return query_dataframe

def web_mercator_to_wgs84(x, y):
  # Converts EPSG:3857 to EPSG:4326
  earth_radius = 6378137  # Earth's radius in meters
  lon_rad = x/earth_radius
  lat_rad = 2*math.atan(math.exp(y/earth_radius)) - math.pi/2

  return (math.degrees(lat_rad), math.degrees(lon_rad))

def run_map(query_dataframe=pd.DataFrame()):
    lat = []
    lon = []

    for x, y in zip(query_dataframe['srx'], query_dataframe['sry']):
        coord = web_mercator_to_wgs84(float(x), float(y))
        lat.append(coord[0])
        lon.append(coord[1])

    fig = go.Figure(go.Scattermap(
    lat=lat,
    lon=lon,
    mode='markers',
    marker=go.scattermap.Marker(size=2)))

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

def main():
    api_token = keyring.get_password("oak311", "api_token")
    dump = DumperData(api_token)
    query_dataframe = dump.run_query(offset=0, limit=100000, where="REQCATEGORY='ILLDUMP' AND date_extract_y(DATETIMEINIT)=2024", order="DATETIMEINIT DESC")
    print(len(query_dataframe))
    run_map(query_dataframe)
    
    dump.close()

if __name__ == "__main__":
    main()