import os
import sys
import json
import requests
from dotenv import load_dotenv

load_dotenv()

req = requests.get(
    "https://api.le-systeme-solaire.net/rest/bodies?data=id,name,semimajorAxis,eccentricity,perihelion,aphelion,parentBody",
    headers={"Authorization": f'Bearer {os.getenv("L_OPEN_DATA")}'},
)

data = req.json()

for i in data["bodies"]:
    i["apogee"] = i["semimajorAxis"] * (1 + i["eccentricity"])
    i["perigee"] = i["semimajorAxis"] * (1 - i["eccentricity"])

with open("data/data.json", "w") as f:
    json.dump(data, f)
