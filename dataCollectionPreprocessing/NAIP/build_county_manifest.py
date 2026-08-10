import csv
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(".deps").resolve()))
import numpy as np

SOURCE = Path(r"C:\Users\larki\Desktop\PollenSense\training\McCoy")
BOUNDARIES = Path("county_boundaries.geojson")
OUTPUT = Path("naip_county_manifest.json")

CODES = {
    "Albuquerque": "ABQ", "Anaheim": "ANA", "Arlington": "ARL", "Atlanta": "ATL",
    "AuroraCO": "AUR", "Austin": "AUS", "Baltimore": "BAL", "Boston": "BOS",
    "Buffalo": "BUF", "CapeCoral": "CPC", "ColoradoSprings": "COS", "Columbus": "CMH",
    "Dallas": "DAL", "Denver": "DEN", "DesMoines": "DSM", "Detroit": "DET",
    "Durham": "DUR", "Fresno": "FAT", "GardenGrove": "GGV", "GrandRapids": "GRR",
    "Greensboro": "GSO", "Honolulu": "HNL", "Houston": "HOU", "HuntingtonBeach": "HNB",
    "Indianapolis": "IND", "Irvine": "IRV", "Jerseycity": "JCY", "Knoxville": "TYS",
    "LasVegas": "LAS", "LosAngeles": "LAX", "Louisville": "SDF", "Madison": "MSN",
    "Miami": "MIA", "Milwaukee": "MKE", "Minneapolis": "MSP", "Nashville": "BNA",
    "NewOrleans": "MSY", "NewYork": "NYC", "Oakland": "OAK", "OklahomaCity": "OKC",
    "Ontario": "ONT", "Orlando": "ORL", "OverlandPark": "OVP", "Phoenix": "PHX",
    "Pittsburgh": "PIT", "Plano": "PLN", "Portland": "PDX", "Providence": "PVD",
    "RanchoCucamonga": "RCM", "Richmond": "RIC", "Rochester": "ROC", "Sacramento": "SAC",
    "SanDiego": "SAN", "SanFrancisco": "SFO", "SanJose": "SJC", "SantaRosa": "STR",
    "Seattle": "SEA", "SiouxFalls": "FSD", "StLouis": "STL", "Stockton": "STK",
    "Tampa": "TPA", "WashingtonDC": "DCA", "Worcester": "ORH",
}

STATE_ABBR = {
    "01":"AL","02":"AK","04":"AZ","05":"AR","06":"CA","08":"CO","09":"CT",
    "10":"DE","11":"DC","12":"FL","13":"GA","15":"HI","16":"ID","17":"IL",
    "18":"IN","19":"IA","20":"KS","21":"KY","22":"LA","23":"ME","24":"MD",
    "25":"MA","26":"MI","27":"MN","28":"MS","29":"MO","30":"MT","31":"NE",
    "32":"NV","33":"NH","34":"NJ","35":"NM","36":"NY","37":"NC","38":"ND",
    "39":"OH","40":"OK","41":"OR","42":"PA","44":"RI","45":"SC","46":"SD",
    "47":"TN","48":"TX","49":"UT","50":"VT","51":"VA","53":"WA","54":"WV",
    "55":"WI","56":"WY",
}

def rings(geometry):
    coords = geometry["coordinates"]
    if geometry["type"] == "Polygon":
        yield coords
    elif geometry["type"] == "MultiPolygon":
        yield from coords

def point_in_ring(x, y, ring):
    inside = False
    j = len(ring) - 1
    for i, (xi, yi) in enumerate(ring):
        xj, yj = ring[j]
        if ((yi > y) != (yj > y)) and x < (xj-xi)*(y-yi)/(yj-yi) + xi:
            inside = not inside
        j = i
    return inside

def contains(geometry, x, y):
    for poly in rings(geometry):
        if point_in_ring(x, y, poly[0]) and not any(point_in_ring(x, y, hole) for hole in poly[1:]):
            return True
    return False

def points_in_ring(xs, ys, ring):
    inside = np.zeros(xs.shape, dtype=bool)
    xj, yj = ring[-1]
    for xi, yi in ring:
        crosses = (yi > ys) != (yj > ys)
        inside ^= crosses & (xs < (xj-xi) * (ys-yi) / ((yj-yi) if yj != yi else 1.0) + xi)
        xj, yj = xi, yi
    return inside

def points_in_geometry(xs, ys, geometry):
    result = np.zeros(xs.shape, dtype=bool)
    for poly in rings(geometry):
        hit = points_in_ring(xs, ys, poly[0])
        for hole in poly[1:]: hit &= ~points_in_ring(xs, ys, hole)
        result |= hit
    return result

def main():
    geo = json.loads(BOUNDARIES.read_text(encoding="utf-8"))
    counties = []
    for f in geo["features"]:
        fid = str(f.get("id") or f["properties"].get("GEO_ID", "")[-5:]).zfill(5)
        pts = [p for poly in rings(f["geometry"]) for ring in poly for p in ring]
        px, py = zip(*pts)
        counties.append((fid, f["properties"].get("NAME", fid), (min(px),min(py),max(px),max(py)), f["geometry"]))

    manifest = []
    unmatched = []
    for path in sorted(SOURCE.glob("*_Final_*.csv")):
        city = re.sub(r"_Final_.*$", "", path.stem)
        found, point_count = {}, 0
        xs, ys = [], []
        with path.open(encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                try:
                    y = float(row.get("latitude_coordinates") or row["latitude_coordinate"])
                    x = float(row.get("longitude_coordinates") or row["longitude_coordinate"])
                    if not (math.isfinite(x) and math.isfinite(y)): continue
                except (KeyError, TypeError, ValueError):
                    continue
                point_count += 1; xs.append(x); ys.append(y)
        if xs:
            xa, ya = np.asarray(xs), np.asarray(ys)
            matched = np.zeros(xa.shape, dtype=bool)
            for fid, name, (xmin,ymin,xmax,ymax), geom in counties:
                candidate = (xa >= xmin) & (xa <= xmax) & (ya >= ymin) & (ya <= ymax)
                if not candidate.any(): continue
                idx = np.flatnonzero(candidate)
                hit = points_in_geometry(xa[idx], ya[idx], geom)
                if hit.any(): found[fid] = name; matched[idx[hit]] = True
            for i in np.flatnonzero(~matched):
                unmatched.append({"city":city,"latitude":ys[i],"longitude":xs[i]})
        manifest.append({
            "city": city, "code": CODES[city], "points": point_count,
            "counties": [{"geoid": f, "state": STATE_ABBR[f[:2]], "county_fips": f[2:], "name": n}
                         for f,n in sorted(found.items())]
        })
    OUTPUT.write_text(json.dumps({"cities":manifest,"unmatched":unmatched}, indent=2), encoding="utf-8")
    print(json.dumps({"cities":len(manifest),"city_county_pairs":sum(len(x["counties"]) for x in manifest),
                      "multi_county_cities":[{"city":x["city"],"code":x["code"],"counties":x["counties"]}
                                              for x in manifest if len(x["counties"])>1],
                      "unmatched_points":len(unmatched)}, indent=2))

if __name__ == "__main__":
    main()
