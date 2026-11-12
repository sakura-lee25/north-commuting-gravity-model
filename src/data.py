"""Fetch and prepare inputs: Census 2021 commuting flows between Northern English local authorities.

Sources
- ONS Census 2021 ODWP01EW (LTLA), via Nomis bulk download (data/raw/ODWP01EW_LTLA.csv).
- ONS Open Geography: LAD Dec 2021 centroids (BNG) and Region Dec 2021 boundaries.
"""
from pathlib import Path
import json
import urllib.request

import numpy as np
import pandas as pd
from shapely.geometry import Point, shape

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
ARC = "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services"
NORTH = {"E12000001": "North East", "E12000002": "North West", "E12000003": "Yorkshire and The Humber"}


def _get(url: str, dest: Path) -> dict:
    if not dest.exists():
        with urllib.request.urlopen(url, timeout=60) as r:
            dest.write_bytes(r.read())
    return json.loads(dest.read_text())


def fetch_geography() -> tuple[pd.DataFrame, dict]:
    lad = _get(f"{ARC}/Local_Authority_Districts_December_2021_UK_BGC_2022/FeatureServer/0/query"
               "?where=1%3D1&outFields=LAD21CD,LAD21NM,BNG_E,BNG_N,Shape__Area&returnGeometry=false&f=json",
               RAW / "lad21_centroids.json")
    rgn = _get(f"{ARC}/Regions_December_2021_EN_BGC_2022/FeatureServer/0/query"
               "?where=1%3D1&outFields=RGN21CD,RGN21NM&outSR=27700&f=geojson",
               RAW / "rgn21_boundaries.geojson")
    lads = pd.DataFrame([f["attributes"] for f in lad["features"]])
    return lads, rgn


def north_lads() -> pd.DataFrame:
    """LADs whose centroid falls inside the North East, North West or Yorkshire and The Humber."""
    lads, rgn = fetch_geography()
    polys = {f["properties"]["RGN21CD"]: shape(f["geometry"]) for f in rgn["features"]
             if f["properties"]["RGN21CD"] in NORTH}
    def region(row):
        p = Point(row.BNG_E, row.BNG_N)
        for code, poly in polys.items():
            if poly.contains(p):
                return code
        return None
    lads["RGN21CD"] = lads.apply(region, axis=1)
    out = lads[lads.RGN21CD.notna()].sort_values("LAD21CD").reset_index(drop=True)
    return out


def od_matrix(zones: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Observed commuters (residence -> workplace), category 3 = working in the UK, not from home."""
    df = pd.read_csv(RAW / "ODWP01EW_LTLA.csv")
    df = df[df["Place of work indicator (4 categories) code"] == 3]
    codes = list(zones.LAD21CD)
    idx = {c: i for i, c in enumerate(codes)}
    df = df[df["Lower tier local authorities code"].isin(idx) & df["LTLA of workplace code"].isin(idx)]
    T = np.zeros((len(codes), len(codes)))
    for o, d, n in df[["Lower tier local authorities code", "LTLA of workplace code", "Count"]].itertuples(index=False):
        T[idx[o], idx[d]] += n
    return T, codes


def distance_matrix(zones: pd.DataFrame) -> np.ndarray:
    """Straight-line km between LAD centroids; intrazonal = mean distance within an equal-area disc."""
    xy = zones[["BNG_E", "BNG_N"]].to_numpy(float)
    D = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1)) / 1000.0
    radius_km = np.sqrt(zones.Shape__Area.to_numpy(float) / np.pi) / 1000.0
    np.fill_diagonal(D, 2.0 / 3.0 * radius_km)
    return D


def build() -> None:
    PROC.mkdir(parents=True, exist_ok=True)
    zones = north_lads()
    T, codes = od_matrix(zones)
    D = distance_matrix(zones)
    zones.to_csv(PROC / "zones.csv", index=False)
    np.save(PROC / "T_obs.npy", T)
    np.save(PROC / "D_km.npy", D)
    print(f"{len(codes)} northern LADs; {T.sum():,.0f} commuters living and working in the North")


if __name__ == "__main__":
    build()
