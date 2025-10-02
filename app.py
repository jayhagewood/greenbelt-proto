# app.py
# Streamlit prototype: Greenbelt ADOS-Proxy + Wealth/Health context
# Sources: Census ACS (B05003B etc.), CDC PLACES, PG County Socrata
# Designed for Cloud deploy with lazy imports

import os
import requests
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="ADOS / Wealth Gap Prototype", layout="wide")

# ---------------------------
# Lazy imports for geo libs
# ---------------------------
def _gpd():
    import geopandas as _g
    return _g

def _shapely():
    from shapely.geometry import Point as _Point, shape as _shape
    return _Point, _shape

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.header("Configuration")

CITY_NAME = st.sidebar.text_input("City", "Greenbelt, Maryland, USA")
STATE_FIPS = st.sidebar.text_input("State FIPS", "24")   # Maryland
COUNTY_FIPS = st.sidebar.text_input("County FIPS", "033")  # Prince George's
ACS_YEAR = st.sidebar.selectbox("ACS 5-year vintage", ["2023","2022","2021"], index=0)

BOUNDARY_SOURCE = st.sidebar.selectbox(
    "Boundary source",
    ["Census Places (TIGER)", "Auto (OSM→Census)", "OSM (Nominatim)"],
    index=0,
)

# ---------------------------
# Helpers
# ---------------------------
@st.cache_data(show_spinner=True, ttl=60*60*24)
def get_city_boundary_from_census_places(place_name: str, state_fips: str):
    """
    Fetch a place boundary directly from the Census TIGERweb ArcGIS REST API (GeoJSON),
    which avoids reading remote ZIP shapefiles (vsizip/vsicurl) that often fail on Cloud.

    Returns a GeoDataFrame with CRS EPSG:4326 containing a single row (the best match).
    """
    import json
    gpd = _gpd()

    base = "https://tigerweb.geo.census.gov/arcgis/rest/services/" \
           "TIGERweb/Places_CouSub_ConCity_SubMCD/MapServer/2/query"

    # Normalize inputs
    state = str(state_fips).zfill(2)
    name_token = place_name.split(",")[0].strip().replace("'", "''")  # escape single quotes

    # First try exact (case-insensitive) name match
    where = f"STATE='{state}' AND UPPER(NAME)=UPPER('{name_token}')"
    params = {
        "where": where,
        "outFields": "*",
        "f": "geojson",
        "outSR": 4326,
        "returnGeometry": "true",
    }
    r = requests.get(base, params=params, timeout=60)
    r.raise_for_status()
    j = r.json()

    # If nothing, try prefix LIKE
    if not j.get("features"):
        where_like = f"STATE='{state}' AND UPPER(NAME) LIKE UPPER('{name_token}%')"
        params["where"] = where_like
        r = requests.get(base, params=params, timeout=60)
        r.raise_for_status()
        j = r.json()

    if not j.get("features"):
        raise RuntimeError(f"Census TIGERweb: no place found for '{place_name}' in state {state}.")

    # Build GeoDataFrame from features
    gdf = gpd.GeoDataFrame.from_features(j["features"], crs="EPSG:4326")

    # Prefer 'city' type if multiple records (NAMELSAD often contains "city", "town", etc.)
    if "NAMELSAD" in gdf.columns:
        gdf["score"] = 0
        gdf.loc[gdf["NAMELSAD"].str.contains("city", case=False, na=False), "score"] += 2
        gdf.loc[gdf["NAME"].str.upper() == name_token.upper(), "score"] += 3
        gdf = gdf.sort_values(["score", "NAME"], ascending=[False, True])

    # Keep the best match
    gdf = gdf.head(1).copy()
    gdf["name"] = place_name
    # Fix potential topology issues
    gdf["geometry"] = gdf["geometry"].buffer(0)

    return gdf[["name", "geometry"]].reset_index(drop=True)


@st.cache_data(show_spinner=True, ttl=60*60*24)
def get_city_boundary_from_census_places(place_name: str, state_fips: str):
    gpd = _gpd()
    url = "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_place_500k.zip"
    places = gpd.read_file(url).to_crs(4326)
    matches = places[places["NAME"].str.contains(place_name.split(",")[0], case=False, na=False)]
    cand = matches[matches["STATEFP"] == state_fips]
    if cand.empty:
        raise RuntimeError("City not found in Census Places shapefile.")
    cand["name"] = cand["NAME"]
    cand["geometry"] = cand["geometry"].buffer(0)
    return cand[["name","geometry"]].reset_index(drop=True)

@st.cache_data(show_spinner=True, ttl=60*60)
def get_census_group_metadata(year:str, group:str="B05003B"):
    url = f"https://api.census.gov/data/{year}/acs/acs5/groups/{group}.json"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.json()

@st.cache_data(show_spinner=True, ttl=60*60)
def get_census_tract_black_nativity(year, state_fips, county_fips, group="B05003B"):
    meta = get_census_group_metadata(year, group)
    url = f"https://api.census.gov/data/{year}/acs/acs5"
    params = {"get": f"NAME,group({group})", "for": "tract:*", "in": f"state:{state_fips} county:{county_fips}"}
    r = requests.get(url, params=params, timeout=120)
    r.raise_for_status()
    raw = r.json()
    df = pd.DataFrame(raw[1:], columns=raw[0])
    varmeta = meta["variables"]
    native_cols = [k for k,v in varmeta.items() if "!!Native" in v.get("label","") and k.endswith("E")]
    foreign_cols = [k for k,v in varmeta.items() if "!!Foreign born" in v.get("label","") and k.endswith("E")]
    total_black_col = f"{group}_001E"
    for c in set(native_cols+foreign_cols+[total_black_col]):
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    df["native_black"] = df[native_cols].sum(axis=1, skipna=True)
    df["foreign_black"] = df[foreign_cols].sum(axis=1, skipna=True)
    df["black_total"] = pd.to_numeric(df.get(total_black_col, np.nan), errors="coerce")
    df["geoid_tract"] = df["state"]+df["county"]+df["tract"]
    return df[["NAME","geoid_tract","native_black","foreign_black","black_total"]]

@st.cache_data(show_spinner=True, ttl=60*60)
def get_pg_tract_geometries(state_fips="24", county_fips="033"):
    gpd = _gpd()
    url = "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_24_tract_500k.zip"
    gdf = gpd.read_file(url).to_crs(4326)
    gdf = gdf[gdf["COUNTYFP"] == county_fips]
    gdf["geoid_tract"] = gdf["GEOID"]
    return gdf[["geoid_tract","geometry","NAMELSAD"]].reset_index(drop=True)

def rate(numer, denom):
    numer = pd.to_numeric(numer, errors="coerce")
    denom = pd.to_numeric(denom, errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom>0, numer/denom, np.nan)

# ---------------------------
# Main app
# ---------------------------
st.title("Greenbelt, MD • Reparations Context Prototype")
st.caption("ADOS-proxy (native-born Black residents) + wealth & health context")

# ---- Boundary ----
with st.spinner("Fetching city boundary…"):
    try:
        if BOUNDARY_SOURCE == "Census Places (TIGER)":
            city_gdf = get_city_boundary_from_census_places(CITY_NAME, STATE_FIPS)
        else:
            city_gdf = get_city_boundary_from_overpass(CITY_NAME)
        st.success("Boundary loaded")
    except Exception as e:
        st.error(f"Boundary fetch failed: {e}")
        city_gdf = None

# ---- ADOS Proxy ----
if city_gdf is not None:
    colA, colB = st.columns([3,2])
    with colA:
        st.subheader("ADOS Proxy Estimate")
        acs_df = get_census_tract_black_nativity(ACS_YEAR, STATE_FIPS, COUNTY_FIPS)
        tracts_gdf = get_pg_tract_geometries()
        gpd = _gpd()
        tr_ = gpd.overlay(tracts_gdf, city_gdf[["geometry"]], how="intersection")
        tr_ids = tr_["geoid_tract"].unique().tolist()
        acs_sel = acs_df[acs_df["geoid_tract"].isin(tr_ids)].copy()

        city_native = acs_sel["native_black"].sum()
        city_foreign = acs_sel["foreign_black"].sum()
        city_black_total = acs_sel["black_total"].sum()
        ados_proxy_share = (city_native / city_black_total) if city_black_total > 0 else np.nan

        k1, k2, k3 = st.columns(3)
        k1.metric("Native-born Black (ADOS proxy)", f"{int(city_native):,}")
        k2.metric("Foreign-born Black", f"{int(city_foreign):,}")
        k3.metric("ADOS Proxy Share", f"{ados_proxy_share:.1%}" if pd.notna(ados_proxy_share) else "—")

    with colB:
        st.subheader("Notes")
        st.info("ADOS-proxy = Native-born Black (alone) from ACS table B05003B. This is a proxy, not an identity marker.")

else:
    st.warning("No boundary — cannot compute ADOS proxy")

st.markdown("---")
st.caption("Prototype — Census ACS 5-year, OSM/Census boundaries")
