# app.py — City ADOS & Equity Context (with BLUF panel)
# =============================================================================
# Phase 1:
#   - City boundary (OSM/Nominatim)
#   - ACS B05003B ADOS-proxy (Native-born Black alone) by tract → city
#   - Wealth/Health indicators (ACS + CDC PLACES)
#   - PG County Socrata samplers (311, Code, Crime)
#   - Optional: Maryland Socrata domain slot
# Phase 2:
#   - Update buttons to fetch/prepare HUD CHAS (place), LODES (RAC), PUMS (ADOS by PUMA)
#   - Auto-detect & display outputs when present
# Fixes:
#   - Robust PLACES loader + safe merge
#   - pydeck GeoJsonLayer mapping for polygon
#   - Arrow-friendly dataframes (drop raw geometry in tables)
#   - BLUF (30s) panel + smart fallbacks for missing/fragile metrics
# =============================================================================

import os
import io
import zipfile
import gzip
import shutil
from pathlib import Path

import requests
import numpy as np
import pandas as pd
import streamlit as st

# Geo
import geopandas as gpd
from shapely.geometry import Point, shape

# Charts / Map
import altair as alt
import pydeck as pdk

st.set_page_config(page_title="City ADOS & Equity Context — Prototype", layout="wide")

# ---------------------------
# Sidebar (BLUF-friendly labels)
# ---------------------------
st.sidebar.header("Config")

CITY_NAME   = st.sidebar.text_input("City Name", "Greenbelt, Maryland, USA")
STATE_FIPS  = st.sidebar.text_input("State Code (FIPS)", "24")       # Maryland = 24
COUNTY_FIPS = st.sidebar.text_input("County Code (FIPS)", "033")     # Prince George's = 033
ACS_YEAR    = st.sidebar.selectbox("ACS Data Year (5-yr)", ["2023", "2022", "2021"], index=0)

st.sidebar.markdown("---")
st.sidebar.markdown("### Socrata Open Data")
SOCRATA_DOMAIN_PG = st.sidebar.text_input("PG County Socrata domain", "data.princegeorgescountymd.gov")
PG_DATASETS = {
    "311 Service Calls":        st.sidebar.text_input("311 dataset id", "8nyi-qgn7"),
    "Code Enforcement":         st.sidebar.text_input("Code/Inspections dataset id", "9hyf-46qb"),
    "Crime Reports – Current":  st.sidebar.text_input("Crime current id", "xjru-idbe"),
    "Crime Reports – Past":     st.sidebar.text_input("Crime historic id", "wb4e-w4nf"),
}

# Optional second Socrata domain (e.g., Maryland Open Data)
SOCRATA_DOMAIN_MD_OPT = st.sidebar.text_input("Optional Socrata domain (e.g., opendata.maryland.gov)", "")

MAX_ROWS = st.sidebar.slider("Max Rows per Dataset", 10_000, 100_000, 50_000, step=10_000)

st.sidebar.markdown("---")
st.sidebar.caption("Tip: Works for any U.S. city that OSM has a polygon for.")

# ---------------------------
# Helpers (cached)
# ---------------------------

@st.cache_data(show_spinner=True, ttl=60*60*24)
def get_city_boundary_from_overpass(place_name: str) -> gpd.GeoDataFrame:
    nom = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params={"q": place_name, "format": "json", "polygon_geojson": 1, "limit": 1, "addressdetails": 0},
        headers={"User-Agent": "city-boundary-prototype/1.0"},
        timeout=60,
    )
    nom.raise_for_status()
    j = nom.json()
    if not j:
        raise RuntimeError(f"Boundary not found for '{place_name}'. Try a different spelling.")
    gj = j[0].get("geojson")
    gdf = gpd.GeoDataFrame({"name": [place_name]}, geometry=[shape(gj)], crs="EPSG:4326")
    gdf["geometry"] = gdf["geometry"].buffer(0)  # fix invalid multipolygons if any
    return gdf

@st.cache_data(show_spinner=True, ttl=60*60*24)
def get_census_group_metadata(year: str, group: str = "B05003B"):
    url = f"https://api.census.gov/data/{year}/acs/acs5/groups/{group}.json"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.json()

@st.cache_data(show_spinner=True, ttl=60*60*24)
def get_census_tract_black_nativity(year: str, state_fips: str, county_fips: str, group: str = "B05003B"):
    meta = get_census_group_metadata(year, group)
    url = f"https://api.census.gov/data/{year}/acs/acs5"
    params = {"get": f"NAME,group({group})", "for": "tract:*", "in": f"state:{state_fips} county:{county_fips}"}
    r = requests.get(url, params=params, timeout=120)
    r.raise_for_status()
    raw = r.json()
    df = pd.DataFrame(raw[1:], columns=raw[0])

    varmeta = meta["variables"]
    native_cols = [k for k, v in varmeta.items()
                   if k.startswith(group + "_") and "!!Native" in v.get("label", "") and k.endswith("E")]
    foreign_cols = [k for k, v in varmeta.items()
                    if k.startswith(group + "_") and "!!Foreign born" in v.get("label", "") and k.endswith("E")]
    total_black_col = f"{group}_001E"

    for c in set(native_cols + foreign_cols + [total_black_col]) & set(df.columns):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["native_black"]   = df[native_cols].sum(axis=1, skipna=True)
    df["foreign_black"]  = df[foreign_cols].sum(axis=1, skipna=True)
    df["black_total"]    = pd.to_numeric(df.get(total_black_col, np.nan), errors="coerce")
    df["geoid_tract"]    = df["state"] + df["county"] + df["tract"]

    keep = df[["NAME", "geoid_tract", "native_black", "foreign_black", "black_total"]].copy()
    return keep

@st.cache_data(show_spinner=True, ttl=60*60*24)
def get_pg_tract_geometries(state_fips="24", county_fips="033"):
    url = "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_24_tract_500k.zip"
    gdf = gpd.read_file(url)
    gdf = gdf.to_crs("EPSG:4326")
    gdf = gdf[gdf["COUNTYFP"] == county_fips].copy()
    gdf["geoid_tract"] = gdf["GEOID"]
    return gdf[["geoid_tract", "geometry", "NAMELSAD"]].reset_index(drop=True)

def _socrata_url(domain, dataset, limit):
    return f"https://{domain}/resource/{dataset}.json?$limit={limit}"

@st.cache_data(show_spinner=True, ttl=30*60)
def fetch_socrata(domain: str, dataset_id: str, limit: int = 50000):
    if not domain or not dataset_id:
        return pd.DataFrame()
    url = _socrata_url(domain, dataset_id, limit)
    try:
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        df = pd.DataFrame(r.json())
        return df
    except Exception as e:
        st.warning(f"Socrata fetch failed for {domain}/{dataset_id}: {e}")
        return pd.DataFrame()

def safe_pointify(df, lon_col="longitude", lat_col="latitude"):
    if lon_col not in df.columns or lat_col not in df.columns:
        return gpd.GeoDataFrame(df, geometry=[], crs="EPSG:4326")
    x = pd.to_numeric(df[lon_col], errors="coerce")
    y = pd.to_numeric(df[lat_col], errors="coerce")
    geom = [Point(a, b) if (not pd.isna(a) and not pd.isna(b)) else None for a, b in zip(x, y)]
    gdf = gpd.GeoDataFrame(df.copy(), geometry=geom, crs="EPSG:4326")
    gdf = gdf[~gdf.geometry.isna()]
    return gdf

@st.cache_data(show_spinner=True, ttl=60*60*24)
def census_get_vars(year: str, state_fips: str, county_fips: str, var_list: list):
    url = f"https://api.census.gov/data/{year}/acs/acs5"
    params = {"get": ",".join(["NAME"] + var_list), "for": "tract:*", "in": f"state:{state_fips} county:{county_fips}"}
    r = requests.get(url, params=params, timeout=120)
    r.raise_for_status()
    raw = r.json()
    df = pd.DataFrame(raw[1:], columns=raw[0])
    for v in var_list:
        if v in df.columns:
            df[v] = pd.to_numeric(df[v], errors="coerce")
    df["geoid_tract"] = df["state"] + df["county"] + df["tract"]
    return df

@st.cache_data(show_spinner=True, ttl=60*60*24)
def census_sum_by_label(year: str, group: str, label_contains: list):
    """Return ACS var names (Estimate columns) whose labels contain ALL substrings."""
    meta = get_census_group_metadata(year, group)
    hits = []
    for var, meta_v in meta["variables"].items():
        if not var.endswith("E"):
            continue
        lab = meta_v.get("label", "")
        if all(s in lab for s in label_contains):
            hits.append(var)
    return hits

def subset_to_city_tracts(tract_df: pd.DataFrame, tracts_gdf: gpd.GeoDataFrame, city_gdf: gpd.GeoDataFrame):
    g_city = city_gdf.to_crs(tracts_gdf.crs)
    tr_ = gpd.overlay(tracts_gdf, g_city[["geometry"]], how="intersection")
    tr_ids = set(tr_["geoid_tract"].unique().tolist())
    return tract_df[tract_df["geoid_tract"].isin(tr_ids)].copy()

@st.cache_data(show_spinner=True, ttl=60*60*24)
def get_cdc_places_tract(level_year: str = "2023"):
    """
    CDC PLACES tract-level indicators (GIS-friendly CSV).
    Returns two columns guaranteed: ['geoid_tract','uninsured_18_64_pct'] (may be empty).
    """
    url = "https://chronicdata.cdc.gov/api/views/cwsq-ngmh/rows.csv?accessType=DOWNLOAD"  # 2023 PLACES Tract GIS-friendly
    r = requests.get(url, timeout=180)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), low_memory=False)

    # Construct GEOID from several possible fields (preserve 11 digits)
    geoid_series = None
    for cand in ["TractFIPS", "TractFIPS10", "LocationID"]:
        if cand in df.columns:
            s = df[cand].astype(str).str.extract(r"(\d{11})", expand=False)
            if s.notna().any():
                geoid_series = s.str.zfill(11)
                break
    if geoid_series is None:
        for c in df.columns:
            if "fips" in c.lower():
                s = df[c].astype(str).str.extract(r"(\d{11})", expand=False)
                if s.notna().any():
                    geoid_series = s.str.zfill(11)
                    break

    # Filter to uninsured 18–64
    if "Measure" in df.columns and "Data_Value" in df.columns:
        df_ins = df[df["Measure"].str.contains("lack of health insurance", case=False, na=False)].copy()
        df_ins.rename(columns={"Data_Value": "uninsured_18_64_pct"}, inplace=True)
    else:
        df_ins = pd.DataFrame(columns=["uninsured_18_64_pct"])

    if geoid_series is not None:
        df_ins["geoid_tract"] = geoid_series.loc[df_ins.index].values

    # Always return canonical columns; may be empty
    return df_ins.reindex(columns=["geoid_tract", "uninsured_18_64_pct"]).dropna(how="all")

def rate(numer, denom):
    numer = pd.to_numeric(numer, errors="coerce")
    denom = pd.to_numeric(denom, errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(denom > 0, numer / denom, np.nan)
    return out

def fmtnum(x, pct=False):
    try:
        if pd.isna(x):
            return "—"
        if pct:
            return f"{float(x)*100:,.1f}%"
        return f"{float(x):,.0f}"
    except Exception:
        return "—"

# --- NEW: helpers for BLUF/fallbacks ---
def safe_mean(series):
    """Mean that returns np.nan instead of warnings when empty/all-nan."""
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().any():
        return float(np.nanmean(s))
    return np.nan

def metric_or_fallback(col, label, value, fmt="num", help_text="", fallback_note="No tract data; use county/CHAS for context"):
    """Show metric with smart fallback if value is nan."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        col.metric(label, "—")
        if fallback_note:
            col.caption(f"⚠️ {fallback_note}")
    else:
        if fmt == "pct":
            col.metric(label, f"{value*100:.1f}%", help=help_text)
        elif fmt == "usd":
            col.metric(label, f"${value:,.0f}", help=help_text)
        elif fmt == "ratio":
            col.metric(label, f"{value:,.1f}×", help=help_text)
        else:
            col.metric(label, f"{value:,.0f}", help=help_text)

# ---------------------------
# Main App — Phase 1
# ---------------------------
st.title("City ADOS & Equity Context — 72-Hour Prototype")
st.caption("ADOS-proxy (Native-born Black alone) from ACS + optional local/state open data feeds")

with st.spinner("Fetching city boundary…"):
    city_gdf = get_city_boundary_from_overpass(CITY_NAME)
st.success("Loaded city boundary")

# ---- Map the city boundary cleanly (pydeck GeoJson) ----
try:
    minx, miny, maxx, maxy = city_gdf.total_bounds
    lon_center = (minx + maxx) / 2
    lat_center = (miny + maxy) / 2

    city_geojson = city_gdf.to_json()
    layer = pdk.Layer(
        "GeoJsonLayer",
        data=city_geojson,
        stroked=True,
        filled=False,
        get_line_color=[20, 120, 240, 200],
        line_width_min_pixels=2,
    )
    view = pdk.ViewState(latitude=lat_center, longitude=lon_center, zoom=11)
    deck = pdk.Deck(layers=[layer], initial_view_state=view, map_style=None)
    st.pydeck_chart(deck)
except Exception:
    # Fallback: plot centroid point
    cent = city_gdf.to_crs(4326).copy()
    cent["latitude"] = cent.geometry.centroid.y
    cent["longitude"] = cent.geometry.centroid.x
    st.map(cent[["latitude", "longitude"]])

with st.expander("City boundary (attributes)"):
    st.dataframe(
        city_gdf.drop(columns=["geometry"]).assign(
            area_sqkm=city_gdf.to_crs(3857).area / 1e6
        )
    )

colA, colB = st.columns([3, 2])

with colA:
    st.subheader("ADOS-proxy from ACS (B05003B)")
    with st.spinner("Pulling ACS + tract geometries…"):
        acs_df    = get_census_tract_black_nativity(ACS_YEAR, STATE_FIPS, COUNTY_FIPS, "B05003B")
        tracts_gd = get_pg_tract_geometries(STATE_FIPS, COUNTY_FIPS)

    # Spatial intersect: tracts intersecting city
    g_city = city_gdf.to_crs(tracts_gd.crs)
    tr_ = gpd.overlay(tracts_gd, g_city[["geometry"]], how="intersection")
    tr_ids = tr_["geoid_tract"].unique().tolist()

    acs_sel = acs_df[acs_df["geoid_tract"].isin(tr_ids)].copy()

    # Aggregate to city
    city_native      = acs_sel["native_black"].sum()
    city_foreign     = acs_sel["foreign_black"].sum()
    city_black_total = acs_sel["black_total"].sum()
    ados_proxy_share = (city_native / city_black_total) if city_black_total > 0 else np.nan

    k1, k2, k3 = st.columns(3)
    k1.metric("U.S.-born Black residents (est.)", f"{int(city_native):,}")
    k2.metric("Foreign-born Black residents (est.)", f"{int(city_foreign):,}")
    k3.metric("ADOS proxy (share of Black)", f"{ados_proxy_share:.1%}" if pd.notna(ados_proxy_share) else "—")

    # Tract-level bar (ADOS proxy)
    acs_map = acs_sel.merge(tracts_gd, on="geoid_tract", how="left")
    acs_map = gpd.GeoDataFrame(acs_map, geometry="geometry", crs=tracts_gd.crs)
    acs_map["ados_proxy_share"] = np.where(acs_map["black_total"] > 0,
                                           acs_map["native_black"] / acs_map["black_total"], np.nan)

    st.markdown("**Tract-level ADOS proxy inside city**")
    try:
        acs_map_plot = (acs_map
                        .sort_values("ados_proxy_share", ascending=False)
                        .assign(tract_short=lambda d: d["geoid_tract"].str[-6:]))
        chart = alt.Chart(
            acs_map_plot.dropna(subset=["ados_proxy_share"])
        ).mark_bar().encode(
            x=alt.X('tract_short:N', title="Census tract (short id)"),
            y=alt.Y('ados_proxy_share:Q', title='Share of U.S.-born Black residents', axis=alt.Axis(format='%')),
            tooltip=[
                alt.Tooltip('geoid_tract:N',         title='Census tract ID'),
                alt.Tooltip('ados_proxy_share:Q',    title='ADOS proxy (share)', format='.1%'),
                alt.Tooltip('native_black:Q',        title='U.S.-born Black (est.)', format=',.0f'),
                alt.Tooltip('black_total:Q',         title='Black total (est.)',     format=',.0f')
            ]
        ).properties(height=280)
        st.altair_chart(chart, use_container_width=True)
    except Exception:
        st.dataframe(acs_map[["geoid_tract", "ados_proxy_share", "native_black", "black_total"]])

with colB:
    st.subheader("Data freshness & caveats")
    st.info(
        "• **ADOS proxy** = U.S.-born Black (alone) from ACS table **B05003B**.\n"
        "• This is a **proxy**, not identity; precise ancestry/parent-nativity needs PUMS (Phase 2).\n"
        "• Tracts are intersected with the city boundary (OSM); small edge effects possible.\n"
        "• Results are **indicative** and subject to ACS sampling error (especially tract level)."
    )
    st.caption(f"ACS {ACS_YEAR} 5-yr • State FIPS {STATE_FIPS}, County FIPS {COUNTY_FIPS}")

st.markdown("---")
st.subheader("Optional: Open Data quick glance")

col1, col2, col3 = st.columns(3)
with col1:
    st.write("**311 Service Calls (PG County)**")
    df_311 = fetch_socrata(SOCRATA_DOMAIN_PG, PG_DATASETS["311 Service Calls"], MAX_ROWS)
    if not df_311.empty:
        g311 = safe_pointify(df_311, lon_col="longitude", lat_col="latitude")
        if not g311.empty:
            g311_in = gpd.sjoin(g311.to_crs("EPSG:4326"), city_gdf.to_crs("EPSG:4326"),
                                predicate="within", how="inner")
            st.write(f"Rows inside boundary: {len(g311_in):,}")
        st.dataframe(df_311.head(25))
    else:
        st.caption("No rows (or dataset id blank).")

with col2:
    st.write("**Code Enforcement (PG County)**")
    df_code = fetch_socrata(SOCRATA_DOMAIN_PG, PG_DATASETS["Code Enforcement"], MAX_ROWS)
    if not df_code.empty:
        st.dataframe(df_code.head(25))
    else:
        st.caption("No rows (or dataset id blank).")

with col3:
    st.write("**Crime (PG County)**")
    df_crime = fetch_socrata(SOCRATA_DOMAIN_PG, PG_DATASETS["Crime Reports – Current"], MAX_ROWS)
    if not df_crime.empty:
        gcr = safe_pointify(df_crime, lon_col="longitude", lat_col="latitude")
        if not gcr.empty:
            gcr_in = gpd.sjoin(gcr.to_crs("EPSG:4326"), city_gdf.to_crs("EPSG:4326"),
                               predicate="within", how="inner")
            st.write(f"Rows inside boundary: {len(gcr_in):,}")
        st.dataframe(df_crime.head(25))
    else:
        st.caption("No rows (or dataset id blank).")

# Optional Maryland portal example (leave IDs blank unless you have one)
if SOCRATA_DOMAIN_MD_OPT.strip():
    st.markdown("---")
    st.subheader("Optional: Maryland Open Data (custom IDs)")
    st.caption("Enter dataset IDs from the state portal; results will show here if the API shape matches.")
    # Example placeholder: uncomment if you add an ID
    # df_md = fetch_socrata(SOCRATA_DOMAIN_MD_OPT, "<dataset_id_here>", MAX_ROWS)
    # if not df_md.empty: st.dataframe(df_md.head(25))

st.markdown("---")
st.header("Wealth & Health Context (tract → city)")

with st.spinner("Loading contextual ACS & PLACES indicators…"):
    tracts_gdf = get_pg_tract_geometries(STATE_FIPS, COUNTY_FIPS)
    acs_core = census_get_vars(
        ACS_YEAR, STATE_FIPS, COUNTY_FIPS,
        var_list=[
            # Income & households
            "B19013_001E",       # median household income ($)
            "B11001_001E",       # total households (for denominators)
            # Home values & rent
            "B25077_001E",       # median home value ($)
            "B25064_001E",       # median gross rent ($)
            # Labor force & unemployment
            "B23025_003E",       # civilian labor force
            "B23025_005E",       # unemployed
            # Homeownership by race: Black alone
            "B25003B_001E",      # total Black-alone occupied housing units
            "B25003B_002E",      # owner-occupied Black-alone units
            "B25003B_003E",      # renter-occupied Black-alone units
            # Asset-income proxy
            "B19053_001E",       # households universe for asset-income
            "B19053_002E",       # households with interest/dividends/rental income
        ],
    )

    # Poverty (Black alone) via label search in B17020B
    pov_vars = census_sum_by_label(ACS_YEAR, "B17020B", ["Black or African American alone", "Below"])
    if pov_vars:
        acs_pov = census_get_vars(ACS_YEAR, STATE_FIPS, COUNTY_FIPS, pov_vars + ["B17020B_001E"])
        acs_pov["pov_black_count"] = acs_pov[pov_vars].sum(axis=1, skipna=True)
        acs_pov.rename(columns={"B17020B_001E": "pov_black_universe"}, inplace=True)
        acs_core = acs_core.merge(
            acs_pov[["geoid_tract", "pov_black_count", "pov_black_universe"]],
            on="geoid_tract", how="left"
        )
    else:
        acs_core["pov_black_count"] = np.nan
        acs_core["pov_black_universe"] = np.nan

    # No vehicle available (mobility proxy) from B08201 via label search
    nov_vars = census_sum_by_label(ACS_YEAR, "B08201", ["No vehicle available"])
    if nov_vars:
        acs_noveh = census_get_vars(ACS_YEAR, STATE_FIPS, COUNTY_FIPS, nov_vars + ["B08201_001E"])
        acs_noveh["no_vehicle_households"] = acs_noveh[nov_vars].sum(axis=1, skipna=True)
        acs_noveh.rename(columns={"B08201_001E": "veh_table_households"}, inplace=True)
        acs_core = acs_core.merge(
            acs_noveh[["geoid_tract", "no_vehicle_households", "veh_table_households"]],
            on="geoid_tract", how="left"
        )
    else:
        acs_core["no_vehicle_households"] = np.nan
        acs_core["veh_table_households"] = np.nan

    # CDC PLACES — Uninsured (18–64)
    places = get_cdc_places_tract("2023")

    # Compute indicators
    acs_core["median_income"]       = acs_core["B19013_001E"]
    acs_core["median_home_value"]   = acs_core["B25077_001E"]
    acs_core["median_gross_rent"]   = acs_core["B25064_001E"]
    acs_core["unemp_rate"]          = rate(acs_core["B23025_005E"], acs_core["B23025_003E"])
    acs_core["black_owner_rate"]    = rate(acs_core["B25003B_002E"], acs_core["B25003B_001E"])
    acs_core["black_poverty_rate"]  = rate(acs_core["pov_black_count"], acs_core["pov_black_universe"])
    acs_core["asset_income_rate"]   = rate(acs_core["B19053_002E"], acs_core["B19053_001E"])
    acs_core["no_vehicle_rate"]     = rate(acs_core["no_vehicle_households"], acs_core["veh_table_households"])

    # Join PLACES uninsured (safe)
    if "geoid_tract" in places.columns:
        acs_core = acs_core.merge(places, on="geoid_tract", how="left")
    else:
        st.warning("CDC PLACES: no tract GEOIDs found; skipping uninsured merge for now.")

    # Limit to city tracts
    acs_city = subset_to_city_tracts(acs_core, tracts_gdf, city_gdf)

# =========================
# BLUF PANEL (30 seconds)
# =========================
st.markdown("---")
st.header("BLUF: What’s the headline for leadership (30 seconds)")

# Compute safe tract-level means/medians for BLUF KPIs
med_income = np.nanmedian(acs_city["median_income"])     if "median_income"     in acs_city else np.nan
med_home   = np.nanmedian(acs_city["median_home_value"]) if "median_home_value" in acs_city else np.nan
med_rent   = np.nanmedian(acs_city["median_gross_rent"]) if "median_gross_rent" in acs_city else np.nan

asset_part = safe_mean(acs_city.get("asset_income_rate", np.nan))   # 0–1
no_vehicle = safe_mean(acs_city.get("no_vehicle_rate", np.nan))     # 0–1
unemp_rate = safe_mean(acs_city.get("unemp_rate", np.nan))          # 0–1
black_pov  = safe_mean(acs_city.get("black_poverty_rate", np.nan))  # 0–1 (may be nan if unavailable)
uninsured  = safe_mean(acs_city.get("uninsured_18_64_pct", np.nan)) # already a percent value (0–100), may be nan

# Price-to-income (higher = harder to buy)
price_to_income = (med_home / med_income) if (pd.notna(med_home) and pd.notna(med_income) and med_income > 0) else np.nan
# Rent burden (typical rent as % of typical monthly income)
rent_burden = (med_rent / (med_income/12.0)) if (pd.notna(med_rent) and pd.notna(med_income) and med_income > 0) else np.nan

# Simple vulnerability index (0–100): normalize “strain” signals
signals = []
uninsured_01 = uninsured/100.0 if pd.notna(uninsured) else np.nan
signals.append(no_vehicle)                                                # 0–1
signals.append(unemp_rate)                                                # 0–1
signals.append(1 - asset_part if pd.notna(asset_part) else np.nan)        # lack of asset participation
signals.append(min(rent_burden/0.30, 2.0) / 2.0 if pd.notna(rent_burden) else np.nan)  # >30% burden scaled
signals.append(uninsured_01 if pd.notna(uninsured_01) else np.nan)

sig = pd.to_numeric(pd.Series(signals), errors="coerce")
vulnerability_index = float(np.nanmean(sig) * 100.0) if sig.notna().any() else np.nan

# Headline composite
hcol = st.columns(1)[0]
metric_or_fallback(
    hcol, "Community Vulnerability (composite)", vulnerability_index,
    fmt="num",
    help_text="0–100 composite (higher = greater strain): no vehicle, unemployment, lack of asset income, rent burden, uninsured"
)

# Row 1
b1, b2, b3 = st.columns(3)
metric_or_fallback(
    b1, "Wealth participation", asset_part, fmt="pct",
    help_text="Share of households with interest/dividends/rent income (wealth proxy)"
)
metric_or_fallback(
    b2, "Home price-to-income", price_to_income, fmt="ratio",
    help_text="Median home value divided by median household income (higher = harder to buy)"
)
metric_or_fallback(
    b3, "Rent burden (typical)", rent_burden, fmt="pct",
    help_text="Median rent as % of median monthly income (≥30% = burdened)"
)

# Row 2
c1, c2, c3 = st.columns(3)
metric_or_fallback(
    c1, "Mobility constraint", no_vehicle, fmt="pct",
    help_text="Households with no vehicle"
)
metric_or_fallback(
    c2, "Adults without health insurance",
    (uninsured/100.0) if pd.notna(uninsured) else np.nan, fmt="pct",
    help_text="CDC PLACES estimate for ages 18–64"
)
metric_or_fallback(
    c3, "Black poverty rate", black_pov, fmt="pct",
    help_text="If blank at tract level, use county/CHAS as fallback"
)

with st.expander("How to read these (plain English)"):
    st.markdown("""
- **Wealth participation**: percent of households reporting interest/dividends/rent income — a practical wealth proxy.
- **Home price-to-income**: how many times the median home price exceeds the median household income (higher = harder to buy).
- **Rent burden**: typical rent as a share of typical monthly income (≥30% is considered burdened).
- **Mobility constraint**: percent of households without a vehicle (limits access to jobs, childcare, healthcare).
- **Adults without health insurance**: share of 18–64 without coverage (CDC PLACES).
- **Black poverty rate**: if blank, tract sample is too small; use county or HUD CHAS city tables.
- **Community Vulnerability**: a simple composite (0–100) combining the strain signals above to help triage attention.
""")

# Relationships: ADOS proxy vs context
st.subheader("Relationships: ADOS proxy vs Wealth & Health")
try:
    acs_df_rel = acs_df.copy()
    acs_df_rel["ados_proxy_share"] = rate(acs_df_rel["native_black"], acs_df_rel["black_total"])
    rel = subset_to_city_tracts(acs_df_rel, tracts_gdf, city_gdf).merge(
        acs_city, on="geoid_tract", suffixes=("","")
    )

    def scatter(x_field, x_title, x_is_pct=False):
        df_plot = rel.dropna(subset=["ados_proxy_share", x_field]).copy()
        enc_x = alt.X(f"{x_field}:Q", title=x_title, axis=alt.Axis(format='.1%' if x_is_pct else None))
        return alt.Chart(df_plot).mark_circle(size=80).encode(
            x=enc_x,
            y=alt.Y("ados_proxy_share:Q", title="Share of U.S.-born Black residents", axis=alt.Axis(format='%')),
            tooltip=[
                alt.Tooltip('geoid_tract:N', title='Census tract ID'),
                alt.Tooltip('ados_proxy_share:Q', title='ADOS proxy (share)', format='.1%'),
                alt.Tooltip(f'{x_field}:Q', title=x_title, format='.1%' if x_is_pct else ',.0f')
            ]
        ).properties(height=280)

    c1 = scatter("median_income", "Median household income ($)")
    c2 = scatter("black_owner_rate", "Black homeownership rate", x_is_pct=True)
    c3 = scatter("black_poverty_rate", "Black poverty rate", x_is_pct=True)
    c4 = scatter("uninsured_18_64_pct", "Uninsured adults (18–64) (%)") if "uninsured_18_64_pct" in rel.columns \
         else scatter("median_gross_rent", "Typical monthly rent ($)")
    c5 = scatter("asset_income_rate", "Households w/ asset income", x_is_pct=True)
    c6 = scatter("no_vehicle_rate", "Households with no vehicle", x_is_pct=True)

    st.altair_chart(alt.hconcat(c1, c2, c3).resolve_scale(y='shared'), use_container_width=True)
    st.altair_chart(alt.hconcat(c4, c5, c6).resolve_scale(y='shared'), use_container_width=True)
except Exception as e:
    st.warning(f"Chart render issue: {e}")
    st.dataframe(acs_city[["geoid_tract","median_income","black_owner_rate","black_poverty_rate",
                           "median_home_value","median_gross_rent","unemp_rate",
                           "asset_income_rate","no_vehicle_rate"]].head(12))

st.markdown("---")
st.subheader("⚠️ Data Caveats & Limitations")
st.info(
    "• **ADOS proxy** = U.S.-born Black (alone) from ACS B05003B (tract level); city totals aggregate intersecting tracts.\n"
    "• Context metrics are tract medians/means (some race-specific where available), not ADOS-only.\n"
    "• **Weighted views** can be added; Phase 2 adds city-level HUD CHAS and strict ADOS via PUMS.\n"
    "• ACS has sampling error; interpret as indicative, not exact. Prototype generalizes to other cities via OSM."
)

# =============================================================================
# Phase 2 — Updater buttons (CHAS, LODES, PUMS) + Auto-detect & render
# =============================================================================
st.markdown("---")
st.header("Phase 2: HUD CHAS, LODES, PUMS (Experimental)")

PHASE2_IN  = Path("phase2/inputs");  PHASE2_IN.mkdir(parents=True, exist_ok=True)
PHASE2_OUT = Path("phase2/outputs"); PHASE2_OUT.mkdir(parents=True, exist_ok=True)

STATE_FIPS_MD = "24"
GREENBELT_PLACE_CODE       = "34775"
GREENBELT_PLACE_GEOID_NUM  = int(STATE_FIPS_MD + GREENBELT_PLACE_CODE)

def update_chas(chas_url: str = "https://www.huduser.gov/portal/sites/default/files/xls/CHAS/CHAS_2016-2020_MD.csv.zip"):
    try:
        r = requests.get(chas_url, timeout=120)
        r.raise_for_status()
    except Exception as e:
        st.error(f"CHAS download failed: {e}")
        return None
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        z.extractall(PHASE2_IN)
    # find MD CSV
    csv_file = None
    for c in PHASE2_IN.glob("CHAS*MD*.csv"):
        csv_file = c; break
    if csv_file is None:
        st.error("CHAS MD CSV not found in ZIP. HUD may have changed filenames.")
        return None
    df = pd.read_csv(csv_file, low_memory=False)
    keep = df[df.get("geoname","").astype(str).str.contains("Greenbelt", case=False, na=False)].copy()
    if keep.empty and "place" in df.columns:
        try:
            keep = df[df["place"].astype(str).astype(float).astype(int) == GREENBELT_PLACE_GEOID_NUM].copy()
        except Exception:
            pass
    out_fp = PHASE2_OUT / "chas_greenbelt.csv"
    keep.to_csv(out_fp, index=False)
    return out_fp

def update_lodes(year: int = 2021):
    rac_url   = f"https://lehd.ces.census.gov/data/lodes/LODES8/md/rac/md_rac_S000_JT00_{year}.csv.gz"
    xwalk_url = "https://lehd.ces.census.gov/data/lodes/LODES8/md/xwalk/md_xwalk.csv.gz"
    rac_fp = PHASE2_IN / f"md_rac_{year}.csv.gz"
    xw_fp  = PHASE2_IN / "md_xwalk.csv.gz"
    try:
        with requests.get(rac_url, stream=True, timeout=180) as r:
            r.raise_for_status()
            with open(rac_fp, 'wb') as f: shutil.copyfileobj(r.raw, f)
    except Exception as e:
        st.error(f"LODES RAC download failed: {e}")
        return None
    if not xw_fp.exists():
        try:
            with requests.get(xwalk_url, stream=True, timeout=180) as r:
                r.raise_for_status()
                with open(xw_fp, 'wb') as f: shutil.copyfileobj(r.raw, f)
        except Exception as e:
            st.error(f"LODES xwalk download failed: {e}")
            return None
    try:
        rac = pd.read_csv(rac_fp, compression="gzip", low_memory=False)
        xw  = pd.read_csv(xw_fp,  compression="gzip", low_memory=False)
    except Exception as e:
        st.error(f"LODES read failed: {e}")
        return None
    join_left  = "h_geocode"   if "h_geocode"   in rac.columns else None
    join_right = "tabblk2020"  if "tabblk2020"  in xw.columns  else None
    if not join_left or not join_right or "place" not in xw.columns:
        st.error("Expected columns not found for LODES/xwalk (need h_geocode, tabblk2020, place).")
        return None
    merged = rac.merge(xw, left_on=join_left, right_on=join_right, how="left")
    try:
        merged["place"] = pd.to_numeric(merged["place"], errors="coerce")
        gb = merged[merged["place"] == int(GREENBELT_PLACE_CODE)].copy()
    except Exception as e:
        st.error(f"LODES filter failed: {e}")
        return None
    out_fp = PHASE2_OUT / f"lodes_greenbelt_rac_{year}.csv"
    gb.to_csv(out_fp, index=False)
    return out_fp

def update_pums(pums_file: str = "phase2/inputs/ipums_pums_md.csv"):
    pfile = Path(pums_file)
    if not pfile.exists():
        st.warning("PUMS file missing. Export from IPUMS (ACS 5-yr, MD) with variables: RACE,HISPAN,NATIVITY,MBPL,FBPL,CITIZEN,PUMA,PERWT")
        return None
    df = pd.read_csv(pfile, low_memory=False)
    need = ["RACE","NATIVITY","MBPL","FBPL","PUMA","PERWT"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        st.error(f"PUMS missing columns: {missing}")
        return None
    hisp = "HISPAN" if "HISPAN" in df.columns else None
    cit  = "CITIZEN" if "CITIZEN" in df.columns else None

    is_black   = df["RACE"].eq(200)  # IPUMS code for Black alone (verify in your extract)
    is_non_hisp= df[hisp].eq(0) if hisp else True
    native     = df["NATIVITY"].eq(1)
    mbpl_us    = pd.to_numeric(df["MBPL"], errors="coerce") < 100
    fbpl_us    = pd.to_numeric(df["FBPL"], errors="coerce") < 100
    citizen_ok = df[cit].eq(1) if cit else True

    ados = is_black & is_non_hisp & native & mbpl_us & fbpl_us & citizen_ok
    df["PERWT"] = pd.to_numeric(df["PERWT"], errors="coerce").fillna(0)

    def safe_div(a, b): return (a / b) if b and b > 0 else np.nan

    grp = (df.groupby("PUMA")
           .apply(lambda g: pd.Series({
               "pop_total_wt": g["PERWT"].sum(),
               "ados_total_wt": g.loc[ados, "PERWT"].sum(),
               "ados_share":   safe_div(g.loc[ados, "PERWT"].sum(), g["PERWT"].sum())
           }))
           .reset_index())

    out_fp = PHASE2_OUT / "pums_md_puma_ados.csv"
    grp.to_csv(out_fp, index=False)
    return out_fp

# Buttons row
b1, b2, b3 = st.columns(3)
with b1:
    if st.button("Update CHAS"):
        fp = update_chas()
        st.success(f"CHAS saved → {fp}" if fp else "CHAS update failed.")
with b2:
    if st.button("Update LODES"):
        fp = update_lodes()
        st.success(f"LODES saved → {fp}" if fp else "LODES update failed.")
with b3:
    if st.button("Update PUMS"):
        fp = update_pums()
        st.success(f"PUMS saved → {fp}" if fp else "PUMS update failed.")

# Auto-detect & render Phase-2 outputs
st.markdown("---")
st.header("Phase 2 — Auto Results (if available)")

# CHAS
st.subheader("HUD CHAS (Place) — Greenbelt")
chas_fp = Path("phase2/outputs/chas_greenbelt.csv")
if chas_fp.exists():
    chas = pd.read_csv(chas_fp, low_memory=False)
    st.caption("Official HUD place-level housing needs for **Greenbelt city** (most scrutiny-ready).")
    with st.expander("Preview CHAS rows"):
        st.dataframe(chas.head(25))
else:
    st.caption("No CHAS output yet. Click **Update CHAS** above.")

# LODES
st.subheader("LODES (RAC) — Residents in Greenbelt")
lodes_candidates = sorted(Path("phase2/outputs").glob("lodes_greenbelt_rac_*.csv"))
if lodes_candidates:
    lodes_fp = lodes_candidates[-1]
    lodes = pd.read_csv(lodes_fp, low_memory=False)
    st.caption(f"LEHD LODES RAC subset for **Greenbelt** (file: `{Path(lodes_fp).name}`)")
    with st.expander("Preview LODES rows"):
        st.dataframe(lodes.head(25))
else:
    st.caption("No LODES output yet. Click **Update LODES** above.")

# PUMS
st.subheader("PUMS (ADOS Eligibility) — PUMA level")
pums_fp = Path("phase2/outputs/pums_md_puma_ados.csv")
if pums_fp.exists():
    pums = pd.read_csv(pums_fp)
    st.caption("ACS PUMS aggregated to **PUMA** (coarser than city). Use as context; disclose geography difference.")
    with st.expander("Preview PUMS PUMA summary"):
        st.dataframe(pums.head(25))
    try:
        pums_plot = pums.dropna(subset=["ados_share"]).sort_values("ados_share", ascending=False).copy()
        pums_plot["ados_share_pct"] = pums_plot["ados_share"] * 100
        chart = alt.Chart(pums_plot).mark_bar().encode(
            x=alt.X("ados_share_pct:Q", title="ADOS share (%)"),
            y=alt.Y("PUMA:N", sort="-x"),
            tooltip=["PUMA", "ados_share_pct"]
        ).properties(height=320)
        st.altair_chart(chart, use_container_width=True)
    except Exception as e:
        st.caption(f"(Chart fallback) {e}")
else:
    st.caption("No PUMS output yet. Click **Update PUMS** above after placing your IPUMS CSV in `phase2/inputs/`.")

st.markdown("—")
st.caption(
    "Phase 2 notes: **CHAS** is official city (place) level; **LODES** is filtered to Greenbelt via the LEHD crosswalk; "
    "**PUMS** is PUMA-level (larger than city). Re-run updates when new vintages release."
)
