# app.py
# Greenbelt, MD • Reparations Context Prototype (robust + Cloud friendly)
# - City boundary: TIGERweb (preferred) -> fallback to OSM (Nominatim)
# - Tracts: Census cartographic boundary (MD 2023) for county 033
# - Intersection is robust (CRS align + small buffer) with county fallback
# - ADOS proxy (ACS B05003B) + simple Wealth/Health context (fast)

import io
import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Greenbelt ADOS / Wealth Gap Prototype", layout="wide")

# ---------------------------
# Lazy geo imports (avoid Cloud boot flakiness)
# ---------------------------
def _gpd():
    import geopandas as _g
    return _g

def _shapely():
    from shapely.geometry import shape as _shape
    return _shape

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.header("Configuration")

CITY_NAME = st.sidebar.text_input("City", "Greenbelt")            # keep simple name for TIGERweb
STATE_FIPS = st.sidebar.text_input("State FIPS", "24")            # Maryland
COUNTY_FIPS = st.sidebar.text_input("County FIPS", "033")         # Prince George's
ACS_YEAR = st.sidebar.selectbox("ACS 5-year vintage", ["2023","2022","2021"], index=0)

ALLOW_OSM_FALLBACK = st.sidebar.checkbox(
    "If TIGERweb fails, allow OSM fallback", value=True,
    help="If TIGERweb cannot find the place, we will try OpenStreetMap."
)

INCLUDE_PLACES = st.sidebar.checkbox(
    "Include CDC PLACES uninsured metric (slower)", value=False,
    help="Downloads ~100MB CSV; leave off if you just need the ACS-side prototypes."
)

st.sidebar.caption("Cloud tip: leave steps button-driven to avoid long cold starts.")
st.sidebar.caption("Use Python 3.10 locally for the most stable geo stack.")

# ---------------------------
# Boundary helpers
# ---------------------------
@st.cache_data(show_spinner=True, ttl=60
