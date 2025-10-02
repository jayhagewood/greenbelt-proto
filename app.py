# app.py
# Greenbelt, MD • Reparations Context Prototype (Streamlit Cloud-safe)
# - City boundary: Census TIGERweb (ArcGIS REST GeoJSON), auto-fallback to OSM (Nominatim)
# - Tract geometries: TIGERweb (GeoJSON), fallback to Cartographic Boundary ZIP (if allowed)
# - ADOS proxy (ACS B05003B) + Wealth/Health context (ACS + CDC PLACES)
# - Lazy geo imports so Cloud starts even if Geo stack is slow to import

import io
import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="ADOS / Wealth Gap Prototype", layout="wide")

# ---------------------------
# Lazy geo imports (avoid Cloud boot hiccups)
# ---------------------------
def _gpd():
    import geopandas as _g
    return _g

def _shapely():
    from shapely.geometry import Point as _Point, shape as _shape
    return _Point, _shape

# ---------------------------
# Sidebar config
# ---------------------------
st.sidebar.header("Configuration")

# IMPORTANT: keep just the short city name by default
CITY_NAME   = st.sidebar.text_input("City (just the name)", "Greenbelt")
STATE_FIPS  = st.sidebar.text_input("State FIPS", "24")       # Maryland
COUNTY_FIPS = st.sidebar.text_input("County FIPS", "033")     # Prince George's County
ACS_YEAR    = st.sidebar.selectbox("ACS 5-year vintage", ["2023","2022","2021"], index=0)

st.sidebar.markdown("---")
st.sidebar.caption("Tip: Add a 'runtime.txt' with '3.10' so Streamlit Cloud uses Python 3.10 (more stable geo stack).")

# ---------------------------
# Boundary helpers (robust, no ZIPs)
# ---------------------------
@st.cache_data(show_spinner=True, ttl=60*60*24)
def get_city_boundary_from_tigerweb(place_name: str, state_fips: str):
    """
    Robust search of Census TIGERweb (ArcGIS REST, GeoJSON) for a place boundary.
    - Cleans the place name ("Greenbelt, Maryland, USA" → "Greenbelt")
    - Tries multiple layers & both STATE / STATEFP fields
    - Tries exact NAME and prefix LIKE
    Returns: GeoDataFrame (EPSG:4326) with one row; raises if nothing found.
    """
    gpd = _gpd()

    base = (
        "https://tigerweb.geo.census.gov/arcgis/rest/services/"
        "TIGERweb/Places_CouSub_ConCity_SubMCD/MapServer"
    )

    # Clean the incoming name down to a simple token
    token = place_name.split(",")[0].strip()
    token = token.replace(" city", "").replace(" town", "").replace(" village", "")
    token_esc = token.replace("'", "''")
    state = str(state_fips).zfill(2)

    # Candidate layers commonly used for places (incorporated, CDPs, etc.)
    layers = [0, 1, 2, 3, 4, 5]
    # Some layers use STATE, others STATEFP
    state_fields = ["STATE", "STATEFP"]

    def _queries(sf):
        return [
            f"{sf}='{state}' AND UPPER(NAME)=UPPER('{token_esc}')",
            f"{sf}='{state}' AND UPPER(NAME) LIKE UPPER('{token_esc}%')",
        ]

    collected = []

    for lyr in layers:
        lyr_url = f"{base}/{lyr}/query"
        for sf in state_fields:
            for where in _queries(sf):
                params = {
                    "where": where,
                    "outFields": "*",
                    "f": "geojson",
                    "outSR": 4326,
                    "returnGeometry": "true",
                }
                try:
                    r = requests.get(lyr_url, params=params, timeout=60)
                    r.raise_for_status()
                    j = r.json()
                    feats = (j.get("features") or [])
                    if feats:
                        gdf_try = gpd.GeoDataFrame.from_features(feats, crs="EPSG:4326")
                        gdf_try["__layer"] = lyr
                        collected.append(gdf_try)
                except Exception:
                    continue
        if collected:
            break

    if not collected:
        raise RuntimeError(
            f"TIGERweb: no place found for '{place_name}' in state {state_fips} "
            f"(tried layers {layers} and fields {state_fields})."
        )

    gdf = pd.concat(collected, ignore_index=True)

    # Score matches: prefer exact NAME and city-type NAMELSAD
    gdf["__score"] = 0
    if "NAME" in gdf.columns:
        gdf.loc[gdf["NAME"].str.upper() == token.upper(), "__score"] += 5
    if "NAMELSAD" in gdf.columns:
        gdf.loc[gdf["NAMELSAD"].str.contains("city", case=False, na=False), "__score"] += 2
        gdf.loc[gdf["NAMELSAD"].str.contains("town|village", case=False, na=False), "__score"] += 1

    gdf = gdf.sort_values(["__score", "__layer", "NAME"], ascending=[False, True, True]).head(1).copy()
    gdf["name"] = token
    gdf["geometry"] = gdf["geometry"].buffer(0)  # topology fix
    return gdf[["name", "geometry"]].reset_index(drop=True)


@st.cache_data(show_spinner=True, ttl=60*60*24)
def get_city_boundary_from_overpass(place_name: str, contact_email="contact@example.com"):
    gpd = _gpd()
    _, shape = _shapely()
    r = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params={"q": place_name, "format": "json", "polygon_geojson": 1, "limit": 1, "addressdetails": 0},
        headers={"User-Agent": f"ados-proto/1.0 ({contact_email})"},
        timeout=60,
    )
    r.raise_for_status()
    j = r.json()
    if not j:
        raise RuntimeError(f"OSM: no boundary found for '{place_name}'.")
    gj = j[0]["geojson"]
    gdf = _gpd().GeoDataFrame({"name":[place_name]}, geometry=[shape(gj)], crs=4326)
    gdf["geometry"] = gdf["geometry"].buffer(0)
    return gdf


@st.cache_data(show_spinner=True, ttl=60*60*24)
def get_city_boundary(place_name: str, state_fips: str):
    """
    Try TIGERweb first; if it fails, automatically fall back to OSM.
    """
    try:
        return get_city_boundary_from_tigerweb(place_name, state_fips)
    except Exception as e1:
        try:
            return get_city_boundary_from_overpass(place_name)
        except Exception as e2:
            raise RuntimeError(f"Boundary fetch failed. TIGERweb said: {e1}. Fallback OSM also failed: {e2}")

# ---------------------------
# TIGERweb tracts (preferred) + fallback ZIP
# ---------------------------
@st.cache_data(show_spinner=True, ttl=60*60*24)
def get_pg_tract_geometries_tigerweb(state_fips="24", county_fips="033"):
    """
    Tracts from TIGERweb (ArcGIS REST, GeoJSON).
    Service: Tracts_Blocks/MapServer/8  (Census Tracts)
    """
    gpd = _gpd()
    base = "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/Tracts_Blocks/MapServer/8/query"
    state = str(state_fips).zfill(2)
    county = str(county_fips).zfill(3)

    # Try STATE/COUNTY or STATEFP/COUNTYFP
    where_clauses = [
        f"STATE='{state}' AND COUNTY='{county}'",
        f"STATEFP='{state}' AND COUNTYFP='{county}'",
    ]

    last_err = None
    for where in where_clauses:
        try:
            r = requests.get(
                base,
                params={
                    "where": where,
                    "outFields": "*",
                    "f": "geojson",
                    "outSR": 4326,
                    "returnGeometry": "true",
                },
                timeout=90,
            )
            r.raise_for_status()
            j = r.json()
            feats = (j.get("features") or [])
            if not feats:
                continue
            gdf = gpd.GeoDataFrame.from_features(feats, crs=4326)
            # Common field names: GEOID or TRACTCE/STATE/COUNTY
            if "GEOID" in gdf.columns:
                gdf["geoid_tract"] = gdf["GEOID"].astype(str).str.zfill(11)
            else:
                # Construct if needed
                if set(["STATE", "COUNTY", "TRACT"]).issubset(gdf.columns):
                    gdf["geoid_tract"] = (
                        gdf["STATE"].astype(str).str.zfill(2)
                        + gdf["COUNTY"].astype(str).str.zfill(3)
                        + gdf["TRACT"].astype(str).str.zfill(6)
                    )
                elif set(["STATEFP", "COUNTYFP", "TRACTCE"]).issubset(gdf.columns):
                    gdf["geoid_tract"] = (
                        gdf["STATEFP"].astype(str).str.zfill(2)
                        + gdf["COUNTYFP"].astype(str).str.zfill(3)
                        + gdf["TRACTCE"].astype(str).str.zfill(6)
                    )
                else:
                    raise RuntimeError("TIGERweb tracts missing expected ID fields.")
            gdf["geometry"] = gdf["geometry"].buffer(0)
            keep_cols = ["geoid_tract", "geometry"]
            if "NAMELSAD" in gdf.columns:
                keep_cols.append("NAMELSAD")
            return gdf[keep_cols].reset_index(drop=True)
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"TIGERweb tracts failed: {last_err}")

@st.cache_data(show_spinner=True, ttl=60*60*24)
def get_pg_tract_geometries_zip(state_fips="24", county_fips="033"):
    """
    Fallback to Cartographic Boundary ZIP if TIGERweb tracts fail.
    """
    gpd = _gpd()
    url = "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_24_tract_500k.zip"
    gdf = gpd.read_file(url).to_crs(4326)
    gdf = gdf[gdf["COUNTYFP"] == str(county_fips).zfill(3)].copy()
    gdf["geoid_tract"] = gdf["GEOID"]
    return gdf[["geoid_tract", "geometry", "NAMELSAD"]].reset_index(drop=True)

@st.cache_data(show_spinner=True, ttl=60*60*24)
def get_pg_tract_geometries(state_fips="24", county_fips="033"):
    try:
        return get_pg_tract_geometries_tigerweb(state_fips, county_fips)
    except Exception:
        # fallback to ZIP
        return get_pg_tract_geometries_zip(state_fips, county_fips)

# ---------------------------
# Census/ACS helpers
# ---------------------------
@st.cache_data(show_spinner=True, ttl=60*60*24)
def get_census_group_metadata(year:str, group:str="B05003B"):
    r = requests.get(f"https://api.census.gov/data/{year}/acs/acs5/groups/{group}.json", timeout=60)
    r.raise_for_status()
    return r.json()

@st.cache_data(show_spinner=True, ttl=60*60*24)
def get_census_tract_black_nativity(year: str, state_fips: str, county_fips: str, group: str = "B05003B"):
    meta = get_census_group_metadata(year, group)
    r = requests.get(
        f"https://api.census.gov/data/{year}/acs/acs5",
        params={"get": f"NAME,group({group})", "for":"tract:*", "in": f"state:{state_fips} county:{county_fips}"},
        timeout=120,
    )
    r.raise_for_status()
    raw = r.json()
    df = pd.DataFrame(raw[1:], columns=raw[0])

    varmeta = meta["variables"]
    native_cols  = [k for k,v in varmeta.items() if k.startswith(group+"_") and "!!Native" in v.get("label","") and k.endswith("E")]
    foreign_cols = [k for k,v in varmeta.items() if k.startswith(group+"_") and "!!Foreign born" in v.get("label","") and k.endswith("E")]
    total_black_col = f"{group}_001E"

    for c in set(native_cols + foreign_cols + [total_black_col]) & set(df.columns):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["native_black"]  = df[native_cols].sum(axis=1, skipna=True)
    df["foreign_black"] = df[foreign_cols].sum(axis=1, skipna=True)
    df["black_total"]   = pd.to_numeric(df.get(total_black_col, np.nan), errors="coerce")
    df["geoid_tract"]   = df["state"] + df["county"] + df["tract"]
    return df[["NAME","geoid_tract","native_black","foreign_black","black_total"]].copy()

def rate(numer, denom):
    numer = pd.to_numeric(numer, errors="coerce")
    denom = pd.to_numeric(denom, errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom>0, numer/denom, np.nan)

def subset_to_city_tracts(tract_df: pd.DataFrame, tracts_gdf, city_gdf):
    gpd = _gpd()
    g_city = city_gdf.to_crs(tracts_gdf.crs)
    inter = gpd.overlay(tracts_gdf, g_city[["geometry"]], how="intersection")
    ids = set(inter["geoid_tract"].unique().tolist())
    return tract_df[tract_df["geoid_tract"].isin(ids)].copy()

# ---------------------------
# Wealth & Health helpers
# ---------------------------
@st.cache_data(show_spinner=True, ttl=60*60*24)
def census_get_vars(year: str, state_fips: str, county_fips: str, var_list: list):
    r = requests.get(
        f"https://api.census.gov/data/{year}/acs/acs5",
        params={"get": ",".join(["NAME"] + var_list), "for": "tract:*", "in": f"state:{state_fips} county:{county_fips}"},
        timeout=120,
    )
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
    meta = get_census_group_metadata(year, group)
    hits = []
    for var, mv in meta["variables"].items():
        if not var.endswith("E"):
            continue
        lab = mv.get("label", "")
        if all(s in lab for s in label_contains):
            hits.append(var)
    return hits

@st.cache_data(show_spinner=True, ttl=60*60*24)
def get_cdc_places_tract():
    url = "https://chronicdata.cdc.gov/api/views/cwsq-ngmh/rows.csv?accessType=DOWNLOAD"  # 2023 GIS-friendly
    r = requests.get(url, timeout=180)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), low_memory=False)

    # GEOID extraction (tolerant)
    geoid = None
    for cand in ["TractFIPS", "TractFIPS10", "LocationID"]:
        if cand in df.columns:
            s = df[cand].astype(str).str.extract(r"(\d{11})", expand=False)
            if s.notna().any():
                geoid = s.str.zfill(11)
                break
    if geoid is None:
        for c in df.columns:
            if "fips" in c.lower():
                s = df[c].astype(str).str.extract(r"(\d{11})", expand=False)
                if s.notna().any():
                    geoid = s.str.zfill(11)
                    break

    if "Measure" in df.columns and "Data_Value" in df.columns:
        df_ins = df[df["Measure"].str.contains("lack of health insurance", case=False, na=False)].copy()
        df_ins.rename(columns={"Data_Value":"uninsured_18_64_pct"}, inplace=True)
    else:
        df_ins = pd.DataFrame(columns=["uninsured_18_64_pct"])

    if geoid is not None:
        df_ins["geoid_tract"] = geoid.loc[df_ins.index].values

    return df_ins.reindex(columns=["geoid_tract","uninsured_18_64_pct"]).dropna(how="all")

# ---------------------------
# UI — Safe, step-by-step
# ---------------------------
st.title("Greenbelt, MD • Reparations Context Prototype")
st.caption("ADOS-proxy (native-born Black residents) + wealth & health context")

# Step 1 — Boundary
if "city_gdf" not in st.session_state:
    st.session_state["city_gdf"] = None

st.subheader("Step 1 — Load city boundary")
if st.button("Fetch boundary now"):
    try:
        st.session_state["city_gdf"] = get_city_boundary(CITY_NAME, STATE_FIPS)
        st.success("Boundary loaded.")
    except Exception as e:
        st.error(f"Boundary fetch failed: {e}")

city_gdf = st.session_state["city_gdf"]

if city_gdf is not None:
    # Show a simple centroid on the map to avoid serializing the polygon
    try:
        gpd = _gpd()
        cent = city_gdf.to_crs(4326).copy()
        cent["latitude"] = cent.geometry.centroid.y
        cent["longitude"] = cent.geometry.centroid.x
        st.map(cent[["latitude","longitude"]], size=60)
    except Exception:
        st.dataframe(city_gdf.drop(columns=["geometry"]))

st.markdown("---")

# Step 2 — ADOS proxy
st.subheader("Step 2 — ADOS proxy from ACS (B05003B)")
if "ados_df" not in st.session_state:
    st.session_state["ados_df"] = None
if "tracts_gdf" not in st.session_state:
    st.session_state["tracts_gdf"] = None
if "ados_map" not in st.session_state:
    st.session_state["ados_map"] = None

if st.button("Load ACS & compute ADOS proxy"):
    if city_gdf is None:
        st.error("Load the boundary first (Step 1).")
    else:
        with st.spinner("Downloading tract geometries & ACS…"):
            st.session_state["tracts_gdf"] = get_pg_tract_geometries(STATE_FIPS, COUNTY_FIPS)
            ados_df = get_census_tract_black_nativity(ACS_YEAR, STATE_FIPS, COUNTY_FIPS, "B05003B")

        tracts = st.session_state["tracts_gdf"]
        if tracts is None or ados_df.empty:
            st.error("Tracts or ACS failed to load.")
        else:
            # Limit to city tracts
            ados_city = subset_to_city_tracts(ados_df, tracts, city_gdf)
            # City totals
            city_native  = ados_city["native_black"].sum()
            city_foreign = ados_city["foreign_black"].sum()
            city_black   = ados_city["black_total"].sum()
            ados_share   = (city_native / city_black) if city_black > 0 else np.nan

            k1, k2, k3 = st.columns(3)
            k1.metric("U.S.-born Black residents (proxy)", f"{int(city_native):,}")
            k2.metric("Foreign-born Black residents", f"{int(city_foreign):,}")
            k3.metric("ADOS proxy (share of Black pop.)", f"{ados_share:.1%}" if pd.notna(ados_share) else "—")

            # Store for later charts
            st.session_state["ados_df"] = ados_city
            try:
                gpd = _gpd()
                ados_map = gpd.GeoDataFrame(
                    ados_city.merge(tracts[["geoid_tract","geometry"]], on="geoid_tract", how="left"),
                    geometry="geometry", crs=tracts.crs
                )
                ados_map["ados_proxy_share"] = rate(ados_map["native_black"], ados_map["black_total"])
                st.session_state["ados_map"] = ados_map
            except Exception as e:
                st.warning(f"Map build issue: {e}")
                st.session_state["ados_map"] = None

ados_map = st.session_state["ados_map"]
if ados_map is not None and not ados_map.empty:
    try:
        import altair as alt
        plot_df = (ados_map
                   .assign(tract_short=lambda d: d["geoid_tract"].str[-6:])
                   .dropna(subset=["ados_proxy_share"]))
        chart = alt.Chart(plot_df).mark_bar().encode(
            x=alt.X("tract_short:N", title="Census tract (short id)"),
            y=alt.Y("ados_proxy_share:Q", title="Share of U.S.-born Black", axis=alt.Axis(format='%')),
            tooltip=[
                alt.Tooltip("geoid_tract:N", title="Tract GEOID"),
                alt.Tooltip("native_black:Q", title="U.S.-born Black", format=",.0f"),
                alt.Tooltip("black_total:Q", title="Black total", format=",.0f"),
                alt.Tooltip("ados_proxy_share:Q", title="ADOS proxy", format=".1%"),
            ],
        ).properties(height=280)
        st.altair_chart(chart, use_container_width=True)
    except Exception as e:
        st.warning(f"Chart error: {e}")
        st.dataframe(ados_map[["geoid_tract","native_black","black_total"]].head(20))

st.markdown("---")

# Step 3 — Wealth & Health context (fast subset)
st.subheader("Step 3 — Wealth & Health (tract → city)")

if "context_df" not in st.session_state:
    st.session_state["context_df"] = None

if st.button("Load Wealth & Health context"):
    if city_gdf is None:
        st.error("Load the boundary first (Step 1).")
    else:
        with st.spinner("Fetching ACS context + CDC PLACES…"):
            tracts_gdf = st.session_state.get("tracts_gdf") or get_pg_tract_geometries(STATE_FIPS, COUNTY_FIPS)
            core = census_get_vars(
                ACS_YEAR, STATE_FIPS, COUNTY_FIPS,
                [
                    "B19013_001E", "B25077_001E", "B25064_001E",  # income, home value, rent
                    "B23025_003E", "B23025_005E",                 # labor force, unemployed
                    "B25003B_001E", "B25003B_002E",               # Black total occ. units, owner-occupied
                    "B19053_001E", "B19053_002E",                 # hh total, hh w/ interest/dividends/rent
                ],
            )
            # Poverty (Black alone)
            pov_vars = census_sum_by_label(ACS_YEAR, "B17020B", ["Black or African American alone", "Below"])
            if pov_vars:
                pov = census_get_vars(ACS_YEAR, STATE_FIPS, COUNTY_FIPS, pov_vars + ["B17020B_001E"])
                pov["pov_black_count"] = pov[pov_vars].sum(axis=1, skipna=True)
                pov.rename(columns={"B17020B_001E":"pov_black_universe"}, inplace=True)
                core = core.merge(pov[["geoid_tract","pov_black_count","pov_black_universe"]], on="geoid_tract", how="left")
            # PLACES uninsured
            places = get_cdc_places_tract()
            if "geoid_tract" in places.columns:
                core = core.merge(places, on="geoid_tract", how="left")

            # Indicators
            core["median_income"]      = core["B19013_001E"]
            core["median_home_value"]  = core["B25077_001E"]
            core["median_gross_rent"]  = core["B25064_001E"]
            core["unemp_rate"]         = rate(core["B23025_005E"], core["B23025_003E"])
            core["black_owner_rate"]   = rate(core["B25003B_002E"], core["B25003B_001E"])
            core["asset_income_rate"]  = rate(core["B19053_002E"], core["B19053_001E"])
            core["black_poverty_rate"] = rate(core["pov_black_count"], core["pov_black_universe"])

            # Limit to city
            st.session_state["context_df"] = subset_to_city_tracts(core, tracts_gdf, city_gdf)

context = st.session_state["context_df"]
if context is not None and not context.empty:
    def _fmt_pct(x):
        return "—" if pd.isna(x) else f"{x*100:,.1f}%"
    def _fmt_usd(x):
        return "—" if pd.isna(x) else f"${x:,.0f}"
    def _fmt_ratio(x):
        return "—" if pd.isna(x) else f"{x:,.1f}×"

    med_income = np.nanmedian(context["median_income"])
    med_home   = np.nanmedian(context["median_home_value"])
    med_rent   = np.nanmedian(context["median_gross_rent"])
    unemp      = float(np.nanmean(context["unemp_rate"]))
    owner_b    = float(np.nanmean(context["black_owner_rate"]))
    asset_p    = float(np.nanmean(context["asset_income_rate"]))
    black_pov  = float(np.nanmean(context["black_poverty_rate"]))
    uninsured  = float(np.nanmean(pd.to_numeric(context.get("uninsured_18_64_pct", np.nan), errors="coerce")))  # 0–100

    price_to_income = (med_home / med_income) if (pd.notna(med_home) and pd.notna(med_income) and med_income>0) else np.nan
    rent_burden     = (med_rent / (med_income/12.0)) if (pd.notna(med_rent) and pd.notna(med_income) and med_income>0) else np.nan

    st.markdown("### BLUF — 30-second headline signals")
    c1, c2, c3 = st.columns(3)
    c1.metric("Wealth participation (households w/ asset income)", _fmt_pct(asset_p))
    c2.metric("Home price-to-income", _fmt_ratio(price_to_income))
    c3.metric("Typical rent burden", _fmt_pct(rent_burden))

    c4, c5, c6 = st.columns(3)
    c4.metric("Black homeownership", _fmt_pct(owner_b))
    c5.metric("Unemployment rate", _fmt_pct(unemp))
    c6.metric("Adults 18–64 without health insurance (PLACES)", "—" if pd.isna(uninsured) else f"{uninsured:,.1f}%")

    st.caption("Notes: rent burden ≥30% is traditionally considered 'cost-burdened'. "
               "Wealth participation uses interest/dividends/rental income as a proxy for asset ownership.")

st.markdown("---")
st.caption("Prototype — Boundaries: TIGERweb (GeoJSON) • Demographics: Census ACS • Health: CDC PLACES")
