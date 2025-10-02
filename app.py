# app.py
# Greenbelt, MD • Reparations Context Prototype (Streamlit-cloud friendly)
# - City boundary: Census TIGERweb (GeoJSON) with optional OSM fallback
# - Tracts: TIGERweb (GeoJSON) by state/county (no remote ZIP reads)
# - ADOS proxy (ACS B05003B) + Wealth/Health BLUF metrics
# - Lazy geopandas/shapely imports to improve reliability on hosted runtimes

import io
import time
import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="ADOS / Wealth Gap Prototype", layout="wide")

# ---------------------------
# Lazy geo imports (avoid boot issues on Cloud)
# ---------------------------
def _gpd():
    import geopandas as _g
    return _g

def _shapely():
    from shapely.geometry import shape as _shape
    return _shape


# ---------------------------
# Sidebar configuration
# ---------------------------
st.sidebar.header("Configuration")

CITY_NAME   = st.sidebar.text_input("City (e.g., 'Greenbelt, Maryland, USA')", "Greenbelt, Maryland, USA")
STATE_FIPS  = st.sidebar.text_input("State FIPS (MD=24)", "24")
COUNTY_FIPS = st.sidebar.text_input("County FIPS (PG=033)", "033")
ACS_YEAR    = st.sidebar.selectbox("ACS 5-year vintage", ["2023", "2022", "2021"], index=0)

ALLOW_OSM_FALLBACK = st.sidebar.checkbox(
    "If TIGERweb fails, allow OSM fallback", value=True,
    help="TIGERweb is preferred for consistency. OSM works but can rate-limit."
)

INCLUDE_PLACES = st.sidebar.checkbox(
    "Include CDC PLACES uninsured metric (slower)", value=False,
    help="Downloads a large CSV; leave off unless needed."
)

st.sidebar.markdown("---")
st.sidebar.caption("Tip: Keep the app step-driven (buttons) to avoid long cold starts on Streamlit Cloud.")


# ---------------------------
# Boundary & Tracts (TIGERweb / OSM) — no ZIP shapefiles
# ---------------------------
@st.cache_data(show_spinner=True, ttl=60*60*24)
def get_city_boundary_from_tigerweb(place_name: str, state_fips: str):
    """Query Census TIGERweb Places layer (GeoJSON) for a city boundary."""
    gpd = _gpd()

    base = (
        "https://tigerweb.geo.census.gov/arcgis/rest/services/"
        "TIGERweb/Places_CouSub_ConCity_SubMCD/MapServer/2/query"
    )

    state = str(state_fips).zfill(2)
    token = place_name.split(",")[0].strip()
    token = token.replace(" city", "").replace(" town", "").replace(" village", "")
    token = token.replace("'", "''")

    params = {
        "where": f"STATE='{state}' AND UPPER(NAME)=UPPER('{token}')",
        "outFields": "STATE,NAME,NAMELSAD",
        "f": "geojson",
        "outSR": 4326,
        "returnGeometry": "true",
    }
    r = requests.get(base, params=params, timeout=60)
    r.raise_for_status()
    j = r.json()
    if not j.get("features"):
        # try prefix like
        params["where"] = f"STATE='{state}' AND UPPER(NAME) LIKE UPPER('{token}%')"
        r = requests.get(base, params=params, timeout=60)
        r.raise_for_status()
        j = r.json()

    if not j.get("features"):
        raise RuntimeError(f"TIGERweb: no place found for '{place_name}' in state {state_fips}.")

    gdf = gpd.GeoDataFrame.from_features(j["features"], crs="EPSG:4326")
    gdf["score"] = 0
    if "NAMELSAD" in gdf.columns:
        gdf.loc[gdf["NAMELSAD"].str.contains("city", case=False, na=False), "score"] += 2
    gdf.loc[gdf["NAME"].str.upper() == token.upper(), "score"] += 3
    gdf = gdf.sort_values(["score", "NAME"], ascending=[False, True]).head(1).copy()
    gdf["name"] = place_name
    gdf["geometry"] = gdf["geometry"].buffer(0)  # topology fix
    return gdf[["name", "geometry"]].reset_index(drop=True)


@st.cache_data(show_spinner=True, ttl=60*60*24)
def get_city_boundary_from_overpass(place_name: str, contact_email="contact@example.com"):
    """OSM/Nominatim polygon (GeoJSON). Used only as a fallback."""
    gpd = _gpd()
    shape = _shapely()
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
    gdf = _gpd().GeoDataFrame({"name": [place_name]}, geometry=[shape(gj)], crs=4326)
    gdf["geometry"] = gdf["geometry"].buffer(0)
    return gdf


@st.cache_data(show_spinner=True, ttl=60*60*24)
def get_city_boundary(place_name: str, state_fips: str, allow_osm_fallback: bool):
    try:
        return get_city_boundary_from_tigerweb(place_name, state_fips), "TIGERweb"
    except Exception as e:
        if not allow_osm_fallback:
            raise
        # fallback to OSM
        gdf = get_city_boundary_from_overpass(place_name)
        return gdf, "OSM"


@st.cache_data(show_spinner=True, ttl=60*60*24)
def get_tracts_from_tigerweb(state_fips: str, county_fips: str):
    """
    Tracts via TIGERweb 2020 Tracts layer (GeoJSON). No shapefile ZIPs.
    MapServer 2 = 2020 Census Tracts.
    """
    gpd = _gpd()
    base = "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/Tracts_Blocks/MapServer/2/query"
    params = {
        "where": f"STATE='{str(state_fips).zfill(2)}' AND COUNTY='{str(county_fips).zfill(3)}'",
        "outFields": "STATE,COUNTY,TRACT,GEOID,NAMELSAD",
        "f": "geojson",
        "outSR": 4326,
        "returnGeometry": "true",
    }
    r = requests.get(base, params=params, timeout=120)
    r.raise_for_status()
    j = r.json()
    if not j.get("features"):
        raise RuntimeError("TIGERweb tracts query returned no features.")
    gdf = gpd.GeoDataFrame.from_features(j["features"], crs="EPSG:4326")
    if "GEOID" in gdf.columns:
        gdf["geoid_tract"] = gdf["GEOID"].astype(str).str.zfill(11)
    else:
        gdf["geoid_tract"] = gdf["STATE"].astype(str).str.zfill(2) + \
                             gdf["COUNTY"].astype(str).str.zfill(3) + \
                             gdf["TRACT"].astype(str).str.zfill(6)
    return gdf[["geoid_tract", "geometry", "NAMELSAD"]].reset_index(drop=True)


def rate(numer, denom):
    numer = pd.to_numeric(numer, errors="coerce")
    denom = pd.to_numeric(denom, errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom > 0, numer / denom, np.nan)


def subset_to_city_tracts(tract_df: pd.DataFrame, tracts_gdf, city_gdf):
    gpd = _gpd()
    inter = gpd.overlay(tracts_gdf, city_gdf[["geometry"]], how="intersection")
    ids = set(inter["geoid_tract"].unique().tolist())
    return tract_df[tract_df["geoid_tract"].isin(ids)].copy(), len(ids)


# ---------------------------
# Census / ACS helpers
# ---------------------------
@st.cache_data(show_spinner=True, ttl=60*60*24)
def get_census_group_metadata(year: str, group: str = "B05003B"):
    r = requests.get(f"https://api.census.gov/data/{year}/acs/acs5/groups/{group}.json", timeout=60)
    r.raise_for_status()
    return r.json()


@st.cache_data(show_spinner=True, ttl=60*60*24)
def get_census_tract_black_nativity(year: str, state_fips: str, county_fips: str, group: str = "B05003B"):
    """Return tract-level native-born vs foreign-born Black counts + totals."""
    meta = get_census_group_metadata(year, group)
    r = requests.get(
        f"https://api.census.gov/data/{year}/acs/acs5",
        params={"get": f"NAME,group({group})", "for": "tract:*", "in": f"state:{state_fips} county:{county_fips}"},
        timeout=120,
    )
    r.raise_for_status()
    raw = r.json()
    df = pd.DataFrame(raw[1:], columns=raw[0])

    varmeta = meta["variables"]
    native_cols  = [k for k, v in varmeta.items() if k.startswith(group + "_") and "!!Native" in v.get("label", "") and k.endswith("E")]
    foreign_cols = [k for k, v in varmeta.items() if k.startswith(group + "_") and "!!Foreign born" in v.get("label", "") and k.endswith("E")]
    total_black_col = f"{group}_001E"

    for c in set(native_cols + foreign_cols + [total_black_col]) & set(df.columns):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["native_black"]  = df[native_cols].sum(axis=1, skipna=True)
    df["foreign_black"] = df[foreign_cols].sum(axis=1, skipna=True)
    df["black_total"]   = pd.to_numeric(df.get(total_black_col, np.nan), errors="coerce")
    df["geoid_tract"]   = df["state"] + df["county"] + df["tract"]
    return df[["NAME", "geoid_tract", "native_black", "foreign_black", "black_total"]].copy()


@st.cache_data(show_spinner=True, ttl=60*60*24)
def census_get_vars(year: str, state_fips: str, county_fips: str, var_list: list):
    """Fetch arbitrary ACS vars for all tracts in a county."""
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
    """Return ACS variable names in a group whose labels contain ALL substrings."""
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
def get_cdc_places_uninsured():
    """CDC PLACES tract GIS-friendly CSV (filter to uninsured 18–64)."""
    url = "https://chronicdata.cdc.gov/api/views/cwsq-ngmh/rows.csv?accessType=DOWNLOAD"  # 2023
    r = requests.get(url, timeout=180)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), low_memory=False)

    # Try standard GEOID columns
    geoid = None
    for cand in ["TractFIPS", "TractFIPS10", "LocationID"]:
        if cand in df.columns:
            s = df[cand].astype(str).str.extract(r"(\d{11})", expand=False)
            if s.notna().any():
                geoid = s.str.zfill(11)
                break
    if geoid is None:
        # fall back: search any fips-like column
        for c in df.columns:
            if "fips" in c.lower():
                s = df[c].astype(str).str.extract(r"(\d{11})", expand=False)
                if s.notna().any():
                    geoid = s.str.zfill(11)
                    break

    if "Measure" in df.columns and "Data_Value" in df.columns:
        df_ins = df[df["Measure"].str.contains("lack of health insurance", case=False, na=False)].copy()
        df_ins.rename(columns={"Data_Value": "uninsured_18_64_pct"}, inplace=True)
    else:
        df_ins = pd.DataFrame(columns=["uninsured_18_64_pct"])

    if geoid is not None:
        df_ins["geoid_tract"] = geoid.loc[df_ins.index].values

    return df_ins.reindex(columns=["geoid_tract", "uninsured_18_64_pct"]).dropna(how="all")


# ---------------------------
# UI — Step flow
# ---------------------------
st.title("Greenbelt, MD • Reparations Context Prototype")
st.caption("ADOS-proxy (native-born Black residents) + wealth & health context")

# Keep state
ss = st.session_state
for k in ["city_gdf", "tracts_gdf", "intersect_count", "ados_city", "ados_map", "context_df", "boundary_source"]:
    if k not in ss:
        ss[k] = None


# Step 1 — Boundary & tracts
st.subheader("Step 1 — Load city boundary & intersecting tracts")
if st.button("Fetch boundary & tracts", type="primary"):
    try:
        city_gdf, src = get_city_boundary(CITY_NAME, STATE_FIPS, ALLOW_OSM_FALLBACK)
        ss["city_gdf"] = city_gdf
        ss["boundary_source"] = src
    except Exception as e:
        st.error(f"Boundary error: {e}")
        ss["city_gdf"] = None

    if ss["city_gdf"] is not None:
        try:
            tracts_gdf = get_tracts_from_tigerweb(STATE_FIPS, COUNTY_FIPS)
            ss["tracts_gdf"] = tracts_gdf
            # intersect once and reuse
            _ = _gpd()
            adf = pd.DataFrame({"geoid_tract": tracts_gdf["geoid_tract"]})
            _, count = subset_to_city_tracts(adf, tracts_gdf, ss["city_gdf"])
            ss["intersect_count"] = count
            st.success(f"Found {count} intersecting tracts in county {COUNTY_FIPS}.")
        except Exception as e:
            st.error(f"Tracts error: {e}")
            ss["tracts_gdf"] = None

if ss["city_gdf"] is not None:
    try:
        gpd = _gpd()
        cent = ss["city_gdf"].to_crs(4326).copy()
        cent["latitude"] = cent.geometry.centroid.y
        cent["longitude"] = cent.geometry.centroid.x
        st.map(cent[["latitude", "longitude"]], size=60)
        st.caption(f"Boundary source: {ss.get('boundary_source', '—')} • NAME: {CITY_NAME}")
    except Exception:
        st.caption(f"Boundary source: {ss.get('boundary_source', '—')}")

st.markdown("---")


# Step 2 — ADOS proxy
st.subheader("Step 2 — ADOS proxy from ACS (B05003B)")
if st.button("Compute ADOS proxy"):
    if ss["city_gdf"] is None or ss["tracts_gdf"] is None:
        st.error("Please complete Step 1 first.")
    else:
        with st.spinner("Fetching ACS and computing…"):
            acs_df = get_census_tract_black_nativity(ACS_YEAR, STATE_FIPS, COUNTY_FIPS, "B05003B")
            # limit to city tracts
            ados_city, _ = subset_to_city_tracts(acs_df, ss["tracts_gdf"], ss["city_gdf"])
            ss["ados_city"] = ados_city

            # City totals
            city_native  = ados_city["native_black"].sum()
            city_foreign = ados_city["foreign_black"].sum()
            city_black   = ados_city["black_total"].sum()
            ados_share   = (city_native / city_black) if city_black > 0 else np.nan

            c1, c2, c3 = st.columns(3)
            c1.metric("U.S.-born Black residents (proxy)", f"{int(city_native):,}")
            c2.metric("Foreign-born Black residents", f"{int(city_foreign):,}")
            c3.metric("ADOS proxy (share of Black pop.)", f"{ados_share:.1%}" if pd.notna(ados_share) else "—")

            # build tract chart data (optional)
            try:
                gpd = _gpd()
                m = _gpd().GeoDataFrame(
                    ados_city.merge(ss["tracts_gdf"][["geoid_tract", "geometry"]], on="geoid_tract", how="left"),
                    geometry="geometry",
                    crs=ss["tracts_gdf"].crs,
                )
                m["ados_proxy_share"] = rate(m["native_black"], m["black_total"])
                ss["ados_map"] = m
            except Exception as e:
                st.warning(f"Map build issue: {e}")
                ss["ados_map"] = None

if ss["ados_city"] is not None and ss["ados_map"] is not None:
    try:
        import altair as alt
        plot_df = (ss["ados_map"]
                   .assign(tract_short=lambda d: d["geoid_tract"].str[-6:])
                   .dropna(subset=["ados_proxy_share"]))
        chart = alt.Chart(plot_df).mark_bar().encode(
            x=alt.X("tract_short:N", title="Census tract (short id)"),
            y=alt.Y("ados_proxy_share:Q", title="Share of U.S.-born Black", axis=alt.Axis(format='%')),
            tooltip=[
                alt.Tooltip("geoid_tract:N", title="Tract"),
                alt.Tooltip("native_black:Q", title="U.S.-born Black", format=",.0f"),
                alt.Tooltip("black_total:Q",  title="Black total",   format=",.0f"),
                alt.Tooltip("ados_proxy_share:Q", title="ADOS proxy", format=".1%"),
            ],
        ).properties(height=280)
        st.altair_chart(chart, use_container_width=True)
    except Exception as e:
        st.warning(f"Chart issue: {e}")
        st.dataframe(ss["ados_city"][["geoid_tract", "native_black", "black_total"]].head(20))

st.markdown("---")


# Step 3 — Wealth & Health context (BLUF)
st.subheader("Step 3 — Wealth & Health context")

if st.button("Compute Wealth & Health"):
    if ss["city_gdf"] is None or ss["tracts_gdf"] is None:
        st.error("Please complete Step 1 first.")
    else:
        with st.spinner("Fetching ACS context" + (" + CDC PLACES…" if INCLUDE_PLACES else "…")):
            core = census_get_vars(
                ACS_YEAR, STATE_FIPS, COUNTY_FIPS,
                [
                    "B19013_001E",  # median household income
                    "B25077_001E",  # median home value
                    "B25064_001E",  # median gross rent
                    "B23025_003E",  # labor force
                    "B23025_005E",  # unemployed
                    "B25003B_001E", # Black units total
                    "B25003B_002E", # Black owner-occupied
                    "B19053_001E",  # households (asset-income universe)
                    "B19053_002E",  # households with interest/dividends/rental income
                ],
            )

            # Poverty (Black alone)
            pov_vars = census_sum_by_label(ACS_YEAR, "B17020B", ["Black or African American alone", "Below"])
            if pov_vars:
                pov = census_get_vars(ACS_YEAR, STATE_FIPS, COUNTY_FIPS, pov_vars + ["B17020B_001E"])
                pov["pov_black_count"] = pov[pov_vars].sum(axis=1, skipna=True)
                pov.rename(columns={"B17020B_001E": "pov_black_universe"}, inplace=True)
                core = core.merge(pov[["geoid_tract", "pov_black_count", "pov_black_universe"]], on="geoid_tract", how="left")

            # CDC PLACES uninsured (optional)
            if INCLUDE_PLACES:
                try:
                    places = get_cdc_places_uninsured()
                    if "geoid_tract" in places.columns:
                        core = core.merge(places, on="geoid_tract", how="left")
                except Exception as e:
                    st.warning(f"PLACES fetch skipped: {e}")

            # Indicators
            core["median_income"]      = core["B19013_001E"]
            core["median_home_value"]  = core["B25077_001E"]
            core["median_gross_rent"]  = core["B25064_001E"]
            core["unemp_rate"]         = rate(core["B23025_005E"], core["B23025_003E"])
            core["black_owner_rate"]   = rate(core["B25003B_002E"], core["B25003B_001E"])
            core["asset_income_rate"]  = rate(core["B19053_002E"], core["B19053_001E"])
            if "pov_black_count" in core.columns:
                core["black_poverty_rate"] = rate(core["pov_black_count"], core["pov_black_universe"])

            ss["context_df"], _ = subset_to_city_tracts(core, ss["tracts_gdf"], ss["city_gdf"])

# BLUF cards
ctx = ss["context_df"]
if ctx is not None and not ctx.empty:
    def _fmt_pct(x):
        return "—" if pd.isna(x) else f"{x*100:,.1f}%"
    def _fmt_usd(x):
        return "—" if pd.isna(x) else f"${x:,.0f}"
    def _fmt_ratio(x):
        return "—" if pd.isna(x) else f"{x:,.1f}×"

    med_income = float(np.nanmedian(ctx["median_income"]))
    med_home   = float(np.nanmedian(ctx["median_home_value"]))
    med_rent   = float(np.nanmedian(ctx["median_gross_rent"]))
    unemp      = float(np.nanmean(ctx["unemp_rate"]))
    owner_b    = float(np.nanmean(ctx["black_owner_rate"]))
    asset_p    = float(np.nanmean(ctx["asset_income_rate"]))
    black_pov  = float(np.nanmean(ctx.get("black_poverty_rate")))
    uninsured  = float(np.nanmean(pd.to_numeric(ctx.get("uninsured_18_64_pct", np.nan), errors="coerce"))) if "uninsured_18_64_pct" in ctx.columns else np.nan

    price_to_income = (med_home / med_income) if (med_income and med_income > 0) else np.nan
    rent_burden     = (med_rent / (med_income/12.0)) if (med_income and med_income > 0) else np.nan

    st.markdown("### BLUF — 30-second headline signals")
    r1c1, r1c2, r1c3 = st.columns(3)
    r1c1.metric("Wealth participation\n(households w/ asset income)", _fmt_pct(asset_p))
    r1c2.metric("Home price-to-income", _fmt_ratio(price_to_income))
    r1c3.metric("Typical rent burden", _fmt_pct(rent_burden))

    r2c1, r2c2, r2c3 = st.columns(3)
    r2c1.metric("Black homeownership", _fmt_pct(owner_b))
    r2c2.metric("Unemployment rate", _fmt_pct(unemp))
    r2c3.metric("Adults 18–64 without health insurance", "—" if pd.isna(uninsured) else f"{uninsured:,.1f}%")

    st.caption(
        "Notes: ‘ADOS proxy’ = share of **U.S.-born Black residents** (ACS table B05003B). "
        "Wealth participation uses presence of interest/dividends/rental income as a proxy for asset ownership. "
        "Rent burden ≥30% is traditionally considered cost-burdened."
    )

st.markdown("---")
st.caption("Sources: Census TIGERweb (boundaries, tracts), Census ACS 5-year (B05003B, etc.), CDC PLACES (optional).")ALLOW_OSM_FALLBACK = st.sidebar.checkbox(
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
