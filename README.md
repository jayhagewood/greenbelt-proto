# Greenbelt ADOS Data Prototype

This is a **72-hour prototype dashboard** built in [Streamlit](https://streamlit.io) to help visualize **ADOS-related metrics** for Greenbelt, Maryland.  

The app combines **U.S. Census ACS**, **CDC PLACES**, and **Prince George’s County open data** to provide quick, BLUF-style insights for leadership.

---

## 🚀 Features
- **ADOS Proxy**: Native-born Black population share (ACS Table B05003B).
- **Wealth & Health Context**:
  - Median household income
  - Black homeownership rate
  - Black poverty rate (if tract sample supports it)
  - Median home value & gross rent
  - Asset-income participation (interest, dividends, rental income)
  - Unemployment & uninsured adults (CDC PLACES)
  - Mobility constraints (households without vehicles)
- **Community Vulnerability Index** (0–100): composite signal for leadership.

---

## 📊 Data Sources
- **U.S. Census Bureau ACS 5-year** (via Census API)  
- **CDC PLACES** (tract-level health outcomes)  
- **OpenStreetMap (Nominatim)** — city boundaries  
- **Prince George’s County Open Data Portal** — 311, code enforcement, crime  

---

## 💡 Notes
- This is a **prototype**: results are based on tract-level estimates and proxies.  
- Some metrics (e.g., Black poverty rate) may not appear if tract samples are too small.  
- Future **Phase 2** will integrate **HUD CHAS**, **PUMS microdata**, and potentially **Color of Wealth**–style KPIs (net worth, debt ratios, inheritance).  

---

## ▶️ Run locally
```bash
conda create -n city python=3.10 -y
conda activate city
pip install -r requirements.txt
streamlit run app.py

🌐 Live Demo

Deployed on Streamlit Community Cloud

👉 App link: (fill this in once deployed)
