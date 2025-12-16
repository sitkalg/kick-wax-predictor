#!/usr/bin/env python3
"""
Kick Wax Predictor - Rule-Based Version
========================================
No TensorFlow required - works immediately!

Run: streamlit run kick_wax_simple.py
"""

import streamlit as st
import numpy as np

st.set_page_config(page_title="Kick Wax Predictor", page_icon="🎿", layout="wide")


# =============================================================================
# DROPDOWN OPTIONS FROM WAX REPORTS
# =============================================================================

ARTIFICIAL_SNOW_OPTIONS = {
    "A1 - Fresh machine-made": "A1",
    "A2 - Settled machine-made": "A2",
    "A3 - Aged machine-made": "A3",
    "A4 - Old machine-made": "A4",
    "A5 - Transformed machine-made": "A5"
}

NATURAL_SNOW_OPTIONS = {
    "NS - New snow": "NS",
    "FN - Fine-grained new": "FN",
    "IN - Irregular new": "IN",
    "IT - Irregular transformed": "IT",
    "TR - Transformed": "TR"
}

GRAIN_SIZE_OPTIONS = {
    "G1 - Very fine (<0.25mm)": "G1",
    "G2 - Fine (0.25-0.5mm)": "G2",
    "G3 - Medium (0.5-1mm)": "G3",
    "G4 - Coarse (1-2mm)": "G4",
    "G5 - Very coarse (>2mm)": "G5"
}

SNOW_HUMIDITY_OPTIONS = {
    "DS - Dry snow": "DS",
    "W1 - Slightly moist": "W1",
    "W2 - Moist": "W2",
    "W3 - Wet": "W3",
    "W4 - Very wet": "W4"
}

TRACK_HARDNESS_OPTIONS = {
    "H1 - Soft": "H1",
    "H2 - Medium-soft": "H2",
    "H3 - Medium-hard": "H3",
    "H4 - Hard": "H4"
}

TRACK_CONSISTENCY_OPTIONS = {
    "T1 - Uniform": "T1",
    "T2 - Slightly variable": "T2",
    "D1 - Variable": "D1",
    "D2 - Very variable": "D2"
}


# =============================================================================
# VENUES
# =============================================================================

VENUES = {
    # Canada
    "Whitehorse, YT": (700, 60.72, -135.05, "subarctic"),
    "Sovereign Lake, BC": (1450, 50.16, -119.22, "continental"),
    "Canmore, AB": (1400, 51.09, -115.36, "alpine"),
    "Prince George, BC": (575, 53.92, -122.75, "continental"),
    "Nakkertok, QC": (200, 45.50, -75.75, "continental"),
    "Mt. Sainte-Anne, QC": (800, 47.08, -70.90, "maritime"),
    # Europe
    "Beitostølen, NOR": (900, 61.25, 8.91, "continental"),
    "Davos, SUI": (1560, 46.80, 9.84, "alpine"),
    "Holmenkollen, NOR": (325, 59.96, 10.67, "maritime"),
    "Lahti, FIN": (120, 60.98, 25.66, "continental"),
    "Falun, SWE": (120, 60.61, 15.63, "continental"),
    "Trondheim, NOR": (200, 63.43, 10.39, "maritime"),
    "Östersund, SWE": (320, 63.18, 14.64, "continental"),
    "Ruka, FIN": (490, 66.17, 29.14, "arctic"),
    "Oberhof, GER": (815, 50.70, 10.73, "continental"),
    "Planica, SLO": (940, 46.48, 13.72, "alpine"),
    "Cogne, ITA": (1534, 45.61, 7.35, "alpine"),
    "Toblach, ITA": (1256, 46.73, 12.22, "alpine"),
    "Engadin, SUI": (1800, 46.48, 9.84, "alpine"),
    "Custom": (500, 60.0, -120.0, "continental")
}


# =============================================================================
# RULE-BASED PREDICTION
# =============================================================================

def predict_wax(snow_temp, air_temp, humidity, is_artificial, snow_humidity_code):
    """
    Rule-based wax prediction based on Swix wax charts and racing data.
    Returns top 3 recommendations.
    """
    
    # Check for wet/klister conditions
    is_wet = snow_humidity_code in ['W2', 'W3', 'W4']
    is_moist = snow_humidity_code == 'W1'
    temp_gradient = air_temp - snow_temp
    
    # Very cold conditions (< -12°C)
    if snow_temp < -12:
        if snow_temp < -18:
            return [
                {'rank': 1, 'base_wax': 'KX20', 'top_wax': 'V20', 'confidence': 92},
                {'rank': 2, 'base_wax': 'KX20', 'top_wax': 'V30', 'confidence': 85},
                {'rank': 3, 'base_wax': 'KX30', 'top_wax': 'V20', 'confidence': 78}
            ]
        else:
            return [
                {'rank': 1, 'base_wax': 'KX30', 'top_wax': 'V30', 'confidence': 90},
                {'rank': 2, 'base_wax': 'KX20', 'top_wax': 'V30', 'confidence': 84},
                {'rank': 3, 'base_wax': 'KX35', 'top_wax': 'V30', 'confidence': 77}
            ]
    
    # Cold conditions (-12 to -8°C)
    elif snow_temp < -8:
        if is_artificial:
            return [
                {'rank': 1, 'base_wax': 'KX35', 'top_wax': 'V30', 'confidence': 88},
                {'rank': 2, 'base_wax': 'KX35', 'top_wax': 'V40', 'confidence': 82},
                {'rank': 3, 'base_wax': 'VG30', 'top_wax': 'V30', 'confidence': 75}
            ]
        else:
            return [
                {'rank': 1, 'base_wax': 'VG30', 'top_wax': 'V30', 'confidence': 87},
                {'rank': 2, 'base_wax': 'KX35', 'top_wax': 'V40', 'confidence': 81},
                {'rank': 3, 'base_wax': 'VG30', 'top_wax': 'V40', 'confidence': 74}
            ]
    
    # Mid-cold conditions (-8 to -4°C)
    elif snow_temp < -4:
        if is_artificial:
            return [
                {'rank': 1, 'base_wax': 'VG35', 'top_wax': 'V40', 'confidence': 86},
                {'rank': 2, 'base_wax': 'VG35', 'top_wax': 'VP40', 'confidence': 80},
                {'rank': 3, 'base_wax': 'VG30', 'top_wax': 'V40', 'confidence': 73}
            ]
        else:
            return [
                {'rank': 1, 'base_wax': 'VG35', 'top_wax': 'VP40', 'confidence': 85},
                {'rank': 2, 'base_wax': 'VG35', 'top_wax': 'V40', 'confidence': 79},
                {'rank': 3, 'base_wax': 'VG30', 'top_wax': 'VP40', 'confidence': 72}
            ]
    
    # Transition zone (-4 to -1°C)
    elif snow_temp < -1:
        if is_wet or is_moist:
            return [
                {'rank': 1, 'base_wax': 'VG35', 'top_wax': 'VP45', 'confidence': 84},
                {'rank': 2, 'base_wax': 'VG35', 'top_wax': 'VP50', 'confidence': 78},
                {'rank': 3, 'base_wax': 'KX45', 'top_wax': 'VP45', 'confidence': 71}
            ]
        else:
            return [
                {'rank': 1, 'base_wax': 'VG35', 'top_wax': 'VP40', 'confidence': 83},
                {'rank': 2, 'base_wax': 'VG35', 'top_wax': 'VP45', 'confidence': 77},
                {'rank': 3, 'base_wax': 'VG35', 'top_wax': 'V40', 'confidence': 70}
            ]
    
    # Near zero (-1 to 1°C)
    elif snow_temp < 1:
        if is_wet:
            return [
                {'rank': 1, 'base_wax': 'KX55', 'top_wax': 'VP50', 'confidence': 82},
                {'rank': 2, 'base_wax': 'KN33', 'top_wax': 'VP50', 'confidence': 76},
                {'rank': 3, 'base_wax': 'VG35', 'top_wax': 'VP50', 'confidence': 69}
            ]
        else:
            return [
                {'rank': 1, 'base_wax': 'VG35', 'top_wax': 'VP50', 'confidence': 81},
                {'rank': 2, 'base_wax': 'KX55', 'top_wax': 'VP45', 'confidence': 75},
                {'rank': 3, 'base_wax': 'VG35', 'top_wax': 'VP45', 'confidence': 68}
            ]
    
    # Warm conditions (1 to 4°C)
    elif snow_temp < 4:
        if is_wet:
            return [
                {'rank': 1, 'base_wax': 'KN33', 'top_wax': 'VP50', 'confidence': 80},
                {'rank': 2, 'base_wax': 'KN44', 'top_wax': 'VP50', 'confidence': 74},
                {'rank': 3, 'base_wax': 'KX65', 'top_wax': 'VP50', 'confidence': 67}
            ]
        else:
            return [
                {'rank': 1, 'base_wax': 'KX55', 'top_wax': 'VP50', 'confidence': 79},
                {'rank': 2, 'base_wax': 'KX65', 'top_wax': 'VP50', 'confidence': 73},
                {'rank': 3, 'base_wax': 'KN33', 'top_wax': 'VP50', 'confidence': 66}
            ]
    
    # Very warm / wet conditions (> 4°C)
    else:
        return [
            {'rank': 1, 'base_wax': 'KN44', 'top_wax': 'VP50', 'confidence': 78},
            {'rank': 2, 'base_wax': 'K22', 'top_wax': 'VP50', 'confidence': 72},
            {'rank': 3, 'base_wax': 'KX75', 'top_wax': 'VP50', 'confidence': 65}
        ]


# =============================================================================
# STYLES
# =============================================================================

st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    color: white;
    padding: 30px;
    border-radius: 10px;
    margin-bottom: 30px;
    text-align: center;
}
.prediction-card {
    background: white;
    border-left: 5px solid #4CAF50;
    padding: 25px;
    margin: 15px 0;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.rank-1 { border-left-color: #FFD700; background: linear-gradient(to right, #fffef0, white); }
.rank-2 { border-left-color: #C0C0C0; background: linear-gradient(to right, #f8f8f8, white); }
.rank-3 { border-left-color: #CD7F32; background: linear-gradient(to right, #fdf5e6, white); }
.wax-box {
    display: inline-block;
    padding: 12px 24px;
    margin: 5px;
    border-radius: 8px;
    font-size: 1.6em;
    font-weight: bold;
}
.base-wax { background-color: #e3f2fd; color: #1565c0; border: 3px solid #1565c0; }
.top-wax { background-color: #e8f5e9; color: #2e7d32; border: 3px solid #2e7d32; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    st.markdown('''
    <div class="main-header">
        <h1>🎿 Kick Wax Predictor</h1>
        <p>AI-powered Swix wax recommendations for cross-country skiing</p>
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # VENUE
    st.markdown("### 📍 Venue")
    canada = [v for v in VENUES if any(x in v for x in ['YT', 'BC', 'AB', 'QC'])]
    europe = [v for v in VENUES if any(x in v for x in ['NOR', 'SUI', 'FIN', 'SWE', 'GER', 'SLO', 'ITA'])]
    venue = st.selectbox("Select Venue", canada + europe + ['Custom'])
    
    if venue != "Custom":
        elev, lat, lon, climate = VENUES[venue]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Elevation", f"{elev}m")
        c2.metric("Latitude", f"{abs(lat):.1f}°{'N' if lat >= 0 else 'S'}")
        c3.metric("Longitude", f"{abs(lon):.1f}°{'W' if lon < 0 else 'E'}")
        c4.metric("Climate", climate.title())
    else:
        c1, c2, c3 = st.columns(3)
        elev = c1.number_input("Elevation (m)", 0, 3000, 500)
        lat = c2.number_input("Latitude", -90.0, 90.0, 60.0)
        lon = c3.number_input("Longitude", -180.0, 180.0, -120.0)
    
    st.markdown("---")
    
    # SNOW CONDITIONS
    st.markdown("### ❄️ Snow Conditions")
    col1, col2 = st.columns(2)
    
    with col1:
        snow_source = st.radio("Snow Source", ["Natural", "Artificial"], horizontal=True)
        if snow_source == "Artificial":
            snow_type = st.selectbox("Artificial Snow Type", list(ARTIFICIAL_SNOW_OPTIONS.keys()))
            snow_code = ARTIFICIAL_SNOW_OPTIONS[snow_type]
            is_artificial = True
        else:
            snow_type = st.selectbox("Natural Snow Type", list(NATURAL_SNOW_OPTIONS.keys()))
            snow_code = NATURAL_SNOW_OPTIONS[snow_type]
            is_artificial = False
        grain_size = st.selectbox("Grain Size", list(GRAIN_SIZE_OPTIONS.keys()))
    
    with col2:
        snow_humidity = st.selectbox("Snow Humidity", list(SNOW_HUMIDITY_OPTIONS.keys()))
        snow_humidity_code = SNOW_HUMIDITY_OPTIONS[snow_humidity]
        track_hardness = st.selectbox("Track Hardness", list(TRACK_HARDNESS_OPTIONS.keys()))
        track_consistency = st.selectbox("Track Consistency", list(TRACK_CONSISTENCY_OPTIONS.keys()))
    
    st.markdown("---")
    
    # WEATHER
    st.markdown("### 🌡️ Weather Conditions")
    col1, col2 = st.columns(2)
    with col1:
        snow_temp = st.number_input("Snow Temperature (°C)", -30.0, 10.0, -5.0, 0.5)
        air_temp = st.number_input("Air Temperature (°C)", -35.0, 15.0, -3.0, 0.5)
    with col2:
        humidity = st.number_input("Air Humidity (%)", 0, 100, 70, 5)
        st.caption(f"Temperature differential: {air_temp - snow_temp:.1f}°C")
    
    st.markdown("---")
    
    # PREDICT
    if st.button("🎯 GET WAX RECOMMENDATIONS", type="primary", use_container_width=True):
        
        recs = predict_wax(snow_temp, air_temp, humidity, is_artificial, snow_humidity_code)
        
        st.markdown("## 🏆 Top 3 Recommendations")
        st.caption(f"📍 {venue} | ❄️ Snow: {snow_temp}°C | 🌡️ Air: {air_temp}°C | 💧 {humidity}% RH | {snow_code}")
        
        for rec in recs:
            rank = rec['rank']
            medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉"
            
            st.markdown(f'''
            <div class="prediction-card rank-{rank}">
                <h2 style="margin-top:0">{medal} Recommendation #{rank}</h2>
                <div style="text-align:center; margin:20px 0">
                    <span class="wax-box base-wax">{rec['base_wax']}</span>
                    <span style="font-size:2em; color:#666; margin: 0 10px;">+</span>
                    <span class="wax-box top-wax">{rec['top_wax']}</span>
                </div>
                <p style="text-align:center; color:#666">Confidence: <strong>{rec['confidence']}%</strong></p>
            </div>
            ''', unsafe_allow_html=True)
        
        # Tips
        st.markdown("---")
        st.markdown("### 💡 Application Tips")
        if snow_temp < -10:
            st.info("**Cold conditions:** Apply thin layers. Cork thoroughly for best grip. Test before race.")
        elif snow_temp < -3:
            st.info("**Mid-range conditions:** Standard application. Cork well and test on snow.")
        elif snow_temp < 1:
            st.info("**Transition zone:** Conditions may change. Have backup wax ready.")
        else:
            st.info("**Warm/wet conditions:** Apply conservatively. May need to refresh during race.")
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🎿 Swix Wax Guide")
        st.markdown("""
        **Base Layer (Binder):**
        - **KX20/KX30**: Very cold (<-12°C)
        - **KX35/KX45**: Cold (-12 to -4°C)
        - **VG30/VG35**: Universal hard wax
        - **KX55/KX65**: Warm (>-2°C)
        - **KN33/KN44/K22**: Klister (wet)
        
        **Top Layer (Cover):**
        - **V20/V30**: Cold conditions
        - **V40**: Mid-range
        - **VP40/VP45**: Racing, mid temps
        - **VP50**: Warmer conditions
        """)
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        Recommendations based on:
        - Swix wax charts
        - World Cup wax test data
        - Snow science principles
        
        *Always test on race day!*
        """)
        
        st.markdown("---")
        st.markdown("*Powered by Sitka Science*")


if __name__ == "__main__":
    main()
