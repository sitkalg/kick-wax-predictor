#!/usr/bin/env python3
"""
Kick Wax Predictor v2.0
- Canadian venues added
- Snow type selection (artificial/natural)
- Commercial wax name output

Run: python -m streamlit run kick_wax_predictor_v2.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os

# Optional: XGBoost import (fails gracefully if not installed)
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

st.set_page_config(page_title="Kick Wax Predictor", page_icon="🎿", layout="wide")


# =============================================================================
# WAX MAPPING - Category to Commercial Names
# =============================================================================

def get_commercial_base_wax(category, temperature):
    """Map base category to commercial Swix wax name based on temperature"""
    cat = str(category).upper()
    
    # KX Series (Klister for cold/transformed snow)
    if 'COLD' in cat and 'KLISTER' in cat:
        if temperature < -15: return 'KX20'
        elif temperature < -10: return 'KX30'
        elif temperature < -5: return 'KX35'
        else: return 'KX40'
    
    # Mid-range klister
    elif 'MID' in cat and 'KLISTER' in cat:
        if temperature < -8: return 'KX30'
        elif temperature < -3: return 'KX35'
        else: return 'KX45'
    
    # Warm klister
    elif 'WARM' in cat and 'KLISTER' in cat:
        if temperature < -1: return 'KX45'
        elif temperature < 2: return 'KX65'
        else: return 'KX75'
    
    # Hard wax bases (VG series)
    elif 'HARD' in cat or 'VG' in cat:
        if temperature < -12: return 'VG30'
        else: return 'VG35'
    
    # Generic klister
    elif 'KLISTER' in cat:
        if temperature < -5: return 'K22'
        elif temperature < 0: return 'KN33'
        else: return 'KN44'
    
    # KX_COLD from your model
    elif 'KX_COLD' in cat or 'KX' in cat:
        if temperature < -15: return 'KX20'
        elif temperature < -10: return 'KX30'
        elif temperature < -5: return 'KX35'
        else: return 'KX45'
    
    # Default fallbacks
    else:
        if temperature < -10: return 'KX30'
        elif temperature < -3: return 'VG35'
        else: return 'KX45'


def get_commercial_top_wax(category, temperature):
    """Map top category to commercial Swix wax name based on temperature"""
    cat = str(category).upper()
    
    # V Series (hard waxes)
    if 'V_COLD' in cat or 'COLD_HARD' in cat:
        if temperature < -15: return 'V05'
        elif temperature < -10: return 'V20'
        else: return 'V30'
    
    elif 'V30' in cat:
        return 'V30'
    
    elif 'V40' in cat and 'VP' not in cat:
        return 'V40'
    
    elif 'V50' in cat and 'VP' not in cat:
        return 'V50'
    
    elif 'MID_V' in cat:
        if temperature < -5: return 'V30'
        else: return 'V40'
    
    elif 'WARM_V' in cat:
        if temperature < 0: return 'V40'
        else: return 'V50'
    
    # VP Series (racing waxes)
    elif 'VP30' in cat or 'COLD_VP' in cat:
        return 'VP30'
    
    elif 'VP_MID' in cat or 'MID_VP' in cat:
        if temperature < -3: return 'VP40'
        else: return 'VP45'
    
    elif 'VP_WARM' in cat or 'WARM_VP' in cat:
        if temperature < 0: return 'VP50'
        else: return 'VP55'
    
    # Klister as top layer
    elif 'KLISTER' in cat:
        if temperature < -3: return 'KX35'
        elif temperature < 0: return 'KN33'
        else: return 'KN44'
    
    # OTHER category
    elif 'OTHER' in cat:
        if temperature < -8: return 'V30'
        elif temperature < -3: return 'V40'
        else: return 'VP50'
    
    # Direct wax names (already commercial)
    elif cat in ['V05', 'V20', 'V30', 'V40', 'V50', 'V60', 
                 'VP30', 'VP40', 'VP45', 'VP50', 'VP55', 'VP60',
                 'KX20', 'KX30', 'KX35', 'KX40', 'KX45', 'KX65', 'KX75',
                 'K22', 'KN33', 'KN44']:
        return cat
    
    # Default based on temperature
    else:
        if temperature < -10: return 'V30'
        elif temperature < -3: return 'V40'
        elif temperature < 0: return 'VP50'
        else: return 'VP55'


# =============================================================================
# VENUES - Including Canadian locations
# =============================================================================

VENUES = {
    # --- CANADA ---
    "Whitehorse, YT": (700, 60.72, -135.05, "subarctic"),
    "Sovereign Lake, BC": (1450, 50.16, -119.22, "continental"),
    "Canmore, AB": (1400, 51.09, -115.36, "alpine"),
    "Prince George, BC": (575, 53.92, -122.75, "continental"),
    "Nakkertok, QC": (200, 45.50, -75.75, "continental"),
    "Mt. Sainte-Anne, QC": (800, 47.08, -70.90, "maritime"),
    
    # --- EUROPE ---
    "Beitostølen, NOR": (900, 61.25, 8.91, "continental"),
    "Davos, SUI": (1560, 46.80, 9.84, "alpine"),
    "Holmenkollen, NOR": (325, 59.96, 10.67, "maritime"),
    "Lahti, FIN": (120, 60.98, 25.66, "continental"),
    "Falun, SWE": (120, 60.61, 15.63, "continental"),
    "Trondheim, NOR": (200, 63.43, 10.39, "maritime"),
    "Östersund, SWE": (320, 63.18, 14.64, "continental"),
    "Ruka, FIN": (490, 66.17, 29.14, "arctic"),
    "Oberhof, GER": (815, 50.72, 10.73, "continental"),
    "Planica, SLO": (940, 46.48, 13.72, "alpine"),
    
    # --- CUSTOM ---
    "Custom": (500, 60.0, -120.0, "continental")
}


# =============================================================================
# SNOW TYPE OPTIONS
# =============================================================================

ARTIFICIAL_SNOW_TYPES = [
    "Machine-made (fresh)",
    "Machine-made (aged)",
    "Machine-made (wet)",
    "Machine-made (icy)"
]

NATURAL_SNOW_TYPES = [
    "New snow (fresh powder)",
    "New snow (settling)",
    "Fine grained (small crystals)",
    "Fine grained (packed)",
    "Coarse grained (old snow)",
    "Coarse grained (corn snow)",
    "Transformed (freeze-thaw)",
    "Transformed (wind-packed)",
    "Mixed (new on old)",
    "Mixed (variable)"
]


# =============================================================================
# PREDICTOR CLASS
# =============================================================================

class XGBoostPredictor:
    """XGBoost-based predictor with commercial wax output"""
    
    def __init__(self):
        self.model_loaded = False
        self.base_model = None
        self.top_model = None
        self.base_classes = np.array(['KX_COLD', 'VG'])
        self.top_classes = np.array(['KLISTER', 'V30', 'V40', 'VP_MID', 'VP_WARM', 'V_COLD'])
        self.scaler_center = None
        self.scaler_scale = None
    
    def load_from_directory(self, model_dir='kick_wax_model'):
        """Load XGBoost models from directory"""
        try:
            if not os.path.exists(model_dir):
                print(f"Model directory not found: {model_dir}")
                return False
            
            print(f"Loading XGBoost model from {model_dir}...")
            
            # Load encoders
            enc_path = os.path.join(model_dir, 'encoders.pkl')
            if os.path.exists(enc_path):
                with open(enc_path, 'rb') as f:
                    enc = pickle.load(f)
                self.base_classes = np.array(enc.get('base_classes', self.base_classes))
                self.top_classes = np.array(enc.get('top_classes', self.top_classes))
                print(f"  ✓ Loaded encoders ({len(self.base_classes)} base, {len(self.top_classes)} top)")
            
            # Load scaler
            scaler_path = os.path.join(model_dir, 'scaler.pkl')
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    s = pickle.load(f)
                self.scaler_center = np.array(s.get('center', np.zeros(3)))
                self.scaler_scale = np.array(s.get('scale', np.ones(3)))
                print(f"  ✓ Loaded scaler")
            
            # Load XGBoost models
            if not HAS_XGBOOST:
                print("  ✗ XGBoost not installed. Run: pip install xgboost")
                return False
            
            base_path = os.path.join(model_dir, 'base_model.json')
            top_path = os.path.join(model_dir, 'top_model.json')
            
            if os.path.exists(base_path) and os.path.exists(top_path):
                self.base_model = xgb.XGBClassifier()
                self.base_model.load_model(base_path)
                print(f"  ✓ Loaded base model")
                
                self.top_model = xgb.XGBClassifier()
                self.top_model.load_model(top_path)
                print(f"  ✓ Loaded top model")
                
                self.model_loaded = True
                print("✓ XGBoost models loaded successfully!")
                return True
            else:
                print(f"  ✗ XGBoost model files not found")
                return False
            
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def create_features(self, conditions):
        """Create feature vector"""
        if isinstance(conditions, pd.DataFrame):
            row = conditions.iloc[0]
        else:
            row = conditions
        
        snow_temp = float(row.get('Snow temperature (C)', -10))
        air_temp = float(row.get('Air temperature (C)', -8))
        humidity = float(row.get('Air humidity (% rH)', 70))
        
        return np.array([[snow_temp, air_temp, humidity]])
    
    def scale_features(self, X):
        """Apply scaling"""
        if self.scaler_center is None:
            return X
        
        n = len(self.scaler_center)
        if X.shape[1] < n:
            X = np.pad(X, ((0, 0), (0, n - X.shape[1])))
        elif X.shape[1] > n:
            X = X[:, :n]
        
        scale = np.where(self.scaler_scale == 0, 1, self.scaler_scale)
        return (X - self.scaler_center) / scale
    
    def predict(self, conditions, top_k=3):
        """Make predictions using XGBoost"""
        if not self.model_loaded:
            return None
        
        try:
            if isinstance(conditions, pd.DataFrame):
                temp = conditions['Snow temperature (C)'].values[0]
            else:
                temp = conditions.get('Snow temperature (C)', -10)
            
            X = self.create_features(conditions)
            X_scaled = self.scale_features(X)
            
            # Get probability predictions
            base_probs = self.base_model.predict_proba(X_scaled)[0]
            top_probs = self.top_model.predict_proba(X_scaled)[0]
            
            return self._generate_recommendations(base_probs, top_probs, temp, top_k)
            
        except Exception as e:
            print(f"Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _generate_recommendations(self, base_probs, top_probs, temperature, top_k=3):
        """Generate recommendations with commercial wax names"""
        base_probs = np.array(base_probs).flatten()
        top_probs = np.array(top_probs).flatten()
        
        n_base = min(2, len(base_probs))
        n_top = min(3, len(top_probs))
        
        base_indices = np.argsort(base_probs)[-n_base:][::-1]
        top_indices = np.argsort(top_probs)[-n_top:][::-1]
        
        results = []
        for bi in base_indices:
            base_category = self.base_classes[bi] if bi < len(self.base_classes) else 'VG'
            # Convert to commercial name
            base_wax = get_commercial_base_wax(str(base_category), temperature)
            bp = float(base_probs[bi])
            
            for ti in top_indices:
                top_category = self.top_classes[ti] if ti < len(self.top_classes) else 'V40'
                # Convert to commercial name
                top_wax = get_commercial_top_wax(str(top_category), temperature)
                tp = float(top_probs[ti])
                
                results.append({
                    'base_wax': base_wax,
                    'top_wax': top_wax,
                    'confidence': bp * tp
                })
        
        # Remove duplicates (same base+top combo)
        seen = set()
        unique_results = []
        for r in results:
            key = (r['base_wax'], r['top_wax'])
            if key not in seen:
                seen.add(key)
                unique_results.append(r)
        
        unique_results.sort(key=lambda x: x['confidence'], reverse=True)
        for i, r in enumerate(unique_results[:top_k]):
            r['rank'] = i + 1
        
        return unique_results[:top_k]


class KickWaxPredictor:
    """Main predictor interface"""
    
    def __init__(self):
        self.model_loaded = False
        self.predictor = XGBoostPredictor()
    
    def load_model(self, path='kick_wax_model'):
        self.model_loaded = self.predictor.load_from_directory(path)
        return self.model_loaded
    
    def predict(self, conditions, top_k=3):
        if self.model_loaded:
            pred = self.predictor.predict(
                pd.DataFrame([conditions]) if isinstance(conditions, dict) else conditions,
                top_k
            )
            if pred:
                return pred
        return self.rule_based(conditions, top_k)
    
    def rule_based(self, c, k=3):
        """Temperature-based rule fallback with commercial wax names"""
        t = c.get('Snow temperature (C)', -10) if isinstance(c, dict) else c['Snow temperature (C)'].values[0]
        is_artificial = c.get('is_artificial', False) if isinstance(c, dict) else False
        
        # Artificial snow tends to be more abrasive, prefer harder waxes
        if is_artificial:
            if t < -12:
                return [
                    {'rank': 1, 'base_wax': 'KX30', 'top_wax': 'V20', 'confidence': 0.85},
                    {'rank': 2, 'base_wax': 'KX20', 'top_wax': 'V30', 'confidence': 0.78},
                    {'rank': 3, 'base_wax': 'VG30', 'top_wax': 'V20', 'confidence': 0.72}
                ]
            elif t < -8:
                return [
                    {'rank': 1, 'base_wax': 'KX35', 'top_wax': 'V30', 'confidence': 0.84},
                    {'rank': 2, 'base_wax': 'VG35', 'top_wax': 'V40', 'confidence': 0.77},
                    {'rank': 3, 'base_wax': 'KX30', 'top_wax': 'V30', 'confidence': 0.71}
                ]
            elif t < -3:
                return [
                    {'rank': 1, 'base_wax': 'VG35', 'top_wax': 'V40', 'confidence': 0.83},
                    {'rank': 2, 'base_wax': 'KX45', 'top_wax': 'VP45', 'confidence': 0.76},
                    {'rank': 3, 'base_wax': 'VG35', 'top_wax': 'VP40', 'confidence': 0.70}
                ]
            elif t < 0:
                return [
                    {'rank': 1, 'base_wax': 'KX45', 'top_wax': 'VP50', 'confidence': 0.80},
                    {'rank': 2, 'base_wax': 'VG35', 'top_wax': 'VP45', 'confidence': 0.73},
                    {'rank': 3, 'base_wax': 'KN33', 'top_wax': 'VP50', 'confidence': 0.67}
                ]
            else:
                return [
                    {'rank': 1, 'base_wax': 'KN44', 'top_wax': 'VP55', 'confidence': 0.78},
                    {'rank': 2, 'base_wax': 'KX65', 'top_wax': 'VP55', 'confidence': 0.71},
                    {'rank': 3, 'base_wax': 'K22', 'top_wax': 'VP50', 'confidence': 0.65}
                ]
        
        # Natural snow
        else:
            if t < -12:
                return [
                    {'rank': 1, 'base_wax': 'KX20', 'top_wax': 'V20', 'confidence': 0.88},
                    {'rank': 2, 'base_wax': 'KX30', 'top_wax': 'V30', 'confidence': 0.82},
                    {'rank': 3, 'base_wax': 'VG30', 'top_wax': 'VP30', 'confidence': 0.76}
                ]
            elif t < -8:
                return [
                    {'rank': 1, 'base_wax': 'VG35', 'top_wax': 'V30', 'confidence': 0.85},
                    {'rank': 2, 'base_wax': 'KX35', 'top_wax': 'V40', 'confidence': 0.79},
                    {'rank': 3, 'base_wax': 'VG30', 'top_wax': 'VP30', 'confidence': 0.73}
                ]
            elif t < -3:
                return [
                    {'rank': 1, 'base_wax': 'VG35', 'top_wax': 'VP40', 'confidence': 0.84},
                    {'rank': 2, 'base_wax': 'VG35', 'top_wax': 'V40', 'confidence': 0.78},
                    {'rank': 3, 'base_wax': 'KX45', 'top_wax': 'VP45', 'confidence': 0.71}
                ]
            elif t < 0:
                return [
                    {'rank': 1, 'base_wax': 'VG35', 'top_wax': 'VP50', 'confidence': 0.82},
                    {'rank': 2, 'base_wax': 'K22', 'top_wax': 'VP45', 'confidence': 0.76},
                    {'rank': 3, 'base_wax': 'VG35', 'top_wax': 'VP55', 'confidence': 0.69}
                ]
            else:
                return [
                    {'rank': 1, 'base_wax': 'KN44', 'top_wax': 'VP55', 'confidence': 0.80},
                    {'rank': 2, 'base_wax': 'K22', 'top_wax': 'VP50', 'confidence': 0.74},
                    {'rank': 3, 'base_wax': 'KN33', 'top_wax': 'VP55', 'confidence': 0.68}
                ]


# =============================================================================
# Streamlit UI
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
.venue-card {
    background: #f5f5f5;
    padding: 15px;
    border-radius: 8px;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)


def main():
    if 'predictor' not in st.session_state:
        st.session_state.predictor = KickWaxPredictor()
        st.session_state.load_attempted = False
    
    predictor = st.session_state.predictor
    
    st.markdown('''
    <div class="main-header">
        <h1>🎿 Kick Wax Predictor</h1>
        <p>AI-powered Swix wax recommendations</p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Model loading
    if not st.session_state.load_attempted:
        col1, col2 = st.columns([1, 2])
        with col1:
            if st.button("🚀 Load AI Model", type="primary"):
                with st.spinner('Loading model...'):
                    st.session_state.load_attempted = True
                    for path in ['kick_wax_model', './kick_wax_model']:
                        if os.path.exists(path):
                            if predictor.load_model(path):
                                break
                    st.rerun()
        with col2:
            st.info("💡 Click to load AI model, or use rule-based predictions below")
    else:
        if predictor.model_loaded:
            st.success("✓ AI Model Active")
        else:
            st.warning("⚠ Using rule-based predictions")
    
    st.markdown("---")
    
    # ===================
    # VENUE SELECTION
    # ===================
    st.markdown("### 📍 Venue")
    
    # Group venues by region
    canada_venues = [v for v in VENUES.keys() if any(x in v for x in ['YT', 'BC', 'AB', 'QC'])]
    europe_venues = [v for v in VENUES.keys() if any(x in v for x in ['NOR', 'SUI', 'FIN', 'SWE', 'GER', 'SLO'])]
    other_venues = ['Custom']
    
    venue_options = ['--- Canada ---'] + canada_venues + ['--- Europe ---'] + europe_venues + ['--- Other ---'] + other_venues
    
    # Filter out section headers for actual selection
    selectable_venues = [v for v in venue_options if not v.startswith('---')]
    
    venue = st.selectbox("Select Venue", selectable_venues)
    
    if venue != "Custom":
        elev, lat, lon, climate = VENUES[venue]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Elevation", f"{elev}m")
        col2.metric("Latitude", f"{abs(lat):.1f}°{'N' if lat >= 0 else 'S'}")
        col3.metric("Longitude", f"{abs(lon):.1f}°{'W' if lon < 0 else 'E'}")
        col4.metric("Climate", climate.title())
    else:
        col1, col2, col3 = st.columns(3)
        elev = col1.number_input("Elevation (m)", 0, 3000, 500)
        lat = col2.number_input("Latitude", -90.0, 90.0, 60.0)
        lon = col3.number_input("Longitude", -180.0, 180.0, -120.0)
        climate = "continental"
    
    st.markdown("---")
    
    # ===================
    # SNOW CONDITIONS
    # ===================
    st.markdown("### ❄️ Snow Conditions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        snow_source = st.radio("Snow Type", ["Natural", "Artificial"], horizontal=True)
        
        if snow_source == "Artificial":
            snow_type = st.selectbox("Artificial Snow Condition", ARTIFICIAL_SNOW_TYPES)
            is_artificial = True
        else:
            snow_type = st.selectbox("Natural Snow Type", NATURAL_SNOW_TYPES)
            is_artificial = False
    
    with col2:
        snow_moisture = st.selectbox("Snow Moisture", ["Dry", "Moist", "Wet", "Very wet", "Slush"], index=1)
        track_hardness = st.selectbox("Track Hardness", ["Soft", "Medium", "Hard", "Icy"], index=1)
    
    st.markdown("---")
    
    # ===================
    # WEATHER CONDITIONS
    # ===================
    st.markdown("### 🌡️ Weather Conditions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        snow_temp = st.number_input("Snow Temperature (°C)", -30.0, 10.0, -8.0, 0.5)
        air_temp = st.number_input("Air Temperature (°C)", -35.0, 15.0, -5.0, 0.5)
    
    with col2:
        humidity = st.number_input("Air Humidity (%)", 0, 100, 70, 5)
        st.caption(f"Temperature differential: {air_temp - snow_temp:.1f}°C")
    
    st.markdown("---")
    
    # ===================
    # PREDICTION
    # ===================
    if st.button("🎯 GET WAX RECOMMENDATIONS", type="primary", use_container_width=True):
        
        conditions = {
            'Snow temperature (C)': snow_temp,
            'Air temperature (C)': air_temp,
            'Air humidity (% rH)': humidity,
            'is_artificial': is_artificial,
            'snow_type': snow_type,
            'snow_moisture': snow_moisture
        }
        
        with st.spinner('🤖 Analyzing conditions...'):
            recs = predictor.predict(conditions, top_k=3)
        
        st.markdown("## 🏆 Top 3 Recommendations")
        
        # Show conditions summary
        st.caption(f"📍 {venue} | ❄️ Snow: {snow_temp}°C | 🌡️ Air: {air_temp}°C | 💧 {humidity}% RH | {snow_source} - {snow_type}")
        
        for rec in recs:
            rank = rec['rank']
            medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉"
            rank_class = f"rank-{rank}"
            
            st.markdown(f'''
            <div class="prediction-card {rank_class}">
                <h2 style="margin-top:0">{medal} #{rank}</h2>
                <div style="text-align:center; margin:20px 0">
                    <span class="wax-box base-wax">{rec['base_wax']}</span>
                    <span style="font-size:2em; color:#666; margin: 0 10px;">+</span>
                    <span class="wax-box top-wax">{rec['top_wax']}</span>
                </div>
                <p style="text-align:center; font-size:1.1em; color:#666">
                    Confidence: <strong>{rec['confidence']*100:.0f}%</strong>
                </p>
            </div>
            ''', unsafe_allow_html=True)
        
        # Wax application tip
        st.markdown("---")
        st.markdown("### 💡 Application Tips")
        if snow_temp < -10:
            st.info("**Cold conditions:** Apply thin layers. Cork thoroughly. Consider multiple thin layers of top wax.")
        elif snow_temp < -3:
            st.info("**Mid-range conditions:** Standard application. Cork well and let wax cool between layers.")
        else:
            st.info("**Warm conditions:** Apply conservatively. Watch for glazing. May need to refresh during race.")
    
    # ===================
    # SIDEBAR
    # ===================
    with st.sidebar:
        st.markdown("### ℹ️ About")
        st.write("AI-powered kick wax recommendations using Swix wax products.")
        
        st.markdown("### 📊 Model Status")
        if predictor.model_loaded:
            st.success("🤖 AI Model: Active")
        else:
            st.warning("📊 Rule-based mode")
        
        st.markdown("### 🎿 Wax Guide")
        st.markdown("""
        **Base Waxes (Binders):**
        - KX20-KX30: Cold klister (-15°C and below)
        - KX35-KX45: Mid-range klister
        - VG30-VG35: Hard wax base
        - KN33-KN44: Universal klister
        
        **Top Waxes:**
        - V05-V30: Cold hard wax
        - V40-V50: Mid-range hard wax  
        - VP30-VP55: Racing wax series
        """)


if __name__ == "__main__":
    main()
