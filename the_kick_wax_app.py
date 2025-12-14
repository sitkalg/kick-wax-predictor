#!/usr/bin/env python3
"""
Kick Wax Predictor - TensorFlow Version
========================================
Matches trained model: 25 features, 2 base classes, 6 top classes

Base classes: HARD_COLD, HARD_WARM
Top classes: V20, V30, V40, VP40, VP45, VP50

Run: streamlit run kick_wax_app.py
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import streamlit as st
import numpy as np
import pickle

# TensorFlow import
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    
    # Register custom layer for model loading
    @keras.utils.register_keras_serializable()
    class TemperatureAttentionLayer(layers.Layer):
        def __init__(self, n_features, **kwargs):
            super().__init__(**kwargs)
            self.n_features = n_features
        def build(self, input_shape):
            self.temp_transform = layers.Dense(32, activation='relu')
            self.attention_weights = layers.Dense(self.n_features, activation='sigmoid')
            super().build(input_shape)
        def call(self, inputs, temperature=None):
            if temperature is None:
                return inputs
            temp_context = self.temp_transform(temperature)
            weights = self.attention_weights(temp_context)
            return inputs * weights
        def get_config(self):
            config = super().get_config()
            config.update({'n_features': self.n_features})
            return config
    
    tf.config.set_visible_devices([], 'GPU')
    HAS_TF = True
except ImportError:
    HAS_TF = False

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
# WAX NAME MAPPING
# =============================================================================

def get_commercial_base_wax(category, temperature):
    """Map HARD_COLD/HARD_WARM to commercial Swix wax"""
    cat = str(category).upper()
    
    if 'COLD' in cat:
        if temperature < -15:
            return 'KX20'
        elif temperature < -10:
            return 'KX30'
        elif temperature < -5:
            return 'KX35'
        else:
            return 'KX45'
    else:  # HARD_WARM
        if temperature < -5:
            return 'VG30'
        else:
            return 'VG35'


# =============================================================================
# PREDICTOR CLASS
# =============================================================================

class KickWaxPredictor:
    def __init__(self):
        self.model_loaded = False
        self.model = None
        self.base_classes = np.array(['HARD_COLD', 'HARD_WARM'])
        self.top_classes = np.array(['V20', 'V30', 'V40', 'VP40', 'VP45', 'VP50'])
        self.scaler_center = None
        self.scaler_scale = None
        self.n_features = 25
    
    def load_model(self, model_dir='kick_wax_model'):
        if not HAS_TF:
            st.error("TensorFlow not available")
            return False
        
        try:
            # Load encoders
            enc_path = os.path.join(model_dir, 'encoders.pkl')
            if os.path.exists(enc_path):
                with open(enc_path, 'rb') as f:
                    enc = pickle.load(f)
                self.base_classes = np.array([str(c) for c in enc.get('base_classes', self.base_classes)])
                self.top_classes = np.array([str(c) for c in enc.get('top_classes', self.top_classes)])
            
            # Load scaler
            scaler_path = os.path.join(model_dir, 'scaler.pkl')
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    s = pickle.load(f)
                self.scaler_center = np.array(s.get('center'))
                self.scaler_scale = np.array(s.get('scale'))
                self.n_features = len(self.scaler_center)
            
            # Load feature info
            feat_path = os.path.join(model_dir, 'feature_info.pkl')
            if os.path.exists(feat_path):
                with open(feat_path, 'rb') as f:
                    fi = pickle.load(f)
                self.n_features = fi.get('n_features', self.n_features)
            
            # Load Keras model
            model_path = os.path.join(model_dir, 'main_model.keras')
            if os.path.exists(model_path):
                custom_objects = {'TemperatureAttentionLayer': TemperatureAttentionLayer}
                self.model = keras.models.load_model(model_path, compile=False, custom_objects=custom_objects)
                self.model_loaded = True
                return True
            
            return False
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return False
    
    def create_features(self, conditions):
        """Create 25-feature vector matching training"""
        snow_temp = conditions.get('snow_temp', -5)
        air_temp = conditions.get('air_temp', -3)
        humidity = conditions.get('humidity', 70)
        elevation = conditions.get('elevation', 500)
        latitude = conditions.get('latitude', 60)
        longitude = conditions.get('longitude', 10)
        is_artificial = conditions.get('is_artificial', False)
        
        features = np.zeros(self.n_features)
        
        # Basic features (indices 0-2)
        features[0] = snow_temp
        features[1] = air_temp
        features[2] = humidity
        
        # Fuzzy temperature zones (indices 3-5)
        features[3] = np.exp(-0.5 * ((snow_temp + 12) / 6) ** 2)  # cold
        features[4] = np.exp(-0.5 * ((snow_temp + 5) / 5) ** 2)   # mid
        features[5] = np.exp(-0.5 * ((snow_temp) / 4) ** 2)       # warm
        total = features[3] + features[4] + features[5]
        if total > 0:
            features[3:6] /= total
        
        # Temperature gradient features (indices 6-11)
        gradient = air_temp - snow_temp
        features[6] = gradient  # air_snow_gradient
        features[7] = 1 if abs(gradient) < 2 else 0  # stable
        features[8] = 1 if gradient > 3 else 0  # warming
        features[9] = 1 if gradient < -3 else 0  # cooling
        features[10] = 1 if abs(gradient) > 5 else 0  # dynamic
        features[11] = 1 if abs(snow_temp) < 2 else 0  # near_zero
        
        # Gradient cluster (index 12)
        if features[7] and snow_temp < -5:
            features[12] = 0
        elif features[7]:
            features[12] = 1
        elif features[8]:
            features[12] = 2
        else:
            features[12] = 3
        
        # Ordinal temperature (index 13)
        temp_bins = [-20, -15, -10, -8, -5, -3, -1, 1, 5]
        features[13] = np.digitize(snow_temp, temp_bins)
        
        # Venue features (indices 14-16)
        features[14] = elevation / 2000  # elevation_norm
        features[15] = latitude / 90  # latitude_norm
        features[16] = longitude / 180 if longitude else 0
        
        # Snow type (indices 17-18)
        features[17] = 1 if is_artificial else 0  # artificial_snow
        features[18] = 0 if is_artificial else 1  # natural_snow
        
        # Physics features (indices 19-20)
        features[19] = 6.11 * np.exp(17.27 * air_temp / (237.3 + air_temp))  # vapor_pressure
        try:
            dew_val = humidity/100 * features[19]/6.11
            if dew_val > 0:
                features[20] = (237.3 * np.log(dew_val)) / (17.27 - np.log(dew_val))
        except:
            features[20] = 0
        
        # Additional features (indices 21-24)
        features[21] = snow_temp * humidity / 100
        features[22] = air_temp - snow_temp
        features[23] = abs(snow_temp)
        features[24] = 1 if snow_temp < -8 else 0
        
        # Scale features
        if self.scaler_center is not None and self.scaler_scale is not None:
            n = min(len(features), len(self.scaler_center))
            scale = np.where(self.scaler_scale[:n] == 0, 1, self.scaler_scale[:n])
            features[:n] = (features[:n] - self.scaler_center[:n]) / scale
        
        return features.reshape(1, -1)
    
    def predict(self, conditions, top_k=3):
        temp = conditions.get('snow_temp', -5)
        
        if self.model_loaded and self.model is not None:
            try:
                X = self.create_features(conditions)
                temp_input = np.array([[temp]])
                
                preds = self.model.predict([X, temp_input], verbose=0)
                
                if isinstance(preds, dict):
                    base_probs = preds['base'][0]
                    top_probs = preds['top'][0]
                else:
                    base_probs = preds[0][0]
                    top_probs = preds[1][0]
                
                return self._generate_recs(base_probs, top_probs, temp, top_k)
            except Exception as e:
                st.warning(f"Model error: {e}, using rules")
        
        return self.rule_based(temp, top_k)
    
    def _generate_recs(self, base_probs, top_probs, temp, top_k):
        base_probs = np.array(base_probs).flatten()
        top_probs = np.array(top_probs).flatten()
        
        results = []
        for bi in range(len(self.base_classes)):
            base_cat = self.base_classes[bi]
            base_wax = get_commercial_base_wax(base_cat, temp)
            
            for ti in range(len(self.top_classes)):
                top_wax = str(self.top_classes[ti])  # Already commercial name
                
                results.append({
                    'base_wax': base_wax,
                    'top_wax': top_wax,
                    'confidence': float(base_probs[bi] * top_probs[ti])
                })
        
        results.sort(key=lambda x: x['confidence'], reverse=True)
        seen = set()
        unique = []
        for r in results:
            key = (r['base_wax'], r['top_wax'])
            if key not in seen:
                seen.add(key)
                unique.append(r)
        
        for i, r in enumerate(unique[:top_k]):
            r['rank'] = i + 1
        return unique[:top_k]
    
    def rule_based(self, temp, top_k=3):
        if temp < -12:
            return [
                {'rank': 1, 'base_wax': 'KX20', 'top_wax': 'V20', 'confidence': 0.85},
                {'rank': 2, 'base_wax': 'KX30', 'top_wax': 'V30', 'confidence': 0.78},
                {'rank': 3, 'base_wax': 'VG30', 'top_wax': 'V30', 'confidence': 0.72}
            ][:top_k]
        elif temp < -8:
            return [
                {'rank': 1, 'base_wax': 'KX35', 'top_wax': 'V30', 'confidence': 0.83},
                {'rank': 2, 'base_wax': 'VG30', 'top_wax': 'V40', 'confidence': 0.77},
                {'rank': 3, 'base_wax': 'KX45', 'top_wax': 'VP40', 'confidence': 0.71}
            ][:top_k]
        elif temp < -3:
            return [
                {'rank': 1, 'base_wax': 'VG35', 'top_wax': 'VP40', 'confidence': 0.82},
                {'rank': 2, 'base_wax': 'VG35', 'top_wax': 'V40', 'confidence': 0.76},
                {'rank': 3, 'base_wax': 'KX45', 'top_wax': 'VP45', 'confidence': 0.70}
            ][:top_k]
        elif temp < 0:
            return [
                {'rank': 1, 'base_wax': 'VG35', 'top_wax': 'VP45', 'confidence': 0.80},
                {'rank': 2, 'base_wax': 'VG35', 'top_wax': 'VP50', 'confidence': 0.74},
                {'rank': 3, 'base_wax': 'KX55', 'top_wax': 'VP50', 'confidence': 0.68}
            ][:top_k]
        else:
            return [
                {'rank': 1, 'base_wax': 'KX55', 'top_wax': 'VP50', 'confidence': 0.78},
                {'rank': 2, 'base_wax': 'KN33', 'top_wax': 'VP50', 'confidence': 0.72},
                {'rank': 3, 'base_wax': 'KN44', 'top_wax': 'VP50', 'confidence': 0.66}
            ][:top_k]


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
    
    # Auto-load model
    if not st.session_state.load_attempted:
        st.session_state.load_attempted = True
        for path in ['kick_wax_model', './kick_wax_model', '.']:
            if predictor.load_model(path):
                break
    
    if predictor.model_loaded:
        st.success(f"✓ TensorFlow Model Active | Base: {list(predictor.base_classes)} | Top: {list(predictor.top_classes)}")
    else:
        st.warning("⚠ Using rule-based predictions (model not found)")
    
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
        conditions = {
            'snow_temp': snow_temp,
            'air_temp': air_temp,
            'humidity': humidity,
            'elevation': elev if venue != "Custom" else elev,
            'latitude': lat if venue != "Custom" else lat,
            'longitude': lon if venue != "Custom" else lon,
            'is_artificial': is_artificial
        }
        
        with st.spinner('🤖 Analyzing conditions...'):
            recs = predictor.predict(conditions, top_k=3)
        
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
                <p style="text-align:center; color:#666">Confidence: <strong>{rec['confidence']*100:.0f}%</strong></p>
            </div>
            ''', unsafe_allow_html=True)
        
        # Tips
        st.markdown("---")
        st.markdown("### 💡 Application Tips")
        if snow_temp < -10:
            st.info("**Cold conditions:** Apply thin layers. Cork thoroughly.")
        elif snow_temp < -3:
            st.info("**Mid-range conditions:** Standard application. Cork well.")
        else:
            st.info("**Warm conditions:** Apply conservatively. May need refresh.")
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 Model Status")
        if predictor.model_loaded:
            st.success("🤖 TensorFlow AI Active")
            st.caption(f"Features: {predictor.n_features}")
            st.caption(f"Base: {len(predictor.base_classes)} classes")
            st.caption(f"Top: {len(predictor.top_classes)} classes")
        else:
            st.warning("📊 Rule-based mode")
        
        st.markdown("---")
        st.markdown("### 🎿 Swix Wax Guide")
        st.markdown("""
        **Base Layer:**
        - KX20/KX30: Very cold
        - KX35/KX45: Cold
        - VG30/VG35: Universal
        - KX55+: Warm
        
        **Top Layer:**
        - V20/V30: Cold
        - V40: Mid
        - VP40/VP45: Racing mid
        - VP50: Warm
        """)
        
        st.markdown("---")
        st.markdown("*Powered by Sitka Science*")


if __name__ == "__main__":
    main()
