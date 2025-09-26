import streamlit as st
import numpy as np
import cv2
from PIL import Image
from joblib import load
import csv, io

# Try to import Google Generative AI with error handling
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError as e:
    GEMINI_AVAILABLE = False
    st.warning("⚠️ Google Generative AI not available. Using static recommendations. Install with: `pip install google-generativeai`")

# ----------------------
# App settings
# ----------------------
st.set_page_config(page_title="Tomato Ripeness (SVM + CIELAB)", page_icon="🍅", layout="wide")

# ----------------------
# Gemini AI Configuration
# ----------------------
def get_gemini_api_key():
    """Get Gemini API key from Streamlit secrets or environment variables"""
    try:
        # Try Streamlit secrets first
        return st.secrets["GEMINI_API_KEY"]
    except:
        try:
            # Fallback to environment variable
            import os
            return os.getenv("GEMINI_API_KEY")
        except:
            return None

if GEMINI_AVAILABLE:
    try:
        GEMINI_API_KEY = get_gemini_api_key()
        if not GEMINI_API_KEY:
            GEMINI_AVAILABLE = False
            st.error("⚠️ Gemini API key not found. Please add GEMINI_API_KEY to Streamlit secrets or environment variables.")
        else:
            genai.configure(api_key=GEMINI_API_KEY)
        
        # Function to get available models for debugging
        def get_available_models():
            """Get list of available Gemini models"""
            try:
                models = genai.list_models()
                return [model.name for model in models if 'generateContent' in model.supported_generation_methods]
            except Exception as e:
                return [f"Error fetching models: {e}"]
                
        # Test API connectivity
        def test_gemini_api():
            """Test if Gemini API is accessible"""
            try:
                models = list(genai.list_models())
                return True, f"Found {len(models)} models"
            except Exception as e:
                return False, str(e)
                
    except Exception as config_error:
        GEMINI_AVAILABLE = False
        st.error(f"Failed to configure Gemini API: {config_error}")

# Minimal CSS for the prediction badge (white text for both themes)
st.markdown("""
<style>
.badge {
  display:inline-block; padding:6px 12px; border-radius:999px;
  font-weight:600; font-size:0.95rem; color:#fff; margin: 0 0 6px 0;
}
hr { margin: 0.6rem 0 1rem 0; }
</style>
""", unsafe_allow_html=True)

st.title("🍅 Tomato Ripeness Classifier (SVM + CIELAB)")
st.caption("Upload a photo. We extract CIELAB color features and predict ripeness using your trained SVM.")

# ----------------------
# Compatibility wrapper for st.image
# ----------------------
def show_image(img, caption=None):
    """Display image with compatibility for Streamlit versions."""
    try:
        st.image(img, caption=caption, use_container_width=True)
    except TypeError:
        st.image(img, caption=caption, use_column_width=True)

# ----------------------
# Model loader (cached)
# ----------------------
def load_model():
    """Load model without caching to avoid corruption issues"""
    try:
        model = load("svm_lab.joblib")
        le = load("label_encoder.joblib")
        
        # Validate the loaded objects
        if not hasattr(model, 'predict'):
            raise ValueError(f"Model is {type(model)}, expected sklearn model with predict method")
        
        if not hasattr(le, 'inverse_transform'):
            raise ValueError(f"Label encoder is {type(le)}, expected sklearn LabelEncoder")
            
        return model, le
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        st.error("Please check that svm_lab.joblib and label_encoder.joblib are valid scikit-learn objects")
        raise

def extract_lab_features(bgr_img, mask=None):
    """Compute mean & std of L*, a*, b* inside mask (or whole image if mask is None)."""
    if bgr_img is None or bgr_img.size == 0:
        return None
    H, W = bgr_img.shape[:2]
    if mask is None:
        mask_bool = np.ones((H, W), dtype=bool)
    else:
        mask_bool = (mask > 0) if mask.dtype != bool else mask
        if mask_bool.sum() < 10:
            return None

    lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0][mask_bool].astype(np.float32)
    a = lab[:, :, 1][mask_bool].astype(np.float32)
    b = lab[:, :, 2][mask_bool].astype(np.float32)

    return np.array([
        float(L.mean()), float(L.std() + 1e-6),
        float(a.mean()), float(a.std() + 1e-6),
        float(b.mean()), float(b.std() + 1e-6),
    ], dtype=np.float32)

def explain_lab(L_mean, a_mean, b_mean):
    """Human-friendly interpretation for CIELAB means."""
    a_delta = a_mean - 128.0
    b_delta = b_mean - 128.0

    if a_delta > 10:
        a_txt = f"a* = {a_mean:.1f} (>128) → **reddish** tendency"
    elif a_delta < -10:
        a_txt = f"a* = {a_mean:.1f} (<128) → **greenish** tendency"
    else:
        a_txt = f"a* = {a_mean:.1f} (≈128) → balanced red/green"

    if b_delta > 10:
        b_txt = f"b* = {b_mean:.1f} (>128) → **yellowish** tendency"
    elif b_delta < -10:
        b_txt = f"b* = {b_mean:.1f} (<128) → **bluish** tendency"
    else:
        b_txt = f"b* = {b_mean:.1f} (≈128) → balanced blue/yellow"

    L_txt = f"L* = {L_mean:.1f} → overall lightness (higher = brighter)"
    return f"- {a_txt}\n- {b_txt}\n- {L_txt}"

def center_crop(bgr, crop_pct):
    """Center-crop image to reduce background influence (crop_pct of shorter side)."""
    crop_pct = int(crop_pct)
    if crop_pct >= 100:
        H, W = bgr.shape[:2]
        return bgr, (0, 0, W, H)
    H, W = bgr.shape[:2]
    s = min(H, W)
    new_s = max(10, int(s * (crop_pct / 100.0)))
    cy, cx = H // 2, W // 2
    y0 = max(0, cy - new_s // 2)
    y1 = min(H, y0 + new_s)
    x0 = max(0, cx - new_s // 2)
    x1 = min(W, x0 + new_s)
    return bgr[y0:y1, x0:x1, :], (x0, y0, x1, y1)

# --- canonical label + color map ---
def canonical_label(s: str) -> str:
    """Normalize to exactly: Unripen / Half Ripened / Fully Ripened."""
    s = (s or "").strip().lower().replace("_", " ").replace("-", " ")
    if s in {"unripen", "unripe", "green"}:
        return "Unripen"
    if "half" in s:
        return "Half Ripened"
    if "full" in s or "ripe" in s:
        return "Fully Ripened"
    return s.title() if s else "Unknown"

COLOR_BY_LABEL = {
    "Unripen":       "#27ae60",  # green
    "Half Ripened":  "#f39c12",  # orange
    "Fully Ripened": "#e74c3c",  # red
}

def label_color(label: str) -> str:
    return COLOR_BY_LABEL.get(canonical_label(label), "#7f8c8d")

# ----------------------
# Harvest recommendation system
# ----------------------
def get_harvest_recommendations(ripeness_label: str) -> dict:
    """
    Provide harvest recommendations based on tomato ripeness for different scenarios.
    
    Args:
        ripeness_label: The predicted ripeness level (Unripen/Half Ripened/Fully Ripened)
    
    Returns:
        Dictionary containing recommendations for different travel/distribution scenarios
    """
    recommendations = {
        "Unripen": {
            "long_travel": {
                "recommendation": "✅ GO",
                "action": "Ship in ventilated containers",
                "shelf_life": "10 days",
                "color": "#27ae60"
            },
            "nearby_town": {
                "recommendation": "⚠️ RISKY",
                "action": "Market as green tomatoes",
                "shelf_life": "7 days",
                "color": "#f39c12"
            },
            "community": {
                "recommendation": "✅ GO",
                "action": "Use for cooking or ripening",
                "shelf_life": "5 days",
                "color": "#27ae60"
            }
        },
        "Half Ripened": {
            "long_travel": {
                "recommendation": "⚠️ RISKY",
                "action": "Use refrigerated transport",
                "shelf_life": "4 days",
                "color": "#f39c12"
            },
            "nearby_town": {
                "recommendation": "✅ GO",
                "action": "Sell to local markets",
                "shelf_life": "5 days",
                "color": "#27ae60"
            },
            "community": {
                "recommendation": "✅ GO",
                "action": "Share with neighbors",
                "shelf_life": "3 days",
                "color": "#27ae60"
            }
        },
        "Fully Ripened": {
            "long_travel": {
                "recommendation": "❌ NO",
                "action": "Process locally instead",
                "shelf_life": "1 day",
                "color": "#e74c3c"
            },
            "nearby_town": {
                "recommendation": "⚠️ RISKY",
                "action": "Sell immediately at discount",
                "shelf_life": "2 days",
                "color": "#f39c12"
            },
            "community": {
                "recommendation": "✅ GO",
                "action": "Consume or preserve today",
                "shelf_life": "1 day",
                "color": "#27ae60"
            }
        }
    }
    
    return recommendations.get(canonical_label(ripeness_label), {})

def format_recommendation_card(title: str, rec_data: dict) -> str:
    """Format a recommendation as a concise HTML card."""
    return f"""
    <div style="
        border-left: 4px solid {rec_data['color']};
        background: rgba(255,255,255,0.05);
        padding: 12px 16px;
        margin: 8px 0;
        border-radius: 6px;
    ">
        <h4 style="margin: 0 0 8px 0; color: {rec_data['color']}; font-size: 1rem;">{rec_data['recommendation']}</h4>
        <p style="margin: 4px 0; font-weight: 600; font-size: 0.9em;">📦 {rec_data['action']}</p>
        <p style="margin: 4px 0; font-size: 0.85em; color: #666;">⏱️ {rec_data['shelf_life']}</p>
    </div>
    """

# ----------------------
# AI-Powered Recommendation Generator
# ----------------------
@st.cache_data(show_spinner=False)
def generate_ai_recommendations(ripeness_label: str, l_mean: float, a_mean: float, b_mean: float) -> dict:
    """
    Generate intelligent harvest recommendations using Gemini AI based on tomato analysis.
    """
    # Check if Gemini is available
    if not GEMINI_AVAILABLE:
        return get_harvest_recommendations(ripeness_label)
    
    try:
        # Use a working model - prefer the newer 2.5-flash model
        working_models = ['gemini-2.5-flash', 'models/gemini-2.5-flash', 'models/gemini-1.5-flash', 'models/gemini-pro', 'gemini-pro']
        
        model = None
        for model_name in working_models:
            try:
                model = genai.GenerativeModel(model_name)
                break
            except Exception as model_error:
                continue
        
        if model is None:
            raise Exception(f"No working Gemini models found. Tried: {working_models}")
        
        prompt = f"""
        Based on tomato ripeness "{ripeness_label}" with color values L*={l_mean:.1f}, a*={a_mean:.1f}, b*={b_mean:.1f}, provide CONCISE harvest recommendations.
        
        Return JSON with 3 scenarios:
        {{
            "long_travel": {{
                "recommendation": "✅ GO" or "⚠️ RISKY" or "❌ NO",
                "action": "One actionable step",
                "shelf_life": "X days"
            }},
            "nearby_town": {{
                "recommendation": "✅ GO" or "⚠️ RISKY" or "❌ NO", 
                "action": "One actionable step",
                "shelf_life": "X days"
            }},
            "community": {{
                "recommendation": "✅ GO" or "⚠️ RISKY" or "❌ NO",
                "action": "One actionable step", 
                "shelf_life": "X days"
            }}
        }}
        
        Keep actions under 10 words. Be direct and practical.
        """
        
        response = model.generate_content(prompt)
        
        # Parse the JSON response
        import json
        recommendations_text = response.text.strip()
        
        # Clean up the response if it has markdown formatting
        if "```json" in recommendations_text:
            recommendations_text = recommendations_text.split("```json")[1].split("```")[0].strip()
        elif "```" in recommendations_text:
            recommendations_text = recommendations_text.split("```")[1].strip()
            
        recommendations = json.loads(recommendations_text)
        
        # Add colors based on recommendation type
        for scenario in recommendations:
            rec_text = recommendations[scenario]["recommendation"]
            if "✅" in rec_text or "GO" in rec_text:
                recommendations[scenario]["color"] = "#27ae60"  # green
            elif "⚠️" in rec_text or "RISKY" in rec_text:
                recommendations[scenario]["color"] = "#f39c12"  # orange
            else:  # ❌ NO
                recommendations[scenario]["color"] = "#e74c3c"  # red
                
        return recommendations
        
    except Exception as e:
        st.error(f"Error generating AI recommendations: {str(e)}")
        # Fallback to static recommendations
        return get_harvest_recommendations(ripeness_label)

# ----------------------
# AI-Powered Logistics Explanation Generator
# ----------------------
@st.cache_data(show_spinner=False)
def generate_logistics_explanation(ripeness_label: str, l_mean: float, a_mean: float, b_mean: float) -> dict:
    """
    Generate easy-to-read logistics explanations using AI based on tomato analysis.
    """
    # Check if Gemini is available
    if not GEMINI_AVAILABLE:
        return get_static_logistics_explanation(ripeness_label)
    
    try:
        # Use a working model
        working_models = ['gemini-2.5-flash', 'models/gemini-2.5-flash', 'models/gemini-1.5-flash', 'models/gemini-pro', 'gemini-pro']
        
        model = None
        for model_name in working_models:
            try:
                model = genai.GenerativeModel(model_name)
                break
            except Exception as model_error:
                continue
        
        if model is None:
            raise Exception(f"No working Gemini models found")
        
        prompt = f"""
        You are explaining tomato logistics to farmers in SIMPLE, EASY language.
        
        Tomato details:
        - Ripeness: {ripeness_label}
        - L*: {l_mean:.1f} (brightness)
        - a*: {a_mean:.1f} (green to red)  
        - b*: {b_mean:.1f} (blue to yellow)
        
        Explain in simple terms:
        1. WHY this ripeness affects transport
        2. BEST packaging method
        3. IDEAL temperature
        4. KEY timing considerations
        
        Return JSON:
        {{
            "why_matters": "Simple 1-sentence explanation why ripeness affects logistics",
            "packaging": "Best container/packaging method in simple terms", 
            "temperature": "Ideal temperature with simple reason",
            "timing": "Key timing advice in everyday language"
        }}
        
        Use everyday language like talking to a friend. Avoid technical jargon.
        """
        
        response = model.generate_content(prompt)
        
        # Parse the JSON response
        import json
        logistics_text = response.text.strip()
        
        # Clean up the response if it has markdown formatting
        if "```json" in logistics_text:
            logistics_text = logistics_text.split("```json")[1].split("```")[0].strip()
        elif "```" in logistics_text:
            logistics_text = logistics_text.split("```")[1].strip()
            
        logistics = json.loads(logistics_text)
                
        return logistics
        
    except Exception as e:
        st.error(f"Error generating logistics explanation: {str(e)}")
        # Fallback to static explanations
        return get_static_logistics_explanation(ripeness_label)

def get_static_logistics_explanation(ripeness_label: str) -> dict:
    """Fallback static logistics explanations"""
    explanations = {
        "Unripen": {
            "why_matters": "Green tomatoes are firm, so they can handle bumpy roads and tight packing without bruising.",
            "packaging": "Use sturdy wooden crates or cardboard boxes with good air holes for breathing.",
            "temperature": "Keep cool (55-60°F) to slow ripening and prevent rot during long trips.",
            "timing": "Start shipping early morning when it's cool, allows 7-10 days travel time."
        },
        "Half Ripened": {
            "why_matters": "These tomatoes are getting softer, so they need gentler handling to avoid damage.",
            "packaging": "Use padded containers with soft material between layers to prevent bruising.",
            "temperature": "Keep slightly cool (60-65°F) to control ripening speed during transport.",
            "timing": "Ship within 2-3 days max, plan for quick delivery to avoid overripening."
        },
        "Fully Ripened": {
            "why_matters": "Red tomatoes are very soft and bruise easily, making long transport very risky.",
            "packaging": "Use individual wrapping or soft padding, avoid stacking to prevent crushing.",
            "temperature": "Keep cold (45-50°F) to slow down spoiling, but not too cold to avoid damage.",
            "timing": "Sell locally within 1-2 days, avoid long transport unless absolutely necessary."
        }
    }
    
    return explanations.get(canonical_label(ripeness_label), explanations["Half Ripened"])

# ----------------------
# Load model
# ----------------------
try:
    model, le = load_model()
    st.success("Model loaded ✅")
    st.caption("**Classes:** " + ", ".join(map(str, le.classes_)))
    
    # Add model debugging information
    st.sidebar.markdown("**Model Info:**")
    st.sidebar.write(f"Model type: {type(model)}")
    st.sidebar.write(f"Label encoder classes: {le.classes_}")
    st.sidebar.write(f"Has predict method: {hasattr(model, 'predict')}")
        
except Exception as e:
    st.error("Could not load model files. Ensure `svm_lab.joblib` and `label_encoder.joblib` are in the app folder.")
    st.code(str(e))
    st.stop()

# ----------------------
# Sidebar options
# ----------------------
with st.sidebar:
    st.header("Options")
    crop_mode = st.radio("Region", ["Center crop", "Whole image"], index=0, help="Use center crop to reduce background influence.")
    crop_pct = st.slider("Center crop size (% of shorter side)", 30, 100, 80, 5, disabled=(crop_mode == "Whole image"))
    show_crop_box = st.checkbox("Show crop box overlay", value=True, disabled=(crop_mode == "Whole image"))
    st.info("CIELAB: a* ↑ → redder, a* ↓ → greener; b* ↑ → yellower, b* ↓ → bluer; L* = lightness.")

# ----------------------
# Main layout
# ----------------------
left, right = st.columns([1.1, 1.2], gap="large")

with left:
    st.subheader("1) Upload tomato image")
    up_img = st.file_uploader("Choose a JPEG/PNG", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    if not up_img:
        st.info("Upload a photo of a tomato to classify.")
        st.stop()
    rgb = np.array(Image.open(up_img).convert("RGB"))
    show_image(rgb, caption="Uploaded image")

# Processing
bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
if crop_mode == "Center crop":
    bgr_crop, (x0, y0, x1, y1) = center_crop(bgr, crop_pct)
else:
    H, W = bgr.shape[:2]
    bgr_crop, (x0, y0, x1, y1) = bgr, (0, 0, W, H)

feats = extract_lab_features(bgr_crop, mask=None)
if feats is None:
    st.error("Could not extract features (crop may be too small). Try whole image or increase crop size.")
    st.stop()

# Add debugging information for features
st.sidebar.markdown("**Debug Info:**")
st.sidebar.write(f"Features shape: {feats.shape}")
st.sidebar.write(f"Features: {feats}")

# Try prediction with error handling - reload model fresh to avoid cache issues
try:
    # Reload model fresh for prediction to avoid cache corruption
    fresh_model, fresh_le = load_model()
    
    # Verify model is valid before prediction
    if not hasattr(fresh_model, 'predict'):
        raise ValueError(f"Fresh model is {type(fresh_model)}, not a valid sklearn model")
    
    # Make prediction
    pred_idx = fresh_model.predict(feats.reshape(1, -1))[0]
    pred_label_raw = fresh_le.inverse_transform([pred_idx])[0]
    display_label = canonical_label(pred_label_raw)
    
    st.sidebar.success(f"✅ Prediction successful: {display_label}")
    
except Exception as e:
    st.error(f"Prediction failed: {str(e)}")
    st.error(f"Feature shape: {feats.shape}, Feature values: {feats}")
    st.error("This might be due to model corruption or version mismatch.")
    
    # Try to reload and test with simple features
    try:
        test_model, _ = load_model()
        st.info(f"Model reloaded as: {type(test_model)}")
        if hasattr(test_model, 'predict'):
            test_feats = np.array([[50, 1, 128, 1, 128, 1]], dtype=np.float32)
            test_pred = test_model.predict(test_feats)
            st.success(f"Test prediction successful: {test_pred}")
        else:
            st.error("Reloaded model still doesn't have predict method")
    except Exception as test_e:
        st.error(f"Even model reload failed: {test_e}")
    
    st.stop()

L_mean, L_std, a_mean, a_std, b_mean, b_std = feats

# Build CSV in-memory
csv_buffer = io.StringIO()
writer = csv.writer(csv_buffer)
writer.writerow([
    "predicted_label","L_mean","L_std","a_mean","a_std","b_mean","b_std",
    "crop_x0","crop_y0","crop_x1","crop_y1","region"
])
writer.writerow([
    display_label, f"{L_mean:.6f}", f"{L_std:.6f}", f"{a_mean:.6f}", f"{a_std:.6f}",
    f"{b_mean:.6f}", f"{b_std:.6f}", int(x0), int(y0), int(x1), int(y1),
    "center_crop" if crop_mode == "Center crop" else "whole_image"
])
csv_bytes = csv_buffer.getvalue().encode("utf-8")

with right:
    st.subheader("2) Prediction & features")

    # Prediction badge
    st.markdown(
        f"<span class='badge' style='background:{label_color(display_label)}'>"
        f"Predicted: {display_label}</span>",
        unsafe_allow_html=True,
    )
    st.caption("Model: SVM (RBF) on CIELAB stats")

    # Metrics
    colB, colC, colD = st.columns(3)
    with colB:
        st.metric("L* mean", f"{L_mean:.1f}")
        st.caption("Lightness")
    with colC:
        st.metric("a* mean", f"{a_mean:.1f}")
        st.caption("Green ↔ Red")
    with colD:
        st.metric("b* mean", f"{b_mean:.1f}")
        st.caption("Blue ↔ Yellow")

    # Visuals
    vis_col1, vis_col2 = st.columns(2)
    with vis_col1:
        disp = rgb.copy()
        if crop_mode == "Center crop" and show_crop_box:
            cv2.rectangle(disp, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 2)
        show_image(disp, caption="Original (with crop overlay)")
    with vis_col2:
        crop_rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
        show_image(crop_rgb, caption="Analyzed region")

    # Features + explanation
    with st.expander("See detailed CIELAB features", expanded=False):
        st.write(f"- **L*** mean: `{L_mean:.2f}` | std: `{L_std:.2f}`")
        st.write(f"- **a*** mean: `{a_mean:.2f}` | std: `{a_std:.2f}`")
        st.write(f"- **b*** mean: `{b_mean:.2f}` | std: `{b_std:.2f}`")
    st.markdown("#### Why this prediction?")
    st.markdown(explain_lab(L_mean, a_mean, b_mean))

    st.download_button(
        "Download features & prediction (CSV)",
        data=csv_bytes,
        file_name="tomato_ripeness_features.csv",
        mime="text/csv",
    )

# ----------------------
# Harvest Recommendations Section
# ----------------------
st.markdown("---")
if GEMINI_AVAILABLE:
    st.markdown("### 🤖 AI-Powered Harvest Recommendations")
    st.caption("Based on the predicted ripeness and CIELAB color analysis, here are intelligent recommendations for different distribution scenarios:")
    
    # Generate AI recommendations with a loading spinner
    with st.spinner("🧠 Generating intelligent recommendations..."):
        recommendations = generate_ai_recommendations(display_label, L_mean, a_mean, b_mean)
else:
    st.markdown("### 🚚 Harvest Recommendations")
    st.caption("Based on the predicted ripeness, here are recommendations for different distribution scenarios:")
    
    # Use static recommendations
    recommendations = get_harvest_recommendations(display_label)

if recommendations:
    # Create three columns for the different scenarios
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Long Distance Travel")
        if "long_travel" in recommendations:
            rec = recommendations["long_travel"]
            st.markdown(format_recommendation_card("Long Distance Transport", rec), unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### Nearby Town/Market")
        if "nearby_town" in recommendations:
            rec = recommendations["nearby_town"]
            st.markdown(format_recommendation_card("Regional Distribution", rec), unsafe_allow_html=True)
    
    with col3:
        st.markdown("#### Local Community")
        if "community" in recommendations:
            rec = recommendations["community"]
            st.markdown(format_recommendation_card("Community Consumption", rec), unsafe_allow_html=True)
    
    # Additional harvest insights
    st.markdown("#### 📋 Smart Harvest Strategy")
    
    # Generate overall strategy based on AI analysis
    if display_label == "Unripen":
        st.success("🟢 **Optimal Strategy:** Maximum flexibility for distribution and storage")
        st.info(f"💡 **Color Analysis:** a*={a_mean:.1f} indicates {'strong green' if a_mean < 120 else 'moderate green'} coloration - excellent for controlled ripening")
    elif display_label == "Half Ripened":
        st.warning("🟡 **Optimal Strategy:** Focus on regional markets and quick turnaround")
        st.info(f"💡 **Color Analysis:** a*={a_mean:.1f} shows transitioning color - ideal timing for 2-3 day distribution window")
    elif display_label == "Fully Ripened":
        st.error("🔴 **Optimal Strategy:** Immediate processing or local consumption priority")
        st.info(f"💡 **Color Analysis:** a*={a_mean:.1f} indicates peak ripeness - maximum flavor but minimal shelf life")
    
    # Add a note about the AI analysis
    st.markdown("---")
    st.caption("🤖 These recommendations are generated using AI analysis of your specific tomato's color characteristics and ripeness level.")
else:
    st.warning("Unable to generate recommendations. Please try uploading a clearer image of the tomato.")

# ----------------------
# AI-Powered Logistics Explanation Section  
# ----------------------
st.markdown("---")
if GEMINI_AVAILABLE:
    st.markdown("### 🚛 AI-Powered Logistics Guide")
    st.caption("Easy-to-understand explanations about the best ways to transport and handle your tomatoes:")
    
    # Generate logistics explanation with a loading spinner
    with st.spinner("🔄 Analyzing optimal logistics..."):
        logistics = generate_logistics_explanation(display_label, L_mean, a_mean, b_mean)
else:
    st.markdown("### 🚛 Logistics Guide")
    st.caption("Easy-to-understand explanations about the best ways to transport and handle your tomatoes:")
    
    # Use static logistics explanations
    logistics = get_static_logistics_explanation(display_label)

if logistics:
    # Create a clean, easy-to-read layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🤔 Why This Matters")
        st.info(f"💡 {logistics['why_matters']}")
        
        st.markdown("#### 📦 Best Packaging")
        st.success(f"📦 {logistics['packaging']}")
    
    with col2:
        st.markdown("#### 🌡️ Temperature Tips")
        st.warning(f"🌡️ {logistics['temperature']}")
        
        st.markdown("#### ⏰ Timing Advice")
        st.error(f"⏰ {logistics['timing']}")
    
    # Add a summary box
    st.markdown("#### 📝 Quick Summary")
    
    # Color-code the summary based on ripeness
    if display_label == "Unripen":
        summary_color = "success"
        summary_icon = "🟢"
        summary_text = "**Great for logistics!** Green tomatoes are tough and travel well with proper care."
    elif display_label == "Half Ripened":
        summary_color = "warning"
        summary_icon = "🟡"
        summary_text = "**Moderate logistics challenge.** Handle gently and move quickly for best results."
    else:  # Fully Ripened
        summary_color = "error"
        summary_icon = "🔴"
        summary_text = "**High logistics risk!** Keep local, handle very carefully, and sell fast."
    
    if summary_color == "success":
        st.success(f"{summary_icon} {summary_text}")
    elif summary_color == "warning":
        st.warning(f"{summary_icon} {summary_text}")
    else:
        st.error(f"{summary_icon} {summary_text}")
        
else:
    st.warning("Unable to generate logistics explanation. Please try uploading a clearer image of the tomato.")

st.info(
    "Tip: If background color dominates, try a smaller center crop to focus on the tomato. "
    "Ripening usually increases **a*** (greener → redder) and often **b*** (more yellow/orange)."
)
