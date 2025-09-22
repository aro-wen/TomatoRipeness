import streamlit as st
import numpy as np
import cv2
from PIL import Image
from joblib import load

# ----------------------
# App settings
# ----------------------
st.set_page_config(page_title="Tomato Ripeness (SVM + CIELAB)", page_icon="🍅", layout="centered")
st.title("🍅 Tomato Ripeness Classifier (SVM + CIELAB)")
st.caption("Upload an image. The app extracts CIELAB color features and predicts ripeness using your trained SVM.")

# ----------------------
# Model loader (cached)
# ----------------------
@st.cache_resource
def load_model():
    model = load("svm_lab.joblib")
    le    = load("label_encoder.joblib")
    return model, le

def extract_lab_features(bgr_img, mask=None):
    """
    Compute mean & std of L*, a*, b* inside mask (or whole image if mask is None).
    OpenCV's LAB ranges: 0..255 for all channels (a*, b* ~128 is neutral).
    Returns (6,) [Lmean, Lstd, amean, astd, bmean, bstd].
    """
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

    feats = np.array([
        float(L.mean()), float(L.std() + 1e-6),
        float(a.mean()), float(a.std() + 1e-6),
        float(b.mean()), float(b.std() + 1e-6),
    ], dtype=np.float32)
    return feats

def explain_lab(L_mean, a_mean, b_mean):
    """
    Human-friendly interpretation:
    - a*: green (low) ↔ red (high), ~128 neutral
    - b*: blue (low) ↔ yellow (high), ~128 neutral
    - L*: lightness
    """
    a_delta = a_mean - 128.0
    b_delta = b_mean - 128.0

    if a_delta > 10:
        a_txt = f"a*={a_mean:.1f} (>128) → **reddish** tendency"
    elif a_delta < -10:
        a_txt = f"a*={a_mean:.1f} (<128) → **greenish** tendency"
    else:
        a_txt = f"a*={a_mean:.1f} (≈128) → balanced red/green"

    if b_delta > 10:
        b_txt = f"b*={b_mean:.1f} (>128) → **yellowish** tendency"
    elif b_delta < -10:
        b_txt = f"b*={b_mean:.1f} (<128) → **bluish** tendency"
    else:
        b_txt = f"b*={b_mean:.1f} (≈128) → balanced blue/yellow"

    L_txt = f"L*={L_mean:.1f} → overall lightness (higher = brighter)"

    return f"- {a_txt}\n- {b_txt}\n- {L_txt}"

def center_crop(bgr, crop_pct):
    """
    Center-crop the image to reduce background influence.
    crop_pct is the *shorter-side* percentage to keep (e.g., 60 keeps 60%).
    Returns: cropped image (BGR), and (x0,y0,x1,y1) in original coords.
    """
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

# ----------------------
# Load model
# ----------------------
try:
    model, le = load_model()
    st.success("Model loaded ✅")
    st.caption("Classes: " + ", ".join(map(str, le.classes_)))
except Exception as e:
    st.error("Could not load model files. Ensure 'svm_lab.joblib' and 'label_encoder.joblib' are in the app folder.")
    st.code(str(e))
    st.stop()

# ----------------------
# Sidebar controls (no camera, no blobs)
# ----------------------
st.sidebar.header("Options")
crop_pct = st.sidebar.slider("Center crop (% of shorter side)", 30, 100, 80, 5)
show_crop_box = st.sidebar.checkbox("Show crop box overlay", value=True)

# ----------------------
# Upload image only
# ----------------------
st.subheader("Upload a tomato image")
up_img = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if not up_img:
    st.info("Upload a photo of a tomato to classify.")
    st.stop()

# Convert to OpenCV BGR
rgb = np.array(Image.open(up_img).convert("RGB"))
bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

# Optional center crop to reduce background
bgr_crop, (x0, y0, x1, y1) = center_crop(bgr, crop_pct)

# Features (whole crop region)
feats = extract_lab_features(bgr_crop, mask=None)
if feats is None:
    st.error("Could not extract features from the image (crop too small?). Try a larger crop.")
    st.stop()

# Predict
pred_idx = model.predict(feats.reshape(1, -1))[0]
pred_label = le.inverse_transform([pred_idx])[0]

# Display image (with optional crop box overlay)
disp = rgb.copy()
if show_crop_box:
    cv2.rectangle(disp, (x0, y0), (x1, y1), (0, 255, 0), 2)
st.subheader("Result")
st.image(disp, caption=f"Predicted ripeness: {pred_label}", use_container_width=True)

# Show features + explanation
L_mean, L_std, a_mean, a_std, b_mean, b_std = feats
st.write("### CIELAB features (on cropped region)")
st.table({
    "Feature": ["L* mean", "L* std", "a* mean", "a* std", "b* mean", "b* std"],
    "Value":   [f"{L_mean:.2f}", f"{L_std:.2f}", f"{a_mean:.2f}", f"{a_std:.2f}", f"{b_mean:.2f}", f"{b_std:.2f}"]
})

st.write("### Why this prediction? (CIELAB interpretation)")
st.markdown(explain_lab(L_mean, a_mean, b_mean))

st.caption(
    "Tip: Adjust the center crop to focus on the tomato and reduce background influence. "
    "As tomatoes ripen, **a*** typically increases (greener → redder) and **b*** often rises (more yellow/orange)."
)
