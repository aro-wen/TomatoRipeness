import streamlit as st
import numpy as np
import cv2
from PIL import Image
from joblib import load
import csv, io

# ----------------------
# App settings
# ----------------------
st.set_page_config(page_title="Tomato Ripeness (SVM + CIELAB)", page_icon="🍅", layout="wide")

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
@st.cache_resource
def load_model():
    model = load("svm_lab.joblib")
    le    = load("label_encoder.joblib")
    return model, le

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
# Load model
# ----------------------
try:
    model, le = load_model()
    st.success("Model loaded ✅")
    st.caption("**Classes:** " + ", ".join(map(str, le.classes_)))
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

pred_idx = model.predict(feats.reshape(1, -1))[0]
pred_label_raw = le.inverse_transform([pred_idx])[0]
display_label = canonical_label(pred_label_raw)

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

st.info(
    "Tip: If background color dominates, try a smaller center crop to focus on the tomato. "
    "Ripening usually increases **a*** (greener → redder) and often **b*** (more yellow/orange)."
)
