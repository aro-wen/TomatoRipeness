# tomatoClassifier.py
# ---------------------------------------------------------------
# Tomato ripeness classifier using CIELAB + SVM on Laboro Tomato.
# - Robustly finds annotation dir: "annotation/" or "annotations/"
# - Resolves images from train/val/test/images (and other common layouts)
# - Uses instance masks if available (polygons or RLE); falls back to bbox
# - Works WITHOUT pycocotools (polygon & uncompressed-RLE supported)
# - Plotting is optional; script won't crash if matplotlib missing
#
# Outputs:
#   - Prints metrics (Accuracy, Macro-F1, classification report)
#   - Saves: svm_lab.joblib, label_encoder.joblib
#
# Requirements:
#   numpy, opencv-python, scikit-learn, joblib, tqdm
#   (optional) matplotlib, pycocotools (or pycocotools-windows on Win)
# ---------------------------------------------------------------

from __future__ import annotations
import os, json, sys
from pathlib import Path
from functools import lru_cache
from typing import Dict, Tuple, List

print("Interpreter in use:", sys.executable)

# ---------- Optional imports ----------
try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False
    print("NOTE: matplotlib not installed; skipping plots.")

try:
    from pycocotools import mask as maskUtils  # type: ignore
    HAVE_COCO = True
except Exception:
    HAVE_COCO = False
    print("NOTE: pycocotools not installed; using polygon/uncompressed-RLE/bbox only.")

import cv2
import numpy as np
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from joblib import dump


# ===================== USER: set your dataset root here =====================
DATA_ROOT = Path(r"C:\Users\ACER NITRO 5 GAMING\.cache\kagglehub\datasets\nexuswho\laboro-tomato\versions\5")
# ===========================================================================


# ---------- Helpers: discover annotation dir & JSON files ----------
def pick_ann_dir(data_root: Path) -> Path:
    for name in ["annotations", "annotation", "Annotations", "Annotation"]:
        p = data_root / name
        if p.exists():
            return p
    raise SystemExit(f"No annotation directory found under {data_root}")

ANN_DIR = pick_ann_dir(DATA_ROOT)

# Prefer common COCO names; otherwise take all .json inside ann dir
PREFERRED_JSON = ["train.json","instances_train.json","val.json","instances_val.json",
                  "test.json","instances_test.json"]
ANN_FILES = [ANN_DIR / n for n in PREFERRED_JSON if (ANN_DIR / n).exists()]
if not ANN_FILES:
    ANN_FILES = sorted(ANN_DIR.glob("*.json"))
if not ANN_FILES:
    raise SystemExit(f"No JSON files found in {ANN_DIR}")

print("Using annotation dir:", ANN_DIR)
print("JSON files found:", [p.name for p in ANN_FILES])


# ---------- Image resolver: robust to many layouts ----------
CANDIDATE_DIRS: List[Path] = []
for sd in [
    "", "images", "image",
    "train", "train/images", "images/train",
    "val",   "val/images",   "images/val",
    "test",  "test/images",  "images/test",
]:
    p = DATA_ROOT / sd
    if p.exists():
        CANDIDATE_DIRS.append(p)

if not CANDIDATE_DIRS:
    raise SystemExit(f"No image directories found under {DATA_ROOT}. "
                     f"Expected e.g. 'train/images', 'val/images', or a flat 'images' folder.")

@lru_cache(maxsize=1)
def _file_index() -> Dict[str, str]:
    """Map lowercase filename -> absolute path, scanning candidate dirs once."""
    idx: Dict[str, str] = {}
    for d in CANDIDATE_DIRS:
        for root, _, files in os.walk(d):
            for f in files:
                fl = f.lower()
                if fl.endswith((".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff")):
                    idx.setdefault(fl, os.path.join(root, f))
    return idx

def resolve_image_path(fname: str) -> str | None:
    """
    Resolve an image filename from the JSON to an absolute path on disk.
    Handles:
      - Bare filename (e.g., IMG_123.jpg)
      - Relative subpath inside JSON (e.g., train/images/IMG_123.jpg)
      - Images living in train/images, val/images, test/images, images/train, etc.
    """
    fpath = Path(fname)

    # If JSON already includes a relative subpath, try it from DATA_ROOT
    p = DATA_ROOT / fpath
    if p.exists():
        return str(p)

    # Try each candidate dir with just the name part
    just_name = fpath.name
    for d in CANDIDATE_DIRS:
        p = d / just_name
        if p.exists():
            return str(p)

    # Last resort: case-insensitive filename index
    return _file_index().get(just_name.lower(), None)


# ---------- Mask building without (or with) pycocotools ----------
def ann_to_mask_generic(ann: dict, img_h: int, img_w: int) -> np.ndarray:
    """
    Convert a COCO-style annotation to a boolean mask.
    - If 'segmentation' is polygon(s): rasterize with OpenCV.
    - If 'segmentation' is uncompressed RLE (counts is list): decode it.
    - If 'segmentation' is compressed RLE (counts is str):
        -> if pycocotools available, decode; else fallback to bbox.
    - If anything fails, fallback to bbox.
    """
    seg = ann.get("segmentation", None)

    # 1) Polygon(s): list of lists [x1,y1,x2,y2,...]
    if isinstance(seg, list) and len(seg) > 0:
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        polys = []
        for poly in seg:
            if isinstance(poly, list) and len(poly) >= 6:
                pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
                pts = np.round(pts).astype(np.int32)
                polys.append(pts)
        if polys:
            cv2.fillPoly(mask, polys, 1)
            return mask.astype(bool)

    # 2) RLE dict
    if isinstance(seg, dict):
        counts = seg.get("counts", None)
        size = seg.get("size", None)

        # 2a) Uncompressed RLE (counts is list)
        if isinstance(counts, list) and isinstance(size, list) and len(size) == 2:
            rle_h, rle_w = int(size[0]), int(size[1])
            flat = np.zeros(rle_h * rle_w, dtype=np.uint8)
            val = 0
            idx = 0
            for c in counts:
                c = int(c)
                if c > 0:
                    flat[idx:idx + c] = val
                    idx += c
                    val = 1 - val
            flat = flat.reshape((rle_h, rle_w), order="F")  # COCO column-major
            if (rle_h, rle_w) != (img_h, img_w):
                flat = cv2.resize(flat, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
            return flat.astype(bool)

        # 2b) Compressed RLE (counts is str) — decode if we have pycocotools
        if HAVE_COCO and isinstance(counts, str):
            try:
                rle = seg
                m = maskUtils.decode(rle)
                if m.shape[0] != img_h or m.shape[1] != img_w:
                    m = cv2.resize(m, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                return (m > 0)
            except Exception:
                pass

    # 3) Fallback: bounding box
    x, y, w, h = ann.get("bbox", [0, 0, 0, 0])
    x0 = max(0, int(np.floor(x))); y0 = max(0, int(np.floor(y)))
    x1 = min(img_w, int(np.ceil(x + w))); y1 = min(img_h, int(np.ceil(y + h)))
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    mask[y0:y1, x0:x1] = 1
    return mask.astype(bool)


# ---------- Feature extraction ----------
def extract_lab_features(bgr_img: np.ndarray, mask_bool: np.ndarray | None) -> np.ndarray | None:
    """
    Compute mean & std of L*, a*, b* within mask. Returns (6,) or None if invalid.
    """
    if bgr_img is None or bgr_img.size == 0:
        return None
    if mask_bool is None:
        mask_bool = np.ones(bgr_img.shape[:2], dtype=bool)

    if mask_bool.sum() < 10:
        return None

    lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0][mask_bool].astype(np.float32)
    a = lab[:, :, 1][mask_bool].astype(np.float32)
    b = lab[:, :, 2][mask_bool].astype(np.float32)

    return np.array([
        float(np.mean(L)), float(np.std(L) + 1e-6),
        float(np.mean(a)), float(np.std(a) + 1e-6),
        float(np.mean(b)), float(np.std(b) + 1e-6),
    ], dtype=np.float32)


# ---------- Data loading per split ----------
def load_split(ann_file: Path) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str,int,str]]]:
    """
    Load features (X), labels (y_text), and metadata list from one COCO split file.
    Returns: X (N,6), y (N,), meta [(file_name, ann_id, class_name), ...]
    """
    split_name = ann_file.stem.lower()
    print(f"\nLoading {split_name} from {ann_file}")

    if not ann_file.exists():
        print(f"  -> {ann_file} not found, skipping.")
        return np.array([]), np.array([]), []

    # Read JSON directly (no need for COCO class)
    data = json.loads(ann_file.read_text())

    images = data.get("images", [])
    anns   = data.get("annotations", [])
    cats   = data.get("categories", [])
    print(f"[{split_name}] images in JSON: {len(images)} | anns in JSON: {len(anns)}")

    # category id -> name (lowercased, normalized)
    cat_map: Dict[int, str] = {}
    for c in cats:
        name = c.get("name", "").strip().lower()
        # Normalize common names
        if "green" in name:
            name = "green"
        elif "half" in name:
            name = "half_ripened"
        elif "full" in name:
            name = "fully_ripened"
        cat_map[int(c["id"])] = name

    # image_id -> (abs_image_path, width, height, file_name)
    id2img: Dict[int, Tuple[str | None, int | None, int | None, str]] = {}
    missing_paths = 0
    for img in images:
        img_id = int(img["id"])
        fname = img.get("file_name")
        full = resolve_image_path(fname) if fname else None
        id2img[img_id] = (full, img.get("width"), img.get("height"), fname)
        if not full:
            missing_paths += 1
    total_imgs = len(id2img)
    print(f"[{split_name}] resolved image paths: {total_imgs - missing_paths}/{total_imgs}")
    if missing_paths:
        print(f"[{split_name}] WARNING: {missing_paths} image(s) not found by filename (will be skipped).")

    X_list, y_list, meta = [], [], []

    for ann in tqdm(anns, total=len(anns), desc=f"Extract {split_name}"):
        img_id = int(ann["image_id"])
        if img_id not in id2img:
            continue
        img_path, w, h, fname = id2img[img_id]
        if not img_path or not os.path.exists(img_path):
            continue

        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        H, W = bgr.shape[:2]

        # Build mask
        m = ann_to_mask_generic(ann, H, W)
        feats = extract_lab_features(bgr, m)
        if feats is None:
            continue

        # Label
        cat_id = int(ann["category_id"])
        cname  = cat_map.get(cat_id, "unknown").lower()
        if cname not in {"green","half_ripened","fully_ripened"}:
            # last-chance normalization
            if "green" in cname:
                cname = "green"
            elif "half" in cname:
                cname = "half_ripened"
            elif "full" in cname:
                cname = "fully_ripened"

        X_list.append(feats)
        y_list.append(cname)
        meta.append((fname, int(ann.get("id", -1)), cname))

    if not X_list:
        return np.array([]), np.array([]), []

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list)
    return X, y, meta


def main():
    # Load all splits we have
    all_X, all_y, all_meta = [], [], []

    for annp in ANN_FILES:
        Xs, ys, metas = load_split(annp)
        if Xs.size and ys.size:
            all_X.append(Xs)
            all_y.append(ys)
            all_meta.extend(metas)

    if not all_y:
        print("No annotations found. Check your DATA_ROOT, annotation filenames, and image folders.")
        return

    X = np.vstack(all_X)
    y = np.concatenate(all_y)

    # Show class distribution
    uniq, counts = np.unique(y, return_counts=True)
    print("\nLabel counts:", dict(zip(uniq, counts)))

    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    print("Classes:", list(le.classes_))

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    # Pipeline + modest grid
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", class_weight="balanced", random_state=42)),
    ])
    param_grid = {
        "clf__C": [1, 3, 10, 30, 100],
        "clf__gamma": ["scale", 0.1, 0.03, 0.01],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gs = GridSearchCV(pipe, param_grid, scoring="f1_macro", cv=cv, n_jobs=-1, refit=True, verbose=1)
    gs.fit(X_train, y_train)

    print("\nBest params:", gs.best_params_)
    print("Best CV macro-F1: {:.4f}".format(gs.best_score_))

    # Evaluate
    best_model = gs.best_estimator_
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")

    print("\n=== Test Results ===")
    print("Accuracy:  {:.4f}".format(acc))
    print("Macro F1:  {:.4f}".format(f1m))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Confusion matrix (optional)
    if HAVE_MPL:
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6,5))
        plt.imshow(cm, interpolation="nearest")
        plt.title("Confusion Matrix (Test)")
        plt.xticks(range(len(le.classes_)), le.classes_, rotation=45, ha="right")
        plt.yticks(range(len(le.classes_)), le.classes_)
        for (i, j), v in np.ndenumerate(cm):
            plt.text(j, i, str(v), ha="center", va="center")
        plt.xlabel("Predicted"); plt.ylabel("True")
        plt.tight_layout(); plt.show()

    # Save
    dump(best_model, "svm_lab.joblib")
    dump(le, "label_encoder.joblib")
    print("\nSaved: svm_lab.joblib and label_encoder.joblib")

    # Example inference helper
    print("\nTip: Use predict_instance(bgr_img, mask_bool, model, label_encoder) for new samples.")

# ---------- Optional: simple inference helper ----------
def predict_instance(bgr_img: np.ndarray, mask_bool: np.ndarray | None, model, label_encoder):
    feats = extract_lab_features(bgr_img, mask_bool)
    if feats is None:
        return None, None
    pred = model.predict(feats.reshape(1, -1))[0]
    return label_encoder.inverse_transform([pred])[0], feats.ravel()


if __name__ == "__main__":
    main()
