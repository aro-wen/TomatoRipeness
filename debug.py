from pathlib import Path
import json, os

DATA_ROOT = Path(r"C:\Users\ACER NITRO 5 GAMING\.cache\kagglehub\datasets\nexuswho\laboro-tomato\versions\5")
ANN_DIR = DATA_ROOT / "annotations"

print("DATA_ROOT:", DATA_ROOT)
print("ANN_DIR exists:", ANN_DIR.exists())

jsons = sorted(ANN_DIR.glob("*.json"))
print("Found JSONs:", [p.name for p in jsons])

def candidates(fname):
    # places the Kaggle releases commonly use
    subdirs = [
        "", "train", "val", "test", "images",
        "train/images", "val/images", "test/images",
        "train/imgs", "val/imgs", "test/imgs"
    ]
    for sd in subdirs:
        p = (DATA_ROOT / sd / fname)
        if p.exists():
            return str(p)
    # last resort: walk (can be slow, but ok for a few files)
    for root, _, files in os.walk(DATA_ROOT):
        if fname in files:
            return str(Path(root) / fname)
    return None

for annp in jsons[:2]:  # check 2 jsons
    print("\n== Checking:", annp.name)
    jj = json.loads(annp.read_text())
    print("images:", len(jj.get("images", [])), "| annotations:", len(jj.get("annotations", [])))
    if jj.get("images"):
        samp = jj["images"][:3]
        for im in samp:
            fn = im["file_name"]
            path = candidates(fn)
            print("  file_name:", fn, "->", ("FOUND" if path else "MISSING"), path or "")
