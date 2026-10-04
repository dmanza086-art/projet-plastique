from pathlib import Path
from collections import Counter
import hashlib
import cv2
import numpy as np
import pandas as pd

ROOT = next(Path("data/raw").glob("Dal Lake*"))
IMG_DIR, MSK_DIR = ROOT / "Raw_Images", ROOT / "Segmentation_Masks"

imgs = {p.stem: p for p in IMG_DIR.glob("*.jpg")}
msks = {p.stem.removesuffix("_mask"): p for p in MSK_DIR.glob("*.png")}
names = sorted(set(imgs) | set(msks), key=lambda s: int(s[3:]))

rows, anomalies = [], []
val_counter, img_sizes, msk_sizes = Counter(), Counter(), Counter()
hashes = {}

for n in names:
    if n not in imgs:
        anomalies.append((n, "masque sans image")); continue
    if n not in msks:
        anomalies.append((n, "image sans masque")); continue

    im = cv2.imread(str(imgs[n]))
    m = cv2.imread(str(msks[n]), cv2.IMREAD_UNCHANGED)
    if im is None:
        anomalies.append((n, "image illisible")); continue
    if m is None:
        anomalies.append((n, "masque illisible")); continue
    if m.ndim == 3:
        anomalies.append((n, f"masque a {m.shape[2]} canaux"))
        m = m.max(axis=2)

    img_sizes[im.shape[:2]] += 1
    msk_sizes[m.shape[:2]] += 1
    if im.shape[:2] != m.shape[:2]:
        anomalies.append((n, f"tailles differentes {im.shape[:2]} / {m.shape[:2]}"))

    vals = tuple(int(v) for v in np.unique(m))
    val_counter[vals] += 1
    hashes.setdefault(hashlib.md5(imgs[n].read_bytes()).hexdigest(), []).append(n)

    rows.append({"name": n, "h": im.shape[0], "w": im.shape[1],
                 "mask_values": str(vals), "plastic_pct": 100 * (m > 0).mean()})

for group in hashes.values():
    if len(group) > 1:
        anomalies.append((",".join(group), "images identiques (doublon exact)"))

df = pd.DataFrame(rows)
Path("data/interim").mkdir(parents=True, exist_ok=True)
df.to_csv("data/interim/stats_brutes.csv", index=False)
pd.DataFrame(anomalies, columns=["element", "probleme"]).to_csv("data/cleaning_log.csv", index=False)

print("Images :", len(imgs), "| Masques :", len(msks))
print("Dimensions images (H, W) :", dict(img_sizes))
print("Dimensions masques (H, W) :", dict(msk_sizes))
print("Valeurs presentes dans les masques :", dict(val_counter))
print("Anomalies :", len(anomalies), "(voir data/cleaning_log.csv)")
print("Images avec plastique (masque non vide) :", (df.plastic_pct > 0).sum())
print("Couverture moyenne (%) :", round(df.plastic_pct.mean(), 3))

reg = pd.read_csv(next(ROOT.glob("Mask_foreground*.csv")))
reg["name"] = reg["image name"].str.replace(".jpg", "", regex=False)
reg = reg.rename(columns={"plastic waste accumulation (in percentage)": "csv_pct"})
mg = df.merge(reg[["name", "csv_pct"]], on="name")
print("Lignes comparees avec le CSV :", len(mg))
print("Ecart absolu moyen CSV vs masque :", round((mg.plastic_pct - mg.csv_pct).abs().mean(), 4))
print("Correlation :", round(mg.plastic_pct.corr(mg.csv_pct), 4))
