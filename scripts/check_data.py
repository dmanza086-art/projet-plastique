"""Contrôle des données préparées (phase 1).
Lancer depuis n'importe où : python scripts/check_data.py"""
import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

RACINE = Path(__file__).resolve().parents[1]
IMAGES = RACINE / "data" / "processed" / "images"
MASQUES = RACINE / "data" / "processed" / "masks"
SPLITS = RACINE / "data" / "splits"
TAILLE = 512
CIBLES = {"train": 0.70, "val": 0.15, "test": 0.15}
erreurs = []

# 1. Fichiers de découpage
splits = {}
for k in CIBLES:
    f = SPLITS / f"{k}.csv"
    if not f.exists():
        erreurs.append(f"{f} introuvable")
        continue
    d = pd.read_csv(f)
    if "name" not in d.columns:
        erreurs.append(f"{f.name} : colonne 'name' absente")
        continue
    if d["name"].duplicated().any():
        erreurs.append(f"{f.name} : noms en double")
    splits[k] = d

if erreurs or len(splits) < 3:
    print("PROBLEMES :")
    for e in erreurs:
        print(" -", e)
    sys.exit(1)

# 2. Aucune image commune entre les ensembles
ens = {k: set(d["name"]) for k, d in splits.items()}
for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
    commun = ens[a] & ens[b]
    if commun:
        erreurs.append(f"{len(commun)} images communes entre {a} et {b}")

# 3. Nombres et proportions
total = sum(len(v) for v in ens.values())
n_img = len(list(IMAGES.glob("*.jpg")))
n_msk = len(list(MASQUES.glob("*.png")))
print(f"Images : {n_img} | Masques : {n_msk} | Lignes des CSV : {total}")
if not (n_img == n_msk == total):
    erreurs.append("le nombre d'images, de masques et de lignes de CSV diffère")
for k, v in ens.items():
    part = len(v) / total
    print(f"  {k:5s} : {len(v)} ({100 * part:.1f} %)")
    if abs(part - CIBLES[k]) > 0.05:
        erreurs.append(f"{k} : proportion {100 * part:.1f} % trop éloignée de {100 * CIBLES[k]:.0f} %")

# 4. Tailles et valeurs, paire par paire
for k, noms in ens.items():
    for nom in sorted(noms):
        im = cv2.imread(str(IMAGES / f"{nom}.jpg"))
        m = cv2.imread(str(MASQUES / f"{nom}.png"), cv2.IMREAD_UNCHANGED)
        if im is None or m is None:
            erreurs.append(f"{nom} ({k}) : image ou masque manquant")
            continue
        if im.shape != (TAILLE, TAILLE, 3):
            erreurs.append(f"{nom} : image de forme {im.shape}")
        if m.shape != (TAILLE, TAILLE):
            erreurs.append(f"{nom} : masque de forme {m.shape}")
        elif not set(np.unique(m).tolist()) <= {0, 1}:
            erreurs.append(f"{nom} : valeurs de masque {np.unique(m).tolist()}")

if erreurs:
    print(f"{len(erreurs)} PROBLEME(S) :")
    for e in erreurs[:20]:
        print(" -", e)
    sys.exit(1)
print("TOUT EST OK")
