"""
created by Dan Abergel
date: 25/09/2025
"""

import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Charger le fichier NIfTI ---
# Remplace ce chemin par celui où ton fichier est situé
nii_path = "/Users/danabergel/Documents/HUJI-Data_science/Thesis-Ariel-Yaffe/Project/Original_data/subjects/subject_101410/FMRI_Rest/MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR.nii.gz"

img = nib.load(nii_path)
data = img.get_fdata()  # tableau numpy 4D (X, Y, Z, T)

print("Shape du tensor fMRI :", data.shape)  # (x, y, z, time)

# --- 2. Choisir un instant t et un layer (slice z) ---
t = 100   # par exemple, volume 100 (tu peux changer)
z = 45    # par exemple, coupe au milieu du cerveau

# Vérification pour éviter les erreurs d'index
if t >= data.shape[3]:
    raise ValueError(f"t={t} est trop grand, max = {data.shape[3]-1}")
if z >= data.shape[2]:
    raise ValueError(f"z={z} est trop grand, max = {data.shape[2]-1}")

# --- 3. Extraire l'image correspondante ---
slice_2d = data[:, :, z, t]

# --- 4. Afficher ---
plt.figure(figsize=(6, 6))
plt.imshow(np.rot90(slice_2d), cmap="gray")
plt.title(f"Slice z={z} à t={t}")
plt.axis("off")
plt.colorbar(label="Intensité BOLD")
plt.show()