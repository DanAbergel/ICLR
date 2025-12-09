import os
import shutil
import random
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import sys
import time

def log(msg):
    print(msg)
    sys.stdout.flush()
    time.sleep(0.01)  # pour forcer l'affichage même en batch SLURM


# ================================================================
# 0. CLEANUP DES SUJETS INCOMPLETS
# ================================================================
BASE_DIR = "/sci/labs/arieljaffe/dan.abergel1/HCP_data"

log("===== CLEANUP DES SUJETS INCOMPLETS =====")

valid_subject_paths = []
valid_subject_folders = []

subjects = os.listdir(BASE_DIR)
log(f"Nombre total d'entrées dans BASE_DIR : {len(subjects)}")

for i, subject in enumerate(subjects):
    subj_dir = os.path.join(BASE_DIR, subject)
    log(f"[{i+1}/{len(subjects)}] Inspection du dossier : {subject}")

    if not os.path.isdir(subj_dir):
        log(" → Ignoré (pas un dossier)")
        continue

    nifti_path = os.path.join(
        subj_dir,
        "MNINonLinear", "Results", "rfMRI_REST1_LR", "rfMRI_REST1_LR.nii.gz"
    )

    if os.path.exists(nifti_path):
        log(" → OK, NIFTI trouvé")
        valid_subject_paths.append(nifti_path)
        valid_subject_folders.append(subj_dir)
    else:
        log(" → ❌ NIFTI manquant — suppression du dossier")
        shutil.rmtree(subj_dir, ignore_errors=True)

log(f"Sujets valides conservés : {len(valid_subject_paths)}")
log("==============================================================\n")

subjects_paths = valid_subject_paths


# ================================================================
# 1. FONCTIONS
# ================================================================
def compute_global_signal(fmri_path):
    log(f"  Chargement NIFTI : {fmri_path}")
    img = nib.load(fmri_path)
    data = img.get_fdata()

    log("  Calcul brain_mask…")
    brain_mask = data.mean(axis=-1) != 0

    log("  Calcul global_signal…")
    global_signal = data[brain_mask].mean(axis=0)

    log("  Normalisation z-score…")
    gs_z = (global_signal - global_signal.mean()) / global_signal.std()

    log("  → global_signal calculé")
    return gs_z


def estimate_stabilization_time(global_signal, window_size=20, epsilon=0.05, min_consecutive=3):
    gs = global_signal
    T = len(gs)
    max_t = T - 2 * window_size

    if max_t <= 0:
        log("  → WARNING : série trop courte")
        return 0

    stable_count = 0
    for t in range(max_t):
        w1 = gs[t : t + window_size].mean()
        w2 = gs[t + window_size : t + 2 * window_size].mean()
        diff = abs(w2 - w1)

        if diff < epsilon:
            stable_count += 1
            if stable_count >= min_consecutive:
                log(f"  → Stabilisation trouvée à t={t}")
                return max(0, t - (min_consecutive - 1) * window_size)
        else:
            stable_count = 0

    log("  → Aucune stabilisation détectée, retour 0")
    return 0


# ================================================================
# 2. VISUALISATION DE 20 SUJETS
# ================================================================
log("===== VISUALISATION DE 20 SUJETS =====")
log(f"Nombre total de sujets valides : {len(subjects_paths)}")

random_paths = random.sample(subjects_paths, min(20, len(subjects_paths)))

plt.figure(figsize=(14, 10))

for idx, sp in enumerate(random_paths):
    log(f"Plot sujet {idx+1}/{len(random_paths)}")
    gs = compute_global_signal(sp)
    plt.plot(gs, alpha=0.6)

plt.axvline(100, linestyle="--", color="black", label="T=100")
plt.axvline(150, linestyle="--", color="red", label="T=150")
plt.title("Global signal (z-score) — 20 sujets aléatoires")
plt.xlabel("TR (t)")
plt.ylabel("Global signal (z-score)")
plt.legend()
plt.show()
log("===== FIN VISUALISATION =====\n")


# ================================================================
# 3. CALCUL T_stab POUR TOUS LES SUJETS
# ================================================================
log("===== CALCUL DES T_stab =====")
all_T_stab = []

for idx, sp in enumerate(subjects_paths):
    log(f"[{idx+1}/{len(subjects_paths)}] Sujet : {sp}")
    try:
        gs = compute_global_signal(sp)
        T_stab = estimate_stabilization_time(gs)
        all_T_stab.append(T_stab)
        log(f"  → T_stab = {T_stab}")
    except Exception as e:
        log(f"  ❌ ERREUR sur {sp} : {e}")

log("===== FIN CALCUL T_stab =====\n")


# ================================================================
# 4. AFFICHAGE HISTOGRAMME
# ================================================================
log("===== HISTOGRAMME DES T_stab =====")

plt.figure(figsize=(10, 6))
plt.hist(all_T_stab, bins=40, color="skyblue", edgecolor="black")
plt.axvline(np.percentile(all_T_stab, 90), color="red", linestyle="--", label="90e percentile")
plt.axvline(np.percentile(all_T_stab, 95), color="green", linestyle="--", label="95e percentile")
plt.title("Distribution des T_stab pour tous les sujets")
plt.xlabel("T_stab")
plt.ylabel("Nombre de sujets")
plt.legend()
plt.show()

T90 = int(np.percentile(all_T_stab, 90))
T95 = int(np.percentile(all_T_stab, 95))
Tmedian = int(np.median(all_T_stab))

log("\n==============================")
log(" Résultats finaux ")
log("==============================")
log(f"Median T_stab : {Tmedian}")
log(f"90e percentile : {T90}")
log(f"95e percentile : {T95}")
log("==============================\n")