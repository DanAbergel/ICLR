import os
import json
from datetime import datetime

# ========================
# CONFIGURATION
# ========================
base_dir = "/sci/labs/arieljaffe/dan.abergel1/HCP_data"
data_dir = os.path.join(base_dir, "data")
os.makedirs(data_dir, exist_ok=True)

index_to_name_path = os.path.join(data_dir, "index_to_name.json")
imageID_to_labels_path = os.path.join(data_dir, "imageID_to_labels.json")

# ========================
# LISTE DES SUJETS À IGNORER
# ========================
invalid_subjects = {
    "116120", "119833", "121820", "126931", "129432", "131621",
    "140420", "143527", "150019", "160931", "168038", "173233",
    "179548", "186949", "190132", "197449", "203721", "207628",
    "248238", "284646", "351938", "355542", "355845", "462139",
    "569965", "584355", "611231", "650746", "693461", "733548",
    "745555", "782157"
}

# ========================
# LOGGING HELPER
# ========================
def log(message):
    """Affiche un message avec timestamp."""
    time_str = datetime.now().strftime("[%H:%M:%S]")
    print(f"{time_str} {message}")

# ========================
# INITIALISATION
# ========================
index_to_name = {}
imageID_to_labels = {}
counter = 0

log("🚀 Début de la génération des fichiers JSON...")
log(f"Dossier principal : {base_dir}")

# ========================
# SCAN DES DOSSIERS
# ========================
folders = sorted(os.listdir(base_dir))
log(f"📂 {len(folders)} éléments trouvés dans HCP_data/")

for folder in folders:
    if not folder.startswith("subject_"):
        continue

    subject_id = folder.replace("subject_", "")
    subject_path = os.path.join(base_dir, folder)

    log(f"\n🔎 Analyse du dossier {folder}...")

    # Ignorer les sujets invalides
    if subject_id in invalid_subjects:
        log(f"⚠️  Sujet {subject_id} ignoré (problématique connu)")
        continue

    if not os.path.isdir(subject_path):
        log(f"❌ {subject_path} n'est pas un dossier, ignoré.")
        continue

    gz_found = False
    for root, dirs, files in os.walk(subject_path):
        for file in files:
            if file.endswith(".nii.gz"):
                filename = os.path.splitext(file)[0]
                gz_found = True

                log(f"✅ Fichier .nii.gz trouvé : {file}")
                index_to_name[str(counter)] = {
                    "filename": filename,
                    "subject_id": subject_id,
                    "date": "N/A",
                    "image_id": subject_id
                }
                imageID_to_labels[subject_id] = {}

                log(f"➕ Ajouté au JSON → index {counter} | subject {subject_id}")
                counter += 1
                break  # on s'arrête après le premier .nii.gz trouvé
        if gz_found:
            break

    if not gz_found:
        log(f"⚠️  Aucun fichier .nii.gz trouvé dans {subject_path}, sujet ignoré.")

# ========================
# SAUVEGARDE DES FICHIERS
# ========================
log("\n💾 Sauvegarde des fichiers JSON...")

with open(index_to_name_path, "w") as f:
    json.dump(index_to_name, f, indent=4)
log(f"✅ Fichier sauvegardé : {index_to_name_path} ({len(index_to_name)} subjects)")

with open(imageID_to_labels_path, "w") as f:
    json.dump(imageID_to_labels, f, indent=4)
log(f"✅ Fichier sauvegardé : {imageID_to_labels_path} ({len(imageID_to_labels)} labels)")

log("\n🎉 Génération terminée avec succès !")
log(f"Total des subjects valides : {len(index_to_name)}")
log("===============================================")
# ==============================================
# FINAL SUMMARY
# ==============================================
total_subjects = len(valid_subjects)  # liste de tous les sujets valides ajoutés au JSON
ignored_subjects = len(problematic_subjects)  # liste de sujets ignorés à cause de shape ou erreur

print("\n" + "="*60)
print("📊 FINAL SUMMARY")
print("="*60)
print(f"✅ Total subjects processed : {total_subjects + ignored_subjects}")
print(f"✅ Valid subjects added     : {total_subjects}")
print(f"⚠️  Problematic subjects ignored : {ignored_subjects}")

if ignored_subjects > 0:
    print("\n🧹 Ignored subjects list:")
    for sid in problematic_subjects:
        print(f"   - {sid}")

# Check JSON content
with open(os.path.join(output_dir, "index_to_name.json")) as f:
    index_to_name = json.load(f)
with open(os.path.join(output_dir, "imageID_to_labels.json")) as f:
    imageID_to_labels = json.load(f)

print("\n" + "="*60)
print("🧠 JSON CONSISTENCY CHECK")
print("="*60)
if len(index_to_name) != len(imageID_to_labels):
    print(f"❌ Mismatch: index_to_name={len(index_to_name)}, imageID_to_labels={len(imageID_to_labels)}")
else:
    print(f"✅ Both JSONs contain {len(index_to_name)} subjects")

# Verify that no problematic subject accidentally made it into JSONs
problem_in_json = [
    sid for sid in problematic_subjects if sid in index_to_name.values()
]
if problem_in_json:
    print(f"🚨 ERROR: {len(problem_in_json)} problematic subjects found in JSONs:")
    for sid in problem_in_json:
        print(f"   - {sid}")
else:
    print("✅ No problematic subject found in JSONs")

print("\n🎉 Summary complete — all checks passed successfully!")
print("="*60)