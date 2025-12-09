import os
import json
import pandas as pd
import numpy as np

# ----------------------------------------------------------
# PATHS
# ----------------------------------------------------------
BASE_PATH = "/sci/labs/arieljaffe/dan.abergel1/HCP_data"
CSV_PATH = f"{BASE_PATH}/metadata/HCP_YA_subjects.csv"
OUTPUT_INDEX = f"{BASE_PATH}/model_input/index_to_name.json"
OUTPUT_LABELS = f"{BASE_PATH}/model_input/imageID_to_labels.json"

# Create output directory if missing
os.makedirs(os.path.dirname(OUTPUT_INDEX), exist_ok=True)

print("🚀 Starting JSON metadata creation...")
print(f"📂 BASE_PATH  = {BASE_PATH}")
print(f"📄 CSV_PATH   = {CSV_PATH}")
print("----------------------------------------------------------")

# ----------------------------------------------------------
# LOAD CSV
# ----------------------------------------------------------
df = pd.read_csv(CSV_PATH)
print(f"✔️ Loaded CSV with {len(df)} rows")

# ----------------------------------------------------------
# Helper: Encode Sex
# ----------------------------------------------------------
def encode_sex(x):
    if isinstance(x, str):
        x = x.strip().upper()
        if x == "M":
            return 1
        if x == "F":
            return 0
    return np.nan


# ----------------------------------------------------------
# BUILD index_to_name.json
# AND imageID_to_labels.json
# ----------------------------------------------------------
index_to_name = {}
imageID_to_labels = {}
counter = 0

print("\n🔎 Scanning BASE_PATH...\n")

for item in os.listdir(BASE_PATH):
    print("----------------------------------------------------------")
    print(f"📁 Checking folder: {item}")

    subject_dir = os.path.join(BASE_PATH, item)

    if not item.startswith("subject_"):
        print("⛔  Skipped — not a subject folder")
        continue

    subject_id = item.replace("subject_", "")
    print(f"✔️ Valid subject detected: {subject_id}")

    # Build expected NIFTI path
    nii_path = os.path.join(
        subject_dir,
        "MNINonLinear",
        "Results",
        "rfMRI_REST1_LR",
        "rfMRI_REST1_LR.nii.gz"
    )

    if not os.path.isfile(nii_path):
        print(f"❌ Missing NIfTI file → {nii_path}")
        continue
    else:
        print(f"📄 Found NIfTI file → {nii_path}")

    # Unique image ID
    image_id = f"{subject_id}_REST1_LR"
    print(f"🆔 Image ID = {image_id}")

    # ------------------------------------------------------
    # index_to_name entry
    # ------------------------------------------------------
    index_to_name[str(counter)] = {
        "filename": nii_path.replace(BASE_PATH + "/", ""),
        "subject_id": subject_id,
        "date": "N/A",
        "image_id": image_id
    }
    print(f"📝 Added index_to_name[{counter}]")

    # ------------------------------------------------------
    # labels entry → from CSV
    # ------------------------------------------------------
    row = df[df["Subject"].astype(str) == subject_id]

    if row.empty:
        print(f"⚠ CSV row NOT FOUND for subject {subject_id}")
        labels = {
            "Sex": np.nan,
            "Age": np.nan,
            "Sex_Binary": np.nan
        }
    else:
        row_dict = row.iloc[0].to_dict()
        print(f"📄 CSV row (raw) = {row_dict}")

        # Convert numpy types to Python types
        row_dict = {
            k: (None if pd.isna(v) else v)
            for k, v in row_dict.items()
        }

        # Add binary sex
        row_dict["Sex_Binary"] = encode_sex(
            row_dict.get("Gender", None) or row_dict.get("Sex", None)
        )

        print(f"🔧 Cleaned CSV row + added Sex_Binary: {row_dict}")
        labels = row_dict

    imageID_to_labels[image_id] = labels
    print(f"🏷 Saved labels for imageID = {image_id}")

    counter += 1
    print(f"➕ Counter → {counter}")

# ----------------------------------------------------------
# SAVE JSON FILES
# ----------------------------------------------------------
print("\n💾 Saving output JSON files...")

with open(OUTPUT_INDEX, "w") as f:
    json.dump(index_to_name, f, indent=2)
print(f"✔️ Saved: {OUTPUT_INDEX}")

with open(OUTPUT_LABELS, "w") as f:
    json.dump(imageID_to_labels, f, indent=2)
print(f"✔️ Saved: {OUTPUT_LABELS}")

print("\n🎉 DONE!")
print(f"📊 Total subjects processed = {counter}")
print("----------------------------------------------------------\n")