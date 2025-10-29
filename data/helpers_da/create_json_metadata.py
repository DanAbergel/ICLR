import os
import json

# === Paths setup ===
base_dir = "/sci/labs/arieljaffe/dan.abergel1/HCP_data"
output_dir = os.path.join(base_dir, "model_input")
os.makedirs(output_dir, exist_ok=True)

index_to_name_path = os.path.join(output_dir, "index_to_name.json")
imageID_to_labels_path = os.path.join(output_dir, "imageID_to_labels.json")

# === Initialize structures ===
index_to_name = {}
imageID_to_labels = {}
index = 0

# === Loop over subjects ===
for subject_dir in sorted(os.listdir(base_dir)):
    subject_path = os.path.join(base_dir, subject_dir)
    if not os.path.isdir(subject_path) or not subject_dir.startswith("subject_"):
        continue

    subject_id = subject_dir.replace("subject_", "")
    nii_path = os.path.join(
        subject_path, "MNINonLinear", "Results", "rfMRI_REST1_LR", "rfMRI_REST1_LR.nii.gz"
    )

    if os.path.exists(nii_path):
        # Clean up filename (remove .nii or .nii.gz)
        filename = os.path.basename(nii_path)
        if filename.endswith(".nii.gz"):
            filename = filename[:-7]
        elif filename.endswith(".nii"):
            filename = filename[:-4]

        entry = {
            "filename": filename,
            "subject_id": subject_id,
            "date": "N/A",
            "image_id": subject_id
        }
        index_to_name[str(index)] = entry

        # Add empty label dictionary for this image_id
        imageID_to_labels[subject_id] = {}

        index += 1
    else:
        print(f"⚠️ Missing file for {subject_id}")

# === Save index_to_name.json ===
with open(index_to_name_path, "w") as f:
    json.dump(index_to_name, f, indent=4)

# === Save imageID_to_labels.json ===
with open(imageID_to_labels_path, "w") as f:
    json.dump(imageID_to_labels, f, indent=4)

# === Logs ===
print(f"✅ index_to_name.json created at: {index_to_name_path}")
print(f"✅ imageID_to_labels.json created at: {imageID_to_labels_path}")
print(f"Total subjects indexed: {index}")