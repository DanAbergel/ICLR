import os, gc, time, psutil, torch, nibabel as nib, numpy as np, shutil
from tqdm import tqdm
from datetime import datetime
from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker
import json

# ==============================================================
# CONFIGURATION
# ==============================================================
base_dir = "/sci/labs/arieljaffe/dan.abergel1/HCP_data"
output_dir = os.path.join(base_dir, "data_full")
os.makedirs(output_dir, exist_ok=True)

index_to_name_path = os.path.join(output_dir, "index_to_name.json")

BATCH_SIZE = 100
standardize = False
EXPECTED_SPATIAL_SHAPE = (46, 55, 46)
T_MIN = 150
T_MAX = 350

final_4d_path = os.path.join(output_dir, "all_4d_downsampled.pt")
final_schaefer_path = os.path.join(output_dir, "time_regions_tensor_not_normalized_schaefer.pt")

# ==============================================================
# HELPERS
# ==============================================================
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def ram():
    m = psutil.virtual_memory()
    log(f"[RAM] used {m.used/1e9:.1f} GB / total {m.total/1e9:.1f} GB")

def extract_schaefer(fmri, atlas):
    masker = NiftiLabelsMasker(labels_img=atlas.maps, standardize=standardize, memory="nilearn_cache")
    return masker.fit_transform(fmri)

# ==============================================================
# PHASE 1 — CREATE BATCHES
# ==============================================================
def create_batches():
    log("=== PHASE 1: Creating batches and index mapping ===")
    log("Loading Schaefer atlas (200 ROIs)...")
    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=200)
    log("Schaefer atlas loaded.")

    subjects = sorted([d for d in os.listdir(base_dir) if d.startswith("subject_")])
    log(f"Discovered {len(subjects)} subject directories.")
    ram()

    log(f"Processing subjects in batches of {BATCH_SIZE}.")

    bad = []
    index_to_name = {}
    global_index = 0
    t0 = time.time()

    for i in range(0, len(subjects), BATCH_SIZE):
        batch_subjects = subjects[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        log(f"[Batch {batch_num}] Subjects {i} → {i + len(batch_subjects) - 1}")
        batch_4d, batch_schaefer = [], []

        for subj in tqdm(batch_subjects, desc=f"Loading batch {batch_num}", ncols=100):
            sid = subj.replace("subject_", "")
            subj_path = os.path.join(base_dir, subj)
            nii_path = os.path.join(subj_path, "MNINonLinear", "Results", "rfMRI_REST1_LR", "rfMRI_REST1_LR.nii.gz")

            if not os.path.exists(nii_path):
                # skip silently
                bad.append((sid, "missing"))
                continue

            try:
                nii = nib.load(nii_path)
                data = nii.get_fdata().astype(np.float32)

                if data.shape[:3] != EXPECTED_SPATIAL_SHAPE:
                    log(f"[SKIP] Subject {sid}: invalid spatial shape {data.shape[:3]}")
                    bad.append((sid, f"invalid spatial shape {data.shape}"))
                    continue

                T = data.shape[3]
                if T < T_MIN:
                    log(f"[SKIP] Subject {sid}: too short T={T}")
                    bad.append((sid, f"too short T {T}"))
                    continue

                # Slice temporal window [T_MIN : T_MAX] (or until end if shorter)
                t_start = T_MIN
                # t_end = min(T_MAX, T)
                data = data[:, :, :, t_start:T]

                # Create a sliced NIfTI to ensure Schaefer uses the same temporal window
                nii_sliced = nib.Nifti1Image(data, affine=nii.affine, header=nii.header)

                tensor = torch.from_numpy(data)
                ts = extract_schaefer(nii_sliced, atlas)

                batch_4d.append(tensor)
                batch_schaefer.append(torch.tensor(ts, dtype=torch.float32))

                index_to_name[str(global_index)] = {
                    "filename": nii_path,
                    "subject_id": sid,
                    "date": "N/A",
                    "image_id": f"{sid}_REST1_LR"
                }
                global_index += 1

                del nii, nii_sliced, data, ts, tensor
                gc.collect()

            except Exception as e:
                log(f"[ERROR] Subject {sid}: {e}")
                bad.append((sid, str(e)))
                continue

        if len(batch_4d) == 0:
            log(f"[Batch {batch_num}] No valid subjects — batch skipped.")
            continue

        b4_path = os.path.join(output_dir, f"batch_4d_{batch_num}.pt")
        bs_path = os.path.join(output_dir, f"batch_schaefer_{batch_num}.pt")

        torch.save(torch.stack(batch_4d), b4_path)
        torch.save(torch.stack(batch_schaefer), bs_path)
        log(f"[Batch {batch_num}] Saved {len(batch_4d)} valid subjects.")
        ram()

        del batch_4d, batch_schaefer
        gc.collect()

    with open(index_to_name_path, "w") as f:
        json.dump(index_to_name, f, indent=2)
    log(f"Index mapping written ({len(index_to_name)} entries).")

    log(f"PHASE 1 completed in {(time.time()-t0)/60:.1f} minutes.")
    log(f"Valid subjects kept: {len(index_to_name)}")
    log(f"Subjects excluded: {len(bad)}")

# ==============================================================
# PHASE 2 — MERGE STREAM
# ==============================================================
def merge_batches(output_path, pattern, label):
    files = sorted([f for f in os.listdir(output_dir) if f.startswith(pattern) and f.endswith(".pt")])
    if not files:
        log(f"❌ No batch files found for pattern '{pattern}'")
        return

    log(f"=== Merging {label} batches ===")
    log(f"Found {len(files)} batch files.")
    first = torch.load(os.path.join(output_dir, files[0]), map_location="cpu")
    total = sum(torch.load(os.path.join(output_dir, f), map_location="cpu").shape[0] for f in files)
    shape = [total] + list(first.shape[1:])
    log(f"Final tensor shape will be {tuple(shape)}")

    final = torch.empty(shape, dtype=first.dtype)
    offset = 0

    for f in tqdm(files, desc=f"Merging {label}", ncols=100):
        batch_file = os.path.join(output_dir, f)
        batch = torch.load(batch_file, map_location="cpu")
        final[offset:offset + batch.shape[0]] = batch
        offset += batch.shape[0]
        del batch
        gc.collect()
        os.remove(batch_file)

    ram()

    torch.save(final, output_path)
    log(f"{label} tensor saved: {output_path}")
    del final
    gc.collect()
    ram()

def merge_all():
    merge_batches(final_4d_path, "batch_4d_", "4D")
    merge_batches(final_schaefer_path, "batch_schaefer_", "Schaefer")
    log("PHASE 2 completed — final tensors ready.")

# ==============================================================
# EXECUTION CONTROL
# ==============================================================
if __name__ == "__main__":
    # create_batches()   # ➜ Phase 1 : crée les batchs (et supprime les sujets invalides)
    merge_all()        # ➜ Phase 2 : fusionne les batchs