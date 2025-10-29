import os, gc, time, psutil, torch, nibabel as nib, numpy as np, shutil
from tqdm import tqdm
from datetime import datetime
from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker

# ==============================================================
# CONFIGURATION
# ==============================================================
base_dir = "/sci/labs/arieljaffe/dan.abergel1/HCP_data"
output_dir = os.path.join(base_dir, "data")
os.makedirs(output_dir, exist_ok=True)

BATCH_SIZE = 100
standardize = False
EXPECTED_SHAPE = (46, 55, 46, 1200)

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
    log("📚 Loading Schaefer atlas (200 ROIs)...")
    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=200)
    log("✅ Atlas loaded successfully.")

    subjects = sorted([d for d in os.listdir(base_dir) if d.startswith("subject_")])
    log(f"✅ Found {len(subjects)} subjects in {base_dir}")
    ram()

    bad = []
    t0 = time.time()

    for i in range(0, len(subjects), BATCH_SIZE):
        batch_subjects = subjects[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        log(f"\n🧠 Batch {batch_num} — subjects {i}-{i + len(batch_subjects) - 1}")
        batch_4d, batch_schaefer = [], []

        for subj in tqdm(batch_subjects, desc=f"Loading batch {batch_num}", ncols=100):
            sid = subj.replace("subject_", "")
            subj_path = os.path.join(base_dir, subj)
            nii_path = os.path.join(subj_path, "MNINonLinear", "Results", "rfMRI_REST1_LR", "rfMRI_REST1_LR.nii.gz")

            if not os.path.exists(nii_path):
                log(f"⚠️ Subject {sid}: missing file, skipping.")
                bad.append((sid, "missing"))
                continue

            try:
                nii = nib.load(nii_path)
                data = nii.get_fdata().astype(np.float32)

                if data.shape != EXPECTED_SHAPE:
                    log(f"⚠️ Subject {sid} has invalid shape {data.shape} — deleted from HCP_data.")
                    shutil.rmtree(subj_path, ignore_errors=True)
                    bad.append((sid, f"invalid shape {data.shape}"))
                    continue

                tensor = torch.from_numpy(data)
                ts = extract_schaefer(nii, atlas)
                batch_4d.append(tensor)
                batch_schaefer.append(torch.tensor(ts, dtype=torch.float32))

                del nii, data, ts, tensor
                gc.collect()

            except Exception as e:
                log(f"❌ Error loading {sid}: {e}")
                bad.append((sid, str(e)))
                continue

        ram()

        if len(batch_4d) == 0:
            log(f"⚠️ No valid subjects in batch {batch_num}, skipping save.")
            continue

        # ✅ Sauvegarde les batchs directement dans data/
        b4_path = os.path.join(output_dir, f"batch_4d_{batch_num}.pt")
        bs_path = os.path.join(output_dir, f"batch_schaefer_{batch_num}.pt")

        torch.save(torch.stack(batch_4d), b4_path)
        torch.save(torch.stack(batch_schaefer), bs_path)
        log(f"✅ Saved batch files: {b4_path} and {bs_path}")

        del batch_4d, batch_schaefer
        gc.collect()
        ram()

    log(f"✅ Phase 1 finished in {(time.time()-t0)/60:.1f} min. Total subjects processed: {len(subjects)}")
    if bad:
        log(f"⚠️ Excluded or deleted {len(bad)} subjects.")
        for s, reason in bad[:5]:
            log(f"   - {s}: {reason}")

# ==============================================================
# PHASE 2 — MERGE STREAMÉ
# ==============================================================
def merge_batches(output_path, pattern, label):
    files = sorted([f for f in os.listdir(output_dir) if f.startswith(pattern) and f.endswith(".pt")])
    if not files:
        log(f"❌ No batch files found for pattern '{pattern}'")
        return

    log(f"\n🔗 Merging {len(files)} {label} batches → {output_path}")
    first = torch.load(os.path.join(output_dir, files[0]), map_location="cpu")
    total = sum(torch.load(os.path.join(output_dir, f), map_location="cpu").shape[0] for f in files)
    shape = [total] + list(first.shape[1:])
    log(f"📐 Target shape : {tuple(shape)}")

    final = torch.empty(shape, dtype=first.dtype)
    offset = 0

    for f in tqdm(files, desc=f"Merging {label}", ncols=100):
        batch = torch.load(os.path.join(output_dir, f), map_location="cpu")
        final[offset:offset + batch.shape[0]] = batch
        offset += batch.shape[0]
        del batch
        gc.collect()
        ram()

    torch.save(final, output_path)
    log(f"✅ Saved {output_path} ({tuple(final.shape)})")
    del final
    gc.collect()
    ram()

def merge_all():
    merge_batches(final_4d_path, "batch_4d_", "4D")
    merge_batches(final_schaefer_path, "batch_schaefer_", "Schaefer")
    log("🎉 Phase 2 complete — all final files ready.")

# ==============================================================
# EXECUTION CONTROL
# ==============================================================
if __name__ == "__main__":
    create_batches()   # ➜ Phase 1 : crée les batchs (et supprime les sujets invalides)
    merge_all()        # ➜ Phase 2 : fusionne les batchs