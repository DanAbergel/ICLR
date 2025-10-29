"""
created by Dan Abergel
date: 25/09/2025
"""

#!/usr/bin/env python3
print("cececec")
from pathlib import Path
import boto3, botocore
from datetime import datetime
import os
import sys
import nibabel as nib
import numpy as np

# ====== CONFIG ======
BASE = Path("/sci/labs/arieljaffe/dan.abergel1/HCP_data")
BUCKET = "hcp-openaccess"
ROOT = "HCP_1200"
REQUEST_PAYER = "requester"
DRY_RUN = False
VERBOSE = True
# ====================

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def s3():
    cfg = botocore.config.Config(
        region_name="us-east-1",
        retries={"max_attempts": 10, "mode": "standard"},
        s3={"addressing_style": "virtual"},
    )
    return boto3.client("s3", config=cfg)

def folder_size_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for p, _, files in os.walk(path):
        for f in files:
            fp = Path(p) / f
            try:
                total += fp.stat().st_size
            except FileNotFoundError:
                pass
    return total

def download_one(client, s3_key: str, local_path: Path):
    local_path.parent.mkdir(parents=True, exist_ok=True)

    # ✅ Check if file already exists and is not empty
    if local_path.exists():
        try:
            if local_path.stat().st_size > 0:
                return "EXISTS"
        except FileNotFoundError:
            pass

    if DRY_RUN:
        if VERBOSE:
            log(f"DRY s3://{BUCKET}/{s3_key}  ->  {local_path}")
        return "DRY"

    log(f"Starting download of s3://{BUCKET}/{s3_key} to {local_path}")
    tmp = local_path.with_suffix(local_path.suffix + ".part")
    client.download_file(
        BUCKET, s3_key, str(tmp), ExtraArgs={"RequestPayer": REQUEST_PAYER}
    )
    tmp.replace(local_path)
    log(f"Finished download of s3://{BUCKET}/{s3_key} to {local_path}")
    return "DOWNLOADED"

def main():
    BASE.mkdir(parents=True, exist_ok=True)
    client = s3()

    # 📊 Initial folder size
    initial_size = folder_size_bytes(BASE)
    initial_size_gb = initial_size / (1024 ** 3)
    log(f"Initial size: {initial_size_gb:.2f} GB")

    # 📡 List all subjects from S3
    log("📡 Listing all subjects from S3...")
    resp = client.list_objects_v2(
        Bucket=BUCKET,
        Prefix=f"{ROOT}/",
        Delimiter="/",
        RequestPayer=REQUEST_PAYER
    )

    subjects = []
    for prefix in resp.get("CommonPrefixes", []):
        sid = prefix["Prefix"].split("/")[-2]
        if sid.isdigit():
            subjects.append(sid)

    total_subjects = len(subjects)
    log(f"✅ {total_subjects} subjects found.")

    # --- Check how many are already downloaded ---
    downloaded_subjects = 0
    total_size_bytes = 0
    for sid in sorted(subjects):
        dst = BASE / f"subject_{sid}/MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR.nii.gz"
        if dst.exists():
            size = dst.stat().st_size
            if size > 0:
                downloaded_subjects += 1
                total_size_bytes += size

    total_size_gb = total_size_bytes / (1024 ** 3)
    remaining_subjects = total_subjects - downloaded_subjects
    avg_size_per_subject_gb = (total_size_gb / downloaded_subjects) if downloaded_subjects > 0 else 0
    estimated_remaining_gb = avg_size_per_subject_gb * remaining_subjects

    log(f"📊 Already downloaded: {downloaded_subjects}/{total_subjects} subjects, {total_size_gb:.2f} GB")
    log(f"📦 Remaining: {remaining_subjects} subjects (~{estimated_remaining_gb:.2f} GB)")

    # 📥 Download remaining subjects
    current_total_size_bytes = total_size_bytes
    current_downloaded = downloaded_subjects
    counts = {"downloaded": 0, "missing": 0, "errors": 0}

    for idx, sid in enumerate(sorted(subjects), 1):
        dst = BASE / f"subject_{sid}/MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR.nii.gz"

        # Skip already existing and downsampled
        if dst.exists() and dst.stat().st_size > 0:
            try:
                img = nib.load(str(dst))
                data = img.get_fdata()
                spatial_shape = data.shape[:3]
                if all(s <= 110 for s in spatial_shape):
                    continue
            except Exception:
                pass
        # Get estimated size if possible
        try:
            head = client.head_object(Bucket=BUCKET, Key=f"{ROOT}/{sid}/MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR.nii.gz", RequestPayer=REQUEST_PAYER)
            estimated_size_gb = head["ContentLength"] / (1024 ** 3)
        except Exception:
            estimated_size_gb = 0.0

        log(f"➡️ [{current_downloaded + 1}/{total_subjects}] Subject {sid} – starting download (~{estimated_size_gb:.2f} GB)")

        try:
            out = download_one(client, f"{ROOT}/{sid}/MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR.nii.gz", dst)
            if out == "DOWNLOADED":
                counts["downloaded"] += 1
                current_downloaded += 1
                current_total_size_bytes += dst.stat().st_size

                # Downsampling
                original_size_gb = dst.stat().st_size / (1024 ** 3)
                img = nib.load(str(dst))
                data = img.get_fdata(dtype=np.float32)
                affine = img.affine.copy()
                data_downsampled = data[::2, ::2, ::2, :]
                affine[:3, :3] *= 2
                downsampled_img = nib.Nifti1Image(data_downsampled, affine, img.header)
                nib.save(downsampled_img, str(dst))
                downsampled_size_gb = dst.stat().st_size / (1024 ** 3)
                saved_gb = original_size_gb - downsampled_size_gb
                log(f"📉 Downsampling complete for subject {sid} – reduced from {original_size_gb:.2f} GB to {downsampled_size_gb:.2f} GB (saved {saved_gb:.2f} GB)")

                current_total_size_gb = current_total_size_bytes / (1024 ** 3)
                remaining_subjects = total_subjects - current_downloaded
                avg_size_per_subject_gb = (current_total_size_gb / current_downloaded) if current_downloaded > 0 else 0
                estimated_remaining_gb = avg_size_per_subject_gb * remaining_subjects
                log(f"✅ Subject {sid} downloaded – total: {current_total_size_gb:.2f} GB | remaining: {remaining_subjects} (~{estimated_remaining_gb:.2f} GB)")
        except botocore.exceptions.ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in ("404", "NoSuchKey"):
                counts["missing"] += 1
            else:
                counts["errors"] += 1

    # 📊 Final summary
    final_size = folder_size_bytes(BASE)
    final_size_gb = final_size / (1024 ** 3)
    added_gb = (final_size - initial_size) / (1024 ** 3)

    log("==== Summary ====")
    log(f"📈 Final folder size: {final_size_gb:.2f} GB")
    log(f"📥 Added memory: {added_gb:.2f} GB")
    log(f"✅ New files downloaded: {counts['downloaded']} | Missing: {counts['missing']} | Errors: {counts['errors']}")

if __name__ == "__main__":
    try:
        main()
    except botocore.exceptions.NoCredentialsError:
        print("❌ AWS credentials not found. Run `aws configure`.", file=sys.stderr)
        sys.exit(2)