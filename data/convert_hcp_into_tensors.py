import nibabel as nib
import torch
import glob
import os
import gc

BATCH_SIZE = 70
HCP_DIR = "/sci/labs/arieljaffe/dan.abergel1/HCP_data"
OUT_DIR = "/sci/labs/arieljaffe/dan.abergel1/HCP_tensors/"
def convert_hcp_into_tensors(dir_path):
    # Ensure the provided directory path exists
    assert os.path.exists(dir_path) , f"The path {dir_path} does not exist."

    print(f"Starting conversion process in directory: {dir_path}")

    # Find all .nii.gz files in the nested directory structure matching the pattern
    nii_files_pathes = glob.glob(f"{dir_path}/subject*/*/*/*/*.nii.gz")
    print(f"Number of .nii.gz files found: {len(nii_files_pathes)}")

    tensor_batch = []  # List to hold tensors for batching
    batch_idx = 0      # Index to keep track of batch files saved

    # Iterate over all found files
    for i, file_path in enumerate(nii_files_pathes):
        print(f"Processing file {i+1}/{len(nii_files_pathes)}: {file_path}")

        # Load the NIfTI image
        nii_image = nib.load(file_path)

        # Convert the image data to a PyTorch tensor of type float32
        image_tensor = torch.tensor(nii_image.get_fdata(), dtype=torch.float32)

        # Free up memory by deleting the loaded image object
        nii_image.uncache()
        del nii_image

        # Append the tensor to the current batch list
        tensor_batch.append(image_tensor)

        # Delete the tensor variable to free memory (optional cleanup)
        del image_tensor

        # Call to garbage collector to cleanup the disk
        gc.collect()

        # Check if the batch has reached the specified size or if this is the last file
        if len(tensor_batch) == BATCH_SIZE or i == len(nii_files_pathes) - 1:
            # Stack the list of tensors into a single tensor batch
            batch_tensor = torch.stack(tensor_batch)

            # Define the filename for saving the batch tensor
            batch_filename = f'batched_tensor_{batch_idx}.pt'

            # Save the tensor batch to disk
            torch.save(batch_tensor, batch_filename)

            print(f"Saved batch {batch_idx} with {len(tensor_batch)} tensors to {batch_filename}")

            # Increment batch index for next batch
            batch_idx += 1

            # Clear the batch list for the next batch of tensors
            tensor_batch = []

    print("Conversion process completed.")

if __name__ == "__main__":
    print("Starting conversion process.")
    os.makedirs(OUT_DIR,exist_ok=True)
    assert os.path.exists(HCP_DIR) , f"The path {HCP_DIR} does not exist."
    convert_hcp_into_tensors(HCP_DIR)
    print("Conversion process completed.")
