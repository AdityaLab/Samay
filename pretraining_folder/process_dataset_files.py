import json
import numpy as np
import os

#May need to edit paths
input_folder = "Time-300B-energy"
output_folder = "csv_output"

def process_dataset_files():
    subfolders = [d for d in os.listdir(input_folder) 
                    if os.path.isdir(os.path.join(input_folder, d)) and not d.startswith('.')]
    for subfolder in subfolders:
        meta_file = os.path.join(input_folder, subfolder, "meta.json")
        bin_file = os.path.join(input_folder, subfolder, "data-1-of-1 (1).bin")
        out_dir = os.path.join(output_folder, subfolder)

        os.makedirs(out_dir, exist_ok=True)

        with open(meta_file, "r") as f:
            meta = json.load(f)

        dtype = np.dtype(meta["dtype"])
        itemsize = dtype.itemsize

        with open(bin_file, "rb") as f:
            raw = f.read()

        bin_size = len(raw)


        for i, scale in enumerate(meta["scales"]):
            offset_bytes = scale["offset"] * itemsize
            length_bytes = scale["length"] * itemsize

            buf = raw[offset_bytes: offset_bytes + length_bytes]
            arr = np.frombuffer(buf, dtype=dtype)

            out_path = os.path.join(out_dir, f"sequence_{i+1}.csv")
            np.savetxt(out_path, arr, delimiter=",")


process_dataset_files()