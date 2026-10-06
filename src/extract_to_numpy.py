import argparse
import glob
import os
import tensorflow as tf
import numpy as np

def process_dataset(dataset_path):
    # Normalize path and remove trailing slashes to get a clean directory name
    dataset_path = os.path.normpath(dataset_path)
    
    if not os.path.isdir(dataset_path):
        print(f"Skipping: '{dataset_path}' is not a valid directory.")
        return

    print(f"\n" + "="*80)
    print(f"Processing: {dataset_path}")
    print(f"="*80)

    try:
        # 1. Load the compressed TF dataset
        tf_dataset = tf.data.Dataset.load(dataset_path, compression="GZIP")
        tf_dataset = tf_dataset.unbatch()
    except Exception as e:
        print(f"Error loading dataset at {dataset_path}: {e}")
        return

    features_list = []
    labels_list = []
    is_dict = False

    print("Streaming data points into memory...")
    for item in tf_dataset.as_numpy_iterator():
        # Scenario A: Tuple (features, label)
        if isinstance(item, tuple) and len(item) == 2:
            features_list.append(item[0])
            labels_list.append(item[1])
            
        # Scenario B: Dictionary (Text / Key-value structures)
        elif isinstance(item, dict):
            is_dict = True
            if not features_list:
                features_list = {key: [] for key in item.keys()}
            for key, val in item.items():
                features_list[key].append(val)
                
        # Scenario C: Pure features array
        else:
            features_list.append(item)

    # 2. Determine output filenames based on input directory path
    # If folder is '.../only_treated_train_dataset_logcounts.gz', 
    # base becomes '.../only_treated_train_dataset_logcounts'
    base_output_path = dataset_path
    if base_output_path.endswith(".gz"):
        base_output_path = base_output_path[:-3]

    print("Writing files to disk...")
    if is_dict:
        out_file = f"{base_output_path}_dict.npz"
        if not os.path.exists(out_file):
            np.savez(out_file, **{k: np.array(v) for k, v in features_list.items()})
            print(f"-> Saved: {out_file}")
    else:
        feat_file = f"{base_output_path}_features.npy"
        if not os.path.exists(feat_file):
            np.save(feat_file, np.array(features_list))
            print(f"-> Saved: {feat_file}")
        
        if labels_list:
            label_file = f"{base_output_path}_labels.npy"
            if not os.path.exists(label_file):
                np.save(label_file, np.array(labels_list))
                print(f"-> Saved: {label_file}")

def main():
    parser = argparse.ArgumentParser(description="Convert GZIP tf.data.Dataset folders to NumPy files.")
    parser.add_argument(
        "paths", 
        nargs="+", 
        help="One or more dataset paths or wildcard patterns (e.g., ../data/processed/only_treated_train_dataset_*)"
    )
    args = parser.parse_args()

    # Resolve all wildcards/globs passed via command line
    resolved_paths = []
    for pattern in args.paths:
        resolved_paths.extend(glob.glob(pattern))

    if not resolved_paths:
        print("No matching directories found for the provided arguments.")
        return

    print(f"Found {len(resolved_paths)} path(s) to process.")
    for path in resolved_paths:
        process_dataset(path)
    print("\nAll extractions completed successfully!")

if __name__ == "__main__":
    main()

