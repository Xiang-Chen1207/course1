
import h5py
import numpy as np
import os
import sys

root = '/mnt/dataset2/benchmark_dataloader/hdf5/TUAB/'
if not os.path.exists(root):
    print(f"Root not found: {root}")
    exit(1)

files = [os.path.join(root, f) for f in os.listdir(root) if f.endswith('.h5')]
print(f"Found {len(files)} files.")

labels = set()

for i, file_path in enumerate(files):
    try:
        with h5py.File(file_path, 'r') as f:
            for trial_key in f.keys():
                if not trial_key.startswith('trial'): continue
                trial = f[trial_key]
                for seg_key in trial.keys():
                    if not seg_key.startswith('segment'): continue
                    seg = trial[seg_key]
                    if 'eeg' in seg:
                        lbl = seg['eeg'].attrs.get('label')
                        if isinstance(lbl, np.ndarray):
                            lbl = lbl.item() if lbl.size == 1 else lbl[0]
                        labels.add(lbl)
    except Exception as e:
        print(f"Error: {e}")
    
    if len(labels) >= 2:
        print(f"Found multiple labels: {labels}")
        break
    if i % 50 == 0:
        print(f"Checked {i+1} files, found labels: {labels}")

print(f"Final labels: {labels}")
