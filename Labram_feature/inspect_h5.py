
import h5py
import os

file_path = '/mnt/dataset2/benchmark_dataloader/hdf5/TUAB/sub_aaaaaaaq.h5'

if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
    # Try finding it in parent directory
    file_path = '/mnt/dataset2/benchmark_dataloader/hdf5/sub_aaaaaaaq.h5'
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        exit(1)

print(f"Inspecting: {file_path}")

try:
    with h5py.File(file_path, 'r') as f:
        print(f"Global Attrs: {dict(f.attrs)}")
        print("Keys:", list(f.keys()))
        eeg_printed = False
        # Inspect first key
        first_key = list(f.keys())[0]
        print(f"Inspecting first key: {first_key}")
        print(f"Type: {type(f[first_key])}")
        if isinstance(f[first_key], h5py.Group):
            print(f"Group keys: {list(f[first_key].keys())}")
            # Inspect first subkey
            first_subkey = list(f[first_key].keys())[0]
            print(f"Inspecting first subkey: {first_subkey}")
            print(f"Type: {type(f[first_key][first_subkey])}")
            if isinstance(f[first_key][first_subkey], h5py.Group):
                 print(f"Subgroup keys: {list(f[first_key][first_subkey].keys())}")
                 if 'eeg' in f[first_key][first_subkey]:
                     print(f"EEG shape: {f[first_key][first_subkey]['eeg'].shape}")
                     eeg_data = f[first_key][first_subkey]['eeg']
                     print(f"EEG first 5 time points: {eeg_data[:, :5]}")
                     eeg_printed = True
            elif isinstance(f[first_key][first_subkey], h5py.Dataset):
                 print(f"Dataset shape: {f[first_key][first_subkey].shape}")

        # Fallback: recursively search for any dataset named 'eeg'
        if not eeg_printed:
            def _print_eeg_dataset(name, obj):
                nonlocal eeg_printed
                if isinstance(obj, h5py.Dataset) and name.split('/')[-1] == 'eeg':
                    print(f"EEG path: {name}")
                    print(f"EEG shape: {obj.shape}")
                    print(f"EEG first 5 time points: {obj[:, :5]}")
                    eeg_printed = True

            f.visititems(_print_eeg_dataset)

except Exception as e:
    print(f"Error: {e}")
