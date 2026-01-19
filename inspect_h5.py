
import h5py
import numpy as np
import sys

file_path = "/mnt/dataset2/benchmark_dataloader/hdf5/TUAB/sub_aaaaaaaq.h5"

try:
    with h5py.File(file_path, 'r') as f:
        print("File keys:", list(f.keys()))
        print("Attributes:", list(f.attrs.keys()))
        for key, val in f.attrs.items():
            print(f"{key}: {val}")
            
        if 'trial0' in f:
             print("Trial0 keys:", list(f['trial0'].keys()))
             print("Trial0 attributes:", list(f['trial0'].attrs.keys()))
             for key, val in f['trial0'].attrs.items():
                print(f"Trial0 attr {key}: {val}")
             
             if 'segment0' in f['trial0']:
                 print("Segment0 keys:", list(f['trial0']['segment0'].keys()))
                 print("Segment0 attributes:", list(f['trial0']['segment0'].attrs.keys()))
                 # Check data shape
                 if 'data' in f['trial0']['segment0']:
                     print("Data shape:", f['trial0']['segment0']['data'].shape)
                 elif 'eeg' in f['trial0']['segment0']:
                     print("EEG shape:", f['trial0']['segment0']['eeg'].shape)
except Exception as e:
    print(e)
