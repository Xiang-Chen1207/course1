
import h5py
import bisect
import numpy as np
import torch
from pathlib import Path
from typing import List
from torch.utils.data import Dataset
import ast
from collections import deque

standard_1020 = [ 
         'FP1', 'FPZ', 'FP2', 
         'AF9', 'AF7', 'AF5', 'AF3', 'AF1', 'AFZ', 'AF2', 'AF4', 'AF6', 'AF8', 'AF10', 
         'T1', 'F9', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'F10', 'T2', 
         'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 
         'A1', 'T9', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'T10', 'A2', 
         'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 
         'P9', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'P10', 
         'PO9', 'PO7', 'PO5', 'PO3', 'PO1', 'POZ', 'PO2', 'PO4', 'PO6', 'PO8', 'PO10', 
         'O1', 'OZ', 'O2', 
         'I1', 'IZ', 'I2', 
     ]

list_path = List[Path]

class SingleShockDataset(Dataset):
    """Read single hdf5 file regardless of label, subject, and paradigm."""
    def __init__(self, file_path: Path, window_size: int=200, stride_size: int=1, start_percentage: float=0, end_percentage: float=1):
        '''
        Extract datasets from file_path.

        param Path file_path: the path of target data
        param int window_size: the length of a single sample
        param int stride_size: the interval between two adjacent samples
        param float start_percentage: Index of percentage of the first sample of the dataset in the data file (inclusive)
        param float end_percentage: Index of percentage of end of dataset sample in data file (not included)
        '''
        self.__file_path = file_path
        self.__window_size = window_size
        self.__stride_size = stride_size
        self.__start_percentage = start_percentage
        self.__end_percentage = end_percentage

        self.__file = None
        self.__length = None
        self.__feature_size = None

        self.__subjects = []
        self.__global_idxes = []
        self.__local_idxes = []
        
        self.__init_dataset()

    def __init_dataset(self) -> None:
        self.__file = h5py.File(str(self.__file_path), 'r')
        self.__subjects = [i for i in self.__file]

        global_idx = 0
        for subject in self.__subjects:
            self.__global_idxes.append(global_idx) # the start index of the subject's sample in the dataset
            subject_len = self.__file[subject]['eeg'].shape[1]
            # total number of samples
            total_sample_num = (subject_len-self.__window_size) // self.__stride_size + 1
            # cut out part of samples
            start_idx = int(total_sample_num * self.__start_percentage) * self.__stride_size 
            end_idx = int(total_sample_num * self.__end_percentage - 1) * self.__stride_size

            self.__local_idxes.append(start_idx)
            global_idx += (end_idx - start_idx) // self.__stride_size + 1
        self.__length = global_idx

        self.__feature_size = [i for i in self.__file[self.__subjects[0]]['eeg'].shape]
        self.__feature_size[1] = self.__window_size

    @property
    def feature_size(self):
        return self.__feature_size

    def __len__(self):
        return self.__length

    def __getitem__(self, idx: int):
        subject_idx = bisect.bisect(self.__global_idxes, idx) - 1
        item_start_idx = (idx - self.__global_idxes[subject_idx]) * self.__stride_size + self.__local_idxes[subject_idx]
        return self.__file[self.__subjects[subject_idx]]['eeg'][:, item_start_idx:item_start_idx+self.__window_size]
    
    def free(self) -> None: 
        if self.__file:
            self.__file.close()
            self.__file = None
    
    def get_ch_names(self):
        return self.__file[self.__subjects[0]]['eeg'].attrs['chOrder']


class ShockDataset(Dataset):
    """integrate multiple hdf5 files"""
    def __init__(self, file_paths: list_path, window_size: int=200, stride_size: int=1, start_percentage: float=0, end_percentage: float=1):
        '''
        Arguments will be passed to SingleShockDataset. Refer to SingleShockDataset.
        '''
        self.__file_paths = file_paths
        self.__window_size = window_size
        self.__stride_size = stride_size
        self.__start_percentage = start_percentage
        self.__end_percentage = end_percentage

        self.__datasets = []
        self.__length = None
        self.__feature_size = None

        self.__dataset_idxes = []
        
        self.__init_dataset()

    def __init_dataset(self) -> None:
        self.__datasets = [SingleShockDataset(file_path, self.__window_size, self.__stride_size, self.__start_percentage, self.__end_percentage) for file_path in self.__file_paths]
        
        # calculate the number of samples for each subdataset to form the integral indexes
        dataset_idx = 0
        for dataset in self.__datasets:
            self.__dataset_idxes.append(dataset_idx)
            dataset_idx += len(dataset)
        self.__length = dataset_idx
        
        self.__feature_size = self.__datasets[0].feature_size

    @property
    def feature_size(self):
        return self.__feature_size

    def __len__(self):
        return self.__length

    def __getitem__(self, idx: int):
        dataset_idx = bisect.bisect(self.__dataset_idxes, idx) - 1
        return self.__datasets[dataset_idx][idx - self.__dataset_idxes[dataset_idx]]
    
    def free(self) -> None:
        for dataset in self.__datasets:
            dataset.free()
    
    def get_ch_names(self):
        return self.__datasets[0].get_ch_names()


class FeaturePredictionDataset(Dataset):
    def __init__(self, dataframe, hdf5_root, window_size=1600, cache_size=4):
        self.dataframe = dataframe.reset_index(drop=True)
        self.hdf5_root = Path(hdf5_root)
        self.window_size = window_size
        self.cache_size = max(int(cache_size), 0)
        
        # Identify feature columns
        metadata_cols = [
            'trial_ids', 'segment_ids', 'session_id', 'primary_label', 'labels',
            'start_time', 'end_time', 'total_time_length', 'merge_count',
            'source_segments', 'source_file', 'sub_id', 'subject_id', 'unique_sub_id', 'dataset_source'
        ]
        self.feature_cols = [c for c in self.dataframe.columns if c not in metadata_cols]
        
        # Group by HDF5 file to minimize file opening overhead
        # Structure: {file_path: [row_indices]}
        # We can't easily change __getitem__ to batch by file unless we use a custom sampler.
        # But we can at least cache the file handle if we assume sequential access (using a custom worker_init_fn or careful caching).
        # HOWEVER, PyTorch DataLoader workers are separate processes.
        # Simple optimization: Cache a few opened file handles (LRU per worker).
        self._file_cache = {}
        self._file_cache_order = deque()
        
    def __len__(self):
        return len(self.dataframe)
        
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        file_name = row['source_file']
        
        # Handle source_segments: "['trial0/segment10']" -> "trial0/segment10"
        segment_path_str = row['source_segments']
        
        # Safer parsing
        try:
            segment_list = ast.literal_eval(segment_path_str)
            if isinstance(segment_list, list) and len(segment_list) > 0:
                segment_path = segment_list[0]
            else:
                # Fallback if eval works but structure unexpected
                segment_path = str(segment_list)
        except:
            # Fallback for simple strings or malformed
            segment_path = segment_path_str.replace('[', '').replace(']', '').replace("'", "").replace('"', "")

        # Construct HDF5 path
        file_path = self.hdf5_root / file_name
        if not file_path.exists():
             # Try TUAB subfolder
             file_path = self.hdf5_root / "TUAB" / file_name
        
        try:
            # LRU cache for file handles (per worker)
            file_path_str = str(file_path)
            if self.cache_size > 0 and file_path_str in self._file_cache:
                f = self._file_cache[file_path_str]
                if file_path_str in self._file_cache_order:
                    self._file_cache_order.remove(file_path_str)
                self._file_cache_order.append(file_path_str)
            else:
                f = h5py.File(file_path, 'r')
                if self.cache_size > 0:
                    self._file_cache[file_path_str] = f
                    self._file_cache_order.append(file_path_str)
                    if len(self._file_cache_order) > self.cache_size:
                        old_path = self._file_cache_order.popleft()
                        old_f = self._file_cache.pop(old_path, None)
                        if old_f is not None:
                            try:
                                old_f.close()
                            except Exception:
                                pass

            # segment_path e.g. "trial0/segment10"
            # split by /
            parts = segment_path.split('/')
            obj = f
            for part in parts:
                obj = obj[part]
            
            # Now obj should be the group containing 'eeg'
            data = obj['eeg'][()]
            
            # Standardize channels
            ch_names = None
            if 'chn_name' in f.attrs:
                ch_names = f.attrs['chn_name']
            elif 'chOrder' in f.attrs:
                ch_names = f.attrs['chOrder']
            elif 'chOrder' in obj['eeg'].attrs:
                ch_names = obj['eeg'].attrs['chOrder']
            
            if ch_names is not None:
                new_data = np.zeros((len(standard_1020), data.shape[1]), dtype=data.dtype)
                
                if len(ch_names) > 0 and isinstance(ch_names[0], bytes):
                    ch_names = [c.decode('utf-8') for c in ch_names]
                
                for i, ch in enumerate(ch_names):
                    ch = ch.upper().strip()
                    if ch in standard_1020:
                        idx = standard_1020.index(ch)
                        new_data[idx] = data[i]
                data = new_data
            else:
                new_data = np.zeros((len(standard_1020), data.shape[1]), dtype=data.dtype)
                # print(f"Warning: No channel info for {file_path}")
                data = new_data
            
            # data shape (21, 2000)
            # Crop to window_size (1600)
            if data.shape[1] > self.window_size:
                start = np.random.randint(0, data.shape[1] - self.window_size + 1)
                data = data[:, start:start+self.window_size]
            elif data.shape[1] < self.window_size:
                pad_len = self.window_size - data.shape[1]
                data = np.pad(data, ((0,0), (0, pad_len)), 'constant')
                
            data = torch.FloatTensor(data)
            
            # Features
            features = row[self.feature_cols].values.astype(np.float32)
            features = torch.FloatTensor(features)
            
            return data, features
            
        except Exception as e:
            print(f"Error loading {file_path} {segment_path}: {e}")
            return torch.zeros((21, self.window_size)), torch.zeros(len(self.feature_cols))
    
    def __del__(self):
        try:
            for f in self._file_cache.values():
                try:
                    f.close()
                except Exception:
                    pass
        except Exception:
            pass

    def get_feature_names(self):
        return self.feature_cols

    def get_ch_names(self):
        return standard_1020
