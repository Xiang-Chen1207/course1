
import h5py
import bisect
import numpy as np
import torch
from pathlib import Path
from typing import List
from torch.utils.data import Dataset
import ast
from collections import OrderedDict

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

# Pre-computed channel index lookup dict for O(1) access instead of O(n) list.index()
STANDARD_1020_INDEX = {ch: i for i, ch in enumerate(standard_1020)}
NUM_STANDARD_CHANNELS = len(standard_1020)

list_path = List[Path]


def _normalize_dataset_key(name):
    if name is None:
        return None
    return str(name).strip().lower()

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
    def __init__(self, dataframe, hdf5_root, window_size=1600, cache_size=4, hdf5_root_map=None,
                 skip_pre_resolve=True, skip_existence_check=True):
        """
        Optimized dataset for feature prediction.

        Args:
            dataframe: DataFrame with feature data and file metadata
            hdf5_root: Root directory for HDF5 files
            window_size: Size of EEG window to extract
            cache_size: Number of HDF5 files to cache (LRU)
            hdf5_root_map: Optional dict mapping dataset names to HDF5 roots
            skip_pre_resolve: If True, skip slow pre-resolution of paths (recommended for large datasets)
            skip_existence_check: If True, skip file existence checks during init (recommended)
        """
        self.dataframe = dataframe.reset_index(drop=True)
        self.hdf5_root = Path(hdf5_root) if hdf5_root else None
        self.window_size = window_size
        self.cache_size = max(int(cache_size), 0)
        self.hdf5_root_map = None
        if hdf5_root_map:
            self.hdf5_root_map = {
                _normalize_dataset_key(k): Path(v) for k, v in hdf5_root_map.items()
            }

        # Identify feature columns
        metadata_cols = [
            'trial_ids', 'segment_ids', 'session_id', 'primary_label', 'labels',
            'start_time', 'end_time', 'total_time_length', 'merge_count',
            'source_segments', 'source_file', 'sub_id', 'subject_id', 'unique_sub_id', 'dataset_source'
        ]
        self.feature_cols = [c for c in self.dataframe.columns if c not in metadata_cols]

        # LRU cache for file handles using OrderedDict for O(1) move_to_end
        self._file_cache = OrderedDict()

        # OPTIMIZATION: Use lazy loading by default - skip slow pre-resolution
        # Pre-resolution with iterrows() is extremely slow for large datasets
        self._skip_pre_resolve = skip_pre_resolve
        self._skip_existence_check = skip_existence_check

        self._resolved_file_paths = None
        self._resolved_segment_paths = None
        self._missing_count = None

        # Pre-extract source_file and source_segments columns as numpy arrays for fast access
        self._source_files = self.dataframe['source_file'].values if 'source_file' in self.dataframe.columns else None
        self._source_segments = self.dataframe['source_segments'].values if 'source_segments' in self.dataframe.columns else None
        self._dataset_sources = self.dataframe['dataset_source'].values if 'dataset_source' in self.dataframe.columns else None

        # Pre-extract feature values as numpy array for fast access
        self._feature_values = self.dataframe[self.feature_cols].values.astype(np.float32)
        # Clean NaN/Inf values once upfront
        self._feature_values = np.nan_to_num(self._feature_values, nan=0.0, posinf=0.0, neginf=0.0)
        
    def __len__(self):
        return len(self.dataframe)

    def get_missing_count(self):
        return self._missing_count

    def get_file_path(self, idx):
        if self._resolved_file_paths is not None:
            return self._resolved_file_paths[idx]
        # Use fast method with pre-extracted arrays
        file_name = self._source_files[idx] if self._source_files is not None else None
        dataset_source = self._dataset_sources[idx] if self._dataset_sources is not None else None
        return self._resolve_file_path_fast(file_name, dataset_source, check_exists=False)

    def get_segment_path(self, idx):
        if self._resolved_segment_paths is not None:
            return self._resolved_segment_paths[idx]
        # Use fast method with pre-extracted arrays
        segment_str = self._source_segments[idx] if self._source_segments is not None else None
        return self._parse_segment_path_fast(segment_str)

    def _parse_segment_path_fast(self, segment_path_str):
        """Fast segment path parsing without ast.literal_eval."""
        if segment_path_str is None or (isinstance(segment_path_str, float) and np.isnan(segment_path_str)):
            return None

        segment_path_str = str(segment_path_str)

        # Fast path: if it doesn't start with '[', it's already a simple path
        if not segment_path_str.startswith('['):
            return segment_path_str.strip()

        # Extract first element from list-like string: "['trial0/segment0', ...]" -> "trial0/segment0"
        # This is faster than ast.literal_eval for simple cases
        try:
            # Remove brackets and split
            inner = segment_path_str.strip()[1:-1]  # Remove [ and ]
            if not inner:
                return None
            # Find first element (before first comma if exists)
            first_elem = inner.split(',')[0].strip()
            # Remove quotes
            first_elem = first_elem.strip("'\"")
            return first_elem
        except Exception:
            # Fallback to original method
            try:
                segment_list = ast.literal_eval(segment_path_str)
                if isinstance(segment_list, list) and len(segment_list) > 0:
                    return segment_list[0]
                return str(segment_list)
            except Exception:
                return segment_path_str.replace('[', '').replace(']', '').replace("'", "").replace('"', '')

    def _parse_segment_path(self, row):
        if 'source_segments' not in row.index:
            return None
        return self._parse_segment_path_fast(row['source_segments'])

    def _resolve_file_path_fast(self, file_name, dataset_source=None, check_exists=False):
        """Fast file path resolution using raw values instead of row objects."""
        if file_name is None:
            return None

        dataset_key = _normalize_dataset_key(dataset_source)

        hdf5_root = self.hdf5_root
        if self.hdf5_root_map and dataset_key in self.hdf5_root_map:
            hdf5_root = self.hdf5_root_map[dataset_key]

        file_name_str = str(file_name)
        if Path(file_name_str).is_absolute():
            file_path = Path(file_name_str)
        else:
            if hdf5_root is None:
                file_path = Path(file_name_str)
            else:
                file_path = hdf5_root / file_name_str

        if not check_exists:
            return file_path

        if file_path.exists():
            return file_path

        if hdf5_root is not None and dataset_source:
            alt_path = hdf5_root / str(dataset_source) / file_name_str
            if alt_path.exists():
                return alt_path

        if hdf5_root is not None:
            alt_path = hdf5_root / "TUAB" / file_name_str
            if alt_path.exists():
                return alt_path

        return None

    def _resolve_file_path(self, row, check_exists=True):
        file_name = row['source_file']
        dataset_source = row['dataset_source'] if 'dataset_source' in row.index else None
        return self._resolve_file_path_fast(file_name, dataset_source, check_exists)
        
    def __getitem__(self, idx):
        # OPTIMIZATION: Use pre-extracted arrays instead of dataframe.iloc[idx]
        # This avoids slow pandas row access
        file_name = self._source_files[idx] if self._source_files is not None else None
        segment_str = self._source_segments[idx] if self._source_segments is not None else None
        dataset_source = self._dataset_sources[idx] if self._dataset_sources is not None else None

        # Resolve paths using fast methods
        file_path = self._resolve_file_path_fast(file_name, dataset_source, check_exists=False)
        segment_path = self._parse_segment_path_fast(segment_str)

        if file_path is None or segment_path is None:
            return torch.zeros((NUM_STANDARD_CHANNELS, self.window_size)), torch.zeros(len(self.feature_cols))

        try:
            # LRU cache for file handles using OrderedDict for O(1) operations
            file_path_str = str(file_path)
            if self.cache_size > 0 and file_path_str in self._file_cache:
                f = self._file_cache[file_path_str]
                # Move to end (most recently used) - O(1) operation
                self._file_cache.move_to_end(file_path_str)
            else:
                f = h5py.File(file_path, 'r')
                if self.cache_size > 0:
                    self._file_cache[file_path_str] = f
                    # Evict oldest if over capacity
                    while len(self._file_cache) > self.cache_size:
                        _, old_f = self._file_cache.popitem(last=False)
                        try:
                            old_f.close()
                        except Exception:
                            pass

            # segment_path e.g. "trial0/segment10"
            parts = segment_path.split('/')
            obj = f
            for part in parts:
                obj = obj[part]

            # Read EEG data
            data = obj['eeg'][()]
            # Sanitize NaN/Inf early
            if isinstance(data, np.ndarray):
                data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

            # Standardize channels
            ch_names = None
            if 'chn_name' in f.attrs:
                ch_names = f.attrs['chn_name']
            elif 'chOrder' in f.attrs:
                ch_names = f.attrs['chOrder']
            elif 'chOrder' in obj['eeg'].attrs:
                ch_names = obj['eeg'].attrs['chOrder']

            if ch_names is not None:
                new_data = np.zeros((NUM_STANDARD_CHANNELS, data.shape[1]), dtype=data.dtype)

                if len(ch_names) > 0 and isinstance(ch_names[0], bytes):
                    ch_names = [c.decode('utf-8') for c in ch_names]

                for i, ch in enumerate(ch_names):
                    ch_upper = ch.upper().strip()
                    ch_idx = STANDARD_1020_INDEX.get(ch_upper)
                    if ch_idx is not None:
                        new_data[ch_idx] = data[i]
                data = new_data
            else:
                new_data = np.zeros((NUM_STANDARD_CHANNELS, data.shape[1]), dtype=data.dtype)
                data = new_data

            # Crop or pad to window_size
            if data.shape[1] > self.window_size:
                start = np.random.randint(0, data.shape[1] - self.window_size + 1)
                data = data[:, start:start + self.window_size]
            elif data.shape[1] < self.window_size:
                pad_len = self.window_size - data.shape[1]
                data = np.pad(data, ((0, 0), (0, pad_len)), 'constant')

            data = torch.from_numpy(data).float()
            data = torch.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

            # OPTIMIZATION: Use pre-extracted feature values
            features = torch.from_numpy(self._feature_values[idx].copy()).float()

            return data, features

        except Exception as e:
            print(f"Error loading {file_path} {segment_path}: {e}")
            return torch.zeros((NUM_STANDARD_CHANNELS, self.window_size)), torch.zeros(len(self.feature_cols))
    
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
