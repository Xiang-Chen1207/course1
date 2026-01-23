"""
数据加载器：从 HDF5 文件加载 EEG 数据
"""
import h5py
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Generator, Tuple, Optional
from pathlib import Path


@dataclass
class SegmentData:
    """单个 segment 的数据结构"""
    trial_id: int
    segment_id: int
    session_id: int
    eeg_data: np.ndarray  # shape: (n_channels, n_timepoints)
    label: int
    start_time: float
    end_time: float
    time_length: float


@dataclass
class SubjectInfo:
    """被试信息"""
    subject_id: str
    dataset_name: str
    task_type: str
    sampling_rate: float
    channel_names: List[str]
    n_channels: int


class EEGDataLoader:
    """EEG 数据加载器"""

    def __init__(self, h5_path: str):
        """
        初始化数据加载器

        Args:
            h5_path: HDF5 文件路径
        """
        self.h5_path = Path(h5_path)
        if not self.h5_path.exists():
            raise FileNotFoundError(f"文件不存在: {h5_path}")

        self._subject_info: Optional[SubjectInfo] = None
        self._trial_info: Dict = {}
        self._file: Optional[h5py.File] = None
        self._trial_names_cache: Optional[List[str]] = None
        self._segment_names_cache: Dict[str, List[str]] = {}
        self._trial_attrs_cache: Dict[str, Dict[str, int]] = {}

    def __enter__(self):
        self._open_file()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()

    def _open_file(self) -> h5py.File:
        """确保文件句柄已打开，并返回句柄。"""
        if self._file is None:
            # 提升 HDF5 读取性能：增大 chunk cache
            self._file = h5py.File(
                self.h5_path,
                'r',
                rdcc_nbytes=64 * 1024 * 1024,
                rdcc_nslots=1_048_576
            )
        return self._file

    def close(self):
        """关闭文件句柄"""
        if self._file is not None:
            try:
                self._file.close()
            finally:
                self._file = None
                self._trial_names_cache = None
                self._segment_names_cache.clear()
                self._trial_attrs_cache.clear()

    def get_subject_info(self) -> SubjectInfo:
        """获取被试信息"""
        if self._subject_info is not None:
            return self._subject_info

        f = self._open_file()
        attrs = dict(f.attrs)
        channel_names = attrs.get('chn_name', [])
        if isinstance(channel_names, np.ndarray):
            channel_names = [ch.decode('utf-8') if isinstance(ch, bytes) else ch
                             for ch in channel_names]

        # Fallback for missing channel names (e.g. TUAB 21 channels)
        if not channel_names:
            try:
                # Try to infer from data shape
                # Iterate to find the first valid segment
                for trial_key in f.keys():
                    if trial_key.startswith('trial'):
                        trial_grp = f[trial_key]
                        for seg_key in trial_grp.keys():
                            if seg_key.startswith('segment') or seg_key.startswith('sample'):
                                seg_grp = trial_grp[seg_key]
                                if 'eeg' in seg_grp:
                                    # Attempt to read chn_name from segment attributes if missing in root
                                    if not channel_names and 'chn_name' in seg_grp['eeg'].attrs:
                                        c_names = seg_grp['eeg'].attrs['chn_name']
                                        if isinstance(c_names, np.ndarray):
                                            channel_names = [ch.decode('utf-8') if isinstance(ch, bytes) else str(ch) for ch in c_names]
                                        elif isinstance(c_names, (list, tuple)):
                                            channel_names = [ch.decode('utf-8') if isinstance(ch, bytes) else str(ch) for ch in c_names]

                                    if channel_names:
                                        break
                                    
                                    # Removed assumption about TUAB 21 channels. 
                                    # Users requested explicit channel finding or failure.
                                    # data_shape = seg_grp['eeg'].shape
                                    # if data_shape[0] == 21: ...
                                    break

                        if channel_names:
                            break
            except Exception as e:
                print(f"Warning: Failed to infer channel names: {e}")

        self._subject_info = SubjectInfo(
            subject_id=str(attrs.get('subject_id', '0')),
            dataset_name=attrs.get('dataset_name', '').decode('utf-8')
            if isinstance(attrs.get('dataset_name', ''), bytes)
            else str(attrs.get('dataset_name', '')),
            task_type=attrs.get('task_type', '').decode('utf-8')
            if isinstance(attrs.get('task_type', ''), bytes)
            else str(attrs.get('task_type', '')),
            sampling_rate=float(attrs.get('rsFreq', 200.0)),
            channel_names=channel_names,
            n_channels=len(channel_names)
        )

        return self._subject_info

    def get_trial_names(self) -> List[str]:
        """获取所有 trial 名称"""
        if self._trial_names_cache is not None:
            return self._trial_names_cache

        f = self._open_file()
        trials = [key for key in f.keys() if key.startswith('trial')]
        # 按数字排序
        trials.sort(key=lambda x: int(x.replace('trial', '')))
        self._trial_names_cache = trials
        return trials

    def get_segment_names(self, trial_name: str) -> List[str]:
        """获取指定 trial 下的所有 segment 名称"""
        if trial_name in self._segment_names_cache:
            return self._segment_names_cache[trial_name]

        f = self._open_file()
        if trial_name not in f:
            return []
        
        # Support both 'segment' and 'sample' prefixes
        segments = [key for key in f[trial_name].keys() 
                   if key.startswith('segment') or key.startswith('sample')]
        
        def _sort_key(x):
            if x.startswith('segment'):
                return int(x.replace('segment', ''))
            elif x.startswith('sample'):
                return int(x.replace('sample', ''))
            return 0
            
        segments.sort(key=_sort_key)
        self._segment_names_cache[trial_name] = segments
        return segments

    def get_segment(self, trial_name: str, segment_name: str) -> SegmentData:
        """获取单个 segment 的数据"""
        f = self._open_file()
        trial_grp = f[trial_name]
        seg_grp = trial_grp[segment_name]
        eeg_dset = seg_grp['eeg']

        # 获取 EEG 数据
        eeg_data = (eeg_dset[...]).astype(np.float32, copy=False)

        # 读取试次属性（缓存）
        trial_attrs = self._trial_attrs_cache.get(trial_name)
        if trial_attrs is None:
            raw_trial_attrs = trial_grp.attrs
            trial_id = int(raw_trial_attrs.get('trial_id', 0))
            session_id = raw_trial_attrs.get('session_id', 1)
            if isinstance(session_id, str) and session_id.startswith('s'):
                try:
                    session_id = int(session_id[1:])
                except ValueError:
                    session_id = 1
            else:
                try:
                    session_id = int(session_id)
                except (ValueError, TypeError):
                    session_id = 1
            trial_attrs = {'trial_id': trial_id, 'session_id': session_id}
            self._trial_attrs_cache[trial_name] = trial_attrs

        # 读取 segment 属性（仅取需要字段）
        eeg_attrs = eeg_dset.attrs

        # 处理 label
        label = eeg_attrs.get('label', 0)
        if isinstance(label, np.ndarray):
            label = int(label[0]) if label.size > 0 else 0
        elif isinstance(label, list):
            label = int(label[0]) if len(label) > 0 else 0
        else:
            try:
                label = int(label)
            except (TypeError, ValueError):
                label = 0

        return SegmentData(
            trial_id=trial_attrs['trial_id'],
            segment_id=int(eeg_attrs.get('segment_id', 0)),
            session_id=trial_attrs['session_id'],
            eeg_data=eeg_data,
            label=label,
            start_time=float(eeg_attrs.get('start_time', 0)),
            end_time=float(eeg_attrs.get('end_time', 0)),
            time_length=float(eeg_attrs.get('time_length', 2.0))
        )

    def get_segments(self, trial_name: str, segment_names: List[str]) -> List[SegmentData]:
        """批量获取同一 trial 下的多个 segment（复用 trial 元信息）"""
        if not segment_names:
            return []

        f = self._open_file()
        trial_grp = f[trial_name]

        # 读取试次属性（缓存）
        trial_attrs = self._trial_attrs_cache.get(trial_name)
        if trial_attrs is None:
            raw_trial_attrs = trial_grp.attrs
            trial_id = int(raw_trial_attrs.get('trial_id', 0))
            session_id = raw_trial_attrs.get('session_id', 1)
            if isinstance(session_id, str) and session_id.startswith('s'):
                try:
                    session_id = int(session_id[1:])
                except ValueError:
                    session_id = 1
            else:
                try:
                    session_id = int(session_id)
                except (ValueError, TypeError):
                    session_id = 1
            trial_attrs = {'trial_id': trial_id, 'session_id': session_id}
            self._trial_attrs_cache[trial_name] = trial_attrs

        results: List[SegmentData] = []
        for segment_name in segment_names:
            seg_grp = trial_grp[segment_name]
            eeg_dset = seg_grp['eeg']
            eeg_data = (eeg_dset[...]).astype(np.float32, copy=False)
            eeg_attrs = eeg_dset.attrs

            label = eeg_attrs.get('label', 0)
            if isinstance(label, np.ndarray):
                label = int(label[0]) if label.size > 0 else 0
            elif isinstance(label, list):
                label = int(label[0]) if len(label) > 0 else 0
            else:
                try:
                    label = int(label)
                except (TypeError, ValueError):
                    label = 0

            results.append(SegmentData(
                trial_id=trial_attrs['trial_id'],
                segment_id=int(eeg_attrs.get('segment_id', 0)),
                session_id=trial_attrs['session_id'],
                eeg_data=eeg_data,
                label=label,
                start_time=float(eeg_attrs.get('start_time', 0)),
                end_time=float(eeg_attrs.get('end_time', 0)),
                time_length=float(eeg_attrs.get('time_length', 2.0))
            ))

        return results

    def iter_segments(self) -> Generator[Tuple[str, str, SegmentData], None, None]:
        """
        迭代所有 segments

        Yields:
            (trial_name, segment_name, SegmentData) 元组
        """
        for trial_name in self.get_trial_names():
            for segment_name in self.get_segment_names(trial_name):
                yield trial_name, segment_name, self.get_segment(trial_name, segment_name)

    def get_all_data(self) -> Dict[str, Dict[str, SegmentData]]:
        """
        获取所有数据

        Returns:
            嵌套字典 {trial_name: {segment_name: SegmentData}}
        """
        data = {}
        for trial_name, segment_name, segment in self.iter_segments():
            if trial_name not in data:
                data[trial_name] = {}
            data[trial_name][segment_name] = segment
        return data

    def get_statistics(self) -> Dict:
        """获取数据集统计信息"""
        stats = {
            'n_trials': 0,
            'n_segments': 0,
            'labels': {0: 0, 1: 0, 2: 0},
            'sessions': {}
        }

        for trial_name in self.get_trial_names():
            stats['n_trials'] += 1
            segments = self.get_segment_names(trial_name)
            stats['n_segments'] += len(segments)

            # 获取第一个 segment 的信息
            if segments:
                seg = self.get_segment(trial_name, segments[0])
                stats['labels'][seg.label] = stats['labels'].get(seg.label, 0) + len(segments)

                session_id = seg.session_id
                if session_id not in stats['sessions']:
                    stats['sessions'][session_id] = {'n_trials': 0, 'n_segments': 0}
                stats['sessions'][session_id]['n_trials'] += 1
                stats['sessions'][session_id]['n_segments'] += len(segments)

        return stats
