#!/usr/bin/env python3
"""
修复 SEED_basic 数据集中缺失特征的 segments
1. 读取 SEED_BASIC_column_mismatch 中的文件列表
2. 对每个受影响的 Subject，重新加载数据，生成 Microstate 模板
3. 重新计算特定 Segment 的特征 (使用多进程并行)
4. 更新单个 Segment CSV 文件
5. 更新汇总文件 all_merged_features.csv
6. 重新生成 all_merged_features_zscored.csv (Subject-specific Z-score)
"""

import sys
import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
import concurrent.futures
import traceback

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from merged_segment_extraction import MergedSegmentExtractor
from eeg_feature_extraction.config import Config
from selective_feature_extraction import apply_preset, FeatureSelectionConfig
from eeg_feature_extraction.data_loader import EEGDataLoader

def process_segment_task(args):
    """
    Worker function to process a single segment.
    args: (seg_idx, merged_seg, extractor, microstate_analyzer, seed_basic_dir, sub_id, h5_filename)
    """
    seg_idx, merged_seg, extractor, microstate_analyzer, seed_basic_dir, sub_id, h5_filename = args
    
    try:
        # 增加超时时间
        extractor.FEATURE_TIMEOUT_SEC = 60 
        
        # 计算特征
        features = extractor.extract_features(merged_seg.eeg_data, microstate_analyzer=microstate_analyzer)
        
        # 补充元数据
        features['trial_ids'] = str(merged_seg.trial_ids)
        features['segment_ids'] = str(merged_seg.segment_ids)
        features['session_id'] = merged_seg.session_id
        features['primary_label'] = merged_seg.primary_label
        features['labels'] = str(merged_seg.labels)
        features['start_time'] = merged_seg.start_time
        features['end_time'] = merged_seg.end_time
        features['total_time_length'] = merged_seg.total_time_length
        features['merge_count'] = merged_seg.merge_count
        features['source_segments'] = str(merged_seg.source_segments)
        features['source_file'] = h5_filename

        # 1. 更新单个 CSV 文件
        seg_csv_path = seed_basic_dir / sub_id / f"merged_segment_{seg_idx:04d}.csv"
        if not seg_csv_path.parent.exists():
            try:
                seg_csv_path.parent.mkdir(parents=True, exist_ok=True)
            except FileExistsError:
                pass
            
        df_seg = pd.DataFrame([features])
        df_seg.to_csv(seg_csv_path, index=False)
        
        return features
        
    except Exception as e:
        print(f"Error processing segment {seg_idx} for {sub_id}: {e}")
        traceback.print_exc()
        return None

def main():
    # 配置路径
    mismatch_dir = Path("/mnt/dataset4/cx/code/EEG_LLM_text/SEED_BASIC_column_mismatch")
    seed_basic_dir = Path("/mnt/dataset4/cx/code/EEG_LLM_text/SEED_basic")
    summary_csv_path = seed_basic_dir / "all_merged_features.csv"
    zscored_csv_path = seed_basic_dir / "all_merged_features_zscored.csv"
    h5_root = Path("/mnt/dataset2/hdf5_datasets/SEED")

    # 参数配置 (必须与原始运行参数一致)
    MERGE_COUNT = 1
    PRESET = "standard"
    MICROSTATE_SEGS = 20
    CROSS_TRIAL = False # 默认是 False
    
    # 并行配置
    MAX_WORKERS = 80  # 根据用户请求，可以使用较多CPU

    print("正在扫描需要修复的 Segments...")
    mismatch_files = list(mismatch_dir.glob("*.csv"))
    if not mismatch_files:
        print("未找到任何 mismatch 文件，退出。")
        return

    # 解析任务：按 Subject 分组
    # 文件名格式: sub_X__merged_segment_Y.csv
    tasks = {}  # subject_id -> [segment_index, ...]
    
    for p in mismatch_files:
        filename = p.name
        try:
            parts = filename.split("__")
            sub_str = parts[0]  # sub_X
            seg_part = parts[1] # merged_segment_Y.csv
            seg_idx = int(seg_part.replace("merged_segment_", "").replace(".csv", ""))
            
            if sub_str not in tasks:
                tasks[sub_str] = []
            tasks[sub_str].append(seg_idx)
        except Exception as e:
            print(f"解析文件名失败: {filename}, error: {e}")

    print(f"共发现 {len(tasks)} 个 Subject 需要修复，涉及 {sum(len(v) for v in tasks.values())} 个 Segments。")

    # 加载汇总表
    if not summary_csv_path.exists():
        print(f"错误: 汇总表不存在: {summary_csv_path}")
        return
    
    df_summary = pd.read_csv(summary_csv_path)
    print(f"原始汇总表行数: {len(df_summary)}")

    # 准备特征提取器配置
    selection_config = apply_preset(PRESET)
    
    # 开始处理
    
    for sub_id, seg_indices in tasks.items():
        print(f"\n正在处理 {sub_id} (需修复 {len(seg_indices)} 个 segments)...")
        
        # 找到 HDF5 文件
        h5_candidates = list(h5_root.rglob(f"{sub_id}.h5")) + list(h5_root.rglob(f"{sub_id}.hdf5"))
        if not h5_candidates:
            print(f"警告: 未找到 {sub_id} 的 HDF5 文件，跳过。")
            continue
        h5_path = str(h5_candidates[0])
        h5_filename = Path(h5_path).name
        
        # 初始化 Extractor
        config = Config()
        extractor = MergedSegmentExtractor(
            config=config,
            selection_config=selection_config,
            merge_count=MERGE_COUNT,
            cross_trial=CROSS_TRIAL,
            n_jobs=1, # 内部不并行，我们在外部并行
            microstate_segments_per_trial=MICROSTATE_SEGS
        )
        
        # 加载数据
        loader = EEGDataLoader(h5_path)
        subject_info = loader.get_subject_info()
        
        # 同步配置
        extractor.config.update_from_electrode_names(subject_info.channel_names)
        extractor.config.sampling_rate = subject_info.sampling_rate
        # 重建计算器
        extractor.psd_computer = extractor.psd_computer.__class__(
            sampling_rate=extractor.config.sampling_rate,
            use_gpu=extractor.config.use_gpu,
            nperseg=extractor.config.nperseg,
            noverlap=extractor.config.noverlap,
            nfft=extractor.config.nfft
        )
        extractor.feature_computers = {}
        extractor._initialize_computers()

        # 生成 Microstate 模板 (耗时步骤)
        microstate_analyzer = None
        if 'microstate' in extractor.selection_config.get_required_groups():
            print("  正在生成 Microstate 模板...")
            microstate_analyzer = extractor._compute_microstate_template(loader, verbose=False)

        # 获取所有 Merged Segments
        if extractor.cross_trial:
            merged_segments = extractor._merge_segments_cross_trial(loader)
        else:
            merged_segments = extractor._merge_segments_within_trial(loader)
            
        print(f"  总 Merged Segments: {len(merged_segments)}")
        
        # 准备并行任务
        # 只处理 seg_indices 中的 segments
        valid_indices = [idx for idx in seg_indices if idx < len(merged_segments)]
        if len(valid_indices) < len(seg_indices):
            print(f"  警告: 忽略 {len(seg_indices) - len(valid_indices)} 个越界索引")
            
        # 使用 ProcessPoolExecutor 并行处理
        results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # 提交任务
            future_to_idx = {}
            for seg_idx in valid_indices:
                merged_seg = merged_segments[seg_idx]
                args = (seg_idx, merged_seg, extractor, microstate_analyzer, seed_basic_dir, sub_id, h5_filename)
                future = executor.submit(process_segment_task, args)
                future_to_idx[future] = seg_idx
            
            # 收集结果
            for future in tqdm(concurrent.futures.as_completed(future_to_idx), total=len(valid_indices), desc=f"并行修复 {sub_id}"):
                idx = future_to_idx[future]
                try:
                    features = future.result()
                    if features:
                        results.append(features)
                except Exception as exc:
                    print(f"  Segment {idx} generated an exception: {exc}")

        # 批量更新 df_summary
        print(f"  正在更新汇总表 ({len(results)} 个结果)...")
        for features in results:
            source_segs_str = features['source_segments']
            source_file_str = features['source_file']
            
            # 在 df_summary 中查找
            mask = (df_summary['source_file'] == source_file_str) & \
                   (df_summary['source_segments'] == source_segs_str)
            
            if mask.any():
                idx_to_update = df_summary.index[mask]
                for col, val in features.items():
                    if col in df_summary.columns:
                        df_summary.loc[idx_to_update, col] = val
            else:
                print(f"  警告: 在汇总表中未找到行: {source_file_str} - {source_segs_str}")

    # 保存更新后的汇总表
    print("\n保存更新后的汇总表...")
    df_summary.to_csv(summary_csv_path, index=False)
    print(f"已保存至 {summary_csv_path}")

    # 更新 Z-score 文件
    print("\n正在生成 Z-score 文件...")
    # 策略: 按 Subject (source_file) 进行 Z-score
    # 排除元数据列
    meta_cols = ['trial_ids', 'segment_ids', 'session_id', 'primary_label', 'labels',
                 'start_time', 'end_time', 'total_time_length', 'merge_count', 
                 'source_segments', 'source_file']
    
    feature_cols = [c for c in df_summary.columns if c not in meta_cols]
    
    df_zscored = df_summary.copy()
    
    # 按 subject 分组进行 z-score
    for source_file, group in tqdm(df_zscored.groupby('source_file'), desc="Z-scoring"):
        # 获取该 subject 的特征数据
        sub_features = group[feature_cols]
        
        # 计算 Z-score: (x - mean) / std
        # 需处理 NaN (忽略 NaN 计算 mean/std，结果仍为 NaN)
        z = (sub_features - sub_features.mean()) / sub_features.std()
        
        # 将计算结果填回
        df_zscored.loc[group.index, feature_cols] = z

    # 保存 Z-score 文件
    df_zscored.to_csv(zscored_csv_path, index=False)
    print(f"已保存 Z-score 文件至 {zscored_csv_path}")
    print("全部完成！")

if __name__ == "__main__":
    main()
