
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
import pandas as pd
from pathlib import Path
from timm.models import create_model
from optim_factory import create_optimizer
from engine_for_pretraining import train_one_epoch
from utils import NativeScalerWithGradNormCount as NativeScaler
import utils
import modeling_pretrain
import modeling_vqnsp
from data_processor.dataset import FeaturePredictionDataset
from sklearn.model_selection import train_test_split
from einops import rearrange
from engine_for_pretraining import random_masking
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

# Dataset paths
DATASETS = {
    'ADHD': '/mnt/cx/EEG_text/data/ADHD.csv',
    'BCIC2A': '/mnt/cx/EEG_text/data/BCIC2A.csv',
    'SEEDIV': '/mnt/cx/EEG_text/data/SEEDIV.csv',
    'SEEDV': '/mnt/cx/EEG_text/data/SEEDV.csv',
    'SleepEDF': '/mnt/cx/EEG_text/data/SLEEPEDF.csv',
    'Workload_MATB': '/mnt/cx/EEG_text/data/WORKLOAD_MATB.csv',
    'TUAB': '/mnt/cx/EEG_text/data/TUAB.csv'
}

# Default HDF5 roots per dataset
DEFAULT_HDF5_ROOT_MAP = {
    'ADHD': '/pretrain-clip/hdf5_datasets/ADHD',
    'BCIC2A': '/pretrain-clip/hdf5_datasets/BCIC2A',
    'SEEDIV': '/pretrain-clip/hdf5_datasets/SEEDIV',
    'SEEDV': '/pretrain-clip/hdf5_datasets/SEEDV',
    'SleepEDF': '/pretrain-clip/hdf5_datasets/SleepEDF',
    'Workload_MATB': '/pretrain-clip/hdf5_datasets/Workload_MATB',
    'TUAB': '/eeg-h5-files/TUAB'
}


def _normalize_dataset_key(name):
    if name is None:
        return None
    return str(name).strip().lower()


def _infer_dataset_name_from_path(csv_path: str) -> str:
    try:
        return Path(csv_path).stem
    except Exception:
        return ""


def parse_hdf5_root_map(arg_value: str):
    if not arg_value:
        return None
    if isinstance(arg_value, str) and arg_value.lower() in {"auto", "default"}:
        return DEFAULT_HDF5_ROOT_MAP
    if os.path.isfile(arg_value) and arg_value.endswith(".json"):
        with open(arg_value, "r", encoding="utf-8") as f:
            return json.load(f)
    # Parse comma-separated key=path pairs
    root_map = {}
    for item in arg_value.split(","):
        if not item:
            continue
        if "=" not in item:
            continue
        key, path = item.split("=", 1)
        root_map[key.strip()] = path.strip()
    return root_map

def load_and_merge_datasets(datasets_config, allowed_datasets=None):
    all_dfs = []
    
    # Define common feature columns (intersection or union, here we assume intersection of important ones)
    # We will let the first dataset define the feature columns and ensure others have them.
    # Actually, FeaturePredictionDataset handles feature columns dynamically, but for merging we need consistency if we concat.
    # Strategy: Read all, find common feature columns, keep only those.
    
    common_features = None
    
    for name, path in datasets_config.items():
        if allowed_datasets is not None and name not in allowed_datasets:
            continue
        # Check if we should load this dataset based on hdf5_root existence
        # The hdf5_root arg might point to a specific root, e.g. /mnt/dataset2/hdf5_datasets
        # We assume dataset folder structure: hdf5_root / NAME / ...
        # If NAME folder doesn't exist, we skip.
        
        # Wait, args.hdf5_root is not available here. We assume path exists.
        if not os.path.exists(path):
            print(f"Warning: Dataset CSV {name} not found at {path}, skipping.")
            continue
            
        print(f"Loading {name} from {path}...")
        df = pd.read_csv(path)
        
        # Standardize Subject ID
        if 'subject_id' in df.columns:
            # Add dataset name prefix to avoid collision (e.g. ADHD_0, BCIC_0)
            df['unique_sub_id'] = f"{name}_" + df['subject_id'].astype(str)
        elif 'source_file' in df.columns:
            # Extract from source_file (e.g. sub_1.h5 -> 1)
            # Assuming format sub_X.h5
            df['subject_id'] = df['source_file'].apply(lambda x: x.split('.')[0].split('_')[-1] if 'sub' in x else x)
            df['unique_sub_id'] = f"{name}_" + df['subject_id'].astype(str)
        else:
            print(f"Error: No subject ID found for {name}")
            continue
            
        # Add dataset source column for tracking
        df['dataset_source'] = name
        
        # Identify feature columns (exclude metadata)
        metadata_cols = [
            'trial_ids', 'segment_ids', 'session_id', 'primary_label', 'labels',
            'start_time', 'end_time', 'total_time_length', 'merge_count',
            'source_segments', 'source_file', 'sub_id', 'subject_id', 'unique_sub_id', 'dataset_source'
        ]
        feat_cols = [c for c in df.columns if c not in metadata_cols]
        
        # Filter out unnamed columns or non-feature columns
        feat_cols = [c for c in feat_cols if not c.startswith('Unnamed')]
        
        if common_features is None:
            common_features = set(feat_cols)
        else:
            common_features = common_features.intersection(set(feat_cols))
            
        all_dfs.append(df)
        
    if not all_dfs:
        raise ValueError("No datasets loaded!")
        
    common_features = sorted(list(common_features))
    print(f"Common features count: {len(common_features)}")
    
    final_dfs = []
    for df in all_dfs:
        # Keep only metadata + common features
        keep_cols = [c for c in df.columns if c in common_features or c in [
            'trial_ids', 'segment_ids', 'session_id', 'primary_label', 'labels',
            'start_time', 'end_time', 'total_time_length', 'merge_count',
            'source_segments', 'source_file', 'unique_sub_id' # Use unique_sub_id as 'sub_id'
        ]]
        
        # Rename unique_sub_id to sub_id for the Dataset class compatibility if needed
        # The Dataset class uses 'sub_id' if present, or creates it.
        # We will create a new dataframe with standard columns
        df_filtered = df[keep_cols].copy()
        df_filtered['sub_id'] = df['unique_sub_id'] 
        final_dfs.append(df_filtered)
        
    merged_df = pd.concat(final_dfs, ignore_index=True)
    
    # Fill NaNs with 0 if any (shouldn't be for common features but just in case)
    merged_df[common_features] = merged_df[common_features].fillna(0)
    
    print(f"Total samples: {len(merged_df)}")
    print(f"Total unique subjects: {merged_df['sub_id'].nunique()}")
    
    return merged_df

def evaluate_regression(model, dataset_eval, device, args):
    model.eval()
    
    if args.distributed:
        sampler_eval = torch.utils.data.DistributedSampler(
            dataset_eval, num_replicas=utils.get_world_size(), rank=utils.get_rank(), shuffle=False
        )
    else:
        sampler_eval = torch.utils.data.SequentialSampler(dataset_eval)

    data_loader = torch.utils.data.DataLoader(
        dataset_eval, sampler=sampler_eval, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=False
    )
    
    all_preds = []
    all_targets = []
    
    ch_names = dataset_eval.get_ch_names()
    input_chans = utils.get_input_chans(ch_names)
    input_chans = torch.tensor(input_chans).to(device)

    if utils.is_main_process():
        print("Evaluating regression...")
        
    with torch.no_grad():
        for batch in data_loader:
            samples, features = batch
            samples = samples.float().to(device) / 100
            features = features.to(device)
            samples = rearrange(samples, 'B N (A T) -> B N A T', T=200)
            
            bool_masked_pos = random_masking(samples.flatten(1, 2), mask_ratio=0.5).to(device)
            
            outputs = model(samples, input_chans=input_chans, bool_masked_pos=bool_masked_pos)
            reg_output = outputs[2]
            
            all_preds.append(reg_output)
            all_targets.append(features)
            
    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    if args.distributed:
        preds_gathered = utils.all_gather_batch([preds])[0]
        targets_gathered = utils.all_gather_batch([targets])[0]
        preds_np = preds_gathered.cpu().numpy()
        targets_np = targets_gathered.cpu().numpy()
    else:
        preds_np = preds.cpu().numpy()
        targets_np = targets.cpu().numpy()

    if utils.is_main_process():
        if len(preds_np) > len(dataset_eval):
            preds_np = preds_np[:len(dataset_eval)]
            targets_np = targets_np[:len(dataset_eval)]

        feature_names = dataset_eval.get_feature_names()
        results = {}
        print("\nFeature Regression Results:")
        for i, name in enumerate(feature_names):
            p = preds_np[:, i]
            t = targets_np[:, i]
            r2 = r2_score(t, p)
            r, _ = pearsonr(t, p)
            results[name] = {'r': r, 'r2': r2}
            # print(f"Feature {name}: r={r:.4f}, r2={r2:.4f}")
            
        # Summary
        mean_r = np.mean([res['r'] for res in results.values()])
        mean_r2 = np.mean([res['r2'] for res in results.values()])
        print(f"Mean R: {mean_r:.4f}, Mean R2: {mean_r2:.4f}")
        return results
    return None

def get_args():
    parser = argparse.ArgumentParser('Unified LaBraM pre-training script', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--save_ckpt_freq', default=10, type=int)

    # tokenizer settings
    parser.add_argument("--tokenizer_weight", type=str, default='/mnt/cx/EEG_text/Labram_feature/checkpoints/vqnsp.pth')
    parser.add_argument("--tokenizer_model", type=str, default="vqnsp_encoder_base_decoder_3x200x12")
    
    # Model parameters
    parser.add_argument('--model', default='labram_base_patch200_1600_8k_vocab', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--rel_pos_bias', action='store_true')
    parser.add_argument('--disable_rel_pos_bias', action='store_true', dest='rel_pos_bias')
    parser.set_defaults(rel_pos_bias=False)
    parser.add_argument('--abs_pos_emb', action='store_true')
    parser.set_defaults(abs_pos_emb=True)
    parser.add_argument('--layer_scale_init_value', default=0.1, type=float)
    parser.add_argument('--input_size', default=1600, type=int)
    parser.add_argument('--drop_path', type=float, default=0.1)

    # Tokenizer parameters
    parser.add_argument('--codebook_size', default=8192, type=int)
    parser.add_argument('--codebook_dim', default=64, type=int)

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str)
    parser.add_argument('--opt_eps', default=1e-8, type=float)
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+')
    parser.add_argument('--clip_grad', type=float, default=None)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--weight_decay_end', type=float, default=None)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--warmup_lr', type=float, default=1e-6)
    parser.add_argument('--min_lr', type=float, default=1e-5)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--warmup_steps', type=int, default=-1)

    parser.add_argument('--output_dir', default='', help='path where to save')
    parser.add_argument('--log_dir', default=None, help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='')
    parser.add_argument('--auto_resume', action='store_true')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--start_epoch', default=0, type=int)    
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.set_defaults(pin_mem=True)
    
    # Distributed
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int)
    
    # Loss weights
    parser.add_argument('--rec_loss_weight', default=1.0, type=float)
    parser.add_argument('--reg_loss_weight', default=0.0, type=float, help='Set > 0 to enable feature prediction loss')

    parser.add_argument('--hdf5_root', default='/pretrain-clip/hdf5_datasets', type=str,
                        help='Root directory for HDF5 files (used when dataset-specific root is unavailable)')
    parser.add_argument('--hdf5_root_map', default='auto', type=str,
                        help='Dataset->HDF5 root map. Use "auto", a JSON file path, or "NAME=/path,..."')
    parser.add_argument('--csv_path', default=None, type=str, help='Path to single dataset CSV (overrides default multi-dataset)')
    parser.add_argument('--debug_data', action='store_true', help='Print detailed data loading debug info and verify first batch')
    parser.add_argument('--dataset_name', default=None, type=str,
                        help='Process only one dataset (e.g., TUAB, ADHD). Ignored if --csv_path is set.')

    return parser.parse_args()

def get_model(args, num_reg_features=0):
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        use_shared_rel_pos_bias=args.rel_pos_bias,
        use_abs_pos_emb=args.abs_pos_emb,
        init_values=args.layer_scale_init_value,
        vocab_size=args.codebook_size,
        num_regression_features=num_reg_features
    )
    return model

def get_visual_tokenizer(args):
    print(f"Creating visual tokenizer: {args.tokenizer_model}")
    model = create_model(
            args.tokenizer_model,
            pretrained=True,
            pretrained_weight=args.tokenizer_weight,
            as_tokenzer=True,
            n_code=args.codebook_size, 
            code_dim=args.codebook_dim,
        ).eval()
    return model

def main(args):
    utils.init_distributed_mode(args)
    print(args)
    device = torch.device(args.device)

    prep_start = time.time()
    def log(msg):
        if utils.is_main_process():
            elapsed = time.time() - prep_start
            print(f"[prep t+{elapsed:.1f}s] {msg}")

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # 1. Load and Merge Data
    log("Starting data preparation...")
    
    # We only want to load TUAB for this specific pipeline if run_experiment_pipeline.sh is used with TUAB_CSV
    # But run_unified_pretraining.py seems to be designed for multiple datasets.
    # Let's check if we should override DATASETS with a single one if provided via args?
    # No, the script hardcodes DATASETS.
    
    # FOR NOW: Let's assume we only want TUAB if it's the only one available or requested.
    # The user script points HDF5_ROOT to /mnt/dataset2/hdf5_datasets.
    # And TUAB_CSV to /mnt/dataset4/cx/code/EEG_LLM_text/TUAB_fast_elm/all_merged_features_zscored.csv
    
    # Let's Modify DATASETS to include TUAB if it's not there, or just use what we have.
    # Actually, the user script `run_experiment_pipeline.sh` defines TUAB_CSV but `run_unified_pretraining.py` uses hardcoded DATASETS.
    # This is a mismatch. The `run_unified_pretraining.py` was likely from a different context (multi-dataset).
    # We should make `run_unified_pretraining.py` accept a csv_path arg to override.
    
    hdf5_root_map = parse_hdf5_root_map(getattr(args, 'hdf5_root_map', None))
    if hdf5_root_map:
        hdf5_root_map = {_normalize_dataset_key(k): v for k, v in hdf5_root_map.items()}

    if hasattr(args, 'csv_path') and args.csv_path:
        # Single dataset mode (e.g. TUAB)
        log(f"Loading CSV: {args.csv_path}")
        merged_df = pd.read_csv(args.csv_path, low_memory=False)
        if 'sub_id' not in merged_df.columns:
             merged_df['sub_id'] = merged_df['source_file'].apply(lambda x: x.split('.')[0])
        if 'dataset_source' not in merged_df.columns:
            inferred_name = _infer_dataset_name_from_path(args.csv_path)
            merged_df['dataset_source'] = inferred_name
    else:
        log("Merging multiple datasets...")
        allowed = None
        if args.dataset_name:
            allowed = {args.dataset_name}
            log(f"Filtering datasets: {allowed}")
        merged_df = load_and_merge_datasets(DATASETS, allowed_datasets=allowed)

    if utils.is_main_process():
        log(f"Merged samples: {len(merged_df)}")
        if 'dataset_source' in merged_df.columns:
            ds_counts = merged_df['dataset_source'].value_counts().to_dict()
            log(f"Dataset source counts: {ds_counts}")
    
    # 2. Split (Train/Val/Test) - Subject-wise
    # Since we merged 5 datasets, we should split carefully.
    # Simple strategy: 80% subjects train, 10% val, 10% test (global split)
    subjects = merged_df['sub_id'].unique()
    if len(subjects) < 3:
        # Fallback for tiny smoke tests
        train_subs = subjects
        val_subs = subjects
        test_subs = subjects
    else:
        train_subs, temp_subs = train_test_split(subjects, test_size=0.2, random_state=args.seed)
        val_subs, test_subs = train_test_split(temp_subs, test_size=0.5, random_state=args.seed)
    
    train_df = merged_df[merged_df['sub_id'].isin(train_subs)]
    val_df = merged_df[merged_df['sub_id'].isin(val_subs)]
    test_df = merged_df[merged_df['sub_id'].isin(test_subs)]
    
    log("Building datasets...")
    dataset_train = FeaturePredictionDataset(train_df, hdf5_root=args.hdf5_root, window_size=args.input_size, hdf5_root_map=hdf5_root_map)
    dataset_val = FeaturePredictionDataset(val_df, hdf5_root=args.hdf5_root, window_size=args.input_size, hdf5_root_map=hdf5_root_map)
    dataset_test = FeaturePredictionDataset(test_df, hdf5_root=args.hdf5_root, window_size=args.input_size, hdf5_root_map=hdf5_root_map)
    
    if utils.is_main_process():
        log(f"Train samples: {len(dataset_train)} ({len(train_subs)} subs)")
        log(f"Val samples: {len(dataset_val)} ({len(val_subs)} subs)")
        log(f"Test samples: {len(dataset_test)} ({len(test_subs)} subs)")

        missing_train = dataset_train.get_missing_count()
        if missing_train is not None:
            log(f"Missing HDF5 files in train (pre-resolve check): {missing_train}")
    
    num_reg_features = len(dataset_train.get_feature_names())
    if utils.is_main_process():
        log(f"Number of regression features: {num_reg_features}")

    dataset_train_list = [dataset_train]
    train_ch_names_list = [dataset_train.get_ch_names()]
    
    # 3. Model & Tokenizer
    log("Creating model...")
    model = get_model(args, num_reg_features=num_reg_features)
    patch_size = model.patch_size
    args.window_size = (1, args.input_size // patch_size)
    args.patch_size = patch_size

    log("Creating visual tokenizer...")
    vqnsp = get_visual_tokenizer(args).to(device)

    # 4. Sampler
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    sampler_rank = global_rank
    num_training_steps_per_epoch = sum([len(dataset) for dataset in dataset_train_list]) // args.batch_size // num_tasks

    sampler_train_list = []
    for dataset in dataset_train_list:
        if args.distributed:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset, num_replicas=num_tasks, rank=sampler_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset)
        sampler_train_list.append(sampler_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train_list = []
    for dataset, sampler in zip(dataset_train_list, sampler_train_list):
        data_loader_train = torch.utils.data.DataLoader(
            dataset, sampler=sampler,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
            persistent_workers=args.num_workers > 0,
            prefetch_factor=2 if args.num_workers > 0 else None,
        )
        data_loader_train_list.append(data_loader_train)

    if args.debug_data and utils.is_main_process():
        log("Debug data enabled: fetching one batch for verification...")
        debug_loader = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=min(2, args.batch_size),
            shuffle=False,
            num_workers=0,
        )
        batch = next(iter(debug_loader))
        samples_dbg, feats_dbg = batch
        log(f"Debug batch samples shape: {tuple(samples_dbg.shape)}")
        log(f"Debug batch features shape: {tuple(feats_dbg.shape)}")
        log(f"Debug sample[0] file: {dataset_train.get_file_path(0)}")
        log(f"Debug sample[0] segment: {dataset_train.get_segment_path(0)}")

    model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if utils.is_main_process():
        print("Model = %s" % str(model_without_ddp))
        print('number of params:', n_parameters)

    total_batch_size = args.batch_size * utils.get_world_size() * args.gradient_accumulation_steps
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()

    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    
    # Create log file
    if utils.is_main_process():
        log_file_path = os.path.join(args.output_dir, "loss_log.csv") if args.output_dir else "loss_log.csv"
        with open(log_file_path, "w") as f:
            f.write("epoch,loss,loss_rec,reg_loss\n")
        
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            for data_loader_train in data_loader_train_list:
                data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch)

        train_stats = train_one_epoch(
            model, vqnsp, data_loader_train_list,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            ch_names_list=train_ch_names_list,
            args=args,
        )
        
        if utils.is_main_process():
            # Save losses
            epoch_loss = train_stats.get('loss', 0)
            epoch_loss_rec = train_stats.get('loss_rec', 0)
            epoch_reg_loss = train_stats.get('reg_loss', 0)
            
            with open(log_file_path, "a") as f:
                f.write(f"{epoch},{epoch_loss},{epoch_loss_rec},{epoch_reg_loss}\n")
            
            if args.output_dir:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, save_ckpt_freq=args.save_ckpt_freq)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch, 'n_parameters': n_parameters}

            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # Evaluate on held-out test set
    evaluate_regression(model_without_ddp, dataset_test, device, args)

if __name__ == '__main__':
    opts = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts)
