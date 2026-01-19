
import argparse
import os
import json
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import pandas as pd
from pathlib import Path
from timm.models import create_model
from optim_factory import create_optimizer
from engine_for_finetuning import train_one_epoch, evaluate
from utils import NativeScalerWithGradNormCount as NativeScaler
import utils
import modeling_finetune
from sklearn.model_selection import train_test_split
from utils import TUABLoader
import h5py

def prepare_tuab_data(hdf5_root, csv_path):
    print(f"Loading TUAB metadata from {csv_path}...")
    df = pd.read_csv(csv_path)
    if 'sub_id' not in df.columns:
         df['sub_id'] = df['source_file'].apply(lambda x: x.split('.')[0])
    
    # Split by subject
    subjects = df['sub_id'].unique()
    train_subs, temp_subs = train_test_split(subjects, test_size=0.2, random_state=42)
    val_subs, test_subs = train_test_split(temp_subs, test_size=0.5, random_state=42)
    
    # Filter function
    def get_samples(sub_list):
        samples = []
        sub_df = df[df['sub_id'].isin(sub_list)]
        for _, row in sub_df.iterrows():
            # Construct sample tuple (file_path, trial_key, segment_key, label)
            # Need to reverse engineer keys from source_segments string or source_file
            # source_segments example: "['trial0/segment1']"
            try:
                # We need absolute path (robust to TUAB/ prefix and root setting)
                source_file = row['source_file']
                if hdf5_root.rstrip('/').endswith('/TUAB') and source_file.startswith('TUAB/'):
                    source_file = source_file.split('TUAB/', 1)[1]

                candidate_paths = [
                    os.path.join(hdf5_root, source_file),
                    os.path.join(hdf5_root, 'TUAB', source_file),
                ]
                file_path = next((p for p in candidate_paths if os.path.exists(p)), candidate_paths[0])
                # Parse trial/segment
                seg_str = row['source_segments'].replace('[','').replace(']','').replace("'", "").replace('"','')
                if ',' in seg_str: seg_str = seg_str.split(',')[0]
                
                trial_key = seg_str.split('/')[0]
                segment_key = seg_str.split('/')[1]
                label = row['primary_label'] # Or labels[0]
                
                samples.append((file_path, trial_key, segment_key, int(label)))
            except Exception as e:
                pass
        return samples

    train_samples = get_samples(train_subs)
    val_samples = get_samples(val_subs)
    test_samples = get_samples(test_subs)
    
    return TUABLoader(train_samples), TUABLoader(val_samples), TUABLoader(test_samples)

def get_args():
    parser = argparse.ArgumentParser('Eval LaBraM Downstream TUAB', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    
    parser.add_argument('--checkpoint', required=True, type=str)
    parser.add_argument('--mode', required=True, type=str, choices=['linear', 'finetune'])
    parser.add_argument('--output_dir', default='./eval_results', help='Path to save results')
    
    parser.add_argument('--csv_path', default='/mnt/dataset4/cx/code/EEG_LLM_text/TUAB_fast_elm/all_merged_features_zscored.csv', type=str)
    parser.add_argument('--hdf5_root', default='/mnt/dataset2/benchmark_dataloader/hdf5', type=str)
    
    parser.add_argument('--model', default='labram_base_patch200_200', type=str)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=10, type=int)
    
    # Model defaults
    parser.add_argument('--input_size', default=200, type=int) # TUAB patches
    parser.add_argument('--drop', type=float, default=0.0)
    parser.add_argument('--attn_drop_rate', type=float, default=0.0)
    parser.add_argument('--drop_path', type=float, default=0.1)
    parser.add_argument('--init_scale', default=0.001, type=float)
    parser.add_argument('--use_mean_pooling', action='store_true')
    parser.set_defaults(use_mean_pooling=True)
    parser.add_argument('--rel_pos_bias', action='store_true')
    parser.set_defaults(rel_pos_bias=True)
    parser.add_argument('--abs_pos_emb', action='store_true')
    parser.set_defaults(abs_pos_emb=False)
    parser.add_argument('--layer_scale_init_value', default=0.1, type=float)
    parser.add_argument('--qkv_bias', action='store_true')
    parser.set_defaults(qkv_bias=True)
    parser.add_argument('--max_num_electrodes', default=140, type=int)
    
    # Training
    parser.add_argument('--opt', default='adamw')
    parser.add_argument('--opt_eps', default=1e-8)
    parser.add_argument('--opt_betas', default=None, nargs='+')
    parser.add_argument('--clip_grad', default=None)
    parser.add_argument('--momentum', default=0.9)
    parser.add_argument('--weight_decay_end', default=None)
    parser.add_argument('--warmup_lr', default=1e-6)
    parser.add_argument('--min_lr', default=1e-6)
    parser.add_argument('--warmup_epochs', default=5)
    parser.add_argument('--warmup_steps', default=-1)
    parser.add_argument('--smoothing', default=0.1, type=float)

    # Distributed
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://')
    
    return parser.parse_args()

def main():
    args = get_args()
    device = torch.device(args.device)
    
    utils.init_distributed_mode(args) # Handle single GPU gracefully
    
    # Data
    dataset_train, dataset_val, dataset_test = prepare_tuab_data(args.hdf5_root, args.csv_path)
    print(f"Train: {len(dataset_train)}, Val: {len(dataset_val)}, Test: {len(dataset_test)}")
    
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    
    # Model
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=1, # Binary
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        attn_drop_rate=args.attn_drop_rate,
        use_mean_pooling=args.use_mean_pooling,
        init_scale=args.init_scale,
        use_rel_pos_bias=args.rel_pos_bias,
        use_abs_pos_emb=args.abs_pos_emb,
        init_values=args.layer_scale_init_value,
        qkv_bias=args.qkv_bias,
        max_num_electrodes=args.max_num_electrodes,
    )
    
    # Load Pretrained
    print(f"Loading backbone from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    checkpoint_model = checkpoint['model'] if 'model' in checkpoint else checkpoint
    
    # Remove head
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model:
            del checkpoint_model[k]
    
    # Filter keys (e.g. relative_position_index)
    all_keys = list(checkpoint_model.keys())
    for key in all_keys:
        if "relative_position_index" in key:
            checkpoint_model.pop(key)
            
    utils.load_state_dict(model, checkpoint_model, prefix='')
    
    if args.mode == 'linear':
        # Freeze backbone
        for param in model.parameters():
            param.requires_grad = False
        for param in model.head.parameters():
            param.requires_grad = True
    
    model.to(device)
    
    optimizer = create_optimizer(args, model)
    loss_scaler = NativeScaler()
    criterion = torch.nn.BCEWithLogitsLoss()
    
    max_accuracy = 0.0
    
    log_file = os.path.join(args.output_dir, f"log_{args.mode}.txt")
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Start training ({args.mode})...")
    for epoch in range(args.epochs):
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer,
            device, epoch, loss_scaler, args.clip_grad,
            log_writer=None,
            start_steps=epoch * len(data_loader_train),
            lr_schedule_values=None, wd_schedule_values=None,
            num_training_steps_per_epoch=len(data_loader_train), update_freq=1,
            is_binary=True
        )
        
        val_stats = evaluate(data_loader_val, model, device, header='Val:', metrics=["accuracy", "roc_auc"], is_binary=True)
        test_stats = evaluate(data_loader_test, model, device, header='Test:', metrics=["accuracy", "roc_auc"], is_binary=True)
        
        print(f"Epoch {epoch}: Train Loss {train_stats['loss']:.4f}, Val Acc {val_stats['accuracy']:.2f}, Test Acc {test_stats['accuracy']:.2f}")
        
        with open(log_file, "a") as f:
            f.write(json.dumps({
                'epoch': epoch,
                'train': train_stats,
                'val': val_stats,
                'test': test_stats
            }) + "\n")
            
        if val_stats['accuracy'] > max_accuracy:
            max_accuracy = val_stats['accuracy']
            torch.save(model.state_dict(), os.path.join(args.output_dir, f"best_model_{args.mode}.pth"))
            
    print(f"Best Val Accuracy: {max_accuracy:.2f}%")

if __name__ == '__main__':
    main()
