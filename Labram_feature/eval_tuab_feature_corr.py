
import argparse
import torch
import torch.backends.cudnn as cudnn
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from timm.models import create_model
from einops import rearrange
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
import utils
from data_processor.dataset import FeaturePredictionDataset
import modeling_pretrain

def evaluate_feature_correlation(model, dataset, device, args, output_dir):
    model.eval()
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    
    all_preds = []
    all_targets = []
    
    ch_names = dataset.get_ch_names()
    input_chans = utils.get_input_chans(ch_names)
    input_chans = torch.tensor(input_chans).to(device)
    
    print("Evaluating feature correlation...")
    with torch.no_grad():
        for batch in data_loader:
            samples, features = batch
            samples = samples.float().to(device) / 100
            features = features.to(device)
            samples = rearrange(samples, 'B N (A T) -> B N A T', T=200)
            
            # No masking for evaluation
            bool_masked_pos = torch.zeros(
                (samples.shape[0], samples.shape[1] * samples.shape[2]),
                dtype=torch.bool,
                device=device,
            )
            
            outputs = model(samples, input_chans=input_chans, bool_masked_pos=bool_masked_pos)
            reg_output = outputs[2] # Regression output
            
            all_preds.append(reg_output.cpu().numpy())
            all_targets.append(features.cpu().numpy())
            
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    feature_names = dataset.get_feature_names()
    results = []
    
    # Create scatter plots directory
    plot_dir = os.path.join(output_dir, "scatter_plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    print(f"\n{'Feature':<30} | {'R':<8} | {'R2':<8}")
    print("-" * 55)
    
    for i, name in enumerate(feature_names):
        p = all_preds[:, i]
        t = all_targets[:, i]
        r2 = r2_score(t, p)
        r, _ = pearsonr(t, p)
        
        results.append({
            'feature': name,
            'r': r,
            'r2': r2
        })
        print(f"{name:<30} | {r:.4f}   | {r2:.4f}")
        
        # Plot scatter
        plt.figure(figsize=(6, 6))
        plt.scatter(t, p, alpha=0.5, s=10)
        plt.xlabel("True Value")
        plt.ylabel("Predicted Value")
        plt.title(f"{name} (R={r:.2f}, R2={r2:.2f})")
        plt.plot([t.min(), t.max()], [t.min(), t.max()], 'r--')
        plt.savefig(os.path.join(plot_dir, f"{name}.png"))
        plt.close()
        
    # Save results to CSV
    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(output_dir, "feature_correlation_metrics.csv"), index=False)
    print(f"\nResults saved to {output_dir}")

def get_args():
    parser = argparse.ArgumentParser('Eval LaBraM feature correlation', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--checkpoint', required=True, type=str, help='Path to pretrained checkpoint')
    parser.add_argument('--output_dir', default='./eval_results', help='Path to save results')
    parser.add_argument('--csv_path', default='/mnt/dataset4/cx/code/EEG_LLM_text/TUAB_fast_elm/all_merged_features_zscored.csv', type=str)
    parser.add_argument('--hdf5_root', default='/mnt/dataset2/benchmark_dataloader/hdf5', type=str)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--model', default='labram_base_patch200_1600_8k_vocab', type=str)
    parser.add_argument('--num_workers', default=10, type=int)
    
    # Model config
    parser.add_argument('--drop_path', type=float, default=0.1)
    parser.add_argument('--rel_pos_bias', action='store_true')
    parser.set_defaults(rel_pos_bias=False)
    parser.add_argument('--abs_pos_emb', action='store_true')
    parser.set_defaults(abs_pos_emb=True)
    parser.add_argument('--layer_scale_init_value', default=0.1, type=float)
    parser.add_argument('--codebook_size', default=8192, type=int)
    parser.add_argument('--input_size', default=1600, type=int)

    return parser.parse_args()

def main():
    args = get_args()
    device = torch.device(args.device)
    
    # Load Dataset
    print(f"Loading TUAB from {args.csv_path}...")
    df = pd.read_csv(args.csv_path)
    
    # Ensure subject ID
    if 'sub_id' not in df.columns:
         df['sub_id'] = df['source_file'].apply(lambda x: x.split('.')[0])
         
    dataset = FeaturePredictionDataset(df, hdf5_root=args.hdf5_root, window_size=args.input_size)
    num_reg_features = len(dataset.get_feature_names())
    print(f"Samples: {len(dataset)}, Features: {num_reg_features}")
    
    # Load Model
    print(f"Loading model from {args.checkpoint}...")
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
    
    # Load weights
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    
    # Filter keys if needed
    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)
    
    model.to(device)
    
    evaluate_feature_correlation(model, dataset, device, args, args.output_dir)

if __name__ == '__main__':
    main()
