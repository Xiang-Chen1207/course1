
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
            reg_output = outputs[2]  # Regression output

            if reg_output is None:
                print("ERROR: Model does not have regression head (reg_output is None).")
                print("This checkpoint was likely trained without feature prediction loss.")
                return

            all_preds.append(reg_output.cpu().numpy())
            all_targets.append(features.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    feature_names = dataset.get_feature_names()
    results = []

    # Handle dimension mismatch between predictions and targets
    num_pred_features = all_preds.shape[1]
    num_target_features = all_targets.shape[1]

    if num_pred_features != num_target_features:
        print(f"\nWARNING: Dimension mismatch!")
        print(f"  Model predicts {num_pred_features} features")
        print(f"  Dataset has {num_target_features} features")
        num_eval_features = min(num_pred_features, num_target_features)
        print(f"  Evaluating first {num_eval_features} features only.\n")
        all_preds = all_preds[:, :num_eval_features]
        all_targets = all_targets[:, :num_eval_features]
        feature_names = feature_names[:num_eval_features]

    # Create scatter plots directory
    plot_dir = os.path.join(output_dir, "scatter_plots")
    os.makedirs(plot_dir, exist_ok=True)

    print(f"\n{'Feature':<30} | {'R':<8} | {'R2':<8}")
    print("-" * 55)

    for i, name in enumerate(feature_names):
        p = all_preds[:, i]
        t = all_targets[:, i]

        # Handle edge cases
        if np.std(t) == 0 or np.std(p) == 0:
            r, r2 = 0.0, 0.0
            print(f"{name:<30} | {'N/A':<8} | {'N/A':<8} (constant values)")
        else:
            r2 = r2_score(t, p)
            r, _ = pearsonr(t, p)
            print(f"{name:<30} | {r:.4f}   | {r2:.4f}")

        results.append({
            'feature': name,
            'r': r,
            'r2': r2
        })

        # Plot scatter
        plt.figure(figsize=(6, 6))
        plt.scatter(t, p, alpha=0.5, s=10)
        plt.xlabel("True Value")
        plt.ylabel("Predicted Value")
        plt.title(f"{name} (R={r:.2f}, R2={r2:.2f})")
        if t.max() > t.min():
            plt.plot([t.min(), t.max()], [t.min(), t.max()], 'r--')
        plt.savefig(os.path.join(plot_dir, f"{name}.png"))
        plt.close()

    # Summary statistics
    valid_results = [r for r in results if r['r'] != 0.0 or r['r2'] != 0.0]
    if valid_results:
        mean_r = np.mean([r['r'] for r in valid_results])
        mean_r2 = np.mean([r['r2'] for r in valid_results])
        print("-" * 55)
        print(f"{'MEAN':<30} | {mean_r:.4f}   | {mean_r2:.4f}")

    # Save results to CSV
    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(output_dir, "feature_correlation_metrics.csv"), index=False)
    print(f"\nResults saved to {output_dir}")

def get_args():
    parser = argparse.ArgumentParser('Eval LaBraM feature correlation', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--checkpoint', required=True, type=str, help='Path to pretrained checkpoint')
    parser.add_argument('--output_dir', default='./eval_results', help='Path to save results')
    parser.add_argument('--csv_path', default='/mnt/cx/EEG_text/data/TUAB.csv', type=str)
    parser.add_argument('--hdf5_root', default='/eeg-h5-files/TUAB', type=str)
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
    dataset_num_features = len(dataset.get_feature_names())
    print(f"Samples: {len(dataset)}, Dataset Features: {dataset_num_features}")

    # Load checkpoint first to get num_regression_features
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint

    # Determine num_regression_features from checkpoint
    num_reg_features = 0
    if 'student.reg_head.weight' in state_dict:
        saved_num_features = state_dict['student.reg_head.weight'].shape[0]
        num_reg_features = saved_num_features
        print(f"Checkpoint has reg_head with {saved_num_features} features")

        if saved_num_features != dataset_num_features:
            print(f"WARNING: Checkpoint has {saved_num_features} regression features, "
                  f"but current dataset has {dataset_num_features} features!")
            print(f"Using checkpoint's feature dimension ({saved_num_features}) for model creation.")
            print(f"Note: Feature correlation evaluation will only use the first {min(saved_num_features, dataset_num_features)} features.")
    else:
        print("WARNING: Checkpoint does not have reg_head (trained without feature prediction).")
        print("Feature correlation evaluation may not work properly.")
        # Use dataset features if checkpoint doesn't have reg_head
        num_reg_features = dataset_num_features

    # Create Model with correct num_regression_features
    print(f"Creating model with num_regression_features={num_reg_features}...")
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
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"Model load result: {msg}")

    model.to(device)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    evaluate_feature_correlation(model, dataset, device, args, args.output_dir)

if __name__ == '__main__':
    main()
