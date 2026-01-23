# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# By Wei-Bang Jiang
# Based on BEiT-v2, timm, DeiT, and DINO code bases
# https://github.com/microsoft/unilm/tree/master/beitv2
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# ---------------------------------------------------------

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

def evaluate_regression(model, dataset_eval, device, args):
    model.eval()
    data_loader = torch.utils.data.DataLoader(
        dataset_eval, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=False
    )
    
    all_preds = []
    all_targets = []
    
    # Get input channels
    ch_names = dataset_eval.get_ch_names()
    input_chans = utils.get_input_chans(ch_names)
    input_chans = torch.tensor(input_chans).to(device)

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
            
            all_preds.append(reg_output.cpu().numpy())
            all_targets.append(features.cpu().numpy())
            
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    feature_names = dataset_eval.get_feature_names()
    results = {}
    for i, name in enumerate(feature_names):
        p = all_preds[:, i]
        t = all_targets[:, i]
        r2 = r2_score(t, p)
        r, _ = pearsonr(t, p)
        results[name] = {'r': r, 'r2': r2}
        print(f"Feature {name}: r={r:.4f}, r2={r2:.4f}")
        
    return results

def get_args():
    parser = argparse.ArgumentParser('LaBraM pre-training script', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--save_ckpt_freq', default=20, type=int)

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
    parser.add_argument('--layer_scale_init_value', default=0.1, type=float, 
                        help="0.1 for base, 1e-5 for large. set 0 to disable layer scale")

    parser.add_argument('--input_size', default=1600, type=int,
                        help='EEG input size for backbone')

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Tokenizer parameters
    parser.add_argument('--codebook_size', default=8192, type=int, help='number of codebook')
    parser.add_argument('--codebook_dim', default=64, type=int, help='number of codebook')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD. 
        (Set the same value with args.weight_decay to keep weight decay no change)""")

    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')    
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)
    
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--gradient_accumulation_steps', default=1, type=int)
    
    # Loss weights
    parser.add_argument('--rec_loss_weight', default=1.0, type=float, help='Weight for reconstruction loss')
    parser.add_argument('--reg_loss_weight', default=15.0, type=float, help='Weight for regression loss')

    parser.add_argument('--csv_path', default='/mnt/cx/EEG_text/data/TUAB.csv', type=str,
                        help='Path to the features CSV file')
    parser.add_argument('--hdf5_root', default='/eeg-h5-files/TUAB', type=str,
                        help='Root directory for HDF5 files')

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

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    # get dataset
    csv_path = '/mnt/dataset4/cx/code/EEG_LLM_text/TUAB_fast_elm/all_merged_features_zscored.csv'
    df = pd.read_csv(csv_path)
    if 'sub_id' not in df.columns:
         df['sub_id'] = df['source_file'].apply(lambda x: x.split('.')[0])
    
    subjects = df['sub_id'].unique()
    # First split: Train (80%) vs Temp (20%)
    train_subs, temp_subs = train_test_split(subjects, test_size=0.2, random_state=args.seed)
    # Second split: Val (10% of total) vs Test (10% of total) -> 50% of Temp
    val_subs, test_subs = train_test_split(temp_subs, test_size=0.5, random_state=args.seed)
    
    train_df = df[df['sub_id'].isin(train_subs)]
    val_df = df[df['sub_id'].isin(val_subs)]
    test_df = df[df['sub_id'].isin(test_subs)]
    
    dataset_train = FeaturePredictionDataset(train_df, hdf5_root='/mnt/dataset2/benchmark_dataloader/hdf5', window_size=args.input_size)
    dataset_val = FeaturePredictionDataset(val_df, hdf5_root='/mnt/dataset2/benchmark_dataloader/hdf5', window_size=args.input_size)
    dataset_test = FeaturePredictionDataset(test_df, hdf5_root='/mnt/dataset2/benchmark_dataloader/hdf5', window_size=args.input_size)
    
    print(f"Train samples: {len(dataset_train)}, Val samples: {len(dataset_val)}, Test samples: {len(dataset_test)}")
    
    num_reg_features = len(dataset_train.get_feature_names())
    print(f"Number of regression features: {num_reg_features}")

    dataset_train_list = [dataset_train]
    train_ch_names_list = [dataset_train.get_ch_names()]
    
    model = get_model(args, num_reg_features=num_reg_features)
    patch_size = model.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (1, args.input_size // patch_size)
    args.patch_size = patch_size

    # prepare visual tokenizer
    vqnsp = get_visual_tokenizer(args).to(device)

    if True:  # args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_rank = global_rank
        num_training_steps_per_epoch = sum([len(dataset) for dataset in dataset_train_list]) // args.batch_size // num_tasks

        sampler_train_list = []
        for dataset in dataset_train_list:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset, num_replicas=num_tasks, rank=sampler_rank, shuffle=True
            )
            sampler_train_list.append(sampler_train)
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

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
        )
        data_loader_train_list.append(data_loader_train)

    model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)

    print("Tokenizer = %s" % str(vqnsp))
    total_batch_size = args.batch_size * utils.get_world_size() * args.gradient_accumulation_steps
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Number of training steps = %d" % num_training_steps_per_epoch)
    print("Number of training examples per epoch = %d" % (total_batch_size * num_training_steps_per_epoch))

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    optimizer = create_optimizer(
        args, model_without_ddp)
    loss_scaler = NativeScaler()

    print("Use step level LR & WD scheduler!")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    
    # Create log file
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

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    if utils.is_main_process():
        # Only main process prints evaluation results, but all processes participate in gather
        pass
    evaluate_regression(model_without_ddp, dataset_test, device, args)


if __name__ == '__main__':
    opts = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts)
