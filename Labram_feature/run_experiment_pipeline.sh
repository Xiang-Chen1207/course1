
#!/bin/bash

# Configuration
PRETRAIN_SCRIPT="run_unified_pretraining.py"
EVAL_FEAT_SCRIPT="eval_tuab_feature_corr.py"
EVAL_DOWN_SCRIPT="eval_tuab_downstream.py"

OUTPUT_ROOT="./experiments_unified"
# TUAB HDF5 root (CSV里的source_file不带TUAB/前缀)
HDF5_ROOT="/mnt/dataset2/benchmark_dataloader/hdf5/TUAB"
TUAB_CSV="/mnt/dataset4/cx/code/EEG_LLM_text/TUAB_fast_elm/all_merged_features_zscored.csv"

# GPU Settings
# Example: "0,1,2,3" or "0"
export CUDA_VISIBLE_DEVICES="1" 
GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l) # Auto-detect count
BATCH_SIZE=32 # Per GPU

mkdir -p $OUTPUT_ROOT

# ==========================================
# 1. Pretrain: Original LaBraM (No Feature Loss)
# ==========================================
echo "Starting Pretraining (Original)..."
conda run -n labram_py310 python -m torch.distributed.run --nproc_per_node=$GPUS --master_port=29505 $PRETRAIN_SCRIPT \
    --output_dir "$OUTPUT_ROOT/pretrain_original" \
    --rec_loss_weight 1.0 \
    --reg_loss_weight 0.0 \
    --hdf5_root $HDF5_ROOT \
    --csv_path $TUAB_CSV \
    --batch_size $BATCH_SIZE \
    --epochs 100 \
    --save_ckpt_freq 20

# ==========================================
# 2. Pretrain: LaBraM + Feature Prediction
# ==========================================
echo "Starting Pretraining (Feature)..."
conda run -n labram_py310 python -m torch.distributed.run --nproc_per_node=$GPUS --master_port=29612 $PRETRAIN_SCRIPT \
    --output_dir "$OUTPUT_ROOT/pretrain_feature" \
    --rec_loss_weight 1.0 \
    --reg_loss_weight 10.0 \
    --hdf5_root $HDF5_ROOT \
    --csv_path $TUAB_CSV \
    --batch_size $BATCH_SIZE \
    --epochs 100 \
    --save_ckpt_freq 20

# ==========================================
# 3. Evaluation Loop
# ==========================================

for MODE in "original" "feature"; do
    CKPT="$OUTPUT_ROOT/pretrain_${MODE}/checkpoint-best.pth"
    # Fallback to last if best not found
    if [ ! -f "$CKPT" ]; then CKPT="$OUTPUT_ROOT/pretrain_${MODE}/checkpoint.pth"; fi
    
    echo "Evaluating Mode: $MODE using $CKPT"
    
    # A. Feature Correlation on TUAB
    echo "Running Feature Correlation..."
    conda run -n labram_py310 python $EVAL_FEAT_SCRIPT \
        --checkpoint $CKPT \
        --output_dir "$OUTPUT_ROOT/eval_${MODE}/feature_corr" \
        --csv_path $TUAB_CSV \
        --hdf5_root "$HDF5_ROOT" \
        --batch_size 128
        
    # B. Downstream Classification (Linear Probing)
    echo "Running Linear Probing..."
    conda run -n labram_py310 python $EVAL_DOWN_SCRIPT \
        --checkpoint $CKPT \
        --mode linear \
        --output_dir "$OUTPUT_ROOT/eval_${MODE}/downstream_linear" \
        --csv_path $TUAB_CSV \
        --hdf5_root "$HDF5_ROOT" \
        --epochs 20 \
        --lr 1e-3
        
    # C. Downstream Classification (Full Finetuning)
    echo "Running Full Fine-tuning..."
    conda run -n labram_py310 python $EVAL_DOWN_SCRIPT \
        --checkpoint $CKPT \
        --mode finetune \
        --output_dir "$OUTPUT_ROOT/eval_${MODE}/downstream_finetune" \
        --csv_path $TUAB_CSV \
        --hdf5_root "$HDF5_ROOT" \
        --epochs 30 \
        --lr 5e-4
done

echo "Pipeline Completed!"
