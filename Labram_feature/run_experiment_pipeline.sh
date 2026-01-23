
#!/bin/bash

# Force unbuffered Python output for real-time training logs
export PYTHONUNBUFFERED=1

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRETRAIN_SCRIPT="$SCRIPT_DIR/run_unified_pretraining.py"
EVAL_FEAT_SCRIPT="$SCRIPT_DIR/eval_tuab_feature_corr.py"
EVAL_DOWN_SCRIPT="$SCRIPT_DIR/eval_tuab_downstream.py"

OUTPUT_ROOT="$SCRIPT_DIR/experiments_unified"

# Pretrain datasets (exclude TUAB)
PRETRAIN_DATASET_LIST=${PRETRAIN_DATASET_LIST:-"ADHD BCIC2A SEEDIV SEEDV SleepEDF Workload_MATB"}

# Downstream dataset (TUAB only)
DOWNSTREAM_DATASET=${DOWNSTREAM_DATASET:-"TUAB"}

# GPU Settings
# Example: "0,1,2,3" or "0"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"1"}
GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l) # Auto-detect count
BATCH_SIZE=32 # Per GPU
LR=5e-5
# Mitigate CUDA memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"expandable_segments:True"}

# Avoid HDF5 file locking issues on shared filesystems
export HDF5_USE_FILE_LOCKING=${HDF5_USE_FILE_LOCKING:-"FALSE"}

# Performance optimizations
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
# Enable cuDNN autotuning for better performance on fixed input sizes
export CUDNN_BENCHMARK=${CUDNN_BENCHMARK:-1}

# Conda environment
CONDA_ENV="labram"

# Debug data loading
DEBUG_DATA=${DEBUG_DATA:-1}

# DataLoader workers (set to 0 to avoid HDF5 multiprocessing hangs)
NUM_WORKERS=${NUM_WORKERS:-0}

# HDF5 file cache size (LRU). Larger = fewer file open/close operations.
# Increase this for slow IO systems (e.g., network storage, HDD)
# Recommended: 32-64 for slow IO, 16 for fast SSD
H5_CACHE_SIZE=${H5_CACHE_SIZE:-32}

# Master ports (auto-pick if empty)
MASTER_PORT_PRETRAIN=${MASTER_PORT_PRETRAIN:-$((29500 + RANDOM % 2000))}
MASTER_PORT_FEATURE=${MASTER_PORT_FEATURE:-$((29500 + RANDOM % 2000))}

mkdir -p $OUTPUT_ROOT

for DATASET_NAME in $PRETRAIN_DATASET_LIST; do
    case "$DATASET_NAME" in
        ADHD)
            HDF5_ROOT="/pretrain-clip/hdf5_datasets/ADHD"
            DATA_CSV="/vePFS-0x0d/home/cx/EEG_text/data/ADHD.csv"
            ;;
        BCIC2A)
            HDF5_ROOT="/pretrain-clip/hdf5_datasets/BCIC2A"
            DATA_CSV="/vePFS-0x0d/home/cx/EEG_text/data/BCIC2A.csv"
            ;;
        SEEDIV)
            HDF5_ROOT="/pretrain-clip/hdf5_datasets/SEEDIV"
            DATA_CSV="/vePFS-0x0d/home/cx/EEG_text/data/SEEDIV.csv"
            ;;
        SEEDV)
            HDF5_ROOT="/pretrain-clip/hdf5_datasets/SEEDV"
            DATA_CSV="/vePFS-0x0d/home/cx/EEG_text/data/SEEDV.csv"
            ;;
        SleepEDF)
            HDF5_ROOT="/pretrain-clip/hdf5_datasets/SleepEDF"
            DATA_CSV="/vePFS-0x0d/home/cx/EEG_text/data/SLEEPEDF.csv"
            ;;
        Workload_MATB)
            HDF5_ROOT="/pretrain-clip/hdf5_datasets/Workload_MATB"
            DATA_CSV="/vePFS-0x0d/home/cx/EEG_text/data/WORKLOAD_MATB.csv"
            ;;
        *)
            echo "Unknown dataset: $DATASET_NAME"
            exit 1
            ;;
    esac

    DATA_OUTPUT_ROOT="$OUTPUT_ROOT/${DATASET_NAME}"
    mkdir -p "$DATA_OUTPUT_ROOT"

    # ==========================================
    # 1. Pretrain: Original LaBraM (No Feature Loss)
    # ==========================================
    echo "Starting Pretraining (Original) for $DATASET_NAME..."
    if [ "$GPUS" -le 1 ]; then
        conda run -n $CONDA_ENV python $PRETRAIN_SCRIPT \
            --output_dir "$DATA_OUTPUT_ROOT/pretrain_original" \
            --rec_loss_weight 1.0 \
            --reg_loss_weight 0.0 \
            --hdf5_root $HDF5_ROOT \
            --csv_path $DATA_CSV \
            --batch_size $BATCH_SIZE \
            --num_workers $NUM_WORKERS \
            --h5_cache_size $H5_CACHE_SIZE \
            --lr $LR \
            --clip_grad 3.0 \
            --epochs 100 \
            --save_ckpt_freq 20 \
            $( [ "$DEBUG_DATA" = "1" ] && echo "--debug_data" )
    else
        conda run -n $CONDA_ENV python -m torch.distributed.run --nproc_per_node=$GPUS --master_port=$MASTER_PORT_PRETRAIN $PRETRAIN_SCRIPT \
        --output_dir "$DATA_OUTPUT_ROOT/pretrain_original" \
        --rec_loss_weight 1.0 \
        --reg_loss_weight 0.0 \
        --hdf5_root $HDF5_ROOT \
        --csv_path $DATA_CSV \
        --batch_size $BATCH_SIZE \
        --num_workers $NUM_WORKERS \
        --h5_cache_size $H5_CACHE_SIZE \
        --lr $LR \
        --clip_grad 3.0 \
        --epochs 100 \
        --save_ckpt_freq 20 \
        $( [ "$DEBUG_DATA" = "1" ] && echo "--debug_data" )
    fi

    # ==========================================
    # 2. Pretrain: LaBraM + Feature Prediction
    # ==========================================
    echo "Starting Pretraining (Feature) for $DATASET_NAME..."
    if [ "$GPUS" -le 1 ]; then
        conda run -n $CONDA_ENV python $PRETRAIN_SCRIPT \
            --output_dir "$DATA_OUTPUT_ROOT/pretrain_feature" \
            --rec_loss_weight 1.0 \
            --reg_loss_weight 10.0 \
            --hdf5_root $HDF5_ROOT \
            --csv_path $DATA_CSV \
            --batch_size $BATCH_SIZE \
            --num_workers $NUM_WORKERS \
            --h5_cache_size $H5_CACHE_SIZE \
            --epochs 100 \
            --save_ckpt_freq 20 \
            $( [ "$DEBUG_DATA" = "1" ] && echo "--debug_data" )
    else
        conda run -n $CONDA_ENV python -m torch.distributed.run --nproc_per_node=$GPUS --master_port=$MASTER_PORT_FEATURE $PRETRAIN_SCRIPT \
        --output_dir "$DATA_OUTPUT_ROOT/pretrain_feature" \
        --rec_loss_weight 1.0 \
        --reg_loss_weight 10.0 \
        --hdf5_root $HDF5_ROOT \
        --csv_path $DATA_CSV \
        --batch_size $BATCH_SIZE \
        --num_workers $NUM_WORKERS \
        --h5_cache_size $H5_CACHE_SIZE \
        --epochs 100 \
        --save_ckpt_freq 20 \
        $( [ "$DEBUG_DATA" = "1" ] && echo "--debug_data" )
    fi

done

# ==========================================
# 3. Downstream on TUAB (using pretrained checkpoints)
# ==========================================

case "$DOWNSTREAM_DATASET" in
    TUAB)
        DOWN_HDF5_ROOT="/eeg-h5-files/TUAB"
        DOWN_CSV="/vePFS-0x0d/home/cx/EEG_text/data/TUAB.csv"
        ;;
    *)
        echo "Unsupported downstream dataset: $DOWNSTREAM_DATASET"
        exit 1
        ;;
esac

for DATASET_NAME in $PRETRAIN_DATASET_LIST; do
    DATA_OUTPUT_ROOT="$OUTPUT_ROOT/${DATASET_NAME}"
    for MODE in "original" "feature"; do
        CKPT="$DATA_OUTPUT_ROOT/pretrain_${MODE}/checkpoint-best.pth"
        if [ ! -f "$CKPT" ]; then CKPT="$DATA_OUTPUT_ROOT/pretrain_${MODE}/checkpoint.pth"; fi

        echo "Evaluating on $DOWNSTREAM_DATASET with pretrain $DATASET_NAME ($MODE) using $CKPT"

        # A. Feature Correlation on TUAB
        echo "Running Feature Correlation..."
        conda run -n $CONDA_ENV python $EVAL_FEAT_SCRIPT \
            --checkpoint $CKPT \
            --output_dir "$DATA_OUTPUT_ROOT/eval_${MODE}/feature_corr_${DOWNSTREAM_DATASET}" \
            --csv_path $DOWN_CSV \
            --hdf5_root "$DOWN_HDF5_ROOT" \
            --batch_size 128 \
            --num_workers $NUM_WORKERS

        # B. Downstream Classification (Linear Probing)
        echo "Running Linear Probing..."
        conda run -n $CONDA_ENV python $EVAL_DOWN_SCRIPT \
            --checkpoint $CKPT \
            --mode linear \
            --output_dir "$DATA_OUTPUT_ROOT/eval_${MODE}/downstream_linear_${DOWNSTREAM_DATASET}" \
            --csv_path $DOWN_CSV \
            --hdf5_root "$DOWN_HDF5_ROOT" \
            --epochs 20 \
            --lr 1e-3 \
            --num_workers $NUM_WORKERS

        # C. Downstream Classification (Full Finetuning)
        echo "Running Full Fine-tuning..."
        conda run -n $CONDA_ENV python $EVAL_DOWN_SCRIPT \
            --checkpoint $CKPT \
            --mode finetune \
            --output_dir "$DATA_OUTPUT_ROOT/eval_${MODE}/downstream_finetune_${DOWNSTREAM_DATASET}" \
            --csv_path $DOWN_CSV \
            --hdf5_root "$DOWN_HDF5_ROOT" \
            --epochs 30 \
            --lr 5e-4 \
            --num_workers $NUM_WORKERS
    done
done

echo "Pipeline Completed!"
