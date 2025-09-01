#!/bin/bash

# Get number of available GPUs
NUM_GPUS=$(nvidia-smi -L | wc -l)

if [ $NUM_GPUS -eq 0 ]; then
    echo "No GPUs detected!"
    exit 1
fi

echo "Detected $NUM_GPUS GPU(s)"

# Define the commands
commands=(
    "python benchmark_point_tracking_mast3r.py --dataset tapvid3d_pstudio --dataset_dir /ocean/projects/cis220039p/mdt2/datasets/dydust3r/tapvid3d_dataset/pstudio --img_width 512 --img_height 288"
    "python benchmark_point_tracking_mast3r.py --dataset tapvid3d_drivetrack --dataset_dir /ocean/projects/cis220039p/mdt2/datasets/dydust3r/tapvid3d_dataset/drivetrack --img_width 512 --img_height 336"
    "python benchmark_point_tracking_mast3r.py --dataset kubric_eval --dataset_dir /ocean/projects/cis220039p/mdt2/datasets/dydust3r/cotracker3_kubric_dataset --img_width 512 --img_height 512"
    "python benchmark_point_tracking_mast3r.py --dataset dynamic_replica_eval --dataset_dir /ocean/projects/cis220039p/mdt2/datasets/dydust3r/dynamic_replica_data --img_width 512 --img_height 288"
)

# Function to run command on specific GPU
run_on_gpu() {
    local gpu_id=$1
    local cmd=$2
    local dataset_name=$(echo "$cmd" | grep -o -- '--dataset [^ ]*' | cut -d' ' -f2)
    
    echo "Starting $dataset_name on GPU $gpu_id"
    CUDA_VISIBLE_DEVICES=$gpu_id $cmd 2>&1 | tee "log_mast3r_${dataset_name}_gpu${gpu_id}.txt"
    echo "Completed $dataset_name on GPU $gpu_id"
}

# Start processes in parallel
pids=()
for i in "${!commands[@]}"; do
    gpu_id=$((i % NUM_GPUS))
    run_on_gpu $gpu_id "${commands[$i]}" &
    pids+=($!)
    echo "Launched job $((i+1))/4 on GPU $gpu_id (PID: ${pids[$i]})"
done

echo "All jobs launched. Waiting for completion..."

# Wait for all processes to complete
for pid in "${pids[@]}"; do
    wait $pid
done

echo "All jobs completed!"
echo "Check log files: log_*_gpu*.txt for individual outputs"