#!/bin/bash
# Task 2(a): launch distributed RTE fine-tuning with gather/scatter gradient sync.
# Total batch size = per_device_train_batch_size * WORLD_SIZE (match Task 1: 64).
#
# Usage:
#   1. Edit MASTER_IP, NODES, GLUE_DIR, PYTHON as needed.
#   2. chmod +x task2a.sh && ./task2a.sh
# Or run manually on each node (tmux), e.g. on rank-0 machine:
#   python3 run_glue.py ... --local_rank 0 --master_ip $MASTER_IP --master_port $PORT --world_size 4

set -e

MASTER_IP="10.10.1.2"   # Rank-0 node's 10.10.1.* address (see README / cloudlab.md)
MASTER_PORT="12345"
WORLD_SIZE=4
# Per-worker batch size: 16 * 4 workers = 64 global (same as Task 1)
PER_DEVICE_BATCH=16
for TASK in task2a task2b task3; 
do
  GLUE_DIR="../glue_data/"
  TASK_NAME=RTE
  PYTHON="${PYTHON:-python3}"
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  RUN_GLUE="${SCRIPT_DIR}/run_glue_${TASK}.py"
  OUTPUT_DIR="../output_cache/${TASK}"
  mkdir -p ${OUTPUT_DIR}

  # One entry per rank: SSH target for that rank (same order as rank 0..3)
  NODES=("10.10.1.2" "10.10.1.3" "10.10.1.4" "10.10.1.1")
  USER="${USER:-$(whoami)}"

  COMMON_ARGS="\
    --model_type bert \
    --model_name_or_path bert-base-cased \
    --task_name ${TASK_NAME} \
    --do_train \
    --do_eval \
    --data_dir ${GLUE_DIR}/${TASK_NAME} \
    --max_seq_length 128 \
    --per_device_train_batch_size ${PER_DEVICE_BATCH} \
    --per_device_eval_batch_size 8 \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --output_dir ${OUTPUT_DIR} \
    --overwrite_output_dir \
    --master_ip ${MASTER_IP} \
    --master_port ${MASTER_PORT} \
    --world_size ${WORLD_SIZE}"

  # Use unbuffered Python so "[rank N] process started" prints immediately
  PYTHON="${PYTHON:-python3 -u}"

  for RANK in $(seq 0 $((WORLD_SIZE - 1))); do
    NODE="${NODES[$RANK]}"
    echo "Launching rank ${RANK} on ${NODE}..."
    ssh -n -o "StrictHostKeyChecking=no" -o "ConnectTimeout=10" "${USER}@${NODE}" \
      "cd ${SCRIPT_DIR} && ${PYTHON} ${RUN_GLUE} ${COMMON_ARGS} --local_rank ${RANK}" &
  done

  wait
  echo "Task ${TASK} jobs finished."
  python ../${TASK}/plot_loss_time.py --dir ${OUTPUT_DIR}
done