#!/usr/bin/env bash

GPU_IDX=0
NUM_THREADS=32
export OMP_NUM_THREADS=$NUM_THREADS
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES=$GPU_IDX
export OMP_SCHEDULE=STATIC
export OMP_PROC_BIND=CLOSE
export GOMP_CPU_AFFINITY="0-31"
DATASET_PATH='/data1/datasets/ANU_ikea_dataset_smaller/'
CAMERA='dev3'
PT_MODEL='charades'
python train_i3d.py --dataset_path $DATASET_PATH --camera $CAMERA --pretrained_model $PT_MODEL --batch_size 8 --steps_per_update 20