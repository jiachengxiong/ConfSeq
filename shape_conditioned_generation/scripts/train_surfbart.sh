export CUDA_VISIBLE_DEVICES=5,6,7
taskset -c 0-40 accelerate launch --main_process_port 29501 --mixed_precision bf16 src/train/train_surfbartv2_merge.py