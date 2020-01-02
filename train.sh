sh sync.sh
rm -rf checkpoints_logs/exp-checkpoints_saber_128 checkpoints_saber_128
CUDA_VISIBLE_DEVICES="1,2" python3.6 train.py