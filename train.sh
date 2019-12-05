sh sync.sh
# rm -rf checkpoints_logs/exp-sp-nonfocal-vocab128-new checkpoints/
# rm -rf checkpoints_logs/exp-sp-ctc-vocab128-radam checkpoints_5x3_radam/
rm -rf checkpoints_logs/exp-sp-ctc-vocab128-radam/
CUDA_VISIBLE_DEVICES="0,1,2" python3.6 train.py