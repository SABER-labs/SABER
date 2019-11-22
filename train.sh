sh sync.sh
rm -rf checkpoints_logs/exp-sp-nonfocal-vocab128-new checkpoints/
# rm -rf checkpoints_logs/exp-sp-nonfocal-vocab128-new
# rm -rf checkpoints_logs/exp-sp-nonfocal-vocab128 checkpoints/
CUDA_VISIBLE_DEVICES="0,1,2" python3.6 train.py