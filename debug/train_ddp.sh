sh sync.sh
rm -rf checkpoints_logs/exp-checkpoints_saber_128 checkpoints_saber_128
CUDA_VISIBLE_DEVICES="0,1,2" python3.6 -m torch.distributed.launch --nproc_per_node=3 train_ddp.py