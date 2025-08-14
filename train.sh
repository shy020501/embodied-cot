export CUDA_VISIBLE_DEVICES=0
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/train.py \
  --vla.type "prism-qwen25-dinosiglip-224px+0_5b+mx-libero-90" \
  --data_root_dir "/home/work/AGI_NIH/data/embodied_features_and_demos_libero" \
  --run_root_dir "/home/work/AGI_NIH/checkpoints/ecot_libero_mini" \
  --wandb_project ecot \
  --wandb_entity AGI_CSI
