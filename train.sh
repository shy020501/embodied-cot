# export CUDA_VISIBLE_DEVICES=0,1
torchrun --standalone --nnodes 1 --nproc-per-node 2 vla-scripts/train.py \
  --vla.type "prism-qwen25-extra-dinosiglip-224px-0_5b+mx-libero-90" \
  --data_root_dir "/home/seunghyo/embodied-cot/data" \
  --run_root_dir "/home/seunghyo/embodied-cot/logs" \
  --wandb_project ecot_qwen_0_5b \
  --wandb_entity AGI_CSI \