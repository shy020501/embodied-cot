export CUDA_VISIBLE_DEVICES=0
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla.type "prism-qwen25-extra-dinosiglip-224px-0_5b+mx-libero-90" \
  --data_root_dir "/home/seunghyo/embodied-cot/data" \
  --dataset_name libero_90 \
  --run_root_dir "/home/seunghyo/embodied-cot/logs" \
  --adapter_tmp_dir "/home/seunghyo/embodied-cot/logs/lora" \
  --lora_rank 32 \
  --max_steps 200000 \
  --batch_size 8 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug False \
  --wandb_project ecot_qwen_0_5b \
  --wandb_entity AGI_CSI