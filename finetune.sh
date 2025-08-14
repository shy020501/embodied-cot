export CUDA_VISIBLE_DEVICES=0
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path "Stanford-ILIAD/prism-qwen25-extra-dinosiglip-224px-0_5b" \
  --data_root_dir "/home/work/AGI_NIH/data/embodied_features_and_demos_libero" \
  --dataset_name libero_90 \
  --run_root_dir "/home/work/AGI_NIH/checkpoints/ecot_libero_finetune" \
  --adapter_tmp_dir "/home/work/AGI_NIH/checkpoints/ecot_libero_finetune/lora" \
  --lora_rank 32 \
  --max_steps 200000 \
  --batch_size 8 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug False \
  --wandb_project ecot \
  --wandb_entity AGI_CSI