export MUJOCO_GL="egl"

python experiments/robot/libero/run_libero_eval.py \
    --model_family llava \
    --task_suite_name libero_90 \
    --center_crop False \
    --use_wandb True \
    --wandb_project ecot_libero \
    --wandb_entity AGI_CSI