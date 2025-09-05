
export CUDA_VISIBLE_DEVICES=1
export TORCH_CUDA_ARCH_LIST="8.9"
export DS_BUILD_OPS=0
export DS_BUILD_FUSED_ADAM=0
export DS_BUILD_CPU_ADAM=1

### Denoiser ###
python -m resemble_enhance.denoiser.train \
    --yaml config/denoiser.yaml \
    --wandb-project "resemble-enhance" \
    --wandb-name "denoiser" \
    runs/run_3/denoiser

### Enhancer ###
## Stage 1 ##
# python -m resemble_enhance.enhancer.train \
#     --yaml config/enhancer_stage1.yaml \
#     --wandb-project "resemble-enhance" \
#     --wandb-name "enhancer-stage1" \
#     runs/run_2/enhancer_stage1
## Stage 2 ##
# python -m resemble_enhance.enhancer.train \
#     --yaml config/enhancer_stage2.yaml \
#     --wandb-project "resemble-enhance" \
#     --wandb-name "enhancer-stage2" \
#     runs/run_2/enhancer_stage2
