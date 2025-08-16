### Denoiser
python -m resemble_enhance.denoiser.train --yaml config/denoiser.yaml runs/denoiser


### Enhancer
# Stage 1
python -m resemble_enhance.enhancer.train --yaml config/enhancer_stage1.yaml runs/enhancer_stage1
# Stage 2
python -m resemble_enhance.enhancer.train --yaml config/enhancer_stage2.yaml runs/enhancer_stage2