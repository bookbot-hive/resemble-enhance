# Resemble Enhance Architecture Guide

## Overview

Resemble Enhance is a two-stage AI speech enhancement system that performs denoising and audio quality improvement. It uses advanced deep learning techniques including Continuous Flow Matching (CFM) and autoencoder architectures to restore high-quality speech from degraded audio.

## System Architecture

```
Input Audio → Denoiser → Enhancer → Enhanced Output
              (Stage 1)   (Stage 2)   (44.1kHz)
```

### Two-Stage Pipeline

1. **Denoiser**: Separates speech from background noise
2. **Enhancer**: Restores audio quality and extends bandwidth to 44.1kHz

## Core Components

### 1. Data Processing Pipeline

#### Dataset Structure
```
data/
├── fg/          # Foreground (clean speech)
├── bg/          # Background (noise)  
└── rir/         # Room Impulse Responses (.npy files)
```

#### Audio Distortion System
The training pipeline applies realistic audio degradations:

- **RIR (Room Impulse Response)**: Simulates acoustic environments
- **Reverb**: Adds room reverberation effects  
- **Gaussian Noise**: Simulates electronic noise
- **Overdrive**: Simulates amplifier distortion
- **EQ/Filtering**: Simulates frequency response issues
- **Background Mixing**: Combines clean speech with noise

**Training Mode**: Random distortions (80% distorted, 20% clean)
**Validation Mode**: Deterministic distortions for reproducibility

### 2. Denoiser Architecture

The denoiser uses a U-Net style architecture to separate speech from noise:
- Operates on mel-spectrograms
- Trained to predict clean speech from noisy input
- Can be used standalone for noise reduction

### 3. Enhancer Architecture

The enhancer consists of three main components:

#### A. Latent Conditional Flow Matching (LCFM)
- **Purpose**: Generates high-quality audio representations
- **Components**:
  - **IRMAE**: Invertible ResNet-style autoencoder for mel-spectrogram processing
  - **CFM**: Continuous Flow Matching model for latent space generation
  - **Solver**: ODE solver for sampling from the flow model

#### B. UnivNet Vocoder
- Converts enhanced mel-spectrograms to waveform
- High-quality neural vocoder optimized for 44.1kHz output
- Includes alias-free components and multi-resolution STFT discriminator

#### C. Training Modes
- **AE Mode**: Autoencoder training (Stage 1)
- **CFM Mode**: Flow matching training (Stage 2)

## Training Process

### Stage 1: Autoencoder + Vocoder Training
```bash
python -m resemble_enhance.enhancer.train --yaml config/enhancer_stage1.yaml runs/enhancer_stage1
```
- Trains the IRMAE autoencoder
- Trains the UnivNet vocoder
- Establishes mel-to-waveform conversion capability

### Stage 2: Flow Matching Training  
```bash
python -m resemble_enhance.enhancer.train --yaml config/enhancer_stage2.yaml runs/enhancer_stage2
```
- Trains the CFM (Continuous Flow Matching) model
- Uses pretrained autoencoder/vocoder from Stage 1
- Learns to generate enhanced audio in latent space

### Optional: Denoiser Pretraining
```bash
python -m resemble_enhance.denoiser.train --yaml config/denoiser.yaml runs/denoiser
```
- Pretrains the denoiser component
- Can be integrated into the enhancer training

## Key Technical Details

### Continuous Flow Matching (CFM)
- Modern generative modeling technique (successor to diffusion models)
- More efficient than diffusion, requiring fewer sampling steps
- Operates in the latent space of the autoencoder
- Uses ODE solvers (midpoint method) for inference

### Audio Processing
- **Sample Rate**: 44.1kHz output
- **Window Size**: Configurable mel-spectrogram parameters
- **Latent Dimensions**: Configurable based on model complexity
- **Training Duration**: 3-second audio clips (configurable)

### Loss Functions
- **Reconstruction Loss**: L1/L2 loss on mel-spectrograms
- **Adversarial Loss**: GAN-style discriminator training
- **Flow Matching Loss**: Continuous normalizing flow objective

## Configuration

Key configuration files:
- `config/denoiser.yaml`: Denoiser hyperparameters
- `config/enhancer_stage1.yaml`: Autoencoder training config  
- `config/enhancer_stage2.yaml`: Flow matching training config

Important hyperparameters:
- `lcfm_training_mode`: "ae" for Stage 1, "cfm" for Stage 2
- `batch_size_per_gpu`: Batch size per GPU
- `training_seconds`: Audio clip length
- `lcfm_z_scale`: Latent space scaling factor

## Inference Pipeline

1. **Load Models**: Denoiser + Enhancer
2. **Denoise**: Remove background noise
3. **Enhance**: 
   - Convert to mel-spectrogram
   - Encode to latent space (IRMAE)
   - Generate enhanced latents (CFM)
   - Decode to mel-spectrogram (IRMAE)  
   - Convert to waveform (UnivNet)

## Usage

### Training
```bash
# Stage 1: Autoencoder + Vocoder
python -m resemble_enhance.enhancer.train \
    --yaml config/enhancer_stage1.yaml \
    runs/enhancer_stage1

# Stage 2: Flow Matching  
python -m resemble_enhance.enhancer.train \
    --yaml config/enhancer_stage2.yaml \
    runs/enhancer_stage2
```

### Inference
```bash
# Full enhancement (denoise + enhance)
resemble-enhance input_dir output_dir

# Denoise only
resemble-enhance input_dir output_dir --denoise_only
```

This architecture enables Resemble Enhance to achieve high-quality speech enhancement by combining the power of modern generative models (CFM) with robust audio processing pipelines and realistic training data augmentation.