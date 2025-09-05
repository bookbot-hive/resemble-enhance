#!/usr/bin/env python3
"""
Test script to verify RIR application in the denoiser training pipeline.
This script exactly mimics how audio is loaded and RIR is applied during training.
"""

import sys
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from resemble_enhance.data.dataset import Dataset
from resemble_enhance.data.distorter.distorter import Distorter
from resemble_enhance.data.distorter.custom import RandomRIR
from resemble_enhance.hparams import HParams
from resemble_enhance.data.utils import rglob_audio_files


def load_audio_like_dataset(audio_path, target_length=None, wav_rate=44100):
    """Load audio exactly like the Dataset class does"""
    print(f"Loading audio: {audio_path}")
    
    # Load audio using librosa (same as Dataset._load_wav)
    wav, sr = librosa.load(audio_path, sr=wav_rate, mono=True)
    
    if target_length is not None and len(wav) > target_length:
        # Random crop like in training
        start = np.random.randint(0, len(wav) - target_length + 1)
        wav = wav[start:start + target_length]
    
    print(f"Loaded audio shape: {wav.shape}, sample rate: {sr}")
    print(f"Audio range: [{wav.min():.6f}, {wav.max():.6f}]")
    
    return wav


def test_rir_pipeline():
    """Test the exact RIR pipeline used in denoiser training"""
    
    print("=" * 60)
    print("TESTING RIR PIPELINE - DENOISER TRAINING")
    print("=" * 60)
    
    # 1. Load hyperparameters (like in training)
    try:
        hp = HParams.load(Path('runs/run_3/denoiser'))
        print(f"✅ Loaded hyperparameters")
        print(f"   - RIR dir: {hp.rir_dir}")
        print(f"   - Wav rate: {hp.wav_rate}")
    except Exception as e:
        print(f"❌ Error loading hyperparameters: {e}")
        print("Using default hyperparameters...")
        # Create minimal HP object
        class MockHP:
            rir_dir = Path('data/rir')
            wav_rate = 44100
        hp = MockHP()
    
    # 2. Find some test audio files
    print(f"\n📁 Looking for audio files...")
    try:
        fg_files = rglob_audio_files(Path('data/fg'))
        if len(fg_files) == 0:
            print("❌ No foreground audio files found in data/fg")
            return
        
        test_file = fg_files[0]  # Use first file
        print(f"✅ Using test file: {test_file}")
    except Exception as e:
        print(f"❌ Error finding audio files: {e}")
        return
    
    # 3. Load audio exactly like Dataset does
    try:
        # Load 3 seconds of audio (like training)
        target_length = int(3.0 * hp.wav_rate)  # 3 seconds
        original_audio = load_audio_like_dataset(test_file, target_length, hp.wav_rate)
    except Exception as e:
        print(f"❌ Error loading audio: {e}")
        return
    
    # 4. Create distorters exactly like in training
    print(f"\n🔧 Creating distorters...")
    
    # Training distorter (with all effects)
    try:
        train_distorter = Distorter(hp, training=True, mode="denoiser")
        print(f"✅ Training distorter created")
        print(f"   Effects: {[type(effect).__name__ for effect in train_distorter.effects]}")
    except Exception as e:
        print(f"❌ Error creating training distorter: {e}")
        train_distorter = None
    
    # Validation distorter (RIR + Reverb only)
    try:
        val_distorter = Distorter(hp, training=False, mode="denoiser")
        print(f"✅ Validation distorter created")
        print(f"   Effects: {[type(effect).__name__ for effect in val_distorter.effects]}")
    except Exception as e:
        print(f"❌ Error creating validation distorter: {e}")
        val_distorter = None
    
    # Pure RIR only (for comparison)
    try:
        rir_only = RandomRIR(hp.rir_dir, deterministic=True)
        print(f"✅ Pure RIR effect created")
        print(f"   RIR files found: {len(rir_only.rir_paths)}")
    except Exception as e:
        print(f"❌ Error creating RIR effect: {e}")
        rir_only = None
    
    # 5. Apply distortions and save results
    print(f"\n🎵 Processing audio...")
    
    output_dir = Path('test/output')
    output_dir.mkdir(exist_ok=True)
    
    # Save original
    original_file = output_dir / "01_original.wav"
    sf.write(original_file, original_audio, hp.wav_rate)
    print(f"💾 Saved: {original_file}")
    
    # Apply RIR only
    if rir_only:
        try:
            rir_audio = rir_only.apply(original_audio.copy(), hp.wav_rate)
            rir_file = output_dir / "02_rir_only.wav"
            sf.write(rir_file, rir_audio, hp.wav_rate)
            print(f"💾 Saved: {rir_file}")
            
            # Calculate difference
            mse = np.mean((original_audio - rir_audio) ** 2)
            print(f"   RIR MSE: {mse:.6f}")
            
        except Exception as e:
            print(f"❌ Error applying RIR: {e}")
    
    # Apply validation distorter (RIR + Reverb)
    if val_distorter:
        try:
            val_audio = val_distorter(original_audio.copy(), hp.wav_rate)
            val_file = output_dir / "03_validation_distorted.wav"
            sf.write(val_file, val_audio, hp.wav_rate)
            print(f"💾 Saved: {val_file}")
            
            # Calculate difference
            mse = np.mean((original_audio - val_audio) ** 2)
            print(f"   Validation MSE: {mse:.6f}")
            
        except Exception as e:
            print(f"❌ Error applying validation distorter: {e}")
    
    # Apply training distorter (full effect chain)
    if train_distorter:
        try:
            train_audio = train_distorter(original_audio.copy(), hp.wav_rate)
            train_file = output_dir / "04_training_distorted.wav"
            sf.write(train_file, train_audio, hp.wav_rate)
            print(f"💾 Saved: {train_file}")
            
            # Calculate difference  
            mse = np.mean((original_audio - train_audio) ** 2)
            print(f"   Training MSE: {mse:.6f}")
            
        except Exception as e:
            print(f"❌ Error applying training distorter: {e}")
    
    # 6. Mimic exact validation pipeline
    print(f"\n🔍 Mimicking exact validation pipeline...")
    
    try:
        # This is exactly what happens in validation:
        # 1. Load audio
        # 2. Apply distorter (creates fg_dwav)
        # 3. Normalize (like _normalize function)
        
        def _normalize(wav):
            """Exact normalization from dataset"""
            return wav / np.abs(wav).max() if np.abs(wav).max() > 0 else wav
        
        # Apply validation distorter and normalize (exact pipeline)
        fg_dwav = _normalize(val_distorter(original_audio.copy(), hp.wav_rate)).astype(np.float32)
        
        # Save this - this is exactly what gets saved as "_target.wav" in validation
        exact_file = output_dir / "05_exact_validation_target.wav"
        sf.write(exact_file, fg_dwav, hp.wav_rate)
        print(f"💾 Saved: {exact_file} (this matches validation '_target.wav')")
        
    except Exception as e:
        print(f"❌ Error in exact validation pipeline: {e}")
    
    print(f"\n✅ Test completed!")
    print(f"📂 Check output files in: {output_dir.absolute()}")
    print(f"\nFiles to listen to:")
    print(f"   01_original.wav           - Clean original audio")
    print(f"   02_rir_only.wav          - Only RIR applied")
    print(f"   03_validation_distorted.wav - RIR + Reverb (validation)")
    print(f"   04_training_distorted.wav   - Full training effects")
    print(f"   05_exact_validation_target.wav - Exact match to validation '_target.wav'")


if __name__ == "__main__":
    test_rir_pipeline()
