#!/usr/bin/env python3
"""
Analyze the RIR test results to understand the differences between files.
"""

import numpy as np
import librosa
import matplotlib.pyplot as plt
from pathlib import Path


def analyze_audio_file(file_path):
    """Analyze an audio file and return statistics"""
    try:
        audio, sr = librosa.load(file_path, sr=44100)
        
        stats = {
            'file': file_path.name,
            'duration': len(audio) / sr,
            'rms': np.sqrt(np.mean(audio**2)),
            'peak': np.max(np.abs(audio)),
            'dynamic_range': np.max(audio) - np.min(audio),
            'zero_crossings': librosa.zero_crossings(audio).sum(),
            'spectral_centroid': np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)),
            'spectral_bandwidth': np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr)),
        }
        
        return stats, audio, sr
    except Exception as e:
        print(f"Error analyzing {file_path}: {e}")
        return None, None, None


def main():
    output_dir = Path('test/output')
    
    if not output_dir.exists():
        print("❌ Output directory not found. Run test_rir_pipeline.py first!")
        return
    
    print("🔍 ANALYZING RIR TEST RESULTS")
    print("=" * 50)
    
    # Find all wav files
    wav_files = sorted(output_dir.glob("*.wav"))
    
    if not wav_files:
        print("❌ No audio files found in output directory!")
        return
    
    print(f"Found {len(wav_files)} audio files:")
    for f in wav_files:
        print(f"   - {f.name}")
    
    print("\n📊 AUDIO STATISTICS:")
    print("-" * 80)
    print(f"{'File':<35} {'RMS':<8} {'Peak':<8} {'SpectCent':<10} {'SpectBW':<10}")
    print("-" * 80)
    
    original_audio = None
    original_stats = None
    
    for wav_file in wav_files:
        stats, audio, sr = analyze_audio_file(wav_file)
        
        if stats is None:
            continue
            
        if 'original' in wav_file.name:
            original_audio = audio
            original_stats = stats
        
        print(f"{stats['file']:<35} {stats['rms']:<8.4f} {stats['peak']:<8.4f} "
              f"{stats['spectral_centroid']:<10.1f} {stats['spectral_bandwidth']:<10.1f}")
    
    # Compare with original
    if original_audio is not None:
        print(f"\n🔄 COMPARISON WITH ORIGINAL:")
        print("-" * 50)
        
        for wav_file in wav_files:
            if 'original' in wav_file.name:
                continue
                
            stats, audio, sr = analyze_audio_file(wav_file)
            if stats is None or audio is None:
                continue
            
            # Calculate MSE and correlation
            mse = np.mean((original_audio - audio) ** 2)
            correlation = np.corrcoef(original_audio, audio)[0, 1]
            
            print(f"{stats['file']:<35} MSE: {mse:<8.6f} Corr: {correlation:<8.4f}")
    
    print(f"\n🎧 LISTENING RECOMMENDATIONS:")
    print("-" * 40)
    print("1. Listen to 01_original.wav first (clean reference)")
    print("2. Compare with 02_rir_only.wav (should hear room acoustics)")
    print("3. Listen to 03_validation_distorted.wav (RIR + reverb)")
    print("4. Compare with 05_exact_validation_target.wav (exact validation match)")
    print("\nIf RIR is working correctly, you should hear:")
    print("- Spatial reverberation/echo effects")
    print("- Changes in perceived room size")
    print("- Different acoustic characteristics")
    
    # Create a simple visualization
    try:
        plt.figure(figsize=(15, 10))
        
        files_to_plot = [f for f in wav_files if any(x in f.name for x in ['original', 'rir_only', 'validation_distorted', 'exact_validation'])]
        
        for i, wav_file in enumerate(files_to_plot[:4]):  # Plot up to 4 files
            stats, audio, sr = analyze_audio_file(wav_file)
            if audio is None:
                continue
                
            plt.subplot(2, 2, i+1)
            
            # Plot waveform
            time = np.linspace(0, len(audio)/sr, len(audio))
            plt.plot(time, audio, alpha=0.7)
            plt.title(f"{wav_file.stem}")
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            plt.grid(True, alpha=0.3)
            
            # Show first 2 seconds only for clarity
            plt.xlim(0, min(2, len(audio)/sr))
        
        plt.tight_layout()
        plot_file = output_dir / "audio_comparison.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"\n📈 Saved visualization: {plot_file}")
        
    except Exception as e:
        print(f"⚠️  Could not create visualization: {e}")


if __name__ == "__main__":
    main()
