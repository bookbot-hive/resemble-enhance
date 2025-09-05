#!/usr/bin/env python3
"""
Test script to verify that the RandomRIR windowed truncation fix prevents looping artifacts.
"""

import sys
from pathlib import Path
import tempfile
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from resemble_enhance.data.distorter.custom import RandomRIR
from resemble_enhance.data.distorter.sox import RandomReverb


def create_test_audio_and_rir():
    """Create test audio and RIR files"""
    temp_dir = Path(tempfile.mkdtemp())
    
    rir_dir = temp_dir / "rir"
    rir_dir.mkdir()
    
    # Create test audio (2 seconds at 16kHz)
    sample_rate = 16000
    duration = 2.0
    samples = int(sample_rate * duration)
    
    # Create a simple sine wave for testing
    t = np.linspace(0, duration, samples)
    audio = 0.3 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    
    # Create a longer RIR (3 seconds) to force "full" convolution behavior
    rir_duration = 3.0
    rir_samples = int(sample_rate * rir_duration)
    
    # Create an exponentially decaying RIR
    rir_t = np.linspace(0, rir_duration, rir_samples)
    rir = 0.5 * np.exp(-rir_t * 2) * np.random.randn(rir_samples) * 0.1
    
    # Save RIR
    rir_path = rir_dir / "test_rir.npy"
    np.save(rir_path, rir)
    
    return temp_dir, audio, sample_rate, rir_dir


def test_rir_chain_effects():
    """Test RandomRIR followed by RandomReverb to check for looping artifacts"""
    print("Testing RandomRIR + RandomReverb chain...")
    
    # Create test data
    temp_dir, audio, sample_rate, rir_dir = create_test_audio_and_rir()
    
    try:
        # Test the chain: RandomRIR -> RandomReverb (as in validation mode)
        rir_effect = RandomRIR(rir_dir, deterministic=True)
        reverb_effect = RandomReverb(deterministic=True)
        
        print(f"Original audio length: {len(audio)} samples ({len(audio)/sample_rate:.2f}s)")
        
        # Apply RandomRIR
        audio_rir = rir_effect.apply(audio.copy(), sample_rate)
        print(f"After RIR length: {len(audio_rir)} samples ({len(audio_rir)/sample_rate:.2f}s)")
        
        # Apply RandomReverb
        audio_final = reverb_effect.apply(audio_rir.copy(), sample_rate)
        print(f"After Reverb length: {len(audio_final)} samples ({len(audio_final)/sample_rate:.2f}s)")
        
        # Analyze for looping artifacts
        # Check if the end of the audio has similar patterns to earlier parts (indication of looping)
        def check_for_looping(audio_data, window_size=1000):
            """Simple check for looping by comparing end segment with earlier segments"""
            if len(audio_data) < window_size * 3:
                return False, 0.0
                
            end_segment = audio_data[-window_size:]
            max_correlation = 0.0
            
            # Check correlation with earlier segments
            for i in range(0, len(audio_data) - window_size * 2, window_size // 4):
                segment = audio_data[i:i + window_size]
                correlation = np.corrcoef(end_segment, segment)[0, 1]
                if not np.isnan(correlation):
                    max_correlation = max(max_correlation, abs(correlation))
            
            # If correlation is too high, it might indicate looping
            is_looping = max_correlation > 0.7
            return is_looping, max_correlation
        
        is_looping, max_corr = check_for_looping(audio_final)
        
        if is_looping:
            print(f"⚠️  Potential looping detected (max correlation: {max_corr:.3f})")
        else:
            print(f"✅ No significant looping detected (max correlation: {max_corr:.3f})")
        
        # Check for abrupt changes at the end that might cause artifacts
        def check_for_abrupt_changes(audio_data, window_size=100):
            """Check for abrupt amplitude changes that might cause artifacts"""
            if len(audio_data) < window_size * 2:
                return False, 0.0
                
            # Compare amplitude at the end vs near the end
            end_rms = np.sqrt(np.mean(audio_data[-window_size:] ** 2))
            near_end_rms = np.sqrt(np.mean(audio_data[-window_size*2:-window_size] ** 2))
            
            if near_end_rms > 0:
                ratio = end_rms / near_end_rms
                abrupt_change = ratio < 0.1 or ratio > 10  # Sudden change in amplitude
                return abrupt_change, ratio
            
            return False, 1.0
        
        has_abrupt_change, amplitude_ratio = check_for_abrupt_changes(audio_final)
        
        if has_abrupt_change:
            print(f"⚠️  Abrupt amplitude change detected (ratio: {amplitude_ratio:.3f})")
        else:
            print(f"✅ Smooth amplitude transition (ratio: {amplitude_ratio:.3f})")
        
        # Save test audio for manual inspection
        output_path = temp_dir / "test_output.wav"
        sf.write(output_path, audio_final, sample_rate)
        
        print(f"\n📁 Test audio saved to: {output_path}")
        print("You can listen to this file to verify there are no looping artifacts.")
        
        # Overall assessment
        if not is_looping and not has_abrupt_change:
            print("\n🎉 SUCCESS: No looping artifacts detected!")
            return True
        else:
            print("\n❌ POTENTIAL ISSUES: Looping artifacts may still be present.")
            return False
            
    finally:
        # Clean up temporary files
        import shutil
        shutil.rmtree(temp_dir)


def test_windowed_truncation():
    """Test the windowed truncation logic specifically"""
    print("\nTesting windowed truncation logic...")
    
    # Create test signals
    original_length = 10000
    convolved_length = 15000  # Longer than original
    
    # Create a signal that ends with non-zero values (would cause artifacts if cut abruptly)
    audio_convolved = np.sin(np.linspace(0, 4*np.pi, convolved_length)) * 0.5
    
    # Simulate the windowed truncation logic
    excess_length = convolved_length - original_length
    window_length = min(1000, excess_length // 2)
    
    # Extract parts
    main_part = audio_convolved[:original_length - window_length]
    windowed_part = audio_convolved[original_length - window_length:original_length]
    
    # Apply fade-out window
    fade_window = np.cos(np.linspace(0, np.pi/2, window_length)) ** 2
    windowed_part = windowed_part * fade_window
    
    # Combine
    final_audio = np.concatenate([main_part, windowed_part])
    
    print(f"Original length: {original_length}")
    print(f"Convolved length: {convolved_length}")
    print(f"Final length: {len(final_audio)}")
    print(f"Window length: {window_length}")
    
    # Check that the end smoothly fades to near-zero
    end_values = final_audio[-100:]
    max_end_value = np.max(np.abs(end_values))
    
    if max_end_value < 0.1:
        print(f"✅ Smooth fade-out achieved (max end value: {max_end_value:.4f})")
        return True
    else:
        print(f"❌ Fade-out insufficient (max end value: {max_end_value:.4f})")
        return False


def main():
    print("Testing RandomRIR looping artifact fix...\n")
    
    # Test windowed truncation logic
    truncation_ok = test_windowed_truncation()
    
    # Test full chain effects
    chain_ok = test_rir_chain_effects()
    
    if truncation_ok and chain_ok:
        print("\n🎉 All tests passed! The RandomRIR fix should prevent looping artifacts.")
    else:
        print("\n❌ Some tests failed. The fix may need further adjustment.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
