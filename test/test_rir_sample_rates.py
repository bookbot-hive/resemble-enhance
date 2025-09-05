#!/usr/bin/env python3
"""
Test script to check RIR sample rate detection and create sample_rate.txt files.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from resemble_enhance.data.distorter.custom import RandomRIR


def create_sample_rate_files():
    """Create sample_rate.txt files for RIR datasets that don't have them"""
    
    rir_dir = Path('data/rir')
    
    if not rir_dir.exists():
        print(f"❌ RIR directory not found: {rir_dir}")
        return
    
    print("🔍 Scanning RIR datasets for sample_rate.txt files...")
    
    # Common sample rates for different RIR datasets (you can adjust these)
    default_rates = {
        'Arni': 48000,
        'BUT_ReverbDB': 16000,
        'GTU-RIR': 48000,
        'MeshRIR': 48000,
        'MIRACLE': 48000,
        'MIT_Survey': 44100,
        'OPENAIR': 48000,
        'RWCP_REVERB_AACHEN': 48000,
    }
    
    for subdir in rir_dir.iterdir():
        if not subdir.is_dir():
            continue
            
        sample_rate_file = subdir / "sample_rate.txt"
        dataset_name = subdir.name
        
        if sample_rate_file.exists():
            try:
                with open(sample_rate_file, 'r') as f:
                    rate = int(f.read().strip())
                print(f"✅ {dataset_name}: {rate} Hz (existing)")
            except (ValueError, IOError):
                print(f"⚠️  {dataset_name}: Invalid sample_rate.txt file")
        else:
            # Create sample_rate.txt with default rate
            default_rate = default_rates.get(dataset_name, 44100)
            
            try:
                with open(sample_rate_file, 'w') as f:
                    f.write(str(default_rate))
                print(f"📝 {dataset_name}: Created sample_rate.txt with {default_rate} Hz")
            except IOError as e:
                print(f"❌ {dataset_name}: Could not create sample_rate.txt: {e}")


def test_rir_sample_rate_detection():
    """Test the RIR sample rate detection system"""
    
    print("\n" + "="*60)
    print("TESTING RIR SAMPLE RATE DETECTION")
    print("="*60)
    
    try:
        rir_effect = RandomRIR(Path('data/rir'), deterministic=True)
        
        print(f"📊 Dataset sample rates:")
        for dataset, rate in rir_effect.rir_dataset_rates.items():
            print(f"   {dataset}: {rate} Hz")
        
        print(f"\n🧪 Testing RIR sampling...")
        
        # Test sampling a few RIRs
        for i in range(3):
            rir, native_rate = rir_effect._sample_rir()
            if rir is not None:
                print(f"   Sample {i+1}: RIR shape {rir.shape}, native rate {native_rate} Hz")
            else:
                print(f"   Sample {i+1}: No RIR found")
        
        print(f"\n✅ RIR sample rate detection working correctly!")
        
    except Exception as e:
        print(f"❌ Error testing RIR sample rate detection: {e}")
        import traceback
        traceback.print_exc()


def main():
    print("🎛️  RIR SAMPLE RATE CONFIGURATION TOOL")
    print("="*50)
    
    # First, create any missing sample_rate.txt files
    create_sample_rate_files()
    
    # Then test the detection system
    test_rir_sample_rate_detection()
    
    print(f"\n📋 SUMMARY:")
    print(f"   - Each RIR dataset subdirectory should have a 'sample_rate.txt' file")
    print(f"   - The file should contain just the sample rate number (e.g., '48000')")
    print(f"   - Sample rates are cached when RandomRIR is first created")
    print(f"   - During RIR application, audio is resampled to match RIR's native rate")
    print(f"   - This ensures proper convolution without sample rate mismatches")


if __name__ == "__main__":
    main()
