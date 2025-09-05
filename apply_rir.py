#!/usr/bin/env python3
"""
Script to apply Room Impulse Response (RIR) convolution to an audio file.
"""

import argparse
import logging
import subprocess
import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy import signal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def apply_rir_convolution(wav_path, rir_path, output_path, rir_rate=44000, bitrate="32k"):
    """
    Apply RIR convolution to an audio file and save as compressed Opus.
    
    Args:
        wav_path: Path to input WAV file
        rir_path: Path to RIR numpy array (.npy file)
        output_path: Path for output Opus file
        rir_rate: Sample rate for RIR processing
        bitrate: Bitrate for Opus encoding (e.g., "32k", "64k", "128k")
    """
    # Load audio file
    logger.info(f"Loading audio from {wav_path}")
    wav, sr = librosa.load(wav_path, sr=None)
    original_length = len(wav)
    
    # Load RIR
    logger.info(f"Loading RIR from {rir_path}")
    rir = np.load(rir_path, allow_pickle=True)
    rir = np.squeeze(rir)
    
    if not isinstance(rir, np.ndarray):
        raise ValueError("RIR must be a numpy array")
    
    # Normalize RIR to prevent excessive amplification
    rir = rir / (np.abs(rir).max() + 1e-7)
    
    # Resample audio to RIR rate
    logger.info(f"Resampling audio from {sr}Hz to {rir_rate}Hz")
    wav_resampled = librosa.resample(wav, orig_sr=sr, target_sr=rir_rate, res_type="kaiser_fast")
    
    # Ensure both wav and rir are 1D arrays for convolution
    wav_resampled = wav_resampled.flatten()
    rir = rir.flatten()
    
    # Apply convolution
    if len(rir) > 0:
        logger.info(f"Applying RIR convolution (wav shape: {wav_resampled.shape}, rir shape: {rir.shape})")
        wav_convolved = signal.convolve(wav_resampled, rir, mode="same")
    else:
        logger.warning("RIR is empty, skipping convolution")
        wav_convolved = wav_resampled
    
    # Normalize to prevent clipping
    actlev = np.max(np.abs(wav_convolved))
    if actlev > 0.99:
        wav_convolved = (wav_convolved / actlev) * 0.98
        logger.info(f"Normalized audio to prevent clipping (peak was {actlev:.3f})")
    
    # Resample back to original sample rate
    logger.info(f"Resampling back to original rate {sr}Hz")
    wav_final = librosa.resample(wav_convolved, orig_sr=rir_rate, target_sr=sr, res_type="kaiser_fast")
    
    # Adjust length to match original
    if abs(original_length - len(wav_final)) > 10:
        logger.warning(f"Length mismatch: original {original_length} vs processed {len(wav_final)}")
    
    if original_length > len(wav_final):
        wav_final = np.pad(wav_final, (0, original_length - len(wav_final)))
    elif original_length < len(wav_final):
        wav_final = wav_final[:original_length]
    
    # Save to temporary WAV file first
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        tmp_wav_path = tmp_wav.name
        logger.info(f"Saving processed audio to temporary file: {tmp_wav_path}")
        sf.write(tmp_wav_path, wav_final, sr)
    
    # Ensure output has proper extension for Opus
    if not str(output_path).endswith(('.opus', '.ogg')):
        output_path = Path(str(output_path).rsplit('.', 1)[0] + '.opus')
        logger.info(f"Changed output extension to .opus: {output_path}")
    
    # Convert to Opus using ffmpeg
    logger.info(f"Converting to Opus format at {bitrate} bitrate")
    try:
        cmd = [
            "ffmpeg",
            "-i", tmp_wav_path,
            "-c:a", "libopus",
            "-b:a", bitrate,
            "-compression_level", "10",
            "-vbr", "on",
            "-y",
            str(output_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info(f"Successfully saved Opus file to {output_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg conversion failed: {e.stderr}")
        raise
    finally:
        # Clean up temporary file
        Path(tmp_wav_path).unlink(missing_ok=True)
    
    logger.info("Done!")


def main():
    parser = argparse.ArgumentParser(description="Apply RIR convolution to audio file and save as Opus")
    parser.add_argument("wav_path", type=Path, help="Path to input WAV file")
    parser.add_argument("rir_path", type=Path, help="Path to RIR numpy array (.npy file)")
    parser.add_argument("output_path", type=Path, help="Path for output Opus file")
    parser.add_argument("--rir-rate", type=int, default=44000, help="Sample rate for RIR processing (default: 44000)")
    parser.add_argument("--bitrate", type=str, default="32k", help="Bitrate for Opus encoding (default: 32k)")
    
    args = parser.parse_args()
    
    # Validate input files exist
    if not args.wav_path.exists():
        raise FileNotFoundError(f"WAV file not found: {args.wav_path}")
    if not args.rir_path.exists():
        raise FileNotFoundError(f"RIR file not found: {args.rir_path}")
    
    # Create output directory if needed
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    
    apply_rir_convolution(args.wav_path, args.rir_path, args.output_path, args.rir_rate, args.bitrate)


if __name__ == "__main__":
    main()