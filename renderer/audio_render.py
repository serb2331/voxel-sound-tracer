# -------------------------
# Audio Rendering with SIR
# -------------------------
import numpy as np
# import soundfile as sf
import scipy.io.wavfile as wavfile


def apply_sir_to_audio(audio, sir, sample_rate):
    """
    Apply the spatial impulse response to an audio signal via convolution.
    
    Args:
        audio: Input audio signal (1D numpy array)
        sir: Spatial impulse response from compute_sir
        sample_rate: Sample rate (must match SIR sample rate)
    
    Returns:
        Convolved audio signal with room acoustics applied
    """
    # Use FFT-based convolution for efficiency
    output = np.convolve(audio, sir, mode='full')
    
    # Normalize to prevent clipping
    max_val = np.max(np.abs(output))
    if max_val > 0:
        output = output * 0.95 / max_val
    
    return output

def import_and_convolve_file(input_sound_file_path, output_sound_file_path, sir):
    # audio, sr = sf.read("input.wav")   # audio may be mono or stereo    
    
    # 1. Load your audio file
    sample_rate, audio = wavfile.read(input_sound_file_path)

    # 2. Resample if needed (must match SIR sample rate)
    if sample_rate != 8000:
        from scipy.signal import resample
        audio = resample(audio, int(len(audio) * 8000 / sample_rate))

    # 3. Convert to mono and normalize
    # further improvement to convolve binaurally
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32) / 32768.0

    # 4. Apply the SIR
    output = apply_sir_to_audio(audio, sir, sample_rate=8000)

    # 5. Save result
    output_int16 = (output * 32767).astype(np.int16)
    wavfile.write(output_sound_file_path, 8000, output_int16)