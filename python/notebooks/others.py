from scipy.signal import find_peaks

def dominant_frequencies_from_audio(audio, sample_rate=16000, n_peaks=4, low_frequency=100, high_frequency=2000, min_distance_hz=20):
    frequencies = numpy.fft.rfftfreq(len(audio), d=1.0 / sample_rate)
    magnitudes = numpy.abs(numpy.fft.rfft(audio))

    mask = (frequencies >= low_frequency) & (frequencies <= high_frequency)
    frequencies_masked = frequencies[mask]
    magnitudes_masked = magnitudes[mask]

    peak_indices, _ = find_peaks(magnitudes_masked, distance=min_distance_hz)
    top_indices = peak_indices[numpy.argsort(magnitudes_masked[peak_indices])[-n_peaks:][::-1]]
    return frequencies_masked[top_indices], magnitudes_masked[top_indices]

from librosa.feature import mfcc

def mel_frequency_cepstral_coefficients_from_frame(frame, sample_rate=16000):
    n = len(frame)
    coefficients = mfcc(y=frame, sr=sample_rate, n_mfcc=13, n_fft=n, hop_length=n, center=False)
    return coefficients[:, 0]

def def hanning(audio): return audio * numpy.hanning(len(audio))
