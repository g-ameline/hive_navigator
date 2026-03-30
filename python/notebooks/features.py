import numpy
import scipy.signal
import librosa

from pipe import Pipe

import passbands

@Pipe
def hourly_features_from_audio_stream(stream, minimum_samples=16000):
    """Takes (hour_timestamp, audio_array).
    Yields (hour_timestamp, feature_dict) — one row per hour."""
    for hour_timestamp, audio in stream:
        if len(audio) < minimum_samples:
            continue
        print(f'\t{hour_timestamp=}')
        yield hour_timestamp, hourly_feature_dict_from_audio(audio)


def hourly_feature_dict_from_audio(
    audio,
    sample_rate=16000,
    frame_length=16000,
    hop_length=8000,
):
    per_frame = {}

    magnitudes = numpy.abs(librosa.stft(
        y=audio, n_fft=frame_length, hop_length=hop_length, center=False,
    ))
    frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=frame_length)

    per_frame.update(spectral_stats_from_magnitudes(magnitudes, frequencies))
    per_frame.update(spectral_flux_from_magnitudes(magnitudes))

    per_frame["mel_frequency_cepstral_coefficients"] = librosa.feature.mfcc(
        y=audio, sr=sample_rate, n_mfcc=13,
        n_fft=frame_length, hop_length=hop_length, center=False,
    )

    per_frame["chroma"] =librosa.feature.spectral.chroma_stft(
        y=audio, sr=sample_rate,
        n_fft=frame_length, hop_length=hop_length, center=False,
    )

    per_frame["zero_crossing_rate"] = librosa.feature.zero_crossing_rate(
        y=audio, frame_length=frame_length, hop_length=hop_length, center=False,
    )[0]

    per_frame["root_mean_square_energy"] = librosa.feature.rms(
        y=audio, frame_length=frame_length, hop_length=hop_length, center=False,
    )[0]

    per_frame.update(passband_per_frame_from_audio(
        audio, sample_rate, frame_length, hop_length,
    ))
    per_frame.update(passband_ratios_per_frame_from(per_frame))

    result = aggregated_from_per_frame_arrays(per_frame)

    result.update(modulation_stats_from_audio(
        audio, passbands.low_and_middle_and_high, sample_rate,
    ))

    return result

def spectral_stats_from_magnitudes(magnitudes, frequencies):
    """Vectorized spectral statistics over all frames.
    magnitudes: (n_freq_bins, n_frames)
    frequencies: (n_freq_bins,)
    Returns dict of 1-d arrays, each (n_frames,)."""
    total = magnitudes.sum(axis=0, keepdims=True) + 1e-10
    weights = magnitudes / total
    freqs = frequencies[:, None]

    centroid = (freqs * weights).sum(axis=0)
    deviations = freqs - centroid[None, :]
    variance = (weights * deviations ** 2).sum(axis=0)
    bandwidth = numpy.sqrt(variance)

    cumulative = numpy.cumsum(magnitudes, axis=0)
    thresholds = 0.85 * cumulative[-1, :]
    rolloff_indices = numpy.array([
        numpy.searchsorted(cumulative[:, i], thresholds[i])
        for i in range(magnitudes.shape[1])
    ])
    rolloff = frequencies[numpy.minimum(rolloff_indices, len(frequencies) - 1)]

    power = magnitudes ** 2
    log_mean = numpy.mean(numpy.log(power + 1e-10), axis=0)
    flatness = numpy.exp(log_mean) / (numpy.mean(power, axis=0) + 1e-10)

    crest = numpy.max(magnitudes, axis=0) / (numpy.mean(magnitudes, axis=0) + 1e-10)

    probs = power / (power.sum(axis=0, keepdims=True) + 1e-10)
    entropy = -(probs * numpy.log2(probs + 1e-10)).sum(axis=0)

    std = bandwidth + 1e-10
    normalized = deviations / std[None, :]
    skewness = (weights * normalized ** 3).sum(axis=0)
    kurtosis = (weights * normalized ** 4).sum(axis=0)

    return {
        "spectral_centroid": centroid,
        "spectral_bandwidth": bandwidth,
        "spectral_rolloff": rolloff,
        "spectral_flatness": flatness,
        "spectral_crest": crest,
        "spectral_entropy": entropy,
        "spectral_skewness": skewness,
        "spectral_kurtosis": kurtosis,
    }


def spectral_flux_from_magnitudes(magnitudes):
    """Frame-to-frame spectral change. First frame gets 0."""
    diffs = numpy.diff(magnitudes, axis=1)
    flux = numpy.sqrt(numpy.mean(diffs ** 2, axis=0))
    return {"spectral_flux": numpy.concatenate([[0.0], flux])}


def passband_per_frame_from_audio(
    audio,
    sample_rate,
    frame_length,
    hop_length,
    order=6,
):
    result = {}
    for name, low_frequency, high_frequency in passbands.low_and_middle_and_high:
        filtered = bandpassed_audio_from_audio(
            audio, sample_rate, low_frequency, high_frequency, order,
        )
        band_rms = librosa.feature.rms(
            y=filtered, frame_length=frame_length,
            hop_length=hop_length, center=False,
        )[0]

        band_magnitudes = numpy.abs(librosa.stft(
            y=filtered, n_fft=frame_length,
            hop_length=hop_length, center=False,
        ))
        frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=frame_length)
        band_total = band_magnitudes.sum(axis=0, keepdims=True) + 1e-10
        band_weights = band_magnitudes / band_total
        freqs = frequencies[:, None]

        band_centroid = (freqs * band_weights).sum(axis=0)
        band_deviations = freqs - band_centroid[None, :]
        band_bandwidth = numpy.sqrt(
            (band_weights * band_deviations ** 2).sum(axis=0)
        )

        result[f"{name}_root_mean_square_energy"] = band_rms
        result[f"{name}_spectral_centroid"] = band_centroid
        result[f"{name}_spectral_bandwidth"] = band_bandwidth

    return result


def passband_ratios_per_frame_from(per_frame):
    low = per_frame["low_root_mean_square_energy"]
    middle = per_frame["middle_root_mean_square_energy"]
    high = per_frame["high_root_mean_square_energy"]
    return {
        "low_to_middle_energy_ratio": low / (middle + 1e-10),
        "low_to_high_energy_ratio": low / (high + 1e-10),
        "middle_to_high_energy_ratio": middle / (high + 1e-10),
    }


def modulation_stats_from_audio(
    audio,
    passbands,
    sample_rate=16000,
    stft_window_size=512,
    stft_hop_size=256,
    modulation_rate_min_hz=1.0,
    modulation_rate_max_hz=30.0,
):
    stft_frame_rate = sample_rate / stft_hop_size
    stft_frequency_bins = numpy.fft.rfftfreq(stft_window_size, d=1.0 / sample_rate)
    hanning_window = numpy.hanning(stft_window_size)

    stft_columns = numpy.array([
        numpy.abs(numpy.fft.rfft(
            audio[start:start + stft_window_size] * hanning_window
        ))
        for start in range(
            0, len(audio) - stft_window_size + 1, stft_hop_size,
        )
    ])

    modulation_spectrum = numpy.abs(numpy.fft.rfft(stft_columns, axis=0))
    modulation_frequencies = numpy.fft.rfftfreq(
        stft_columns.shape[0], d=1.0 / stft_frame_rate,
    )
    mod_mask = (
        (modulation_frequencies >= modulation_rate_min_hz)
        & (modulation_frequencies <= modulation_rate_max_hz)
    )
    restricted = modulation_spectrum[mod_mask, :]
    modulation_frequencies_restricted = modulation_frequencies[mod_mask]

    stats = {}
    for name, low_frequency, high_frequency in passbands:
        freq_mask = (
            (stft_frequency_bins >= low_frequency)
            & (stft_frequency_bins <= high_frequency)
        )
        band = restricted[:, freq_mask]
        stats[f"{name}_modulation_energy"] = float(band.sum())
        energy_per_rate = band.sum(axis=1)
        stats[f"{name}_peak_modulation_frequency"] = float(
            modulation_frequencies_restricted[energy_per_rate.argmax()]
        )
    return stats


def aggregated_from_per_frame_arrays(per_frame):
    """Collapses per-frame arrays into mean/std scalars."""
    result = {}
    for key, values in per_frame.items():
        if values.ndim == 1:
            result[f"{key}_mean"] = float(numpy.mean(values))
            result[f"{key}_std"] = float(numpy.std(values))
        elif values.ndim == 2:
            for i in range(values.shape[0]):
                result[f"{key}_{i}_mean"] = float(numpy.mean(values[i]))
                result[f"{key}_{i}_std"] = float(numpy.std(values[i]))
    return result


def bandpassed_audio_from_audio(
    audio,
    sample_rate,
    low_frequency,
    high_frequency,
    order=6,
):
    sos = scipy.signal.butter(
        order, [low_frequency, high_frequency],
        btype="band", fs=sample_rate, output="sos",
    )
    return scipy.signal.sosfilt(sos, audio)
