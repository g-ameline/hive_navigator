import numpy
import scipy.signal
import scipy.fft
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
        print(f"\t{hour_timestamp=}")
        yield hour_timestamp, hourly_feature_dict_from_audio(audio)


# ── spectral statistics ─────────────────────────────────────────────


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


# ── UrBAN hand-crafted features ─────────────────────────────────────


def hive_power_features_from_magnitudes(
    magnitudes,
    frequencies,
    hive_low_frequency=122.0,
    hive_high_frequency=515.0,
):
    power = magnitudes ** 2
    hive_band_mask = (
        (frequencies >= hive_low_frequency) & (frequencies <= hive_high_frequency)
    )

    hive_power = power[hive_band_mask].sum(axis=0)
    total_power = power.sum(axis=0) + 1e-10
    audio_band_density_ratio = hive_power / total_power

    hive_power_diff = numpy.diff(hive_power)
    audio_density_variation = numpy.concatenate([[0.0], hive_power_diff])

    return {
        "hive_power": hive_power,
        "audio_band_density_ratio": audio_band_density_ratio,
        "audio_density_variation": audio_density_variation,
    }


def audio_band_coefficients_from_magnitudes(magnitudes, frequencies, n_bands=16):
    power = magnitudes ** 2
    band_edges = numpy.linspace(frequencies[0], frequencies[-1], n_bands + 1)

    result = {}
    for band_index in range(n_bands):
        band_mask = (
            (frequencies >= band_edges[band_index])
            & (frequencies < band_edges[band_index + 1])
        )
        result[f"audio_band_coefficient_{band_index}"] = power[band_mask].sum(axis=0)

    return result


# ── linear-frequency cepstral coefficients ──────────────────────────


def linear_frequency_cepstral_coefficients_from_magnitudes(
    magnitudes,
    frequencies,
    n_coefficients=13,
    n_filters=26,
):
    """LFCCs: like MFCCs but with linearly spaced triangular filterbank."""
    power = (magnitudes ** 2).T  # (n_frames, n_freq_bins)
    filterbank = linear_filterbank_from_parameters(
        magnitudes.shape[0], frequencies[-1] * 2, n_filters,
    )
    filtered = numpy.dot(power, filterbank.T)
    log_filtered = numpy.log(filtered + 1e-10)
    lfccs = scipy.fft.dct(log_filtered, type=2, axis=1, norm="ortho")[:, :n_coefficients]
    return lfccs.T  # (n_coefficients, n_frames)


def linear_filterbank_from_parameters(n_frequency_bins, sample_rate, n_filters):
    """Triangular filterbank with linearly spaced center frequencies."""
    max_frequency = sample_rate / 2.0
    centers = numpy.linspace(0, max_frequency, n_filters + 2)
    frequency_bins = numpy.linspace(0, max_frequency, n_frequency_bins)

    filterbank = numpy.zeros((n_filters, n_frequency_bins))
    for i in range(n_filters):
        left, center, right = centers[i], centers[i + 1], centers[i + 2]

        rising = (frequency_bins >= left) & (frequency_bins <= center)
        filterbank[i, rising] = (
            (frequency_bins[rising] - left) / (center - left + 1e-10)
        )

        falling = (frequency_bins > center) & (frequency_bins <= right)
        filterbank[i, falling] = (
            (right - frequency_bins[falling]) / (right - center + 1e-10)
        )

    return filterbank


# ── gammatone cepstral coefficients ─────────────────────────────────


def gammatone_cepstral_coefficients_from_magnitudes(
    magnitudes,
    frequencies,
    sample_rate,
    n_coefficients=13,
    n_filters=40,
    low_frequency=50.0,
):
    """GFCCs: cepstral coefficients from gammatone-shaped filterbank."""
    power = (magnitudes ** 2).T  # (n_frames, n_freq_bins)
    filterbank = gammatone_filterbank_from_parameters(
        magnitudes.shape[0], sample_rate, n_filters, low_frequency,
    )
    filtered = numpy.dot(power, filterbank.T)
    log_filtered = numpy.log(filtered + 1e-10)
    gfccs = scipy.fft.dct(log_filtered, type=2, axis=1, norm="ortho")[:, :n_coefficients]
    return gfccs.T  # (n_coefficients, n_frames)


def erb_frequency_from_hz(frequency_hz):
    """Hz to ERB-rate (Glasberg & Moore 1990)."""
    return 9.265 * numpy.log(1.0 + frequency_hz / (24.7 * 9.265))


def hz_from_erb_frequency(erb):
    """ERB-rate back to Hz."""
    return 24.7 * 9.265 * (numpy.exp(erb / 9.265) - 1.0)


def gammatone_filterbank_from_parameters(
    n_frequency_bins,
    sample_rate,
    n_filters,
    low_frequency=50.0,
    filter_order=4,
):
    """Approximate gammatone filterbank in the frequency domain.
    Each filter is a rounded-exponential shape centered on ERB-spaced frequencies."""
    max_frequency = sample_rate / 2.0
    frequency_bins = numpy.linspace(0, max_frequency, n_frequency_bins)

    erb_low = erb_frequency_from_hz(low_frequency)
    erb_high = erb_frequency_from_hz(max_frequency)
    center_erbs = numpy.linspace(erb_low, erb_high, n_filters)
    center_frequencies = hz_from_erb_frequency(center_erbs)

    filterbank = numpy.zeros((n_filters, n_frequency_bins))
    for i, center in enumerate(center_frequencies):
        erb_bandwidth = 24.7 * (4.37 * center / 1000.0 + 1.0)
        normalized_distance = numpy.abs(frequency_bins - center) / (erb_bandwidth + 1e-10)
        filterbank[i] = (1.0 + normalized_distance) * numpy.exp(-normalized_distance)

    row_sums = filterbank.sum(axis=1, keepdims=True) + 1e-10
    filterbank = filterbank / row_sums

    return filterbank


# ── bark bands ──────────────────────────────────────────────────────


def bark_from_hz(frequency_hz):
    """Hz to Bark scale (Traunmüller 1990)."""
    return 26.81 / (1.0 + 1960.0 / (frequency_hz + 1e-10)) - 0.53


def bark_bands_from_magnitudes(
    magnitudes,
    frequencies,
    sample_rate,
    n_bands=27,
):
    """Power summed in Bark-spaced bands."""
    power = magnitudes ** 2
    bark_values = bark_from_hz(frequencies)
    bark_min = max(0.0, bark_values[frequencies > 0][0]) if any(frequencies > 0) else 0.0
    bark_max = bark_values[-1]
    band_edges = numpy.linspace(bark_min, bark_max, n_bands + 1)

    result = {}
    for band_index in range(n_bands):
        band_mask = (
            (bark_values >= band_edges[band_index])
            & (bark_values < band_edges[band_index + 1])
        )
        band_power = power[band_mask].sum(axis=0) if band_mask.any() else numpy.zeros(magnitudes.shape[1])
        result[f"bark_band_{band_index}"] = band_power

    return result


# ── ERB bands ───────────────────────────────────────────────────────


def erb_bands_from_magnitudes(
    magnitudes,
    frequencies,
    sample_rate,
    n_bands=40,
    low_frequency=50.0,
):
    """Power summed in ERB-spaced bands."""
    power = magnitudes ** 2
    max_frequency = sample_rate / 2.0
    erb_low = erb_frequency_from_hz(low_frequency)
    erb_high = erb_frequency_from_hz(max_frequency)
    band_edges_erb = numpy.linspace(erb_low, erb_high, n_bands + 1)
    band_edges_hz = hz_from_erb_frequency(band_edges_erb)

    erb_values = erb_frequency_from_hz(frequencies)

    result = {}
    for band_index in range(n_bands):
        band_mask = (
            (erb_values >= band_edges_erb[band_index])
            & (erb_values < band_edges_erb[band_index + 1])
        )
        band_power = power[band_mask].sum(axis=0) if band_mask.any() else numpy.zeros(magnitudes.shape[1])
        result[f"erb_band_{band_index}"] = band_power

    return result


# ── spectral contrast ───────────────────────────────────────────────


def spectral_contrast_from_magnitudes(
    magnitudes,
    frequencies,
    sample_rate,
    n_bands=6,
    quantile=0.2,
):
    """Peak-to-valley ratio per sub-band (octave-spaced).
    Returns spectral_contrast and spectral_valley arrays."""
    power = magnitudes ** 2

    fmin = 100.0
    band_edges = [fmin * (2.0 ** i) for i in range(n_bands + 1)]
    band_edges[-1] = min(band_edges[-1], sample_rate / 2.0)

    contrast_frames = []
    valley_frames = []

    for frame_index in range(power.shape[1]):
        frame_contrast = []
        frame_valley = []
        for band_index in range(n_bands):
            band_mask = (
                (frequencies >= band_edges[band_index])
                & (frequencies < band_edges[band_index + 1])
            )
            band_power = power[band_mask, frame_index]
            if len(band_power) == 0:
                frame_contrast.append(0.0)
                frame_valley.append(0.0)
                continue
            sorted_power = numpy.sort(band_power)
            n_quantile = max(1, int(len(sorted_power) * quantile))
            valley = numpy.mean(sorted_power[:n_quantile])
            peak = numpy.mean(sorted_power[-n_quantile:])
            frame_valley.append(float(numpy.log(valley + 1e-10)))
            frame_contrast.append(float(numpy.log(peak + 1e-10) - numpy.log(valley + 1e-10)))
        contrast_frames.append(frame_contrast)
        valley_frames.append(frame_valley)

    contrast_array = numpy.array(contrast_frames).T  # (n_bands, n_frames)
    valley_array = numpy.array(valley_frames).T

    result = {}
    for band_index in range(n_bands):
        result[f"spectral_contrast_{band_index}"] = contrast_array[band_index]
        result[f"spectral_valley_{band_index}"] = valley_array[band_index]

    return result


# ── strong peak ─────────────────────────────────────────────────────


def strong_peak_from_magnitudes(magnitudes):
    """Ratio of strongest spectral peak to mean magnitude, per frame."""
    peak = numpy.max(magnitudes, axis=0)
    mean = numpy.mean(magnitudes, axis=0) + 1e-10
    return {"spectral_strong_peak": peak / mean}


# ── high frequency content ──────────────────────────────────────────


def high_frequency_content_from_magnitudes(magnitudes, frequencies):
    """Sum of magnitude weighted by frequency bin index (emphasizes high freqs)."""
    bin_indices = numpy.arange(magnitudes.shape[0], dtype=numpy.float64)[:, None]
    hfc = (magnitudes ** 2 * bin_indices).sum(axis=0)
    return {"high_frequency_content": hfc}


# ── passband features ───────────────────────────────────────────────


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
            (band_weights * band_deviations ** 2).sum(axis=0),
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


# ── modulation spectrogram ──────────────────────────────────────────


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
            audio[start : start + stft_window_size] * hanning_window,
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
            modulation_frequencies_restricted[energy_per_rate.argmax()],
        )
    return stats


# ── aggregation ─────────────────────────────────────────────────────


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


# ── utilities ───────────────────────────────────────────────────────


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


def hourly_feature_dict_from_audio(
    audio,
    sample_rate=16000,
    frame_length=16000,
    hop_length=8000,
):
    per_frame = {}

    audio = noise_reduced_audio_from_audio(audio, sample_rate)
    pre_emphasized = pre_emphasized_audio_from_audio(audio)

    magnitudes = numpy.abs(librosa.stft(
        y=audio, n_fft=frame_length, hop_length=hop_length, center=False,
    ))
    frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=frame_length)

    pre_emphasized_magnitudes = numpy.abs(librosa.stft(
        y=pre_emphasized, n_fft=frame_length, hop_length=hop_length, center=False,
    ))

    per_frame.update(spectral_stats_from_magnitudes(magnitudes, frequencies))
    per_frame.update(spectral_flux_from_magnitudes(magnitudes))
    per_frame.update(dominant_frequencies_from_magnitudes(magnitudes, frequencies))

    per_frame["mel_frequency_cepstral_coefficients"] = librosa.feature.mfcc(
        y=pre_emphasized, sr=sample_rate, n_mfcc=13,
        n_fft=frame_length, hop_length=hop_length, center=False,
    )

    per_frame["linear_frequency_cepstral_coefficients"] = (
        linear_frequency_cepstral_coefficients_from_magnitudes(
            pre_emphasized_magnitudes, frequencies,
            n_coefficients=13, n_filters=26,
        )
    )

    per_frame["gammatone_frequency_cepstral_coefficients"] = (
        gammatone_cepstral_coefficients_from_magnitudes(
            pre_emphasized_magnitudes, frequencies, sample_rate,
            n_coefficients=13, n_filters=40,
        )
    )

    per_frame["chroma"] = librosa.feature.spectral.chroma_stft(
        y=audio, sr=sample_rate,
        n_fft=frame_length, hop_length=hop_length, center=False,
    )

    per_frame["zero_crossing_rate"] = librosa.feature.zero_crossing_rate(
        y=audio, frame_length=frame_length, hop_length=hop_length, center=False,
    )[0]

    per_frame["root_mean_square_energy"] = librosa.feature.rms(
        y=audio, frame_length=frame_length, hop_length=hop_length, center=False,
    )[0]

    per_frame.update(hive_power_features_from_magnitudes(magnitudes, frequencies))
    per_frame.update(audio_band_coefficients_from_magnitudes(magnitudes, frequencies))
    per_frame.update(bark_bands_from_magnitudes(magnitudes, frequencies, sample_rate))
    per_frame.update(erb_bands_from_magnitudes(magnitudes, frequencies, sample_rate))
    per_frame.update(spectral_contrast_from_magnitudes(magnitudes, frequencies, sample_rate))
    per_frame.update(strong_peak_from_magnitudes(magnitudes))
    per_frame.update(high_frequency_content_from_magnitudes(magnitudes, frequencies))

    per_frame.update(passband_per_frame_from_audio(
        audio, sample_rate, frame_length, hop_length,
    ))
    per_frame.update(passband_ratios_per_frame_from(per_frame))

    result = aggregated_from_per_frame_arrays(per_frame)

    result.update(modulation_stats_from_audio(
        audio, passbands.low_and_middle_and_high, sample_rate,
    ))

    return result

def noise_reduced_audio_from_audio(
    audio,
    sample_rate=16000,
    n_fft=1024,
    hop_length=512,
    n_noise_frames=20,
):
    """Spectral amplitude subtraction with minimum-statistics noise estimate.
    Estimates noise as the mean of the N frames with lowest total energy."""
    full_stft = librosa.stft(y=audio, n_fft=n_fft, hop_length=hop_length)
    magnitudes = numpy.abs(full_stft)
    phases = numpy.angle(full_stft)

    frame_energies = (magnitudes ** 2).sum(axis=0)
    quietest_indices = numpy.argsort(frame_energies)[:n_noise_frames]
    noise_power_mean = numpy.mean(
        magnitudes[:, quietest_indices] ** 2, axis=1, keepdims=True,
    )

    clean_power = numpy.maximum(magnitudes ** 2 - noise_power_mean, 0.0)
    clean_stft = numpy.sqrt(clean_power) * numpy.exp(1j * phases)
    return librosa.istft(clean_stft, hop_length=hop_length, length=len(audio))

# ── pre-emphasis ────────────────────────────────────────────────────


def pre_emphasized_audio_from_audio(audio, coefficient=0.97):
    """First-order high-pass pre-emphasis filter."""
    return numpy.append(audio[0], audio[1:] - coefficient * audio[:-1])


# ── dominant frequency tracking ─────────────────────────────────────


def dominant_frequencies_from_magnitudes(magnitudes, frequencies, n_peaks=3):
    """Top-N spectral peaks per frame by magnitude.
    Returns dict with dominant_frequency_0, _1, _2 arrays (n_frames,)."""
    result = {}
    for frame_index in range(n_peaks):
        result[f"dominant_frequency_{frame_index}"] = numpy.zeros(magnitudes.shape[1])

    for col in range(magnitudes.shape[1]):
        frame = magnitudes[:, col]
        peak_indices = numpy.argsort(frame)[::-1][:n_peaks]
        peak_indices_sorted = numpy.sort(peak_indices)
        for rank, idx in enumerate(peak_indices_sorted):
            result[f"dominant_frequency_{rank}"][col] = frequencies[idx]

    return result



