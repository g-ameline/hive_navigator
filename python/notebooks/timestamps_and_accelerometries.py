import numpy
from pipe import Pipe

from scipy.signal import find_peaks

@Pipe
def triples_from_hive_acceleromtry_filepath(filepath):
    expected_header = 'timestamp,f1,m1,f2,m2,f3,m3'
    with open(filepath, mode='r') as stream:
        assert stream.readline().strip() == expected_header, "unexpected header"
    raw = numpy.genfromtxt(filepath, delimiter=',', skip_header=1, usecols=range(1, 7), dtype=numpy.float32)
    
    def assert_accelerometry_array_valid(raw):
        assert raw.ndim == 2, f"expected 2d array, got {raw.ndim}d"
        assert raw.shape[1] == 6, f"expected 6 columns, got {raw.shape[1]}"
        assert raw.shape[0] > 0, "no data rows"
        assert not numpy.any(numpy.isnan(raw)), "values contain NaN"
        assert not numpy.any(numpy.isinf(raw)), "values overflow float32"

    assert_accelerometry_array_valid(raw)
    timestamps = numpy.genfromtxt(filepath, delimiter=',', skip_header=1, usecols=0, dtype='datetime64[s]')
    assert timestamps.shape[0] == raw.shape[0], (
        f"timestamp count {timestamps.shape[0]} != data row count {raw.shape[0]} — unparseable timestamps"
    )
    frequencies = raw[:, [0, 2, 4]]
    magnitudes = raw[:, [1, 3, 5]]
    return timestamps, frequencies, magnitudes

@Pipe
def flatten_triple_frequencies_and_triple_magnitudes(
    timestamps_and_triple_frequencies_and_triple_magnitudes,    
):
    timestamps, triple_frequencies, triple_magnitudes = timestamps_and_triple_frequencies_and_triple_magnitudes
    return(
        numpy.repeat(timestamps, 3),
        triple_frequencies.flatten(),
        triple_magnitudes.flatten()
    )

@Pipe
def only_first_peak_from_accelerometry(
    timestamps_and_triple_frequencies_and_triple_magnitudes,    
):
    timestamps, triple_frequencies, triple_magnitudes = timestamps_and_triple_frequencies_and_triple_magnitudes    
    return (
        timestamps,
        triple_frequencies[:, 0],
        triple_magnitudes[:, 0]
    )

# @Pipe
# def harmonic_frequency_bin_triple_from_accelerometry(
#     timestamps_and_frequencies_and_magnitudes,    
#     bin_number= 2200,
#     prominence=100,
#     width=(None, 1.2),
# ):
#     timestamps, frequencies, magnitudes = timestamps_and_frequencies_and_magnitudes    
    
#     counts, bin_edges = numpy.histogram(frequencies, bins=bin_number)
#     bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
#     peak_indices, properties = find_peaks(
#         counts,
#         prominence=prominence,
#         width=width,
#     )
#     harmonic_bins = [(bin_edges[peak_index],bin_edges[peak_index+1]) for peak_index in peak_indices]

@Pipe
def harmonic_frequency_bin_triple_from_accelerometry(
    timestamps_and_frequencies_and_magnitudes,    
    neighbor_width=5,
    bin_count=5000,
):
    timestamps, frequencies, magnitudes = timestamps_and_frequencies_and_magnitudes    
    counts, bin_edges = numpy.histogram(frequencies, bins=bin_count)

    kernel = numpy.ones(2 * neighbor_width + 1) / (2 * neighbor_width + 1)
    local_mean = numpy.convolve(counts, kernel, mode='same')
    contrast = counts / numpy.maximum(local_mean, 1)

    top_3_indices = numpy.argsort(contrast)[-3:]
    top_3_indices = numpy.sort(top_3_indices)

    harmonic_bins = [(bin_edges[i], bin_edges[i + 1]) for i in top_3_indices]
    return harmonic_bins

def peaks(frequencies): 
    counts, bin_edges = numpy.histogram(frequencies.sort(), bins='auto')
    print(bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    peak_indices, _ = find_peaks(
        counts, 
        width = 3,
        prominence=400,
    )
    harmonic_bins = [(bin_edges[i], bin_edges[i + 1]) for i in peak_indices]
    return harmonic_bins

@Pipe
def harmonicless_accelerometry_from_accelerometry_and_harmonic_frequency_bins(
    timestamps_and_frequencies_and_magnitudes,    
    harmonic_frequency_bins
):
    timestamps, frequencies, magnitudes = timestamps_and_frequencies_and_magnitudes    
    mask = numpy.ones(len(frequencies), dtype=bool)
    for low, high in harmonic_frequency_bins:
        mask &= ~((frequencies >= low) & (frequencies <= high))
    
    return (
        timestamps[mask],
        frequencies[mask],
        magnitudes[mask],
    )

@Pipe
def harmonicless_accelerometry_from_accelerometry_from_accelerometry(
    timestamps_and_frequencies_and_magnitudes,    
):
    harmonic_frequency_bins = (
        timestamps_and_frequencies_and_magnitudes    
        | harmonic_frequency_bin_triple_from_accelerometry
    )
    return (
        harmonicless_accelerometry_from_accelerometry_and_harmonic_frequency_bins(
            harmonic_frequency_bins = harmonic_frequency_bins
        )
    )


@Pipe
def concatenated_accelerometry_from_accelerometries(accelerometries):
    triples = list(accelerometries)
    return (
        numpy.concatenate([t[0] for t in triples]),
        numpy.concatenate([t[1] for t in triples]),
        numpy.concatenate([t[2] for t in triples]),
    )


@Pipe
def high_magnitude_only_accelerometry(
    timestamps_and_frequencies_and_magnitudes,    
    threshold = 6,
):
    timestamps, frequencies, magnitudes = timestamps_and_frequencies_and_magnitudes
    mask = magnitudes >= threshold
    return (
        timestamps[mask],
        frequencies[mask],
        magnitudes[mask],
    )





import matplotlib.pyplot

def plot_magnitude_vs_frequency_gap_to_rank1(triple_frequencies, triple_magnitudes, rank=1):
    gap = numpy.abs(triple_frequencies[:, rank] - triple_frequencies[:, 0])
    figure, axis = matplotlib.pyplot.subplots(figsize=(14, 5))
    scatter = axis.scatter(
        gap,
        triple_magnitudes[:, rank],
        c=triple_magnitudes[:, 0],
        s=0.5,
        alpha=0.2,
        cmap='viridis',
    )
    axis.set_xlabel(f'|peak{rank+1} - peak1| frequency gap (Hz)')
    axis.set_ylabel('magnitude (log)')
    axis.set_yscale('log')
    figure.colorbar(scatter, ax=axis, label='peak1 magnitude')
    axis.set_title(f'peak {rank+1}: magnitude vs distance from peak 1')
    figure.tight_layout()

import matplotlib.pyplot

def plot_frequency_gap_between_ranks(triple_frequencies):
    gap_1_2 = numpy.abs(triple_frequencies[:, 0] - triple_frequencies[:, 1])
    gap_1_3 = numpy.abs(triple_frequencies[:, 0] - triple_frequencies[:, 2])

    figure, axes = matplotlib.pyplot.subplots(1, 2, figsize=(14, 4))
    axes[0].hist(gap_1_2, bins=300, color='green', edgecolor='none')
    axes[0].set_title('|peak1 - peak2| frequency gap')
    axes[1].hist(gap_1_3, bins=300, color='blue', edgecolor='none')
    axes[1].set_title('|peak1 - peak3| frequency gap')
    for axis in axes:
        axis.set_xlabel('Hz')
        axis.set_yscale('log')
    figure.tight_layout()




def _daily_band_statistic_from_accelerometry(
    timestamps_and_frequencies_and_magnitudes,    
    low_hz, high_hz, statistic
):
    timestamps, frequencies, magnitudes = timestamps_and_frequencies_and_magnitudes
    in_band_mask = (frequencies >= low_hz) & (frequencies <= high_hz)
    days = timestamps.astype('datetime64[D]')
    unique_days = numpy.unique(days)

    def day_stat(day):
        values = magnitudes[in_band_mask & (days == day)]
        return statistic(values) if values.size > 0 else numpy.nan

    return unique_days, numpy.array([day_stat(day) for day in unique_days])

@Pipe
def daily_mean_band_magnitude_from_accelerometry(
    timestamps_and_frequencies_and_magnitudes,    
    low_hz, high_hz
):
    return _daily_band_statistic_from_accelerometry(
        timestamps_and_frequencies_and_magnitudes,    
        low_hz, high_hz, numpy.mean,
    )

@Pipe
def daily_total_band_magnitude_from_accelerometry(
    timestamps_and_frequencies_and_magnitudes,    
    low_hz, high_hz
):
    return _daily_band_statistic_from_accelerometry(
        timestamps_and_frequencies_and_magnitudes,    
        low_hz, high_hz, numpy.sum,
    )

@Pipe
def daily_band_peak_count_from_accelerometry(
    timestamps_and_frequencies_and_magnitudes,    
    low_hz, high_hz
):
    return _daily_band_statistic_from_accelerometry(
    timestamps_and_frequencies_and_magnitudes,    
        low_hz, high_hz, len,
    )

@Pipe
def daily_in_band_peak_ratio_from_accelerometry(
    timestamps_and_frequencies_and_magnitudes,    
    low_hz, high_hz
):
    timestamps, frequencies, magnitudes = timestamps_and_frequencies_and_magnitudes
    in_band_mask = (frequencies >= low_hz) & (frequencies <= high_hz)
    days = timestamps.astype('datetime64[D]')
    unique_days = numpy.unique(days)

    def day_ratio(day):
        day_mask = days == day
        total = numpy.sum(day_mask)
        return numpy.sum(in_band_mask & day_mask) / total if total > 0 else numpy.nan

    return unique_days, numpy.array([day_ratio(day) for day in unique_days])

@Pipe
def daily_band_to_spectrum_magnitude_ratio_from_accelerometry(
    timestamps_and_frequencies_and_magnitudes,    
    low_hz, high_hz
):
    timestamps, frequencies, magnitudes = timestamps_and_frequencies_and_magnitudes
    in_band_mask = (frequencies >= low_hz) & (frequencies <= high_hz)
    days = timestamps.astype('datetime64[D]')
    unique_days = numpy.unique(days)

    def day_ratio(day):
        day_mask = days == day
        band_values = magnitudes[in_band_mask & day_mask]
        all_values = magnitudes[day_mask]
        if band_values.size == 0 or all_values.size == 0:
            return numpy.nan
        return numpy.median(band_values) / numpy.median(all_values)

    return unique_days, numpy.array([day_ratio(day) for day in unique_days])


def rolling_mean_from_daily_values(day_timestamps, values, window_days=3):
    kernel = numpy.ones(window_days) / window_days
    return day_timestamps, numpy.convolve(values, kernel, mode='same')


def binary_activity_from_daily_values(day_timestamps, values, threshold):
    return day_timestamps, (values >= threshold).astype(numpy.uint8)


@Pipe
def time_filtered_accelerometry(
    timestamps_and_frequencies_and_magnitudes,
    start_hour=8,
    end_hour=18,
):
    timestamps, frequencies, magnitudes = timestamps_and_frequencies_and_magnitudes
    hours = timestamps.astype('datetime64[h]').astype(int) % 24
    mask = (hours >= start_hour) & (hours < end_hour)
    return (
        timestamps[mask],
        frequencies[mask],
        magnitudes[mask],
    )

