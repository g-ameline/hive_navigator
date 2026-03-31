import numpy
import matplotlib.pyplot
import matplotlib.colors
import matplotlib.dates
import matplotlib.gridspec
import matplotlib.transforms
import matplotlib.lines
import librosa
import librosa.display
import soundfile
import times
from datetime import datetime
from pipe import Pipe

matplotlib.pyplot.rcParams['figure.dpi'] = 100
matplotlib.pyplot.rcParams['savefig.dpi'] = 100


def zscore_heatmap(
    dataframe,
    hive_name,
    feature_columns=None,
    vmin=-3.0,
    vmax=3.0,
):
    hive = dataframe[dataframe["hive"] == hive_name].sort_values("timestamp")
    if feature_columns is None:
        feature_columns = [c for c in hive.columns if c.endswith("_zscore")]

    time_labels = hive["timestamp"].apply(
        lambda t: str(t.astype("datetime64[h]"))
    ).values
    matrix = hive[feature_columns].values.T

    figure, axis = matplotlib.pyplot.subplots(
        figsize=(max(16, len(time_labels) * 0.15), max(6, len(feature_columns) * 0.3))
    )
    mesh = axis.pcolormesh(
        matrix,
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        shading="nearest",
    )
    figure.colorbar(mesh, ax=axis, label="z-score")

    step = max(1, len(time_labels) // 30)
    axis.set_xticks(range(0, len(time_labels), step))
    axis.set_xticklabels(time_labels[::step], rotation=45, ha="right", fontsize=6)

    short_names = [c.removesuffix("_zscore") for c in feature_columns]
    axis.set_yticks(range(len(short_names)))
    axis.set_yticklabels(short_names, fontsize=7)

    axis.set_title(f"{hive_name} — z-scored features")
    figure.tight_layout()
    return figure


def spectrum(audio, sample_rate=16000, low_frequency=50, high_frequency=2000):
    freqs = numpy.fft.rfftfreq(len(audio), d=1.0 / sample_rate)
    magnitudes = numpy.abs(numpy.fft.rfft(audio))
    mask = (freqs >= low_frequency) & (freqs <= high_frequency)

    figure, axis = matplotlib.pyplot.subplots(figsize=(12, 4))
    axis.semilogy(freqs[mask], magnitudes[mask])
    axis.set_xlabel("Hz")
    axis.set_ylabel("Magnitude")
    figure.tight_layout()
    return figure


def spectrogram(
    timestamp_frame_pairs,
    sample_rate=16000,
    low_frequency=0,
    high_frequency=600,
    queen_event_hints=True,
):
    timestamps = numpy.array([t for t, _ in timestamp_frame_pairs])
    frames = [f for _, f in timestamp_frame_pairs]

    freqs = numpy.fft.rfftfreq(len(frames[0]), d=1.0 / sample_rate)
    mask = (freqs >= low_frequency) & (freqs <= high_frequency)
    freqs_masked = freqs[mask]

    spectrogram_matrix = numpy.array([
        numpy.abs(numpy.fft.rfft(frame))[mask]
        for frame in frames
    ])

    log_spectrogram = numpy.log1p(spectrogram_matrix)
    vmin = numpy.percentile(log_spectrogram, 2)
    vmax = numpy.percentile(log_spectrogram, 98)

    n_frames = len(frames)
    x_indices = numpy.arange(n_frames)

    figure, axis = matplotlib.pyplot.subplots(figsize=(16, 5))
    mesh = axis.pcolormesh(
        x_indices,
        freqs_masked,
        log_spectrogram.T,
        shading="nearest",
        cmap="inferno",
        vmin=vmin,
        vmax=vmax,
    )
    figure.colorbar(mesh, ax=axis, label="log(1 + magnitude)")

    def tick_positions_and_labels(timestamps, max_ticks=12):
        step = max(1, len(timestamps) // max_ticks)
        positions = list(range(0, len(timestamps), step))
        labels = [
            str(timestamps[i].astype("datetime64[s]").astype(datetime).strftime("%m/%d %H:%M"))
            for i in positions
        ]
        return positions, labels

    positions, labels = tick_positions_and_labels(timestamps)
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)

    if queen_event_hints:
        queen_start = numpy.datetime64(times.queen_event_window[0])
        queen_end = numpy.datetime64(times.queen_event_window[1])
        t_min, t_max = timestamps[0], timestamps[-1]
        if t_min < queen_start < t_max:
            for i in range(len(timestamps)):
                if timestamps[i] >= queen_start:
                    axis.axvline(i, color="red", linewidth=2, linestyle="--")
                    break
        if t_min < queen_end < t_max:
            for i in range(len(timestamps) - 1, -1, -1):
                if timestamps[i] <= queen_end:
                    axis.axvline(i, color="red", linewidth=2, linestyle="--")
                    break
        queen_day_start = numpy.datetime64(times.queen_event_day_window[0])
        queen_day_end = numpy.datetime64(times.queen_event_day_window[1])
        if t_min < queen_day_start < t_max:
            for i in range(len(timestamps)):
                if timestamps[i] >= queen_day_start:
                    axis.axvline(i, color="white", linewidth=2, linestyle="--")
                    break
        if t_min < queen_day_end < t_max:
            for i in range(len(timestamps) - 1, -1, -1):
                if timestamps[i] <= queen_day_end:
                    axis.axvline(i, color="white", linewidth=2, linestyle="--")
                    break

    axis.set_ylabel("Hz")
    figure.tight_layout()
    return figure


def mfcc_heatmap(timestamp_mfcc_couples):
    timestamps, mfccs = zip(*timestamp_mfcc_couples)
    matrix = numpy.stack(mfccs, axis=1)
    figure, axis = matplotlib.pyplot.subplots(figsize=(14, 4))
    axis.imshow(matrix, aspect="auto", origin="lower", interpolation="nearest")
    axis.set_ylabel("coefficient index")
    axis.set_xlabel("frame index")
    figure.colorbar(axis.images[0], ax=axis)
    figure.tight_layout()
    return figure


def accelerometry_overview(
    timestamps_and_frequencies_and_magnitudes,
    hive_number,
    queen_state_hint=False,
):
    timestamps, frequencies, magnitudes = timestamps_and_frequencies_and_magnitudes
    cmap_blue_red = matplotlib.colors.LinearSegmentedColormap.from_list("blue_red", ["#08306b", "#d73027"])

    times_as_float = timestamps.astype('datetime64[m]').astype(float)
    start_date = timestamps[0].astype('datetime64[D]')
    end_date = timestamps[-1].astype('datetime64[D]') + 1

    figure, axis = matplotlib.pyplot.subplots(figsize=(18, 6))

    day = start_date
    while day <= end_date:
        night_start = (day - numpy.timedelta64(1, 'D')).astype('datetime64[m]').astype(float) + 18 * 60
        night_end = day.astype('datetime64[m]').astype(float) + 6 * 60
        axis.axvspan(night_start, night_end, color='navy', alpha=0.08)
        day += numpy.timedelta64(1, 'D')

    vmin, vmax = numpy.percentile(magnitudes, [5, 95])

    scatter = axis.scatter(
        times_as_float,
        frequencies,
        c=magnitudes,
        s=1,
        cmap=cmap_blue_red,
        vmin=vmin,
        vmax=vmax,
        alpha=0.5,
    )
    figure.colorbar(scatter, ax=axis, label='magnitude')

    event_styles = {'removed': ('red', '--'), 'introduced': ('green', '--')}
    if hive_number in times.queenless_period_per_hive:
        start, end = times.queenless_period_per_hive[hive_number]
        if start is not None:
            x_position = numpy.datetime64(start, 'm').astype(float)
            color, linestyle = event_styles['removed']
            axis.axvline(x_position, color=color, linestyle=linestyle, linewidth=1.5, alpha=0.8,
                         label=f'queen removed ({start})')
        if end is not None:
            x_position = numpy.datetime64(end, 'm').astype(float)
            color, linestyle = event_styles['introduced']
            axis.axvline(x_position, color=color, linestyle=linestyle, linewidth=1.5, alpha=0.8,
                         label=f'queen reintroduce ({end})')
        axis.legend(loc='upper right')

    axis.set_ylabel('frequency (hertz)')
    axis.set_xlabel('time')
    axis.set_title(f'hive {hive_number} — accelerometer fast Fourier transform peaks over time')

    tick_dates = numpy.arange(start_date, end_date + 1, numpy.timedelta64(1, 'D'))
    tick_positions = tick_dates.astype('datetime64[m]').astype(float)
    tick_labels = [f"{str(d)[8:10]}/{str(d)[5:7]}/{str(d)[2:4]}" for d in tick_dates]
    axis.set_xticks(tick_positions)
    axis.set_xticklabels(tick_labels, rotation=45)

    figure.tight_layout()
    return figure


def magnitudes_over_frequencies_from_accelerometry(
    timestamps_and_frequencies_and_magnitudes,
    hive_number,
):
    _timestamps, frequencies, magnitudes = timestamps_and_frequencies_and_magnitudes
    figure, axis = matplotlib.pyplot.subplots(figsize=(14, 6))

    axis.scatter(
        frequencies,
        magnitudes,
        s=0.5,
        alpha=0.1,
        color='black',
    )
    axis.set_yscale('log')
    axis.set_xlabel('frequency (Hz)')
    axis.set_ylabel('magnitude (log)')
    axis.set_title(f'hive {hive_number} — frequency vs magnitude distribution')

    figure.tight_layout()
    return figure


def magnitudes_over_frequencies_by_rank_from_triple_accelerometry(
    timestamps_and_triple_frequencies_and_triple_magnitudes,
    hive_number,
):
    timestamps, triple_frequencies, triple_magnitudes = timestamps_and_triple_frequencies_and_triple_magnitudes

    figure, axes = matplotlib.pyplot.subplots(1, 3, figsize=(18, 5), sharey=True, sharex=True)

    colors = ['red', 'green', 'blue']
    labels = ['peak 1', 'peak 2', 'peak 3']

    for rank, axis in enumerate(axes):
        axis.scatter(
            triple_frequencies[:, rank],
            triple_magnitudes[:, rank],
            s=0.5,
            alpha=0.1,
            color=colors[rank],
        )
        axis.set_yscale('log')
        axis.set_xlabel('frequency (Hz)')
        axis.set_title(labels[rank])

    axes[0].set_ylabel('magnitude (log)')
    axes[-1].set_title(f'hive {hive_number} — frequency vs magnitude by peak rank')

    figure.tight_layout()
    return figure


def histogram_from_accelerometry(
    timestamps_and_frequencies_and_magnitudes,
    hive_number,
):
    timestamps, frequencies, magnitudes = timestamps_and_frequencies_and_magnitudes

    figure, axis = matplotlib.pyplot.subplots(figsize=(14, 6))

    axis.hist(
        frequencies,
        bins=500,
        color='black',
        edgecolor='none',
    )
    axis.set_yscale('log')
    axis.set_xlabel('frequency (Hz)')
    axis.set_ylabel('count')
    axis.set_title(f'hive {hive_number} — distribution of FFT peaks')

    figure.tight_layout()
    return figure


def magnitude_histogram_from_accelerometry(
    timestamps_and_frequencies_and_magnitudes,
    hive_number,
):
    timestamps, frequencies, magnitudes = timestamps_and_frequencies_and_magnitudes

    figure, axis = matplotlib.pyplot.subplots(figsize=(14, 6))

    axis.hist(
        magnitudes,
        bins=500,
        color='black',
        edgecolor='none',
    )
    axis.set_yscale('log')
    axis.set_xlabel('magnitude')
    axis.set_ylabel('count')
    axis.set_title(f'hive {hive_number} — frequency distribution of FFT peaks')

    figure.tight_layout()
    return figure


def curves(day_value_pairs, curve_names, title, ylabel="", queen_state_hint=False):
    COLORS = [
        'steelblue', 'darkorange', 'seagreen', 'crimson',
        'mediumpurple', 'goldenrod', 'teal', 'hotpink',
        'slategray', 'limegreen', 'dodgerblue',
    ]

    figure, axis = matplotlib.pyplot.subplots(figsize=(16, 5))

    for (days, values), name, color in zip(day_value_pairs, curve_names, COLORS):
        days_as_float = days.astype('datetime64[D]').astype(float)
        axis.plot(
            days_as_float, values,
            color=color, linewidth=1.5, marker='o', markersize=3, alpha=0.8,
            label=name,
        )

    if queen_state_hint:
        for hive_number, events in times.QUEEN_EVENTS.items():
            event_styles = {'removed': ('red', '--'), 'introduced': ('green', '--')}
            for date_string, event_type in events:
                x_position = numpy.datetime64(date_string, 'D').astype(float)
                color, linestyle = event_styles[event_type]
                axis.axvline(x_position, color=color, linestyle=linestyle, linewidth=1.5, alpha=0.8,
                             label=f'hive {hive_number} queen {event_type} ({date_string})')

    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.legend()
    tick_labels = [str(d)[:10] for d in days]
    axis.set_xticks(days_as_float)
    axis.set_xticklabels(tick_labels, rotation=45, ha='right')

    figure.tight_layout()
    return figure

def spectrogram_from_filepath(
    filepath,
    title=None,
    log_scale=False,
    notable_frequencies=True,
):
    samples, sample_rate = soundfile.read(filepath)
    stft = librosa.stft(samples)
    spectrogram_db = librosa.amplitude_to_db(numpy.abs(stft), ref=numpy.max)

    figure = matplotlib.pyplot.figure(figsize=(20, 5))

    if notable_frequencies:
        COLOR_BANDPASS     = "#00FFCC"
        COLOR_CARRIER      = "#FFFFFF"
        COLOR_WARBLE       = "#66FF66"
        COLOR_NON_SWARM    = "#00CED1"
        COLOR_QUEEN_WORKER = "#ADFF2F"
        COLOR_PRE_SWARM    = "#7FFFD4"

        notable_hz_entries = [
            (100, "bandpass lower boundary",        False, COLOR_BANDPASS),
            (225, "worker warble lower edge",       False, COLOR_WARBLE),
            (250, "wing-beat carrier center",       True,  COLOR_CARRIER),
            (285, "worker warble upper edge",       False, COLOR_WARBLE),
            (300, "non-swarm dominant upper edge",  False, COLOR_NON_SWARM),
            (400, "queen/worker peaks lower edge",  False, COLOR_QUEEN_WORKER),
            (500, "pre-swarm dominant lower edge",  False, COLOR_PRE_SWARM),
            (550, "queen/worker peaks upper edge",  False, COLOR_QUEEN_WORKER),
            (600, "pre-swarm dominant upper edge",  False, COLOR_PRE_SWARM),
        ]

        legend_entries = [
            ("500–600 Hz — pre-swarm dominant",    COLOR_PRE_SWARM),
            ("400–550 Hz — queen/worker peaks",    COLOR_QUEEN_WORKER),
            ("300 Hz — non-swarm dominant upper",  COLOR_NON_SWARM),
            ("225–285 Hz — worker warble range",   COLOR_WARBLE),
            ("250 Hz — wing-beat carrier center",  COLOR_CARRIER),
            ("100 Hz — bandpass lower boundary",   COLOR_BANDPASS),
        ]

        grid = matplotlib.gridspec.GridSpec(1, 2, figure=figure, width_ratios=[0.015, 1], wspace=0.15)
        grid.update(left=0.04, right=0.93)
    else:
        grid = matplotlib.gridspec.GridSpec(1, 2, figure=figure, width_ratios=[0.015, 1], wspace=0.15)
        grid.update(left=0.04, right=0.95)

    cax = figure.add_subplot(grid[0, 0])
    axis = figure.add_subplot(grid[0, 1])

    image = librosa.display.specshow(
        spectrogram_db,
        sr=sample_rate,
        x_axis="time",
        y_axis="log" if log_scale else "hz",
        ax=axis,
    )
    axis.set_title(title or filepath)
    axis.set_ylim(1 if log_scale else 0, 1024)

    if notable_frequencies:
        label_transform = matplotlib.transforms.blended_transform_factory(axis.transAxes, axis.transData)

        for hz, _label, is_center, color in notable_hz_entries:
            axis.axhline(
                hz,
                xmin=0.5, xmax=1.0,
                color=color,
                linewidth=2.0 if is_center else 1.5,
                linestyle="--",
                alpha=0.95 if is_center else 0.75,
            )
            axis.text(
                1.01, hz, f"{hz} Hz",
                transform=label_transform,
                fontsize=7,
                color="black",
                verticalalignment="center",
                fontweight="bold" if is_center else "normal",
                clip_on=False,
            )

        legend_handles = [
            matplotlib.lines.Line2D([0], [0], color=color, linewidth=2, linestyle="--", label=name)
            for name, color in legend_entries
        ]
        axis.legend(
            handles=legend_handles,
            loc="upper right",
            fontsize=7,
            framealpha=1.0,
            facecolor="white",
            edgecolor="black",
            fancybox=True,
        )

    figure.colorbar(image, cax=cax, format="%+2.0f dB")

    figure.tight_layout()
    return figure
