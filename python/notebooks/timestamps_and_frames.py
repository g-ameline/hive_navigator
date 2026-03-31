import os
import soundfile
import numpy
from pipe import where, select, chain, Pipe
from datetime import datetime

import paths

def hourly_audio_slices_from_filepath(filepath, sample_rate=16000):
    """Splits one FLAC file at hour boundaries.
    Yields (hour_timestamp, audio_array) — one or two per file."""
    print(f'proessing {filepath}')
    _, timestamp_str = paths.metadata_from_filepath(filepath)
    file_start = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
    audio, _ = soundfile.read(filepath, dtype="float32")

    seconds_into_hour = file_start.minute * 60 + file_start.second
    samples_to_next_hour = (3600 - seconds_into_hour) * sample_rate
    hour_timestamp = numpy.datetime64(
        file_start.replace(minute=0, second=0, microsecond=0)
    )

    if samples_to_next_hour >= len(audio):
        yield hour_timestamp, audio
    else:
        yield hour_timestamp, audio[:samples_to_next_hour]
        yield hour_timestamp + numpy.timedelta64(1, "h"), audio[samples_to_next_hour:]


def filtered_flac_filepaths_from_folderpath(folderpath, date_range, hour_range):
    def file_hour_minutes_from_filepath(filepath):
        _, timestamp_str = paths.metadata_from_filepath(filepath)
        return int(timestamp_str[9:11]) * 60 + int(timestamp_str[11:13])

    def is_file_in_date_range(filepath):
        _, timestamp_str = paths.metadata_from_filepath(filepath)
        file_start = numpy.datetime64(
            datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
        )
        margin = numpy.timedelta64(1, "h")
        return (
            file_start + margin >= numpy.datetime64(date_range[0])
            and file_start - margin <= numpy.datetime64(date_range[1])
        )

    def is_file_in_hour_range(filepath):
        file_minutes = file_hour_minutes_from_filepath(filepath)
        start_h, start_m = (int(x) for x in hour_range[0].split(":"))
        end_h, end_m = (int(x) for x in hour_range[1].split(":"))
        start_minutes = start_h * 60 + start_m
        end_minutes = end_h * 60 + end_m
        margin = 35
        if start_minutes <= end_minutes:
            return file_minutes + margin >= start_minutes and file_minutes <= end_minutes
        return file_minutes + margin >= start_minutes or file_minutes <= end_minutes

    flac_files = (
        paths.sorted_filepaths_from_folderpath(folderpath)
        | where(lambda filepath: filepath.endswith(".flac"))
    )
    if date_range is not None:
        flac_files = flac_files | where(is_file_in_date_range)
    if hour_range is not None:
        flac_files = flac_files | where(is_file_in_hour_range)
    return flac_files


def hourly_audio_from_hive_folderpath(
    folderpath,
    date_range=None,
    hour_range=None,
):
    """Yields (hour_timestamp, audio_1d_array) — one per hour of recording."""
    flac_files = filtered_flac_filepaths_from_folderpath(
        folderpath, date_range, hour_range,
    )

    current_hour = None
    chunks = []

    for filepath in flac_files:
        for hour, audio_slice in (filepath | hourly_audio_slices_from_filepath):
            if current_hour is not None and hour != current_hour:
                yield current_hour, numpy.concatenate(chunks)
                chunks = []
            current_hour = hour
            chunks.append(audio_slice)

    if chunks:
        yield current_hour, numpy.concatenate(chunks)
