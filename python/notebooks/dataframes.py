import os
import numpy
import pandas

import paths
import times

from pipe import Pipe


def from_filepath(filepath):
    return pandas.read_csv(filepath, parse_dates=["timestamp"])


def flattened_from_feature_dict(features):
    flat = {}
    for key, value in features.items():
        if isinstance(value, numpy.ndarray):
            for i, v in enumerate(value):
                flat[f"{key}_{i}"] = float(v)
        else:
            flat[key] = value
    return flat


def feature_columns_from_dataframe(dataframe):
    skip = {"timestamp", "hive", "time_slice"}
    return [
        c for c in dataframe.columns
        if c not in skip
        and not isinstance(dataframe[c].iloc[0], numpy.ndarray)
    ]


@Pipe
def dataframe_from_hourly_stream(stream, hive_number):
    rows = []
    for hour_timestamp, feature_dict in stream:
        hour = int(
            (hour_timestamp - hour_timestamp.astype("datetime64[D]"))
            .astype("timedelta64[h]")
            .astype(int)
        )
        rows.append({
            "timestamp": hour_timestamp,
            "hive": paths.hive_name_from_number(hive_number),
            "time_slice": times.slice_label_from_hour(hour),
            **feature_dict,
        })
    yield pandas.DataFrame(rows)


def saved_csv_filepath_from_features_dataframe(
    dataframe,
    dataframe_filename,
    save_index=False,
):
    if not os.path.exists(paths.features_folderpath):
        print(f"\ncreating data_folder: {paths.features_folderpath}")
        os.makedirs(paths.features_folderpath)
    filepath = os.path.join(paths.features_folderpath, dataframe_filename)
    dataframe.to_csv(filepath, index=save_index)
    return filepath


def zscored_dataframe_from_dataframe(dataframe, baseline_end="2026-03-09"):
    feature_columns = feature_columns_from_dataframe(dataframe)
    baseline = dataframe[dataframe["timestamp"] < numpy.datetime64(baseline_end)]
    assert len(baseline) > 0, "no baseline rows before " + baseline_end

    baseline_stats = (
        baseline
        .groupby(["hive", "time_slice"])[feature_columns]
        .agg(["mean", "std"])
    )

    result = dataframe.copy()
    for column in feature_columns:
        means = dataframe.set_index(["hive", "time_slice"]).index.map(
            lambda key: baseline_stats.loc[key, (column, "mean")]
            if key in baseline_stats.index else numpy.nan
        )
        stds = dataframe.set_index(["hive", "time_slice"]).index.map(
            lambda key: baseline_stats.loc[key, (column, "std")]
            if key in baseline_stats.index else numpy.nan
        )
        result[f"{column}_zscore"] = (
            (dataframe[column].values - means.values)
            / (stds.values + 1e-10)
        )

    return result


def queenstate_from_row(row):
    hive_number = int(row["hive"].split("_")[1])
    period = times.queenless_period_per_hive.get(hive_number, [])
    if not period:
        return 'queenright'
    start, end = period
    start_date = numpy.datetime64('0001-01-01') if start == None else pandas.Timestamp(start)
    end_date = numpy.datetime64('9999-12-31') if end == None else pandas.Timestamp(end)

    if start_date <= row["timestamp"] <= end_date:
        return 'queenless'
    else:
        return 'queenright'

def numeric_columns_from_dataframe(dataframe, excluded_columns = {"timestamp", "hive", "time_slice", "queenlessness"} ):
    return [
        column for column in dataframe.select_dtypes(include="number").columns
        if column not in excluded_columns and not column.startswith("Unnamed")
    ]
def zscored_dataframe_from_dataframe_and_baseline(dataframe, baseline_dataframe):
    numeric_columns = numeric_columns_from_dataframe(baseline_dataframe)

    stats_per_slice = {
        time_slice: {
            "mean": group[numeric_columns].mean().values,
            "std": group[numeric_columns].std().values,
        }
        for time_slice, group in baseline_dataframe.groupby("time_slice")
    }

    result = numpy.full((len(dataframe), len(numeric_columns)), numpy.nan)

    for time_slice, group in dataframe.groupby("time_slice"):
        assert time_slice in stats_per_slice, f"no baseline stats for time_slice {time_slice}"
        stats = stats_per_slice[time_slice]
        row_indices = dataframe.index.get_indexer(group.index)
        result[row_indices] = (
            (group[numeric_columns].values - stats["mean"])
            / (stats["std"] + 1e-10)
        )

    return pandas.DataFrame(result, index=dataframe.index, columns=numeric_columns)
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist


def reordered_columns_from_dataframe(dataframe, metric="correlation", method="average"):
    indices = leaves_list(linkage(pdist(dataframe.values.T, metric=metric), method=method))
    return dataframe.iloc[:, indices]

