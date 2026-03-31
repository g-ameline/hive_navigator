# app.py
import sys
import os

_app_dir = os.path.dirname(os.path.abspath(__file__))
_notebooks_dir = os.path.join(_app_dir, "python", "notebooks")
sys.path.insert(0, _notebooks_dir)
os.chdir(_notebooks_dir)

# import sys
# sys.path.insert(0, "python/notebooks")
# import os
# os.chdir("python/notebooks")
import streamlit
import pandas
import numpy
from matplotlib import pyplot

import dataframes
import anomalies
import paths

from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


# ── data loading (cached) ────────────────────────────────────────────

@streamlit.cache_data
def all_features_dataframe():
    return dataframes.from_filepath(paths.all_merged_features_filepath)


# ── sidebar controls ─────────────────────────────────────────────────

streamlit.set_page_config(layout="wide", page_title="Hive Anomaly Explorer")
streamlit.title("Hive Anomaly Explorer")

all_features = all_features_dataframe()

DETECTORS = {
    "One-Class Support Vector Machine": lambda: OneClassSVM(),
    "Isolation Forest": lambda: IsolationForest(random_state=42),
    "Local Outlier Factor": lambda: LocalOutlierFactor(novelty=True),
}

detector_name = streamlit.sidebar.selectbox(
    "Anomaly detector",
    list(DETECTORS.keys()),
)

investigated_hive = streamlit.sidebar.selectbox(
    "Queenless hive to investigate",
    ["hive_03", "hive_04", "both"],
)

aggregation_method = streamlit.sidebar.selectbox(
    "Baseline aggregation",
    ["mean", "median"],
)

# ── hour-of-day filter ───────────────────────────────────────────────

hour_range = streamlit.sidebar.slider(
    "Hour-of-day range",
    min_value=0,
    max_value=23,
    value=(0, 23),
)

# ── observation period ───────────────────────────────────────────────

all_dates = pandas.to_datetime(all_features["timestamp"]).dt.date
min_date, max_date = all_dates.min(), all_dates.max()

date_range = streamlit.sidebar.date_input(
    "Observation period",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

# ── feature subset ───────────────────────────────────────────────────

metadata_columns = {"timestamp", "hive", "queenlessness"}
feature_columns = sorted(set(all_features.columns) - metadata_columns)

selected_features = streamlit.sidebar.multiselect(
    "Features to include",
    feature_columns,
    default=feature_columns,
)

assert len(selected_features) > 0, "Select at least one feature."

# ── filtering ────────────────────────────────────────────────────────

def filtered_dataframe_from_dataframe(dataframe):
    timestamps = pandas.to_datetime(dataframe["timestamp"])
    hour_mask = timestamps.dt.hour.between(hour_range[0], hour_range[1])
    assert len(date_range) == 2, "Select a start and end date."
    date_mask = timestamps.dt.date.between(date_range[0], date_range[1])
    return dataframe[hour_mask & date_mask]


filtered_features = filtered_dataframe_from_dataframe(all_features)

queenright = filtered_features[filtered_features["queenlessness"] == False]

match investigated_hive:
    case "hive_03":
        queenless = filtered_features[
            (filtered_features["queenlessness"] == True)
            & (filtered_features["hive"] == "hive_03")
        ]
    case "hive_04":
        queenless = filtered_features[
            (filtered_features["queenlessness"] == True)
            & (filtered_features["hive"] == "hive_04")
        ]
    case "both":
        queenless = filtered_features[filtered_features["queenlessness"] == True]

streamlit.sidebar.metric("Queenright samples", len(queenright))
streamlit.sidebar.metric("Queenless samples", len(queenless))

# ── scoring ──────────────────────────────────────────────────────────

keep_columns = list(metadata_columns & set(queenright.columns)) + selected_features

@streamlit.cache_resource
def scores_from_queenright_and_queenless_and_detector_name(
    _queenright, _queenless, _detector_name, _selected_features,
):
    detector = DETECTORS[_detector_name]()
    scorers = anomalies.scorers_from_inliers_dataframe_and_detector(
        _queenright[_selected_features + ["timestamp", "hive", "queenlessness"]],
        detector,
    )
    queenright_scores = anomalies.anomaly_scores_from_dataframe_and_scorers(
        _queenright[_selected_features + ["timestamp", "hive", "queenlessness"]],
        scorers,
    )
    queenless_scores = anomalies.anomaly_scores_from_dataframe_and_scorers(
        _queenless[_selected_features + ["timestamp", "hive", "queenlessness"]],
        scorers,
    )
    return queenright_scores, queenless_scores


queenright_scores, queenless_scores = scores_from_queenright_and_queenless_and_detector_name(
    queenright, queenless, detector_name, selected_features,
)

# ── histograms ───────────────────────────────────────────────────────

col_left, col_right = streamlit.columns(2)

with col_left:
    streamlit.subheader("Queenright score distribution")
    fig_qr, ax_qr = pyplot.subplots()
    ax_qr.hist(queenright_scores, bins=50, alpha=0.7)
    streamlit.pyplot(fig_qr)

with col_right:
    streamlit.subheader(f"Queenless score distribution ({investigated_hive})")
    fig_ql, ax_ql = pyplot.subplots()
    ax_ql.hist(queenless_scores, bins=50, alpha=0.7, color="orange")
    streamlit.pyplot(fig_ql)

# ── discrimination ───────────────────────────────────────────────────

streamlit.subheader("Population discrimination")

discrimination = anomalies.discrimination_from_scores_and_scores(
    queenless_scores, queenright_scores,
)
streamlit.metric("Discrimination (area under the curve)", f"{discrimination['area_under_curve']:.3f}")

fig_disc = anomalies.discrimination_figure_from_investigated_scores_and_baseline_scores(
    queenless_scores, queenright_scores,
)
streamlit.pyplot(fig_disc)

# ── mosaic ───────────────────────────────────────────────────────────

streamlit.subheader("Z-score mosaic")

baseline_zscores = dataframes.zscored_dataframe_from_dataframe_and_baseline(
    queenright, queenright,
)
investigated_zscores = dataframes.zscored_dataframe_from_dataframe_and_baseline(
    queenless, queenright,
)

baseline_zscores = dataframes.reordered_columns_from_dataframe(baseline_zscores)
investigated_zscores = investigated_zscores[baseline_zscores.columns]

fig_mosaic = anomalies.mosaic_from_zscored_dataframes_and_scores(
    investigated_zscores=investigated_zscores,
    investigated_anomaly_scores=queenless_scores,
    investigated_timestamps=queenless["timestamp"],
    baseline_zscores=baseline_zscores,
    baseline_anomaly_scores=queenright_scores,
    baseline_timestamps=queenright["timestamp"],
    baseline_hives=queenright["hive"],
    aggregation_method=aggregation_method,
)
streamlit.pyplot(fig_mosaic)

# ── download ─────────────────────────────────────────────────────────

streamlit.subheader("Export")

export_dataframe = pandas.DataFrame({
    "timestamp": queenless["timestamp"].values,
    "hive": queenless["hive"].values,
    "anomaly_score": queenless_scores,
})

streamlit.download_button(
    "Download queenless anomaly scores (csv)",
    export_dataframe.to_csv(index=False),
    file_name="anomaly_scores.csv",
    mime="text/csv",
)
