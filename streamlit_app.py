# app.py
import sys
import os

_app_dir = os.path.dirname(os.path.abspath(__file__))
_notebooks_dir = os.path.join(_app_dir, "python", "notebooks")
sys.path.insert(0, _notebooks_dir)
os.chdir(_notebooks_dir)

import streamlit
import pandas
import numpy
from matplotlib import pyplot

import dataframes
import anomalies
import paths
import times

from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


# ── constants ────────────────────────────────────────────────────────

DETECTORS = {
    "One-Class Support Vector Machine": lambda: OneClassSVM(),
    "Isolation Forest": lambda: IsolationForest(random_state=42),
    "Local Outlier Factor": lambda: LocalOutlierFactor(novelty=True),
}

METADATA_COLUMNS = ["timestamp", "hive", "time_slice", "queenlessness"]

DEFAULT_TIME_SLICES = ["11-12", "12-13", "13-14"]


# ── helpers ──────────────────────────────────────────────────────────

def preselected_feature_columns_from_outliers_and_inlier(
    outlier_dataframes, inlier_dataframe,
):
    inlier_zscores = dataframes.zscored_dataframe_from_dataframe_and_baseline(
        inlier_dataframe, inlier_dataframe,
    )
    inlier_means = inlier_zscores.mean()
    anomalous = set(inlier_zscores.columns)
    for outlier_dataframe in outlier_dataframes:
        if len(outlier_dataframe) == 0:
            continue
        outlier_zscores = dataframes.zscored_dataframe_from_dataframe_and_baseline(
            outlier_dataframe, inlier_dataframe,
        )
        sign_product = outlier_zscores.mean() * inlier_means
        divergent = set(sign_product[sign_product < 0].index)
        anomalous = anomalous.intersection(divergent)
    return sorted(anomalous)


# ── data loading (cached) ────────────────────────────────────────────

@streamlit.cache_data
def loaded_all_features():
    dataframe = dataframes.from_filepath(paths.all_merged_features_filepath)
    dataframe = dataframe.drop_duplicates(
        subset=["timestamp", "time_slice", "hive"], keep="first",
    )
    return dataframe.dropna()


# ── page config & introduction ───────────────────────────────────────

streamlit.set_page_config(layout="wide", page_title="Hive Anomaly Explorer")
streamlit.title("🐝 Hive Anomaly Explorer")

streamlit.markdown("""
This tool helps you explore whether a beehive that lost its queen sounds
and behaves differently from healthy, queenright hives.

**How it works:** a model is trained exclusively on data from queenright hives —
this is the "normal" baseline. Every hour of recording is then scored by the model:
a low anomaly score means the hive sounds unusual compared to that baseline,
which may indicate something is wrong.

**What you will see:**

1. **Discrimination plots** — do the queenless scores actually look different
   from the baseline? The further apart the two distributions, the stronger
   the signal.
2. **Z-score mosaic** — an hour-by-hour heatmap showing which audio features
   deviate from normal and when. This helps pinpoint the timing and nature
   of the change.

Use the sidebar on the left to configure the analysis.
""")

streamlit.divider()


# ── load data ────────────────────────────────────────────────────────

all_features = loaded_all_features()


# ── sidebar: data selection ──────────────────────────────────────────

streamlit.sidebar.header("Data selection")

available_time_slices = sorted(all_features["time_slice"].unique())

selected_time_slices = streamlit.sidebar.multiselect(
    "Hour slices to analyse",
    available_time_slices,
    default=[s for s in DEFAULT_TIME_SLICES if s in available_time_slices],
    help="Midday slices (11–14 h) tend to carry the strongest bee activity signal.",
)

if not selected_time_slices:
    streamlit.warning("Select at least one time slice to proceed.")
    streamlit.stop()

filtered_features = all_features[
    all_features["time_slice"].isin(selected_time_slices)
]

queenright_dataframe = filtered_features[
    filtered_features["queenlessness"] == False
]
queenless_03_dataframe = filtered_features[
    (filtered_features["queenlessness"] == True)
    & (filtered_features["hive"] == "hive_03")
]
queenless_04_dataframe = filtered_features[
    (filtered_features["queenlessness"] == True)
    & (filtered_features["hive"] == "hive_04")
]

investigated_hive = streamlit.sidebar.selectbox(
    "Queenless hive to investigate",
    ["hive_04", "hive_03", "both"],
    help=(
        "Hive 04 was queenless 9–12 March, then received a queen. "
        "Hive 03 became queenless from 12 March onward."
    ),
)

match investigated_hive:
    case "hive_03":
        queenless_dataframe = queenless_03_dataframe
    case "hive_04":
        queenless_dataframe = queenless_04_dataframe
    case "both":
        queenless_dataframe = filtered_features[
            filtered_features["queenlessness"] == True
        ]


# ── sidebar: features ───────────────────────────────────────────────

streamlit.sidebar.header("Features")

all_numeric_columns = sorted(
    dataframes.numeric_columns_from_dataframe(all_features),
)

default_features = preselected_feature_columns_from_outliers_and_inlier(
    [queenless_03_dataframe, queenless_04_dataframe],
    queenright_dataframe,
)

selected_features = streamlit.sidebar.multiselect(
    "Features to include",
    all_numeric_columns,
    default=[f for f in default_features if f in all_numeric_columns],
    help=(
        "Default pre-selection keeps only features whose z-scores diverge "
        "in opposite directions between queenright and queenless data."
    ),
)

if not selected_features:
    streamlit.warning("Select at least one feature to proceed.")
    streamlit.stop()

keep_columns = [
    c for c in METADATA_COLUMNS if c in queenright_dataframe.columns
] + selected_features

queenright_selected = queenright_dataframe[keep_columns]
queenless_selected = queenless_dataframe[keep_columns]


# ── sidebar: anomaly detector ────────────────────────────────────────

streamlit.sidebar.header("Anomaly detector")

detector_name = streamlit.sidebar.selectbox(
    "Algorithm",
    list(DETECTORS.keys()),
)

streamlit.sidebar.metric("Queenright samples", len(queenright_selected))
streamlit.sidebar.metric("Queenless samples", len(queenless_selected))


# ── scoring ──────────────────────────────────────────────────────────

@streamlit.cache_resource
def fitted_scorers_from_parameters(
    _queenright_dataframe,
    detector_name,
    features_key,
    slices_key,
):
    detector = DETECTORS[detector_name]()
    return anomalies.scorers_from_inliers_dataframe_and_detector(
        _queenright_dataframe, detector,
    )


scorers = fitted_scorers_from_parameters(
    queenright_selected,
    detector_name,
    tuple(selected_features),
    tuple(selected_time_slices),
)

queenright_scores = anomalies.anomaly_scores_from_dataframe_and_scorers(
    queenright_selected, scorers,
)
queenless_scores = anomalies.anomaly_scores_from_dataframe_and_scorers(
    queenless_selected, scorers,
)


def anomaly_scores_from_dataframe(dataframe):
    return anomalies.anomaly_scores_from_dataframe_and_scorers(dataframe, scorers)


# ── discrimination ───────────────────────────────────────────────────

streamlit.header("Population discrimination")

discrimination = anomalies.discrimination_from_scores_and_scores(
    queenless_scores, queenright_scores,
)

metric_columns = streamlit.columns(4)
metric_columns[0].metric(
    "Area under curve",
    f"{discrimination['area_under_curve']:.3f}",
)
metric_columns[1].metric(
    "Cohen's d",
    f"{discrimination['cohens_d']:.2f}",
)
metric_columns[2].metric(
    "Investigated median",
    f"{discrimination['investigated_median']:.3f}",
)
metric_columns[3].metric(
    "Baseline median",
    f"{discrimination['baseline_median']:.3f}",
)

discrimination_figure = (
    anomalies.discrimination_figure_from_investigated_scores_and_baseline_scores(
        queenless_scores, queenright_scores,
    )
)
streamlit.pyplot(discrimination_figure)
pyplot.close(discrimination_figure)

streamlit.divider()


# ── sidebar: mosaic parameters ───────────────────────────────────────

streamlit.sidebar.header("Mosaic parameters")

baseline_aggregation = streamlit.sidebar.selectbox(
    "Baseline aggregation",
    ["mean", "worst", "best", "furthest"],
    help=(
        "How to summarise the queenright baseline for each time slot. "
        "'mean' averages all queenright hives; "
        "'worst' picks the most anomalous hive at each hour; "
        "'best' picks the least anomalous; "
        "'furthest' picks the hive most different from the investigated one."
    ),
)

observation_period = streamlit.sidebar.selectbox(
    "Observation period",
    ["join", "intersection", "investigated", "baseline"],
    help=(
        "'join' shows the full date range of both datasets; "
        "'intersection' shows only overlapping dates; "
        "'investigated' restricts to the queenless period; "
        "'baseline' restricts to the queenright period."
    ),
)


# ── mosaic ───────────────────────────────────────────────────────────

streamlit.header("Z-score mosaic")

mosaic_figure = anomalies.investigate_anomaly(
    investigated_features_dataframe=queenless_selected,
    baseline_features_dataframe=queenright_selected,
    baseline_aggregation=baseline_aggregation,
    observation_period=observation_period,
    anomaly_scores_from_dataframe=anomaly_scores_from_dataframe,
    plotly=True,
)
streamlit.plotly_chart(mosaic_figure, use_container_width=True)


# ── export ───────────────────────────────────────────────────────────

streamlit.divider()
streamlit.header("Export")

export_dataframe = pandas.DataFrame({
    "timestamp": queenless_selected["timestamp"].values,
    "hive": queenless_selected["hive"].values,
    "anomaly_score": queenless_scores.values,
})

streamlit.download_button(
    "Download queenless anomaly scores (CSV)",
    export_dataframe.to_csv(index=False),
    file_name="anomaly_scores.csv",
    mime="text/csv",
)
