from scipy.stats import mannwhitneyu


def discrimination_from_scores_and_scores(investigated_scores, baseline_scores):
    investigated = numpy.asarray(investigated_scores)
    baseline = numpy.asarray(baseline_scores)

    u_statistic, p_value = mannwhitneyu(investigated, baseline, alternative="less")
    n_investigated = len(investigated)
    n_baseline = len(baseline)
    area_under_curve = 1.0 - u_statistic / (n_investigated * n_baseline)

    pooled_std = numpy.concatenate([investigated, baseline]).std()
    cohens_d = (baseline.mean() - investigated.mean()) / pooled_std

    return {
        "mann_whitney_u": u_statistic,
        "p_value": p_value,
        "area_under_curve": area_under_curve,
        "cohens_d": cohens_d,
        "investigated_median": numpy.median(investigated),
        "baseline_median": numpy.median(baseline),
    }


import sklearn.base
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler


def scorers_from_inliers_dataframe_and_detector(inliers_dataframe, detector):
    scorers = {}
    for time_slice, group in inliers_dataframe.groupby("time_slice"):
        numeric_columns = group.select_dtypes(include="number").columns
        scaler = StandardScaler().fit(group[numeric_columns])
        fitted_detector = sklearn.base.clone(detector).fit(scaler.transform(group[numeric_columns]))

        def scores_from_dataframe(dataframe, _s=scaler, _d=fitted_detector, _c=numeric_columns):
            return _d.score_samples(_s.transform(dataframe[_c]))

        scorers[time_slice] = scores_from_dataframe
    return scorers


def anomaly_scores_from_dataframe_and_scorers(dataframe, scorers):
    anomaly_scores = pandas.Series(numpy.nan, index=dataframe.index)
    for time_slice, group in dataframe.groupby("time_slice"):
        assert time_slice in scorers, f"no scorer for time_slice {time_slice}"
        anomaly_scores.loc[group.index] = scorers[time_slice](group)
    return anomaly_scores



import matplotlib.pyplot
import matplotlib.patches
import matplotlib.colors
import numpy
import pandas


FEATURE_GROUP_DIVERGING_COLORS = [
    ((0.12, 0.47, 0.71), (0.84, 0.15, 0.16)),
    ((0.17, 0.63, 0.17), (0.89, 0.47, 0.76)),
    ((0.58, 0.40, 0.74), (1.00, 0.50, 0.05)),
    ((0.09, 0.75, 0.81), (0.74, 0.74, 0.13)),
    ((0.55, 0.34, 0.29), (0.50, 0.50, 0.50)),
    ((0.84, 0.15, 0.16), (0.12, 0.47, 0.71)),
    ((1.00, 0.50, 0.05), (0.17, 0.63, 0.17)),
    ((0.89, 0.47, 0.76), (0.09, 0.75, 0.81)),
]

MISSING_GRAY = 0.92

_GROUP_CMAP_CACHE = {}


def feature_group_from_column_name(column_name):
    name = column_name
    parts = name.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        name = parts[0]
    if name.endswith("_mean"):
        return name.split("_")[0] + "_mean"
    elif name.endswith("_std"):
        return name.split("_")[0] + "_std"
    else:
        return name.split("_")[0]


def _diverging_cmap_from_colors(color_negative, color_positive):
    return matplotlib.colors.LinearSegmentedColormap.from_list(
        "custom", [color_negative, (1, 1, 1), color_positive],
    )


def cmap_for_group(group):
    if group not in _GROUP_CMAP_CACHE:
        index = len(_GROUP_CMAP_CACHE) % len(FEATURE_GROUP_DIVERGING_COLORS)
        neg, pos = FEATURE_GROUP_DIVERGING_COLORS[index]
        _GROUP_CMAP_CACHE[group] = _diverging_cmap_from_colors(neg, pos)
    return _GROUP_CMAP_CACHE[group]


def column_cmaps_from_columns(columns):
    groups = [feature_group_from_column_name(c) for c in columns]
    return [cmap_for_group(g) for g in groups]


def group_colored_image_from_zscores_and_column_cmaps(zscores_values, column_cmaps):
    nan_mask = numpy.isnan(zscores_values).any(axis=1)
    filled = numpy.nan_to_num(zscores_values, nan=0.0)
    mapped = numpy.clip(filled / 3.0, -1, 1) * 0.5 + 0.5
    rows, cols = mapped.shape
    image = numpy.empty((rows, cols, 3))
    for col_index, cmap in enumerate(column_cmaps):
        image[:, col_index, :] = cmap(mapped[:, col_index])[:, :3]
    image[nan_mask] = MISSING_GRAY
    return image


def full_timestamps_from_bounds_and_hours(start_date, end_date, hours):
    dates = pandas.date_range(
        pandas.Timestamp(start_date).normalize(),
        pandas.Timestamp(end_date).normalize(),
        freq="D",
    )
    return pandas.DatetimeIndex([
        d + pandas.Timedelta(hours=h)
        for d in dates
        for h in sorted(hours)
    ])


def reindexed_zscores_and_scores_from_originals_and_index(
    zscores, anomaly_scores, timestamps, full_index,
):
    zscores_by_time = zscores.copy()
    zscores_by_time.index = pandas.to_datetime(timestamps).values
    scores_by_time = pandas.Series(
        numpy.asarray(anomaly_scores),
        index=pandas.to_datetime(timestamps).values,
    )
    return zscores_by_time.reindex(full_index), scores_by_time.reindex(full_index)


def aggregated_panel_from_zscores_and_metadata(
    zscores, anomaly_scores, timestamps, hives, method, numeric_columns,
):
    grouped = pandas.DataFrame(zscores.values, columns=numeric_columns)
    grouped["timestamp"] = pandas.to_datetime(timestamps).values
    grouped["_anomaly_score"] = numpy.asarray(anomaly_scores)
    grouped["hive"] = numpy.asarray(hives)

    match method:
        case int():
            hive_label = f"hive_{method:02d}"
            mask = grouped["hive"] == hive_label
            result_zscores = grouped[mask].set_index("timestamp")[numeric_columns]
            result_anomaly_scores = grouped[mask].set_index("timestamp")["_anomaly_score"]

        case "mean":
            result_zscores = grouped.groupby("timestamp")[numeric_columns].mean()
            result_anomaly_scores = grouped.groupby("timestamp")["_anomaly_score"].mean()

        case "min":
            result_zscores = grouped.groupby("timestamp")[numeric_columns].min()
            result_anomaly_scores = grouped.groupby("timestamp")["_anomaly_score"].min()

        case "max":
            result_zscores = grouped.groupby("timestamp")[numeric_columns].max()
            result_anomaly_scores = grouped.groupby("timestamp")["_anomaly_score"].max()

        case "furthest":
            def furthest_row(group):
                return group.loc[group[numeric_columns].abs().max(axis=1).idxmax()]
            result = grouped.groupby("timestamp").apply(furthest_row, include_groups=False)
            result_zscores = result[numeric_columns]
            result_anomaly_scores = result["_anomaly_score"]

        case "worst":
            def worst_row(group):
                return group.loc[group["_anomaly_score"].idxmin()]
            result = grouped.groupby("timestamp").apply(worst_row, include_groups=False)
            result_zscores = result[numeric_columns]
            result_anomaly_scores = result["_anomaly_score"]

        case "best":
            def best_row(group):
                return group.loc[group["_anomaly_score"].idxmax()]
            result = grouped.groupby("timestamp").apply(best_row, include_groups=False)
            result_zscores = result[numeric_columns]
            result_anomaly_scores = result["_anomaly_score"]

        case _:
            assert False, f"unknown aggregation method: {method}"

    return result_zscores.sort_index(), result_anomaly_scores.sort_index()


def mosaic_from_zscored_dataframes_and_scores(
    investigated_zscores,
    investigated_anomaly_scores,
    baseline_zscores,
    baseline_anomaly_scores,
    timestamps,
    aggregation_label,
):
    numeric_columns = list(investigated_zscores.columns)
    column_cmaps = column_cmaps_from_columns(numeric_columns)

    all_valid_scores = numpy.concatenate([
        investigated_anomaly_scores.dropna().values,
        baseline_anomaly_scores.dropna().values,
    ])
    anomaly_score_vmin = all_valid_scores.min() if len(all_valid_scores) > 0 else -1
    anomaly_score_vmax = all_valid_scores.max() if len(all_valid_scores) > 0 else 1

    n_rows = len(timestamps)
    pixels_per_row = 0.15
    panel_height = n_rows * pixels_per_row
    total_height = max(16, panel_height * 2)
    width = max(14, len(numeric_columns) * 0.3)

    figure, axes = matplotlib.pyplot.subplots(
        nrows=2, ncols=2,
        figsize=(width, total_height),
        gridspec_kw={
            "width_ratios": [1, len(numeric_columns)],
            "height_ratios": [1, 1],
        },
    )

    (investigated_score_axis, investigated_features_axis) = axes[0]
    (baseline_score_axis, baseline_features_axis) = axes[1]

    purples_cmap = matplotlib.pyplot.get_cmap("Purples_r").copy()
    purples_cmap.set_bad(color=(MISSING_GRAY, MISSING_GRAY, MISSING_GRAY))

    def render_panel(score_axis, features_axis, zscores_full, scores_full, title):
        colored_image = group_colored_image_from_zscores_and_column_cmaps(
            zscores_full.values, column_cmaps,
        )
        features_axis.imshow(colored_image, aspect="auto", interpolation="nearest")
        features_axis.set_xticks(range(len(numeric_columns)))
        features_axis.set_xticklabels(numeric_columns, fontsize=6, rotation=90)
        column_groups = [feature_group_from_column_name(c) for c in numeric_columns]
        unique_groups = list(dict.fromkeys(column_groups))
        for tick_label, group in zip(features_axis.get_xticklabels(), column_groups):
            tick_label.set_color(cmap_for_group(group)(0.0)[:3])
        features_axis.set_title(title, fontsize=10)

        boundary_positions = [i - 0.5 for i in range(len(timestamps))]
        boundary_labels = [t.strftime("%H") for t in timestamps]
        features_axis.yaxis.tick_left()
        features_axis.yaxis.set_label_position("left")
        features_axis.set_yticks(boundary_positions)
        features_axis.set_yticklabels(boundary_labels, fontsize=7)

        masked_scores = numpy.ma.masked_invalid(scores_full.values.reshape(-1, 1))
        score_mappable = score_axis.imshow(
            masked_scores,
            aspect="auto", cmap=purples_cmap,
            vmin=anomaly_score_vmin, vmax=anomaly_score_vmax,
            interpolation="nearest",
        )
        score_axis.set_xticks([0])
        score_axis.set_xticklabels(["anomaly\nscore"], fontsize=8)

        valid_scores = scores_full.dropna().values
        if len(valid_scores) > 0:
            anomaly_threshold = numpy.percentile(valid_scores, 1)
            for i in range(len(scores_full)):
                if not numpy.isnan(scores_full.iloc[i]) and scores_full.iloc[i] < anomaly_threshold:
                    score_axis.add_patch(matplotlib.patches.Rectangle(
                        (-0.5, i - 0.5), 1, 1, linewidth=2, edgecolor="red", facecolor="none",
                    ))

        date_strings = [t.strftime("%d/%m") for t in timestamps]
        day_boundary_indices = [0] + [
            i for i in range(1, len(date_strings))
            if date_strings[i] != date_strings[i - 1]
        ]
        day_boundary_positions = [i - 0.5 for i in day_boundary_indices]
        score_axis.set_yticks(day_boundary_positions)
        score_axis.set_yticklabels(
            [date_strings[i] for i in day_boundary_indices],
            fontsize=9, fontweight="bold",
        )

        return score_mappable

    score_mappable = render_panel(
        investigated_score_axis, investigated_features_axis,
        investigated_zscores, investigated_anomaly_scores,
        "investigated data",
    )
    render_panel(
        baseline_score_axis, baseline_features_axis,
        baseline_zscores, baseline_anomaly_scores,
        f"baseline data (aggregation: {aggregation_label})",
    )

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    colorbar_axis = inset_axes(
        investigated_score_axis,
        width="100%", height="4%",
        loc="lower left",
        bbox_to_anchor=(0, 1.02, 1, 1),
        bbox_transform=investigated_score_axis.transAxes,
        borderpad=0,
    )
    figure.colorbar(score_mappable, cax=colorbar_axis, orientation="horizontal")
    colorbar_axis.set_xlabel("anomaly score", fontsize=7)
    colorbar_axis.xaxis.set_label_position("top")
    colorbar_axis.tick_params(labelsize=6)

    figure.subplots_adjust(hspace=0.15, wspace=0.05, top=0.98, bottom=0.05)
    return figure


import plotly.graph_objects
import plotly.subplots


def plotly_mosaic_from_zscored_dataframes_and_scores(
    investigated_zscores,
    investigated_anomaly_scores,
    baseline_zscores,
    baseline_anomaly_scores,
    timestamps,
    aggregation_label,
):
    numeric_columns = list(investigated_zscores.columns)
    timestamp_labels = [t.strftime("%m/%d %Hh") for t in timestamps]

    figure = plotly.subplots.make_subplots(
        rows=1, cols=4,
        column_widths=[0.01, 0.49, 0.01, 0.49],
        horizontal_spacing=0.005,
        subplot_titles=[
            "score", "investigated",
            "score", f"baseline ({aggregation_label})",
        ],
    )

    figure.add_trace(
        plotly.graph_objects.Heatmap(
            z=investigated_anomaly_scores.values.reshape(-1, 1),
            y=timestamp_labels,
            x=[""],
            xgap=0,
            colorscale="Purples_r",
            showscale=False,
            hovertemplate="score: %{z:.3f}<extra></extra>",
        ),
        row=1, col=1,
    )

    figure.add_trace(
        plotly.graph_objects.Heatmap(
            z=investigated_zscores.values,
            y=timestamp_labels,
            x=numeric_columns,
            colorscale="RdBu_r",
            zmid=0,
            zmin=-3,
            zmax=3,
            colorbar=dict(
                title=dict(text="z-score", side="top"),
                orientation="h",
            ),
            hovertemplate="%{x}<br>%{y}<br>z=%{z:.2f}<extra></extra>",
        ),
        row=1, col=2,
    )

    figure.add_trace(
        plotly.graph_objects.Heatmap(
            z=baseline_anomaly_scores.values.reshape(-1, 1),
            y=timestamp_labels,
            x=[""],
            xgap=0,
            colorscale="Purples_r",
            showscale=False,
            hovertemplate="score: %{z:.3f}<extra></extra>",
        ),
        row=1, col=3,
    )

    figure.add_trace(
        plotly.graph_objects.Heatmap(
            z=baseline_zscores.values,
            y=timestamp_labels,
            x=numeric_columns,
            colorscale="RdBu_r",
            zmid=0,
            zmin=-3,
            zmax=3,
            showscale=False,
            hovertemplate="%{x}<br>%{y}<br>z=%{z:.2f}<extra></extra>",
        ),
        row=1, col=4,
    )

    n_rows = len(timestamps)
    height = max(800, n_rows * 12)

    figure.update_layout(
        height=height,
        width=max(1200, len(numeric_columns) * 12),
        yaxis=dict(autorange="reversed"),
        yaxis2=dict(autorange="reversed", showticklabels=False),
        yaxis3=dict(autorange="reversed", showticklabels=False),
        yaxis4=dict(autorange="reversed", showticklabels=False),
        margin=dict(l=80, r=20, t=40, b=120),
    )

    return figure


from sklearn.metrics import roc_curve
from scipy.stats import gaussian_kde
import numpy
import matplotlib.pyplot


def discrimination_figure_from_investigated_scores_and_baseline_scores(
    investigated_anomaly_scores,
    baseline_anomaly_scores,
):
    investigated = numpy.asarray(investigated_anomaly_scores)
    baseline = numpy.asarray(baseline_anomaly_scores)

    figure, (density_axis, roc_axis) = matplotlib.pyplot.subplots(
        ncols=2, figsize=(12, 4.5),
    )

    score_min = min(investigated.min(), baseline.min())
    score_max = max(investigated.max(), baseline.max())
    grid = numpy.linspace(score_min, score_max, 300)

    baseline_density = gaussian_kde(baseline)(grid)
    investigated_density = gaussian_kde(investigated)(grid)

    density_axis.fill_between(grid, baseline_density, alpha=0.4, label="baseline")
    density_axis.fill_between(grid, investigated_density, alpha=0.4, label="investigated")
    density_axis.axvline(numpy.median(baseline), linestyle="--", color="C0", linewidth=1, label=f"baseline median: {numpy.median(baseline):.3f}")
    density_axis.axvline(numpy.median(investigated), linestyle="--", color="C1", linewidth=1, label=f"investigated median: {numpy.median(investigated):.3f}")
    density_axis.set_xlabel("anomaly score")
    density_axis.set_ylabel("density")
    density_axis.set_title("score distributions")
    density_axis.legend(fontsize=8)

    labels = numpy.concatenate([numpy.ones(len(investigated)), numpy.zeros(len(baseline))])
    scores = numpy.concatenate([-investigated, -baseline])
    false_positive_rates, true_positive_rates, _ = roc_curve(labels, scores)
    area_under_curve = numpy.trapezoid(true_positive_rates, false_positive_rates)

    roc_axis.plot(false_positive_rates, true_positive_rates, linewidth=2, label=f"area under curve = {area_under_curve:.3f}")
    roc_axis.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="random baseline")
    roc_axis.set_xlabel("false positive rate")
    roc_axis.set_ylabel("true positive rate")
    roc_axis.set_title("receiver operating characteristic")
    roc_axis.legend(fontsize=8)
    roc_axis.set_aspect("equal")

    figure.tight_layout()
    return figure


import pandas
import numpy
import dataframes
import anomalies


def _aligned_mosaic_data_from_features(
    investigated_features_dataframe,
    baseline_features_dataframe,
    baseline_aggregation,
    observation_period,
    anomaly_scores_from_dataframe,
):
    baseline_zscores = dataframes.zscored_dataframe_from_dataframe_and_baseline(
        baseline_features_dataframe, baseline_features_dataframe,
    )
    investigated_zscores = dataframes.zscored_dataframe_from_dataframe_and_baseline(
        investigated_features_dataframe, baseline_features_dataframe,
    )

    baseline_zscores = dataframes.reordered_columns_from_dataframe(baseline_zscores)
    investigated_zscores = investigated_zscores[baseline_zscores.columns]

    baseline_anomaly_scores = anomaly_scores_from_dataframe(baseline_features_dataframe)
    investigated_anomaly_scores = anomaly_scores_from_dataframe(investigated_features_dataframe)

    inv_timestamps = pandas.to_datetime(investigated_features_dataframe["timestamp"])
    bas_timestamps = pandas.to_datetime(baseline_features_dataframe["timestamp"])

    # --- aggregate baseline ---
    numeric_columns = list(investigated_zscores.columns)
    bas_zscores_agg, bas_scores_agg = anomalies.aggregated_panel_from_zscores_and_metadata(
        baseline_zscores, baseline_anomaly_scores, bas_timestamps,
        baseline_features_dataframe["hive"].values, baseline_aggregation,
        numeric_columns,
    )
    bas_timestamps_agg = pandas.to_datetime(bas_zscores_agg.index)

    # --- observation period ---
    match observation_period:
        case "intersection":
            obs_start = max(inv_timestamps.min(), bas_timestamps_agg.min())
            obs_end = min(inv_timestamps.max(), bas_timestamps_agg.max())
        case "join" | None:
            obs_start = min(inv_timestamps.min(), bas_timestamps_agg.min())
            obs_end = max(inv_timestamps.max(), bas_timestamps_agg.max())
        case "investigated":
            obs_start = inv_timestamps.min()
            obs_end = inv_timestamps.max()
        case "baseline":
            obs_start = bas_timestamps_agg.min()
            obs_end = bas_timestamps_agg.max()
        case (start, end):
            obs_start, obs_end = pandas.Timestamp(start), pandas.Timestamp(end)

    # --- window both panels ---
    inv_mask = (inv_timestamps >= obs_start) & (inv_timestamps <= obs_end)
    investigated_zscores = investigated_zscores[inv_mask.values]
    investigated_anomaly_scores = investigated_anomaly_scores[inv_mask.values]
    inv_timestamps = inv_timestamps[inv_mask]

    bas_mask = (bas_timestamps_agg >= obs_start) & (bas_timestamps_agg <= obs_end)
    bas_zscores_agg = bas_zscores_agg[bas_mask]
    bas_scores_agg = bas_scores_agg[bas_mask]
    bas_timestamps_agg = bas_timestamps_agg[bas_mask]

    # --- build full timestamp grid and reindex ---
    all_present_hours = set(
        inv_timestamps.dt.hour.tolist()
        + bas_timestamps_agg.hour.tolist()
    )
    full_index = anomalies.full_timestamps_from_bounds_and_hours(obs_start, obs_end, all_present_hours)

    inv_zscores_full, inv_scores_full = anomalies.reindexed_zscores_and_scores_from_originals_and_index(
        investigated_zscores, investigated_anomaly_scores, inv_timestamps, full_index,
    )
    bas_zscores_full, bas_scores_full = anomalies.reindexed_zscores_and_scores_from_originals_and_index(
        bas_zscores_agg, bas_scores_agg, bas_timestamps_agg, full_index,
    )

    aggregation_label = (
        f"hive_{baseline_aggregation:02d}" if isinstance(baseline_aggregation, int)
        else baseline_aggregation
    )

    return dict(
        investigated_zscores=inv_zscores_full,
        investigated_anomaly_scores=inv_scores_full,
        baseline_zscores=bas_zscores_full,
        baseline_anomaly_scores=bas_scores_full,
        timestamps=full_index,
        aggregation_label=aggregation_label,
    )

def investigate_anomaly(
    investigated_features_dataframe,
    baseline_features_dataframe,
    baseline_aggregation,
    anomaly_scores_from_dataframe,
    observation_period='intersection',
    plotly=False,
):
    data = _aligned_mosaic_data_from_features(
        investigated_features_dataframe,
        baseline_features_dataframe,
        baseline_aggregation,
        observation_period,
        anomaly_scores_from_dataframe = anomaly_scores_from_dataframe,
    )
    if plotly:
        return anomalies.plotly_mosaic_from_zscored_dataframes_and_scores(**data)
    return anomalies.mosaic_from_zscored_dataframes_and_scores(**data)


