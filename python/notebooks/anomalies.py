from scipy.stats import mannwhitneyu



def discrimination_from_scores_and_scores(investigated_scores, baseline_scores):
    investigated = numpy.asarray(investigated_scores)
    baseline = numpy.asarray(baseline_scores)

    u_statistic, p_value = mannwhitneyu(investigated, baseline, alternative="less")
    n_investigated = len(investigated)
    n_baseline = len(baseline)
    area_under_curve = 1.0 - u_statistic / (n_investigated * n_baseline)

    pooled_mean = numpy.concatenate([investigated, baseline]).mean()
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
from sklearn.covariance import EllipticEnvelope
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


def anomaly_scores_from_dataframe_from_inliers_dataframe(inliers_dataframe):
    scorers = scorers_from_inliers_dataframe_and_detector(
        inliers_dataframe,
        
        # IsolationForest(random_state=42),
        # LocalOutlierFactor(novelty=True),
        OneClassSVM(),
        
        # # EllipticEnvelope(random_state=42), # not suitable for hour slices

    )
    return lambda dataframe: anomaly_scores_from_dataframe_and_scorers(dataframe, scorers)



import matplotlib.pyplot
import matplotlib.patches
import numpy
import pandas


def mosaic_from_zscored_dataframes_and_scores(
    investigated_zscores,
    investigated_anomaly_scores,
    investigated_timestamps,
    baseline_zscores,
    baseline_anomaly_scores,
    baseline_timestamps,
    baseline_hives,
    aggregation_method="mean",
):
    numeric_columns = list(investigated_zscores.columns)

    def aggregated_panel_from_zscores_and_metadata(zscores, anomaly_scores, timestamps, hives, method):
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

    investigated_timestamps = pandas.to_datetime(investigated_timestamps)
    time_min, time_max = investigated_timestamps.min(), investigated_timestamps.max()

    baseline_timestamps_parsed = pandas.to_datetime(baseline_timestamps)
    time_mask = (baseline_timestamps_parsed >= time_min) & (baseline_timestamps_parsed <= time_max)
    bounded_zscores = baseline_zscores[time_mask.values]
    bounded_anomaly_scores = numpy.asarray(baseline_anomaly_scores)[time_mask.values]
    bounded_timestamps = baseline_timestamps_parsed[time_mask.values]
    bounded_hives = numpy.asarray(baseline_hives)[time_mask.values]

    baseline_zscores_agg, baseline_anomaly_scores_agg = aggregated_panel_from_zscores_and_metadata(
        bounded_zscores, bounded_anomaly_scores, bounded_timestamps, bounded_hives, aggregation_method,
    )
    baseline_timestamps_agg = pandas.to_datetime(baseline_zscores_agg.index)

    all_anomaly_scores = numpy.concatenate([
        numpy.asarray(investigated_anomaly_scores),
        numpy.asarray(baseline_anomaly_scores_agg),
    ])
    anomaly_score_vmin = all_anomaly_scores.min()
    anomaly_score_vmax = all_anomaly_scores.max()

    pixels_per_row = 0.15
    height_investigated = len(investigated_timestamps) * pixels_per_row
    height_baseline = len(baseline_timestamps_agg) * pixels_per_row
    total_height = max(16, height_investigated + height_baseline)
    width = max(14, len(numeric_columns) * 0.3)

    figure, axes = matplotlib.pyplot.subplots(
        nrows=2, ncols=2,
        figsize=(width, total_height),
        gridspec_kw={
            "width_ratios": [1, len(numeric_columns)],
            "height_ratios": [height_investigated, height_baseline],
        },
    )

    (investigated_score_axis, investigated_features_axis) = axes[0]
    (baseline_score_axis, baseline_features_axis) = axes[1]

    def render_panel(score_axis, features_axis, zscores_values, anomaly_scores_values, timestamps, title):
        features_image = features_axis.imshow(
            zscores_values.astype(float),
            aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3, interpolation="nearest",
        )
        features_axis.set_xticks(range(len(numeric_columns)))
        features_axis.set_xticklabels(numeric_columns, fontsize=6, rotation=90)
        features_axis.set_title(title, fontsize=10)


        score_axis.imshow(
            numpy.array(anomaly_scores_values).reshape(-1, 1),
            aspect="auto", cmap="Purples_r",
            vmin=anomaly_score_vmin, vmax=anomaly_score_vmax,
            interpolation="nearest",
        )
        score_axis.set_xticks([0])
        score_axis.set_xticklabels(["anomaly\nscore"], fontsize=8)

        anomaly_array = numpy.array(anomaly_scores_values)
        anomaly_threshold = numpy.percentile(anomaly_array, 1)
        for i in range(len(anomaly_array)):
            if anomaly_array[i] < anomaly_threshold:
                score_axis.add_patch(matplotlib.patches.Rectangle(
                    (-0.5, i - 0.5), 1, 1, linewidth=2, edgecolor="red", facecolor="none",
                ))

        day_indices, hour_indices = [], []
        day_labels, hour_labels = [], []
        for i, timestamp in enumerate(timestamps):
            if timestamp.hour == 0:
                day_indices.append(i)
                day_labels.append(timestamp.strftime("%b %d"))
            else:
                hour_indices.append(i)
                hour_labels.append(timestamp.strftime("%H:%M"))

        score_axis.set_yticks(day_indices)
        score_axis.set_yticklabels(day_labels, fontsize=9, fontweight="bold")
        score_axis.set_yticks(hour_indices, minor=True)
        score_axis.set_yticklabels(hour_labels, minor=True, fontsize=5)
        score_axis.tick_params(axis="y", which="major", length=8, width=1.5)
        score_axis.tick_params(axis="y", which="minor", length=3, width=0.5)
        features_axis.set_yticks([], minor=False)
        features_axis.set_yticks([], minor=True)
        return features_image

    aggregation_label = (
        f"hive_{aggregation_method:02d}" if isinstance(aggregation_method, int)
        else aggregation_method
    )

    features_image = render_panel(
        investigated_score_axis, investigated_features_axis,
        investigated_zscores.values, investigated_anomaly_scores, investigated_timestamps,
        "investigated data",
    )
    render_panel(
        baseline_score_axis, baseline_features_axis,
        baseline_zscores_agg.values, baseline_anomaly_scores_agg.values, baseline_timestamps_agg,
        f"baseline data (aggregation: {aggregation_label})",
    )

    colorbar_axis = figure.add_axes([0.25, 0.98, 0.5, 0.01])
    figure.colorbar(features_image, cax=colorbar_axis, orientation="horizontal")
    colorbar_axis.set_xlabel("z-score (std from baseline mean)", fontsize=8)
    colorbar_axis.xaxis.set_label_position("top")

    figure.subplots_adjust(hspace=0.15, wspace=0.05, top=0.96, bottom=0.05)
    return figure


from sklearn.metrics import roc_curve
from scipy.stats import gaussian_kde
import numpy
import matplotlib.pyplot


def discrimination_figure_from_investigated_scores_and_baseline_scores(
    investigated_anomaly_scores,
    baseline_anomaly_scores
):
    investigated = numpy.asarray(investigated_anomaly_scores)
    baseline = numpy.asarray(baseline_anomaly_scores)

    figure, (density_axis, roc_axis) = matplotlib.pyplot.subplots(
        ncols=2, figsize=(12, 4.5),
    )

    # --- density panel ---
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

    # --- receiver operating characteristic panel ---
    labels = numpy.concatenate([numpy.ones(len(investigated)), numpy.zeros(len(baseline))])
    scores = numpy.concatenate([-investigated, -baseline])  # negate so higher = more anomalous
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
