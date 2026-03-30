# import numpy
# from pipe import Pipe

# @Pipe
# def put_slice_aggregations(stream):
#     current_key = None
#     batch = []

#     for timestamp, features, tags, aggregations in stream:
#         key = group_key_from_tags(timestamp, tags)
#         assert current_key is None or key >= current_key, f"stream not sorted: {key} after {current_key}"
#         if current_key is not None and key != current_key:
#             agg = aggregated_from_feature_dicts([f for _, f, _, _ in batch])
#             for ts, feat, tgs, _ in batch:
#                 yield ts, feat, tgs, agg
#             batch = []


        
#         current_key = key
#         batch.append((timestamp, features, tags, aggregations))

#     if batch:
#         agg = aggregated_from_feature_dicts([f for _, f, _, _ in batch])
#         for ts, feat, tgs, _ in batch:
#             yield ts, feat, tgs, agg

# def group_key_from_tags(timestamp, tags):
#     date = str(timestamp.astype("datetime64[D]"))
#     return (tags["hive"], date, tags["time_slice"])

# def scalar_features_from_features(features):
#     return {k: v for k, v in features.items() if not isinstance(v, numpy.ndarray)}


# def vector_features_from_features(features):
#     return {k: v for k, v in features.items() if isinstance(v, numpy.ndarray)}


# def aggregated_from_feature_dicts(feature_dicts):
#     scalars = [scalar_features_from_features(f) for f in feature_dicts]
#     vectors = [vector_features_from_features(f) for f in feature_dicts]

#     result = {}

#     scalar_keys = scalars[0].keys()
#     for key in scalar_keys:
#         values = numpy.array([s[key] for s in scalars])
#         result[f"{key}_mean"] = float(numpy.mean(values))
#         result[f"{key}_std"] = float(numpy.std(values))

#     vector_keys = vectors[0].keys()
#     for key in vector_keys:
#         stacked = numpy.stack([v[key] for v in vectors])
#         means = numpy.mean(stacked, axis=0)
#         stds = numpy.std(stacked, axis=0)
#         for i, (m, s) in enumerate(zip(means, stds)):
#             result[f"{key}_{i}_mean"] = float(m)
#             result[f"{key}_{i}_std"] = float(s)

#     return result



