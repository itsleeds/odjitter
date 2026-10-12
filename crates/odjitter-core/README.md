# odjitter-core

Core logic for odjitter, with no file input or output.

It samples weighted points, checks minimum distances between points using the haversine formula, and disaggregates OD counts into pairs of points.

It is used by the `odjitter` crate, which handles the CSV, GeoJSON and FlatGeobuf files.
