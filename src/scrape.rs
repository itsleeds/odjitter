use std::io::BufReader;

use anyhow::{bail, Result};
use fs_err::File;
use geo::CoordsIter;
use geo_types::Geometry;
use geojson::FeatureReader;

use crate::WeightedPoint;

/// Extracts all points from a GeoJSON file for use as origin/destination subpoints.
///
/// This function reads a GeoJSON file and extracts coordinates from all features, optionally
/// weighting each point based on a numeric property. The extracted points can be used with
/// the `Subsample::WeightedPoints` variant for weighted sampling during jittering.
///
/// # Arguments
///
/// * `path` - Path to the GeoJSON file containing point, linestring, or polygon features
/// * `weight_key` - Optional name of a numeric property to use as the relative weight for each
///   point. If `None`, all points are equally weighted (weight = 1.0). Higher weights make
///   points more likely to be selected during sampling.
///
/// # Returns
///
/// Returns a vector of `WeightedPoint` objects, each containing a geographic point and its
/// associated weight.
///
/// # Errors
///
/// Returns an error if:
/// - The file cannot be opened or read
/// - The GeoJSON is malformed
/// - A `weight_key` is specified but a feature doesn't have that property or it's not numeric
///
/// # Examples
///
/// ```rust,no_run
/// use odjitter::scrape_points;
///
/// // Extract points with equal weights
/// let points = scrape_points("road_network.geojson", None)?;
///
/// // Extract points with custom weights based on a property
/// let weighted_points = scrape_points("schools.geojson", Some("capacity".to_string()))?;
/// # Ok::<(), anyhow::Error>(())
/// ```
///
/// # Note
///
/// The returned points are not deduplicated. If the input geometry contains duplicate
/// coordinates, they will appear multiple times in the output, each with their respective weight.
pub fn scrape_points(path: &str, weight_key: Option<String>) -> Result<Vec<WeightedPoint>> {
    let reader = FeatureReader::from_reader(BufReader::new(File::open(path)?));
    let mut points = Vec::new();
    for feature in reader.features() {
        let feature = feature?;
        let weight = if let Some(ref key) = weight_key {
            if let Some(weight) = feature.property(key).and_then(|x| x.as_f64()) {
                weight
            } else {
                bail!("Feature doesn't have a numeric {} key: {:?}", key, feature);
            }
        } else {
            1.0
        };
        if let Some(geom) = feature.geometry {
            let geom: Geometry<f64> = geom.try_into()?;
            for pt in geom.coords_iter() {
                points.push(WeightedPoint {
                    point: pt.into(),
                    weight,
                });
            }
        }
    }
    Ok(points)
}
