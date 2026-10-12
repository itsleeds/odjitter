use std::collections::HashSet;
use std::fmt;

use rand::rngs::StdRng;
use rand::seq::SliceRandom;

pub const MAX_ATTEMPTS: usize = 1000;
const RADIUS: f64 = 6371008.8;

pub type Key = [u64; 4];

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WeightedPoint {
    pub x: f64,
    pub y: f64,
    pub weight: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Error {
    NoPoints,
    Attempts,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Error::NoPoints => write!(f, "zone has no points to sample"),
            Error::Attempts => write!(
                f,
                "no pair found at the minimum distance after {MAX_ATTEMPTS} attempts"
            ),
        }
    }
}

impl std::error::Error for Error {}

pub fn repeats(count: f64, threshold: usize) -> f64 {
    if count == 0.0 {
        1.0
    } else {
        (count / threshold as f64).ceil()
    }
}

pub fn distance(a: Point, b: Point) -> f64 {
    let theta1 = a.y.to_radians();
    let theta2 = b.y.to_radians();
    let delta_theta = (b.y - a.y).to_radians();
    let delta_lambda = (b.x - a.x).to_radians();
    let h = (delta_theta / 2.0).sin().powi(2)
        + theta1.cos() * theta2.cos() * (delta_lambda / 2.0).sin().powi(2);
    RADIUS * (2.0 * h.sqrt().asin())
}

pub fn pick(points: &[WeightedPoint], rng: &mut StdRng) -> Option<Point> {
    points
        .choose_weighted(rng, |p| p.weight)
        .ok()
        .map(|p| Point { x: p.x, y: p.y })
}

pub fn key(o: Point, d: Point) -> Key {
    let bits = |v: f64| if v == 0.0 { 0 } else { v.to_bits() };
    [bits(o.x), bits(o.y), bits(d.x), bits(d.y)]
}

pub fn pairs<E, O, D, F>(
    count: usize,
    min_distance: f64,
    mut origin: O,
    mut destination: D,
    mut seen: Option<&mut HashSet<Key>>,
    rng: &mut StdRng,
    mut out: F,
) -> Result<(), E>
where
    O: FnMut(&mut StdRng) -> Option<Point>,
    D: FnMut(&mut StdRng) -> Option<Point>,
    F: FnMut(Point, Point) -> Result<(), E>,
    E: From<Error>,
{
    for _ in 0..count {
        let mut tries = 0;
        loop {
            tries += 1;
            if tries > MAX_ATTEMPTS {
                return Err(Error::Attempts.into());
            }
            let o = origin(rng).ok_or(Error::NoPoints)?;
            let d = destination(rng).ok_or(Error::NoPoints)?;
            if distance(o, d) < min_distance {
                continue;
            }
            if let Some(seen) = seen.as_mut() {
                if !seen.insert(key(o, d)) {
                    continue;
                }
            }
            out(o, d)?;
            break;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn single_point_intrazonal_pair_errors() {
        let pts = [WeightedPoint {
            x: 0.0,
            y: 0.0,
            weight: 1.0,
        }];
        let mut rng = StdRng::seed_from_u64(42);
        let mut n = 0;
        let res: Result<(), Error> = pairs(
            1,
            1.0,
            |r| pick(&pts, r),
            |r| pick(&pts, r),
            None,
            &mut rng,
            |_, _| {
                n += 1;
                Ok(())
            },
        );
        assert_eq!(res, Err(Error::Attempts));
        assert_eq!(n, 0);
    }

    #[test]
    fn intrazonal_pairs_respect_min_distance() {
        let pts = [
            WeightedPoint {
                x: 0.0,
                y: 0.0,
                weight: 1.0,
            },
            WeightedPoint {
                x: 0.01,
                y: 0.0,
                weight: 1.0,
            },
        ];
        let mut rng = StdRng::seed_from_u64(42);
        let mut out = Vec::new();
        let res: Result<(), Error> = pairs(
            5,
            1.0,
            |r| pick(&pts, r),
            |r| pick(&pts, r),
            None,
            &mut rng,
            |o, d| {
                out.push((o, d));
                Ok(())
            },
        );
        assert_eq!(res, Ok(()));
        assert_eq!(out.len(), 5);
        assert!(out.iter().all(|(o, d)| distance(*o, *d) >= 1.0));
    }

    #[test]
    fn pick_empty_is_none() {
        let mut rng = StdRng::seed_from_u64(42);
        assert_eq!(pick(&[], &mut rng), None);
    }

    #[test]
    fn repeats_rounds_up() {
        assert_eq!(repeats(0.0, 50), 1.0);
        assert_eq!(repeats(120.0, 50), 3.0);
    }
}
