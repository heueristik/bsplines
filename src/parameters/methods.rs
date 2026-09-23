use nalgebra::DVector;

use crate::{
    error::{Error, Result},
    parameters::Parameters,
    points::{DataPoints, Points},
};

/// Returns the length of the vector. The vector is divided by a power of two near its largest component first,
/// so the squares neither overflow nor underflow. A power of two scales exactly, so the length equals the
/// direct norm wherever that one neither overflows nor underflows.
fn norm_without_overflow(vector: &DVector<f64>) -> f64 {
    let largest = vector.amax();
    if largest == 0.0 || !largest.is_finite() {
        return largest;
    }
    let scale = largest.log2().floor().exp2();
    (vector / scale).norm() * scale
}

/// Checks that the chord sum can divide the chords: it is positive and finite.
fn check_chord_sum(sum: f64) -> Result<()> {
    if sum == 0.0 {
        return Err(Error::CoincidentDataPoints);
    }
    if !sum.is_finite() {
        return Err(Error::NonFiniteValue);
    }
    Ok(())
}

/// Generates one parameter per data point, equally spaced on [0, 1] — eq. (9.3) in `Piegl1997`:
///
/// ūg = g / m,   g = 0, …, m
///
/// with the parameters ū and the number of polyline segments m.
///
/// Not recommended for unevenly spaced data, as it can produce erratic shapes such as loops.
pub fn equally_spaced(polyline_segments: usize) -> Parameters {
    let mut u_bar = DVector::zeros(polyline_segments + 1);

    for g in 1..polyline_segments {
        u_bar[g] = g as f64 / polyline_segments as f64;
    }
    u_bar[polyline_segments] = 1f64;

    Parameters { vector: u_bar }
}

/// Generates the parameters by the centripetal method — eq. (9.6) in `Piegl1997`:
///
/// ūg = ūg₋₁ + √|Qg − Qg₋₁| ∕ d,   d = Σg √|Qg − Qg₋₁|
///
/// with the parameters ū, the data points Q, and the total sum d.
///
/// Dampens the effect of outlier points on the parametrization.
pub fn centripetal(points: &DataPoints) -> Result<Parameters> {
    let polyline_segments = points.polyline_segments();

    let mut sum = 0.0;

    for g in 1..=polyline_segments {
        let chord = points.matrix().column(g) - points.matrix().column(g - 1);
        sum += norm_without_overflow(&chord).sqrt()
    }
    check_chord_sum(sum)?;

    let mut u_bar = DVector::zeros(polyline_segments + 1);

    for g in 1..polyline_segments {
        let chord = points.matrix().column(g) - points.matrix().column(g - 1);
        u_bar[g] = u_bar[g - 1] + norm_without_overflow(&chord).sqrt() / sum;
    }

    u_bar[polyline_segments] = 1.0;

    Ok(Parameters { vector: u_bar })
}

/// Generates the parameters by the chord-length method — eqs. (9.4) and (9.5) in `Piegl1997`:
///
/// ūg = ūg₋₁ + |Qg − Qg₋₁| ∕ d,   d = Σg |Qg − Qg₋₁|
///
/// with the parameters ū, the data points Q, and the total chord length d.
pub fn chord_length(points: &DataPoints) -> Result<Parameters> {
    let polyline_segments = points.polyline_segments();

    let mut sum = 0f64;

    for g in 1..=polyline_segments {
        let chord = points.matrix().column(g) - points.matrix().column(g - 1);
        sum += norm_without_overflow(&chord);
    }
    check_chord_sum(sum)?;

    let mut u_bar = DVector::zeros(polyline_segments + 1);
    for g in 1..polyline_segments {
        let chord = points.matrix().column(g) - points.matrix().column(g - 1);
        u_bar[g] = u_bar[g - 1] + norm_without_overflow(&chord) / sum;
    }

    u_bar[polyline_segments] = 1f64;

    Ok(Parameters { vector: u_bar })
}

#[cfg(test)]
mod tests {
    use nalgebra::{dmatrix, dvector};

    use super::*;

    mod equally_spaced {
        use super::*;

        #[test]
        fn test() {
            let points = DataPoints::new(dmatrix![1.0, 2.0, 3.0, 4.0, 5.0;]);
            let parameters = equally_spaced(points.polyline_segments());
            assert_eq!(parameters.vector, dvector![0., 0.25, 0.5, 0.75, 1.]);
        }
    }

    mod chord_length {
        use super::*;

        #[test]
        fn linear_1() {
            let points = DataPoints::new(dmatrix![1.0, 2.0, 3.0, 4.0, 5.0;]);
            let parameters = chord_length(&points).unwrap();
            assert_eq!(parameters.vector, dvector![0., 0.25, 0.5, 0.75, 1.]);
        }

        #[test]
        fn linear_2() {
            let points = DataPoints::new(dmatrix![1.0, 3.0, 5.0;]);
            let parameters = chord_length(&points).unwrap();
            assert_eq!(parameters.vector, dvector![0., 0.5, 1.]);
        }

        #[test]
        fn non_linear_1() {
            let points = DataPoints::new(dmatrix![1.0, 2.0, 5.0;]);
            let parameters = chord_length(&points).unwrap();
            assert_eq!(parameters.vector, dvector![0., 0.25, 1.]);
        }

        #[test]
        fn non_linear_2() {
            let points = DataPoints::new(dmatrix![1.0, 4.0, 5.0;]);
            let parameters = chord_length(&points).unwrap();
            assert_eq!(parameters.vector, dvector![0., 0.75, 1.]);
        }
    }

    mod centripetal {
        use super::*;

        #[test]
        fn linear_1() {
            let points = DataPoints::new(dmatrix![1.0, 2.0, 3.0, 4.0, 5.0;]);
            let parameters = centripetal(&points).unwrap();
            assert_eq!(parameters.vector, dvector![0., 0.25, 0.5, 0.75, 1.]);
        }

        #[test]
        fn linear_2() {
            let points = DataPoints::new(dmatrix![1.0, 3.0, 5.0;]);
            let parameters = centripetal(&points).unwrap();
            assert_eq!(parameters.vector, dvector![0., 0.5, 1.]);
        }

        #[test]
        fn non_linear_1() {
            let points = DataPoints::new(dmatrix![1.0, 2.0, 11.0;]);
            let parameters = centripetal(&points).unwrap();
            assert_eq!(parameters.vector, dvector![0., 0.25, 1.]);
        }

        #[test]
        fn non_linear_2() {
            let points = DataPoints::new(dmatrix![1.0, 10.0, 11.0;]);
            let parameters = centripetal(&points).unwrap();
            assert_eq!(parameters.vector, dvector![0., 0.75, 1.]);
        }
    }
}
