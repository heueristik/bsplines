use crate::{
    parameters::Parameters,
    points::{DataPoints, Points},
    types::VecD,
};

/// Generates one parameter per data point, equally spaced on [0, 1] — eq. (9.3) in `Piegl1997`:
///
/// ūg = g / m,   g = 0, …, m
///
/// with the parameters ū and the number of polyline segments m.
///
/// Not recommended for unevenly spaced data, as it can produce erratic shapes such as loops.
pub fn equally_spaced(polyline_segments: usize) -> Parameters {
    let mut u_bar = VecD::zeros(polyline_segments + 1);

    for g in 1..polyline_segments {
        u_bar[g] = g as f64 / polyline_segments as f64;
    }
    u_bar[polyline_segments] = 1f64;

    Parameters::new(u_bar)
}

/// Generates the parameters by the centripetal method — eq. (9.6) in `Piegl1997`:
///
/// ūg = ūg₋₁ + √|Qg − Qg₋₁| ∕ d,   d = Σg √|Qg − Qg₋₁|
///
/// with the parameters ū, the data points Q, and the total sum d.
///
/// Dampens the effect of outlier points on the parametrization.
pub fn centripetal(points: &DataPoints) -> Parameters {
    let polyline_segments = points.polyline_segments();

    let mut sum = 0.0;

    for g in 1..=polyline_segments {
        let chord = points.get(g) - points.get(g - 1);
        sum += chord.norm().sqrt()
    }

    debug_assert!(
        sum >= f64::EPSILON.sqrt(),
        "the chord length sum {} is too small; use the equally spaced method",
        sum
    );

    let mut u_bar = VecD::zeros(polyline_segments + 1);

    for g in 1..polyline_segments {
        let chord = points.get(g) - points.get(g - 1);
        u_bar[g] = u_bar[g - 1] + chord.norm().sqrt() / sum;
    }

    u_bar[polyline_segments] = 1.0;

    Parameters::new(u_bar)
}

/// Generates the parameters by the chord-length method — eqs. (9.4) and (9.5) in `Piegl1997`:
///
/// ūg = ūg₋₁ + |Qg − Qg₋₁| ∕ d,   d = Σg |Qg − Qg₋₁|
///
/// with the parameters ū, the data points Q, and the total chord length d.
pub fn chord_length(points: &DataPoints) -> Parameters {
    let polyline_segments = points.polyline_segments();

    let mut sum = 0f64;

    for g in 1..=polyline_segments {
        let chord = points.get(g) - points.get(g - 1);
        sum += chord.norm();
    }

    let mut u_bar = VecD::zeros(polyline_segments + 1);
    for g in 1..polyline_segments {
        let chord = points.get(g) - points.get(g - 1);
        u_bar[g] = u_bar[g - 1] + chord.norm() / sum;
    }

    u_bar[polyline_segments] = 1f64;

    Parameters::new(u_bar)
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
            let parameters = chord_length(&points);
            assert_eq!(parameters.vector, dvector![0., 0.25, 0.5, 0.75, 1.]);
        }

        #[test]
        fn linear_2() {
            let points = DataPoints::new(dmatrix![1.0, 3.0, 5.0;]);
            let parameters = chord_length(&points);
            assert_eq!(parameters.vector, dvector![0., 0.5, 1.]);
        }

        #[test]
        fn non_linear_1() {
            let points = DataPoints::new(dmatrix![1.0, 2.0, 5.0;]);
            let parameters = chord_length(&points);
            assert_eq!(parameters.vector, dvector![0., 0.25, 1.]);
        }

        #[test]
        fn non_linear_2() {
            let points = DataPoints::new(dmatrix![1.0, 4.0, 5.0;]);
            let parameters = chord_length(&points);
            assert_eq!(parameters.vector, dvector![0., 0.75, 1.]);
        }
    }

    mod centripetal {
        use super::*;

        #[test]
        fn linear_1() {
            let points = DataPoints::new(dmatrix![1.0, 2.0, 3.0, 4.0, 5.0;]);
            let parameters = centripetal(&points);
            assert_eq!(parameters.vector, dvector![0., 0.25, 0.5, 0.75, 1.]);
        }

        #[test]
        fn linear_2() {
            let points = DataPoints::new(dmatrix![1.0, 3.0, 5.0;]);
            let parameters = centripetal(&points);
            assert_eq!(parameters.vector, dvector![0., 0.5, 1.]);
        }

        #[test]
        fn non_linear_1() {
            let points = DataPoints::new(dmatrix![1.0, 2.0, 11.0;]);
            let parameters = centripetal(&points);
            assert_eq!(parameters.vector, dvector![0., 0.25, 1.]);
        }

        #[test]
        fn non_linear_2() {
            let points = DataPoints::new(dmatrix![1.0, 10.0, 11.0;]);
            let parameters = centripetal(&points);
            assert_eq!(parameters.vector, dvector![0., 0.75, 1.]);
        }
    }
}
