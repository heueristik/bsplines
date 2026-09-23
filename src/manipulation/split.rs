//! Splits a curve into two independent curves.

use nalgebra::DVector;

use crate::{
    Curve,
    error::Result,
    knots::{Knots, normalize},
    manipulation::insert::{check_input, insert_knot},
    points::{ControlPoints, Points},
};

/// Splits the curve into two independent curves at the parameter `u` and normalizes both knot vectors.
pub(crate) fn split(curve: &Curve, u: f64) -> Result<(Curve, Curve)> {
    let multiplicity = check_input(curve, u)?;
    let degree = curve.degree();

    // With the multiplicity p, the knot u cuts the curve into two pieces that share the control point at u.
    // Each insertion moves the knot span of u one index up.
    let span = curve.knots.find_span(u, 0);
    let unsplit = (curve.knots.vector().clone(), curve.control_points.matrix().clone());
    let (knots, points) = (span..span + degree - multiplicity)
        .fold(unsplit, |(knots, points), insertion_span| insert_knot(&knots, &points, degree, insertion_span, u));

    // One more knot u clamps each piece at the joint with p + 1 equal knots.
    let knot_values = knots.as_slice();
    let first = knot_values.partition_point(|&knot| knot < u);
    let mut left_knots = DVector::from_vec(knot_values[..first + degree].iter().copied().chain([u]).collect());
    let mut right_knots = DVector::from_vec([u].into_iter().chain(knot_values[first..].iter().copied()).collect());
    normalize(&mut left_knots);
    normalize(&mut right_knots);

    let left_points = points.columns(0, first).into_owned();
    let right_count = right_knots.len() - degree - 1;
    let right_points = points.columns(points.ncols() - right_count, right_count).into_owned();

    let left = Curve::new(Knots::new(degree, left_knots)?, ControlPoints::new(left_points))?;
    let right = Curve::new(Knots::new(degree, right_knots)?, ControlPoints::new(right_points))?;
    Ok((left, right))
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use nalgebra::{dmatrix, dvector};
    use rstest::{fixture, rstest};

    use super::*;
    use crate::error::Error;

    #[fixture]
    /// A one-dimensional, linear test curve with default degree two.
    fn curve(#[default(2)] degree: usize) -> Curve {
        let curve = Curve::with_uniform_knots(ControlPoints::new(dmatrix![1., 2., 3., 4., 5., 6.;]), degree).unwrap();
        assert_eq!(curve.knots.vector(), &dvector![0., 0., 0., 0.25, 0.5, 0.75, 1., 1., 1.]);
        curve
    }

    #[rstest]
    fn cannot_split_start(curve: Curve) {
        let u = 0.0;
        let result = split(&curve, u);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), Error::OutsideDomainInterior { u, min: 0.0, max: 1.0 });
    }

    #[rstest]
    fn cannot_split_end(curve: Curve) {
        let u = 1.0;
        let result = split(&curve, u);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), Error::OutsideDomainInterior { u, min: 0.0, max: 1.0 });
    }

    #[rstest]
    fn split_errors_for_a_nan_parameter(curve: Curve) {
        assert!(matches!(split(&curve, f64::NAN), Err(Error::OutsideDomainInterior { .. })));
    }

    #[rstest]
    fn split_close_to_start(curve: Curve) {
        let eps = f64::EPSILON;
        let (left, right) = split(&curve, 0.0 + eps).unwrap();

        assert_relative_eq!(left.knots.vector(), &dvector![0., 0., 0., 1., 1., 1.], epsilon = eps.sqrt());
        assert_relative_eq!(left.control_points.matrix(), &dmatrix![1., 1., 1.;], epsilon = eps.sqrt());

        assert_relative_eq!(right.knots.vector(), curve.knots.vector(), epsilon = eps.sqrt());
        assert_relative_eq!(right.control_points.matrix(), curve.control_points.matrix(), epsilon = eps.sqrt());
    }

    #[rstest]
    fn split_close_to_end(curve: Curve) {
        let eps = f64::EPSILON;
        let (left, right) = split(&curve, 1.0 - eps).unwrap();

        assert_relative_eq!(left.knots.vector(), curve.knots.vector(), epsilon = eps.sqrt());
        assert_relative_eq!(left.control_points.matrix(), curve.control_points.matrix(), epsilon = eps.sqrt());

        assert_relative_eq!(right.knots.vector(), &dvector![0., 0., 0., 1., 1., 1.], epsilon = eps.sqrt());
        assert_relative_eq!(right.control_points.matrix(), &dmatrix![6., 6., 6.;], epsilon = eps.sqrt());
    }

    #[rstest]
    fn normalized(curve: Curve) {
        let (left, right) = split(&curve, 0.5).unwrap();
        assert_eq!(left.knots.vector(), &dvector![0., 0., 0., 0.5, 1., 1., 1.]);
        assert_eq!(left.control_points.matrix(), &dmatrix![1., 2., 3., 3.5;]);

        assert_eq!(right.knots.vector(), &dvector![0., 0., 0., 0.5, 1., 1., 1.]);
        assert_eq!(right.control_points.matrix(), &dmatrix![3.5, 4., 5., 6.;]);
    }
}
