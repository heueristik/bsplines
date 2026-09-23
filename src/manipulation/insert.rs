//! Inserts a knot into a curve.

use std::ops::AddAssign;

use nalgebra::{DMatrix, DVector};

use crate::{
    Curve,
    error::{Error, Result},
    points::Points,
};

/// Inserts the knot `u` into the curve by Boehm's algorithm, keeping the curve shape unchanged.
/// The parameter must lie in the domain interior (0, 1), and the multiplicity of `u`
/// must not already exceed the degree.
pub(crate) fn insert(curve: &mut Curve, u: f64) -> Result<()> {
    check_input(curve, u)?;

    let span = curve.knots.find_span(u, 0);
    let (knots, points) = insert_knot(curve.knots.vector(), curve.control_points.matrix(), curve.degree(), span, u);
    curve.knots.derivatives[0] = knots;
    curve.control_points.derivatives[0] = points;
    curve.derive();
    Ok(())
}

/// Checks that the parameter `u` lies in the domain interior (0, 1) and that its multiplicity does not exceed the
/// degree. Returns the multiplicity of `u`.
pub(crate) fn check_input(curve: &Curve, u: f64) -> Result<usize> {
    // The negated form also rejects NaN.
    if !(u > 0.0 && u < 1.0) {
        return Err(Error::OutsideDomainInterior { u, min: 0.0, max: 1.0 });
    }

    let degree = curve.degree();
    let multiplicity = curve.knots.multiplicity(u);
    if multiplicity > degree {
        return Err(Error::MultiplicityExceedsDegree { u, multiplicity, degree });
    }
    Ok(multiplicity)
}

/// Returns the knots and the control points with the knot `u` inserted once into the knot span `span`.
pub(crate) fn insert_knot(
    knots: &DVector<f64>,
    points: &DMatrix<f64>,
    degree: usize,
    span: usize,
    u: f64,
) -> (DVector<f64>, DMatrix<f64>) {
    let new_knots = knots.clone().insert_row(span + 1, u);

    // Only the control points from `span - degree + 1` to `span` change.
    let control_point_count = points.ncols();

    let mut new_points = DMatrix::zeros(points.nrows(), control_point_count + 1);

    let head_count = span - degree + 1;
    new_points.columns_mut(0, head_count).copy_from(&points.columns(0, head_count));

    let tail_count = control_point_count - span;
    new_points
        .columns_mut(new_points.ncols() - tail_count, tail_count)
        .copy_from(&points.columns(points.ncols() - tail_count, tail_count));

    let mut alpha: f64;
    for i in (span - degree + 1)..=span {
        alpha = (u - knots[i]) / (knots[i + degree] - knots[i]);

        new_points.column_mut(i).add_assign((1. - alpha) * points.column(i - 1) + alpha * points.column(i));
    }

    (new_knots, new_points)
}

#[cfg(test)]
mod tests {
    use nalgebra::{dmatrix, dvector};

    use crate::points::ControlPoints;

    use super::*;

    #[test]
    fn insert_errors_for_a_nan_parameter() {
        let mut curve = Curve::with_uniform_knots(ControlPoints::new(dmatrix![-1., 0., 1.;]), 2).unwrap();
        let knots_before = curve.knots.vector().clone();

        assert!(matches!(insert(&mut curve, f64::NAN), Err(Error::OutsideDomainInterior { .. })));
        assert_eq!(curve.knots.vector(), &knots_before);
    }

    #[test]
    fn insert_after_a_repeated_knot_keeps_the_knots_sorted_and_the_shape() {
        let mut curve = Curve::with_uniform_knots(ControlPoints::new(dmatrix![0., 1., 3., 2., 4.;]), 2).unwrap();
        insert(&mut curve, 0.5).unwrap();
        insert(&mut curve, 0.5).unwrap();
        assert_eq!(curve.knots.multiplicity(0.5), 2, "the knot 0.5 repeats");
        let parameters: Vec<f64> = (0..=20).map(|step| f64::from(step) / 20.0).collect();
        let before: Vec<_> = parameters.iter().map(|&u| curve.evaluate(u).unwrap()).collect();

        insert(&mut curve, 0.7).unwrap();

        let knot_values = curve.knots.vector();
        assert!(knot_values.as_slice().is_sorted(), "the knots {knot_values:?} are sorted");
        for (&u, point) in parameters.iter().zip(&before) {
            approx::assert_relative_eq!(curve.evaluate(u).unwrap(), point, epsilon = 1e-12);
        }
    }

    #[test]
    fn degree_1() {
        let mut curve = Curve::with_uniform_knots(ControlPoints::new(dmatrix![-1., 1.;]), 1).unwrap();
        assert_eq!(curve.knots.vector(), &dvector![0., 0., 1., 1.]);

        insert(&mut curve, 0.5).unwrap();
        assert_eq!(curve.knots.vector(), &dvector![0., 0., 0.5, 1., 1.]);
        assert_eq!(curve.control_points.matrix(), &dmatrix![-1., 0., 1.;]);
    }

    #[test]
    fn degree_2() {
        let mut curve = Curve::with_uniform_knots(ControlPoints::new(dmatrix![-1., 0., 1.;]), 2).unwrap();
        assert_eq!(curve.knots.vector(), &dvector![0., 0., 0., 1., 1., 1.]);
        assert_eq!(curve.control_points.matrix(), &dmatrix![-1., 0., 1.;]);

        insert(&mut curve, 0.5).unwrap();
        assert_eq!(curve.knots.vector(), &dvector![0., 0., 0., 0.5, 1., 1., 1.]);
        assert_eq!(curve.control_points.matrix(), &dmatrix![-1., -0.5, 0.5, 1.;]);
    }

    #[test]
    fn degree_2_preexisting_knot() {
        let mut curve = Curve::with_uniform_knots(ControlPoints::new(dmatrix![-1.5, -0.5, 0.5, 1.5;]), 2).unwrap();
        assert_eq!(curve.knots.vector(), &dvector![0., 0., 0., 0.5, 1., 1., 1.]);
        assert_eq!(curve.control_points.matrix(), &dmatrix![-1.5, -0.5, 0.5, 1.5;]);

        insert(&mut curve, 0.5).unwrap();
        assert_eq!(curve.knots.vector(), &dvector![0., 0., 0., 0.5, 0.5, 1., 1., 1.]);
        assert_eq!(curve.control_points.matrix(), &dmatrix![-1.5, -0.5, 0.0, 0.5, 1.5;]);
    }

    #[test]
    fn degree_1_repeated_knot() {
        let mut curve = Curve::with_uniform_knots(ControlPoints::new(dmatrix![-1., 1.;]), 1).unwrap();
        assert_eq!(curve.knots.vector(), &dvector![0., 0., 1., 1.]);

        insert(&mut curve, 0.5).unwrap();
        assert_eq!(curve.knots.vector(), &dvector![0., 0., 0.5, 1., 1.]);
        assert_eq!(curve.control_points.matrix(), &dmatrix![-1., 0., 1.;]);

        insert(&mut curve, 0.5).unwrap();
        assert_eq!(curve.knots.vector(), &dvector![0., 0., 0.5, 0.5, 1., 1.]);
        assert_eq!(curve.control_points.matrix(), &dmatrix![-1., 0., 0., 1.;]);
    }

    #[test]
    fn degree_2_repeated_knots() {
        let mut curve = Curve::with_uniform_knots(ControlPoints::new(dmatrix![-1., 0., 1.;]), 2).unwrap();
        let u = 0.5;
        let expected_point = dvector![0.0];

        assert_eq!(curve.knots.vector(), &dvector![0., 0., 0., 1., 1., 1.]);
        assert_eq!(curve.control_points.matrix(), &dmatrix![-1., 0., 1.;]);
        assert_eq!(curve.evaluate(u).unwrap(), expected_point);

        insert(&mut curve, u).unwrap();
        assert_eq!(curve.knots.vector(), &dvector![0., 0., 0., u, 1., 1., 1.]);
        assert_eq!(curve.control_points.matrix(), &dmatrix![-1., -0.5, 0.5, 1.;]);
        assert_eq!(curve.evaluate(u).unwrap(), expected_point);

        insert(&mut curve, u).unwrap();
        assert_eq!(curve.knots.vector(), &dvector![0., 0., 0., u, u, 1., 1., 1.]);
        assert_eq!(curve.control_points.matrix(), &dmatrix![-1., -0.5, 0.0, 0.5, 1.;]);
        assert_eq!(curve.evaluate(u).unwrap(), expected_point);

        insert(&mut curve, u).unwrap();
        assert_eq!(curve.knots.vector(), &dvector![0., 0., 0., u, u, u, 1., 1., 1.]);
        assert_eq!(curve.control_points.matrix(), &dmatrix![-1., -0.5, 0.0, 0.0, 0.5, 1.;]);
    }
}
