use std::ops::SubAssign;

use nalgebra::{DMatrix, DVector};

use crate::{
    error::Result,
    fit::{Penalization, check_input, decompose_normal_matrix, difference_operator},
    knots::Knots,
    parameters::Parameters,
    points::{DataPoints, Points},
};

pub fn fit(
    knots: &Knots,
    points: &DataPoints,
    parameters: &Parameters,
    penalization: Option<Penalization>,
) -> Result<DMatrix<f64>> {
    check_input(knots, points, parameters, &penalization)?;

    let residuals = calculate_residuals(knots, points, parameters);
    let constant_terms = calculate_constant_terms(knots, points, parameters, &residuals);
    let basis_matrix = calculate_basis_matrix(knots, points, parameters);

    let svd =
        decompose_normal_matrix(knots, &basis_matrix, &penalization, Box::new(calculate_finite_difference_matrix))?;
    let internal_control_points = svd
        .solve(&constant_terms.transpose(), f64::EPSILON.sqrt())
        .expect("the SVD was computed with both U and V^T")
        .transpose();

    let polygon_segments = knots.polygon_segments();
    let polyline_segments = points.polyline_segments();
    let mut control_points = DMatrix::zeros(points.dimension(), polygon_segments + 1);

    // Fix the first and last control point to the end data points.
    control_points.column_mut(0).copy_from(&points.matrix().column(0));
    control_points.column_mut(polygon_segments).copy_from(&points.matrix().column(polyline_segments));

    for i in 1..=polygon_segments - 1 {
        control_points.column_mut(i).copy_from(&internal_control_points.column(i - 1));
    }

    Ok(control_points)
}

/// Returns the residual vectors R: the internal data points reduced by the contributions
/// of the two fixed end control points.
fn calculate_residuals(knots: &Knots, points: &DataPoints, parameters: &Parameters) -> DMatrix<f64> {
    let polygon_segments = knots.polygon_segments();
    let polyline_segments = points.polyline_segments();
    let dimension = points.dimension();

    let mut residuals = DMatrix::zeros(dimension, polyline_segments + 1);

    let u_bar = parameters.vector();

    for g in 1..=polyline_segments - 1 {
        residuals.column_mut(g).copy_from(&points.matrix().column(g));
        let u = u_bar[g];

        residuals.column_mut(g).sub_assign(knots.basis_of_derivative_curve(0, 0, u) * points.matrix().column(0));
        residuals.column_mut(g).sub_assign(
            knots.basis_of_derivative_curve(0, polygon_segments, u) * points.matrix().column(polyline_segments),
        );
    }

    residuals
}

fn calculate_constant_terms(
    knots: &Knots,
    points: &DataPoints,
    parameters: &Parameters,
    residuals: &DMatrix<f64>,
) -> DMatrix<f64> {
    let polygon_segments = knots.polygon_segments();
    let polyline_segments = points.polyline_segments();
    let dimension = points.dimension();

    let u_bar = parameters.vector();

    let mut constant_terms = DMatrix::zeros(dimension, polygon_segments - 1);

    let mut sum = DVector::zeros(dimension);
    for i in 1..=polygon_segments - 1 {
        sum *= 0.0;

        for g in 1..=polyline_segments - 1 {
            let u = u_bar[g];
            sum += knots.basis_of_derivative_curve(0, i, u) * residuals.column(g);
        }
        constant_terms.column_mut(i - 1).copy_from(&sum);
    }

    constant_terms
}

fn calculate_basis_matrix(knots: &Knots, points: &DataPoints, parameters: &Parameters) -> DMatrix<f64> {
    let polygon_segments = knots.polygon_segments();
    let polyline_segments = points.polyline_segments();

    let u_bar = parameters.vector();

    let mut basis_matrix = DMatrix::zeros(polyline_segments - 1, polygon_segments - 1);
    for g in 1..=polyline_segments - 1 {
        let u = u_bar[g];
        for i in 1..=polygon_segments - 1 {
            basis_matrix[(g - 1, i - 1)] = knots.basis_of_derivative_curve(0, i, u);
        }
    }
    basis_matrix
}

fn calculate_finite_difference_matrix(difference_order: usize, knots: &Knots) -> DMatrix<f64> {
    let polygon_segments = knots.polygon_segments();
    assert!(
        difference_order <= polygon_segments - 2,
        "the difference order {} must not exceed n - 2 = {}",
        difference_order,
        polygon_segments - 2
    );

    let mut difference_matrix = DMatrix::zeros(polygon_segments - 1 - difference_order, polygon_segments - 1);

    for i in 0..=polygon_segments - difference_order - 2 {
        for j in 0..=polygon_segments - 2 {
            difference_matrix[(i, j)] = difference_operator(i, j, difference_order) as f64;
        }
    }

    difference_matrix
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use nalgebra::{dmatrix, dvector};

    use crate::{
        Curve,
        knots::KnotMethod::{Averaging, Uniform},
        parameters::ParameterMethod::{ChordLength, EquallySpaced},
        points::ControlPoints,
    };

    use super::*;
    use crate::fit::test_data_points;

    #[test]
    fn finite_difference_matrix_order_1() {
        let knots = Knots::new(1, dvector![0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0]).unwrap();
        let matrix = calculate_finite_difference_matrix(1, &knots);
        let expected = dmatrix![
            -1.0, 1.0, 0.0;
             0.0,-1.0, 1.0;
        ];
        assert_eq!(matrix, expected);
    }

    #[test]
    fn finite_difference_matrix_order_2() {
        let knots = Knots::new(1, dvector![0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0]).unwrap();
        let matrix = calculate_finite_difference_matrix(2, &knots);
        let expected = dmatrix![
             1.0,-2.0, 1.0;
        ];
        assert_eq!(matrix, expected);
    }

    #[test]
    fn unpenalized_linear() {
        let points = DataPoints::new(dmatrix![
            1., 2., 3., 4., 5.;
            1., 2., 3., 4., 5.;
        ]);

        let parameters = Parameters::generate(&points, EquallySpaced);
        let knots = Knots::generate(1, points.polyline_segments(), &parameters, Uniform).unwrap();
        assert_eq!(crate::fit::loose::fit(&knots, &points, &parameters, None).unwrap(), *points.matrix());
    }

    #[test]
    fn penalized_linear() {
        let points = DataPoints::new(dmatrix![
            1., 2., 3., 4., 5.;
            1., 2., 3., 4., 5.;
        ]);

        let parameters = Parameters::generate(&points, ChordLength);
        let knots = Knots::generate(1, points.polyline_segments(), &parameters, Averaging).unwrap();
        assert_relative_eq!(
            fit(&knots, &points, &parameters, Some(Penalization { strength: 0.5, difference_order: 2 })).unwrap(),
            points.matrix(),
            epsilon = f64::EPSILON.sqrt()
        );
    }

    #[test]
    fn unpenalized_nonlinear() {
        let degree = 1;
        let data_points = test_data_points(10);

        let polygon_segments = data_points.polyline_segments();
        let parameters = Parameters::generate(&data_points, EquallySpaced);
        let knots = Knots::generate(degree, polygon_segments, &parameters, Uniform).unwrap();

        assert_relative_eq!(
            fit(&knots, &data_points, &parameters, None).unwrap(),
            data_points.matrix(),
            epsilon = f64::EPSILON.sqrt()
        );
    }

    #[test]
    fn penalized_nonlinear() {
        let degree = 2;
        let data_points = test_data_points(10);

        let parameters = Parameters::generate(&data_points, EquallySpaced);
        let knots = Knots::generate(degree, data_points.polyline_segments(), &parameters, Uniform).unwrap();
        let points =
            fit(&knots, &data_points, &parameters, Some(Penalization { strength: 1.0, difference_order: 2 })).unwrap();
        let curve = Curve::new(knots, ControlPoints::new(points)).unwrap();

        assert_relative_eq!(curve.evaluate(0.5).unwrap(), dvector![0.0, 0.0], epsilon = f64::EPSILON.sqrt());
    }
}
