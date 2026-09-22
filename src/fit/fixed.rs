use std::ops::SubAssign;

use crate::{
    error::Result,
    fit::{Penalization, compute_svd, difference_operator, input_checks},
    knots::Knots,
    parameters::Parameters,
    points::{DataPoints, Points},
    types::{MatD, VecD},
};

pub fn fit(
    knots: &Knots,
    points: &DataPoints,
    parameters: &Parameters,
    penalization: Option<Penalization>,
) -> Result<MatD> {
    input_checks(knots, points, parameters, &penalization)?;

    let residuals = calculate_residuals(knots, points, parameters);
    let constant_terms = calculate_constant_terms_matrix(knots, points, parameters, &residuals);
    let basis_matrix = calculate_basis_matrix(knots, points, parameters);

    let svd = compute_svd(knots, &basis_matrix, &penalization, Box::new(calculate_finite_difference_matrix))?;
    let internal_control_points = svd
        .solve(&constant_terms.transpose(), f64::EPSILON.sqrt())
        .expect("the SVD was computed with both U and V^T")
        .transpose();

    let polygon_segments = knots.polygon_segments();
    let polyline_segments = points.polyline_segments();
    let mut control_points = MatD::zeros(points.dimension(), polygon_segments + 1);

    // Fix the first and last control point to the end data points.
    control_points.column_mut(0).copy_from(&points.get(0));
    control_points.column_mut(polygon_segments).copy_from(&points.get(polyline_segments));

    for i in 1..=polygon_segments - 1 {
        control_points.column_mut(i).copy_from(&internal_control_points.column(i - 1));
    }

    Ok(control_points)
}

/// Returns the residual vectors R: the internal data points reduced by the contributions
/// of the two fixed end control points.
fn calculate_residuals(knots: &Knots, points: &DataPoints, parameters: &Parameters) -> MatD {
    let polygon_segments = knots.polygon_segments();
    let polyline_segments = points.polyline_segments();
    let dimension = points.dimension();

    let mut residuals = MatD::zeros(dimension, polyline_segments + 1);

    let u_bar = parameters.vector();

    for g in 1..=polyline_segments - 1 {
        residuals.column_mut(g).copy_from(&points.get(g));
        let u = u_bar[g];

        residuals.column_mut(g).sub_assign(knots.basis(0, u) * points.get(0));
        residuals.column_mut(g).sub_assign(knots.basis(polygon_segments, u) * points.get(polyline_segments));
    }

    residuals
}

fn calculate_constant_terms_matrix(
    knots: &Knots,
    points: &DataPoints,
    parameters: &Parameters,
    residuals: &MatD,
) -> MatD {
    let polygon_segments = knots.polygon_segments();
    let polyline_segments = points.polyline_segments();
    let dimension = points.dimension();

    let u_bar = parameters.vector();

    let mut constant_terms = MatD::zeros(dimension, polygon_segments - 1);

    let mut sum = VecD::zeros(dimension);
    for i in 1..=polygon_segments - 1 {
        sum *= 0.0;

        for g in 1..=polyline_segments - 1 {
            let u = u_bar[g];
            sum += knots.basis(i, u) * residuals.column(g);
        }
        constant_terms.column_mut(i - 1).copy_from(&sum);
    }

    constant_terms
}

fn calculate_basis_matrix(knots: &Knots, points: &DataPoints, parameters: &Parameters) -> MatD {
    let polygon_segments = knots.polygon_segments();
    let polyline_segments = points.polyline_segments();

    let u_bar = parameters.vector();

    let mut basis_matrix = MatD::zeros(polyline_segments - 1, polygon_segments - 1);
    for g in 1..=polyline_segments - 1 {
        let u = u_bar[g];
        for i in 1..=polygon_segments - 1 {
            basis_matrix[(g - 1, i - 1)] = knots.basis(i, u);
        }
    }
    basis_matrix
}

fn calculate_finite_difference_matrix(kappa: usize, knots: &Knots) -> MatD {
    let polygon_segments = knots.polygon_segments();
    assert!(
        kappa <= polygon_segments - 2,
        "the difference order kappa = {} must not exceed n - 2 = {}",
        kappa,
        polygon_segments - 2
    );

    let mut difference_matrix = MatD::zeros(polygon_segments - 1 - kappa, polygon_segments - 1);

    for i in 0..=polygon_segments - kappa - 2 {
        for j in 0..=polygon_segments - 2 {
            difference_matrix[(i, j)] = difference_operator(i, j, kappa) as f64;
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
    fn finite_difference_matrix_kappa_1() {
        let knots = Knots::new(1, dvector![0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0]);
        let matrix = calculate_finite_difference_matrix(1, &knots);
        let expected = dmatrix![
            -1.0, 1.0, 0.0;
             0.0,-1.0, 1.0;
        ];
        assert_eq!(matrix, expected);
    }

    #[test]
    fn finite_difference_matrix_kappa_2() {
        let knots = Knots::new(1, dvector![0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0]);
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
            fit(&knots, &points, &parameters, Some(Penalization { lambda: 0.5, kappa: 2 })).unwrap(),
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
        let points = fit(&knots, &data_points, &parameters, Some(Penalization { lambda: 1.0, kappa: 2 })).unwrap();
        let curve = Curve::new(knots, ControlPoints::new(points)).unwrap();

        assert_relative_eq!(curve.evaluate(0.5).unwrap(), dvector![0.0, 0.0], epsilon = f64::EPSILON.sqrt());
    }
}
