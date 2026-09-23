use std::ops::SubAssign;

use nalgebra::{DMatrix, DVector};

use crate::{
    error::Result,
    fit::{Penalization, check_input, decompose_normal_matrix},
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
    check_input(knots, points, parameters, &penalization, knots.polygon_segments().saturating_sub(1))?;

    let polygon_segments = knots.polygon_segments();
    let polyline_segments = points.polyline_segments();
    let mut control_points = DMatrix::zeros(points.dimension(), polygon_segments + 1);

    // Fix the first and last control point to the end data points.
    control_points.column_mut(0).copy_from(&points.matrix().column(0));
    control_points.column_mut(polygon_segments).copy_from(&points.matrix().column(polyline_segments));

    if polygon_segments == 1 {
        return Ok(control_points);
    }

    let basis_matrix = knots.calculate_basis_matrix(parameters.vector());
    let residuals = calculate_residuals(points, &basis_matrix);
    let constant_terms = calculate_constant_terms(&residuals, &basis_matrix);
    // The internal parameters and the internal basis functions form the system of the internal control points.
    let internal_basis_matrix = basis_matrix.view((1, 1), (polyline_segments - 1, polygon_segments - 1)).into_owned();

    let svd = decompose_normal_matrix(knots, &internal_basis_matrix, &penalization)?;
    let internal_control_points = svd
        .solve(&constant_terms.transpose(), f64::EPSILON.sqrt())
        .expect("the SVD was computed with both U and V^T")
        .transpose();

    for i in 1..=polygon_segments - 1 {
        control_points.column_mut(i).copy_from(&internal_control_points.column(i - 1));
    }

    Ok(control_points)
}

/// Returns the residual vectors R: the internal data points reduced by the contributions
/// of the two fixed end control points.
fn calculate_residuals(points: &DataPoints, basis_matrix: &DMatrix<f64>) -> DMatrix<f64> {
    let polygon_segments = basis_matrix.ncols() - 1;
    let polyline_segments = points.polyline_segments();

    let mut residuals = DMatrix::zeros(points.dimension(), polyline_segments + 1);

    for g in 1..=polyline_segments - 1 {
        residuals.column_mut(g).copy_from(&points.matrix().column(g));
        residuals.column_mut(g).sub_assign(basis_matrix[(g, 0)] * points.matrix().column(0));
        residuals
            .column_mut(g)
            .sub_assign(basis_matrix[(g, polygon_segments)] * points.matrix().column(polyline_segments));
    }

    residuals
}

fn calculate_constant_terms(residuals: &DMatrix<f64>, basis_matrix: &DMatrix<f64>) -> DMatrix<f64> {
    let polygon_segments = basis_matrix.ncols() - 1;
    let polyline_segments = basis_matrix.nrows() - 1;
    let dimension = residuals.nrows();

    let mut constant_terms = DMatrix::zeros(dimension, polygon_segments - 1);

    let mut sum = DVector::zeros(dimension);
    for i in 1..=polygon_segments - 1 {
        sum *= 0.0;

        for g in 1..=polyline_segments - 1 {
            sum += basis_matrix[(g, i)] * residuals.column(g);
        }
        constant_terms.column_mut(i - 1).copy_from(&sum);
    }

    constant_terms
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
    fn unpenalized_linear() {
        let points = DataPoints::new(dmatrix![
            1., 2., 3., 4., 5.;
            1., 2., 3., 4., 5.;
        ]);

        let parameters = Parameters::generate(&points, EquallySpaced).unwrap();
        let knots = Knots::generate(1, points.polyline_segments(), &parameters, Uniform).unwrap();
        assert_eq!(crate::fit::loose::fit(&knots, &points, &parameters, None).unwrap(), *points.matrix());
    }

    #[test]
    fn penalized_linear() {
        let points = DataPoints::new(dmatrix![
            1., 2., 3., 4., 5.;
            1., 2., 3., 4., 5.;
        ]);

        let parameters = Parameters::generate(&points, ChordLength).unwrap();
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
        let parameters = Parameters::generate(&data_points, EquallySpaced).unwrap();
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

        let parameters = Parameters::generate(&data_points, EquallySpaced).unwrap();
        let knots = Knots::generate(degree, data_points.polyline_segments(), &parameters, Uniform).unwrap();
        let points =
            fit(&knots, &data_points, &parameters, Some(Penalization { strength: 1.0, difference_order: 2 })).unwrap();
        let curve = Curve::new(knots, ControlPoints::new(points)).unwrap();

        assert_relative_eq!(curve.evaluate(0.5).unwrap(), dvector![0.0, 0.0], epsilon = f64::EPSILON.sqrt());
    }
}
