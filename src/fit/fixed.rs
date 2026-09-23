use nalgebra::DMatrix;

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
    // The internal parameters and the internal basis functions form the system of the internal control points.
    let internal_basis_matrix = basis_matrix.view((1, 1), (polyline_segments - 1, polygon_segments - 1)).into_owned();
    let constant_terms = internal_basis_matrix.transpose() * calculate_residuals(points, &basis_matrix).transpose();

    let svd = decompose_normal_matrix(knots, &internal_basis_matrix, &penalization)?;
    let internal_control_points =
        svd.solve(&constant_terms, f64::EPSILON.sqrt()).expect("the SVD was computed with both U and V^T");
    control_points.columns_mut(1, polygon_segments - 1).tr_copy_from(&internal_control_points);

    Ok(control_points)
}

/// Returns the residual vectors R: the internal data points reduced by the contributions
/// of the two fixed end control points.
fn calculate_residuals(points: &DataPoints, basis_matrix: &DMatrix<f64>) -> DMatrix<f64> {
    let polygon_segments = basis_matrix.ncols() - 1;
    let polyline_segments = points.polyline_segments();
    let point_matrix = points.matrix();

    // The basis functions of the two end control points at the internal parameters.
    let first_basis_values = basis_matrix.view((1, 0), (polyline_segments - 1, 1));
    let last_basis_values = basis_matrix.view((1, polygon_segments), (polyline_segments - 1, 1));

    point_matrix.columns(1, polyline_segments - 1) -
        point_matrix.column(0) * first_basis_values.transpose() -
        point_matrix.column(polyline_segments) * last_basis_values.transpose()
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
