use nalgebra::DMatrix;

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

    let basis_matrix = calculate_basis_matrix(knots, points, parameters);

    let svd =
        decompose_normal_matrix(knots, &basis_matrix, &penalization, Box::new(calculate_finite_difference_matrix))?;
    let constant_terms = basis_matrix.transpose() * points.matrix().transpose();
    let control_points =
        svd.solve(&constant_terms, f64::EPSILON.sqrt()).expect("the SVD was computed with both U and V^T").transpose();

    Ok(control_points)
}

fn calculate_basis_matrix(knots: &Knots, points: &DataPoints, parameters: &Parameters) -> DMatrix<f64> {
    let polygon_segments = knots.polygon_segments();
    let polyline_segments = points.polyline_segments();

    let u_bar = parameters.vector();

    let mut basis_matrix = DMatrix::zeros(polyline_segments + 1, polygon_segments + 1);
    for g in 0..=polyline_segments {
        let u = u_bar[g];
        for i in 0..=polygon_segments {
            basis_matrix[(g, i)] = knots.basis(i, u);
        }
    }
    basis_matrix
}

fn calculate_finite_difference_matrix(difference_order: usize, knots: &Knots) -> DMatrix<f64> {
    let polygon_segments = knots.polygon_segments();
    assert!(
        difference_order <= polygon_segments,
        "the difference order {} must not exceed n = {}",
        difference_order,
        polygon_segments
    );

    let mut difference_matrix = DMatrix::zeros(polygon_segments + 1 - difference_order, polygon_segments + 1);

    for i in 0..=polygon_segments - difference_order {
        for j in 0..=polygon_segments {
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
        let knots = Knots::new(1, dvector![0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0]);
        let matrix = calculate_finite_difference_matrix(1, &knots);
        let expected = dmatrix![
            -1.0, 1.0, 0.0, 0.0, 0.0;
             0.0,-1.0, 1.0, 0.0, 0.0;
             0.0, 0.0,-1.0, 1.0, 0.0;
             0.0, 0.0, 0.0,-1.0, 1.0;
        ];
        assert_eq!(matrix, expected);
    }

    #[test]
    fn finite_difference_matrix_order_2() {
        let knots = Knots::new(1, dvector![0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0]);
        let matrix = calculate_finite_difference_matrix(2, &knots);
        let expected = dmatrix![
             1.0,-2.0, 1.0, 0.0, 0.0;
             0.0, 1.0,-2.0, 1.0, 0.0;
             0.0, 0.0, 1.0,-2.0, 1.0;
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
        assert_eq!(fit(&knots, &points, &parameters, None).unwrap(), *points.matrix());
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
        let points = crate::fit::fixed::fit(
            &knots,
            &data_points,
            &parameters,
            Some(Penalization { strength: 1.0, difference_order: 2 }),
        )
        .unwrap();
        let curve = Curve::new(knots, ControlPoints::new(points)).unwrap();

        assert_relative_eq!(curve.evaluate(0.5).unwrap(), dvector![0.0, 0.0], epsilon = f64::EPSILON.sqrt());
    }
}
