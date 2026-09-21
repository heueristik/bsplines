use crate::{
    error::Result,
    fit::{Penalization, compute_svd, difference_operator, input_checks},
    knots::Knots,
    parameters::Parameters,
    points::{DataPoints, Points},
    types::MatD,
};

pub fn fit(
    knots: &Knots,
    points: &DataPoints,
    parameters: &Parameters,
    penalization: Option<Penalization>,
) -> Result<MatD> {
    input_checks(knots, points, parameters, &penalization)?;

    let basis_matrix = calculate_basis_matrix(knots, points, parameters);

    let svd = compute_svd(knots, &basis_matrix, &penalization, Box::new(calculate_finite_difference_matrix))?;
    let constant_terms = basis_matrix.transpose() * points.matrix().transpose();
    let control_points =
        svd.solve(&constant_terms, f64::EPSILON.sqrt()).expect("the SVD was computed with both U and V^T").transpose();

    Ok(control_points)
}

fn calculate_basis_matrix(knots: &Knots, points: &DataPoints, parameters: &Parameters) -> MatD {
    let degree = knots.degree();
    let polygon_segments = knots.polygon_segments();
    let polyline_segments = points.polyline_segments();

    let u_bar = parameters.vector();

    let mut basis_matrix = MatD::zeros(polyline_segments + 1, polygon_segments + 1);
    for g in 0..=polyline_segments {
        let u = u_bar[g];
        for i in 0..=polygon_segments {
            basis_matrix[(g, i)] = knots.evaluate(0, i, degree, u);
        }
    }
    basis_matrix
}

fn calculate_finite_difference_matrix(kappa: usize, knots: &Knots) -> MatD {
    let polygon_segments = knots.polygon_segments();
    assert!(
        kappa <= polygon_segments,
        "the difference order kappa = {} must not exceed n = {}",
        kappa,
        polygon_segments
    );

    let mut difference_matrix = MatD::zeros(polygon_segments + 1 - kappa, polygon_segments + 1);

    for i in 0..=polygon_segments - kappa {
        for j in 0..=polygon_segments {
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
        Curve, knots,
        knots::KnotMethod::{Averaging, Uniform},
        parameters,
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
            -1.0, 1.0, 0.0, 0.0, 0.0;
             0.0,-1.0, 1.0, 0.0, 0.0;
             0.0, 0.0,-1.0, 1.0, 0.0;
             0.0, 0.0, 0.0,-1.0, 1.0;
        ];
        assert_eq!(matrix, expected);
    }

    #[test]
    fn finite_difference_matrix_kappa_2() {
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

        let parameters = parameters::generate(&points, EquallySpaced);
        let knots = knots::generate(1, points.polyline_segments(), &parameters, Uniform).unwrap();
        assert_eq!(fit(&knots, &points, &parameters, None).unwrap(), *points.matrix());
    }

    #[test]
    fn penalized_linear() {
        let points = DataPoints::new(dmatrix![
            1., 2., 3., 4., 5.;
            1., 2., 3., 4., 5.;
        ]);

        let parameters = parameters::generate(&points, ChordLength);
        let knots = knots::generate(1, points.polyline_segments(), &parameters, Averaging).unwrap();
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
        let parameters = parameters::generate(&data_points, EquallySpaced);
        let knots = knots::generate(degree, polygon_segments, &parameters, Uniform).unwrap();

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

        let parameters = parameters::generate(&data_points, EquallySpaced);
        let knots = knots::generate(degree, data_points.polyline_segments(), &parameters, Uniform).unwrap();
        let points =
            crate::fit::fixed::fit(&knots, &data_points, &parameters, Some(Penalization { lambda: 1.0, kappa: 2 }))
                .unwrap();
        let curve = Curve::new(knots, ControlPoints::new(points)).unwrap();

        assert_relative_eq!(curve.evaluate(0.5).unwrap(), dvector![0.0, 0.0], epsilon = f64::EPSILON.sqrt());
    }
}
