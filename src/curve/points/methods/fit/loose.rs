use crate::{
    curve::{
        knots::Knots,
        parameters::Parameters,
        points::{
            methods::fit::{compute_svd, difference_operator, input_checks, FitError, Penalization},
            DataPoints, Points,
        },
    },
    types::MatD,
};

pub fn fit(
    knots: &Knots,
    points: &DataPoints,
    params: &Parameters,
    penalization: Option<Penalization>,
) -> Result<MatD, FitError> {
    input_checks(knots, points, params, &penalization)?;

    let Nmat = calculate_coefficient_matrix(knots, points, params);

    let svd = compute_svd(knots, &Nmat, &penalization, Box::new(calculate_finite_difference_matrix))?;
    let mat = Nmat.transpose() * points.matrix().transpose();
    let control_points = svd.solve(&mat, f64::EPSILON.sqrt()).unwrap().transpose();

    Ok(control_points)
}

fn calculate_coefficient_matrix(knots: &Knots, points: &DataPoints, params: &Parameters) -> MatD {
    let p = knots.degree();
    let n = knots.segments();
    let m = points.segments();

    let U_bar = params.vector();

    let mut Nmat = MatD::zeros(m + 1, n + 1);
    for g in 0..=m {
        let u = U_bar[g];
        for i in 0..=n {
            Nmat[(g, i)] = knots.evaluate(0, i, p, u);
        }
    }
    Nmat
}

fn calculate_finite_difference_matrix(kappa: usize, knots: &Knots) -> MatD {
    let n = knots.segments();
    assert!(kappa <= n, "The parameter kappa = {} must be greater than the number of spline segments n = {}", kappa, n);

    let mut delta_mat = MatD::zeros(n + 1 - kappa, n + 1);

    for i in 0..=n - kappa {
        for j in 0..=n {
            delta_mat[(i, j)] = difference_operator(i, j, kappa) as f64;
        }
    }

    delta_mat
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use nalgebra::{dmatrix, dvector};

    use crate::curve::{
        knots,
        knots::Method::{Averaging, Uniform},
        parameters,
        parameters::Method::{ChordLength, EquallySpaced},
        points::{methods::fit::test_data_points, ControlPoints},
        Curve,
    };

    use super::*;

    #[test]
    fn calculate_finite_difference_matrix_kappa1_test() {
        let knots = Knots::new(1, dvector![0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0]);
        let mat = calculate_finite_difference_matrix(1, &knots);
        let expected = dmatrix![
            -1.0, 1.0, 0.0, 0.0, 0.0;
             0.0,-1.0, 1.0, 0.0, 0.0;
             0.0, 0.0,-1.0, 1.0, 0.0;
             0.0, 0.0, 0.0,-1.0, 1.0;
        ];
        assert_eq!(mat, expected);
        // TODO check
    }

    #[test]
    fn calculate_finite_difference_matrix_kappa2_test() {
        let knots = Knots::new(1, dvector![0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0]);
        let mat = calculate_finite_difference_matrix(2, &knots);
        let expected = dmatrix![
             1.0,-2.0, 1.0, 0.0, 0.0;
             0.0, 1.0,-2.0, 1.0, 0.0;
             0.0, 0.0, 1.0,-2.0, 1.0;
        ];
        assert_eq!(mat, expected);
        // TODO check
    }

    #[test]
    fn unpenalized_linear() {
        let points = DataPoints::new(dmatrix![
            1., 2., 3., 4., 5.;
            1., 2., 3., 4., 5.;
        ]);

        let params = parameters::generate(&points, EquallySpaced);
        let knots = knots::generate(1, points.segments(), &params, Uniform).unwrap();
        assert_eq!(fit(&knots, &points, &params, None).unwrap(), *points.matrix());
    }

    #[test]
    fn penalized_linear() {
        let points = DataPoints::new(dmatrix![
            1., 2., 3., 4., 5.;
            1., 2., 3., 4., 5.;
        ]);

        let params = parameters::generate(&points, ChordLength);
        let knots = knots::generate(1, points.segments(), &params, Averaging).unwrap();
        assert_relative_eq!(
            fit(&knots, &points, &params, Some(Penalization { lambda: 0.5, kappa: 2 })).unwrap(),
            points.matrix(),
            epsilon = f64::EPSILON.sqrt()
        );
    }

    #[test]
    fn unpenalized_nonlinear() {
        let p = 1;
        let data_points = test_data_points(10);

        let n = data_points.segments();
        let params = parameters::generate(&data_points, EquallySpaced);
        let knots = knots::generate(p, n, &params, Uniform).unwrap();

        assert_relative_eq!(
            fit(&knots, &data_points, &params, None).unwrap(),
            data_points.matrix(),
            epsilon = f64::EPSILON.sqrt()
        );
    }

    #[test]
    fn penalized_nonlinear() {
        let p = 2;
        let data_points = test_data_points(10);

        let params = parameters::generate(&data_points, EquallySpaced);
        let knots = knots::generate(p, data_points.segments(), &params, Uniform).unwrap();
        let points = crate::curve::points::methods::fit::fixed::fit(
            &knots,
            &data_points,
            &params,
            Some(Penalization { lambda: 1.0, kappa: 2 }),
        )
        .unwrap();
        let c = Curve::new(knots, ControlPoints::new(points)).unwrap();

        assert_relative_eq!(c.evaluate(0.5).unwrap(), dvector![0.0, 0.0], epsilon = f64::EPSILON.sqrt());
    }
}
