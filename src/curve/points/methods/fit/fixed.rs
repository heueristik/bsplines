use std::ops::SubAssign;

use crate::{
    curve::{
        knots::Knots,
        parameters::Parameters,
        points::{
            methods::fit::{compute_svd, difference_operator, input_checks, FitError, Penalization},
            DataPoints, Points,
        },
    },
    types::{MatD, VecD},
};

pub fn fit(
    knots: &Knots,
    points: &DataPoints,
    params: &Parameters,
    penalization: Option<Penalization>,
) -> Result<MatD, FitError> {
    input_checks(knots, points, params, &penalization)?;

    let Q = generate_qvectors(knots, points, params);
    let Qmat = calculate_constant_terms_matrix(knots, points, params, &Q);
    let Nmat = calculate_coefficient_matrix(knots, points, params);

    let svd = compute_svd(knots, &Nmat, &penalization, Box::new(calculate_finite_difference_matrix))?;
    let internal_control_points = svd.solve(&Qmat.transpose(), f64::EPSILON.sqrt()).unwrap().transpose();

    let n = knots.segments();
    let m = points.segments();
    let mut control_points = MatD::zeros(points.dimension(), n + 1);

    // Set the first and last control point to the original data points
    control_points.column_mut(0).copy_from(&points.get(0));
    control_points.column_mut(n).copy_from(&points.get(m));

    // TODO copy matrix view instead of individual columns
    for i in 1..=n - 1 {
        control_points.column_mut(i).copy_from(&internal_control_points.column(i - 1));
    }

    Ok(control_points)
}

fn generate_qvectors(knots: &Knots, points: &DataPoints, params: &Parameters) -> MatD {
    let p = knots.degree();
    let n = knots.segments();
    let m = points.segments();
    let dim = points.dimension();

    let mut Q = MatD::zeros(dim, m + 1);

    let U_bar = params.vector();

    for g in 1..=m - 1 {
        Q.column_mut(g).copy_from(&points.get(g));
        let u = U_bar[g];

        Q.column_mut(g).sub_assign(knots.evaluate(0, 0, p, u) * points.get(0));
        Q.column_mut(g).sub_assign(knots.evaluate(0, n, p, u) * points.get(m));
    }

    Q
}

fn calculate_constant_terms_matrix(knots: &Knots, points: &DataPoints, params: &Parameters, Q: &MatD) -> MatD {
    let p = knots.degree();
    let n = knots.segments(); //settings.intended_control_points_count - 1;
    let m = points.segments();
    let dim = points.dimension();

    let U_bar = params.vector();

    let mut Qmat = MatD::zeros(dim, n - 1);

    let mut accum = VecD::zeros(dim);
    for i in 1..=n - 1 {
        accum *= 0.0;

        for g in 1..=m - 1 {
            let u = U_bar[g];
            accum += knots.evaluate(0, i, p, u) * Q.column(g);
        }
        Qmat.column_mut(i - 1).copy_from(&accum);
    }

    Qmat
}

fn calculate_coefficient_matrix(knots: &Knots, points: &DataPoints, params: &Parameters) -> MatD {
    let p = knots.degree();
    let n = knots.segments();
    let m = points.segments();

    let U_bar = params.vector();

    let mut Nmat = MatD::zeros(m - 1, n - 1);
    for g in 1..=m - 1 {
        let u = U_bar[g];
        for i in 1..=n - 1 {
            Nmat[(g - 1, i - 1)] = knots.evaluate(0, i, p, u);
        }
    }
    Nmat
}

fn calculate_finite_difference_matrix(kappa: usize, knots: &Knots) -> MatD {
    let n = knots.segments(); //settings.intended_control_points_count - 1;
    assert!(
        kappa <= n - 2,
        "The parameter kappa = {} must be greater than the number of spline segments n - 2 = {}",
        kappa,
        n - 2
    );

    let mut delta_mat = MatD::zeros(n - 1 - kappa, n - 1);

    for i in 0..=n - kappa - 2 {
        for j in 0..=n - 2 {
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
            -1.0, 1.0, 0.0;
             0.0,-1.0, 1.0;
        ];
        assert_eq!(mat, expected);
        // TODO check
    }

    #[test]
    fn calculate_finite_difference_matrix_kappa2_test() {
        let knots = Knots::new(1, dvector![0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0]);
        let mat = calculate_finite_difference_matrix(2, &knots);
        let expected = dmatrix![
             1.0,-2.0, 1.0;
        ];
        assert_eq!(mat, expected);
        // TODO check
    }

    // TODO reuse tests for fixed and loose
    #[test]
    fn unpenalized_linear() {
        let points = DataPoints::new(dmatrix![
            1., 2., 3., 4., 5.;
            1., 2., 3., 4., 5.;
        ]);

        let params = parameters::generate(&points, EquallySpaced);
        let knots = knots::generate(1, points.segments(), &params, Uniform).unwrap();
        assert_eq!(
            crate::curve::points::methods::fit::loose::fit(&knots, &points, &params, None).unwrap(),
            *points.matrix()
        );
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
        let points = fit(&knots, &data_points, &params, Some(Penalization { lambda: 1.0, kappa: 2 })).unwrap();
        let c = Curve::new(knots, ControlPoints::new(points)).unwrap();

        assert_relative_eq!(c.evaluate(0.5).unwrap(), dvector![0.0, 0.0], epsilon = f64::EPSILON.sqrt());
    }
}
