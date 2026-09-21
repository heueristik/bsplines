use nalgebra::SVD;

use crate::{
    knots::Knots,
    parameters::Parameters,
    points::{DataPoints, Points},
    types::MatD,
};

pub fn interpolate(knots: &Knots, points: &DataPoints, params: &Parameters) -> MatD {
    let p = knots.degree();
    let m = points.polyline_segments();
    // Interpolation uses one control point per data point.
    let n = m;

    let u_bar = params.vector();

    let mut n_mat = MatD::zeros(points.count(), points.count());
    for i in 0..=n {
        for g in 0..=m {
            n_mat[(g, i)] = knots.evaluate(0, i, p, u_bar[g]);
        }
    }

    let svd = SVD::new(n_mat, true, true);
    let mat = points.matrix().transpose();
    svd.solve(&mat, f64::EPSILON.sqrt()).expect("the SVD was computed with both U and V^T").transpose()
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use nalgebra::dmatrix;

    use crate::{knots, knots::KnotMethod::Averaging, parameters, parameters::ParameterMethod::ChordLength};

    use super::*;

    #[test]
    fn linear() {
        let points = DataPoints::new(dmatrix![
            1., 2., 3., 4.;
            1., 2., 3., 4.;
        ]);

        let params = parameters::generate(&points, ChordLength);
        let knots = knots::generate(1, points.polyline_segments(), &params, Averaging).unwrap();

        assert_eq!(interpolate(&knots, &points, &params), *points.matrix());
    }

    #[test]
    fn quadratic() {
        let points = DataPoints::new(dmatrix![
            1., 2., 3., 4.;
            1., 2., 3., 4.;
        ]);
        let params = parameters::generate(&points, ChordLength);
        let knots = knots::generate(2, points.polyline_segments(), &params, Averaging).unwrap();

        assert_relative_eq!(
            interpolate(&knots, &points, &params),
            dmatrix![
                1., 1.75, 3.25, 4.;
                1., 1.75, 3.25, 4.;
            ],
            epsilon = f64::EPSILON.sqrt()
        );
    }
}
