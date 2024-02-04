use nalgebra::SVD;

use crate::{
    curve::{
        knots::Knots,
        parameters::Parameters,
        points::{DataPoints, Points},
    },
    types::MatD,
};

pub fn interpolate(knots: &Knots, points: &DataPoints, params: &Parameters) -> MatD {
    let p = knots.degree();
    let m = points.segments();
    let n = m;

    let Ubar = params.vector();

    let mut Nmat = MatD::zeros(points.count(), points.count());
    for i in 0..=n {
        for g in 0..=m {
            Nmat[(g, i)] = knots.evaluate(0, i, p, Ubar[g]);
        }
    }

    let svd = SVD::new(Nmat, true, true);
    let mat = points.matrix().transpose();
    svd.solve(&mat, f64::EPSILON.sqrt()).unwrap().transpose()
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use nalgebra::dmatrix;

    use crate::curve::{knots, knots::Method::Averaging, parameters, parameters::Method::ChordLength};

    use super::*;

    #[test]
    fn linear() {
        let points = DataPoints::new(dmatrix![
            1., 2., 3., 4.;
            1., 2., 3., 4.;
        ]);

        let params = parameters::generate(&points, ChordLength);
        let knots = knots::generate(1, points.segments(), &params, Averaging).unwrap();

        assert_eq!(interpolate(&knots, &points, &params), points.matrix);
    }

    #[test]
    fn quadratic() {
        let points = DataPoints::new(dmatrix![
            1., 2., 3., 4.;
            1., 2., 3., 4.;
        ]);
        let params = parameters::generate(&points, ChordLength);
        let knots = knots::generate(2, points.segments(), &params, Averaging).unwrap();

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
