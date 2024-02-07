#![cfg_attr(feature = "doc-images",
cfg_attr(all(),
doc = ::embed_doc_image::embed_image!("split-before", "doc-images/plots/manipulation/split-before.svg"),
doc = ::embed_doc_image::embed_image!("split-after", "doc-images/plots/manipulation/split-after.svg")))]
//! Splits a curve into two independent ones.
//!
//! | A curve before splitting. | The two independent curves after splitting at `u = 1/2`. |
//! |:-------------------------:|:--------------------------------------------------------:|
//! | ![][split-before]         | ![][split-after]                                         |
//!
//! The splitting is conducted by adding the respective knot `p+1`-times, which allows for splitting the knot vector.
//! The knot vector can then be re-normalized on the interval `[0,1]`.

use thiserror::Error;

use crate::{
    curve::{
        knots::{normalize, DomainKnotComparatorType, Knots},
        points::{ControlPoints, Points},
        Curve, CurveError,
    },
    manipulation::insert::insert,
    types::{MatD, VecD, VecHelpers},
};

#[derive(Error, Debug, PartialEq)]
pub enum SplitError {
    #[error("The curve cannot be disconnected. The Multiplicity of `{multiplicity}` at u = {u}` exceeds `p = {p}.")]
    MultiplicityTooHigh { u: f64, p: usize, multiplicity: usize },
    #[error("Parameter `u = {u}` lies outside the interval `({lower_bound}, {upper_bound})`.")]
    OutOfBounds { u: f64, lower_bound: f64, upper_bound: f64 },

    #[error("Curve generation failed with error.")]
    CurveError(#[from] CurveError),
}

pub fn split(c: &Curve, u: f64) -> Result<(Curve, Curve), SplitError> {
    split_and_normalize(c, u, (true, true))
}

pub fn split_and_normalize(
    c: &Curve,
    u: f64,
    normalizeKnotVectors: (bool, bool),
) -> Result<(Curve, Curve), SplitError> {
    if u <= 0.0 || u >= 1.0 {
        return Err(SplitError::OutOfBounds { u, lower_bound: 0.0, upper_bound: 1.0 });
    }

    let p = c.degree();

    let mut bs_inserted = c.clone();

    let l = c.knots.find_idx(u, 0, DomainKnotComparatorType::LeftOrEqual);
    let multiplicity = c.knots.vector().iter().skip(l).take_while(|&&x| x == u).count();

    if multiplicity > p {
        return Err(SplitError::MultiplicityTooHigh { u, p, multiplicity });
    }

    for _ in 0..p - multiplicity {
        insert(&mut bs_inserted, u).unwrap();
    }

    let U = bs_inserted.knots.vector();
    let P = bs_inserted.points.matrix();

    if multiplicity > 0 {
        let left = {
            // TODO reduce index calcs
            let mut U_left = VecD::zeros(l + p + 1);
            U_left.head_mut(l + p).copy_from(&U.head(l + p));
            U_left[l + p] = u;

            if normalizeKnotVectors.0 {
                normalize(&mut U_left);
            }
            let bot_cols = U_left.len() - (p + 2) + 1;
            let P_left: MatD = P.columns(0, bot_cols).into();

            Curve::new(Knots::new(p, U_left), ControlPoints::new(P_left))?
        };

        let right = {
            let mut U_right = VecD::zeros(U.len() + 1 - l);
            U_right[0] = u;
            U_right.tail_mut(U.len() - l).copy_from(&U.tail(U.len() - l));

            if normalizeKnotVectors.1 {
                normalize(&mut U_right);
            }
            let bot_cols = U_right.len() - (p + 2) + 1;
            let P_right: MatD = P.columns(P.ncols() - bot_cols, bot_cols).into();

            Curve::new(Knots::new(p, U_right), ControlPoints::new(P_right))?
        };
        Ok((left, right))
    } else {
        let left = {
            let mut U_left = VecD::zeros(l + p + 1 + 1);
            U_left.head_mut(l + p + 1).copy_from(&U.head(l + p + 1));
            U_left[l + p + 1] = u;

            if normalizeKnotVectors.0 {
                normalize(&mut U_left);
            }

            let top_cols = U_left.len() + 1 - (p + 2);
            let P_left: MatD = P.columns(0, top_cols).into();

            Curve::new(Knots::new(p, U_left), ControlPoints::new(P_left))?
        };

        let right = {
            let mut U_right = VecD::zeros(U.len() + 1 - (l + 1)); // length of U - the elements that occur before the split idx
            U_right[0] = u;
            U_right.tail_mut(U.len() - (l + 1)).copy_from(&U.tail(U.len() - (l + 1)));

            if normalizeKnotVectors.1 {
                normalize(&mut U_right);
            }
            let bot_cols = U_right.len() + 1 - (p + 2);
            let P_right: MatD = P.columns(P.ncols() - bot_cols, bot_cols).into();

            Curve::new(Knots::new(p, U_right), ControlPoints::new(P_right))?
        };
        Ok((left, right))
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use nalgebra::{dmatrix, dvector};
    use rstest::{fixture, rstest};

    use crate::curve::{
        generation::{generate, Generation::Manual},
        knots::Generation::Uniform,
    };

    use super::*;

    #[fixture]
    /// A one-dimensional, linear test curve with default degree two.
    fn c(#[default(2)] degree: usize) -> Curve {
        let c =
            generate(Manual { degree, points: ControlPoints::new(dmatrix![1., 2., 3., 4., 5., 6.;]), knots: Uniform })
                .unwrap();
        assert_eq!(c.knots.vector(), &dvector![0., 0., 0., 0.25, 0.5, 0.75, 1., 1., 1.]);
        c
    }

    #[rstest]
    fn cannot_split_start(c: Curve) {
        let u = 0.0;
        let res = split_and_normalize(&c, u, (true, true));
        assert!(res.is_err());
        assert_eq!(res.unwrap_err(), SplitError::OutOfBounds { u, lower_bound: 0.0, upper_bound: 1.0 });
    }

    #[rstest]
    fn cannot_split_end(c: Curve) {
        let u = 1.0;
        let res = split_and_normalize(&c, u, (true, true));
        assert!(res.is_err());
        assert_eq!(res.unwrap_err(), SplitError::OutOfBounds { u, lower_bound: 0.0, upper_bound: 1.0 });
    }

    /*#[rstest]
    fn cannot_split_wrong_mult() {
        todo!("add multiplicity test");
    }*/

    #[rstest]
    fn split_close_to_start(c: Curve) {
        let eps = f64::EPSILON;
        let (left, right) = split_and_normalize(&c, 0.0 + eps, (true, true)).unwrap();

        assert_relative_eq!(left.knots.vector(), &dvector![0., 0., 0., 1., 1., 1.], epsilon = eps.sqrt());
        assert_relative_eq!(left.points.matrix(), &dmatrix![1., 1., 1.;], epsilon = eps.sqrt());

        assert_relative_eq!(right.knots.vector(), c.knots.vector(), epsilon = eps.sqrt());
        assert_relative_eq!(right.points.matrix(), c.points.matrix(), epsilon = eps.sqrt());
    }

    #[rstest]
    fn split_close_to_end(c: Curve) {
        let eps = f64::EPSILON;
        let (left, right) = split_and_normalize(&c, 1.0 - eps, (true, true)).unwrap();

        assert_relative_eq!(left.knots.vector(), c.knots.vector(), epsilon = eps.sqrt());
        assert_relative_eq!(left.points.matrix(), c.points.matrix(), epsilon = eps.sqrt());

        assert_relative_eq!(right.knots.vector(), &dvector![0., 0., 0., 1., 1., 1.], epsilon = eps.sqrt());
        assert_relative_eq!(right.points.matrix(), &dmatrix![6., 6., 6.;], epsilon = eps.sqrt());
    }

    #[rstest]
    fn normalized(c: Curve) {
        let (left, right) = split_and_normalize(&c, 0.5, (true, true)).unwrap();
        assert_eq!(left.knots.vector(), &dvector![0., 0., 0., 0.5, 1., 1., 1.]);
        assert_eq!(left.points.matrix(), &dmatrix![1., 2., 3., 3.5;]);

        assert_eq!(right.knots.vector(), &dvector![0., 0., 0., 0.5, 1., 1., 1.]);
        assert_eq!(right.points.matrix(), &dmatrix![3.5, 4., 5., 6.;]);
    }

    #[rstest]
    fn unnormalized(c: Curve) {
        let (left, right) = split_and_normalize(&c, 0.5, (false, false)).unwrap();
        assert_eq!(left.knots.vector(), &dvector![0., 0., 0., 0.25, 0.5, 0.5, 0.5]);
        assert_eq!(left.points.matrix(), &dmatrix![1., 2., 3., 3.5;]);

        assert_eq!(right.knots.vector(), &dvector![0.5, 0.5, 0.5, 0.75, 1., 1., 1.]);
        assert_eq!(right.points.matrix(), &dmatrix![3.5, 4., 5., 6.;]);
    }
}
