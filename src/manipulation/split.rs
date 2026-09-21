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

use crate::{
    Curve,
    error::{Error, Result},
    knots::{Knots, normalize},
    manipulation::insert::insert,
    points::{ControlPoints, Points},
    types::{MatD, VecD, VecHelpers},
};

pub fn split(c: &Curve, u: f64) -> Result<(Curve, Curve)> {
    split_and_normalize(c, u, (true, true))
}

pub fn split_and_normalize(c: &Curve, u: f64, normalize_knot_vectors: (bool, bool)) -> Result<(Curve, Curve)> {
    if u <= 0.0 || u >= 1.0 {
        return Err(Error::OutsideDomainInterior { u, min: 0.0, max: 1.0 });
    }

    let p = c.degree();

    let mut bs_inserted = c.clone();

    let l = c.knots.find_span(u, 0);
    let multiplicity = c.knots.vector().iter().skip(l).take_while(|&&x| x == u).count();

    if multiplicity > p {
        return Err(Error::MultiplicityExceedsDegree { u, multiplicity, degree: p });
    }

    for _ in 0..p - multiplicity {
        insert(&mut bs_inserted, u)?;
    }

    let knots = bs_inserted.knots.vector();
    let points = bs_inserted.points.matrix();

    if multiplicity > 0 {
        let left = {
            // TODO reduce index calcs
            let mut left_knots = VecD::zeros(l + p + 1);
            left_knots.head_mut(l + p).copy_from(&knots.head(l + p));
            left_knots[l + p] = u;

            if normalize_knot_vectors.0 {
                normalize(&mut left_knots);
            }
            let bot_cols = left_knots.len() - (p + 2) + 1;
            let left_points: MatD = points.columns(0, bot_cols).into();

            Curve::new(Knots::new(p, left_knots), ControlPoints::new(left_points))?
        };

        let right = {
            let mut right_knots = VecD::zeros(knots.len() + 1 - l);
            right_knots[0] = u;
            right_knots.tail_mut(knots.len() - l).copy_from(&knots.tail(knots.len() - l));

            if normalize_knot_vectors.1 {
                normalize(&mut right_knots);
            }
            let bot_cols = right_knots.len() - (p + 2) + 1;
            let right_points: MatD = points.columns(points.ncols() - bot_cols, bot_cols).into();

            Curve::new(Knots::new(p, right_knots), ControlPoints::new(right_points))?
        };
        Ok((left, right))
    } else {
        let left = {
            let mut left_knots = VecD::zeros(l + p + 1 + 1);
            left_knots.head_mut(l + p + 1).copy_from(&knots.head(l + p + 1));
            left_knots[l + p + 1] = u;

            if normalize_knot_vectors.0 {
                normalize(&mut left_knots);
            }

            let top_cols = left_knots.len() + 1 - (p + 2);
            let left_points: MatD = points.columns(0, top_cols).into();

            Curve::new(Knots::new(p, left_knots), ControlPoints::new(left_points))?
        };

        let right = {
            let mut right_knots = VecD::zeros(knots.len() + 1 - (l + 1)); // length of knots - the elements that occur before the
            // split idx
            right_knots[0] = u;
            right_knots.tail_mut(knots.len() - (l + 1)).copy_from(&knots.tail(knots.len() - (l + 1)));

            if normalize_knot_vectors.1 {
                normalize(&mut right_knots);
            }
            let bot_cols = right_knots.len() + 1 - (p + 2);
            let right_points: MatD = points.columns(points.ncols() - bot_cols, bot_cols).into();

            Curve::new(Knots::new(p, right_knots), ControlPoints::new(right_points))?
        };
        Ok((left, right))
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use nalgebra::{dmatrix, dvector};
    use rstest::{fixture, rstest};

    use crate::{
        generation::{Generation::Manual, generate},
        knots::KnotGeneration::Uniform,
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
        assert_eq!(res.unwrap_err(), Error::OutsideDomainInterior { u, min: 0.0, max: 1.0 });
    }

    #[rstest]
    fn cannot_split_end(c: Curve) {
        let u = 1.0;
        let res = split_and_normalize(&c, u, (true, true));
        assert!(res.is_err());
        assert_eq!(res.unwrap_err(), Error::OutsideDomainInterior { u, min: 0.0, max: 1.0 });
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
