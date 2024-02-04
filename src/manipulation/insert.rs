#![cfg_attr(feature = "doc-images",
cfg_attr(all(),
doc = ::embed_doc_image::embed_image!("insert-before", "doc-images/plots/manipulation/insert-before.svg"),
doc = ::embed_doc_image::embed_image!("insert-after", "doc-images/plots/manipulation/insert-after.svg")))]
//! Inserts an additional knot into the curve.
//!
//! | A curve before knot insertion. | The curve after knot insertion at `$u=4/5$`. |
//! |:------------------------------:|:--------------------------------------------:|
//! | ![][insert-before]             | ![][insert-after]                            |

use std::ops::AddAssign;

use thiserror::Error;

use crate::{
    curve::{knots::DomainKnotComparatorType, points::Points, Curve},
    types::MatD,
};

#[derive(Error, Debug, PartialEq)]
pub enum InsertError {
    #[error("Parameter `u = {u}` lies outside the interval `({lower_bound}, {upper_bound})`.")]
    OutOfBounds { u: f64, lower_bound: f64, upper_bound: f64 },

    #[error(
        "The knot `u = {u}` has a multiplicity of `m = {m}` already. \
    Therefore, the knot cannot be inserted as this would exceed the maximum \
    multiplicity corresponding to the curve degree with `p = {p}`."
    )]
    MultiplicityError { u: f64, m: usize, p: usize },
}

/// Knot insertion algorithm by Boehm
/// `u` the knot to be inserted. The value must be in `$u\in(0, 1)$`
pub fn insert(c: &mut Curve, u: f64) -> Result<(), InsertError> {
    if u <= 0.0 || u >= 1.0 {
        return Err(InsertError::OutOfBounds { u, lower_bound: 0.0, upper_bound: 1.0 });
    }

    let p = c.degree();

    let m = c.knots.multiplicity(u);
    if c.knots.multiplicity(u) > p {
        return Err(InsertError::MultiplicityError { u, m, p });
    }

    let dim = c.points.dimension();

    let U_old = c.knots.vector();
    let P_old = c.points.matrix();

    let l = c.knots.find_idx(u, 0, DomainKnotComparatorType::LeftOrEqual);

    // Insert u into the knot vector
    let U_new = U_old.clone().insert_row(l + 1, u);

    // Compute the new control points.
    // Only the control points `l-p+1` to `l` change.
    let control_point_count = c.points.count();

    let mut P_new = MatD::zeros(dim, control_point_count + 1);

    let top_cols = l - p + 1;
    P_new.columns_mut(0, top_cols).copy_from(&P_old.columns(0, top_cols));

    let bot_cols = control_point_count - l;
    P_new.columns_mut(P_new.ncols() - bot_cols, bot_cols).copy_from(&P_old.columns(P_old.ncols() - bot_cols, bot_cols));

    let mut alpha: f64;
    for i in (l - p + 1)..=l {
        alpha = (u - U_old[i]) / (U_old[i + p] - U_old[i]);

        P_new.column_mut(i).add_assign((1. - alpha) * P_old.column(i - 1) + alpha * P_old.column(i));
    }

    c.knots.Uk[0] = U_new;
    c.points.Pk[0] = P_new;
    c.calculate_derivatives();
    Ok(())
}

#[cfg(test)]
mod tests {
    use nalgebra::{dmatrix, dvector};

    use crate::curve::{
        generation::{generate, Generation::Manual},
        knots::Generation::Uniform,
        points::ControlPoints,
    };

    use super::*;

    #[test]
    fn degree_1() {
        let mut c =
            generate(Manual { degree: 1, points: ControlPoints::new(dmatrix![-1., 1.;]), knots: Uniform }).unwrap();
        assert_eq!(c.knots.vector(), &dvector![0., 0., 1., 1.]);

        insert(&mut c, 0.5).unwrap();
        assert_eq!(c.knots.vector(), &dvector![0., 0., 0.5, 1., 1.]);
        assert_eq!(c.points.matrix(), &dmatrix![-1., 0., 1.;]);
    }

    #[test]
    fn degree_2() {
        let mut c =
            generate(Manual { degree: 2, points: ControlPoints::new(dmatrix![-1., 0., 1.;]), knots: Uniform }).unwrap();
        assert_eq!(c.knots.vector(), &dvector![0., 0., 0., 1., 1., 1.]);
        assert_eq!(c.points.matrix(), &dmatrix![-1., 0., 1.;]);

        insert(&mut c, 0.5).unwrap();
        assert_eq!(c.knots.vector(), &dvector![0., 0., 0., 0.5, 1., 1., 1.]);
        assert_eq!(c.points.matrix(), &dmatrix![-1., -0.5, 0.5, 1.;]);
    }

    #[test]
    fn degree_2_preexisting_knot() {
        let mut c =
            generate(Manual { degree: 2, points: ControlPoints::new(dmatrix![-1.5, -0.5, 0.5, 1.5;]), knots: Uniform })
                .unwrap();
        assert_eq!(c.knots.vector(), &dvector![0., 0., 0., 0.5, 1., 1., 1.]);
        assert_eq!(c.points.matrix(), &dmatrix![-1.5, -0.5, 0.5, 1.5;]);

        insert(&mut c, 0.5).unwrap();
        assert_eq!(c.knots.vector(), &dvector![0., 0., 0., 0.5, 0.5, 1., 1., 1.]);
        assert_eq!(c.points.matrix(), &dmatrix![-1.5, -0.5, 0.0, 0.5, 1.5;]);
    }

    #[test]
    fn degree_1_repeated_knot() {
        let mut c =
            generate(Manual { degree: 1, points: ControlPoints::new(dmatrix![-1., 1.;]), knots: Uniform }).unwrap();
        assert_eq!(c.knots.vector(), &dvector![0., 0., 1., 1.]);

        insert(&mut c, 0.5).unwrap();
        assert_eq!(c.knots.vector(), &dvector![0., 0., 0.5, 1., 1.]);
        assert_eq!(c.points.matrix(), &dmatrix![-1., 0., 1.;]);

        insert(&mut c, 0.5).unwrap();
        assert_eq!(c.knots.vector(), &dvector![0., 0., 0.5, 0.5, 1., 1.]);
        assert_eq!(c.points.matrix(), &dmatrix![-1., 0., 0., 1.;]);
    }

    #[test]
    fn degree_2_repeated_knots() {
        let mut c =
            generate(Manual { degree: 2, points: ControlPoints::new(dmatrix![-1., 0., 1.;]), knots: Uniform }).unwrap();
        let u = 0.5;
        let expectedEvaluationResult = dvector![0.0];

        assert_eq!(c.knots.vector(), &dvector![0., 0., 0., 1., 1., 1.]);
        assert_eq!(c.points.matrix(), &dmatrix![-1., 0., 1.;]);
        //assert_eq!(c.evaluate(u).unwrap(), expectedEvaluationResult);

        insert(&mut c, u).unwrap();
        assert_eq!(c.knots.vector(), &dvector![0., 0., 0., u, 1., 1., 1.]);
        assert_eq!(c.points.matrix(), &dmatrix![-1., -0.5, 0.5, 1.;]);
        //assert_eq!(c.evaluate(u).unwrap(), expectedEvaluationResult);

        insert(&mut c, u).unwrap();
        assert_eq!(c.knots.vector(), &dvector![0., 0., 0., u, u, 1., 1., 1.]);
        assert_eq!(c.points.matrix(), &dmatrix![-1., -0.5, 0.0, 0.5, 1.;]);
        assert_eq!(c.evaluate(u).unwrap(), expectedEvaluationResult);

        insert(&mut c, u).unwrap();
        assert_eq!(c.knots.vector(), &dvector![0., 0., 0., u, u, u, 1., 1., 1.]);
        assert_eq!(c.points.matrix(), &dmatrix![-1., -0.5, 0.0, 0.0, 0.5, 1.;]);
    }
}
