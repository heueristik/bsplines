#![cfg_attr(feature = "doc-images",
cfg_attr(all(),
doc = ::embed_doc_image::embed_image!("insert-before", "doc-images/plots/manipulation/insert-before.svg"),
doc = ::embed_doc_image::embed_image!("insert-after", "doc-images/plots/manipulation/insert-after.svg")))]
//! Inserts an additional knot into the curve.
//!
//! | A curve before knot insertion. | The curve after knot insertion at `u=4/5`. |
//! |:------------------------------:|:--------------------------------------------:|
//! | ![][insert-before]             | ![][insert-after]                            |

use std::ops::AddAssign;

use crate::{
    Curve,
    error::{Error, Result},
    points::Points,
    types::MatD,
};

/// Knot insertion algorithm by Boehm
/// `u` the knot to be inserted. The value must be in `u ∈ (0, 1)`
pub fn insert(c: &mut Curve, u: f64) -> Result<()> {
    if u <= 0.0 || u >= 1.0 {
        return Err(Error::OutsideDomainInterior { u, min: 0.0, max: 1.0 });
    }

    let p = c.degree();

    let m = c.knots.multiplicity(u);
    if c.knots.multiplicity(u) > p {
        return Err(Error::MultiplicityExceedsDegree { u, multiplicity: m, degree: p });
    }

    let dim = c.points.dimension();

    let old_knots = c.knots.vector();
    let old_points = c.points.matrix();

    let l = c.knots.find_span(u, 0);

    // Insert u into the knot vector
    let new_knots = old_knots.clone().insert_row(l + 1, u);

    // Compute the new control points.
    // Only the control points `l-p+1` to `l` change.
    let control_point_count = c.points.count();

    let mut new_points = MatD::zeros(dim, control_point_count + 1);

    let top_cols = l - p + 1;
    new_points.columns_mut(0, top_cols).copy_from(&old_points.columns(0, top_cols));

    let bot_cols = control_point_count - l;
    new_points
        .columns_mut(new_points.ncols() - bot_cols, bot_cols)
        .copy_from(&old_points.columns(old_points.ncols() - bot_cols, bot_cols));

    let mut alpha: f64;
    for i in (l - p + 1)..=l {
        alpha = (u - old_knots[i]) / (old_knots[i + p] - old_knots[i]);

        new_points.column_mut(i).add_assign((1. - alpha) * old_points.column(i - 1) + alpha * old_points.column(i));
    }

    c.knots.derivatives[0] = new_knots;
    c.points.derivatives[0] = new_points;
    c.calculate_derivatives();
    Ok(())
}

#[cfg(test)]
mod tests {
    use nalgebra::{dmatrix, dvector};

    use crate::{
        generation::{Generation::Manual, generate},
        knots::KnotGeneration::Uniform,
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
        let expected_evaluation_result = dvector![0.0];

        assert_eq!(c.knots.vector(), &dvector![0., 0., 0., 1., 1., 1.]);
        assert_eq!(c.points.matrix(), &dmatrix![-1., 0., 1.;]);
        //assert_eq!(c.evaluate(u).unwrap(), expected_evaluation_result);

        insert(&mut c, u).unwrap();
        assert_eq!(c.knots.vector(), &dvector![0., 0., 0., u, 1., 1., 1.]);
        assert_eq!(c.points.matrix(), &dmatrix![-1., -0.5, 0.5, 1.;]);
        //assert_eq!(c.evaluate(u).unwrap(), expected_evaluation_result);

        insert(&mut c, u).unwrap();
        assert_eq!(c.knots.vector(), &dvector![0., 0., 0., u, u, 1., 1., 1.]);
        assert_eq!(c.points.matrix(), &dmatrix![-1., -0.5, 0.0, 0.5, 1.;]);
        assert_eq!(c.evaluate(u).unwrap(), expected_evaluation_result);

        insert(&mut c, u).unwrap();
        assert_eq!(c.knots.vector(), &dvector![0., 0., 0., u, u, u, 1., 1., 1.]);
        assert_eq!(c.points.matrix(), &dmatrix![-1., -0.5, 0.0, 0.0, 0.5, 1.;]);
    }
}
