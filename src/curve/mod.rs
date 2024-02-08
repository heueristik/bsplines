//#![warn(missing_docs)]
//#![warn(missing_doc_code_examples)]
#![cfg_attr(feature = "doc-images",
cfg_attr(all(),
doc = ::embed_doc_image::embed_image!("eq-curve", "doc-images/equations/curve.svg")))]
//! Implements the B-spline curve.
//!
//! A B-spline curve can be defined by
//!
//! ![B-spline curve][eq-curve]
//!
//! with the
//! - parameter `u ∈ [0,1]` defining a point on the curve,
//! - derivative order `k`,
//! - number of polynomial spline segments `n`,
//! - spline degree `p`,
//! - `k`-th derivative [knot vector][knots] `U`,
//! - `n+1-k` [spline basis function][basis] `N` of degree `p` defined by the [knot vector][knots] `U`, and
//! - `n+1-k`, `N`-dimensional [control points][points] `P`.

use embed_doc_image::embed_doc_image;
use thiserror::Error;

use crate::{
    curve::{
        knots::Knots,
        points::{methods::fit::FitError, ControlPoints, Points},
    },
    manipulation::{
        insert::{insert, InsertError},
        merge::{merge, merge_with_constraints, ConstrainedCurve, Constraints},
        split::{split, SplitError},
    },
    types::VecD,
};

pub mod basis;
pub mod generation;
pub mod knots;
pub mod parameters;
pub mod points;

#[embed_doc_image("spline", "doc-images/plots/derivatives.svg")]
#[derive(Debug, Clone)]
pub struct Curve {
    pub knots: Knots,
    pub points: ControlPoints,
}

#[derive(Error, Debug, PartialEq)]
pub enum CurveError {
    #[error("Parameter `u = {u}` lies outside the interval `[{lower_bound}, {upper_bound}]`.")]
    ParameterOutOfBounds { u: f64, lower_bound: f64, upper_bound: f64 },
    #[error(
        "The number of polynomial segments `n = {n}` of the curve must be \
        greater than or equal to its polynomial degree `p = {p}."
    )]
    DegreeAndSegmentsMismatch { p: usize, n: usize },

    #[error("The derivative order `k = {k}` cannot be greater than curve degree `p = {p}.")]
    DegreeAndDerivativeOrderMismatch { p: usize, k: usize },

    #[error("The maximal derivative order k_max = {k_max} cannot be greater than the spline degree p = {p}.")]
    DerivativeNotAvailable { k_max: usize, p: usize },

    #[error("Degree `p = {p}` is too low and must be greater than `{limit}`")]
    DegreeTooLow { p: usize, limit: usize },

    #[error("Curve generation failed with error {err}.")]
    FitError { err: FitError },
    // TODO
    //#[error("Knot generation failed with error {err}.")]
    //KnotError { err: KnotError },
}

impl Curve {
    /// Returns a B-Spline
    ///
    /// # Arguments
    ///
    /// * `degree` - The degree of the spline
    ///
    /// # Examples
    /// ```
    /// use nalgebra::{dmatrix, dvector};
    /// use bsplines::curve::Curve;
    /// use bsplines::curve::generation::{generate, Generation::Manual};
    /// use bsplines::curve::knots;
    /// use bsplines::curve::points::ControlPoints;
    ///
    /// // Create a coordinate matrix containing with five 3D points.
    /// let points = ControlPoints::new(dmatrix![
    /// // 1    2    3    4    5
    ///  -2.0,-2.0,-1.0, 0.5, 1.5; // x
    ///  -1.0, 0.0, 1.0, 1.0, 2.0; // y
    ///   0.0, 0.5, 1.5,-0.5,-1.0; // z
    /// ]);
    /// let degree = 2;
    /// let knots = knots::methods::uniform(degree, points.segments()).unwrap();
    /// let curve = Curve::new(knots, points).unwrap();
    /// println!("{:?}", curve.evaluate(0.5));
    /// ```
    pub fn new(knots: Knots, points: ControlPoints) -> Result<Self, CurveError> {
        // TODO more sanity checks

        match (knots.degree(), points.segments()) {
            (p, n) if n < p => Err(CurveError::DegreeAndSegmentsMismatch { p, n }),
            _ => {
                let mut c = Self { knots, points };
                c.calculate_derivatives();
                Ok(c)
            }
        }
    }

    pub fn degree(&self) -> usize {
        self.knots.degree()
    }

    pub fn segments(&self) -> usize {
        self.points.segments()
    }

    /// Returns the dimension of the curve.
    pub fn dimension(&self) -> usize {
        self.points.dimension()
    }

    pub fn evaluate(&self, u: f64) -> Result<VecD, CurveError> {
        self.evaluate_derivative(u, 0)
    }

    pub fn evaluate_derivative(&self, u: f64, k: usize) -> Result<VecD, CurveError> {
        if !(0.0..=1.0).contains(&u) {
            return Err(CurveError::ParameterOutOfBounds { u, lower_bound: 0.0, upper_bound: 1.0 });
        }

        let p = self.degree();

        let mut value = VecD::zeros(self.points.dimension());

        if k <= p {
            let n = self.segments();
            let l = self.knots.find_idx(u, k, knots::DomainKnotComparatorType::LeftOrEqual);

            for i in l - (p - k)..=n - k {
                value += self.knots.evaluate(k, i, p, u) * self.points.matrix_derivative(k).column(i);
            }
        }
        Ok(value)
    }

    pub fn max_derivative(&self) -> usize {
        let kmax_knots = self.knots.max_derivative();
        let kmax_points = self.points.max_derivative();

        assert_eq!(
            kmax_knots, kmax_points,
            "The available derivatives of the knots and control points do not match up {} != {}.",
            kmax_knots, kmax_points
        );

        kmax_knots
    }

    /// Reverses the curve.
    pub fn reverse(&mut self) -> &mut Self {
        self.knots.reverse();
        self.points.reverse();
        self
    }

    /// Prepends another curve.
    ///
    /// The end of another curve is attached to the beginning of this curve,
    /// while maintaining continuity of all derivatives
    /// This affects the first and last `p` control points of the two curves, respectively,
    /// and removes `p` control points in total.
    ///
    /// # Examples
    ///
    /// ```
    /// use approx::relative_eq;
    /// use nalgebra::dmatrix;
    /// use bsplines::curve::Curve;
    /// use bsplines::curve::generation::generate;
    /// use bsplines::curve::generation::Generation::Manual;
    /// use bsplines::curve::knots::Generation::Uniform;
    /// use bsplines::curve::points::{ControlPoints, Points};
    ///
    /// let mut a = generate(Manual {
    ///             degree: 2,
    ///             points: ControlPoints::new(dmatrix![ 1.0, 2.0, 3.0;]),
    ///             knots: Uniform,
    ///         }).unwrap();
    /// let b = generate(Manual {
    ///             degree: 2,
    ///             points: ControlPoints::new(dmatrix![-3.0,-2.0,-1.0;]),
    ///             knots: Uniform,
    ///         }).unwrap();
    ///  let c = a.prepend(&b);
    ///
    ///  relative_eq!(c.points.matrix(), &dmatrix![-3.0,-2.0, 2.0, 3.0;], epsilon = f64::EPSILON);
    /// ```
    pub fn prepend(&mut self, other: &Self) -> &mut Self {
        let c = merge(other, self).unwrap();
        self.knots = c.knots;
        self.points = c.points;
        self
    }

    /// Prepends another curve with maximally `p-1` constraints.
    pub fn prepend_constrained(
        &mut self,
        constraints_self: Constraints,
        other: &Self,
        constraints_other: Constraints,
    ) -> &mut Self {
        let c = merge_with_constraints(
            &ConstrainedCurve { curve: other, constraints: constraints_self },
            &ConstrainedCurve { curve: self, constraints: constraints_other },
        )
        .unwrap();
        self.knots = c.knots;
        self.points = c.points;
        self
    }

    /// Appends another curve.
    ///
    /// The end of this curve is attached to the beginning of the other curve,
    /// while maintaining continuity of all derivatives
    /// This affects the first and last `p` control points of the two curves, respectively,
    /// and removes `p` control points in total.
    ///
    /// # Examples
    ///
    /// ```
    /// use approx::relative_eq;
    /// use nalgebra::dmatrix;
    /// use bsplines::curve::Curve;
    /// use bsplines::curve::generation::generate;
    /// use bsplines::curve::generation::Generation::Manual;
    /// use bsplines::curve::knots::Generation::Uniform;
    /// use bsplines::curve::points::{ControlPoints, Points};
    ///
    /// let mut a = generate(Manual {
    ///             degree: 2,
    ///             points: ControlPoints::new(dmatrix![-3.0,-2.0,-1.0;]),
    ///             knots: Uniform,
    ///         }).unwrap();
    /// let b = generate(Manual {
    ///             degree: 2,
    ///             points: ControlPoints::new(dmatrix![ 1.0, 2.0, 3.0;]),
    ///             knots: Uniform,
    ///         }).unwrap();
    ///
    ///  let c = a.append(&b);
    ///
    ///  relative_eq!(c.points.matrix(), &dmatrix![-3.0,-2.0, 2.0, 3.0;], epsilon = f64::EPSILON);
    /// ```
    pub fn append(&mut self, other: &Self) -> &mut Self {
        let c = merge(self, other).unwrap();
        self.knots = c.knots;
        self.points = c.points;
        self
    }

    /// Appends another curve with maximally `p-1` constraints.
    pub fn append_constrained(
        &mut self,
        constraints_self: Constraints,
        other: &Self,
        constraints_other: Constraints,
    ) -> &mut Self {
        let c = merge_with_constraints(
            &ConstrainedCurve { curve: self, constraints: constraints_self },
            &ConstrainedCurve { curve: other, constraints: constraints_other },
        )
        .unwrap();
        self.knots = c.knots;
        self.points = c.points;
        self
    }

    /// Splits a curve into two at parameter `u`.
    ///
    /// # Arguments
    /// * `u` - The parameter`u` that must lie in the interval `(0,1)`.
    pub fn split(&self, u: f64) -> Result<(Self, Self), SplitError> {
        split(self, u)
    }

    /// Inserts a knot into the curve at parameter `u`.
    ///
    /// # Arguments
    /// * `u` - The parameter`u` that must lie in the interval `(0,1)`.
    pub fn insert(&mut self, u: f64) -> Result<&mut Self, InsertError> {
        self.insert_times(u, 1)?;
        Ok(self)
    }

    /// Inserts a knot `x` times into the curve at parameter `u`.
    ///
    /// # Arguments
    /// * `u` - The parameter`u` that must lie in the interval `(0,1)`.
    /// * `x` - The number of insertions of parameter `u`.
    pub fn insert_times(&mut self, u: f64, x: usize) -> Result<&mut Self, InsertError> {
        for _ in 0..x {
            insert(self, u)?;
        }
        Ok(self)
    }

    pub(crate) fn calculate_derivatives(&mut self) {
        self.knots.derive();
        self.points.derive(&self.knots);
    }

    /// Returns the curve describing the `k`-th derivative of the current one.
    /// # Arguments
    ///
    /// * `k` - The derivative
    pub fn get_derivative_curve(&self, k: usize) -> Self {
        let knots = Knots::new(self.degree() - k, self.knots.vector_derivative(k).clone());
        let points = ControlPoints::new(self.points.matrix_derivative(k).clone());
        Curve { knots, points }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use nalgebra::{dmatrix, dvector};
    use rstest::fixture;

    use points::DataPoints;

    use crate::curve::{
        generation::{
            generate,
            Generation::{Interpolation, Manual},
        },
        knots::Generation::Uniform,
    };

    use super::*;

    #[fixture]
    /// A one-dimensional, linear test curve with default degree two.
    fn c(#[default(2)] degree: usize) -> Curve {
        let c = generate(Manual {
            degree,
            points: ControlPoints::new(dmatrix![
                1., 3., 5.;
                2., 4., 6.;
            ]),
            knots: Uniform,
        })
        .unwrap();
        assert_eq!(c.knots.vector(), &dvector![0., 0., 0., 1., 1., 1.]);
        c
    }

    mod evaluate {
        use rstest::rstest;

        use super::*;

        #[rstest]
        fn non_existing_derivative(c: Curve) {
            let k = 3;
            assert_eq!(c.evaluate_derivative(0.5, k), Ok(dvector![0., 0.]));
        }

        #[rstest] //TODO test for orders > 0
        fn outside_lower_bound(c: Curve) {
            let u = -0.1;
            assert_eq!(
                c.evaluate_derivative(u, 0),
                Err(CurveError::ParameterOutOfBounds { u, lower_bound: 0.0, upper_bound: 1.0 })
            );
        }

        #[rstest]
        fn outside_upper_bound(c: Curve) {
            let u = 1.1;
            assert_eq!(
                c.evaluate_derivative(u, 0),
                Err(CurveError::ParameterOutOfBounds { u, lower_bound: 0.0, upper_bound: 1.0 })
            );
        }

        #[test]
        fn derivative_out_of_bounds_error() {
            let p = 2;
            let points = dmatrix![1., 1., 1., 1.;];
            let mut c = generate(Manual { degree: p, points: ControlPoints::new(points), knots: Uniform }).unwrap();

            c.insert(0.5).unwrap();

            assert_eq!(c.evaluate_derivative(0.9, 1).unwrap(), dvector![0.]);
        }

        #[test]
        fn evaluate_p_repeated_knots() {
            let p = 3;
            let points = dmatrix![-1., -0.5, 0.5, 1.;];
            let mut c = generate(Manual { degree: p, points: ControlPoints::new(points), knots: Uniform }).unwrap();
            let u = 0.5;
            let expectedEvaluationResult = dvector![0.0];
            assert_eq!(c.knots.vector(), &dvector![0., 0., 0., 0., 1., 1., 1., 1.]);
            assert_eq!(c.evaluate(0.0).unwrap(), dvector![-1.]);
            assert_eq!(c.evaluate(1.0).unwrap(), dvector![1.]);

            insert(&mut c, u).unwrap();
            assert_eq!(c.knots.vector(), &dvector![0., 0., 0., 0., u, 1., 1., 1., 1.]);
            assert_eq!(c.points.matrix(), &dmatrix![-1., -0.75, 0.0, 0.75, 1.;]);
            assert_eq!(c.evaluate(u).unwrap(), expectedEvaluationResult);
            assert_eq!(c.evaluate(0.0).unwrap(), dvector![-1.]);
            assert_eq!(c.evaluate(1.0).unwrap(), dvector![1.]);

            insert(&mut c, u).unwrap();
            assert_eq!(c.knots.vector(), &dvector![0., 0., 0., 0., u, u, 1., 1., 1., 1.]);
            assert_eq!(c.points.matrix(), &dmatrix![-1., -0.75, -0.375, 0.375, 0.75, 1.;]);
            //assert_eq!(c.evaluate_naive(u).unwrap(), expectedEvaluationResult);
            assert_eq!(c.evaluate(u).unwrap(), expectedEvaluationResult);
            assert_eq!(c.evaluate(0.0).unwrap(), dvector![-1.]);
            assert_eq!(c.evaluate(1.0).unwrap(), dvector![1.]);

            insert(&mut c, u).unwrap();
            assert_eq!(c.knots.vector(), &dvector![0., 0., 0., 0., u, u, u, 1., 1., 1., 1.]);
            assert_eq!(c.points.matrix(), &dmatrix![-1., -0.75, -0.375, 0.0, 0.375, 0.75, 1.;]);
            //assert_eq!(c.evaluate_naive(u).unwrap(), expectedEvaluationResult);
            assert_eq!(c.evaluate(u).unwrap(), expectedEvaluationResult);
            assert_eq!(c.evaluate(0.0).unwrap(), dvector![-1.]);
            assert_eq!(c.evaluate(1.0).unwrap(), dvector![1.]);
        }

        #[rstest]
        fn start(c: Curve) {
            assert_eq!(c.evaluate_derivative(0., 0).unwrap(), dvector![1., 2.])
        }

        #[rstest]
        fn middle(c: Curve) {
            assert_eq!(c.evaluate_derivative(0.5, 0).unwrap(), dvector![3., 4.])
        }

        #[rstest]
        fn end(c: Curve) {
            assert_eq!(c.evaluate_derivative(1., 0).unwrap(), dvector![5., 6.])
        }
    }

    #[test]
    fn reverse() {
        let mut c = generate(Manual {
            degree: 2,
            points: ControlPoints::new(dmatrix![
                1., 3., 5.;
                2., 4., 6.;
            ]),
            knots: Uniform,
        })
        .unwrap();

        let knots_before = c.knots.vector().clone();
        let points_before = c.points.matrix().clone();
        c.reverse();

        let points_after = dmatrix![
                 5., 3., 1.;
                 6., 4., 2.;
        ];
        assert_eq!(c.knots.vector(), &knots_before);
        assert_eq!(c.points.matrix(), &points_after);

        c.reverse();

        assert_eq!(c.knots.vector(), &knots_before);
        assert_eq!(c.points.matrix(), &points_before);
    }

    #[test]
    fn interpolate_linear() {
        let points = DataPoints::new(dmatrix![
            1., 2., 3., 4.;
            1., 2., 3., 4.;
        ]);

        let c = generate(Interpolation { degree: 1, points: &points }).unwrap();

        assert_eq!(c.evaluate(0.0).unwrap(), dvector![1., 1.]);
        assert_relative_eq!(c.evaluate(1. / 3.).unwrap(), dvector![2., 2.], epsilon = f64::EPSILON.sqrt());
        assert_eq!(c.evaluate(2. / 3.).unwrap(), dvector![3., 3.]);
        assert_eq!(c.evaluate(1.0).unwrap(), dvector![4., 4.]);
    }
}
