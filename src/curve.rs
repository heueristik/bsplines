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
//! - number of control polygon segments `n`,
//! - spline degree `p`,
//! - `k`-th derivative [knot vector][crate::knots] `U`,
//! - `n+1-k` [spline basis function][crate::basis] `N` of degree `p-k` defined by the [knot vector][crate::knots] `U`,
//!   and
//! - `n+1-k`, `N`-dimensional [control points][crate::points] `P`.

use embed_doc_image::embed_doc_image;

use crate::{
    error::{Error, Result},
    fit::FitBuilder,
    interpolation,
    knots::{self, KnotMethod, Knots},
    manipulation::{
        insert::insert,
        merge::{ConstrainedCurve, Constraints, merge, merge_with_constraints},
        split::split,
    },
    parameters::{self, ParameterMethod},
    points::{ControlPoints, DataPoints, Points},
    types::VecD,
};

#[embed_doc_image("spline", "doc-images/plots/derivatives.svg")]
#[derive(Debug, Clone)]
pub struct Curve {
    pub(crate) knots: Knots,
    pub(crate) points: ControlPoints,
}

impl Curve {
    /// Returns a curve defined by the given knot vector and control points.
    ///
    /// # Examples
    /// ```
    /// use bsplines::{Curve, Knots, points::ControlPoints};
    /// use nalgebra::dmatrix;
    ///
    /// // Create a coordinate matrix containing five 3D points.
    /// let points = ControlPoints::new(dmatrix![
    /// // 1    2    3    4    5
    ///  -2.0,-2.0,-1.0, 0.5, 1.5; // x
    ///  -1.0, 0.0, 1.0, 1.0, 2.0; // y
    ///   0.0, 0.5, 1.5,-0.5,-1.0; // z
    /// ]);
    /// let degree = 2;
    /// let knots = Knots::uniform(degree, points.polygon_segments()).unwrap();
    /// let curve = Curve::new(knots, points).unwrap();
    /// println!("{:?}", curve.evaluate(0.5));
    /// ```
    pub fn new(knots: Knots, points: ControlPoints) -> Result<Self> {
        match (knots.degree(), points.polygon_segments()) {
            (degree, polygon_segments) if polygon_segments < degree => {
                Err(Error::TooFewPolygonSegments { degree, polygon_segments })
            }
            _ => {
                let mut curve = Self { knots, points };
                curve.calculate_derivatives();
                Ok(curve)
            }
        }
    }

    /// Returns a curve of the given degree using the points as control points,
    /// on a clamped, uniform knot vector.
    ///
    /// # Examples
    /// ```
    /// use bsplines::{Curve, points::ControlPoints};
    /// use nalgebra::dmatrix;
    ///
    /// let curve = Curve::with_uniform_knots(2, ControlPoints::new(dmatrix![-2.0,-1.0, 0.5, 1.5;])).unwrap();
    /// ```
    pub fn with_uniform_knots(degree: usize, points: ControlPoints) -> Result<Self> {
        let knots = Knots::uniform(degree, points.polygon_segments())?;
        Self::new(knots, points)
    }

    /// Returns a curve of the given degree interpolating the data points,
    /// using equally spaced parameters and a uniform knot vector.
    ///
    /// # Examples
    /// ```
    /// use bsplines::{Curve, points::DataPoints};
    /// use nalgebra::dmatrix;
    ///
    /// let data = DataPoints::new(dmatrix![
    ///     1.0, 2.0, 3.0, 4.0;
    ///     1.0, 2.0, 3.0, 4.0;
    /// ]);
    /// let curve = Curve::interpolate(&data, 2).unwrap();
    /// ```
    pub fn interpolate(data: &DataPoints, degree: usize) -> Result<Self> {
        Self::interpolate_with(data, degree, ParameterMethod::EquallySpaced, KnotMethod::Uniform)
    }

    /// Returns a curve of the given degree interpolating the data points,
    /// with explicitly chosen parameter and knot generation methods.
    pub fn interpolate_with(
        data: &DataPoints,
        degree: usize,
        parameter_method: ParameterMethod,
        knot_method: KnotMethod,
    ) -> Result<Self> {
        let parameters = parameters::generate(data, parameter_method);
        let knots = knots::generate(degree, data.polyline_segments(), &parameters, knot_method)?;
        let points =
            ControlPoints::new_with_capacity(interpolation::interpolate(&knots, data, &parameters), degree + 1);
        Self::new(knots, points)
    }

    /// Returns a builder for a least-squares fit of the data points
    /// with a curve of the given degree.
    ///
    /// # Examples
    /// ```
    /// use bsplines::{Curve, fit::Penalization, points::DataPoints};
    /// use nalgebra::dmatrix;
    ///
    /// let data = DataPoints::new(dmatrix![
    ///     1.0, 2.0, 3.0, 4.0, 5.0;
    ///     1.0, 2.0, 3.0, 4.0, 5.0;
    /// ]);
    /// let curve = Curve::fit(&data, 2).polygon_segments(3).loose_ends().build().unwrap();
    /// ```
    pub fn fit<'a>(data: &'a DataPoints, degree: usize) -> FitBuilder<'a> {
        FitBuilder::new(data, degree)
    }

    /// Returns the knot vector and its derivatives.
    pub fn knots(&self) -> &Knots {
        &self.knots
    }

    /// Returns the control points and their derivatives.
    pub fn points(&self) -> &ControlPoints {
        &self.points
    }

    /// Returns the degree p of the curve.
    pub fn degree(&self) -> usize {
        self.knots.degree()
    }

    /// Returns the number of segments n of the control polygon.
    pub fn polygon_segments(&self) -> usize {
        self.points.polygon_segments()
    }

    /// Returns the dimension of the curve.
    pub fn dimension(&self) -> usize {
        self.points.dimension()
    }

    /// Evaluates the curve at the parameter `u`.
    pub fn evaluate(&self, u: f64) -> Result<VecD> {
        self.evaluate_derivative(u, 0)
    }

    /// Evaluates the `k`-th derivative of the curve at the parameter `u`:
    ///
    /// C⁽ᵏ⁾(u) = Σᵢ Nᵢ,ₚ₋ₖ(u) · Pᵢ⁽ᵏ⁾
    ///
    /// with the basis functions N of degree p − k on the derivative knot vector
    /// and the control points P of the derivative curve, summed over the local
    /// polynomial segment.
    ///
    /// Derivative orders beyond the degree return the zero vector,
    /// since all higher derivatives of a polynomial of degree p vanish.
    pub fn evaluate_derivative(&self, u: f64, derivative: usize) -> Result<VecD> {
        if !(0.0..=1.0).contains(&u) {
            return Err(Error::OutsideDomain { u, min: 0.0, max: 1.0 });
        }

        let degree = self.degree();

        let mut value = VecD::zeros(self.points.dimension());

        if derivative <= degree {
            let polygon_segments = self.polygon_segments();
            let span = self.knots.find_span(u, derivative);

            for i in span - (degree - derivative)..=polygon_segments - derivative {
                value +=
                    self.knots.evaluate(derivative, i, degree, u) * self.points.matrix_derivative(derivative).column(i);
            }
        }
        Ok(value)
    }

    /// Returns the highest derivative order for which knots and control points are available.
    pub fn max_derivative(&self) -> usize {
        let knots_max = self.knots.max_derivative();
        let points_max = self.points.max_derivative();

        assert_eq!(
            knots_max, points_max,
            "the available derivatives of the knots and control points differ: {} != {}",
            knots_max, points_max
        );

        knots_max
    }

    /// Reverses the curve.
    pub fn reverse(&mut self) -> &mut Self {
        self.knots.reverse();
        self.points.reverse();
        self
    }

    /// Prepends another curve.
    ///
    /// The end of the other curve is attached to the beginning of this curve,
    /// while maintaining continuity of all derivatives.
    /// This affects the first and last `p` control points of the two curves, respectively,
    /// and removes `p` control points in total.
    ///
    /// # Examples
    ///
    /// ```
    /// use approx::assert_relative_eq;
    /// use bsplines::{Curve, points::{ControlPoints, Points}};
    /// use nalgebra::dmatrix;
    ///
    /// let mut curve = Curve::with_uniform_knots(2, ControlPoints::new(dmatrix![ 1.0, 2.0, 3.0;])).unwrap();
    /// let other = Curve::with_uniform_knots(2, ControlPoints::new(dmatrix![-3.0,-2.0,-1.0;])).unwrap();
    /// let merged = curve.prepend(&other).unwrap();
    ///
    /// assert_relative_eq!(merged.points().matrix(), &dmatrix![-3.0,-2.0, 2.0, 3.0;], epsilon = f64::EPSILON.sqrt());
    /// ```
    pub fn prepend(&mut self, other: &Self) -> Result<&mut Self> {
        let merged = merge(other, self)?;
        self.knots = merged.knots;
        self.points = merged.points;
        Ok(self)
    }

    /// Prepends another curve with maximally `p-1` constraints.
    pub fn prepend_constrained(
        &mut self,
        constraints_self: Constraints,
        other: &Self,
        constraints_other: Constraints,
    ) -> Result<&mut Self> {
        let merged = merge_with_constraints(
            &ConstrainedCurve { curve: other, constraints: constraints_other },
            &ConstrainedCurve { curve: self, constraints: constraints_self },
        )?;
        self.knots = merged.knots;
        self.points = merged.points;
        Ok(self)
    }

    /// Appends another curve.
    ///
    /// The end of this curve is attached to the beginning of the other curve,
    /// while maintaining continuity of all derivatives.
    /// This affects the first and last `p` control points of the two curves, respectively,
    /// and removes `p` control points in total.
    ///
    /// # Examples
    ///
    /// ```
    /// use approx::assert_relative_eq;
    /// use bsplines::{Curve, points::{ControlPoints, Points}};
    /// use nalgebra::dmatrix;
    ///
    /// let mut curve = Curve::with_uniform_knots(2, ControlPoints::new(dmatrix![-3.0,-2.0,-1.0;])).unwrap();
    /// let other = Curve::with_uniform_knots(2, ControlPoints::new(dmatrix![ 1.0, 2.0, 3.0;])).unwrap();
    /// let merged = curve.append(&other).unwrap();
    ///
    /// assert_relative_eq!(merged.points().matrix(), &dmatrix![-3.0,-2.0, 2.0, 3.0;], epsilon = f64::EPSILON.sqrt());
    /// ```
    pub fn append(&mut self, other: &Self) -> Result<&mut Self> {
        let merged = merge(self, other)?;
        self.knots = merged.knots;
        self.points = merged.points;
        Ok(self)
    }

    /// Appends another curve with maximally `p-1` constraints.
    pub fn append_constrained(
        &mut self,
        constraints_self: Constraints,
        other: &Self,
        constraints_other: Constraints,
    ) -> Result<&mut Self> {
        let merged = merge_with_constraints(
            &ConstrainedCurve { curve: self, constraints: constraints_self },
            &ConstrainedCurve { curve: other, constraints: constraints_other },
        )?;
        self.knots = merged.knots;
        self.points = merged.points;
        Ok(self)
    }

    /// Splits the curve into two independent curves at the parameter `u`,
    /// normalizing both knot vectors to the domain [0, 1].
    /// The parameter must lie in the domain interior (0, 1).
    pub fn split(&self, u: f64) -> Result<(Self, Self)> {
        split(self, u)
    }

    /// Inserts a knot at the parameter `u` without changing the curve shape.
    /// The parameter must lie in the domain interior (0, 1).
    pub fn insert(&mut self, u: f64) -> Result<&mut Self> {
        self.insert_times(u, 1)?;
        Ok(self)
    }

    /// Inserts a knot at the parameter `u` the given number of times.
    /// The parameter must lie in the domain interior (0, 1).
    pub fn insert_times(&mut self, u: f64, times: usize) -> Result<&mut Self> {
        for _ in 0..times {
            insert(self, u)?;
        }
        Ok(self)
    }

    pub(crate) fn calculate_derivatives(&mut self) {
        self.knots.derive();
        self.points.derive(&self.knots);
    }

    /// Returns the curve describing the `k`-th derivative of this curve.
    pub fn derivative_curve(&self, derivative: usize) -> Self {
        let knots = Knots::new(self.degree() - derivative, self.knots.vector_derivative(derivative).clone());
        let points = ControlPoints::new(self.points.matrix_derivative(derivative).clone());
        Curve { knots, points }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use nalgebra::{dmatrix, dvector};
    use rstest::fixture;

    use crate::points::DataPoints;

    use super::*;

    #[fixture]
    /// A two-dimensional, linear test curve with default degree two.
    fn curve(#[default(2)] degree: usize) -> Curve {
        let curve = Curve::with_uniform_knots(
            degree,
            ControlPoints::new(dmatrix![
                1., 3., 5.;
                2., 4., 6.;
            ]),
        )
        .unwrap();
        assert_eq!(curve.knots.vector(), &dvector![0., 0., 0., 1., 1., 1.]);
        curve
    }

    mod evaluate {
        use rstest::rstest;

        use super::*;

        #[rstest]
        fn evaluate_derivative_returns_zero_above_the_degree(curve: Curve) {
            let derivative = 3;
            assert_eq!(curve.evaluate_derivative(0.5, derivative), Ok(dvector![0., 0.]));
        }

        #[rstest]
        fn evaluate_derivative_errors_below_the_domain(curve: Curve) {
            let u = -0.1;
            assert_eq!(curve.evaluate_derivative(u, 0), Err(Error::OutsideDomain { u, min: 0.0, max: 1.0 }));
        }

        #[rstest]
        fn evaluate_derivative_errors_above_the_domain(curve: Curve) {
            let u = 1.1;
            assert_eq!(curve.evaluate_derivative(u, 0), Err(Error::OutsideDomain { u, min: 0.0, max: 1.0 }));
        }

        #[test]
        fn evaluate_derivative_succeeds_near_the_end_after_insertion() {
            let degree = 2;
            let points = dmatrix![1., 1., 1., 1.;];
            let mut curve = Curve::with_uniform_knots(degree, ControlPoints::new(points)).unwrap();

            curve.insert(0.5).unwrap();

            assert_eq!(curve.evaluate_derivative(0.9, 1).unwrap(), dvector![0.]);
        }

        #[test]
        fn evaluate_is_unchanged_by_repeated_insertion() {
            let degree = 3;
            let points = dmatrix![-1., -0.5, 0.5, 1.;];
            let mut curve = Curve::with_uniform_knots(degree, ControlPoints::new(points)).unwrap();
            let u = 0.5;
            let expected_point = dvector![0.0];
            assert_eq!(curve.knots.vector(), &dvector![0., 0., 0., 0., 1., 1., 1., 1.]);
            assert_eq!(curve.evaluate(0.0).unwrap(), dvector![-1.]);
            assert_eq!(curve.evaluate(1.0).unwrap(), dvector![1.]);

            insert(&mut curve, u).unwrap();
            assert_eq!(curve.knots.vector(), &dvector![0., 0., 0., 0., u, 1., 1., 1., 1.]);
            assert_eq!(curve.points.matrix(), &dmatrix![-1., -0.75, 0.0, 0.75, 1.;]);
            assert_eq!(curve.evaluate(u).unwrap(), expected_point);
            assert_eq!(curve.evaluate(0.0).unwrap(), dvector![-1.]);
            assert_eq!(curve.evaluate(1.0).unwrap(), dvector![1.]);

            insert(&mut curve, u).unwrap();
            assert_eq!(curve.knots.vector(), &dvector![0., 0., 0., 0., u, u, 1., 1., 1., 1.]);
            assert_eq!(curve.points.matrix(), &dmatrix![-1., -0.75, -0.375, 0.375, 0.75, 1.;]);
            assert_eq!(curve.evaluate(u).unwrap(), expected_point);
            assert_eq!(curve.evaluate(0.0).unwrap(), dvector![-1.]);
            assert_eq!(curve.evaluate(1.0).unwrap(), dvector![1.]);

            insert(&mut curve, u).unwrap();
            assert_eq!(curve.knots.vector(), &dvector![0., 0., 0., 0., u, u, u, 1., 1., 1., 1.]);
            assert_eq!(curve.points.matrix(), &dmatrix![-1., -0.75, -0.375, 0.0, 0.375, 0.75, 1.;]);
            assert_eq!(curve.evaluate(u).unwrap(), expected_point);
            assert_eq!(curve.evaluate(0.0).unwrap(), dvector![-1.]);
            assert_eq!(curve.evaluate(1.0).unwrap(), dvector![1.]);
        }

        #[rstest]
        fn start(curve: Curve) {
            assert_eq!(curve.evaluate_derivative(0., 0).unwrap(), dvector![1., 2.])
        }

        #[rstest]
        fn middle(curve: Curve) {
            assert_eq!(curve.evaluate_derivative(0.5, 0).unwrap(), dvector![3., 4.])
        }

        #[rstest]
        fn end(curve: Curve) {
            assert_eq!(curve.evaluate_derivative(1., 0).unwrap(), dvector![5., 6.])
        }
    }

    #[test]
    fn reverse() {
        let mut curve = Curve::with_uniform_knots(
            2,
            ControlPoints::new(dmatrix![
                1., 3., 5.;
                2., 4., 6.;
            ]),
        )
        .unwrap();

        let knots_before = curve.knots.vector().clone();
        let points_before = curve.points.matrix().clone();
        curve.reverse();

        let points_after = dmatrix![
                 5., 3., 1.;
                 6., 4., 2.;
        ];
        assert_eq!(curve.knots.vector(), &knots_before);
        assert_eq!(curve.points.matrix(), &points_after);

        curve.reverse();

        assert_eq!(curve.knots.vector(), &knots_before);
        assert_eq!(curve.points.matrix(), &points_before);
    }

    /// Two quadratic curves with a gap: the left curve ends at −0.5, the right curve starts at 0.5.
    fn curves_with_a_gap() -> (Curve, Curve) {
        let left = Curve::with_uniform_knots(2, ControlPoints::new(dmatrix![-2., -1., -0.5;])).unwrap();
        let right = Curve::with_uniform_knots(2, ControlPoints::new(dmatrix![0.5, 1., 2.;])).unwrap();
        (left, right)
    }

    #[test]
    fn prepend_constrained_keeps_the_constrained_start_of_self_fixed() {
        let (left, mut curve) = curves_with_a_gap();
        let start = curve.evaluate(0.0).unwrap();
        let joint = 0.5;

        let mut unconstrained = curve.clone();
        unconstrained.prepend(&left).unwrap();
        assert_eq!(
            unconstrained.knots.vector(),
            &dvector![0., 0., 0., joint, 1., 1., 1.],
            "the joint is the internal knot"
        );
        assert_ne!(unconstrained.evaluate(joint).unwrap(), start, "the unconstrained merge moves the joint");

        curve
            .prepend_constrained(Constraints { parameters: vec![0.0] }, &left, Constraints { parameters: vec![] })
            .unwrap();

        assert_relative_eq!(curve.evaluate(joint).unwrap(), start, epsilon = f64::EPSILON.sqrt());
    }

    #[test]
    fn append_constrained_keeps_the_constrained_end_of_self_fixed() {
        let (mut curve, right) = curves_with_a_gap();
        let end = curve.evaluate(1.0).unwrap();
        let joint = 0.5;

        let mut unconstrained = curve.clone();
        unconstrained.append(&right).unwrap();
        assert_eq!(
            unconstrained.knots.vector(),
            &dvector![0., 0., 0., joint, 1., 1., 1.],
            "the joint is the internal knot"
        );
        assert_ne!(unconstrained.evaluate(joint).unwrap(), end, "the unconstrained merge moves the joint");

        curve
            .append_constrained(Constraints { parameters: vec![1.0] }, &right, Constraints { parameters: vec![] })
            .unwrap();

        assert_relative_eq!(curve.evaluate(joint).unwrap(), end, epsilon = f64::EPSILON.sqrt());
    }

    #[test]
    fn interpolate_linear() {
        let points = DataPoints::new(dmatrix![
            1., 2., 3., 4.;
            1., 2., 3., 4.;
        ]);

        let curve = Curve::interpolate(&points, 1).unwrap();

        assert_eq!(curve.evaluate(0.0).unwrap(), dvector![1., 1.]);
        assert_relative_eq!(curve.evaluate(1. / 3.).unwrap(), dvector![2., 2.], epsilon = f64::EPSILON.sqrt());
        assert_eq!(curve.evaluate(2. / 3.).unwrap(), dvector![3., 3.]);
        assert_eq!(curve.evaluate(1.0).unwrap(), dvector![4., 4.]);
    }
}
