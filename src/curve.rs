//! Implements the B-spline curve.

use nalgebra::DVector;

use crate::{
    error::{Error, Result},
    fit::FitBuilder,
    interpolation,
    knots::{KnotMethod, Knots},
    manipulation::{
        insert::insert,
        merge::{Constraints, merge},
        split::split,
    },
    parameters::{ParameterMethod, Parameters},
    points::{ControlPoints, DataPoints, Points},
};

/// A B-spline curve: a parametric function that maps the domain [0, 1] into N-dimensional space.
///
/// The curve and its derivatives follow from
///
/// ![B-spline curve][eq-curve]
///
/// with the parameter u ∈ [0, 1], the derivative order k, the number of polygon segments n,
/// the degree p, the knot vector U of the k-th derivative (see [`Knots`]), the n + 1 − k
/// [basis functions][Knots::basis] N of degree p − k, and the n + 1 − k control points P
/// of the k-th derivative (see [`ControlPoints`]).
///
/// The plot shows a cubic curve in red and its first, second, and third derivative
/// in purple, blue, and teal.
///
/// ![A cubic curve and its derivatives][curve-derivatives]
#[cfg_attr(feature = "doc-images", doc = ::embed_doc_image::embed_image!("eq-curve", "doc-images/equations/curve.svg"))]
#[cfg_attr(feature = "doc-images", doc = ::embed_doc_image::embed_image!("curve-derivatives", "doc-images/plots/derivatives.svg"))]
#[derive(Debug, Clone)]
pub struct Curve {
    pub(crate) knots: Knots,
    pub(crate) control_points: ControlPoints,
}

impl Curve {
    /// Returns a curve defined by the given knot vector and control points. The knot vector must be
    /// clamped and normalized to [0, 1], and its n + p + 2 knots need n + 1 control points with finite
    /// coordinates.
    ///
    /// # Examples
    /// ```
    /// use bsplines::{ControlPoints, Curve, Knots};
    /// use nalgebra::dmatrix;
    ///
    /// // Create a coordinate matrix containing five 3D points.
    /// let control_points = ControlPoints::new(dmatrix![
    /// // 1    2    3    4    5
    ///  -2.0,-2.0,-1.0, 0.5, 1.5; // x
    ///  -1.0, 0.0, 1.0, 1.0, 2.0; // y
    ///   0.0, 0.5, 1.5,-0.5,-1.0; // z
    /// ]);
    /// let degree = 2;
    /// let knots = Knots::uniform(degree, control_points.polygon_segments()).unwrap();
    /// let curve = Curve::new(knots, control_points).unwrap();
    /// println!("{:?}", curve.evaluate(0.5));
    /// ```
    pub fn new(knots: Knots, control_points: ControlPoints) -> Result<Self> {
        if !knots.is_clamped() {
            return Err(Error::UnclampedKnots);
        }
        if !knots.is_normalized() {
            return Err(Error::UnnormalizedKnots);
        }

        let expected = knots.polygon_segments() + 1;
        let count = control_points.count();
        if count != expected {
            return Err(Error::ControlPointCountMismatch { expected, count });
        }
        let points = control_points.matrix();
        if let Some(index) = points.column_iter().position(|point| point.iter().any(|x| !x.is_finite())) {
            return Err(Error::NonFiniteControlPoint { index });
        }

        let mut curve = Self { knots, control_points };
        curve.derive();
        Ok(curve)
    }

    /// Returns a curve of the given degree with the given control points
    /// on a clamped, uniform knot vector.
    ///
    /// # Examples
    /// ```
    /// use bsplines::{ControlPoints, Curve};
    /// use nalgebra::dmatrix;
    ///
    /// let curve = Curve::with_uniform_knots(ControlPoints::new(dmatrix![-2.0,-1.0, 0.5, 1.5;]), 2).unwrap();
    /// ```
    pub fn with_uniform_knots(control_points: ControlPoints, degree: usize) -> Result<Self> {
        let knots = Knots::uniform(degree, control_points.polygon_segments())?;
        Self::new(knots, control_points)
    }

    /// Returns a curve of the given degree interpolating the data points,
    /// using equally spaced parameters and a uniform knot vector.
    ///
    /// # Examples
    /// ```
    /// use bsplines::{Curve, DataPoints};
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
        let parameters = Parameters::generate(data, parameter_method)?;
        let knots = Knots::generate(degree, data.polyline_segments(), &parameters, knot_method)?;
        let control_points = ControlPoints::new(interpolation::interpolate(&knots, data, &parameters)?);
        Self::new(knots, control_points)
    }

    /// Returns a builder for a least-squares fit of the data points
    /// with a curve of the given degree.
    ///
    /// # Examples
    /// ```
    /// use bsplines::{Curve, DataPoints};
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

    /// Returns the knot vector.
    pub fn knots(&self) -> &Knots {
        &self.knots
    }

    /// Returns the control points.
    pub fn control_points(&self) -> &ControlPoints {
        &self.control_points
    }

    /// Returns the degree p of the curve.
    pub fn degree(&self) -> usize {
        self.knots.degree()
    }

    /// Returns the number of segments n of the control polygon.
    pub fn polygon_segments(&self) -> usize {
        self.control_points.polygon_segments()
    }

    /// Returns the dimension of the curve.
    pub fn dimension(&self) -> usize {
        self.control_points.dimension()
    }

    /// Evaluates the curve at the parameter `u`.
    pub fn evaluate(&self, u: f64) -> Result<DVector<f64>> {
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
    pub fn evaluate_derivative(&self, u: f64, derivative: usize) -> Result<DVector<f64>> {
        if !(0.0..=1.0).contains(&u) {
            return Err(Error::OutsideDomain { u, min: 0.0, max: 1.0 });
        }

        let degree = self.degree();

        let mut value = DVector::zeros(self.control_points.dimension());

        if derivative <= degree {
            let basis_degree = degree - derivative;
            let knots = self.knots.vector_derivative(derivative).as_slice();
            // The basis functions that are not zero at u end at the last knot at or below u,
            // limited to the n − k + 1 basis functions of the derivative curve.
            let last = knots
                .partition_point(|&knot| knot <= u)
                .saturating_sub(1)
                .clamp(basis_degree, self.polygon_segments() - derivative);

            for i in last - basis_degree..=last {
                value += self.knots.basis_of_derivative_curve(derivative, i, u) *
                    self.control_points.matrix_derivative(derivative).column(i);
            }
        }
        Ok(value)
    }

    /// Reverses the direction of the curve: the point at the parameter u moves to 1 − u.
    ///
    /// | The curve.          | The reversed curve. |
    /// |:-------------------:|:-------------------:|
    /// | ![][reverse-before] | ![][reverse-after]  |
    #[cfg_attr(feature = "doc-images", doc = ::embed_doc_image::embed_image!("reverse-before", "doc-images/plots/manipulation/reverse-before.svg"))]
    #[cfg_attr(feature = "doc-images", doc = ::embed_doc_image::embed_image!("reverse-after", "doc-images/plots/manipulation/reverse-after.svg"))]
    pub fn reverse(&mut self) -> &mut Self {
        self.knots.reverse();
        self.control_points.reverse();
        self
    }

    /// Prepends another curve: attaches the end of the other curve to the start of this curve
    /// and keeps all derivatives continuous at the joint — see [`Curve::append`].
    ///
    /// # Examples
    ///
    /// ```
    /// use approx::assert_relative_eq;
    /// use bsplines::{ControlPoints, Curve, Points};
    /// use nalgebra::dmatrix;
    ///
    /// let mut curve = Curve::with_uniform_knots(ControlPoints::new(dmatrix![ 1.0, 2.0, 3.0;]), 2).unwrap();
    /// let other = Curve::with_uniform_knots(ControlPoints::new(dmatrix![-3.0,-2.0,-1.0;]), 2).unwrap();
    /// let merged = curve.prepend(&other).unwrap();
    ///
    /// assert_relative_eq!(merged.control_points().matrix(), &dmatrix![-3.0,-2.0, 2.0, 3.0;], epsilon = f64::EPSILON.sqrt());
    /// ```
    pub fn prepend(&mut self, other: &Self) -> Result<&mut Self> {
        let merged = merge(other, self, &Constraints::default())?;
        self.knots = merged.knots;
        self.control_points = merged.control_points;
        Ok(self)
    }

    /// Prepends another curve like [`Curve::prepend`], but keeps the points at the constrained
    /// parameters fixed. The other curve is the left one, this curve is the right one.
    pub fn prepend_constrained(&mut self, other: &Self, constraints: Constraints) -> Result<&mut Self> {
        let merged = merge(other, self, &constraints)?;
        self.knots = merged.knots;
        self.control_points = merged.control_points;
        Ok(self)
    }

    /// Appends another curve: attaches the end of this curve to the start of the other curve
    /// and keeps all derivatives continuous at the joint — see `Tai2003`. The merge moves the last
    /// p control points of this curve and the first p control points of the other curve, and it
    /// removes p control points in total.
    ///
    /// | Two curves.       | The merged curve. |
    /// |:-----------------:|:-----------------:|
    /// | ![][merge-before] | ![][merge-after]  |
    ///
    /// # Examples
    ///
    /// ```
    /// use approx::assert_relative_eq;
    /// use bsplines::{ControlPoints, Curve, Points};
    /// use nalgebra::dmatrix;
    ///
    /// let mut curve = Curve::with_uniform_knots(ControlPoints::new(dmatrix![-3.0,-2.0,-1.0;]), 2).unwrap();
    /// let other = Curve::with_uniform_knots(ControlPoints::new(dmatrix![ 1.0, 2.0, 3.0;]), 2).unwrap();
    /// let merged = curve.append(&other).unwrap();
    ///
    /// assert_relative_eq!(merged.control_points().matrix(), &dmatrix![-3.0,-2.0, 2.0, 3.0;], epsilon = f64::EPSILON.sqrt());
    /// ```
    #[cfg_attr(feature = "doc-images", doc = ::embed_doc_image::embed_image!("merge-before", "doc-images/plots/manipulation/merge-before.svg"))]
    #[cfg_attr(feature = "doc-images", doc = ::embed_doc_image::embed_image!("merge-after", "doc-images/plots/manipulation/merge-after.svg"))]
    pub fn append(&mut self, other: &Self) -> Result<&mut Self> {
        let merged = merge(self, other, &Constraints::default())?;
        self.knots = merged.knots;
        self.control_points = merged.control_points;
        Ok(self)
    }

    /// Appends another curve like [`Curve::append`], but keeps the points at the constrained
    /// parameters fixed. This curve is the left one, the other curve is the right one.
    ///
    /// # Examples
    ///
    /// ```
    /// use approx::assert_relative_eq;
    /// use bsplines::{Constraints, ControlPoints, Curve};
    /// use nalgebra::dmatrix;
    ///
    /// let mut curve = Curve::with_uniform_knots(ControlPoints::new(dmatrix![-2.0,-1.0,-0.5;]), 2).unwrap();
    /// let other = Curve::with_uniform_knots(ControlPoints::new(dmatrix![ 0.5, 1.0, 2.0;]), 2).unwrap();
    /// let end = curve.evaluate(1.0).unwrap();
    ///
    /// // Keep the end of this curve fixed, so the joint at u = 0.5 stays at its old end point.
    /// curve.append_constrained(&other, Constraints { left: vec![1.0], right: vec![] }).unwrap();
    ///
    /// assert_relative_eq!(curve.evaluate(0.5).unwrap(), end, epsilon = f64::EPSILON.sqrt());
    /// ```
    pub fn append_constrained(&mut self, other: &Self, constraints: Constraints) -> Result<&mut Self> {
        let merged = merge(self, other, &constraints)?;
        self.knots = merged.knots;
        self.control_points = merged.control_points;
        Ok(self)
    }

    /// Splits the curve into two independent curves at the parameter `u`,
    /// normalizing both knot vectors to the domain [0, 1].
    /// The parameter must lie in the domain interior (0, 1).
    ///
    /// | The curve.        | The two curves after the split at u = 1/2. |
    /// |:-----------------:|:------------------------------------------:|
    /// | ![][split-before] | ![][split-after]                           |
    #[cfg_attr(feature = "doc-images", doc = ::embed_doc_image::embed_image!("split-before", "doc-images/plots/manipulation/split-before.svg"))]
    #[cfg_attr(feature = "doc-images", doc = ::embed_doc_image::embed_image!("split-after", "doc-images/plots/manipulation/split-after.svg"))]
    pub fn split(&self, u: f64) -> Result<(Self, Self)> {
        split(self, u)
    }

    /// Inserts a knot at the parameter `u` without changing the curve shape.
    /// The parameter must lie in the domain interior (0, 1). Call it again to raise the multiplicity of `u`.
    ///
    /// | The curve.         | The curve after the insertion at u = 4/5. |
    /// |:------------------:|:-----------------------------------------:|
    /// | ![][insert-before] | ![][insert-after]                         |
    #[cfg_attr(feature = "doc-images", doc = ::embed_doc_image::embed_image!("insert-before", "doc-images/plots/manipulation/insert-before.svg"))]
    #[cfg_attr(feature = "doc-images", doc = ::embed_doc_image::embed_image!("insert-after", "doc-images/plots/manipulation/insert-after.svg"))]
    pub fn insert_knot(&mut self, u: f64) -> Result<&mut Self> {
        insert(self, u)?;
        Ok(self)
    }

    pub(crate) fn derive(&mut self) {
        self.knots.derive();
        self.control_points.derive(&self.knots);
    }

    /// Returns the curve describing the `k`-th derivative of this curve.
    /// The derivative order must not exceed the degree p.
    pub fn derivative_curve(&self, derivative: usize) -> Result<Self> {
        let knots = self.knots.derivative_knots(derivative)?;
        let control_points = ControlPoints::new(self.control_points.matrix_derivative(derivative).clone());
        Curve::new(knots, control_points)
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use nalgebra::{DMatrix, dmatrix, dvector};
    use rstest::fixture;

    use crate::points::DataPoints;

    use super::*;

    #[fixture]
    /// A two-dimensional, linear test curve with default degree two.
    fn curve(#[default(2)] degree: usize) -> Curve {
        let curve = Curve::with_uniform_knots(
            ControlPoints::new(dmatrix![
                1., 3., 5.;
                2., 4., 6.;
            ]),
            degree,
        )
        .unwrap();
        assert_eq!(curve.knots.vector(), &dvector![0., 0., 0., 1., 1., 1.]);
        curve
    }

    #[test]
    fn new_errors_for_a_control_point_count_that_does_not_match_the_knots() {
        let knots = Knots::uniform(2, 2).unwrap();
        let expected = knots.polygon_segments() + 1;
        let count = expected + 2;
        assert_eq!(
            Curve::new(knots, ControlPoints::new(DMatrix::zeros(1, count))).err(),
            Some(Error::ControlPointCountMismatch { expected, count })
        );
    }

    #[test]
    fn new_errors_for_a_control_point_that_is_not_finite() {
        let index = 1;
        let mut points = dmatrix![0.0, 1.0, 2.0;];
        points[(0, index)] = f64::INFINITY;
        assert_eq!(
            Curve::new(Knots::uniform(2, 2).unwrap(), ControlPoints::new(points)).err(),
            Some(Error::NonFiniteControlPoint { index })
        );
    }

    #[test]
    fn new_errors_for_unclamped_knots() {
        let knots = Knots::new(2, dvector![0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0]).unwrap();
        let points = DMatrix::zeros(1, knots.polygon_segments() + 1);
        assert_eq!(Curve::new(knots, ControlPoints::new(points)).err(), Some(Error::UnclampedKnots));
    }

    #[test]
    fn new_errors_for_more_than_p_plus_1_equal_end_knots() {
        let knots = Knots::new(2, dvector![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]).unwrap();
        let points = DMatrix::zeros(1, knots.polygon_segments() + 1);
        assert_eq!(Curve::new(knots, ControlPoints::new(points)).err(), Some(Error::UnclampedKnots));
    }

    #[test]
    fn new_errors_for_unnormalized_knots() {
        let knots = Knots::new(1, dvector![0.0, 0.0, 1.0, 2.0, 2.0]).unwrap();
        let points = DMatrix::zeros(1, knots.polygon_segments() + 1);
        assert_eq!(Curve::new(knots, ControlPoints::new(points)).err(), Some(Error::UnnormalizedKnots));
    }

    mod evaluate {
        use rstest::rstest;

        use super::*;

        #[test]
        fn evaluate_derivative_equals_the_sum_over_all_basis_functions() {
            let degree = 3;
            let mut curve =
                Curve::with_uniform_knots(ControlPoints::new(dmatrix![0., 1., 3., 2., 4., 5., 3., 6.;]), degree)
                    .unwrap();
            curve.insert_knot(0.4).unwrap();
            curve.insert_knot(0.4).unwrap();
            assert_eq!(curve.knots().multiplicity(0.4), degree, "the internal knot 0.4 repeats p times");

            for derivative in 0..=degree {
                let derivative_curve = curve.derivative_curve(derivative).unwrap();
                let knots = derivative_curve.knots();
                let points = derivative_curve.control_points().matrix();
                let parameters = knots.vector().iter().copied().chain((0..=20).map(|step| f64::from(step) / 20.0));

                for u in parameters {
                    let sum = (0..points.ncols())
                        .map(|i| knots.basis(i, u).unwrap() * points.column(i))
                        .fold(DVector::zeros(points.nrows()), |sum, term| sum + term);
                    assert_relative_eq!(curve.evaluate_derivative(u, derivative).unwrap(), sum, epsilon = 1e-9);
                }
            }
        }

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
            let mut curve = Curve::with_uniform_knots(ControlPoints::new(points), degree).unwrap();

            curve.insert_knot(0.5).unwrap();

            assert_eq!(curve.evaluate_derivative(0.9, 1).unwrap(), dvector![0.]);
        }

        #[test]
        fn evaluate_is_unchanged_by_repeated_insertion() {
            let degree = 3;
            let points = dmatrix![-1., -0.5, 0.5, 1.;];
            let mut curve = Curve::with_uniform_knots(ControlPoints::new(points), degree).unwrap();
            let u = 0.5;
            let expected_point = dvector![0.0];
            assert_eq!(curve.knots.vector(), &dvector![0., 0., 0., 0., 1., 1., 1., 1.]);
            assert_eq!(curve.evaluate(0.0).unwrap(), dvector![-1.]);
            assert_eq!(curve.evaluate(1.0).unwrap(), dvector![1.]);

            insert(&mut curve, u).unwrap();
            assert_eq!(curve.knots.vector(), &dvector![0., 0., 0., 0., u, 1., 1., 1., 1.]);
            assert_eq!(curve.control_points.matrix(), &dmatrix![-1., -0.75, 0.0, 0.75, 1.;]);
            assert_eq!(curve.evaluate(u).unwrap(), expected_point);
            assert_eq!(curve.evaluate(0.0).unwrap(), dvector![-1.]);
            assert_eq!(curve.evaluate(1.0).unwrap(), dvector![1.]);

            insert(&mut curve, u).unwrap();
            assert_eq!(curve.knots.vector(), &dvector![0., 0., 0., 0., u, u, 1., 1., 1., 1.]);
            assert_eq!(curve.control_points.matrix(), &dmatrix![-1., -0.75, -0.375, 0.375, 0.75, 1.;]);
            assert_eq!(curve.evaluate(u).unwrap(), expected_point);
            assert_eq!(curve.evaluate(0.0).unwrap(), dvector![-1.]);
            assert_eq!(curve.evaluate(1.0).unwrap(), dvector![1.]);

            insert(&mut curve, u).unwrap();
            assert_eq!(curve.knots.vector(), &dvector![0., 0., 0., 0., u, u, u, 1., 1., 1., 1.]);
            assert_eq!(curve.control_points.matrix(), &dmatrix![-1., -0.75, -0.375, 0.0, 0.375, 0.75, 1.;]);
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

    mod derivative_curve {
        use rstest::rstest;

        use super::*;

        #[rstest]
        fn derivative_curve_errors_above_the_degree(curve: Curve) {
            let degree = curve.degree();
            let derivative = degree + 1;
            assert_eq!(
                curve.derivative_curve(derivative).err(),
                Some(Error::DerivativeExceedsDegree { derivative, degree })
            );
        }

        #[test]
        fn derivative_curve_carries_the_higher_derivatives_of_the_curve() {
            let curve = Curve::with_uniform_knots(ControlPoints::new(dmatrix![-1., -0.5, 0.5, 2., 1.;]), 3).unwrap();
            let u = 0.25;
            assert_ne!(curve.evaluate_derivative(u, 2).unwrap(), dvector![0.], "the second derivative is not zero");

            let first_derivative = curve.derivative_curve(1).unwrap();

            assert_eq!(first_derivative.evaluate_derivative(u, 1), curve.evaluate_derivative(u, 2));
        }
    }

    #[test]
    fn reversed_curve_has_the_derivatives_of_the_chain_rule() {
        let degree = 3;
        let points =
            DMatrix::from_fn(2, 7, |row, column| if row == 0 { column as f64 } else { (column as f64 * 1.3).sin() });
        let curve = Curve::with_uniform_knots(ControlPoints::new(points), degree).unwrap();
        let mut reversed = curve.clone();
        reversed.reverse();

        // The parameters avoid the knots, where the derivative of order p jumps.
        for u in (0..20).map(|step| f64::from(step) / 20.0 + 0.013) {
            for derivative in 0..=degree {
                let sign = if derivative % 2 == 0 { 1.0 } else { -1.0 };
                let expected = sign * curve.evaluate_derivative(1.0 - u, derivative).unwrap();
                assert_relative_eq!(reversed.evaluate_derivative(u, derivative).unwrap(), expected, epsilon = 1e-9);
            }
        }
    }

    #[test]
    fn reverse() {
        let mut curve = Curve::with_uniform_knots(
            ControlPoints::new(dmatrix![
                1., 3., 5.;
                2., 4., 6.;
            ]),
            2,
        )
        .unwrap();

        let knots_before = curve.knots.vector().clone();
        let points_before = curve.control_points.matrix().clone();
        curve.reverse();

        let points_after = dmatrix![
                 5., 3., 1.;
                 6., 4., 2.;
        ];
        assert_eq!(curve.knots.vector(), &knots_before);
        assert_eq!(curve.control_points.matrix(), &points_after);

        curve.reverse();

        assert_eq!(curve.knots.vector(), &knots_before);
        assert_eq!(curve.control_points.matrix(), &points_before);
    }

    /// Two quadratic curves with a gap: the left curve ends at −0.5, the right curve starts at 0.5.
    fn curves_with_a_gap() -> (Curve, Curve) {
        let left = Curve::with_uniform_knots(ControlPoints::new(dmatrix![-2., -1., -0.5;]), 2).unwrap();
        let right = Curve::with_uniform_knots(ControlPoints::new(dmatrix![0.5, 1., 2.;]), 2).unwrap();
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

        curve.prepend_constrained(&left, Constraints { left: vec![], right: vec![0.0] }).unwrap();

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

        curve.append_constrained(&right, Constraints { left: vec![1.0], right: vec![] }).unwrap();

        assert_relative_eq!(curve.evaluate(joint).unwrap(), end, epsilon = f64::EPSILON.sqrt());
    }

    #[test]
    fn curve_of_degree_40_starts_and_ends_at_its_end_control_points() {
        let degree = 40;
        let points = DMatrix::from_fn(1, degree + 5, |_, column| column as f64);
        let last = points.ncols() - 1;
        let curve = Curve::with_uniform_knots(ControlPoints::new(points.clone()), degree).unwrap();

        assert_relative_eq!(curve.evaluate(0.0).unwrap()[0], points[0], epsilon = 1e-9);
        assert_relative_eq!(curve.evaluate(1.0).unwrap()[0], points[last], epsilon = 1e-9);
    }

    #[test]
    fn interpolate_with_errors_when_a_basis_function_is_zero_at_its_parameter() {
        let data = DataPoints::new(dmatrix![
            0.0, 0.01, 0.02, 0.03, 0.04, 10.0;
            0.0,  0.5, -0.5,  0.5, -0.5,  0.0;
        ]);
        let parameters = Parameters::generate(&data, ParameterMethod::ChordLength).unwrap();
        assert!(parameters.vector()[3] < 0.25, "the basis function 3 of the uniform knots starts at 0.25");

        let result = Curve::interpolate_with(&data, 2, ParameterMethod::ChordLength, KnotMethod::Uniform);
        assert_eq!(result.err(), Some(Error::SingularInterpolation { index: 3 }));

        let result = Curve::interpolate_with(&data, 2, ParameterMethod::ChordLength, KnotMethod::Averaging);
        assert!(result.is_ok(), "averaged knots suit the chord-length parameters");
    }

    #[test]
    fn interpolate_errors_for_no_data_points() {
        assert!(Curve::interpolate(&DataPoints::new(DMatrix::zeros(2, 0)), 2).is_err());
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
