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
    /// Returns a curve defined by the given knot vector and control points.
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
        match (knots.degree(), control_points.polygon_segments()) {
            (degree, polygon_segments) if polygon_segments < degree => {
                Err(Error::TooFewPolygonSegments { degree, polygon_segments })
            }
            _ => {
                let mut curve = Self { knots, control_points };
                curve.derive();
                Ok(curve)
            }
        }
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
        let parameters = Parameters::generate(data, parameter_method);
        let knots = Knots::generate(degree, data.polyline_segments(), &parameters, knot_method)?;
        let control_points = ControlPoints::new(interpolation::interpolate(&knots, data, &parameters));
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
            let polygon_segments = self.polygon_segments();
            let span = self.knots.find_span(u, derivative);

            for i in span - (degree - derivative)..=polygon_segments - derivative {
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
        let degree = self.degree();
        if derivative > degree {
            return Err(Error::DerivativeExceedsDegree { derivative, degree });
        }

        let knots = Knots::new(degree - derivative, self.knots.vector_derivative(derivative).clone());
        let control_points = ControlPoints::new(self.control_points.matrix_derivative(derivative).clone());
        Ok(Curve { knots, control_points })
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
