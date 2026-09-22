//! Implements the knot vector.

use std::ops::MulAssign;

use nalgebra::{DVector, DVectorView};

use crate::{
    basis,
    error::{Error, Result},
    parameters::Parameters,
    vector_views::VectorViews,
};

pub(crate) mod methods;

/// The knot vector U of a curve and the knot vectors of its derivatives.
///
/// The knot vector of the k-th derivative curve holds n + p + 2 − 2k knots in non-decreasing order:
///
/// ![The knot vector][eq-knots]
///
/// with the number of polygon segments n, the degree p, and the derivative order k. The first
/// p − k + 1 knots are 0 and the last p − k + 1 knots are 1, which leaves the n − p internal knots
/// between them. The knots from index p − k to n + 1 − k span the domain. [`KnotMethod`] lists
/// the methods that generate a knot vector.
#[cfg_attr(feature = "doc-images", doc = ::embed_doc_image::embed_image!("eq-knots", "doc-images/equations/knots.svg"))]
#[derive(Debug, Clone)]
pub struct Knots {
    /// The knot vectors of the curve and of its derivatives, indexed by derivative order.
    pub(crate) derivatives: Vec<DVector<f64>>,
    pub(crate) degree: usize,
}

/// The method generating a clamped knot vector for a curve of degree p with n polygon segments.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KnotMethod {
    /// Spaces the internal knots equally — eq. (9.7) in `Piegl1997`.
    /// Use only with evenly distributed control points.
    Uniform,
    /// Averages the parameters over knot spans — eqs. (9.68) and (9.69) in `Piegl1997`.
    /// Requires chord-length parameters; guarantees a well-conditioned fitting system.
    DeBoor,
    /// Averages runs of p consecutive parameters — eq. (9.8) in `Piegl1997`.
    /// The recommended companion to chord-length parameters for interpolation.
    Averaging,
}

impl Knots {
    /// Generates a clamped knot vector with the given method from the parameters ū.
    /// The parameters must belong to at least n + 1 data points.
    pub fn generate(
        degree: usize,
        polygon_segments: usize,
        parameters: &Parameters,
        method: KnotMethod,
    ) -> Result<Self> {
        let polyline_segments = parameters.polyline_segments();
        if polygon_segments > polyline_segments {
            return Err(Error::TooFewPolylineSegments { polygon_segments, polyline_segments });
        }

        match method {
            KnotMethod::Uniform => methods::uniform(degree, polygon_segments),
            KnotMethod::DeBoor => methods::de_boor(degree, polygon_segments, parameters),
            KnotMethod::Averaging => methods::averaging(degree, polygon_segments, parameters),
        }
    }

    /// Returns a clamped, uniform knot vector for a curve of the given degree
    /// with `n` polygon segments.
    pub fn uniform(degree: usize, polygon_segments: usize) -> Result<Self> {
        methods::uniform(degree, polygon_segments)
    }

    /// Returns knots for a curve of the given degree from the given knot values,
    /// deriving the knot vectors of all derivative orders. The degree p needs at least 2p + 2 finite
    /// knots in non-decreasing order, and the domain from knot p to knot n + 1 must not have length zero.
    pub fn new(degree: usize, knots: DVector<f64>) -> Result<Self> {
        let count = knots.len();
        if count < degree.saturating_mul(2).saturating_add(2) {
            return Err(Error::TooFewKnots { count, degree });
        }
        if let Some(index) = knots.iter().position(|u| !u.is_finite()) {
            return Err(Error::NonFiniteKnot { index });
        }
        if let Some(index) = (1..count).find(|&index| knots[index] < knots[index - 1]) {
            return Err(Error::DecreasingKnots { index });
        }
        if knots[degree] == knots[count - degree - 1] {
            return Err(Error::ZeroLengthDomain);
        }

        let mut derivatives: Vec<DVector<f64>> = Vec::with_capacity(degree + 1);
        derivatives.push(knots);

        let mut knots = Knots { derivatives, degree };
        knots.derive();
        Ok(knots)
    }

    /// Returns the knot vector of the curve.
    pub fn vector(&self) -> &DVector<f64> {
        &self.derivatives[0]
    }

    /// Returns the knot vector of the `k`-th derivative curve.
    pub(crate) fn vector_derivative(&self, derivative: usize) -> &DVector<f64> {
        &self.derivatives[derivative]
    }

    /// Returns the degree p of the curve the knots parametrize.
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// Returns the number of polygon segments n of a curve on this knot vector.
    pub fn polygon_segments(&self) -> usize {
        self.derivatives[0].len() - (self.degree + 2)
    }

    fn internal_count(&self) -> usize {
        self.polygon_segments() - self.degree
    }

    /// Returns a view of the internal knots.
    pub fn internal(&self) -> DVectorView<'_, f64> {
        self.derivatives[0].segment(self.degree + 1, self.internal_count())
    }

    fn domain_count(&self) -> usize {
        self.polygon_segments() - self.degree + 2
    }

    /// Returns a view of the domain knots, spanning from knot p to knot n + 1.
    pub fn domain(&self) -> DVectorView<'_, f64> {
        self.domain_derivative(0)
    }

    fn domain_derivative(&self, derivative: usize) -> DVectorView<'_, f64> {
        self.derivatives[derivative].segment(self.degree - derivative, self.domain_count())
    }

    /// Returns how often the knot value `u` occurs in the domain.
    pub fn multiplicity(&self, u: f64) -> usize {
        self.domain().iter().filter(|&x| *x == u).count()
    }

    pub(crate) fn reverse(&mut self) -> &mut Self {
        for knots in self.derivatives.iter_mut() {
            reverse(knots);
        }
        self
    }

    /// Normalizes all knot vectors to the domain [0, 1].
    pub fn normalize(&mut self) -> &mut Self {
        for knots in self.derivatives.iter_mut() {
            normalize(knots);
        }
        self
    }

    /// Derives the knot vectors of all derivative orders from the curve's knot vector.
    /// The `k`-th derivative knot vector drops the first and last knot of the previous order.
    pub(crate) fn derive(&mut self) {
        self.derivatives.truncate(1);
        for derivative in 1..=self.degree {
            let previous = &self.derivatives[derivative - 1];
            let trimmed = previous.segment(1, previous.len() - 2).clone_owned();

            self.derivatives.push(trimmed);
        }
    }

    /// Returns the index `i` of the last domain knot on the interval
    /// `[u_{p-k}^{(k)}, u_{n+1-k}^{(k)}]` that is less than or equal to `u`,
    /// stopping at the first knot of a repeated run (cf. algorithm A2.1 in `Piegl1997`).
    pub(crate) fn find_span(&self, u: f64, derivative: usize) -> usize {
        let knots = self.vector_derivative(derivative);
        let last = self.polygon_segments() + 1 - derivative;
        let mut span = self.degree() - derivative;

        while u >= knots[span + 1] && span + 1 < last {
            span += 1;
            if knots[span + 1] == knots[span] {
                break;
            }
        }
        span
    }

    /// Evaluates the `i`-th basis function of degree p at the parameter `u`
    /// by the Cox-de Boor-Mansfield recurrence:
    ///
    /// ![The Cox-de Boor-Mansfield recurrence relation][eq-basis-function]
    ///
    /// with the basis functions of degree 0
    ///
    /// ![Basis function of degree zero][eq-basis-function-zero]
    ///
    /// and the prefactors
    ///
    /// ![Prefactors][eq-basis-prefactor]
    ///
    /// with the knots U, the degree p, the number of polygon segments n, and the derivative order k,
    /// which is 0 for the curve itself. The condition ⋁ (i = n − k ⋀ u = Uₙ₊₁₋ₖ) closes the last
    /// interval, so the last basis function covers u = 1.
    ///
    /// Returns `None` for an index above n. Call it on [`Knots::derivative_knots`] to evaluate
    /// the basis functions of a derivative curve.
    #[cfg_attr(feature = "doc-images", doc = ::embed_doc_image::embed_image!("eq-basis-function", "doc-images/equations/basis-function.svg"))]
    #[cfg_attr(feature = "doc-images", doc = ::embed_doc_image::embed_image!("eq-basis-function-zero", "doc-images/equations/basis-function-zero.svg"))]
    #[cfg_attr(feature = "doc-images", doc = ::embed_doc_image::embed_image!("eq-basis-prefactor", "doc-images/equations/basis-prefactor.svg"))]
    pub fn basis(&self, index: usize, u: f64) -> Option<f64> {
        (index <= self.polygon_segments()).then(|| self.basis_of_derivative_curve(0, index, u))
    }

    /// Returns the knot vector of the `k`-th derivative curve: this knot vector without its first
    /// and last k knots, with the degree p − k. Its [basis functions][Knots::basis] are the basis
    /// functions of the derivative curve. The derivative order must not exceed the degree p.
    ///
    /// # Examples
    /// ```
    /// use bsplines::Knots;
    ///
    /// let degree = 3;
    /// let polygon_segments = 4;
    /// let knots = Knots::uniform(degree, polygon_segments).unwrap();
    ///
    /// let first_derivative = knots.derivative_knots(1).unwrap();
    /// assert_eq!(first_derivative.degree(), degree - 1);
    /// assert_eq!(first_derivative.polygon_segments(), polygon_segments - 1);
    ///
    /// // The basis functions sum to one at every parameter of the domain.
    /// let sum: f64 = (0..=first_derivative.polygon_segments())
    ///     .map(|index| first_derivative.basis(index, 0.25).unwrap())
    ///     .sum();
    /// assert_eq!(sum, 1.0);
    /// ```
    pub fn derivative_knots(&self, derivative: usize) -> Result<Self> {
        let degree = self.degree;
        if derivative > degree {
            return Err(Error::DerivativeExceedsDegree { derivative, degree });
        }

        Knots::new(degree - derivative, self.derivatives[derivative].clone())
    }

    /// Evaluates the `i`-th basis function of the `k`-th derivative curve at the parameter `u`:
    /// the basis function of degree p − k on the knot vector of that derivative.
    pub(crate) fn basis_of_derivative_curve(&self, derivative: usize, index: usize, u: f64) -> f64 {
        debug_assert!(index <= self.polygon_segments() - derivative, "the basis index must not exceed n − k");
        let knots = &self.derivatives[derivative];
        let basis_degree = self.degree - derivative;

        basis::basis(knots, index, basis_degree, derivative, self.polygon_segments(), u)
    }

    /// Returns whether exactly the first p + 1 knots are equal and exactly the last p + 1 knots are equal,
    /// so a curve starts and ends at its end control points.
    pub fn is_clamped(&self) -> bool {
        let knot_values = self.vector();
        let (degree, last) = (self.degree, knot_values.len() - 1);
        let is_head_clamped = knot_values[0] == knot_values[degree] && knot_values[degree] < knot_values[degree + 1];
        let is_tail_clamped = knot_values[last - degree] == knot_values[last] &&
            knot_values[last - degree - 1] < knot_values[last - degree];
        is_head_clamped && is_tail_clamped
    }

    /// Returns whether the knot values span exactly the domain [0, 1].
    pub fn is_normalized(&self) -> bool {
        let knot_values = self.vector();
        knot_values[0] == 0.0 && knot_values[knot_values.len() - 1] == 1.0
    }

    /// Returns whether the knot values equal the clamped, uniform knot vector
    /// of the same degree and number of polygon segments.
    pub fn is_uniform(&self) -> bool {
        methods::uniform(self.degree, self.polygon_segments()).is_ok_and(|uniform| uniform.vector() == self.vector())
    }
}

pub(crate) fn reverse(knots: &mut DVector<f64>) {
    let nrows = knots.nrows();
    let half_nrows = knots.len() / 2;

    for i in 0..half_nrows {
        knots.swap_rows(i, nrows - 1 - i);
    }

    knots.add_scalar_mut(-1.0);
    knots.mul_assign(-1.0);
}

pub(crate) fn reversed(knots: &DVector<f64>) -> DVector<f64> {
    let mut copy = knots.clone();
    reverse(&mut copy);
    copy
}

/// Normalizes the knot values to the domain [0, 1] in place.
pub(crate) fn normalize(knots: &mut DVector<f64>) {
    let old_lim = (knots.min(), knots.max());

    rescale(knots, old_lim, (0.0, 1.0))
}

fn rescale(knots: &mut DVector<f64>, old_lim: (f64, f64), new_lim: (f64, f64)) {
    let len = knots.len();
    *knots -= DVector::repeat(len, old_lim.0);
    *knots /= old_lim.1 - old_lim.0;
    *knots *= new_lim.1 - new_lim.0;
    *knots += DVector::repeat(len, new_lim.0);
}

#[cfg(test)]
mod tests {
    use nalgebra::dvector;
    use rstest::rstest;

    use super::*;

    const SEGMENTS: usize = 4;

    fn knots_example(degree: usize) -> Knots {
        methods::uniform(degree, SEGMENTS).unwrap()
    }

    #[rstest(degree, case(1), case(2), case(3))]
    fn segments(degree: usize) {
        assert_eq!(knots_example(degree).polygon_segments(), SEGMENTS);
    }

    #[rstest(degree, expected, case(1, 3), case(2, 2), case(3, 1))]
    fn internal_count(degree: usize, expected: usize) {
        assert_eq!(knots_example(degree).internal_count(), expected);
    }

    #[test]
    fn internal_degree_1() {
        assert_eq!(knots_example(1).internal(), dvector![0.25, 0.5, 0.75]);
    }

    #[test]
    fn internal_degree_2() {
        assert_eq!(knots_example(2).internal(), dvector![1.0 / 3.0, 2.0 / 3.0]);
    }

    #[test]
    fn internal_degree_3() {
        assert_eq!(knots_example(3).internal(), dvector![0.5]);
    }

    #[rstest(degree, expected, case(1, 5), case(2, 4), case(3, 3))]
    fn domain_count(degree: usize, expected: usize) {
        assert_eq!(knots_example(degree).domain_count(), expected);
    }

    #[test]
    fn domain_degree_1() {
        assert_eq!(knots_example(1).domain(), dvector![0.0, 0.25, 0.5, 0.75, 1.0]);
    }

    #[test]
    fn domain_degree_2() {
        assert_eq!(knots_example(2).domain(), dvector![0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]);
    }

    #[test]
    fn domain_degree_2_derivative() {
        assert_eq!(knots_example(2).domain_derivative(1), dvector![0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]);
    }

    #[test]
    fn domain_degree_3() {
        assert_eq!(knots_example(3).domain(), dvector![0.0, 0.5, 1.0]);
    }

    #[test]
    fn new_errors_for_too_few_knots() {
        let degree = 2;
        let smallest = dvector![0., 0., 0., 1., 1., 1.];
        assert_eq!(smallest.len(), 2 * degree + 2, "p + 1 control points need 2p + 2 knots");
        assert!(Knots::new(degree, smallest).is_ok());

        let too_short = dvector![0., 0., 0., 1., 1.];
        let count = too_short.len();
        assert_eq!(Knots::new(degree, too_short).err(), Some(Error::TooFewKnots { count, degree }));
    }

    #[test]
    fn multiplicity() {
        let knots = Knots::new(2, dvector![0., 0., 0., 0.25, 0.5, 0.5, 0.75, 1., 1., 1.]).unwrap();

        assert_eq!(knots.multiplicity(0.2), 0);
        assert_eq!(knots.multiplicity(0.25), 1);
        assert_eq!(knots.multiplicity(0.5), 2);
        assert_eq!(knots.multiplicity(0.), 1);
        assert_eq!(knots.multiplicity(1.), 1);
    }

    #[test]
    fn knot_derivation() {
        let degree = 3;
        let knots = methods::uniform(degree, SEGMENTS).unwrap();

        let u0 = dvector![0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0];
        let u1 = dvector![0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0];
        let u2 = dvector![0.0, 0.0, 0.5, 1.0, 1.0];
        let u3 = dvector![0.0, 0.5, 1.0];

        assert_eq!(knots.vector_derivative(0), &u0);
        assert_eq!(knots.vector_derivative(1), &u1);
        assert_eq!(knots.vector_derivative(2), &u2);
        assert_eq!(knots.vector_derivative(3), &u3);
    }

    #[test]
    fn derivative_knots_carry_the_basis_functions_of_the_derivative_curve() {
        let knots = Knots::new(3, dvector![0., 0., 0., 0., 0.25, 0.5, 0.5, 1., 1., 1., 1.]).unwrap();

        for derivative in 0..=knots.degree() {
            let derivative_knots = knots.derivative_knots(derivative).unwrap();
            assert_eq!(derivative_knots.vector(), knots.vector_derivative(derivative));

            for index in 0..=derivative_knots.polygon_segments() {
                for u in (0..=8).map(|eighth| f64::from(eighth) / 8.0) {
                    assert_eq!(
                        derivative_knots.basis(index, u),
                        Some(knots.basis_of_derivative_curve(derivative, index, u))
                    );
                }
            }
        }
    }

    #[test]
    fn derivative_knots_errors_above_the_degree() {
        let knots = knots_example(2);
        let degree = knots.degree();
        let derivative = degree + 1;
        assert_eq!(
            knots.derivative_knots(derivative).err(),
            Some(Error::DerivativeExceedsDegree { derivative, degree })
        );
    }

    #[test]
    fn basis_returns_none_beyond_the_last_index() {
        let knots = knots_example(2);
        let last = knots.polygon_segments();

        assert_eq!(knots.basis(last, 1.0), Some(1.0), "the last basis function is 1 at the end of clamped knots");
        assert_eq!(knots.basis(last + 1, 1.0), None);
    }

    #[test]
    fn normalize() {
        let mut knots = Knots::new(1, dvector![1.0, 1.0, 1.5, 2.0, 2.0]).unwrap();
        knots.normalize();
        assert_eq!(knots.vector(), &dvector![0.0, 0.0, 0.5, 1.0, 1.0]);
    }

    #[test]
    fn reverse() {
        let mut knots = Knots::new(1, dvector![0.0, 0.0, 0.6, 1.0, 1.0]).unwrap();
        knots.reverse();
        assert_eq!(knots.vector(), &dvector![0.0, 0.0, 0.4, 1.0, 1.0]);
    }

    #[test]
    fn generate_errors_for_fewer_data_points_than_control_points() {
        let parameters = Parameters::new(dvector![0.0, 0.5, 1.0]).unwrap();
        let polyline_segments = parameters.polyline_segments();
        let polygon_segments = polyline_segments + 1;
        assert_eq!(
            Knots::generate(2, polygon_segments, &parameters, KnotMethod::Averaging).err(),
            Some(Error::TooFewPolylineSegments { polygon_segments, polyline_segments })
        );
    }

    #[test]
    fn new_errors_for_a_knot_that_is_not_finite() {
        let index = 2;
        let mut knot_values = dvector![0.0, 0.0, 0.5, 1.0, 1.0];
        knot_values[index] = f64::NAN;
        assert_eq!(Knots::new(1, knot_values).err(), Some(Error::NonFiniteKnot { index }));
    }

    #[test]
    fn new_errors_for_a_decreasing_knot() {
        assert_eq!(
            Knots::new(1, dvector![0.0, 0.0, 0.6, 0.4, 1.0, 1.0]).err(),
            Some(Error::DecreasingKnots { index: 3 })
        );
    }

    #[test]
    fn new_errors_for_a_domain_of_length_zero() {
        let degree = 1;
        let knot_values = dvector![0.0, 0.5, 0.5, 1.0];
        let last_domain_knot = knot_values.len() - degree - 1;
        assert_eq!(knot_values[degree], knot_values[last_domain_knot], "the domain starts and ends at 0.5");
        assert_eq!(Knots::new(degree, knot_values).err(), Some(Error::ZeroLengthDomain));
    }

    #[test]
    fn is_clamped_test() {
        assert!(Knots::new(1, dvector![0.0, 0.0, 0.5, 1.0, 1.0]).unwrap().is_clamped());
        assert!(Knots::new(1, dvector![2.0, 2.0, 2.5, 3.0, 3.0]).unwrap().is_clamped());
        assert!(!Knots::new(1, dvector![0.0, 0.25, 0.5, 1.0, 1.0]).unwrap().is_clamped());
        assert!(
            !Knots::new(1, dvector![0.0, 0.0, 0.0, 0.5, 1.0, 1.0]).unwrap().is_clamped(),
            "p + 2 equal first knots"
        );
        assert!(!Knots::new(1, dvector![0.0, 0.0, 0.5, 1.0, 1.0, 1.0]).unwrap().is_clamped(), "p + 2 equal last knots");
    }

    #[test]
    fn is_normalized_test() {
        assert!(Knots::new(1, dvector![0.0, 0.0, 0.5, 1.0, 1.0]).unwrap().is_normalized());
        assert!(!Knots::new(1, dvector![0.0, 0.0, 1.0, 1.5, 1.5]).unwrap().is_normalized());
    }

    #[test]
    fn is_uniform_test() {
        assert!(Knots::new(1, dvector![0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0]).unwrap().is_uniform());
        assert!(!Knots::new(1, dvector![0.0, 0.0, 0.25, 0.75, 1.0, 1.0]).unwrap().is_uniform());
    }

    #[rstest(u, expected, case(0.24, 1), case(0.25, 2), case(0.26, 2), case(0.74, 3), case(0.75, 4), case(0.76, 4))]
    fn find_span_test(u: f64, expected: usize) {
        assert_eq!(knots_example(1).find_span(u, 0), expected);
    }
}
