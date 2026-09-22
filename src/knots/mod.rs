#![cfg_attr(feature = "doc-images",
cfg_attr(all(),
doc = ::embed_doc_image::embed_image!("eq-knots", "doc-images/equations/knots.svg")))]
//! Implements the knot vector defining the [basis functions][Knots::basis].
//!
//! The knot vector parametrizing the `k`-th degree curve is composed of `n+p+2 - 2k` scalar values
//! in ascending order, called 'knots'.
//!
//! ![The knot vector][eq-knots]
//!
//! The head and tail contains of `p-k+1` knots of value `0` and `1`, respectively.
//! This leaves `n-p` internal knots in the center.
//! The interval from index `i = p-k,..., n+1-k` is called 'domain'.
//!
//! Different knot vector generation methods are available via [KnotMethod].

use std::ops::MulAssign;

use nalgebra::{DVector, DVectorView};

use crate::{basis, error::Result, parameters::Parameters, vector_views::VectorViews};

pub(crate) mod methods;

/// The knot vector U of a curve and the knot vectors of its derivatives.
#[derive(Debug, Clone)]
pub struct Knots {
    /// The knot vectors of the curve and of its derivatives, indexed by derivative order.
    pub(crate) derivatives: Vec<DVector<f64>>,
    pub(crate) degree: usize,
}

/// The method generating a clamped knot vector for a curve of degree p with n polygon segments.
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
    pub fn generate(
        degree: usize,
        polygon_segments: usize,
        parameters: &Parameters,
        method: KnotMethod,
    ) -> Result<Self> {
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
    /// deriving the knot vectors of all derivative orders.
    pub fn new(degree: usize, knots: DVector<f64>) -> Self {
        let mut derivatives: Vec<DVector<f64>> = Vec::with_capacity(degree + 1);
        derivatives.push(knots);

        let mut knots = Knots { derivatives, degree };
        knots.derive();
        knots
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
    #[cfg_attr(feature = "doc-images", doc = ::embed_doc_image::embed_image!("eq-basis-function", "doc-images/equations/basis-function.svg"))]
    #[cfg_attr(feature = "doc-images", doc = ::embed_doc_image::embed_image!("eq-basis-function-zero", "doc-images/equations/basis-function-zero.svg"))]
    #[cfg_attr(feature = "doc-images", doc = ::embed_doc_image::embed_image!("eq-basis-prefactor", "doc-images/equations/basis-prefactor.svg"))]
    pub fn basis(&self, index: usize, u: f64) -> f64 {
        self.basis_of_derivative_curve(0, index, u)
    }

    /// Evaluates the `i`-th basis function of the `k`-th derivative curve at the parameter `u`:
    /// the basis function of degree p − k on the knot vector of that derivative.
    pub(crate) fn basis_of_derivative_curve(&self, derivative: usize, index: usize, u: f64) -> f64 {
        let knots = &self.derivatives[derivative];
        let basis_degree = self.degree - derivative;

        basis::basis(knots, index, basis_degree, derivative, self.polygon_segments(), u)
    }

    /// Returns whether the first and last knot value are each repeated p + 1 times,
    /// so a curve starts and ends at its end control points.
    pub fn is_clamped(&self) -> bool {
        let knot_values = self.vector();
        let clamp_size = self.degree + 1;

        let is_head_clamped = knot_values.iter().take(clamp_size).all(|&u| u == 0.0);
        let is_tail_clamped = knot_values.iter().rev().take(clamp_size).all(|&u| u == 1.0);

        is_head_clamped && is_tail_clamped
    }

    /// Returns whether the knot values span exactly the domain [0, 1].
    pub fn is_normalized(&self) -> bool {
        let knot_values = self.vector();

        let is_min_zero = knot_values.iter().min_by(|a, b| a.partial_cmp(b).unwrap()) == Some(&0.0);
        let is_max_unity = knot_values.iter().max_by(|a, b| a.partial_cmp(b).unwrap()) == Some(&1.0);

        is_min_zero && is_max_unity
    }

    /// Returns whether the knot values are in non-decreasing order.
    pub fn is_sorted(&self) -> bool {
        let mut values = self.derivatives[0].iter();
        match values.next() {
            None => true,
            Some(first) => values
                .scan(first, |state, next| {
                    let cmp = *state <= next;
                    *state = next;
                    Some(cmp)
                })
                .all(|b| b),
        }
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
    fn multiplicity() {
        let knots = Knots::new(2, dvector![0., 0., 0., 0.25, 0.5, 0.5, 0.75, 1., 1., 1.]);

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
    fn normalize() {
        let mut knots = Knots::new(1, dvector![1.0, 1.0, 1.5, 2.0, 2.0]);
        knots.normalize();
        assert_eq!(knots.vector(), &dvector![0.0, 0.0, 0.5, 1.0, 1.0]);
    }

    #[test]
    fn reverse() {
        let mut knots = Knots::new(1, dvector![0.0, 0.0, 0.6, 1.0, 1.0]);
        knots.reverse();
        assert_eq!(knots.vector(), &dvector![0.0, 0.0, 0.4, 1.0, 1.0]);
    }

    #[test]
    fn is_sorted_test() {
        assert!(Knots::new(1, dvector![0.0, 0.0, 0.5, 1.0, 1.0]).is_sorted());
        assert!(!Knots::new(1, dvector![0.0, 1.0, 0.5, 1.0, 1.0]).is_sorted());
    }

    #[test]
    fn is_clamped_test() {
        assert!(Knots::new(1, dvector![0.0, 0.0, 0.5, 1.0, 1.0]).is_clamped());
        assert!(!Knots::new(1, dvector![0.0, 1.0, 0.5, 1.0, 1.0]).is_clamped());
    }

    #[test]
    fn is_normalized_test() {
        assert!(Knots::new(1, dvector![0.0, 0.0, 0.5, 1.0, 1.0]).is_normalized());
        assert!(!Knots::new(1, dvector![0.0, 0.0, 1.5, 1.0, 1.0]).is_normalized());
    }

    #[test]
    fn is_uniform_test() {
        assert!(Knots::new(1, dvector![0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0]).is_uniform());
        assert!(!Knots::new(1, dvector![0.0, 0.0, 0.25, 0.75, 1.0, 1.0]).is_uniform());
    }

    #[rstest(u, expected, case(0.24, 1), case(0.25, 2), case(0.26, 2), case(0.74, 3), case(0.75, 4), case(0.76, 4))]
    fn find_span_test(u: f64, expected: usize) {
        assert_eq!(knots_example(1).find_span(u, 0), expected);
    }
}
