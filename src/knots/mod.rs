#![cfg_attr(feature = "doc-images",
cfg_attr(all(),
doc = ::embed_doc_image::embed_image!("eq-knots", "doc-images/equations/knots.svg")))]
//! Implements the knot vector defining the [spline basis functions][basis].
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

use crate::{
    basis,
    error::Result,
    parameters::Parameters,
    types::{VecD, VecDView, VecHelpers},
};

pub(crate) mod methods;

/// The knot vector U of a curve and the knot vectors of its derivatives.
#[derive(Debug, Clone)]
pub struct Knots {
    /// The knot vectors of the curve and of its derivatives, indexed by derivative order.
    pub(crate) derivatives: Vec<VecD>,
    pub(crate) degree: usize,
    pub(crate) max_derivative: usize,
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

/// Generates a clamped knot vector with the given method from the parameters ū.
pub fn generate(degree: usize, polygon_segments: usize, params: &Parameters, method: KnotMethod) -> Result<Knots> {
    match method {
        KnotMethod::Uniform => methods::uniform(degree, polygon_segments),
        KnotMethod::DeBoor => methods::de_boor(degree, polygon_segments, params),
        KnotMethod::Averaging => methods::averaging(degree, polygon_segments, params),
    }
}

impl Knots {
    /// Returns a clamped, uniform knot vector for a curve of the given degree
    /// with `n` polygon segments.
    pub fn uniform(degree: usize, polygon_segments: usize) -> Result<Self> {
        methods::uniform(degree, polygon_segments)
    }

    /// Returns knots for a curve of the given degree from the given knot values,
    /// deriving the knot vectors of all derivative orders.
    pub fn new(degree: usize, knots: VecD) -> Self {
        let mut derivatives: Vec<VecD> = Vec::with_capacity(degree + 1);
        derivatives.push(knots);

        let mut knots = Knots { derivatives, degree, max_derivative: 0 };
        knots.derive();
        knots
    }

    /// Returns the knot vector of the curve.
    pub fn vector(&self) -> &VecD {
        &self.derivatives[0]
    }

    /// Returns the mutable knot vector of the curve.
    /// Call [`Knots::derive`] afterwards to refresh the derivative knot vectors.
    pub fn vector_mut(&mut self) -> &mut VecD {
        &mut self.derivatives[0]
    }

    /// Returns the knot vector of the `k`-th derivative curve.
    pub fn vector_derivative(&self, derivative: usize) -> &VecD {
        &self.derivatives[derivative]
    }

    /// Returns the mutable knot vector of the `k`-th derivative curve.
    pub fn vector_derivative_mut(&mut self, derivative: usize) -> &mut VecD {
        &mut self.derivatives[derivative]
    }

    /// Returns the degree p of the curve the knots parametrize.
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// Returns the number of polygon segments n of a curve on this knot vector.
    pub fn polygon_segments(&self) -> usize {
        self.derivatives[0].len() - (self.degree + 2)
    }

    /// Returns the number of knots of the `k`-th derivative knot vector.
    pub fn len(&self, k: usize) -> usize {
        self.derivatives[k].len()
    }

    /// Returns the number of internal knots, i.e. those between the clamps.
    pub fn internal_count(&self) -> usize {
        self.polygon_segments() - self.degree
    }

    /// Returns a view of the internal knots.
    pub fn internal(&self) -> VecDView<'_> {
        self.derivatives[0].segment(self.degree + 1, self.internal_count())
    }

    /// Returns the `i`-th internal knot.
    pub fn internal_knot(&self, i: usize) -> f64 {
        self.internal()[i]
    }

    /// Returns the number of knots in the domain.
    pub fn domain_count(&self) -> usize {
        self.polygon_segments() - self.degree + 2
    }

    /// Returns a view of the domain knots, spanning from knot p to knot n + 1.
    pub fn domain(&self) -> VecDView<'_> {
        self.domain_derivative(0)
    }

    /// Returns a view of the domain knots of the `k`-th derivative knot vector.
    pub fn domain_derivative(&self, k: usize) -> VecDView<'_> {
        self.derivatives[k].segment(self.degree - k, self.domain_count())
    }

    /// Returns the `i`-th domain knot.
    pub fn domain_knot(&self, i: usize) -> f64 {
        self.domain()[i]
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

    /// Rescales all knot vectors from the limits `old_lim` to `new_lim`.
    pub fn rescale(&mut self, old_lim: (f64, f64), new_lim: (f64, f64)) {
        for knots in self.derivatives.iter_mut() {
            rescale(knots, old_lim, new_lim);
        }
    }

    /// Returns the highest derivative order for which a knot vector is available.
    pub fn max_derivative(&self) -> usize {
        self.max_derivative
    }

    /// Derives the knot vectors of all derivative orders from the curve's knot vector.
    /// The `k`-th derivative knot vector drops the first and last knot of the previous order.
    pub fn derive(&mut self) {
        let p = self.degree;

        self.derivatives.truncate(1);
        for k in 1..=p {
            // obtain the `k`-th derivative knot vector from the previous `k-1`-th derivative knot vector segment
            // by dropping the first and last segment
            let segment_of_previous_order_knot_vector =
                self.derivatives[k - 1].segment(1, self.len(k - 1) - 2).clone_owned();

            self.derivatives.push(segment_of_previous_order_knot_vector);
        }
        self.max_derivative = p;
    }

    /// Returns the index `i` of the last domain knot on the interval
    /// `[u_{p-k}^{(k)}, u_{n+1-k}^{(k)}]` that is less than or equal to `u`,
    /// stopping at the first knot of a repeated run (cf. algorithm A2.1 in `Piegl1997`).
    pub(crate) fn find_span(&self, u: f64, k: usize) -> usize {
        let knots = self.vector_derivative(k);
        let pk = self.degree() - k;
        let lim = self.polygon_segments() + 1 - k;
        let mut i = pk;

        while u >= knots[i + 1] && i + 1 < lim {
            i += 1;
            if knots[i + 1] == knots[i] {
                break;
            }
        }
        i
    }

    /// Evaluates the `i`-th basis function of the `k`-th derivative knot vector at the parameter `u`,
    /// where `p` is the degree of the curve itself, so the basis degree is p − k.
    pub fn evaluate(&self, k: usize, i: usize, p: usize, u: f64) -> f64 {
        let knots = &self.derivatives[k];
        let n = self.polygon_segments();
        let pk = p - k;

        basis::basis(knots, i, pk, k, n, u)
    }
}

/// Returns whether the first and last knot value are each repeated p + 1 times,
/// so a curve starts and ends at its end control points.
pub fn is_clamped(knots: &Knots) -> bool {
    let u0 = knots.vector();
    let clamp_size = knots.degree + 1;

    let is_head_clamped = u0.iter().take(clamp_size).all(|&u| u == 0.0);
    let is_tail_clamped = u0.iter().rev().take(clamp_size).all(|&u| u == 1.0);

    is_head_clamped && is_tail_clamped
}

/// Returns whether the knot values span exactly the domain [0, 1].
pub fn is_normalized(knots: &Knots) -> bool {
    let u0 = knots.vector();

    let is_min_zero = u0.iter().min_by(|a, b| a.partial_cmp(b).unwrap()) == Some(&0.0);
    let is_max_unity = u0.iter().max_by(|a, b| a.partial_cmp(b).unwrap()) == Some(&1.0);

    is_min_zero && is_max_unity
}

/// Returns whether the knot values are in non-decreasing order.
pub fn is_sorted(knots: &Knots) -> bool {
    let mut it = knots.derivatives[0].iter();
    match it.next() {
        None => true,
        Some(first) => it
            .scan(first, |state, next| {
                let cmp = *state <= next;
                *state = next;
                Some(cmp)
            })
            .all(|b| b),
    }
}

/// Returns whether the knot values equal a clamped, uniform knot vector.
pub fn is_uniform(knots: &Knots) -> Result<bool> {
    let u0 = knots.vector();

    let expected = methods::uniform(knots.degree(), knots.polygon_segments())?;

    Ok(u0.eq(expected.vector()))
}

pub(crate) fn reverse(knots: &mut VecD) {
    let nrows = knots.nrows();
    let half_nrows = knots.len() / 2;

    for i in 0..half_nrows {
        knots.swap_rows(i, nrows - 1 - i);
    }

    knots.add_scalar_mut(-1.0);
    knots.mul_assign(-1.0);
}

pub(crate) fn reversed(knots: &VecD) -> VecD {
    let mut copy = knots.clone();
    reverse(&mut copy);
    copy
}

/// Normalizes the knot values to the domain [0, 1] in place.
pub fn normalize(knots: &mut VecD) {
    let old_lim = (knots.min(), knots.max());

    rescale(knots, old_lim, (0.0, 1.0))
}

/// Returns a copy of the knot values normalized to the domain [0, 1].
pub fn normalized(knots: &mut VecD) -> VecD {
    let mut copy = knots.clone();
    normalize(&mut copy);
    copy
}

fn rescale(knots: &mut VecD, old_lim: (f64, f64), new_lim: (f64, f64)) {
    let n = knots.len();
    *knots -= VecD::repeat(n, old_lim.0);
    *knots /= old_lim.1 - old_lim.0;
    *knots *= new_lim.1 - new_lim.0;
    *knots += VecD::repeat(n, new_lim.0);
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
        assert!(is_sorted(&Knots::new(1, dvector![0.0, 0.0, 0.5, 1.0, 1.0])));
        assert!(!is_sorted(&Knots::new(1, dvector![0.0, 1.0, 0.5, 1.0, 1.0])));
    }

    #[test]
    fn is_clamped_test() {
        assert!(is_clamped(&Knots::new(1, dvector![0.0, 0.0, 0.5, 1.0, 1.0])));
        assert!(!is_clamped(&Knots::new(1, dvector![0.0, 1.0, 0.5, 1.0, 1.0])));
    }

    #[test]
    fn is_normed_test() {
        assert!(is_normalized(&Knots::new(1, dvector![0.0, 0.0, 0.5, 1.0, 1.0])));
        assert!(!is_normalized(&Knots::new(1, dvector![0.0, 0.0, 1.5, 1.0, 1.0])));
    }

    #[test]
    fn is_uniform_test() {
        assert!(is_uniform(&Knots::new(1, dvector![0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0])).unwrap());
        assert!(!is_uniform(&Knots::new(1, dvector![0.0, 0.0, 0.25, 0.75, 1.0, 1.0])).unwrap());
    }

    #[rstest(u, expected, case(0.24, 1), case(0.25, 2), case(0.26, 2), case(0.74, 3), case(0.75, 4), case(0.76, 4))]
    fn test_find_span(u: f64, expected: usize) {
        assert_eq!(knots_example(1).find_span(u, 0), expected);
    }
}
