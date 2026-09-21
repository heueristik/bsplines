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
//! Different [knot vector generation methods][methods] are available.

use std::ops::MulAssign;

use thiserror::Error;

use crate::{
    curve::{CurveError, basis, parameters, parameters::Parameters},
    types::{VecD, VecDView, VecHelpers},
};

pub mod methods;

#[derive(Debug, Clone)]
pub struct Knots {
    /// The knot vectors of the curve and of its derivatives, indexed by derivative order.
    pub(crate) derivatives: Vec<VecD>,
    pub(crate) degree: usize,
    pub(crate) max_derivative: usize,
}

pub enum DomainKnotComparatorType {
    Left,
    LeftOrEqual,
    RightOrEqual,
    Right,
}

pub enum Generation {
    Uniform,
    Manual { knots: Knots },
    Method { parameter_method: parameters::Method, knot_method: Method },
}

pub enum Method {
    Uniform,
    DeBoor,
    Averaging,
}

#[derive(Error, Debug, PartialEq)]
pub enum KnotError {
    #[error("Parameter `u = {u}` lies outside the interval `[{lower_bound}, {upper_bound}]`.")]
    ParameterOutOfBounds { u: f64, lower_bound: f64, upper_bound: f64 },
    // TODO needed?
    //#[error("Knot `u = {u}` has a muliplicity of {multiplicity}.")]
    //InvalidMultiplicity { u: f64, multiplicity: usize },
}

pub fn generate(degree: usize, segments: usize, params: &Parameters, method: Method) -> Result<Knots, CurveError> {
    match method {
        Method::Uniform => methods::uniform(degree, segments),
        Method::DeBoor => methods::de_boor(degree, segments, params),
        Method::Averaging => methods::averaging(degree, segments, params),
    }
}

impl Knots {
    // generates params automatically, if none are provided.
    pub fn new(degree: usize, knots: VecD) -> Self {
        let mut derivatives: Vec<VecD> = Vec::with_capacity(degree + 1);
        derivatives.push(knots);

        // TODO Add multiplicity check

        let mut knots = Knots { derivatives, degree, max_derivative: 0 };
        knots.derive();
        knots
    }

    pub fn vector(&self) -> &VecD {
        &self.derivatives[0]
    }

    pub fn vector_mut(&mut self) -> &mut VecD {
        &mut self.derivatives[0]
    }

    /// # Arguments
    /// * `k` - The `k`-th derivative knot vector.
    pub fn vector_derivative(&self, derivative: usize) -> &VecD {
        &self.derivatives[derivative]
    }

    pub fn vector_derivative_mut(&mut self, derivative: usize) -> &mut VecD {
        &mut self.derivatives[derivative]
    }

    pub fn degree(&self) -> usize {
        self.degree
    }

    pub fn segments(&self) -> usize {
        self.derivatives[0].len() - (self.degree + 2)
    }

    pub fn len(&self, k: usize) -> usize {
        self.derivatives[k].len()
    }

    pub fn internal_count(&self) -> usize {
        self.segments() - self.degree
    }

    pub fn internal(&self) -> VecDView<'_> {
        self.derivatives[0].segment(self.degree + 1, self.internal_count())
    }

    pub fn internal_knot(&self, i: usize) -> f64 {
        self.internal()[i]
    }

    pub fn domain_count(&self) -> usize {
        self.segments() - self.degree + 2
    }

    pub fn domain(&self) -> VecDView<'_> {
        self.domain_derivative(0)
    }

    pub fn domain_derivative(&self, k: usize) -> VecDView<'_> {
        self.derivatives[k].segment(self.degree - k, self.domain_count())
    }

    pub fn domain_knot(&self, i: usize) -> f64 {
        self.domain()[i]
    }

    pub fn multiplicity(&self, u: f64) -> usize {
        self.domain().iter().filter(|&x| *x == u).count()
    }

    pub(crate) fn reverse(&mut self) -> &mut Self {
        for knots in self.derivatives.iter_mut() {
            reverse(knots);
        }
        self
    }

    pub fn normalize(&mut self) -> &mut Self {
        for knots in self.derivatives.iter_mut() {
            normalize(knots);
        }
        self
    }

    pub fn rescale(&mut self, old_lim: (f64, f64), new_lim: (f64, f64)) {
        for knots in self.derivatives.iter_mut() {
            rescale(knots, old_lim, new_lim);
        }
    }

    pub fn max_derivative(&self) -> usize {
        self.max_derivative
    }

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

    /// Returns the index `i` of the knot on the domain interval
    /// `[u_{p-k}^{(k)}, u_{n+1-k}^{(k)}]`,
    /// being lower, equal, or higher than `u`.
    pub fn find_idx(&self, u: f64, k: usize, comparator: DomainKnotComparatorType) -> usize {
        let knots = self.vector_derivative(k);
        let pk = self.degree() - k;
        match comparator {
            DomainKnotComparatorType::Left => {
                let lim = self.segments() + 1 - k;
                let mut i = pk;

                while u > knots[i + 1] && i + 1 < lim {
                    i += 1;
                }
                i
            }
            DomainKnotComparatorType::LeftOrEqual => {
                let lim = self.segments() + 1 - k;
                let mut i = pk;

                while u >= knots[i + 1] && i + 1 < lim {
                    i += 1;
                    if knots[i + 1] == knots[i] {
                        break;
                    }
                }
                i
            }
            DomainKnotComparatorType::RightOrEqual => {
                let mut i = knots.len() - 1 - pk;

                while u <= knots[i - 1] && i > pk {
                    i -= 1;
                    if knots[i - 1] == knots[i] {
                        break;
                    }
                }
                i
            }
            DomainKnotComparatorType::Right => {
                let mut i = knots.len() - 1 - pk;

                while u < knots[i - 1] && i - 1 > pk {
                    i -= 1;
                }
                i
            }
        }
    }

    /// `p` the degree of this basis function of the kth degree spline - not of the 0th degree spline
    pub fn evaluate(&self, k: usize, i: usize, p: usize, u: f64) -> f64 {
        let knots = &self.derivatives[k];
        let n = self.segments();
        let pk = p - k;

        basis::basis(knots, i, pk, k, n, u)
    }
}

pub fn is_clamped(knots: &Knots) -> bool {
    let u0 = knots.vector();
    let clamp_size = knots.degree + 1;

    let is_head_clamped = u0.iter().take(clamp_size).all(|&u| u == 0.0);
    let is_tail_clamped = u0.iter().rev().take(clamp_size).all(|&u| u == 1.0);

    is_head_clamped && is_tail_clamped
}

pub fn is_normalized(knots: &Knots) -> bool {
    let u0 = knots.vector();

    let is_min_zero = u0.iter().min_by(|a, b| a.partial_cmp(b).unwrap()) == Some(&0.0);
    let is_max_unity = u0.iter().max_by(|a, b| a.partial_cmp(b).unwrap()) == Some(&1.0);

    is_min_zero && is_max_unity
}

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

pub fn is_uniform(knots: &Knots) -> Result<bool, CurveError> {
    let u0 = knots.vector();

    let expected = methods::uniform(knots.degree(), knots.segments())?;

    Ok(u0.eq(expected.vector()))
    // u0.relative_eq(expected.vector(None), NUMERICAL_PRECISION, 0.0) // TODO test
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

pub fn normalize(knots: &mut VecD) {
    let old_lim = (knots.min(), knots.max());

    rescale(knots, old_lim, (0.0, 1.0))
}

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
        assert_eq!(knots_example(degree).segments(), SEGMENTS);
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
        assert!(!is_uniform(&Knots::new(1, dvector![0.0, 1.0, 0.5, 1.0, 1.0])).unwrap());
        //TODO assert_eq!(is_clamped(&Knots::new(1, dvector![2.0, 2.0, 0.5, 3.0, 3.0])), true);
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

    #[rstest(u, expected, case(0.24, 1), case(0.25, 1), case(0.26, 2), case(0.74, 3), case(0.75, 3), case(0.76, 4))]
    fn test_find_idx_of_left_domain_knot(u: f64, expected: usize) {
        assert_eq!(knots_example(1).find_idx(u, 0, DomainKnotComparatorType::Left), expected);
    }

    #[rstest(u, expected, case(0.24, 1), case(0.25, 2), case(0.26, 2), case(0.74, 3), case(0.75, 4), case(0.76, 4))]
    fn test_find_idx_of_left_or_equal_domain_knot(u: f64, expected: usize) {
        assert_eq!(knots_example(1).find_idx(u, 0, DomainKnotComparatorType::LeftOrEqual), expected);
    }

    #[rstest(u, expected, case(0.24, 2), case(0.25, 2), case(0.26, 3), case(0.74, 4), case(0.75, 4), case(0.76, 5))]
    fn test_find_idx_of_right_or_equal_domain_knot(u: f64, expected: usize) {
        assert_eq!(knots_example(1).find_idx(u, 0, DomainKnotComparatorType::RightOrEqual), expected);
    }

    #[rstest(u, expected, case(0.24, 2), case(0.25, 3), case(0.26, 3), case(0.74, 4), case(0.75, 5), case(0.76, 5))]
    fn test_find_idx_of_right_domain_knot(u: f64, expected: usize) {
        assert_eq!(knots_example(1).find_idx(u, 0, DomainKnotComparatorType::Right), expected);
    }
}
