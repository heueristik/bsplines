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

use crate::curve::basis;
use crate::{
    curve::{parameters, parameters::Parameters, CurveError},
    types::{KnotVectorDerivatives, VecD, VecDView, VecHelpers},
};

pub mod methods;

#[derive(Debug, Clone)]
pub struct Knots {
    pub(crate) Uk: KnotVectorDerivatives,
    pub(crate) p: usize,
    pub(crate) k_max: usize,
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
        let mut Uk: Vec<VecD> = Vec::with_capacity(degree + 1);
        Uk.push(knots);

        // TODO Add multiplicity check

        let mut knots = Knots { Uk, p: degree, k_max: 0 };
        knots.derive();
        knots
    }

    pub fn vector(&self) -> &VecD {
        &self.Uk[0]
    }

    pub fn vector_mut(&mut self) -> &mut VecD {
        &mut self.Uk[0]
    }

    /// # Arguments
    /// * `k` - The `k`-th derivative knot vector.
    pub fn vector_derivative(&self, derivative: usize) -> &VecD {
        &self.Uk[derivative]
    }

    pub fn vector_derivative_mut(&mut self, derivative: usize) -> &mut VecD {
        &mut self.Uk[derivative]
    }

    pub fn degree(&self) -> usize {
        self.p
    }

    pub fn segments(&self) -> usize {
        self.segments_derivative(0)
    }

    pub fn segments_derivative(&self, k: usize) -> usize {
        self.Uk[k].len() - (self.p + 2 - 2 * k)
    }

    pub fn len(&self, k: usize) -> usize {
        self.Uk[k].len()
    }

    pub fn internal_count(&self) -> usize {
        self.segments() - self.p
    }

    pub fn internal(&self) -> VecDView {
        self.Uk[0].segment(self.p + 1, self.internal_count())
    }

    pub fn internal_knot(&self, i: usize) -> f64 {
        self.internal()[i]
    }

    pub fn domain_count(&self) -> usize {
        self.segments() - self.p + 2
    }

    pub fn domain(&self) -> VecDView {
        self.domain_derivative(0)
    }

    pub fn domain_derivative(&self, k: usize) -> VecDView {
        self.Uk[k].segment(self.p - k, self.domain_count())
    }

    pub fn domain_knot(&self, i: usize) -> f64 {
        self.domain()[i]
    }

    pub fn multiplicity(&self, u: f64) -> usize {
        self.domain().iter().filter(|&x| *x == u).count()
    }

    pub(crate) fn reverse(&mut self) -> &mut Self {
        for knots in self.Uk.iter_mut() {
            reverse(knots);
        }
        self
    }

    pub fn normalize(&mut self) -> &mut Self {
        for knots in self.Uk.iter_mut() {
            normalize(knots);
        }
        self
    }

    pub fn rescale(&mut self, old_lim: (f64, f64), new_lim: (f64, f64)) {
        for knots in self.Uk.iter_mut() {
            rescale(knots, old_lim, new_lim);
        }
    }

    pub fn max_derivative(&self) -> usize {
        self.k_max
    }

    pub fn derive(&mut self) {
        let p = self.p;

        self.Uk.truncate(1);
        for k in 1..=p {
            // obtain the `k`-th derivative knot vector from the previous `k-1`-th derivative knot vector segment
            // by dropping the first and last segment
            let segment_of_previous_order_knot_vector = self.Uk[k - 1].segment(1, self.len(k - 1) - 2).clone_owned();

            self.Uk.push(segment_of_previous_order_knot_vector);
        }
        self.k_max = p;
    }

    /// Returns the index `i` of the knot on the domain interval
    /// `[u_{p-k}^{(k)}, u_{n+1-k}^{(k)})`,
    /// being less than or equal to `u`.
    fn find_index(&self, u: f64, k: usize) -> Option<usize> {
        let Uk = self.vector_derivative(k);
        let pk = self.degree() - k;

        let first = pk;

        if u < Uk[first] {
            return None;
        }

        let last = Uk.len() - pk - 1;
        let mut low = first;
        let mut high = last;

        while low < high {
            let mid = high - (high - low) / 2;

            if Uk[mid] > u {
                high = mid - 1;
            } else {
                low = mid;
            }
        }
        Some(high)
    }

    /// Returns the index `i` of the knot on the domain interval
    /// `[u_{p-k}^{(k)}, u_{n+1-k}^{(k)}]`,
    /// being lower, equal, or higher than `u`.
    pub fn find_idx(&self, u: f64, k: usize, comparator: DomainKnotComparatorType) -> usize {
        let Uk = self.vector_derivative(k);
        let pk = self.degree() - k;
        match comparator {
            DomainKnotComparatorType::Left => {
                let lim = self.segments() + 1 - k;
                let mut i = pk;

                while u > Uk[i + 1] && i + 1 < lim {
                    i += 1;
                }
                i
            }
            DomainKnotComparatorType::LeftOrEqual => {
                let lim = self.segments() + 1 - k;
                let mut i = pk;

                while u >= Uk[i + 1] && i + 1 < lim {
                    i += 1;
                    if Uk[i + 1] == Uk[i] {
                        break;
                    }
                }
                i
            }
            DomainKnotComparatorType::RightOrEqual => {
                let mut i = Uk.len() - 1 - pk;

                while u <= Uk[i - 1] && i > pk {
                    i -= 1;
                    if Uk[i - 1] == Uk[i] {
                        break;
                    }
                }
                i
            }
            DomainKnotComparatorType::Right => {
                let mut i = Uk.len() - 1 - pk;

                while u < Uk[i - 1] && i - 1 > pk {
                    i -= 1;
                }
                i
            }
        }
    }

    /// `p` the degree of this basis function of the kth degree spline - not of the 0th degree spline
    pub fn evaluate(&self, k: usize, i: usize, p: usize, u: f64) -> f64 {
        let Uk = &self.Uk[k];
        let n = self.segments();
        let pk = p - k;

        basis::basis(Uk, i, pk, k, n, u)
    }
}

fn is_valid(knots: Knots) -> bool {
    knots.segments() == knots.len(0) - (knots.degree() + 2)
}

pub fn is_clamped(knots: &Knots) -> bool {
    let U0 = knots.vector();
    let clamp_size = knots.p + 1;

    let is_head_clamped = U0.iter().take(clamp_size).all(|&u| u == 0.0);
    let is_tail_clamped = U0.iter().rev().take(clamp_size).all(|&u| u == 1.0);

    is_head_clamped && is_tail_clamped
}

pub fn is_normed(knots: &Knots) -> bool {
    let U0 = knots.vector();

    let is_min_zero = U0.iter().min_by(|a, b| a.partial_cmp(b).unwrap()) == Some(&0.0);
    let is_max_unity = U0.iter().max_by(|a, b| a.partial_cmp(b).unwrap()) == Some(&1.0);

    is_min_zero && is_max_unity
}

pub fn is_sorted(knots: &Knots) -> bool {
    let mut it = knots.Uk[0].iter();
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
    let U0 = knots.vector();

    let expected = methods::uniform(knots.degree(), knots.segments())?;

    Ok(U0.eq(expected.vector()))
    // U0.relative_eq(expected.vector(None), NUMERICAL_PRECISION, 0.0) // TODO test
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

fn rescaled_knot(mut knot: f64, old_lim: (f64, f64), new_lim: (f64, f64)) -> f64 {
    knot -= old_lim.0;
    knot /= old_lim.1 - old_lim.0;
    knot *= new_lim.1 - new_lim.0;
    knot += new_lim.0;

    knot
}

fn rescaled(knots: &VecD, old_lim: (f64, f64), new_lim: (f64, f64)) -> VecD {
    let mut copy = knots.clone();
    rescale(&mut copy, old_lim, new_lim);
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

        let U0 = dvector![0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0];
        let U1 = dvector![0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0];
        let U2 = dvector![0.0, 0.0, 0.5, 1.0, 1.0];
        let U3 = dvector![0.0, 0.5, 1.0];

        assert_eq!(knots.vector_derivative(0), &U0);
        assert_eq!(knots.vector_derivative(1), &U1);
        assert_eq!(knots.vector_derivative(2), &U2);
        assert_eq!(knots.vector_derivative(3), &U3);
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
        assert!(is_normed(&Knots::new(1, dvector![0.0, 0.0, 0.5, 1.0, 1.0])));
        assert!(!is_normed(&Knots::new(1, dvector![0.0, 0.0, 1.5, 1.0, 1.0])));
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

    #[rstest(u, expected, case(0.24, 1), case(0.25, 2), case(0.26, 2), case(0.74, 3), case(0.75, 4), case(0.76, 4))]
    fn test_find_idx_of_left_or_equal_domain_knot_bisection(u: f64, expected: usize) {
        assert_eq!(knots_example(1).find_index(u, 0).unwrap(), expected);
    }

    #[test]
    fn test_find_idx_of_left_or_equal_domain_knot_bisection_limits() {
        assert_eq!(knots_example(1).find_index(-0.1, 0), None);
        assert_eq!(knots_example(1).find_index(0.0, 0), Some(1));
        assert_eq!(knots_example(1).find_index(1.0, 0), Some(5));
        assert_eq!(knots_example(1).find_index(1.2, 0), Some(5));
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
